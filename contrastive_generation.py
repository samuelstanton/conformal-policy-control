import datasets
import hydra
import logging
import os
import pandas as pd
import pprint
import torch

from botorch.test_functions import SyntheticTestFunction
from finetune_utils import (
    formatting_texts_func_plain_pairs_higher,
    formatting_texts_func_edit_pairs,
    parse_particle_and_score,
    truncate_after_right_bracket_w_logps,
)
from holo.test_functions.closed_form import Ehrlich, RoughMtFuji
from model_client import ModelClient
from omegaconf import DictConfig, OmegaConf
from scipy.spatial import distance
from tqdm import tqdm
from transformers import GenerationConfig
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union



# def run_unconditional_generation(cfg: DictConfig, model_dir: str, logger: logging.Logger = None):
#     ''' Generates unconditional samples from model
#     '''
#     gen_config = hydra.utils.instantiate(cfg.generation_config)
#     model_client = ModelClient(
#         model_name_or_path=model_dir, #cfg.model_name_or_path,
#         logger=logger,
#         max_generate_length=gen_config.max_new_tokens,
#         device="cuda" if torch.cuda.is_available() else "cpu",
#     )



@torch.no_grad()
def generate_texts_batched_contrastive_mixture(
    models_list: List["ModelClient"],          # List of reference models
    input_texts_list: List[List[str]],              # [prompts_B1, ..., prompts_Bk, prompts_A]
    # contrastive_proportion: float = 1.0,                             # Weight for log(P_A / P_mix)
    # ref_proportion: float = 0.0,                              # Weight for log(P_mix)
    # temperature: float = 1.0,
    # max_new_tokens: int = 100,
    return_likelihood_ratios: bool = True,
    mixture_weights: Optional[List[float]] = None,  # Optional weights for mixture (default uniform)
    logger: logging.Logger = None,
    **kwargs,
) -> Union[List[str], Tuple[List[str], List[float]]]:
    """
    Contrastive decoding against a mixture of reference models where both the target model (A)
    and each reference model (Bi) compute next-token probabilities averaged over their own prompt sets.
    Uses log( mean(P) ) rather than mean( log(P) ).
    """
    num_models = len(models_list)
    num_ref_models = num_models - 1
    if num_models == 0:
        raise ValueError("At least one model must be provided")
    if len(input_texts_list) != num_models:
        raise ValueError(f"input_texts_list must have length num_models (got {len(input_texts_list)})")

    target_model = models_list[-1]
    reference_models = models_list[:-1]

    target_input_texts = input_texts_list[-1]
    if len(target_input_texts) == 0:
        raise ValueError("Target prompt list must be non-empty")

    # # Optional: enforce equal counts; remove if variable counts are acceptable
    # num_sequences = len(target_input_texts)
    # for i, ref_prompts in enumerate(input_texts_list[:-1]):
    #     if len(ref_prompts) != num_sequences:
    #         raise ValueError(f"input_texts_list[{i}] length ({len(ref_prompts)}) must match target prompts length ({num_sequences})")

    # Mixture weights (normalize)
    if mixture_weights is None:
        mixture_weights = [1.0 / num_ref_models] * num_ref_models
    else:
        if len(mixture_weights) != num_ref_models:
            raise ValueError("mixture_weights length must match number of reference models")
        total_w = sum(mixture_weights)
        if total_w <= 0:
            raise ValueError("mixture_weights must sum to a positive value")
        mixture_weights = [w / total_w for w in mixture_weights]

    # Tokenize all prompts once
    target_input_ids = target_model._tokenize_batch(target_input_texts).input_ids
    if target_model.device is not None:
        target_input_ids = target_input_ids.to(target_model.device)

    ref_input_ids_list = []
    for ref_prompts in input_texts_list[:-1]:
        ref_ids = target_model._tokenize_batch(ref_prompts).input_ids
        if target_model.device is not None:
            ref_ids = ref_ids.to(target_model.device)
        ref_input_ids_list.append(ref_ids)

    # Shared generated suffix across all prompts
    generated_suffix = torch.empty(
        (1, 0), dtype=target_input_ids.dtype, device=target_input_ids.device
    )

    per_step_log_ratios: List[float] = []
    eps = 1e-12  # numerical stability

    output_strs = []
    all_output_token_ids = []
    all_token_logps = []

    for _ in tqdm(range(target_model.max_generate_length), desc="Contrastive generation (avg probs)..."):
        # Target: compute probabilities for each prompt, then average probs (not logs)
        suffix_repeated_target = generated_suffix.repeat(target_input_ids.size(0), 1)  # [N_target, T]
        target_batch = torch.cat([target_input_ids, suffix_repeated_target], dim=1)    # [N_target, L_t+T]
        outputs_a = target_model.model(target_batch)
        logits_a_all = outputs_a.logits[:, -1, :]                                       # [N_target, V]
        probs_a_all = torch.softmax(logits_a_all / target_model.temperature, dim=-1)                 # [N_target, V]
        probs_a_avg = probs_a_all.mean(0, keepdim=True)                                  # [1, V]
        log_probs_a = torch.log(probs_a_avg.clamp_min(eps))                              # [1, V]

        # References: average probs per model over its prompts, then weighted sum across models
        mixture_probs = None  # [1, V]
        for w, (ref_ids, ref_model) in zip(mixture_weights, zip(ref_input_ids_list, reference_models)):
            suffix_repeated_ref = generated_suffix.repeat(ref_ids.size(0), 1)           # [N_ref_i, T]
            ref_batch = torch.cat([ref_ids, suffix_repeated_ref], dim=1)                # [N_ref_i, L_i+T]
            ref_out = ref_model.model(ref_batch)
            ref_logits_all = ref_out.logits[:, -1, :]                                   # [N_ref_i, V]
            ref_probs_all = torch.softmax(ref_logits_all / ref_model.temperature, dim=-1)         # [N_ref_i, V]
            ref_probs_avg = ref_probs_all.mean(0, keepdim=True)                          # [1, V]
            weighted = w * ref_probs_avg
            mixture_probs = weighted if mixture_probs is None else (mixture_probs + weighted)

        mixture_log_probs = torch.log(mixture_probs.clamp_min(eps))                      # [1, V]

        # Contrastive score per token
        # contrastive_scores = contrastive_proportion * (log_probs_a - mixture_log_probs) + ref_proportion * mixture_log_probs  # [1, V]
        contrastive_scores = log_probs_a - mixture_log_probs  # [1, V]
        # logger.info(f"contrastive_scores : {contrastive_scores}, log_probs_a : {log_probs_a}, mixture_log_probs : {mixture_log_probs}")

        # Greedy next token (change to sampling if desired)
        next_token = torch.argmax(contrastive_scores, dim=-1, keepdim=True)  # [1,1]

        # Track per-step log-ratio if requested
        if return_likelihood_ratios:
            step_lr = torch.gather(log_probs_a, -1, next_token).squeeze(-1) - torch.gather(mixture_log_probs, -1, next_token).squeeze(-1)
            per_step_log_ratios.append(step_lr.item())
            # logger.info(f"return_likelihood_ratios : {return_likelihood_ratios}, step_lr : {step_lr}")

        # Append to the shared suffix
        generated_suffix = torch.cat([generated_suffix, next_token], dim=1)

        # Early stop on EOS
        if next_token.item() == target_model.tokenizer.eos_token_id:
            break

    logger.info(f"float(sum(per_step_log_ratios)) : {float(sum(per_step_log_ratios))}")
    logger.info(f"float(exp(sum(per_step_log_ratios))) : {float(np.exp(sum(per_step_log_ratios)))}")
    # Decode using first prompt to strip context; only the suffix is returned
    decoded_outputs = target_model.tokenizer.batch_decode(
        torch.cat([target_input_ids[0:1], generated_suffix], dim=1)[:, target_input_ids.shape[-1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    logger.info(f"decoded_outputs : {decoded_outputs}")
    # logger.info(f"float(sum(per_step_log_ratios)) : {float(sum(per_step_log_ratios))}")
    # logger.info(f"float(sum(per_step_log_ratios)) : {float(np.log(sum(per_step_log_ratios)))}")

    if return_likelihood_ratios:
        return decoded_outputs, float(sum(per_step_log_ratios))
    return decoded_outputs

    # output_strs.extend(decoded_outputs)
    # if return_likelihood_ratios:
    #     token_logps = self.model.compute_transition_scores(
    #         outputs.sequences, outputs.scores, normalize_logits=True
    #     )
    #     for oti in output_token_ids[:, input_ids.shape[-1] :]:
    #         all_output_token_ids.append(oti)
    #     for o_logps in token_logps:
    #         all_token_logps.append(o_logps)
    # if return_likelihood_ratios:
    #     return (output_strs, all_output_token_ids, all_token_logps)
    # return output_strs




def run_contrastive_generation(cfg: DictConfig, logger: logging.Logger = None):
    test_fn_params = pd.read_json(cfg.test_fn_fp, orient="records", lines=True).to_dict(
        "records"
    )[0]
    if cfg.test_fn_type == "ehrlich":
        test_fn = Ehrlich(
            num_states=test_fn_params["num_states"],
            dim=test_fn_params["dim"],
            num_motifs=test_fn_params["num_motifs"],
            motif_length=test_fn_params["motif_length"],
            quantization=test_fn_params["quantization"],
            noise_std=test_fn_params["noise_std"],
            negate=test_fn_params["negate"],
            random_seed=test_fn_params["random_seed"],
        )
    else:
        test_fn = RoughMtFuji(
            dim=test_fn_params["dim"],
            additive_term=test_fn_params["additive_term"],
            random_term_std=test_fn_params["random_term_std"],
            noise_std=test_fn_params["noise_std"],
            negate=test_fn_params["negate"],
            random_seed=test_fn_params["random_seed"],
        )
    gen_config = hydra.utils.instantiate(cfg.generation_config)


    ## Populate model_client_list: list of models
    model_client_list = []
    for model_name_or_path in cfg.model_name_or_path_list:
        model_client = ModelClient(
            model_name_or_path=model_name_or_path,
            logger=logger,
            temperature = gen_config.temperature,
            max_generate_length=gen_config.max_new_tokens,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        model_client_list.append(model_client)


    ## Populate input_texts_lists: a list of lists, each containing input prompts for each model.
    input_ds_list = []
    input_texts_list = []
    for input_fp in cfg.data_path_list:
        input_df = pd.read_json(input_fp, orient="records", lines=True)
        # logger.info(f"input_df["score"] : {input_df["score"]}")
        # logger.info(f"input_df["higher_score_particle"] : {input_df["higher_score_particle"]}")
        input_ds_list.append(datasets.Dataset.from_pandas(input_df))
        input_texts_list.append(formatting_texts_func_plain_pairs_higher(input_ds_list[-1]))
        ## May need to / want to write a new 'formatting_texts_func_single_sequences', to do job of 'formatting_texts_func_edit_pairs' for seed files


    if len(model_client_list) != len(input_ds_list):
        raise ValueError(f"Error: len(model_client_list)={len(model_client_list)} != {len(input_ds_list)}=len(input_ds_list)")


    # for iter in tqdm(range(1, cfg.max_iterations + 1), desc="Contrastive generation iterations..."):

    logger.info(
        f"Generating texts with len(input_texts_list[-1])={len(input_texts_list[-1])}, len(set(input_texts_list[-1]))={len(set(input_texts_list[-1]))}"
    )

    ## TO DO: Resolve connection between 'generate_texts_batched_contrastive_mixture' and here
    ## Note: Probably only need to return the likelihood ratio
    output_sequence, output_log_lik_ratio = generate_texts_batched_contrastive_mixture(
        model_client_list,
        input_texts_list,
        # temperature=cfg.temperature,
        # max_new_tokens=cfg.max_new_tokens,
        generation_config=gen_config,
        return_likelihood_ratios=True,
        logger=logger
        # mixture_weights=mixture_weights
        # subsample_seeds=cfg.subsample_seeds
    )

    logger.info(f"output_sequence      : {output_sequence}")
    logger.info(f"output_log_lik_ratio : {output_log_lik_ratio}")

    trunc_outputs = []
    trunc_output_logps = []
    for token_ids, token_logps in tqdm(
        zip(output_token_ids, output_token_logps), desc="Truncating outputs.."
    ):
        trunc_output, logps = truncate_after_right_bracket_w_logps(
            token_ids, token_logps, model_client.tokenizer, length_normalized=True
        )
        trunc_outputs.append(trunc_output)
        trunc_output_logps.append(logps)
    logger.info(
        f"Len of trunc_outputs: {len(trunc_outputs)}\nLen of trunc_output_logps: {len(trunc_output_logps)}\
        \nLen of input_ds before loop : {len(input_ds)}"
    )
    # store outputs and create inputs for the next iteration
    # prev_input_ds = input_ds
    # input_ds = []
    # for trajectory_idx in range(len(all_trajectories)):

    num_particles_generated = 0
    all_outputs = []
    for output_idx in range(gen_config.num_return_sequences):
        output = trunc_outputs[output_idx]
        output_logp = trunc_output_logps[output_idx]
        logger.info(f'output : {output}')
        output_particle_and_score = parse_particle_and_score(output, test_fn)
        logger.info(f'output_particle_and_score : {output_particle_and_score}')

        num_particles_generated += 1
        if output_particle_and_score is None:
            continue
        input_particle = prev_input_ds[trajectory_idx][
            cfg.higher_score_particle_field
        ]
        hamming_dist = distance.hamming(
            input_particle, output_particle_and_score[0]
        )
        # If any of the outputs is parsable, then we continue to iteratively
        # generate for that example.
        all_outputs.append(
            {
                "particle": output_particle_and_score[0],
                "score": output_particle_and_score[1],
                "loglikelihood": output_logp,
                "num_particles_generated": num_particles_generated,
                "hamming_distance": hamming_dist,
            }
        )
    # # Only include the highest-likelihood output in the pool for a given example
    # # in the inputs for the next round. If no particles have non-NaN log-likelihood, then
    # # use the original seed.
    # candidates = [
    #     d
    #     for d in all_trajectories[trajectory_idx]
    #     if d["loglikelihood"] is not None
    # ]
    # if len(candidates) > 0:
    #     max_likelihood_gen = max(candidates, key=lambda d: d["loglikelihood"])
    # else:
    #     max_likelihood_gen = all_trajectories[trajectory_idx][0]
    # input_ds.append(
    #     {
    #         cfg.higher_score_particle_field: max_likelihood_gen["particle"],
    #         "score": max_likelihood_gen["score"],
    #     }
    # )

    # input_ds = datasets.Dataset.from_list(input_ds)
    # # Give each trajectory an ID and flatten out the list of outputs!
    # all_trajectories = [
    #     {"trajectory_id": example_id, **d}
    #     for example_id, trajectory in enumerate(all_trajectories)
    #     for d in trajectory
    # ]
    # logger.info(f'all_outputs : {all_outputs}')
    all_outputs = pd.DataFrame(all_outputs)
    all_outputs.to_json(
        os.path.join(cfg.output_dir, cfg.output_filename), orient="records", lines=True
    )


@hydra.main(config_path="config", config_name="contrastive_generation")
def main(cfg: DictConfig):
    logging.basicConfig(level=cfg.log_level.upper(), force=True)
    logging.info(
        f"Running {__file__} with the following arguments:\n{pprint.pformat(OmegaConf.to_container(cfg))}"
    )
    logger = logging.getLogger(__file__)
    run_contrastive_generation(cfg, logger)


if __name__ == "__main__":
    main()
