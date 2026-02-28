from __future__ import annotations

import datasets
import hydra
import logging
import os
import pandas as pd
import pprint
import random
import time
import torch

from ..test_functions.finetune_utils import (
    formatting_texts_func_edit_pairs,
    parse_particle_and_score,
    parse_particle_and_score_permissive,
    truncate_after_right_bracket_w_logps,
)
from holo.test_functions.closed_form import Ehrlich, RoughMtFuji
from ..core.model_client import ModelClient
from omegaconf import DictConfig, OmegaConf
from transformers import GenerationConfig
from tqdm import tqdm


def run_iterative_generation_inmemory(
    input_ds: datasets.Dataset,
    model_client: ModelClient,
    test_fn: Ehrlich | RoughMtFuji,
    gen_config: GenerationConfig,
    cfg: DictConfig,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Core generation loop that operates entirely in memory.

    Args:
        input_ds: datasets.Dataset with seed sequences for generation.
        model_client: Pre-initialized ModelClient instance.
        test_fn: Ehrlich or RoughMtFuji test function for scoring.
        gen_config: HuggingFace generation config (instantiated).
        cfg: Hydra config (needs batch_size, max_iterations, subsample_seeds,
             permissive_parsing, higher_score_particle_field,
             lower_score_particle_field).
        logger: Logger instance.

    Returns:
        DataFrame with columns: particle, score, loglikelihood, num_particles_generated.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    num_particles_generated = 0
    all_outputs = []
    logger.info(f"cfg.max_iterations : {cfg.max_iterations}")
    for iter in tqdm(range(1, cfg.max_iterations + 1), desc="Generation iterations..."):
        ## Formatting text inputs
        input_texts = formatting_texts_func_edit_pairs(
            input_ds,
            include_target=False,
            higher_score_particle_field=cfg.higher_score_particle_field,
            lower_score_particle_field=cfg.lower_score_particle_field,
        )
        logger.info(
            f"Generating texts with cfg.subsample_seeds={cfg.subsample_seeds}, len(input_texts)={len(input_texts)}, len(set(input_texts))={len(set(input_texts))}"
        )

        ## Generate raw output texts
        _, output_token_ids, output_token_logps = model_client.generate_texts_batched(
            input_texts,
            batch_size=cfg.batch_size,
            generation_config=gen_config,
            return_likelihoods=True,
            subsample_seeds=cfg.subsample_seeds,
        )

        ## Truncated outputs
        trunc_outputs = []
        trunc_output_logps = []
        for token_ids, token_logps in tqdm(
            zip(output_token_ids, output_token_logps), desc="Truncating outputs.."
        ):
            trunc_output, logps = truncate_after_right_bracket_w_logps(
                token_ids,
                token_logps,
                model_client.tokenizer,
                length_normalized=False,  # True
            )
            trunc_outputs.append(trunc_output)
            trunc_output_logps.append(logps)

        # store outputs and create inputs for the next iteration
        for output_idx in range(len(trunc_outputs)):
            output = trunc_outputs[output_idx]
            output_logp = trunc_output_logps[output_idx]

            if cfg.permissive_parsing:
                output_particle_and_score = parse_particle_and_score_permissive(
                    output, test_fn
                )
            else:
                output_particle_and_score = parse_particle_and_score(output, test_fn)

            num_particles_generated += 1
            if output_particle_and_score is None:
                continue

            all_outputs.append(
                {
                    "particle": output_particle_and_score[0],
                    "score": output_particle_and_score[1],
                    "loglikelihood": output_logp,
                    "num_particles_generated": num_particles_generated,
                }
            )

    return pd.DataFrame(all_outputs)


def run_iterative_generation(cfg: DictConfig, logger: logging.Logger = None):
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

    # Add a small random delay to stagger CUDA initialization across processes
    # This helps avoid race conditions when multiple processes try to set CUDA devices simultaneously
    if torch.cuda.is_available():
        delay = random.uniform(0.1, 0.5)  # Random delay between 0.1-0.5 seconds
        time.sleep(delay)

    # Retry ModelClient initialization with exponential backoff to handle CUDA busy errors
    # This addresses race conditions when multiple processes initialize CUDA simultaneously
    # Always try CUDA first, only fall back to CPU after all retries are exhausted
    max_retries = 5
    retry_delay = 1.0
    model_client = None
    fallback_to_cpu = False
    for attempt in range(max_retries):
        try:
            model_client = ModelClient(
                model_name_or_path=cfg.model_name_or_path,
                logger=logger,
                max_generate_length=gen_config.max_new_tokens,
                device="cuda",  # Always try CUDA first
            )
            break
        except Exception as e:
            # Check if it's a CUDA-related error (could be RuntimeError, AcceleratorError, etc.)
            error_str = str(e)
            is_cuda_error = (
                "CUDA" in error_str
                or "busy" in error_str.lower()
                or "unavailable" in error_str.lower()
                or "AcceleratorError" in type(e).__name__
            )

            if is_cuda_error and attempt < max_retries - 1:
                wait_time = retry_delay * (2**attempt)  # Exponential backoff
                logger.warning(
                    f"CUDA initialization failed (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait_time:.1f} seconds..."
                )
                time.sleep(wait_time)
            elif is_cuda_error:
                logger.warning(
                    f"CUDA initialization failed after {max_retries} attempts: {e}. "
                    f"Falling back to CPU."
                )
                fallback_to_cpu = True
                break
            else:
                # Re-raise if it's not a CUDA-related error
                raise

    # Only fall back to CPU if all CUDA attempts failed
    if model_client is None and fallback_to_cpu:
        logger.info("Attempting to initialize ModelClient on CPU as fallback...")
        model_client = ModelClient(
            model_name_or_path=cfg.model_name_or_path,
            logger=logger,
            max_generate_length=gen_config.max_new_tokens,
            device="cpu",
        )
    elif model_client is None:
        raise RuntimeError("Failed to initialize ModelClient after all retry attempts.")
    df = pd.read_json(cfg.data_path, orient="records", lines=True)
    if cfg.sanity_check:
        logger.info(
            "Running in sanity check mode... reducing data down to 10 examples."
        )
        df = df.sample(n=min(10, len(df)))

    if cfg.higher_score_field in df.columns and cfg.lower_score_field in df.columns:
        # Before sampling, save all the particles as tuples in a set so that we can check whether
        # generations are regurgitations from the training data
        set(
            [tuple(ex[cfg.higher_score_particle_field]) for _, ex in df.iterrows()]
        ).union(
            set([tuple(ex[cfg.lower_score_particle_field]) for _, ex in df.iterrows()])
        )
        # Now dedupe the lower_score_particles and sample the lowest scoring examples from the data
        # to use as seeds for generation
        # df = df.drop_duplicates(subset=[cfg.lower_score_particle_field])
        logger.info(f"sample_size : {cfg.sample_size}")
        if cfg.sampling_method == "best_scoring" and not cfg.first_iter:
            ## Only use best_scoring if is not first iteration
            logger.info("sampling_method : best_scoring")
            df = df.sort_values(by=[cfg.lower_score_field], ascending=True)[
                : cfg.sample_size
            ]
        elif cfg.sampling_method == "uniform" or cfg.first_iter:
            ## Always use "uniform" for first iteration, for a safe initial policy
            logger.info("sampling_method : uniform")
            df = df.sample(n=min(len(df), cfg.sample_size), random_state=cfg.seed)
        elif cfg.sampling_method == "combination":
            half_sample_size = int(cfg.sample_size / 2)
            df = pd.concat(
                [
                    df.sort_values(by=[cfg.lower_score_field], ascending=True)[
                        :half_sample_size
                    ],
                    df.sample(n=min(len(df), half_sample_size), random_state=cfg.seed),
                ]
            )
        else:
            raise ValueError(f"Unknown sampling method '{cfg.sampling_method}.'")

        # Start by using the lower score particle from each pair as the seed
        ## ds: Dataset of *pairs*, by the best (lowest) sample_size num scoring particles
        ds = datasets.Dataset.from_pandas(df)

        ## input_ds: Dataset of *sequences*, by the best (lowest) sample_size num scoring particles
        input_ds = datasets.Dataset.from_list(
            [
                {
                    cfg.higher_score_particle_field: ex[cfg.lower_score_particle_field],
                    "score": ex[cfg.lower_score_field],
                }
                for ex in ds
            ]
        )

        ## If selected seeds here, then write them to disk (needed for computing seed-marginalized likelihoods used in policy control)
        input_df = input_ds.to_pandas()
        input_df.to_json(
            os.path.join(cfg.output_dir, f"seeds_{cfg.output_filename}"),
            orient="records",
            lines=True,
        )

    else:
        ## Else, assume that seeds are pre-selected, don't need to select
        set([tuple(ex[cfg.higher_score_particle_field]) for _, ex in df.iterrows()])

        # Start by using the lower score particle from each pair as the seed
        ## ds: Dataset of *pairs*, by the best (lowest) sample_size num scoring particles
        input_ds = datasets.Dataset.from_pandas(df)

    all_outputs = run_iterative_generation_inmemory(
        input_ds, model_client, test_fn, gen_config, cfg, logger
    )
    all_outputs.to_json(
        os.path.join(cfg.output_dir, cfg.output_filename), orient="records", lines=True
    )


@hydra.main(config_path="../../../config", config_name="iterative_generation")
def main(cfg: DictConfig):
    logging.basicConfig(level=cfg.log_level.upper(), force=True)
    logging.info(
        f"Running {__file__} with the following arguments:\n{pprint.pformat(OmegaConf.to_container(cfg))}"
    )
    logger = logging.getLogger(__file__)
    run_iterative_generation(cfg, logger)


if __name__ == "__main__":
    main()
