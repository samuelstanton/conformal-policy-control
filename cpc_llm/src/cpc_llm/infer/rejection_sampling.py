from __future__ import annotations

import datasets
import functools
import numpy as np
import pandas as pd
import random
import time
import torch

import logging
import os

from typing import List
from omegaconf import DictConfig, OmegaConf
from transformers import GenerationConfig

from ..infrastructure.file_handler import LocalOrS3Client
from ..infrastructure.orchestration import (
    run_compute_liks_all_models_and_cal_data,
    run_iterative_generation,
)
from ..calibrate.process_likelihoods import constrain_likelihoods
from ..core.model_client import ModelClient
from ..core.compute_liks_all_models_one_target import compute_likelihoods_inmemory
from .iterative_generation2 import run_iterative_generation_inmemory
from holo.test_functions.closed_form import Ehrlich, RoughMtFuji


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-memory helpers (used when job_submission_system == "direct")
# ---------------------------------------------------------------------------


def _is_direct_mode(cfg: DictConfig) -> bool:
    return getattr(cfg, "job_submission_system", "slurm") == "direct"


@functools.lru_cache(maxsize=32)
def _read_jsonl_cached(filepath: str) -> pd.DataFrame:
    """Read a JSONL file, caching the result across AR iterations."""
    return pd.read_json(filepath, orient="records", lines=True)


def _load_test_fn(ga_data_dir: str) -> Ehrlich | RoughMtFuji:
    """Load the test function from the ehrlich.jsonl parameter file.

    Args:
        ga_data_dir: Directory containing the ehrlich.jsonl parameter file.

    Returns:
        An Ehrlich or RoughMtFuji test function instance.
    """
    test_fn_fp = os.path.join(ga_data_dir, "ehrlich.jsonl")
    params = pd.read_json(test_fn_fp, orient="records", lines=True).to_dict("records")[
        0
    ]
    if "num_motifs" in params:
        return Ehrlich(
            num_states=params["num_states"],
            dim=params["dim"],
            num_motifs=params["num_motifs"],
            motif_length=params["motif_length"],
            quantization=params["quantization"],
            noise_std=params["noise_std"],
            negate=params["negate"],
            random_seed=params["random_seed"],
        )
    return RoughMtFuji(
        dim=params["dim"],
        additive_term=params["additive_term"],
        random_term_std=params["random_term_std"],
        noise_std=params["noise_std"],
        negate=params["negate"],
        random_seed=params["random_seed"],
    )


def _init_model_client_with_retry(
    model_name_or_path: str,
    max_new_tokens: int,
    _logger: logging.Logger,
    temperature: float = 1.0,
) -> ModelClient:
    """Initialize a ModelClient with CUDA exponential-backoff retry.

    Args:
        model_name_or_path: HuggingFace model identifier or local path.
        max_new_tokens: Maximum number of tokens to generate.
        _logger: Logger instance for status messages.
        temperature: Temperature for likelihood computation scaling.

    Returns:
        An initialized ModelClient on CUDA (or CPU as fallback).
    """
    max_retries = 5
    retry_delay = 1.0
    model_client = None
    fallback_to_cpu = False

    if torch.cuda.is_available():
        time.sleep(random.uniform(0.1, 0.5))

    for attempt in range(max_retries):
        try:
            model_client = ModelClient(
                model_name_or_path=model_name_or_path,
                logger=_logger,
                max_generate_length=max_new_tokens,
                temperature=temperature,
                device="cuda",
            )
            return model_client
        except Exception as e:
            error_str = str(e)
            is_cuda_error = (
                "CUDA" in error_str
                or "busy" in error_str.lower()
                or "unavailable" in error_str.lower()
                or "AcceleratorError" in type(e).__name__
            )
            if is_cuda_error and attempt < max_retries - 1:
                wait_time = retry_delay * (2**attempt)
                _logger.warning(
                    f"CUDA init failed (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait_time:.1f}s..."
                )
                time.sleep(wait_time)
            elif is_cuda_error:
                fallback_to_cpu = True
                _logger.warning(
                    f"CUDA init failed after {max_retries} attempts: {e}. "
                    "Falling back to CPU."
                )
                break
            else:
                raise

    if fallback_to_cpu:
        return ModelClient(
            model_name_or_path=model_name_or_path,
            logger=_logger,
            max_generate_length=max_new_tokens,
            temperature=temperature,
            device="cpu",
        )
    raise RuntimeError(f"Failed to initialize ModelClient for {model_name_or_path}")


def _generate_inmemory(
    cfg: DictConfig,
    seeds_fp: str,
    model_client: ModelClient,
    test_fn: Ehrlich | RoughMtFuji,
    temp: float,
    higher_score_particle_field: str,
    lower_score_particle_field: str,
    higher_score_field: str,
    lower_score_field: str,
    random_seed: int,
) -> pd.DataFrame:
    """Read seeds, select, and generate proposals entirely in memory.

    Args:
        cfg: Pipeline config with iterative_generation settings.
        seeds_fp: Path to the JSONL seeds file.
        model_client: Pre-initialized ModelClient for generation.
        test_fn: Test function for scoring generated particles.
        temp: Generation temperature.
        higher_score_particle_field: Column name for higher-score particles.
        lower_score_particle_field: Column name for lower-score particles.
        higher_score_field: Column name for higher scores.
        lower_score_field: Column name for lower scores.
        random_seed: Random seed for reproducible seed selection.

    Returns:
        DataFrame with columns: particle, score, loglikelihood, num_particles_generated.
    """
    df = _read_jsonl_cached(seeds_fp)

    if cfg.sanity_check:
        logger.info(
            "Running in sanity check mode... reducing data down to 10 examples."
        )
        df = df.sample(n=min(10, len(df)))

    sample_size = cfg.iterative_generation.args.sample_size
    sampling_method = cfg.iterative_generation.args.sampling_method

    if higher_score_field in df.columns and lower_score_field in df.columns:
        logger.info(f"sample_size : {sample_size}")
        if sampling_method == "best_scoring":
            logger.info("sampling_method : best_scoring")
            df = df.sort_values(by=[lower_score_field], ascending=True)[:sample_size]
        elif sampling_method == "uniform":
            logger.info("sampling_method : uniform")
            df = df.sample(n=min(len(df), sample_size), random_state=random_seed)
        elif sampling_method == "combination":
            half = int(sample_size / 2)
            df = pd.concat(
                [
                    df.sort_values(by=[lower_score_field], ascending=True)[:half],
                    df.sample(n=min(len(df), half), random_state=random_seed),
                ]
            )
        else:
            raise ValueError(f"Unknown sampling method '{sampling_method}'")

        ds = datasets.Dataset.from_pandas(df)
        input_ds = datasets.Dataset.from_list(
            [
                {
                    higher_score_particle_field: ex[lower_score_particle_field],
                    "score": ex[lower_score_field],
                }
                for ex in ds
            ]
        )
    else:
        input_ds = datasets.Dataset.from_pandas(df)

    gen_config = GenerationConfig(
        do_sample=True,
        num_beams=1,
        temperature=temp,
        num_return_sequences=cfg.generation_sampling_num_return_sequences,
        max_new_tokens=cfg.iterative_generation.args.generation_config.max_new_tokens,
    )

    # Build a minimal config for run_iterative_generation_inmemory
    gen_cfg = OmegaConf.create(
        {
            "max_iterations": cfg.iterative_generation.args.max_iterations,
            "batch_size": cfg.sampling_gen_batch_size,
            "subsample_seeds": cfg.iterative_generation.args.subsample_seeds,
            "permissive_parsing": cfg.iterative_generation.permissive_parsing,
            "higher_score_particle_field": higher_score_particle_field,
            "lower_score_particle_field": lower_score_particle_field,
        }
    )

    return run_iterative_generation_inmemory(
        input_ds, model_client, test_fn, gen_config, gen_cfg, logger
    )


def _compute_liks_all_models_inmemory(
    cfg: DictConfig,
    target_df: pd.DataFrame,
    seeds_fp_list: List[str],
    model_dir_list: List[str],
    model_indices: List[int],
    _lik_model_clients: dict[str, ModelClient] | None = None,
) -> pd.DataFrame:
    """Compute likelihoods for all models in memory, without subprocesses."""
    input_data_list = [_read_jsonl_cached(fp) for fp in seeds_fp_list]

    # batch_size lives at the top level of the Hydra subprocess config
    # (compute_liks_all_models_and_cal_data.yaml), not in the pipeline
    # config's args sub-dict.  Default is 10.
    lik_batch_size = getattr(cfg.compute_likelihooods_all_models.args, "batch_size", 10)
    lik_cfg = OmegaConf.create(
        {
            "batch_size": lik_batch_size,
            "generation_config": {
                "temperature": cfg.temperature,
                "max_new_tokens": cfg.compute_likelihooods_all_models.args.generation_config.max_new_tokens,
            },
            "overwrite_cmp_lik_all": cfg.overwrite_cmp_lik_all,
        }
    )

    return compute_likelihoods_inmemory(
        target_df,
        input_data_list,
        list(model_dir_list),
        list(model_indices),
        lik_cfg,
        logger,
        model_clients=_lik_model_clients,
    )


# ---------------------------------------------------------------------------
# Main AR-sampling functions
# ---------------------------------------------------------------------------


def generate_proposals_for_AR_sampling(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    model_dir_list: List[str],
    seeds_fp_list: List[str],
    output_dir: str,
    betas_list: List[float],
    psis_list: List[float],  ## Normalization constants
    n_target: int,
    ga_data_dir: str,
    temps: List[int] = [1.0],
    depth: int = 0,  ## Recursion depth
    higher_score_particle_field: str = "higher_score_particle",
    lower_score_particle_field: str = "lower_score_particle",
    higher_score_field: str = "higher_score",
    lower_score_field: str = "lower_score",
    proposal: str = None,  ## Proposal distribution (safe or unconstrained), or None --> means running for filtering
    post_policy_control: bool = False,  ## Whether calling post policy control (True <--> generating risk-controlled actions), or pre control (False <--> Generating proposals)
    call_idx: int = 1,
    proportion_of_target_n_accepted: float = 0.0,
    global_random_seed: int = 0,
    _gen_model_client: ModelClient | dict[str, ModelClient] | None = None,
    _test_fn: Ehrlich | RoughMtFuji | None = None,
    _lik_model_clients: dict[str, ModelClient] | None = None,
) -> str:

    n_models = len(model_dir_list)
    use_inmemory = _is_direct_mode(cfg)

    ## Initialize data frames for storing data for accepted samples
    unconstrained_lik_cols = [f"lik_r{i}" for i in range(n_models)]
    unconstrained_col_names = ["particle", "score"] + unconstrained_lik_cols
    # accepted_unconstrained_df = pd.DataFrame(columns=unconstrained_col_names)

    constrained_lik_cols = [f"con_lik_r{i}" for i in range(n_models)]
    ["particle", "score"] + constrained_lik_cols
    # accepted_constrained_df = pd.DataFrame(columns=constrained_col_names)

    ## Compute a deterministic random seed for this call
    random_seed_curr = (
        global_random_seed * 10000 + post_policy_control * 1000 + call_idx
    )

    if proposal == "unconstrained":
        # accepted_curr = []

        temps_curr = temps if len(model_dir_list) > 1 else [cfg.temperature_init]

        if use_inmemory:
            # ---- In-memory path: no subprocesses, no intermediate files ----
            gen_df_ = _generate_inmemory(
                cfg,
                seeds_fp_list[-1],
                _gen_model_client,
                _test_fn,
                temps_curr[-1],
                higher_score_particle_field,
                lower_score_particle_field,
                higher_score_field,
                lower_score_field,
                random_seed_curr,
            )

            if len(gen_df_) == 0:
                return None, None, None, None, None, None

            gen_liks_df = _compute_liks_all_models_inmemory(
                cfg,
                gen_df_,
                seeds_fp_list,
                model_dir_list,
                list(range(n_models)),
                _lik_model_clients=_lik_model_clients,
            )
            # Synthetic filepath for output-dir derivation downstream
            gen_liks_fp = os.path.join(output_dir, "inmemory_liks.jsonl")
        else:
            # ---- Subprocess path (SLURM / legacy direct) ----
            ## Sample using unconstrained model as proposal
            _, iter_gen_outputs_list, hd = run_iterative_generation(
                cfg,
                fs,
                seeds_fp_list[-1],  # combined_sft_dataset_fp,
                ga_data_dir,
                model_dir_list[-1],  # sft_dir,
                output_dir,
                higher_score_particle_field=higher_score_particle_field,
                lower_score_particle_field=lower_score_particle_field,
                higher_score_field=higher_score_field,
                lower_score_field=lower_score_field,
                temps=temps_curr,
                model_idx=len(model_dir_list)
                - 1,  ## Index for model being called for generation
                call_idx=call_idx,  ## Index for times this generation has been called, including current, for same model directory
                proportion_of_target_n_accepted=proportion_of_target_n_accepted,
                post_policy_control=post_policy_control,
                global_random_seed=global_random_seed,
            )
            # call_idx += 1

            gen_df_ = pd.read_json(
                iter_gen_outputs_list[-1], orient="records", lines=True
            )

            if len(gen_df_) == 0:
                return None, None, None, None, None, None

            ## Compute unconstrained likelihoods for all models on the output proposal samples
            gen_liks_fp_list, hd = run_compute_liks_all_models_and_cal_data(
                cfg,
                fs,
                seeds_fp_list=seeds_fp_list,
                prev_cal_data_fp_list=[],
                model_dir_list=model_dir_list,
                target_fp=iter_gen_outputs_list[-1],
                temps=[cfg.temperature],
            )
            gen_liks_fp = gen_liks_fp_list[-1]

            gen_liks_df = pd.read_json(gen_liks_fp, orient="records", lines=True)

        gen_liks_df = gen_liks_df[unconstrained_col_names]
        gen_liks_mat = gen_liks_df[
            unconstrained_lik_cols
        ].to_numpy()  ## Shape (n_prop, n_models)

        if cfg.conformal_policy_control.constrain_against == "init":
            idx_safe_model = 0

            constrained_liks_mat = np.zeros(np.shape(gen_liks_mat))
            constrained_liks_mat[:, 0] = gen_liks_mat[:, 0]

            for c in range(1, len(unconstrained_lik_cols)):
                gen_liks_mat[:, [0, c]]

                constrained_liks_mat[:, c] = constrain_likelihoods(
                    cfg,
                    gen_liks_mat[:, [0, c]],
                    [betas_list[0], betas_list[c]],
                    [psis_list[0], psis_list[c]],
                )[:, -1]  ## Shape (n_prop, n_models)

            unconstrained_liks = gen_liks_mat[:, -1]
            safe_liks = constrained_liks_mat[:, idx_safe_model]
            lik_ratios_unconstrained_over_safe = unconstrained_liks / safe_liks

        else:
            idx_safe_model = -2

            constrained_liks_mat = constrain_likelihoods(
                cfg, gen_liks_mat, betas_list, psis_list
            )  ## Shape (n_prop, n_models)

            unconstrained_liks = gen_liks_mat[:, -1]

            if constrained_liks_mat.shape[1] > 1:
                ## If is not original safe model, \pi_{\theta_0}
                safe_liks = constrained_liks_mat[:, -2]
            else:
                ## Else is original safe model, \pi_{\theta_0}, so unconstrained and constrained likelihoods are the same
                ## (Lik ratios should be == 1, and bound == inf, so should accept everything)
                safe_liks = constrained_liks_mat[:, -1]

            lik_ratios_unconstrained_over_safe = unconstrained_liks / safe_liks

        constrained_liks_df_ = pd.DataFrame(
            constrained_liks_mat, columns=constrained_lik_cols
        )
        constrained_liks_df = pd.concat(
            [gen_liks_df[["particle", "score"]], constrained_liks_df_], axis=1
        )

        len(gen_liks_df)

    elif proposal == "safe":
        if cfg.conformal_policy_control.constrain_against == "init":
            if use_inmemory:
                # ---- In-memory path ----
                gen_df_ = _generate_inmemory(
                    cfg,
                    seeds_fp_list[0],
                    _gen_model_client,
                    _test_fn,
                    cfg.temperature_init,
                    higher_score_particle_field,
                    lower_score_particle_field,
                    higher_score_field,
                    lower_score_field,
                    random_seed_curr,
                )

                if len(gen_df_) == 0:
                    return None, None, None, None, None, None

                gen_liks_df = _compute_liks_all_models_inmemory(
                    cfg,
                    gen_df_,
                    seeds_fp_list,
                    model_dir_list,
                    list(range(n_models)),
                    _lik_model_clients=_lik_model_clients,
                )
                gen_liks_fp = os.path.join(output_dir, "inmemory_liks.jsonl")
            else:
                # ---- Subprocess path ----
                _, iter_gen_outputs_list, hd = run_iterative_generation(
                    cfg,
                    fs,
                    seeds_fp_list[0],  # combined_sft_dataset_fp,
                    ga_data_dir,
                    model_dir_list[0],  # sft_dir,
                    output_dir,
                    higher_score_particle_field=higher_score_particle_field,
                    lower_score_particle_field=lower_score_particle_field,
                    higher_score_field=higher_score_field,
                    lower_score_field=lower_score_field,
                    temps=[cfg.temperature_init],
                    model_idx=0,  # len(model_dir_list) - 1, ## Index for model being called for generation
                    call_idx=call_idx,  ## Index for times this generation has been called, including current, for same model directory
                    proportion_of_target_n_accepted=proportion_of_target_n_accepted,
                    post_policy_control=post_policy_control,
                    global_random_seed=global_random_seed,
                )
                gen_df_ = pd.read_json(
                    iter_gen_outputs_list[-1], orient="records", lines=True
                )

                if len(gen_df_) == 0:
                    return None, None, None, None, None, None

                ## Compute unconstrained likelihoods for all models on the output proposal samples
                gen_liks_fp_list, hd = run_compute_liks_all_models_and_cal_data(
                    cfg,
                    fs,
                    seeds_fp_list=seeds_fp_list,
                    prev_cal_data_fp_list=[],
                    model_dir_list=model_dir_list,
                    target_fp=iter_gen_outputs_list[-1],
                    temps=[cfg.temperature],
                )
                gen_liks_fp = gen_liks_fp_list[-1]

                gen_liks_df = pd.read_json(gen_liks_fp, orient="records", lines=True)

            gen_liks_df = gen_liks_df[unconstrained_col_names]
            gen_liks_mat = gen_liks_df[unconstrained_lik_cols].to_numpy()

            constrained_liks_mat = np.zeros(np.shape(gen_liks_mat))
            constrained_liks_mat[:, 0] = gen_liks_mat[:, 0]
            for c in range(1, len(unconstrained_lik_cols)):
                gen_liks_mat[:, [0, c]]

                constrained_liks_mat[:, c] = constrain_likelihoods(
                    cfg,
                    gen_liks_mat[:, [0, c]],
                    [betas_list[0], betas_list[c]],
                    [psis_list[0], psis_list[c]],
                )[:, -1]  ## Shape (n_prop, n_models)

            unconstrained_liks = gen_liks_mat[:, -1]
            safe_liks = constrained_liks_mat[:, 0]
            lik_ratios_unconstrained_over_safe = unconstrained_liks / safe_liks

            constrained_liks_df_ = pd.DataFrame(
                constrained_liks_mat, columns=constrained_lik_cols
            )
            constrained_liks_df = pd.concat(
                [gen_liks_df[["particle", "score"]], constrained_liks_df_], axis=1
            )

        else:
            ## Shouldn't need this, handled in base case of recursion
            # temps_curr = temps if len(model_dir_list) - 1 > 1 else [cfg.temperature_init] ## If model list after removing one model is not just initial, use temps

            ## Sample using unconstrained model as proposal
            (
                gen_liks_tmin1_df,
                gen_liks_tmin1_fp,
                constrained_gen_liks_tmin1_df,
                constrained_gen_liks_tmin1_fp,
            ) = accept_reject_sample_and_get_likelihoods(
                cfg,
                fs,
                model_dir_list[:-1],
                seeds_fp_list[:-1],
                output_dir,
                betas_list[:-1],
                psis_list[:-1],  ## Normalization constants
                n_target,
                ga_data_dir,
                temps,
                depth + 1,
                higher_score_particle_field=higher_score_particle_field,
                lower_score_particle_field=lower_score_particle_field,
                higher_score_field=higher_score_field,
                lower_score_field=lower_score_field,
                proposal="safe",
                global_random_seed=global_random_seed,
            )

            if use_inmemory:
                # ---- In-memory path for likelihood of most recent model ----
                gen_liks_df = _compute_liks_all_models_inmemory(
                    cfg,
                    gen_liks_tmin1_df,
                    [seeds_fp_list[-1]],
                    [model_dir_list[-1]],
                    [len(model_dir_list) - 1],
                    _lik_model_clients=_lik_model_clients,
                )
                gen_liks_fp = os.path.join(output_dir, "inmemory_liks.jsonl")
            else:
                ## Compute unconstrained likelihoods for most recent model (not passed to recursion) on output proposal samples
                gen_liks_fp_list, hd = run_compute_liks_all_models_and_cal_data(
                    cfg,
                    fs,
                    seeds_fp_list=[seeds_fp_list[-1]],
                    prev_cal_data_fp_list=[],
                    model_dir_list=[model_dir_list[-1]],
                    target_fp=gen_liks_tmin1_fp,  ## Should add a column for time t to gen_liks_tmin1_fp
                    model_indices=[
                        len(model_dir_list) - 1
                    ],  ## Index for most recent model
                    temps=temps,
                )
                gen_liks_fp = gen_liks_fp_list[-1]

                gen_liks_df = pd.read_json(gen_liks_fp, orient="records", lines=True)

            gen_liks_df = gen_liks_df[unconstrained_col_names]
            gen_liks_mat = gen_liks_df[
                unconstrained_lik_cols
            ].to_numpy()  ## Shape (n_prop, n_models)

            gen_liks_df_t0_safe_and_t_unconstrained_mat = pd.concat(
                [constrained_gen_liks_tmin1_df.iloc[:, -1], gen_liks_df.iloc[:, -1]],
                axis=1,
            ).to_numpy()  ## Double check this

            constrained_liks_mat = constrain_likelihoods(
                cfg,
                gen_liks_df_t0_safe_and_t_unconstrained_mat,
                betas_list[-2:],
                psis_list[-2:],
            )

            unconstrained_liks = gen_liks_mat[:, -1]
            if constrained_liks_mat.shape[1] > 1:
                ## If is not original safe model, \pi_{\theta_0}
                safe_liks = constrained_liks_mat[:, -2]
            else:
                ## Else is original safe model, \pi_{\theta_0}, so unconstrained and constrained likelihoods are the same
                ## (Lik ratios should be == 1, and bound == inf, so should accept everything)
                safe_liks = constrained_liks_mat[:, -1]

            lik_ratios_unconstrained_over_safe = unconstrained_liks / safe_liks

            constrained_liks_df_ = pd.DataFrame(
                constrained_liks_mat, columns=constrained_lik_cols[-2:]
            )
            constrained_liks_df = pd.concat(
                [constrained_gen_liks_tmin1_df, constrained_liks_df_.iloc[:, -1]],
                axis=1,
            )

    else:
        raise ValueError(f"Unrecognized proposal name : {proposal}")

    return (
        gen_liks_df,
        gen_liks_fp,
        constrained_liks_df,
        lik_ratios_unconstrained_over_safe,
        unconstrained_liks,
        safe_liks,
    )


def accept_reject_sample_and_get_likelihoods(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    model_dir_list: List[str],
    seeds_fp_list: List[str],
    output_dir: str,
    betas_list: List[float],
    psis_list: List[float],  ## Normalization constants
    n_target: int,
    ga_data_dir: str,
    temps: List[int] = [1.0],
    depth: int = 0,  ## Recursion depth
    higher_score_particle_field: str = "higher_score_particle",
    lower_score_particle_field: str = "lower_score_particle",
    higher_score_field: str = "higher_score",
    lower_score_field: str = "lower_score",
    proposal: str = None,  ## Proposal distribution (safe or unconstrained), or None --> means running for filtering
    post_policy_control: bool = False,  ## Whether calling post policy control (True <--> generating risk-controlled actions), or pre control (False <--> Generating proposals)
    safe_prop_mix_weight: float = 1.0,  ## if proposal == "mixture":  weight in (0, 1) to assign to safe proposal
    env_const: float = 1.0,  ## Recalculated envelope constant,
    global_random_seed: int = 0.0,
) -> str:

    n_models = len(model_dir_list)
    use_inmemory = _is_direct_mode(cfg)

    # accepted = [] ## List containing indicators for whether each considered proposal sample is accepted
    n_accepted = 0

    ## Initialize data frames for storing data for accepted samples
    unconstrained_lik_cols = [f"lik_r{i}" for i in range(n_models)]
    unconstrained_col_names = ["particle", "score"] + unconstrained_lik_cols
    constrained_lik_cols = [f"con_lik_r{i}" for i in range(n_models)]
    constrained_col_names = ["particle", "score"] + constrained_lik_cols

    accepted_unconstrained_dfs = []
    accepted_constrained_dfs = []

    call_idx = 0

    # ---- Pre-load models for in-memory mode ----
    gen_model_client = None
    test_fn = None
    lik_model_clients: dict[str, ModelClient] = {}
    if use_inmemory:
        test_fn = _load_test_fn(ga_data_dir)

        # Pre-load likelihood models first so gen can reuse them
        max_new_tokens_lik = (
            cfg.compute_likelihooods_all_models.args.generation_config.max_new_tokens
        )
        lik_temperature = cfg.temperature
        for model_path in model_dir_list:
            if model_path not in lik_model_clients:
                logger.info(f"Pre-loading likelihood model: {model_path}")
                lik_model_clients[model_path] = _init_model_client_with_retry(
                    model_path, max_new_tokens_lik, logger, temperature=lik_temperature
                )

        # Reuse lik clients for generation when the model path matches
        # (compute_likelihoods_avg does not use max_generate_length or temperature)
        max_new_tokens = cfg.iterative_generation.args.generation_config.max_new_tokens
        if proposal == "unconstrained":
            if model_dir_list[-1] in lik_model_clients:
                gen_model_client = lik_model_clients[model_dir_list[-1]]
            else:
                gen_model_client = _init_model_client_with_retry(
                    model_dir_list[-1], max_new_tokens, logger
                )
        elif proposal == "safe":
            if cfg.conformal_policy_control.constrain_against == "init":
                if model_dir_list[0] in lik_model_clients:
                    gen_model_client = lik_model_clients[model_dir_list[0]]
                else:
                    gen_model_client = _init_model_client_with_retry(
                        model_dir_list[0], max_new_tokens, logger
                    )
            # For safe/non-init, generation happens via recursion (no pre-load needed here)
        elif proposal == "mixture":
            gen_model_client = {
                "unconstrained": lik_model_clients.get(
                    model_dir_list[-1],
                    _init_model_client_with_retry(
                        model_dir_list[-1], max_new_tokens, logger
                    ),
                ),
                "safe": lik_model_clients.get(
                    model_dir_list[0],
                    _init_model_client_with_retry(
                        model_dir_list[0], max_new_tokens, logger
                    ),
                ),
            }

    # if betas_list[-1] >= 1:
    if proposal == "unconstrained":
        ## If beta_t >= 1, then using unconstrained policy as proposal

        ## If pre conformal policy control, running constrained model (ie, not alpha>=1.0), also check that number of calls has not yet exceeded max number
        # unconstrained_pre_cpc_call_num_check = call_idx < cfg.conformal_policy_control.accept_reject.max_unconstrained_proposal_calls_pre_cpc if not post_policy_control else True
        unconstrained_pre_cpc_call_num_check = True

        if cfg.conformal_policy_control.alpha >= 1.0:
            unconstrained_pre_cpc_call_num_check = (
                True  ## Set to True if running unconstrained
            )

        while (
            n_accepted < n_target
            and unconstrained_pre_cpc_call_num_check
            and call_idx < cfg.conformal_policy_control.accept_reject.max_total_AR_calls
        ):
            ## If pre conformal policy control, running constrained model (ie, not alpha>=1.0), also check that, if the specified number of calls have occurred, if at least 1/4 way to n_target_pre_cpc
            if (
                not post_policy_control
                and cfg.conformal_policy_control.alpha < 1.0
                and call_idx
                >= cfg.conformal_policy_control.accept_reject.n_opt_prop_calls_pre_cpc_quarter_check
                and n_accepted < n_target / 4
            ):
                unconstrained_pre_cpc_call_num_check = False
                break

            accepted_curr = []

            proportion_of_target_n_accepted = n_accepted / n_target

            (
                gen_liks_df,
                gen_liks_fp,
                constrained_liks_df,
                lik_ratios_unconstrained_over_safe,
                unconstrained_liks,
                safe_liks,
            ) = generate_proposals_for_AR_sampling(
                cfg,
                fs,
                model_dir_list,
                seeds_fp_list,
                output_dir,
                betas_list,
                psis_list,  ## Normalization constants
                n_target,
                ga_data_dir,
                temps,
                depth,  ## Recursion depth
                higher_score_particle_field,
                lower_score_particle_field,
                higher_score_field,
                lower_score_field,
                proposal,  ## Proposal distribution (safe or unconstrained), or None --> means running for filtering
                post_policy_control,  ## Whether calling post policy control (True <--> generating risk-controlled actions), or pre control (False <--> Generating proposals)
                call_idx,
                proportion_of_target_n_accepted,
                global_random_seed=global_random_seed,
                _gen_model_client=gen_model_client,
                _test_fn=test_fn,
                _lik_model_clients=lik_model_clients or None,
            )

            call_idx += 1

            if gen_liks_df is None:
                continue

            n_prop = len(gen_liks_df)

            ## Accept or reject each proposal

            ## Arbitrary way of standardizing random seeds so that is consistent when rerunning from checkpoint (but uses different random seed for each call)
            ar_random_seed = call_idx if not post_policy_control else 1000 + call_idx
            np.random.seed(ar_random_seed)

            ## Initialize MH state from first proposal
            target_lik = min(unconstrained_liks[0], betas_list[-1] * safe_liks[0])
            prop_lik = unconstrained_liks[0]

            for i in range(n_prop):
                u = np.random.uniform()

                ## Initial state for MH sampling
                if n_accepted == 0:
                    ## Keep updating the initial state until first acceptance
                    prev_target_lik = target_lik
                    prev_prop_lik = prop_lik

                ## Current target and proposal likelihoods (up to normalizing constant)
                target_lik = min(unconstrained_liks[i], betas_list[-1] * safe_liks[i])
                prop_lik = unconstrained_liks[i]

                if post_policy_control and (
                    cfg.conformal_policy_control.ind_metropolis_hastings
                    or call_idx > cfg.conformal_policy_control.num_AR_before_MH
                ):
                    ## Conditions for running IMH
                    acc_prob = min(
                        1, (target_lik / prev_target_lik) * (prev_prop_lik / prop_lik)
                    )

                else:
                    ## Condition for rejection sampling or until first acceptance of MH sampling
                    acc_prob = min(
                        1, betas_list[-1] / lik_ratios_unconstrained_over_safe[i]
                    )

                # betas_list[-1]/lik_ratios_unconstrained_over_safe[i]:
                if u < acc_prob:
                    ## Update target and proposal likelihoods for last accepted sample
                    prev_target_lik = target_lik
                    prev_prop_lik = prop_lik

                    accepted_curr.append(True)
                    n_accepted += 1

                    if n_accepted >= n_target:
                        break

                else:
                    accepted_curr.append(False)

            accepted_unconstrained_dfs.append(
                gen_liks_df[: len(accepted_curr)][accepted_curr]
            )
            accepted_constrained_dfs.append(
                constrained_liks_df[: len(accepted_curr)][accepted_curr]
            )

    elif proposal == "safe":
        ## Else, beta_t < 1, then using safe policy as proposal

        while (
            n_accepted < n_target
            and call_idx < cfg.conformal_policy_control.accept_reject.max_total_AR_calls
        ):
            accepted_curr = []

            ## Sample using unconstrained model as proposal

            if cfg.conformal_policy_control.constrain_against == "init":
                proportion_of_target_n_accepted = n_accepted / n_target

                (
                    gen_liks_df,
                    gen_liks_fp,
                    constrained_liks_df,
                    lik_ratios_unconstrained_over_safe,
                    unconstrained_liks,
                    safe_liks,
                ) = generate_proposals_for_AR_sampling(
                    cfg,
                    fs,
                    model_dir_list,
                    seeds_fp_list,
                    output_dir,
                    betas_list,
                    psis_list,  ## Normalization constants
                    n_target,
                    ga_data_dir,
                    temps,
                    depth,  ## Recursion depth
                    higher_score_particle_field,
                    lower_score_particle_field,
                    higher_score_field,
                    lower_score_field,
                    proposal,  ## Proposal distribution (safe or unconstrained), or None --> means running for filtering
                    post_policy_control,  ## Whether calling post policy control (True <--> generating risk-controlled actions), or pre control (False <--> Generating proposals)
                    call_idx,
                    proportion_of_target_n_accepted,
                    global_random_seed=global_random_seed,
                    _gen_model_client=gen_model_client,
                    _test_fn=test_fn,
                    _lik_model_clients=lik_model_clients or None,
                )
                call_idx += 1

                if gen_liks_df is None:
                    continue

            else:
                ## Shouldn't need this, handled in base case of recursion
                # temps_curr = temps if len(model_dir_list) - 1 > 1 else [cfg.temperature_init] ## If model list after removing one model is not just initial, use temps

                ## Sample using unconstrained model as proposal
                (
                    gen_liks_tmin1_df,
                    gen_liks_tmin1_fp,
                    constrained_gen_liks_tmin1_df,
                    constrained_gen_liks_tmin1_fp,
                ) = accept_reject_sample_and_get_likelihoods(
                    cfg,
                    fs,
                    model_dir_list[:-1],
                    seeds_fp_list[:-1],
                    output_dir,
                    betas_list[:-1],
                    psis_list[:-1],  ## Normalization constants
                    n_target,
                    ga_data_dir,
                    temps,
                    depth + 1,
                    higher_score_particle_field=higher_score_particle_field,
                    lower_score_particle_field=lower_score_particle_field,
                    higher_score_field=higher_score_field,
                    lower_score_field=lower_score_field,
                    proposal="safe",
                    global_random_seed=global_random_seed,
                )
                call_idx += 1

                if use_inmemory:
                    # ---- In-memory path ----
                    gen_liks_df = _compute_liks_all_models_inmemory(
                        cfg,
                        gen_liks_tmin1_df,
                        [seeds_fp_list[-1]],
                        [model_dir_list[-1]],
                        [len(model_dir_list) - 1],
                        _lik_model_clients=lik_model_clients or None,
                    )
                    gen_liks_fp = os.path.join(output_dir, "inmemory_liks.jsonl")
                else:
                    ## Compute unconstrained likelihoods for most recent model (not passed to recursion) on output proposal samples
                    gen_liks_fp_list, hd = run_compute_liks_all_models_and_cal_data(
                        cfg,
                        fs,
                        seeds_fp_list=[seeds_fp_list[-1]],
                        prev_cal_data_fp_list=[],
                        model_dir_list=[model_dir_list[-1]],
                        target_fp=gen_liks_tmin1_fp,  ## Should add a column for time t to gen_liks_tmin1_fp
                        model_indices=[
                            len(model_dir_list) - 1
                        ],  ## Index for most recent model
                        temps=temps,
                    )
                    gen_liks_fp = gen_liks_fp_list[-1]

                    gen_liks_df = pd.read_json(
                        gen_liks_fp, orient="records", lines=True
                    )

                gen_liks_df = gen_liks_df[unconstrained_col_names]
                gen_liks_mat = gen_liks_df[
                    unconstrained_lik_cols
                ].to_numpy()  ## Shape (n_prop, n_models)

                gen_liks_df_t0_safe_and_t_unconstrained_mat = pd.concat(
                    [
                        constrained_gen_liks_tmin1_df.iloc[:, -1],
                        gen_liks_df.iloc[:, -1],
                    ],
                    axis=1,
                ).to_numpy()  ## Double check this

                constrained_liks_mat = constrain_likelihoods(
                    cfg,
                    gen_liks_df_t0_safe_and_t_unconstrained_mat,
                    betas_list[-2:],
                    psis_list[-2:],
                )

                if constrained_liks_mat.shape[1] > 1:
                    ## If is not original safe model, \pi_{\theta_0}
                    lik_ratios_unconstrained_over_safe = (
                        gen_liks_mat[:, -1] / constrained_liks_mat[:, -2]
                    )
                else:
                    ## Else is original safe model, \pi_{\theta_0}, so unconstrained and constrained likelihoods are the same
                    ## (Lik ratios should be == 1, and bound == inf, so should accept everything)
                    lik_ratios_unconstrained_over_safe = (
                        gen_liks_mat[:, -1] / constrained_liks_mat[:, -1]
                    )

                constrained_liks_df_ = pd.DataFrame(
                    constrained_liks_mat, columns=constrained_lik_cols[-2:]
                )
                constrained_liks_df = pd.concat(
                    [constrained_gen_liks_tmin1_df, constrained_liks_df_.iloc[:, -1]],
                    axis=1,
                )

            n_prop = len(gen_liks_df)

            ## Arbitrary way of standardizing random seeds so that is consistent when rerunning from checkpoint (but uses different random seed for each call)
            ar_random_seed = call_idx if not post_policy_control else 1000 + call_idx
            np.random.seed(ar_random_seed)

            ## Accept or reject each proposal
            for i in range(n_prop):
                u = np.random.uniform()

                ## Initial state for MH sampling
                if n_accepted == 0:
                    if i == 0:
                        prev_target_lik = min(
                            unconstrained_liks[i], betas_list[-1] * safe_liks[i]
                        )  # / psis_list[-1]
                        prev_prop_lik = safe_liks[i]

                    else:
                        ## Can keep updating the initial state until first acceptance
                        prev_target_lik = target_lik
                        prev_prop_lik = prop_lik

                ## Current target and proposal likelihoods (up to normalizing constant)
                target_lik = min(
                    unconstrained_liks[i], betas_list[-1] * safe_liks[i]
                )  # / psis_list[-1]
                prop_lik = safe_liks[i]

                if post_policy_control and (
                    cfg.conformal_policy_control.ind_metropolis_hastings
                    or call_idx > cfg.conformal_policy_control.num_AR_before_MH
                ):
                    ## Conditions for running IMH
                    acc_prob = min(
                        1, (target_lik / prev_target_lik) * (prev_prop_lik / prop_lik)
                    )

                else:
                    ## Condition for rejection sampling or until first acceptance of MH sampling
                    acc_prob = min(
                        1, lik_ratios_unconstrained_over_safe[i] / betas_list[-1]
                    )

                # betas_list[-1]/lik_ratios_unconstrained_over_safe[i]:
                if u < acc_prob:
                    ## Update target and proposal likelihoods for last accepted sample
                    prev_target_lik = target_lik
                    prev_prop_lik = prop_lik

                    accepted_curr.append(True)
                    n_accepted += 1

                    if n_accepted >= n_target:
                        break

                else:
                    accepted_curr.append(False)

            accepted_unconstrained_dfs.append(
                gen_liks_df[: len(accepted_curr)][accepted_curr]
            )
            accepted_constrained_dfs.append(
                constrained_liks_df[: len(accepted_curr)][accepted_curr]
            )

    elif proposal == "mixture":
        accepted_curr_dict = {"safe": [], "unconstrained": []}

        n_proposed_dict = {"safe": 0, "unconstrained": 0}
        N_prop_dict = {"safe": 0, "unconstrained": 0}

        (
            gen_liks_df_dict,
            gen_liks_fp_dict,
            constrained_liks_df_dict,
            lik_ratios_unconstrained_over_safe_dict,
        ) = {}, {}, {}, {}
        unconstrained_liks_dict, safe_liks_dict = {}, {}

        while (
            n_accepted < n_target
            and call_idx < cfg.conformal_policy_control.accept_reject.max_total_AR_calls
        ):
            for proposal_curr in ["safe", "unconstrained"]:
                proportion_of_target_n_accepted = n_accepted / n_target

                ## If have already proposed the number of total proposals, then redraw samples for that policy:
                if n_proposed_dict[proposal_curr] >= N_prop_dict[proposal_curr]:
                    # For mixture mode, select the right model client
                    mix_model_client = None
                    if use_inmemory and isinstance(gen_model_client, dict):
                        mix_model_client = gen_model_client[proposal_curr]
                    (
                        gen_liks_df_dict[proposal_curr],
                        gen_liks_fp_dict[proposal_curr],
                        constrained_liks_df_dict[proposal_curr],
                        lik_ratios_unconstrained_over_safe_dict[proposal_curr],
                        unconstrained_liks_dict[proposal_curr],
                        safe_liks_dict[proposal_curr],
                    ) = generate_proposals_for_AR_sampling(
                        cfg,
                        fs,
                        model_dir_list,
                        seeds_fp_list,
                        output_dir,
                        betas_list,
                        psis_list,  ## Normalization constants
                        n_target,
                        ga_data_dir,
                        temps,
                        depth,  ## Recursion depth
                        higher_score_particle_field,
                        lower_score_particle_field,
                        higher_score_field,
                        lower_score_field,
                        proposal_curr,  ## Proposal distribution (safe or unconstrained), or None --> means running for filtering
                        post_policy_control,  ## Whether calling post policy control (True <--> generating risk-controlled actions), or pre control (False <--> Generating proposals)
                        call_idx,
                        proportion_of_target_n_accepted,
                        global_random_seed=global_random_seed,
                        _gen_model_client=mix_model_client,
                        _test_fn=test_fn,
                        _lik_model_clients=lik_model_clients or None,
                    )

                    call_idx += 1
                    if gen_liks_df_dict[proposal_curr] is None:
                        ## If no proposals generated, break out of for loop
                        break

                    ## Mixture proposal probabilies for constrained likelihoods
                    constrained_liks_df_dict[proposal_curr].iloc[:, -1] = (
                        safe_prop_mix_weight * safe_liks_dict[proposal_curr]
                        + (1 - safe_prop_mix_weight)
                        * unconstrained_liks_dict[proposal_curr]
                    )

                    N_prop_dict[proposal_curr] = len(
                        gen_liks_df_dict[proposal_curr]
                    )  ## Reset number of available proposals
                    n_proposed_dict[proposal_curr] = (
                        0  ## Reset number of used proposals to 0
                    )
                    accepted_curr_dict[
                        proposal_curr
                    ] = []  ## Reset running list of current acceptances

            if gen_liks_df_dict[proposal_curr] is None:
                ## If no proposals were generated, continue to restart the while loop (and try generating proposals again)
                continue

            while (
                n_proposed_dict["safe"] < N_prop_dict["safe"]
                and n_proposed_dict["unconstrained"] < N_prop_dict["unconstrained"]
            ):
                ## Arbitrary way of standardizing random seeds so that is consistent when rerunning from checkpoint (but uses different random seed for each call)
                ar_random_seed = (
                    call_idx if not post_policy_control else 1000 + call_idx
                )
                np.random.seed(ar_random_seed)

                ## Select proposal from the mixture
                u_mix = np.random.uniform()
                if u_mix < safe_prop_mix_weight or (
                    n_accepted == 0 and safe_prop_mix_weight > 0.5
                ):
                    proposal_curr = "safe"
                else:
                    proposal_curr = "unconstrained"

                lik_ratios_unconstrained_over_safe = (
                    lik_ratios_unconstrained_over_safe_dict[proposal_curr]
                )
                safe_liks = safe_liks_dict[proposal_curr]
                unconstrained_liks = unconstrained_liks_dict[proposal_curr]

                u = np.random.uniform()
                i_curr = n_proposed_dict[proposal_curr]

                ## Initial state for MH sampling
                if n_accepted == 0:
                    if i_curr == 0:
                        if cfg.conformal_policy_control.use_overlap_mix_weight:
                            prev_target_lik = (
                                min(
                                    unconstrained_liks[i_curr],
                                    betas_list[-1] * safe_liks[i_curr],
                                )
                                / psis_list[-1]
                            )
                        else:
                            prev_target_lik = min(
                                unconstrained_liks[i_curr],
                                betas_list[-1] * safe_liks[i_curr],
                            )

                        # prev_prop_lik = safe_liks[i_curr]  #safe_prop_mix_weight * safe_liks[i_curr] + (1 - safe_prop_mix_weight) * unconstrained_liks[i_curr]
                        prev_prop_lik = (
                            safe_prop_mix_weight * safe_liks[i_curr]
                            + (1 - safe_prop_mix_weight) * unconstrained_liks[i_curr]
                        )

                    else:
                        ## Can keep updating the initial state until first acceptance
                        prev_target_lik = target_lik
                        prev_prop_lik = prop_lik

                if cfg.conformal_policy_control.use_overlap_mix_weight:
                    target_lik = (
                        min(
                            unconstrained_liks[i_curr],
                            betas_list[-1] * safe_liks[i_curr],
                        )
                        / psis_list[-1]
                    )
                else:
                    target_lik = min(
                        unconstrained_liks[i_curr], betas_list[-1] * safe_liks[i_curr]
                    )

                prop_lik = (
                    safe_prop_mix_weight * safe_liks[i_curr]
                    + (1 - safe_prop_mix_weight) * unconstrained_liks[i_curr]
                )

                # if call_idx > cfg.conformal_policy_control.num_AR_before_MH:

                if post_policy_control and (
                    cfg.conformal_policy_control.ind_metropolis_hastings
                    or call_idx > cfg.conformal_policy_control.num_AR_before_MH
                ):
                    ## Conditions for running IMH
                    acc_prob = min(
                        1, (target_lik / prev_target_lik) * (prev_prop_lik / prop_lik)
                    )

                else:
                    ## Condition for rejection sampling or until first acceptance of MH sampling
                    acc_prob = min(1, (target_lik / (prop_lik * env_const)))

                n_proposed_dict[proposal_curr] += 1

                # betas_list[-1]/lik_ratios_unconstrained_over_safe[i]:
                if u < acc_prob:
                    ## Update states for MH
                    prev_target_lik = target_lik
                    prev_prop_lik = prop_lik

                    # if u < acc_prob:
                    accepted_curr_dict[proposal_curr].append(True)
                    n_accepted += 1

                    if n_accepted >= n_target:
                        break

                else:
                    accepted_curr_dict[proposal_curr].append(False)

            for proposal_curr in ["safe", "unconstrained"]:
                accepted_unconstrained_dfs.append(
                    gen_liks_df_dict[proposal_curr][
                        : len(accepted_curr_dict[proposal_curr])
                    ][accepted_curr_dict[proposal_curr]]
                )
                accepted_constrained_dfs.append(
                    constrained_liks_df_dict[proposal_curr][
                        : len(accepted_curr_dict[proposal_curr])
                    ][accepted_curr_dict[proposal_curr]]
                )

    else:
        raise ValueError(f"Unknown proposal name : {proposal}")

    # ---- Free pre-loaded models to reclaim GPU memory ----
    # gen_model_client may be shared with lik_model_clients; clear lik dict
    # first (releases all references), then delete gen if it was separate.
    lik_model_clients.clear()
    if gen_model_client is not None:
        del gen_model_client
    torch.cuda.empty_cache()

    # ## Save accepted with unconstrained likelihoods
    # t = len(model_dir_list)-1
    # accepted_unconstrained_df = gen_liks_df[accepted]

    if accepted_unconstrained_dfs:
        accepted_unconstrained_df = pd.concat(
            accepted_unconstrained_dfs, ignore_index=True
        )
    else:
        accepted_unconstrained_df = pd.DataFrame(columns=unconstrained_col_names)

    if accepted_constrained_dfs:
        accepted_constrained_df = pd.concat(accepted_constrained_dfs, ignore_index=True)
    else:
        accepted_constrained_df = pd.DataFrame(columns=constrained_col_names)

    base_output_name = f"alpha{cfg.conformal_policy_control.alpha}_gens_likelihood_{cfg.iterative_generation.args.sample_size}sample_{cfg.iterative_generation.args.max_iterations}iter_temp{temps[-1]}_{cfg.generation_sampling_num_return_sequences}seqs.jsonl"

    if proposal == "mixture":
        gen_liks_fp = gen_liks_fp_dict[proposal_curr]

    if depth == 0:
        if post_policy_control:
            ## Output filenames

            u_output_filename = f"accepted_uLiks_{base_output_name}"  # f"accepted_uLiks_{base_output_name}"
            c_output_filename = f"accepted_cLiks_{base_output_name}"  # f"accepted_cLiks_{base_output_name}"

            accepted_unconstrained_gen_liks_fp = os.path.join(
                os.path.dirname(gen_liks_fp), u_output_filename
            )
            accepted_constrained_gen_liks_fp = os.path.join(
                os.path.dirname(gen_liks_fp), c_output_filename
            )

        else:
            if gen_liks_fp is None and not unconstrained_pre_cpc_call_num_check:
                ## If terminating due to exceeding limit on number of unconstrained pre-CPC, return None
                return None, None, None, None

            accepted_unconstrained_gen_liks_fp = os.path.join(
                os.path.dirname(gen_liks_fp),
                f"prop_{proposal}_beta{betas_list[-1]}_cn{call_idx}_{base_output_name}",
            )
            accepted_constrained_gen_liks_fp = os.path.join(
                os.path.dirname(gen_liks_fp),
                f"prop_{proposal}_beta{betas_list[-1]}_cn{call_idx}_{base_output_name}",
            )

    else:
        accepted_unconstrained_gen_liks_fp = os.path.join(
            os.path.dirname(gen_liks_fp), f"{depth}u_cn{call_idx}_{base_output_name}"
        )
        accepted_constrained_gen_liks_fp = os.path.join(
            os.path.dirname(gen_liks_fp), f"{depth}c_cn{call_idx}_{base_output_name}"
        )

    if cfg.overwrite_ig or not fs.exists(accepted_unconstrained_gen_liks_fp):
        accepted_unconstrained_df.to_json(
            accepted_unconstrained_gen_liks_fp, orient="records", lines=True
        )

    if cfg.overwrite_ig or not fs.exists(accepted_constrained_gen_liks_fp):
        accepted_constrained_df.to_json(
            accepted_constrained_gen_liks_fp, orient="records", lines=True
        )

    return (
        accepted_unconstrained_df,
        accepted_unconstrained_gen_liks_fp,
        accepted_constrained_df,
        accepted_constrained_gen_liks_fp,
    )
