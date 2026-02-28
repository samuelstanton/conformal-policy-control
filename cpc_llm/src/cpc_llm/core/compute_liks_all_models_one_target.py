"""Compute the likelihood scores that an LLM assigns to particular sequences, averaged over a set of input seeds, for all models in a list."""

import hydra
import logging
import os
import numpy as np
import pandas as pd
import pprint
import time
import torch

from ..test_functions.finetune_utils import formatting_texts_func_single_seq

from ..core.model_client import ModelClient
from omegaconf import DictConfig, OmegaConf


def compute_likelihoods_inmemory(
    target_df: pd.DataFrame,
    input_data_list,
    model_name_or_path_list,
    model_indices,
    cfg: DictConfig,
    logger: logging.Logger = None,
) -> pd.DataFrame:
    """Compute likelihoods for target_df under each model, entirely in memory.

    Args:
        target_df: DataFrame with particle/score columns (and possibly existing lik_r* columns).
        input_data_list: List of DataFrames, one per model, containing seed sequences.
        model_name_or_path_list: List of model paths to load for likelihood computation.
        model_indices: List of integer indices identifying each model (for column naming).
        cfg: Hydra config (needs batch_size, generation_config, overwrite_cmp_lik_all).
        logger: Logger instance.

    Returns:
        DataFrame with lik_r{i} columns appended for each model index.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    lik_col_names_old = [col for col in target_df.columns if "lik_r" in col]
    lik_col_names_new = [f"lik_r{m}" for m in model_indices]

    logger.info(f"pre selection lik_col_names_old : {lik_col_names_old}")
    logger.info(f"pre selection lik_col_names_new : {lik_col_names_new}")

    if cfg.overwrite_cmp_lik_all:
        logger.info("Overwrite")
        old_not_in_new = [c not in lik_col_names_new for c in lik_col_names_old]
        lik_col_names_old = list(np.array(lik_col_names_old)[old_not_in_new])
        if "score" in target_df.columns:
            target_df = target_df[["particle", "score"] + lik_col_names_old]
        else:
            target_df = target_df[target_df.columns[0:2].tolist() + lik_col_names_old]
    else:
        logger.info("No Overwrite")
        new_not_in_old = [c not in lik_col_names_old for c in lik_col_names_new]
        lik_col_names_new = list(np.array(lik_col_names_new)[new_not_in_old])
        model_indices = list(np.array(model_indices)[new_not_in_old])
        model_name_or_path_list = list(
            np.array(model_name_or_path_list)[new_not_in_old]
        )
        input_data_list = [
            input_data_list[i] for i, keep in enumerate(new_not_in_old) if keep
        ]

    logger.info(f"post selection lik_col_names_old : {lik_col_names_old}")
    logger.info(f"post selection lik_col_names_new : {lik_col_names_new}")
    logger.info(f"post selection model_indices : {model_indices}")
    logger.info(f"post selection model_name_or_path_list : {model_name_or_path_list}")

    ## Format target
    target_data = target_df.to_dict("list")
    formatted_targets = formatting_texts_func_single_seq(target_data)

    ## For each model, compute likelihoods of each target sequence, averaged over seeds
    all_timestep_likelihoods = []
    for i, model_name_or_path in enumerate(model_name_or_path_list):
        # GPU memory management
        torch.cuda.empty_cache()

        logger.info(f"torch.cuda.is_available() : {torch.cuda.is_available()}")

        # Retry ModelClient initialization with exponential backoff
        max_retries = 5
        retry_delay_base = 1.0
        model_client = None
        fallback_to_cpu = False
        for attempt in range(max_retries):
            try:
                model_client = ModelClient(
                    model_name_or_path=model_name_or_path,
                    logger=logger,
                    temperature=cfg.generation_config.temperature,
                    max_generate_length=cfg.generation_config.max_new_tokens,
                    device="cuda",
                )
                break
            except Exception as e:
                error_str = str(e)
                is_cuda_error = (
                    "CUDA" in error_str
                    or "busy" in error_str.lower()
                    or "unavailable" in error_str.lower()
                    or "AcceleratorError" in type(e).__name__
                )
                if is_cuda_error and attempt < max_retries - 1:
                    wait_time = retry_delay_base * (2**attempt)
                    logger.warning(
                        f"CUDA initialization failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {wait_time:.1f} seconds..."
                    )
                    time.sleep(wait_time)
                elif is_cuda_error:
                    fallback_to_cpu = True
                    logger.error(
                        f"CUDA initialization failed after {max_retries} attempts for model {model_name_or_path}: {e}. "
                        "Falling back to CPU."
                    )
                    break
                else:
                    raise
        if model_client is None and fallback_to_cpu:
            model_client = ModelClient(
                model_name_or_path=model_name_or_path,
                logger=logger,
                temperature=cfg.generation_config.temperature,
                max_generate_length=cfg.generation_config.max_new_tokens,
                device="cpu",
            )

        ## Format input seeds
        input_df = input_data_list[i]
        input_data = input_df.to_dict("list")
        formatted_inputs = formatting_texts_func_single_seq(input_data)

        logger.info(f"len(formatted_targets) : {len(formatted_targets)}")

        ## Compute likelihoods
        target_likelihoods = model_client.compute_likelihoods_avg(
            formatted_inputs,
            formatted_targets,
            batch_size=min(cfg.batch_size, len(formatted_targets)),
            logger=logger,
        )
        logger.info(f"target_likelihoods : {target_likelihoods}")
        all_timestep_likelihoods.append(target_likelihoods)

    ## Assemble result DataFrame
    if "score" in target_df.columns:
        if len(lik_col_names_new) > 0 and len(all_timestep_likelihoods) > 0:
            target_all_likelihoods_df = pd.DataFrame(
                np.c_[
                    target_df[["particle", "score"] + lik_col_names_old],
                    np.array(all_timestep_likelihoods).T,
                ],
                columns=["particle", "score"] + lik_col_names_old + lik_col_names_new,
            )
        else:
            logger.info("No new likelihoods to compute, using existing data")
            target_all_likelihoods_df = target_df[
                ["particle", "score"] + lik_col_names_old
            ].copy()
    else:
        if len(lik_col_names_new) > 0 and len(all_timestep_likelihoods) > 0:
            target_all_likelihoods_df = pd.DataFrame(
                np.c_[
                    target_df[target_df.columns[0:2].tolist() + lik_col_names_old],
                    np.array(all_timestep_likelihoods).T,
                ],
                columns=target_df.columns[0:2].tolist()
                + lik_col_names_old
                + lik_col_names_new,
            )
        else:
            logger.info("No new likelihoods to compute, using existing data")
            target_all_likelihoods_df = target_df[
                target_df.columns[0:2].tolist() + lik_col_names_old
            ].copy()

    logger.info(
        f"target_all_likelihoods_df.columns : {target_all_likelihoods_df.columns}"
    )
    return target_all_likelihoods_df


def compute_likelihoods_all_models_one_target(
    cfg: DictConfig, logger: logging.Logger = None
):
    """
    Loops through all models (and their corresponding input seed prompts) and computes the likelihoods
    output by those models for a single target calibration dataset (the most recently sampled cal dataset)
    """
    logger.info("Here in compute_likelihoods_all_models_one_target")

    n_models = len(cfg.model_name_or_path_list)
    if n_models != len(cfg.input_data_path_list):
        ValueError(
            f"Num models ({n_models}) != Num input seed files ({cfg.input_data_path_list})"
        )

    ## Load target data
    target_df = pd.read_json(cfg.target_data_path, orient="records", lines=True)

    logger.info(f"target_df : {target_df}")

    model_indices = (
        cfg.model_indices
        if len(cfg.model_indices) > 0
        else [i for i in range(n_models)]
    )
    model_name_or_path_list = list(cfg.model_name_or_path_list)
    input_data_path_list = list(cfg.input_data_path_list)

    ## Load input DataFrames from file paths
    input_data_list = [
        pd.read_json(fp, orient="records", lines=True) for fp in input_data_path_list
    ]

    output_fp = os.path.join(cfg.output_dir, cfg.output_filename)

    target_all_likelihoods_df = compute_likelihoods_inmemory(
        target_df,
        input_data_list,
        model_name_or_path_list,
        model_indices,
        cfg,
        logger,
    )
    target_all_likelihoods_df.to_json(output_fp, orient="records", lines=True)


@hydra.main(
    config_path="../../../config", config_name="compute_liks_all_models_and_cal_data"
)
def main(cfg: DictConfig):
    logging.basicConfig(level=cfg.log_level.upper(), force=True)
    logging.info(
        f"Running {__file__} with the following arguments:\n{pprint.pformat(OmegaConf.to_container(cfg))}"
    )
    logger = logging.getLogger(__file__)

    ## Compute all model likelihoods on most recently drawn calibration data
    logger.info("Running compute_likelihoods_all_models_one_target")
    compute_likelihoods_all_models_one_target(cfg, logger)

    ## Compute most recent model likelihoods on all historically drawn calibration data
    # compute_likelihoods_one_model_all_cals(cfg, logger)


if __name__ == "__main__":
    main()
