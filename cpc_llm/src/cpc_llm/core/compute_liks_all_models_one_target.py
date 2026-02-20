"""Compute the likelihood scores that an LLM assigns to particular sequences, averaged over a set of input seeds, for all models in a list.
"""
import datasets
import hydra
import json
import logging
import os
import numpy as np
import pandas as pd
import pprint
import random
import s3fs
import time
import torch

from ..test_functions.finetune_utils import formatting_texts_func_edit_pairs, formatting_texts_func_single_seq
# from ..core.model_client import ModelClient
from ..core.model_client import *
from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union



def compute_likelihoods_all_models_one_target(cfg: DictConfig, logger: logging.Logger = None):
    """
    Loops through all models (and their corresponding input seed prompts) and computes the likelihoods
    output by those models for a single target calibration dataset (the most recently sampled cal dataset)
    """
    logger.info("Here in compute_likelihoods_all_models_one_target")

    n_models = len(cfg.model_name_or_path_list)
    if (n_models != len(cfg.input_data_path_list)):
        ValueError(f"Num models ({n_models}) != Num input seed files ({cfg.input_data_path_list})")


    ## Load target data
    target_df = pd.read_json(cfg.target_data_path, orient="records", lines=True)

    logger.info(f"target_df : {target_df}")

    model_indices = cfg.model_indices if len(cfg.model_indices) > 0 else [i for i in range(n_models)]
    model_name_or_path_list = cfg.model_name_or_path_list
    input_data_path_list = cfg.input_data_path_list

    lik_col_names_old = []
    for col in target_df.columns:
        if 'lik_r' in col:
            lik_col_names_old.append(col)

    logger.info(f"pre selection lik_col_names_old : {lik_col_names_old}")


    ## Names of new likelihoods computing here
    # lik_col_names_new = [f'lik_r{model_indices[i]}' for i in range(n_models)]
    lik_col_names_new = [f'lik_r{m}' for m in model_indices]

    logger.info(f"pre selection lik_col_names_new : {lik_col_names_new}")


    if cfg.overwrite_cmp_lik_all:
        logger.info("Overwrite")
        ## If want to overwrite computation, then subset target_df to only keep columns not trying to compute now
        old_not_in_new = [c not in lik_col_names_new for c in lik_col_names_old]
        lik_col_names_old = list(np.array(lik_col_names_old)[old_not_in_new])
        if 'score' in target_df.columns:
            target_df = target_df[['particle', 'score'] + lik_col_names_old]
        else:
           ## Handles case where want to compute likelihoods for contrastive-generation particle
            target_df = target_df[target_df.columns[0:2] + lik_col_names_old]

    else:
        logger.info("No Overwrite")
        ## Else, don't want to overwrite where not needed, so subset columns that want to compute to those not already computed
        new_not_in_old = [c not in lik_col_names_old for c in lik_col_names_new]
        lik_col_names_new = list(np.array(lik_col_names_new)[new_not_in_old])
        model_indices = list(np.array(model_indices)[new_not_in_old])
        model_name_or_path_list = list(np.array(model_name_or_path_list)[new_not_in_old])
        input_data_path_list = list(np.array(input_data_path_list)[new_not_in_old])


    logger.info(f"post selection lik_col_names_old : {lik_col_names_old}")
    logger.info(f"post selection lik_col_names_new : {lik_col_names_new}")
    logger.info(f"post selection model_indices : {model_indices}")
    logger.info(f"post selection model_name_or_path_list : {model_name_or_path_list}")
    logger.info(f"post selection input_data_path_list : {input_data_path_list}")


    ## Format target
    target_data = target_df.to_dict("list")
    
    formatted_targets = formatting_texts_func_single_seq(target_data)

        # lik_col_names_old_in_new = []
        # for c in lik_col_names_old:
        #     if c in lik_col_names_new:
        #         lik_col_names_old_in_new.append(c)


    ## For each model, compute likelihoods of each target sequence, averaged over seeds
    all_timestep_likelihoods = [] 
    for i, model_name_or_path in enumerate(model_name_or_path_list):
        
        # GPU memory management
        torch.cuda.empty_cache()
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        # try:
        #     if torch.cuda.is_available():
        #         torch.cuda.empty_cache()
        # except (RuntimeError, torch.cuda.Error) as e:
        #     logger.warning(f"Could not clear CUDA cache: {e}")

        logger.info(f"torch.cuda.is_available() : {torch.cuda.is_available()}")

        # Add a small random delay to stagger CUDA initialization across processes
        # This helps avoid race conditions when multiple processes try to set CUDA devices simultaneously
        if torch.cuda.is_available():
            delay = random.uniform(0.1, 0.5)
            time.sleep(delay)

        # Retry ModelClient initialization with exponential backoff to handle CUDA busy errors.
        # Always prefer CUDA first; only fall back to CPU if max retries are exhausted.
        max_retries = 5
        retry_delay = 1.0
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
                    wait_time = retry_delay * (2 ** attempt)
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
        # except torch.cuda.OutOfMemoryError:
        #     logger.warning("CUDA OOM, falling back to CPU")
        #     self.device = "cpu"
        #     self.model = self.model.to(self.device)
        # except torch.cuda.Error as e:
        #     logger.warning(f"CUDA error: {e}, falling back to CPU")
        #     self.device = "cpu"
        #     self.model = self.model.to(self.device)


        ## Load input_text
        input_fp = cfg.input_data_path_list[i]
        input_df = pd.read_json(input_fp, orient="records", lines=True)
        # input_ds = datasets.Dataset.from_pandas(input_df)
        input_data = input_df.to_dict("list")
        # breakpoint()
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

    output_fp = os.path.join(cfg.output_dir, cfg.output_filename)



    if 'score' in target_df.columns:
        logger.info(f"target_df[['particle', 'score'] + lik_col_names_old] : {target_df[['particle', 'score'] + lik_col_names_old]}")
        logger.info(f"np.array(all_timestep_likelihoods).T : {np.array(all_timestep_likelihoods).T}")
        target_all_likelihoods_df = pd.DataFrame(np.c_[target_df[['particle', 'score'] + lik_col_names_old], np.array(all_timestep_likelihoods).T], columns=['particle', 'score'] + lik_col_names_old + lik_col_names_new)
    else:
        ## Handles case where want to compute likelihoods for contrastive-generation particle
        target_all_likelihoods_df = pd.DataFrame(np.c_[target_df[target_df.columns[0:2] + lik_col_names_old], np.array(all_timestep_likelihoods).T], columns=target_df.columns[0:2] + lik_col_names_old + lik_col_names_new)
    logger.info(f"target_all_likelihoods_df.columns : {target_all_likelihoods_df.columns}")
    target_all_likelihoods_df.to_json(output_fp, orient="records", lines=True)




@hydra.main(config_path="../../../config", config_name="compute_liks_all_models_and_cal_data")
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
