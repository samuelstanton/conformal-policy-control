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

from finetune_utils import formatting_texts_func_edit_pairs, formatting_texts_func_single_seq
from model_client import ModelClient
from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union


CUDA_ERROR = getattr(torch.cuda, "CudaError", RuntimeError)
# def compute_likelihoods_all_models_one_cal(cfg: DictConfig, logger: logging.Logger = None):
#     """
#     Loops through all models (and their corresponding input seed prompts) and computes the likelihoods
#     output by those models for a single target calibration dataset (the most recently sampled cal dataset)
#     """

#     n_models = len(cfg.model_name_or_path_list)
#     if (n_models != len(cfg.input_data_path_list)):
#         ValueError(f"Num models ({n_models}) != Num input seed files ({cfg.input_data_path_list})")


#     ## Load target data
#     target_df = pd.read_json(cfg.target_data_path, orient="records", lines=True)
#     target_data = target_df.to_dict("list")
#     # breakpoint()
#     formatted_targets = formatting_texts_func_single_seq(target_data)


#     ## For each model, compute likelihoods of each target sequence, averaged over seeds
#     all_timestep_likelihoods = [] 
#     for i, model_name_or_path in enumerate(cfg.model_name_or_path_list):

#         ## Load model
#         model_client = ModelClient(
#             model_name_or_path=model_name_or_path,
#             logger=logger,
#             temperature = cfg.generation_config.temperature,
#             max_generate_length=cfg.generation_config.max_new_tokens,
#             device="cuda" if torch.cuda.is_available() else "cpu",
#         )

#         ## Load input_text
#         input_fp = cfg.input_data_path_list[i]
#         input_df = pd.read_json(input_fp, orient="records", lines=True)
#         # input_ds = datasets.Dataset.from_pandas(input_df)
#         input_data = input_df.to_dict("list")
#         # breakpoint()
#         formatted_inputs = formatting_texts_func_single_seq(input_data)

#         ## Compute likelihoods
#         target_likelihoods = model_client.compute_likelihoods_avg(
#             formatted_inputs,
#             formatted_targets,
#             batch_size=cfg.batch_size,
#             logger=logger,
#         )
#         # logger.info(f"target_likelihoods : {target_likelihoods}")
#         all_timestep_likelihoods.append(target_likelihoods)

#     output_fp = os.path.join(cfg.output_dir, cfg.output_filename)

#     lik_col_names = [f'lik_r{i}' for i in range(n_models)]
#     # logger.info(f"target_df.columns : {target_df.columns}")
#     # logger.info(f"lik_col_names     : {lik_col_names}")
#     # logger.info(f"all_timestep_likelihoods : {all_timestep_likelihoods}")
#     target_all_likelihoods_df = pd.DataFrame(np.c_[target_df, np.array(all_timestep_likelihoods).T], columns=np.concatenate((target_df.columns, lik_col_names)))
#     target_all_likelihoods_df.to_json(output_fp, orient="records", lines=True)



def compute_likelihoods_one_model_all_data(cfg: DictConfig, logger: logging.Logger = None):


    if len(cfg.prev_target_data_path_list) == 0:
        logger.info("No previous calibration data given (expected if is first iteration), skipping 'compute_likelihoods_one_model_all_data'")
        return

    logger.info(f"torch.cuda.is_available()   : {torch.cuda.is_available()}")
    try:
        logger.info(f"torch.cuda.device_count()   : {torch.cuda.device_count()}")
        logger.info(f"torch.cuda.current_device() : {torch.cuda.current_device()}")
    except (RuntimeError, CUDA_ERROR) as e:
        logger.warning(f"Could not get CUDA device info: {e}")

    # # GPU memory management
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (RuntimeError, CUDA_ERROR) as e:
        logger.warning(f"Could not clear CUDA cache: {e}")

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
                model_name_or_path=cfg.model_name_or_path_list[-1],  # Use most recent model
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
                    f"CUDA initialization failed after {max_retries} attempts: {e}. Falling back to CPU."
                )
                break
            else:
                raise
    if model_client is None and fallback_to_cpu:
        model_client = ModelClient(
            model_name_or_path=cfg.model_name_or_path_list[-1],
            logger=logger,
            temperature=cfg.generation_config.temperature,
            max_generate_length=cfg.generation_config.max_new_tokens,
            device="cpu",
        )
    # logger.info(f"model_client.device() : {model_client.device()}")

    # model_client.to('cuda')
    # except torch.cuda.OutOfMemoryError:
    #     logger.warning("CUDA OOM, falling back to CPU")
    #     self.device = "cpu"
    #     self.model = self.model.to(self.device)
    # except torch.cuda.Error as e:
    #     logger.warning(f"CUDA error: {e}, falling back to CPU")
    #     self.device = "cpu"
    #     self.model = self.model.to(self.device)


    ## Load most recent input prompt seeds
    input_fp = cfg.input_data_path_list[-1]
    input_df = pd.read_json(input_fp, orient="records", lines=True)
    # input_ds = datasets.Dataset.from_pandas(input_df)
    input_data = input_df.to_dict("list")
    # breakpoint()
    formatted_inputs = formatting_texts_func_single_seq(input_data)


    # last_timestep_idx = len(cfg.prev_target_data_path_list) - 1

    ## This will also be model idx; eg, if 3 prev cal sets {0, 1, 2}, then current model idx is 3
    num_prev_cal = len(cfg.prev_target_data_path_list) 
    

    for i, target_data_path in enumerate(cfg.prev_target_data_path_list):

        ## Load target data
        target_df = pd.read_json(target_data_path, orient="records", lines=True)

        ## Ensure that only get previous target data with 
        lik_col_names_prev = [f'lik_r{c}' for c in range(num_prev_cal)]
        if 'score' in target_df.columns:
            target_df = target_df[['particle', 'score'] + lik_col_names_prev]
        else:
            ## Handles case where want to compute likelihoods for contrastive-generation particle
            target_df = target_df[target_df.columns[0:2] + lik_col_names_prev]


        
        target_data = target_df.to_dict("list")
        # breakpoint()
        formatted_targets = formatting_texts_func_single_seq(target_data)


        ## Compute likelihoods
        target_likelihoods = model_client.compute_likelihoods_avg(
            formatted_inputs,
            formatted_targets,
            batch_size=min(cfg.batch_size, len(formatted_targets)),
            logger=logger,
        )

        # output_fp = os.path.join(cfg.output_dir, cfg.output_filename)
        # logger.info()
        lik_col_name = [f'lik_r{num_prev_cal}']

        logger.info(f"target_df.columns : {target_df.columns}")
        logger.info(f"lik_col_name      : {lik_col_name}")

        target_all_likelihoods_df = pd.DataFrame(np.c_[target_df, np.array(target_likelihoods).T], columns=np.concatenate((target_df.columns, lik_col_name)))
        target_all_likelihoods_df.to_json(target_data_path, orient="records", lines=True)




@hydra.main(config_path="config", config_name="compute_liks_all_models_and_cal_data")
def main(cfg: DictConfig):
    logging.basicConfig(level=cfg.log_level.upper(), force=True)
    logging.info(
        f"Running {__file__} with the following arguments:\n{pprint.pformat(OmegaConf.to_container(cfg))}"
    )
    logger = logging.getLogger(__file__)

    ## Compute all model likelihoods on most recently drawn calibration data
    # compute_likelihoods_all_models_one_cal(cfg, logger)
    logger.info("Running compute_likelihoods_one_model_all_data")
    ## Compute most recent model likelihoods on all historically drawn calibration data
    compute_likelihoods_one_model_all_data(cfg, logger)


if __name__ == "__main__":
    main()
