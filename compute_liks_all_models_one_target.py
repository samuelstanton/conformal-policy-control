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
import s3fs
import torch

from finetune_utils import formatting_texts_func_edit_pairs, formatting_texts_func_single_seq
from model_client import ModelClient
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
        target_df = target_df[['particle', 'score'] + lik_col_names_old]

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

        logger.info(f"torch.cuda.is_available() : {torch.cuda.is_available()}")

        ## Load model
        # try:
        model_client = ModelClient(
            model_name_or_path=model_name_or_path,
            logger=logger,
            temperature = cfg.generation_config.temperature,
            max_generate_length=cfg.generation_config.max_new_tokens,
            device="cuda" if torch.cuda.is_available() else "cpu",
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

        ## Compute likelihoods
        target_likelihoods = model_client.compute_likelihoods_avg(
            formatted_inputs,
            formatted_targets,
            batch_size=cfg.batch_size,
            logger=logger,
        )
        # logger.info(f"target_likelihoods : {target_likelihoods}")
        all_timestep_likelihoods.append(target_likelihoods)

    output_fp = os.path.join(cfg.output_dir, cfg.output_filename)

    # lik_col_names_old = [f'lik_r{model_indices[i]}' for i in range(n_models)]

    ## Names of old likelihood columns already computed


    # logger.info(f"target_df.columns : {target_df.columns}")
    # logger.info(f"lik_col_names     : {lik_col_names}")
    # logger.info(f"all_timestep_likelihoods : {all_timestep_likelihoods}")

    # target_all_likelihoods_df = pd.DataFrame(np.c_[target_df, np.array(all_timestep_likelihoods).T], columns=np.concatenate((target_df.columns, lik_col_names)))


    target_all_likelihoods_df = pd.DataFrame(np.c_[target_df[['particle', 'score'] + lik_col_names_old], np.array(all_timestep_likelihoods).T], columns=['particle', 'score'] + lik_col_names_old + lik_col_names_new)
    logger.info(f"target_all_likelihoods_df.columns : {target_all_likelihoods_df.columns}")
    target_all_likelihoods_df.to_json(output_fp, orient="records", lines=True)



# def compute_likelihoods_one_model_all_cals(cfg: DictConfig, logger: logging.Logger = None):


#     if len(cfg.prev_target_data_path_list) == 0:
#         logger.info("No previous calibration data given (expected if is first iteration), skipping 'compute_likelihoods_one_model_all_cals'")
#         return

#     ## Load model
#     model_client = ModelClient(
#         model_name_or_path=cfg.model_name_or_path_list[-1], ## Use most recent model
#         logger=logger,
#         temperature = cfg.generation_config.temperature,
#         max_generate_length=cfg.generation_config.max_new_tokens,
#         device="cuda" if torch.cuda.is_available() else "cpu",
#     )



#     ## Load most recent input prompt seeds
#     input_fp = cfg.input_data_path_list[-1]
#     input_df = pd.read_json(input_fp, orient="records", lines=True)
#     # input_ds = datasets.Dataset.from_pandas(input_df)
#     input_data = input_df.to_dict("list")
#     # breakpoint()
#     formatted_inputs = formatting_texts_func_single_seq(input_data)


#     # last_timestep_idx = len(cfg.prev_target_data_path_list) - 1

#     ## For each model, compute likelihoods of each target sequence, averaged over seeds
#     # all_timestep_likelihoods = [] 
#     num_prev_cal = len(cfg.prev_target_data_path_list)
#     for i, target_data_path in enumerate(cfg.prev_target_data_path_list):

#         ## Load target data
#         target_df = pd.read_json(target_data_path, orient="records", lines=True)
#         target_data = target_df.to_dict("list")
#         # breakpoint()
#         formatted_targets = formatting_texts_func_single_seq(target_data)


#         ## Compute likelihoods
#         target_likelihoods = model_client.compute_likelihoods_avg(
#             formatted_inputs,
#             formatted_targets,
#             batch_size=cfg.batch_size,
#             logger=logger,
#         )

#         # output_fp = os.path.join(cfg.output_dir, cfg.output_filename)
#         # logger.info()
#         lik_col_name = [f'lik_r{num_prev_cal}']

#         logger.info(f"target_df.columns : {target_df.columns}")
#         logger.info(f"lik_col_name      : {lik_col_name}")

#         target_all_likelihoods_df = pd.DataFrame(np.c_[target_df, np.array(target_likelihoods).T], columns=np.concatenate((target_df.columns, lik_col_name)))
#         target_all_likelihoods_df.to_json(target_data_path, orient="records", lines=True)




@hydra.main(config_path="config", config_name="compute_liks_all_models_and_cal_data")
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
