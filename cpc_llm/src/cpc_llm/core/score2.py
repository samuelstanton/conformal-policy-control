"""Compute the likelihood scores that an LLM assigns to particular sequences.
"""

import hydra
import json
import logging
import os
import pandas as pd
import pprint
import s3fs
import torch

from ..test_functions.finetune_utils import formatting_texts_func_edit_pairs, formatting_texts_func_single_seq
from ..core.model_client import ModelClient
from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union



def score(cfg: DictConfig, logger: logging.Logger = None):
    model_client = ModelClient(
        model_name_or_path=cfg.model_name_or_path,
        logger=logger,
        max_generate_length=500, 
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    # breakpoint()
    input_df = pd.read_json(cfg.input_data_path, orient="records", lines=True)
    target_df = pd.read_json(cfg.target_data_path, orient="records", lines=True)
    if cfg.sanity_check:
        logger.warning(
            "Running in sanity check mode. Will reduce data down to 10 target examples, 1 seed."
        )
        input_df = input_df.iloc[0,:]
        target_df = target_df.sample(10)
    input_data = input_df.to_dict("list")
    target_data = target_df.to_dict("list")
    # inputs = df[cfg.input_field].to_list()
    # targets = df[cfg.target_field].to_list()

    '''Example of formatted_inputs
    p formatted_inputs[0:2]
    ['<inc> [12, 31, 2, 27, 15, 6, 14, 9, 12, 31, 11, 10, 25, 1, 15, 11, 19, 24, 10, 4, 17, 28, 1, 14, 31, 28, 16, 15, 9, 14, 16, 15]\n', 
     '<inc> [12, 31, 2, 4, 15, 6, 9, 9, 12, 31, 11, 10, 25, 1, 15, 11, 19, 24, 10, 5, 17, 27, 1, 14, 31, 28, 16, 15, 11, 14, 16, 15]\n']
    '''
    
    formatted_inputs = formatting_texts_func_single_seq(input_data)

    '''Example of formatted_targets
    p formatted_targets[0:2]
    ['[12, 31, 2, 15, 15, 6, 14, 9, 12, 31, 11, 10, 25, 1, 15, 11, 19, 24, 10, 5, 19, 27, 1, 14, 31, 28, 16, 15, 11, 14, 16, 15]', 
     '[12, 31, 2, 4, 15, 6, 14, 9, 12, 31, 11, 10, 25, 1, 14, 11, 19, 24, 10, 5, 17, 27, 1, 14, 31, 28, 15, 15, 14, 14, 16, 15]']
    '''
    # breakpoint()
    formatted_targets = formatting_texts_func_single_seq(target_data)
    # formatted_targets = [json.dumps(target) for target in data[cfg.target_field]]
    # avg_likelihoods = model_client.compute_likelihoods(
    # breakpoint()
    avg_likelihoods = model_client.compute_likelihoods_avg(
        formatted_inputs,
        formatted_targets,
        batch_size=cfg.batch_size,
        logger=logger,
    )
    
    output_fp = os.path.join(cfg.output_dir, cfg.output_filename)
    data = target_df.to_dict("records")
    for avg_likelihood, record in zip(avg_likelihoods, data):
        record["likelihood"] = avg_likelihood
    output_df = pd.DataFrame(data)
    output_df.to_json(output_fp, orient="records", lines=True)


@hydra.main(config_path="../../../config", config_name="score")
def main(cfg: DictConfig):
    logging.basicConfig(level=cfg.log_level.upper(), force=True)
    logging.info(
        f"Running {__file__} with the following arguments:\n{pprint.pformat(OmegaConf.to_container(cfg))}"
    )
    logger = logging.getLogger(__file__)
    score(cfg, logger)


if __name__ == "__main__":
    main()
