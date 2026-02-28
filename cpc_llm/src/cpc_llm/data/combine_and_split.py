from typing import List
from omegaconf import DictConfig
from ..infrastructure.file_handler import LocalOrS3Client
import pandas as pd
import numpy as np
import math
import os
import logging

logger = logging.getLogger(__name__)


def append_df_len_to_fp(fp, df):
    return "{0}_{2}.{1}".format(*fp.rsplit(".", 1) + [f"{len(df)}"])


def combine_datasets(
    cfg: DictConfig, fs: LocalOrS3Client, input_fps: List[str], output_fp: str
):
    """Combines multiple JSONL files."""
    if not cfg.overwrite and fs.exists(output_fp):
        logger.info(f"{output_fp} already exists. Skipping...")
        return
    combined_df = pd.concat(
        [pd.read_json(input_fp, orient="records", lines=True) for input_fp in input_fps]
    )
    if "hamming_distance" in combined_df.columns:
        hd_avg = combined_df[~np.isnan(combined_df["hamming_distance"])][
            "hamming_distance"
        ].mean()
        if np.isnan(hd_avg):
            hd_avg = None
        else:
            hd_avg = hd_avg.item()
    else:
        hd_avg = None
    combined_df = combined_df.drop_duplicates(subset=["particle"])
    combined_df.to_json(output_fp, orient="records", lines=True)
    logger.info(f"Datasets combined and written to {output_fp}.")
    return hd_avg


def combine_new_with_old_datasets(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    old_fps: List[str],
    curr_fp: str,
    random_seed: int,
):
    """Combines the current dataset with some old data."""
    if cfg.proportion_of_old_data == 0.0 or len(old_fps) == 0:
        return curr_fp
    output_fp = ".".join(curr_fp.split(".")[:-1]) + "_combined.jsonl"
    if not cfg.overwrite and fs.exists(output_fp):
        logger.info(f"{output_fp} already exists. Skipping...")
        return output_fp

    curr = pd.read_json(curr_fp, orient="records", lines=True)
    combined_datasets = [curr]
    num_curr_rows = len(curr)
    num_old_datasets = len(old_fps)
    # if cfg.proportion_of_old_data < 1.0:
    num_rows_per_old_fp = int(
        math.ceil(
            max(num_curr_rows * cfg.proportion_of_old_data, cfg.min_num_data_old)
            / num_old_datasets
            # / (num_old_datasets * (1.0 - cfg.proportion_of_old_data))
        )
    )
    for old_fp in old_fps:
        df = pd.read_json(old_fp, orient="records", lines=True)
        # if cfg.proportion_of_old_data < 1.0:
        df = df.sample(n=min(len(df), num_rows_per_old_fp), random_state=random_seed)
        combined_datasets.append(df)
    combined_datasets = pd.concat(combined_datasets)

    combined_datasets.to_json(output_fp, orient="records", lines=True)
    logger.info(f"Combined current dataset with old datasets. Written to {output_fp}.")
    return output_fp


def train_cal_split_gen_outputs(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    gen_outputs: str,
    sft_dir: str,
    sample_num_cal: int = None,
    sample_num_train: int = None,
    first_iter: bool = False,
    setting: str = "sft_CAinit",
    random_seed: int = 0,
):

    gen_outputs_df = pd.read_json(gen_outputs, orient="records", lines=True)

    if first_iter:
        cal_output_path = os.path.join(
            sft_dir,
            f"{setting}_alpha{cfg.conformal_policy_control.alpha}_cal_gens_all_likelihoods_temp{cfg.temperature_init}.jsonl",
        )
        train_output_path = os.path.join(
            sft_dir,
            f"{setting}_alpha{cfg.conformal_policy_control.alpha}_train_gens_all_likelihoods_temp{cfg.temperature_init}.jsonl",
        )
    else:
        cal_output_path = os.path.join(
            sft_dir,
            f"{setting}_alpha{cfg.conformal_policy_control.alpha}_cal_gens_all_likelihoods_temp{cfg.temperature}.jsonl",
        )
        train_output_path = os.path.join(
            sft_dir,
            f"{setting}_alpha{cfg.conformal_policy_control.alpha}_train_gens_all_likelihoods_temp{cfg.temperature}.jsonl",
        )

    if (
        len(gen_outputs_df)
        < cfg.conformal_policy_control.accept_reject.n_target_post_cpc
    ):
        cal_output_path = append_df_len_to_fp(cal_output_path, gen_outputs_df)
        train_output_path = append_df_len_to_fp(train_output_path, gen_outputs_df)

    overwrite_split_flag = (
        cfg.overwrite_split_init if first_iter else cfg.overwrite_split
    )

    if (
        not overwrite_split_flag
        and fs.exists(cal_output_path)
        and fs.exists(train_output_path)
    ):
        ## If not overwriting and both files exist, then load and return dataframes and paths
        cal_df = pd.read_json(cal_output_path, orient="records", lines=True)
        train_df = pd.read_json(train_output_path, orient="records", lines=True)

    else:
        ## Else, either overwriting or files do not exist, so create them

        ## Cal data (sample exchangeably, without replacement, from generated samples)
        if sample_num_cal is not None:
            ## If want to sample desired number, and 2x that desired number is available
            cal_df = gen_outputs_df.sample(n=sample_num_cal, random_state=random_seed)
        else:
            cal_df = gen_outputs_df.sample(
                frac=cfg.split.cal_frac, random_state=random_seed
            )

        cal_df.to_json(cal_output_path, orient="records", lines=True)

        ## Training data (sample exchangeably, w/o replacement, from *non-cal, deduplicated* generated samples)
        non_cal_gen_outputs_df = gen_outputs_df.drop(cal_df.index)  ## non cal
        non_cal_gen_outputs_df_unique = non_cal_gen_outputs_df.drop_duplicates(
            subset=["particle"]
        )  ## de-duplicate

        if cfg.split.train_sampling_method == "uniform":
            ## Randomly sample sequences for training set
            if sample_num_train is not None:
                train_df = non_cal_gen_outputs_df_unique.sample(
                    n=sample_num_train, random_state=random_seed
                )
            elif cfg.split.train_frac_from_non_cal < 1.0:
                train_df = non_cal_gen_outputs_df_unique.sample(
                    frac=cfg.split.train_frac_from_non_cal, random_state=random_seed
                )
            else:
                train_df = non_cal_gen_outputs_df_unique

        elif cfg.split.train_sampling_method == "best_scoring":
            ## Deterministically select top scoring sequences
            if sample_num_train is not None:
                train_df = non_cal_gen_outputs_df_unique.nlargest(
                    sample_num_train, "score"
                )
            elif cfg.split.train_frac_from_non_cal < 1.0:
                num_train = int(
                    cfg.split.train_frac_from_non_cal
                    * len(non_cal_gen_outputs_df_unique)
                )
                train_df = non_cal_gen_outputs_df_unique.nlargest(num_train, "score")
            else:
                train_df = non_cal_gen_outputs_df_unique
        else:
            raise ValueError(
                "cfg.split.train_sampling_method {cfg.split.train_sampling_method} not recognized"
            )
        train_df.to_json(train_output_path, orient="records", lines=True)

    return cal_df, cal_output_path, train_df, train_output_path
