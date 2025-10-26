import hydra
import json
import logging
import math
import numpy as np
import os
import pandas as pd
import sys

from file_handler import LocalOrS3Client
from omegaconf import DictConfig, OmegaConf
from slurm_utils import submit_cmd_to_slurm, wait_for_slurm_jobs_to_complete
from tqdm import tqdm
from typing import Any, Dict, List, Mapping, Optional


logger = logging.getLogger(__name__)


def check_col_names(df):
    lik_cols = []
    for c in df.columns:
        if (c[0] == 'l' and c[1]=='i') or c[0] == 'c':
            lik_cols.append(c)

    col_indices = [int(c[-1]) for c in lik_cols]
    for i in range(len(col_indices)):

        if i > 0 and col_indices[i] - col_indices[i-1] != 1:
            raise ValueError(f"col indices not increasing {df.columns}")


def mixture_pdf_from_densities_mat(constrained_densities_all_steps, mixture_weights):
    '''
    constrained_densities_cal_test_all_steps : dim (n_samples, n_models) Note: columns correspond to t=0, ..., T-1
    mixture_weights         : dim (T), array of relative weights to put on each of *prior* distributions, from t=0, ..., T-1
                       Note : mixture_weights[i] = n_cal_model_i
    '''
    mixture_weights_normed = mixture_weights / np.sum(mixture_weights)

    # print(f'constrained_densities_cal_test_all_steps.T shape {np.shape(constrained_densities_cal_test_all_steps.T)}')
    # print(f'mixture_weights_normed shape : {np.shape(mixture_weights_normed)}')

    mixture_pdfs = constrained_densities_all_steps @ mixture_weights_normed

    return mixture_pdfs



'''Process matrix of unconstrained likelihoods into constrained likelihoods'''
def constrain_likelihoods(
    cfg: DictConfig,
    likelihoods_mat, ## 2-D np array, shape (n_prop, *); flexible num columns, equal to n_models total from safe model to curr
    betas, ## 1-D np array or list of lik-ratio bounds
    psis ## 1-D np array or list of normalization constants
):  
    n_prop, n_models = np.shape(likelihoods_mat)

    if n_models > 2 and cfg.conformal_policy_control.constrain_against == 'init':
        ## If constraining against initial safe policy, only want first model and current model
        raise ValueError("Modified to only constrain likelihoods relative to original safe policy")

    constrained_likelihoods_mat = np.zeros((n_prop, n_models))

    ## First col of likelihoods_mat should already be safe/constrained
    constrained_likelihoods_mat[:, 0] = likelihoods_mat[:, 0]  


    ## Compute constrained likelihoods for each subsequent policy and bound
    # for i in range(1, n_models):
    if cfg.conformal_policy_control.constrain_against == 'init':
        constrained_likelihoods_mat[:, 1] = np.where(likelihoods_mat[:, 1] / constrained_likelihoods_mat[:, 0] < betas[1], likelihoods_mat[:, 1] / psis[1], constrained_likelihoods_mat[:, 0] * (betas[1] / psis[1]))
    else:
        for i in range(1, n_models):
            constrained_likelihoods_mat[:, i] = np.where(likelihoods_mat[:, i] / constrained_likelihoods_mat[:, i-1] < betas[i], likelihoods_mat[:, i] / psis[i], constrained_likelihoods_mat[:, i-1] * (betas[i] / psis[i]))
        
    return constrained_likelihoods_mat


'''Sort and coarsen grid of lik-ratio values to search over'''
def prepare_grid(
    cfg: DictConfig,
    V, ## 1-D np array, lik-ratio values (unsorted) to process into grid
    n_grid: int = 50, ## int, approximately how many values want to have in resulting grid
    proposal: str = 'unconstrained', ## str, 'unconstrained' or 'safe' to indicate prop dist
):
    G = np.sort(np.unique(V)) ## Want to search in increasing order for fixed-sequence testing (ie, most promising to least promising)
    max_G = G[0]

    ## If wanted to return power-series decreasing grid, rather than based on empirical quantiles
    # return np.concatenate(([np.inf], [v / (1.2**(i)) for i, v in enumerate(np.linspace(sys.float_info.min, max_G, num=n_grid))]))

    if proposal == 'unconstrained':

            # raise ValueError(f"min_value_to_add is None")

        # G = G[G > min_value_to_add] ## For unconstrained, only consider bounds at least equal to 1

        ## Coarsen grid to approximately n_grid elements
        n_curr = len(G)
        k = max(int(n_curr / n_grid), 1)
        G = G[::k]
        # G = np.concatenate(([min_value_to_add], G, [np.inf])) ## For unconstrained, ensure also consider unconstrained policy in grid (np.inf) and 1 as bounds
        G = np.concatenate((G, [np.inf])) ## For unconstrained, ensure also consider unconstrained policy in grid (np.inf) and 1 as bounds

            # G = np.concatenate(([1], G, [np.inf])) ## For unconstrained, ensure also consider unconstrained policy in grid (np.inf) and 1 as bounds
        # G = np.concatenate(([np.inf], G)) ## For unconstrained, ensure also consider unconstrained policy in grid (np.inf) and 1 as bounds

    elif proposal == 'safe':
        # G = G[G<1] ## For safe policy, only consider bounds no greater than 1

        ## Coarsen grid to approximately n_grid elements
        n_curr = len(G)
        k = max(int(n_curr / n_grid), 1) #if n_curr / int(n_curr / n_grid) > n_grid else 1
        G = G[::k]
        G = np.concatenate(([sys.float_info.min], G)) ## For safe, ensure that include minimum positive float value

    elif proposal == 'mixed':
        ## Coarsen grid to approximately n_grid elements
        n_curr = len(G)
        k = max(int(n_curr / n_grid), 1) #if n_curr / int(n_curr / n_grid) > n_grid else 1
        G = G[::k]
        G = np.concatenate(([sys.float_info.min], G, [np.inf])) ## For safe, ensure that include minimum positive float value

    else:
        raise ValueError(f"unrecognized proposal name : {proposal}")

    return G




def get_all_strs_from_nested_dict(nested_dict: Dict[str, Any]) -> List[str]:
    outputs = []
    for k, v in nested_dict.items():
        if not isinstance(v, Mapping):
            outputs.append(f"{k}={v}")
        else:
            for o in get_all_strs_from_nested_dict(v):
                outputs.append(f"{k}.{o}")
    return outputs


def generate_ga_dataset(cfg: DictConfig, fs: LocalOrS3Client) -> str:
    """
    Either run genetic algorithm to generate dataset, or simply return directory containing data.
    """
    python_cmd_str = "python -m synthetic_dataset_generator "
    opts = get_all_strs_from_nested_dict(cfg["evol_dataset_gen"]["args"])
    opts_str = " ".join(opts)
    python_cmd_str += f"{opts_str} "
    num_motifs = cfg["evol_dataset_gen"]["args"]["test_function"]["num_motifs"]
    motif_length = cfg["evol_dataset_gen"]["args"]["test_function"]["motif_length"]
    dim = cfg["evol_dataset_gen"]["args"]["test_function"]["dim"]
    num_particles = cfg["evol_dataset_gen"]["args"]["optimizer"]["num_particles"]
    mutation_prob = cfg["evol_dataset_gen"]["args"]["optimizer"]["mutation_prob"]
    num_opt_steps = cfg["evol_dataset_gen"]["args"]["num_opt_steps"]
    ds_name = f"c{num_motifs}_k{motif_length}_l{dim}_n{num_particles}_pm{mutation_prob}_steps{num_opt_steps}"
    output_dir = (
        f"{cfg.parent_output_dir}/{cfg.run_name}/{ds_name}"
        if cfg.parent_output_dir is not None
        else f"{cfg.local_output_dir}/{cfg.run_name}/{ds_name}"
    )
    if not output_dir.startswith("s3://"):
        os.makedirs(output_dir, exist_ok=True)
    python_cmd_str += f"output_dir={output_dir} "
    output_fp = f"{output_dir}/plain_pairs.jsonl"
    if not cfg.overwrite_ga and fs.exists(output_fp):
        logger.info(f"{output_fp} already exists. Skipping...")
        return output_dir
    slurm_dump_dir = f"{cfg.local_output_dir}/slurm_logs"
    os.makedirs(slurm_dump_dir, exist_ok=True)
    if cfg.run_evol_dataset_gen:
        slurm_kwargs = OmegaConf.to_container(cfg.evol_dataset_gen.slurm_args)
        slurm_kwargs["job_name"] = "ga_seeds"
        submit_cmd_to_slurm(
            python_cmd_str,
            slurm_dump_dir,
            blocking=True,
            path_to_repo=cfg.path_to_repo,
            **slurm_kwargs,
        )

    return output_dir


def create_propen_sft_dataset(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    source_dataset_fp: str,
    filename_prefix: str = "",
    n: int = None,
    initial_sft: bool = False, ## Whether is initialization (False : means policy improvement or extrapolation)
    **extra_kwargs,
) -> str:
    python_cmd_str = "python -m synthetic_dataset_formatter "
    if initial_sft:
        opts = get_all_strs_from_nested_dict(cfg["propen_dataset_formatting_initial_sft"]["args"])
    else:
        opts = get_all_strs_from_nested_dict(cfg["propen_dataset_formatting_sft"]["args"])
    opts_str = " ".join(opts)
    opts_str += (
        f" source_dataset_path={source_dataset_fp} format=dense_neighborhood_pairs "
    )
    if initial_sft:
        pdf_cfg = cfg.propen_dataset_formatting_initial_sft.args
    else:
        pdf_cfg = cfg.propen_dataset_formatting_sft.args

    if initial_sft:
        ## Want same initial SFT dataset for both policy control and no PC
        output_fn = f"{filename_prefix}dense_neighborhood_pairs_xthres{pdf_cfg.dist_x_threshold}_maxinfs{pdf_cfg.max_proportion_infeasible}_{pdf_cfg.n_neighbors}nn.jsonl"
    else:
        ## If not initial SFT, then keep track of whether doing policy control (ie, alpha level)
        output_fn = f"alpha{cfg.conformal_policy_control.alpha}_{filename_prefix}dense_neighborhood_pairs_xthres{pdf_cfg.dist_x_threshold}_maxinfs{pdf_cfg.max_proportion_infeasible}_{pdf_cfg.n_neighbors}nn.jsonl"

    output_fp = (
        f"{cfg.parent_output_dir}/{cfg.run_name}/{output_fn}"
        if cfg.parent_output_dir is not None
        else f"{cfg.local_output_dir}/{cfg.run_name}/{output_fn}"
    )
    if not output_fp.startswith("s3://"):
        os.makedirs(os.path.dirname(output_fp), exist_ok=True)
    opts_str += f"output_path={output_fp} "
    for k, v in extra_kwargs.items():
        opts_str += f"{k}={v} "
    if n is not None:
        opts_str += f"n={n} "
    python_cmd_str += f"{opts_str} "
    overwrite_sft_flag = cfg.overwrite_init_sft_formatter if initial_sft else cfg.overwrite_sft_formatter
    if not overwrite_sft_flag and fs.exists(output_fp):
        logger.info(f"{output_fp} already exists. Skipping...")
        return output_fp
    slurm_dump_dir = f"{cfg.local_output_dir}/slurm_logs"
    os.makedirs(slurm_dump_dir, exist_ok=True)
    if cfg.run_propen_sft_dataset_formatting:
        if initial_sft:
            slurm_kwargs = OmegaConf.to_container(
                cfg.propen_dataset_formatting_initial_sft.slurm_args
            )
            slurm_kwargs["job_name"] = "propen_initial_sft_formatting"
        else:
            slurm_kwargs = OmegaConf.to_container(
                cfg.propen_dataset_formatting_sft.slurm_args
            )
            slurm_kwargs["job_name"] = "propen_sft_formatting"
        submit_cmd_to_slurm(
            python_cmd_str,
            slurm_dump_dir,
            blocking=True,
            path_to_repo=cfg.path_to_repo,
            **slurm_kwargs,
        )
    return output_fp


def create_propen_preference_dataset(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    source_dataset_fp: str,
    filename_prefix: str = "",
    n: int = None,
) -> str:
    python_cmd_str = "python -m synthetic_dataset_formatter "
    opts = get_all_strs_from_nested_dict(
        cfg["propen_dataset_formatting_preference"]["args"]
    )
    opts_str = " ".join(opts)
    opts_str += (
        f" source_dataset_path={source_dataset_fp} format=dense_preference_pairs "
    )
    pdf_cfg = cfg.propen_dataset_formatting_preference.args
    output_fn = f"{filename_prefix}xthres{pdf_cfg.dist_x_threshold}_maxinfs{pdf_cfg.max_proportion_infeasible}_{pdf_cfg.n_neighbors}nn.jsonl"
    output_fp = (
        f"{cfg.parent_output_dir}/{cfg.run_name}/{output_fn}"
        if cfg.parent_output_dir is not None
        else f"{cfg.local_output_dir}/{cfg.run_name}/{output_fn}"
    )
    if not output_fp.startswith("s3://"):
        os.makedirs(os.path.dirname(output_fp), exist_ok=True)
    opts_str += f"output_path={output_fp} "
    if n is not None:
        opts_str += f"n={n} "

    python_cmd_str += f"{opts_str} "
    if not cfg.overwrite_dpo_formatter and fs.exists(output_fp):
        logger.info(f"{output_fp} already exists. Skipping...")
        return output_fp

    slurm_dump_dir = f"{cfg.local_output_dir}/slurm_logs"
    os.makedirs(slurm_dump_dir, exist_ok=True)
    if cfg.run_propen_dpo_dataset_formatting:
        slurm_kwargs = OmegaConf.to_container(
            cfg.propen_dataset_formatting_preference.slurm_args
        )
        slurm_kwargs["job_name"] = "propen_dpo_formatting"
        submit_cmd_to_slurm(
            python_cmd_str,
            slurm_dump_dir,
            blocking=True,
            path_to_repo=cfg.path_to_repo,
            **slurm_kwargs,
        )
    return output_fp


def model_already_trained(
    cfg: DictConfig, fs: LocalOrS3Client, s3_output_dir: str, local_output_dir: str
) -> Optional[str]:
    model_fp_names = ["model.safetensors.index.json", "model.safetensors"]
    model_dir = s3_output_dir if cfg.parent_output_dir is not None else local_output_dir
    for model_fn in model_fp_names:
        fp = f"{model_dir}/{model_fn}"
        if fs.exists(fp):
            return model_dir
    return None

def gpt_model_already_trained(
    cfg: DictConfig, fs: LocalOrS3Client, s3_output_dir: str, local_output_dir: str
) -> Optional[str]:
    # if cfg.overwrite_gpt:
    #     return None
    model_fp_names = ["model.safetensors.index.json", "model.safetensors"]
    model_dir = s3_output_dir if cfg.parent_output_dir is not None else local_output_dir
    for model_fn in model_fp_names:
        fp = f"{model_dir}/{model_fn}"
        if fs.exists(fp):
            return model_dir
    return None


def train_cal_split_gen_outputs(cfg: DictConfig, 
                                fs: LocalOrS3Client, 
                                gen_outputs : str, 
                                sft_dir : str, 
                                sample_num_cal: int = None, 
                                sample_num_train: int = None, 
                                first_iter: bool = False, 
                                setting: str = "sft_CAinit"):

    gen_outputs_df = pd.read_json(gen_outputs, orient="records", lines=True)
    # if first_iter:
    #     output_filename_suffix = f"gens_likelihood_{cfg.iterative_generation.init_args.sample_size}sample_{cfg.iterative_generation.init_args.max_iterations}iter"
    # else:
    # output_filename_suffix = f"gens_likelihood_{cfg.iterative_generation.args.sample_size}sample_{cfg.iterative_generation.args.max_iterations}iter"


    if first_iter:
        cal_output_path = os.path.join(sft_dir, f"{setting}_alpha{cfg.conformal_policy_control.alpha}_cal_gens_all_likelihoods_temp{cfg.temperature_init}.jsonl")
        train_output_path = os.path.join(sft_dir, f"{setting}_alpha{cfg.conformal_policy_control.alpha}_train_gens_all_likelihoods_temp{cfg.temperature_init}.jsonl")
    else:
        cal_output_path = os.path.join(sft_dir, f"{setting}_alpha{cfg.conformal_policy_control.alpha}_cal_gens_all_likelihoods_temp{cfg.temperature}.jsonl")
        train_output_path = os.path.join(sft_dir, f"{setting}_alpha{cfg.conformal_policy_control.alpha}_train_gens_all_likelihoods_temp{cfg.temperature}.jsonl")

    overwrite_split_flag = cfg.overwrite_split_init if first_iter else cfg.overwrite_split

    if not overwrite_split_flag and fs.exists(cal_output_path) and fs.exists(train_output_path):
        ## If not overwriting and both files exist, then load and return dataframes and paths
        cal_df = pd.read_json(cal_output_path, orient="records", lines=True)
        train_df = pd.read_json(train_output_path, orient="records", lines=True)

    else:
        ## Else, either overwriting or files do not exist, so create them

        ## Cal data (sample exchangeably, without replacement, from generated samples)
        if sample_num_cal is not None:
            ## If want to sample desired number, and 2x that desired number is available
            cal_df = gen_outputs_df.sample(n=sample_num_cal, random_state=cfg.random_seed)
        else:
            cal_df = gen_outputs_df.sample(frac=cfg.split.cal_frac, random_state=cfg.random_seed)
        # cal_output_path = "cal_gens_all_likelihoods_temp1.0.jsonl"
        # cal_df = cal_df
        cal_df.to_json(cal_output_path, orient="records", lines=True)


        ## Training data (sample exchangeably, w/o replacement, from *non-cal, deduplicated* generated samples)
        non_cal_gen_outputs_df = gen_outputs_df.drop(cal_df.index) ## non cal
        non_cal_gen_outputs_df_unique = non_cal_gen_outputs_df.drop_duplicates(subset=["particle"]) ## de-duplicate

        if cfg.split.train_sampling_method == "uniform":
            ## Randomly sample sequences for training set
            if sample_num_train is not None:
                train_df = non_cal_gen_outputs_df_unique.sample(n=sample_num_train, random_state=cfg.random_seed)
            elif cfg.split.train_frac_from_non_cal < 1.0:
                train_df = non_cal_gen_outputs_df_unique.sample(frac=cfg.split.train_frac_from_non_cal, random_state=cfg.random_seed)
            else:
                train_df = non_cal_gen_outputs_df_unique
        elif cfg.split.train_sampling_method == "best_scoring":
            ## Deterministically select top scoring sequences
            if sample_num_train is not None:
                train_df = non_cal_gen_outputs_df_unique.nlargest(sample_num_train, "score")
            elif cfg.split.train_frac_from_non_cal < 1.0:
                num_train = int(cfg.split.train_frac_from_non_cal * len(non_cal_gen_outputs_df_unique))
                train_df = non_cal_gen_outputs_df_unique.nlargest(num_train, "score")
            else:
                train_df = non_cal_gen_outputs_df_unique
        else:
            raise ValueError("cfg.split.train_sampling_method {cfg.split.train_sampling_method} not recognized")
        # train_output_path = "cal_gens_all_likelihoods_temp1.0.jsonl"
        train_df.to_json(train_output_path, orient="records", lines=True)

    return cal_df, cal_output_path, train_df, train_output_path


def train_gpt(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    #data_fp: str,
    ga_data_dir: str, ## For GPT pretraining, will use ga_data_dir as main data directory
    gpt_run_name: str,
    model_dir: str = "EleutherAI/pythia-14m",
    train_from_scratch: bool = False,
) -> str:
    

    ## Creating/Getting Output Directories
    test_fn_fp = f"{ga_data_dir}/ehrlich.jsonl" ## Path to Ehrlich function parameters
    os.makedirs(f"{cfg.local_output_dir}/{cfg.run_name}", exist_ok=True)
    output_dir = f"{cfg.local_output_dir}/{cfg.run_name}/{gpt_run_name}"
    s3_output_dir = (
        f"{cfg.parent_output_dir}/{cfg.run_name}/{gpt_run_name}"
        if cfg.parent_output_dir is not None
        else "null"
    )


    ## Loading args
    args = f"--config-name=pythia-2.8b_edit_pairs data_fp={ga_data_dir}/plain_pairs.jsonl " ## Modified relative to train_sft
    args += " ".join(get_all_strs_from_nested_dict(cfg["gpt"]["args"])) + " "
    args += f"test_fn_type=ehrlich test_fn_fp={test_fn_fp} "
    args += f"job_name={gpt_run_name} s3_output_dir={s3_output_dir} "
    args += f"model_config.model_name_or_path={model_dir} "
    args += f"sanity_check={cfg.sanity_check} "
    
    # train from scratch
    if train_from_scratch and hasattr(cfg, "initial_model_config"):
        args += f"train_from_scratch=True "
        for k, v in cfg.initial_model_config.items():
            args += f"+init_model_config.{k}={v} "

    slurm_dump_dir = f"{cfg.local_output_dir}/slurm_logs"

    ## Check if there's already a directory with a pretrained GPT model
    trained_model_dir = gpt_model_already_trained(cfg, fs, s3_output_dir, output_dir)
    if trained_model_dir is not None and not cfg.overwrite_gpt:
        logger.info(f"Trained model already exists in {trained_model_dir}. Skipping...")
        return trained_model_dir
    else:
        if trained_model_dir is None:
            logger.info(f"Did not find trained model, Continuing to train...")
        else:
            trained_model_dir = None
            logger.info(f"Config says to overwrite model (cfg.overwrite_gpt={cfg.overwrite_gpt}), Continuing to train...")
    os.makedirs(slurm_dump_dir, exist_ok=True)
    
    if cfg.run_gpt:
        ## Submit commands for GPT pretraining

        slurm_cfg = cfg.gpt.slurm_args
        # run with ddp (TODO: switch to fsdp)
        gpus_per_node = (
            slurm_cfg.gpus_per_node
            if hasattr(slurm_cfg, "gpus_per_node")
            else int(slurm_cfg.gres.split(":")[-1])
        )
        ## If overwriting GPT, then delete previous checkpoints
        py_cmd = ""
        if cfg.overwrite_gpt:
            py_cmd += f"rm -rf {output_dir}/checkpoint-*\n"

        py_cmd += f"torchrun --standalone --nnodes={slurm_cfg.nodes} --nproc-per-node={gpus_per_node} "
        py_cmd += f"-m finetune_ehrlich {args} training_args.output_dir={output_dir}\n"

        # store return code for the finetuning job so that we can return it later
        py_cmd += f"RETURN_CODE=$?\n"

        # add extra commands for deleting local checkpoints after the job finishes
        # if S3 was used
        if cfg.parent_output_dir is not None:
            py_cmd += f"rm -rf {output_dir}/checkpoint-*/model-*.safetensors\n"
            py_cmd += f"rm -rf {output_dir}/model-*.safetensors\n"
            py_cmd += f"rm -rf {output_dir}/checkpoint-*/optimizer.pt\n"

        # return the exit code of the finetuning job
        py_cmd += "exit ${RETURN_CODE}\n"

        slurm_kwargs = OmegaConf.to_container(cfg.gpt.slurm_args)
        slurm_kwargs["job_name"] = "gpt"
        submit_cmd_to_slurm(
            py_cmd,
            slurm_dump_dir,
            blocking=True,
            path_to_repo=cfg.path_to_repo,
            **slurm_kwargs,
        )
    return_path = s3_output_dir if cfg.parent_output_dir is not None else output_dir
    return return_path



def train_initial_sft(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    data_fp: str,
    ga_data_dir: str,
    initial_sft_run_name: str,
    model_dir: str = "EleutherAI/pythia-2.8b",
    train_from_scratch: bool = False,
) -> str:

    ## Creating/Getting Output Directories
    test_fn_fp = f"{ga_data_dir}/ehrlich.jsonl" ## Path to Ehrlich function parameters
    os.makedirs(f"{cfg.local_output_dir}/{cfg.run_name}", exist_ok=True)
    output_dir = f"{cfg.local_output_dir}/{cfg.run_name}/{initial_sft_run_name}"
    s3_output_dir = (
        f"{cfg.parent_output_dir}/{cfg.run_name}/{initial_sft_run_name}"
        if cfg.parent_output_dir is not None
        else "null"
    )

    ## Loading args
    args = f"--config-name=pythia-2.8b_edit_pairs data_fp={data_fp} "
    args += " ".join(get_all_strs_from_nested_dict(cfg["initial_sft"]["args"])) + " "
    args += f"test_fn_type=ehrlich test_fn_fp={test_fn_fp} "
    args += f"job_name={initial_sft_run_name} s3_output_dir={s3_output_dir} "
    args += f"model_config.model_name_or_path={model_dir} "
    args += f"sanity_check={cfg.sanity_check} "
    

    # train from scratch
    if train_from_scratch and hasattr(cfg, "initial_model_config"):
        args += f"train_from_scratch=True "
        for k, v in cfg.initial_model_config.items():
            args += f"+init_model_config.{k}={v} "

    slurm_dump_dir = f"{cfg.local_output_dir}/slurm_logs"

    trained_model_dir = model_already_trained(cfg, fs, s3_output_dir, output_dir)
    if trained_model_dir is not None and not cfg.overwrite_initial_sft:
        logger.info(f"Trained model already exists in {trained_model_dir}. Skipping...")
        return trained_model_dir
    else:
        if trained_model_dir is None:
            logger.info(f"Did not find trained model, Continuing to train...")
        else:
            trained_model_dir = None
            logger.info(f"Config says to overwrite model (cfg.overwrite_initial_sft={cfg.overwrite_initial_sft}), Continuing to train...")
    os.makedirs(slurm_dump_dir, exist_ok=True)
    if cfg.run_initial_sft:
        slurm_cfg = cfg.initial_sft.slurm_args
        # run with ddp (TODO: switch to fsdp)
        gpus_per_node = (
            slurm_cfg.gpus_per_node
            if hasattr(slurm_cfg, "gpus_per_node")
            else int(slurm_cfg.gres.split(":")[-1])
        )
        ## If overwriting Initial SFT, then delete previous checkpoints
        py_cmd = ""
        if cfg.overwrite_initial_sft:
            py_cmd += f"rm -rf {output_dir}/checkpoint-*\n"

        py_cmd += f"torchrun --standalone --nnodes={slurm_cfg.nodes} --nproc-per-node={gpus_per_node} "
        py_cmd += f"-m finetune_ehrlich {args} training_args.output_dir={output_dir}\n"

        # store return code for the finetuning job so that we can return it later
        py_cmd += f"RETURN_CODE=$?\n"

        # add extra commands for deleting local checkpoints after the job finishes
        # if S3 was used
        if cfg.parent_output_dir is not None:
            py_cmd += f"rm -rf {output_dir}/checkpoint-*/model-*.safetensors\n"
            py_cmd += f"rm -rf {output_dir}/model-*.safetensors\n"
            py_cmd += f"rm -rf {output_dir}/checkpoint-*/optimizer.pt\n"

        # return the exit code of the finetuning job
        py_cmd += "exit ${RETURN_CODE}\n"

        slurm_kwargs = OmegaConf.to_container(cfg.initial_sft.slurm_args)
        slurm_kwargs["job_name"] = "initial_sft"
        submit_cmd_to_slurm(
            py_cmd,
            slurm_dump_dir,
            blocking=True,
            path_to_repo=cfg.path_to_repo,
            **slurm_kwargs,
        )
    return_path = s3_output_dir if cfg.parent_output_dir is not None else output_dir
    return return_path




def train_sft(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    data_fp: str,
    ga_data_dir: str,
    sft_run_name: str,
    model_dir: str = "EleutherAI/pythia-2.8b",
    train_from_scratch: bool = False,
) -> str:

    ## Creating/Getting Output Directories
    test_fn_fp = f"{ga_data_dir}/ehrlich.jsonl" ## Path to Ehrlich function parameters
    os.makedirs(f"{cfg.local_output_dir}/{cfg.run_name}", exist_ok=True)
    output_dir = f"{cfg.local_output_dir}/{cfg.run_name}/{sft_run_name}"
    s3_output_dir = (
        f"{cfg.parent_output_dir}/{cfg.run_name}/{sft_run_name}"
        if cfg.parent_output_dir is not None
        else "null"
    )

    ## Loading args
    args = f"--config-name=pythia-2.8b_edit_pairs data_fp={data_fp} "
    args += " ".join(get_all_strs_from_nested_dict(cfg["sft"]["args"])) + " "
    args += f"test_fn_type=ehrlich test_fn_fp={test_fn_fp} "
    args += f"job_name={sft_run_name} s3_output_dir={s3_output_dir} "
    args += f"model_config.model_name_or_path={model_dir} "
    args += f"sanity_check={cfg.sanity_check} "
    

    # train from scratch
    if train_from_scratch and hasattr(cfg, "initial_model_config"):
        args += f"train_from_scratch=True "
        for k, v in cfg.initial_model_config.items():
            args += f"+init_model_config.{k}={v} "

    slurm_dump_dir = f"{cfg.local_output_dir}/slurm_logs"

    trained_model_dir = model_already_trained(cfg, fs, s3_output_dir, output_dir)
    if trained_model_dir is not None and not cfg.overwrite_sft:
        logger.info(f"Trained model already exists in {trained_model_dir}. Skipping...")
        return trained_model_dir
    else:
        if trained_model_dir is None:
            logger.info(f"Did not find trained model, Continuing to train...")
        else:
            trained_model_dir = None
            logger.info(f"Config says to overwrite model (cfg.overwrite_sft={cfg.overwrite_sft}), Continuing to train...")
    os.makedirs(slurm_dump_dir, exist_ok=True)
    if cfg.run_sft:
        slurm_cfg = cfg.sft.slurm_args
        # run with ddp (TODO: switch to fsdp)
        gpus_per_node = (
            slurm_cfg.gpus_per_node
            if hasattr(slurm_cfg, "gpus_per_node")
            else int(slurm_cfg.gres.split(":")[-1])
        )
        ## If overwriting SFT, then delete previous checkpoints
        py_cmd = ""
        if cfg.overwrite_sft:
            py_cmd += f"rm -rf {output_dir}/checkpoint-*\n"

        py_cmd += f"torchrun --standalone --nnodes={slurm_cfg.nodes} --nproc-per-node={gpus_per_node} "
        py_cmd += f"-m finetune_ehrlich {args} training_args.output_dir={output_dir}\n"

        # store return code for the finetuning job so that we can return it later
        py_cmd += f"RETURN_CODE=$?\n"

        # add extra commands for deleting local checkpoints after the job finishes
        # if S3 was used
        if cfg.parent_output_dir is not None:
            py_cmd += f"rm -rf {output_dir}/checkpoint-*/model-*.safetensors\n"
            py_cmd += f"rm -rf {output_dir}/model-*.safetensors\n"
            py_cmd += f"rm -rf {output_dir}/checkpoint-*/optimizer.pt\n"

        # return the exit code of the finetuning job
        py_cmd += "exit ${RETURN_CODE}\n"

        slurm_kwargs = OmegaConf.to_container(cfg.sft.slurm_args)
        slurm_kwargs["job_name"] = "sft"
        submit_cmd_to_slurm(
            py_cmd,
            slurm_dump_dir,
            blocking=True,
            path_to_repo=cfg.path_to_repo,
            **slurm_kwargs,
        )
    return_path = s3_output_dir if cfg.parent_output_dir is not None else output_dir
    return return_path




def train_dpo(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    ref_model_path: str,
    data_fp: str,
    ga_data_dir: str,
    run_name: str,
) -> str:
    test_fn_fp = f"{ga_data_dir}/ehrlich.jsonl"
    output_dir = f"{cfg.local_output_dir}/{cfg.run_name}/{run_name}"
    s3_output_dir = (
        f"{cfg.parent_output_dir}/{cfg.run_name}/{run_name}"
        if cfg.parent_output_dir is not None
        else "null"
    )
    trained_model_dir = model_already_trained(cfg, fs, s3_output_dir, output_dir)
    if trained_model_dir is not None and not cfg.overwrite_dpo:
        logger.info(f"Trained model already exists in {trained_model_dir}. Skipping...")
        return trained_model_dir
    else:
        if trained_model_dir is None:
            logger.info(f"Did not find trained model, Continuing to train...")
        else:
            trained_model_dir = None
            logger.info(f"Config says to overwrite model (cfg.overwrite_dpo={cfg.overwrite_dpo}), Continuing to train...")
    
    slurm_dump_dir = f"{cfg.local_output_dir}/slurm_logs"

    os.makedirs(slurm_dump_dir, exist_ok=True)
    args = f"--config-name=pythia-2.8b-dpo "
    args += " ".join(get_all_strs_from_nested_dict(cfg["dpo"]["args"])) + " "
    args += f"data_fp={data_fp} "
    args += f"dpo_config.run_name={run_name} "
    args += f"model_config.model_name_or_path={ref_model_path} "
    args += f"test_fn_fp={test_fn_fp} "
    args += f"job_name={run_name} "
    args += f"s3_output_dir={s3_output_dir} "
    args += f"dpo_script_args.sanity_check={cfg.sanity_check} "

    slurm_dump_dir = f"{cfg.local_output_dir}/slurm_logs"
    os.makedirs(slurm_dump_dir, exist_ok=True)
    slurm_cfg = cfg.dpo.slurm_args
    # run with ddp (TODO: switch to fsdp)
    gpus_per_node = (
        slurm_cfg.gpus_per_node
        if hasattr(slurm_cfg, "gpus_per_node")
        else int(slurm_cfg.gres.split(":")[-1])
    )
    py_cmd = f"torchrun --standalone --nnodes={slurm_cfg.nodes} --nproc-per-node={gpus_per_node} "
    py_cmd += f"-m dpo {args} dpo_config.output_dir={output_dir}\n"

    # store return code for the training job so that we can return it later
    py_cmd += f"RETURN_CODE=$?\n"

    # add extra commands for deleting local checkpoints after the job finishes
    # since they will have already been transferred to S3
    if cfg.parent_output_dir is not None:
        py_cmd += f"rm -rf {output_dir}/checkpoint-*/model-*.safetensors\n"
        py_cmd += f"rm -rf {output_dir}/model-*.safetensors\n"
        py_cmd += f"rm -rf {output_dir}/checkpoint-*/optimizer.pt\n"

    # return the exit code of the training job
    py_cmd += "exit ${RETURN_CODE}\n"

    slurm_kwargs = OmegaConf.to_container(cfg.dpo.slurm_args)
    slurm_kwargs["job_name"] = "dpo"
    submit_cmd_to_slurm(
        py_cmd,
        slurm_dump_dir,
        blocking=True,
        path_to_repo=cfg.path_to_repo,
        **slurm_kwargs,
    )
    return_path = s3_output_dir if cfg.parent_output_dir is not None else output_dir
    return return_path


def train_marge(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    ref_model_path: str,
    data_fp: str,
    ga_data_dir: str,
    run_name: str,
) -> str:
    test_fn_fp = f"{ga_data_dir}/ehrlich.jsonl"
    output_dir = f"{cfg.local_output_dir}/{cfg.run_name}/{run_name}"
    s3_output_dir = (
        f"{cfg.parent_output_dir}/{cfg.run_name}/{run_name}"
        if cfg.parent_output_dir is not None
        else "null"
    )
    trained_model_dir = model_already_trained(cfg, fs, s3_output_dir, output_dir)
    if trained_model_dir is not None and not cfg.overwrite_marge:
        logger.info(f"Trained model already exists in {trained_model_dir}. Skipping...")
        return trained_model_dir
    else:
        if trained_model_dir is None:
            logger.info(f"Did not find trained model, Continuing to train...")
        else:
            trained_model_dir = None
            logger.info(f"Config says to overwrite model (cfg.overwrite_marge={cfg.overwrite_marge}), Continuing to train...")
    args = f"--config-name=pythia-2.8b-marge "
    args += " ".join(get_all_strs_from_nested_dict(cfg["marge"]["args"])) + " "
    args += f"data_fp={data_fp} "
    args += f"marge_config.run_name={run_name} "
    args += f"model_config.model_name_or_path={ref_model_path} "
    args += f"test_fn_fp={test_fn_fp} "
    args += f"job_name={run_name} "
    args += f"s3_output_dir={s3_output_dir} "
    args += f"dpo_script_args.sanity_check={cfg.sanity_check} "

    slurm_dump_dir = f"{cfg.local_output_dir}/slurm_logs"
    os.makedirs(slurm_dump_dir, exist_ok=True)
    slurm_cfg = cfg.marge.slurm_args
    # run with ddp (TODO: switch to fsdp)
    gpus_per_node = (
        slurm_cfg.gpus_per_node
        if hasattr(slurm_cfg, "gpus_per_node")
        else int(slurm_cfg.gres.split(":")[-1])
    )
    py_cmd = f"torchrun --standalone --nnodes={slurm_cfg.nodes} --nproc-per-node={gpus_per_node} "
    py_cmd += f"-m marge {args} marge_config.output_dir={output_dir}\n"

    # store return code for the training job so that we can return it later
    py_cmd += f"RETURN_CODE=$?\n"

    # add extra commands for deleting local checkpoints after the job finishes
    # since they will have already been transferred to S3
    if cfg.parent_output_dir is not None:
        py_cmd += f"rm -rf {output_dir}/checkpoint-*/model-*.safetensors\n"
        py_cmd += f"rm -rf {output_dir}/model-*.safetensors\n"
        py_cmd += f"rm -rf {output_dir}/checkpoint-*/optimizer.pt\n"

    # return the exit code of the training job
    py_cmd += "exit ${RETURN_CODE}\n"

    slurm_kwargs = OmegaConf.to_container(cfg.marge.slurm_args)
    slurm_kwargs["job_name"] = "marge"
    submit_cmd_to_slurm(
        py_cmd,
        slurm_dump_dir,
        blocking=True,
        path_to_repo=cfg.path_to_repo,
        **slurm_kwargs,
    )
    return_path = s3_output_dir if cfg.parent_output_dir is not None else output_dir
    return return_path



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
    cfg: DictConfig, fs: LocalOrS3Client, old_fps: List[str], curr_fp: str
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
    num_rows_per_old_fp = int(
        math.ceil(
            num_curr_rows
            * cfg.proportion_of_old_data
            / (num_old_datasets * (1.0 - cfg.proportion_of_old_data))
        )
    )
    for old_fp in old_fps:
        df = pd.read_json(old_fp, orient="records", lines=True)
        df = df.sample(n=min(len(df), num_rows_per_old_fp), random_state=cfg.seed)
        combined_datasets.append(df)
    combined_datasets = pd.concat(combined_datasets)

    combined_datasets.to_json(output_fp, orient="records", lines=True)
    logger.info(f"Combined current dataset with old datasets. Written to {output_fp}.")
    return output_fp


# def run_unconditional_generation(
#     cfg: DictConfig,
#     fs: LocalOrS3Client,
#     data_fp: str,
#     data_dir: str,
#     model_dir: str,
#     particle_field: str = "particle",
#     # lower_score_particle_field: str = "lower_score_particle",
#     score_field: str = "score",
#     # higher_score_field: str = "higher_score",
#     temps: List[float] = [1.0],
# ) -> str:
#     """
#     Runs unconditional generation jobs, combines the outputs, and returns the combined output filepath.
#     """
#     opt_str = " ".join(
#         get_all_strs_from_nested_dict(cfg["unconditional_generation"]["args"])
#     )
#     ## Note: In unconditional_generation, the training particles provided are *not* use as prompts, they are 
#     ## only provided to check how much the density model is reproducing examples from training
#     args = f"{opt_str} data_path={data_fp} model_name_or_path={model_dir} output_dir={model_dir} "
#     args += f"test_fn_fp={data_dir}/ehrlich.jsonl "
#     args += f"particle_field={particle_field} "
#     # args += f"lower_score_particle_field={lower_score_particle_field} "
#     args += f"score_field={score_field} "
#     # args += f"higher_score_field={higher_score_field} "
#     args += f"sanity_check={cfg.sanity_check} "

#     output_filename_prefix = f"gens_uncon_likelihood_{cfg.unconditional_generation.args.sample_size}sample_{cfg.unconditional_generation.args.max_iterations}iter"
#     # greedy_decoding_gen_args = f"generation_config.do_sample=False generation_config.num_beams=1 batch_size={cfg.greedy_gen_batch_size}"
#     temp_sampling_gen_args = [
#         f"generation_config.do_sample=True generation_config.num_beams=1 "
#         + f"+generation_config.temperature={temp} "
#         + f"generation_config.num_return_sequences={cfg.unconditional_sampling_num_return_sequences} "
#         + f"batch_size={cfg.sampling_gen_batch_size} "
#         for temp in temps
#     ]
    
#     all_gen_args = temp_sampling_gen_args
#     output_filenames = [f"{output_filename_prefix}_temp{temp}_{cfg.unconditional_sampling_num_return_sequences}seqs.jsonl" for temp in temps]

#     output_filepaths = [f"{model_dir}/{output_fn}" for output_fn in output_filenames]
#     combined_outputs_fp = f"{model_dir}/{output_filename_prefix}.jsonl"
#     slurm_dump_dir = f"{cfg.local_output_dir}/slurm_logs"
#     os.makedirs(slurm_dump_dir, exist_ok=True)
#     hd = None
#     if cfg.run_uncon_gen:
#         all_args = []
#         for gen_args, output_fn in zip(all_gen_args, output_filenames):
#             if not cfg.overwrite_ig and fs.exists(f"{model_dir}/{output_fn}"):
#                 logger.info(f"{model_dir}/{output_fn} already exists. Skipping...")
#             else:
#                 all_args.append(f"{args} {gen_args} output_filename={output_fn}")
#         all_python_commands = [f"python -m unconditional_generation {a}" for a in all_args]
#         slurm_kwargs = OmegaConf.to_container(cfg.unconditional_generation.slurm_args)
#         slurm_kwargs["job_name"] = "uncon_gen"
#         job_submissions = [
#             submit_cmd_to_slurm(
#                 py_cmd,
#                 slurm_dump_dir,
#                 blocking=False,
#                 path_to_repo=cfg.path_to_repo,
#                 **slurm_kwargs,
#             )
#             for py_cmd in all_python_commands
#         ]
#         wait_for_slurm_jobs_to_complete(job_submissions)
#         hd = combine_datasets(cfg, fs, output_filepaths, combined_outputs_fp)
#     return combined_outputs_fp, hd





def run_initial_generation(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    data_fp: str,
    data_dir: str,
    model_dir: str,
    higher_score_particle_field: str = "higher_score_particle",
    lower_score_particle_field: str = "lower_score_particle",
    lower_score_field: str = "lower_score",
    higher_score_field: str = "higher_score",
    temps: List[float] = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6],
    return_seeds: bool = True
):
    """
    Runs initial generation jobs, combines the outputs, and returns the combined output filepath.
    """
    opt_str = " ".join(
        get_all_strs_from_nested_dict(cfg["initial_generation"]["args"])
    )
    args = f"{opt_str} data_path={data_fp} model_name_or_path={model_dir} output_dir={model_dir} "
    args += f"test_fn_fp={data_dir}/ehrlich.jsonl "
    args += f"higher_score_particle_field={higher_score_particle_field} "
    args += f"lower_score_particle_field={lower_score_particle_field} "
    args += f"lower_score_field={lower_score_field} "
    args += f"higher_score_field={higher_score_field} "
    args += f"sanity_check={cfg.sanity_check} "

    output_filename_prefix = f"alpha{cfg.conformal_policy_control.alpha}_gens_init_likelihood_{cfg.initial_generation.args.sample_size}sample_{cfg.initial_generation.args.max_iterations}iter"
    # else:
    #     ## If last iteration of initial SFT, then use sample_size from cfg.iterative_generation (not from cfg.initial_generation)
    #     output_filename_prefix = f"gens_init_likelihood_{cfg.iterative_generation.args.sample_size}sample_{cfg.initial_generation.args.max_iterations}iter"

    greedy_decoding_gen_args = f"generation_config.do_sample=False generation_config.num_beams=1 batch_size={cfg.greedy_gen_batch_size}"
    temp_sampling_gen_args = [
        f"generation_config.do_sample=True generation_config.num_beams=1 "
        + f"+generation_config.temperature={temp} "
        + f"generation_config.num_return_sequences={cfg.init_generation_sampling_num_return_sequences} "
        + f"batch_size={cfg.sampling_gen_batch_size} "
        for temp in temps
    ]
    if cfg.greedy_decoding:
        all_gen_args = [greedy_decoding_gen_args, *temp_sampling_gen_args]
        output_filenames = [
            f"{output_filename_prefix}_greedy.jsonl",
            *[
                f"{output_filename_prefix}_temp{temp}_{cfg.init_generation_sampling_num_return_sequences}seqs.jsonl"
                for temp in temps
            ],
        ]
    else:
        all_gen_args = temp_sampling_gen_args
        output_filenames = [f"{output_filename_prefix}_temp{temp}_{cfg.init_generation_sampling_num_return_sequences}seqs.jsonl" for temp in temps]

    seeds_filenames = [f'seeds_{output_filename}' for output_filename in output_filenames]
    seeds_filepaths = [f"{model_dir}/{seeds_fn}" for seeds_fn in seeds_filenames]

    output_filepaths = [f"{model_dir}/{output_fn}" for output_fn in output_filenames]
    combined_outputs_fp = f"{model_dir}/{output_filename_prefix}.jsonl"
    slurm_dump_dir = f"{cfg.local_output_dir}/slurm_logs"
    os.makedirs(slurm_dump_dir, exist_ok=True)
    hd = None


    if cfg.run_init_gen:
        all_args = []
        for gen_args, output_fn in zip(all_gen_args, output_filenames):
            if not cfg.overwrite_initg and fs.exists(f"{model_dir}/{output_fn}"):
                logger.info(f"{model_dir}/{output_fn} already exists. Skipping...")
            else:
                all_args.append(f"{args} {gen_args} output_filename={output_fn}")
        all_python_commands = [f"python -m initial_generation {a}" for a in all_args]
        slurm_kwargs = OmegaConf.to_container(cfg.initial_generation.slurm_args)
        slurm_kwargs["job_name"] = "init_gen"
        job_submissions = [
            submit_cmd_to_slurm(
                py_cmd,
                slurm_dump_dir,
                blocking=False,
                path_to_repo=cfg.path_to_repo,
                **slurm_kwargs,
            )
            for py_cmd in all_python_commands
        ]
        wait_for_slurm_jobs_to_complete(job_submissions)
        hd = combine_datasets(cfg, fs, output_filepaths, combined_outputs_fp)
    if return_seeds:
        return combined_outputs_fp, output_filepaths, hd, seeds_filepaths
    else:
        return combined_outputs_fp, hd




def run_contrastive_generation(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    data_fp_list: List[str],
    data_dir: str,
    model_dir_list: List[str],
    particle_field: str = "higher_score_particle",
    score_field: str = "score",
    temps: List[float] = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6],
) -> str:
    """
    Runs contrastive generation jobs, combines the outputs, and returns the combined output filepath.
    """
    opt_str = " ".join(
        get_all_strs_from_nested_dict(cfg["contrastive_generation"]["args"])
    )

    ## Format lists of strings into a long string that python and hydra can interpret
    data_fp_list_str = f"\\['{data_fp_list[0]}'"
    model_dir_list_str = f"\\['{model_dir_list[0]}'"
    for i in range(1, len(data_fp_list)):
        data_fp_list_str += f",'{data_fp_list[i]}'"
        model_dir_list_str += f",'{model_dir_list[i]}'"
    data_fp_list_str += "\\]"
    model_dir_list_str += "\\]"


    args = f"{opt_str} data_path_list={data_fp_list_str} model_name_or_path_list={model_dir_list_str} output_dir={model_dir_list[-1]} "
    args += f"test_fn_fp={data_dir}/ehrlich.jsonl "
    args += f"particle_field={particle_field} "
    args += f"score_field={score_field} "
    args += f"sanity_check={cfg.sanity_check} "

    output_filename_prefix = f"alpha{cfg.conformal_policy_control.alpha}_contrast_gens_likelihood_{cfg.contrastive_generation.args.sample_size}sample"
    greedy_decoding_gen_args = f"generation_config.do_sample=False generation_config.num_beams=1 batch_size={cfg.greedy_gen_batch_size}"
    temp_sampling_gen_args = [
        f"generation_config.do_sample=True generation_config.num_beams=1 "
        + f"+generation_config.temperature={temp} "
        + f"generation_config.num_return_sequences={cfg.generation_sampling_num_return_sequences} "
        + f"batch_size={cfg.sampling_gen_batch_size} "
        for temp in temps
    ]
    if cfg.contrastive_generation.greedy_decoding:
        all_gen_args = [greedy_decoding_gen_args, *temp_sampling_gen_args]
        output_filenames = [
            f"{output_filename_prefix}_greedy.jsonl",
            *[
                f"{output_filename_prefix}_temp{temp}_{cfg.generation_sampling_num_return_sequences}seqs.jsonl"
                for temp in temps
            ],
        ]
    else:
        all_gen_args = temp_sampling_gen_args
        output_filenames = [f"{output_filename_prefix}_temp{temp}_{cfg.generation_sampling_num_return_sequences}seqs.jsonl" for temp in temps]

    output_filepaths = [f"{model_dir_list[-1]}/{output_fn}" for output_fn in output_filenames]
    combined_outputs_fp = f"{model_dir_list[-1]}/{output_filename_prefix}.jsonl"
    slurm_dump_dir = f"{cfg.local_output_dir}/slurm_logs"
    os.makedirs(slurm_dump_dir, exist_ok=True)
    hd = None
    if cfg.run_contrast_gen:
        all_args = []
        for gen_args, output_fn in zip(all_gen_args, output_filenames):
            if not cfg.overwrite_cg and fs.exists(f"{model_dir_list[-1]}/{output_fn}"):
                logger.info(f"{model_dir_list[-1]}/{output_fn} already exists. Skipping contrastive generation...")
            else:
                logger.info(f"Running contrastive generation...")
                all_args.append(f"{args} {gen_args} output_filename={output_fn}")
        all_python_commands = [f"python -m contrastive_generation {a}" for a in all_args]
        slurm_kwargs = OmegaConf.to_container(cfg.contrastive_generation.slurm_args)
        slurm_kwargs["job_name"] = "contrast_gen"
        job_submissions = [
            submit_cmd_to_slurm(
                py_cmd,
                slurm_dump_dir,
                blocking=False,
                path_to_repo=cfg.path_to_repo,
                **slurm_kwargs,
            )
            for py_cmd in all_python_commands
        ]
        wait_for_slurm_jobs_to_complete(job_submissions)
        hd = combine_datasets(cfg, fs, output_filepaths, combined_outputs_fp)
    return combined_outputs_fp, hd



def run_iterative_generation(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    data_fp: str, ## Either path to paired training data (to select seeds from) or path to pre-selected seeds (unpaired)
    data_dir: str,
    model_dir: str,
    output_dir: str = None,
    higher_score_particle_field: str = "higher_score_particle",
    lower_score_particle_field: str = "lower_score_particle",
    lower_score_field: str = "lower_score",
    higher_score_field: str = "higher_score",
    temps: List[float] = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6],
    return_seeds: bool = False,
    first_iter: bool = False,
    model_idx: int = 0, ## Time index of model using for (unconstrained) generation
    call_idx: int = 0, ## Index for this generation has been called, including current, for same model directory
    proportion_of_target_n_accepted: float = None, ## If being run as submodule of AR-sampling, the proportion of target samples accepted
    # proposal: str = 'unconstrained',
):
    """
    Runs iterative generation jobs, combines the outputs, and returns the combined output filepath.
    """


    if output_dir == None:
        output_dir = model_dir

    opt_str = " ".join(
        get_all_strs_from_nested_dict(cfg["iterative_generation"]["args"])
    )
    args = f"{opt_str} data_path={data_fp} model_name_or_path={model_dir} output_dir={output_dir} "
    args += f"test_fn_fp={data_dir}/ehrlich.jsonl "
    args += f"higher_score_particle_field={higher_score_particle_field} "
    args += f"lower_score_particle_field={lower_score_particle_field} "
    args += f"lower_score_field={lower_score_field} "
    args += f"higher_score_field={higher_score_field} "
    args += f"sanity_check={cfg.sanity_check} "
    # args += f"first_iter={cfg.first_iter}"

    if first_iter:
        args += f"sample_size={cfg.iterative_generation.init_args.sample_size} "
        args += f"max_iterations={cfg.iterative_generation.init_args.max_iterations} "
        args += f"sampling_method={cfg.iterative_generation.init_args.sampling_method} "
        output_filename_prefix = f"gens_likelihood_{cfg.iterative_generation.init_args.sample_size}sample_{cfg.iterative_generation.init_args.max_iterations}iter"
    else:
        args += f"sample_size={cfg.iterative_generation.args.sample_size} "
        args += f"max_iterations={cfg.iterative_generation.args.max_iterations} "
        args += f"sampling_method={cfg.iterative_generation.args.sampling_method} "
        if proportion_of_target_n_accepted is not None:
            output_filename_prefix = f"alpha{cfg.conformal_policy_control.alpha}_model{model_idx}_cn{call_idx}_propAcc{proportion_of_target_n_accepted:.3g}_gens_likelihood_{cfg.iterative_generation.args.sample_size}sample_{cfg.iterative_generation.args.max_iterations}iter"
        else:
            output_filename_prefix = f"alpha{cfg.conformal_policy_control.alpha}_model{model_idx}_cn{call_idx}_gens_likelihood_{cfg.iterative_generation.args.sample_size}sample_{cfg.iterative_generation.args.max_iterations}iter"

    greedy_decoding_gen_args = f"generation_config.do_sample=False generation_config.num_beams=1 batch_size={cfg.greedy_gen_batch_size}"
    temp_sampling_gen_args = [
        f"generation_config.do_sample=True generation_config.num_beams=1 "
        + f"+generation_config.temperature={temp} "
        + f"generation_config.num_return_sequences={cfg.generation_sampling_num_return_sequences} "
        + f"batch_size={cfg.sampling_gen_batch_size} "
        for temp in temps
    ]
    if cfg.greedy_decoding:
        all_gen_args = [greedy_decoding_gen_args, *temp_sampling_gen_args]
        output_filenames = [
            f"{output_filename_prefix}_greedy.jsonl",
            *[
                f"{output_filename_prefix}_temp{temp}_{cfg.generation_sampling_num_return_sequences}seqs.jsonl"
                for temp in temps
            ],
        ]
    else:
        all_gen_args = temp_sampling_gen_args

        output_filenames = [f"{output_filename_prefix}_temp{temp}_{cfg.generation_sampling_num_return_sequences}seqs.jsonl" for temp in temps]

    seeds_filenames = [f'seeds_{output_filename}' for output_filename in output_filenames]
    seeds_filepaths = [f"{output_dir}/{seeds_fn}" for seeds_fn in seeds_filenames]
    
    output_filepaths = [f"{output_dir}/{output_fn}" for output_fn in output_filenames]
    combined_outputs_fp = f"{output_dir}/{output_filename_prefix}.jsonl"
    slurm_dump_dir = f"{cfg.local_output_dir}/slurm_logs"
    os.makedirs(slurm_dump_dir, exist_ok=True)
    hd = None
    if cfg.run_iter_gen:
        all_args = []
        for gen_args, output_fn in zip(all_gen_args, output_filenames):
            ## If first_iter, use overwrite_initg flag rather than overwrite_ig
            overwrite_flag = cfg.overwrite_initg if first_iter else cfg.overwrite_ig
            if not overwrite_flag and fs.exists(f"{output_dir}/{output_fn}"):
                logger.info(f"{output_dir}/{output_fn} already exists. Skipping...")
            else:
                all_args.append(f"{args} {gen_args} output_filename={output_fn}")
        all_python_commands = [f"python -m iterative_generation2 {a}" for a in all_args]
        slurm_kwargs = OmegaConf.to_container(cfg.iterative_generation.slurm_args)
        slurm_kwargs["job_name"] = "iter_gen"
        job_submissions = [
            submit_cmd_to_slurm(
                py_cmd,
                slurm_dump_dir,
                blocking=False,
                path_to_repo=cfg.path_to_repo,
                **slurm_kwargs,
            )
            for py_cmd in all_python_commands
        ]
        wait_for_slurm_jobs_to_complete(job_submissions)
        hd = combine_datasets(cfg, fs, output_filepaths, combined_outputs_fp)

    if return_seeds:
        return combined_outputs_fp, output_filepaths, hd, seeds_filepaths
    else:
        return combined_outputs_fp, output_filepaths, hd




def get_seeds_from_training_data(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    training_data_fp: str,
    output_dir: str,
    sample_size: int,
    sampling_method: str = "best_scoring",
    higher_score_particle_field: str = "higher_score_particle",
    lower_score_particle_field: str = "lower_score_particle",
    lower_score_field: str = "lower_score",
    higher_score_field: str = "higher_score",
    pi_optimizer_name: str = "sft",
    setting: str = "sft_CAinit",
    random_seed: int = 0,
) -> str:

    train_df = pd.read_json(training_data_fp, orient="records", lines=True)

    train_df = train_df.loc[train_df[lower_score_particle_field].astype(str).drop_duplicates().index]


    if sampling_method == "best_scoring":
        train_df = train_df.sort_values(by=[lower_score_field], ascending=True)[: sample_size]
    elif sampling_method == "uniform":
        train_df = train_df.sample(n=min(len(train_df), sample_size), random_state=random_seed)
    else:
        raise ValueError(f"Unknown sampling method '{sampling_method}.'")

    train_df_selected = train_df[[lower_score_particle_field, lower_score_field]]

    output_fp = os.path.join(output_dir, f'seeds_from_{os.path.basename(training_data_fp)}')

    ## Rename so that selected particles are used as prompts instead of outputs
    train_df_selected = train_df_selected.rename(columns={lower_score_particle_field: higher_score_particle_field, lower_score_field: 'score'})
    # train_df_selected = train_df_selected.rename(columns={lower_score_particle_field: higher_score_particle_field, lower_score_field: 'score'})
    
    if pi_optimizer_name == "dpo":
        train_df_selected = train_df_selected.rename(columns={higher_score_particle_field : 'prompt', lower_score_particle_field: 'chosen',higher_score_field : 'prompt_score', lower_score_field: 'chosen_score'})

    output_fp = os.path.join(os.path.dirname(output_fp), f"{setting}_{os.path.basename(output_fp)}")
    train_df_selected.to_json(output_fp, orient="records", lines=True)

    return output_fp


def get_num_safe_actions(cfg, cal_infeasible_indicators, cal_lik_numerator, cal_lik_denominator, prop_lik_numerator, prop_lik_denominator, n_target):
    
    ## Unnormalized cal weights
    w_cal = cal_lik_numerator / cal_lik_denominator
    sum_w_cal = np.sum(w_cal)

    ## Unnormalized estimated prop weight
    w_test = np.mean(prop_lik_numerator / prop_lik_denominator)

    for n in range(1, n_target + 1)[::-1]:
        w_test_curr = n * w_test # * 2 #* (1 + cfg.conformal_policy_control.alpha)

        sum_w_cal_test = sum_w_cal + w_test_curr

        w_cal_normalized = w_cal / sum_w_cal_test
        w_test_curr_normalized = w_test_curr / sum_w_cal_test


        if (np.sum(w_cal_normalized[cal_infeasible_indicators]) + w_test_curr_normalized <= cfg.conformal_policy_control.alpha):
            return n
    return 0


def importance_weighted_monte_carlo_integration(
    LRs_unconstrained_over_safe, ## 1D numpy array
    beta_t, ## float
    proposal: str = "safe" ## "unconstrained"
):

    if proposal == "unconstrained":
        ## If beta_t >= 1: Assume proposal is unconstrained
        return np.mean(np.minimum(beta_t/LRs_unconstrained_over_safe, 1))

    elif proposal == "safe":
        ## Else, beta_t < 1: Assume proposal is safe
        return np.mean(np.minimum(LRs_unconstrained_over_safe, beta_t))
    else:
        raise ValueError(f"Unrecognized proposal name : {proposal}")


def iwmci_intersection_est(
    LRs_unconstrained_over_safe, ## 1D numpy array
    unconstrained_liks,
    safe_liks,
    beta_t, ## float
    psi_t,
    # intersection_target: str = "safe", ## The policy that want to compute intersection (w constrained policy) for
    proposal: str = "safe"
):
    # iwmci_est = importance_weighted_monte_carlo_integration(LRs_unconstrained_over_safe, beta_t, "safe")
    # return np.mean(LRs_unconstrained_over_safe[LRs_unconstrained_over_safe < beta_t]) / iwmci_est
    # if intersection_target not in ["safe", "unconstrained"]:
    #     raise ValueError(f"intersection_target name not recognized : {intersection_target}")
    
    
    if proposal not in ["safe", "unconstrained"]:
        raise ValueError(f"proposal name not recognized : {proposal}")
    

    constrained_density_est = np.minimum(safe_liks * (beta_t / psi_t), unconstrained_liks / psi_t)

    # proposal_liks = unconstrained_liks if proposal == "unconstrained" else safe_liks
    # intersection_target_liks = unconstrained_liks if intersection_target == "unconstrained" else safe_liks
    
    if proposal == "unconstrained":
    #     breakpoint()
        ## If beta_t >= 1: Assume proposal is unconstrained
        # constrained_density_est = np.minimum(safe_liks * (beta_t / psi_t), unconstrained_liks / psi_t)
        return np.mean(np.minimum(constrained_density_est, unconstrained_liks) / unconstrained_liks)


    elif proposal == "safe":
    #     ## Else, beta_t < 1: Assume proposal is safe
    #     # return np.mean(np.minimum(LRs_unconstrained_over_safe, beta_t))
    #     # constrained_density_est = np.minimum(safe_liks * (beta_t / psi_t), unconstrained_liks / psi_t)
        return np.mean(np.minimum(constrained_density_est, safe_liks) / safe_liks)

    else:
        raise ValueError(f"Unrecognized proposal name : {proposal}")



# def iwmci_safe_est(
#     LRs_unconstrained_over_safe, ## 1D numpy array
#     beta_t, ## float
#     proposal: str = "safe"
# ):
#     # iwmci_est = importance_weighted_monte_carlo_integration(LRs_unconstrained_over_safe, beta_t, "safe")
#     # return np.mean(LRs_unconstrained_over_safe[LRs_unconstrained_over_safe < beta_t]) / iwmci_est
#     if proposal == "unconstrained":
#         ## If beta_t >= 1: Assume proposal is unconstrained
#         return np.mean((beta_t/LRs_unconstrained_over_safe)[beta_t/LRs_unconstrained_over_safe < 1])

#     elif proposal == "safe":
#         ## Else, beta_t < 1: Assume proposal is safe
#         return np.mean(LRs_unconstrained_over_safe[LRs_unconstrained_over_safe < beta_t])
#     else:
#         raise ValueError(f"Unrecognized proposal name : {proposal}")





def accept_reject_sample_and_get_likelihoods(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    model_dir_list: List[str],
    seeds_fp_list: List[str],
    output_dir: str,
    betas_list: List[float],
    psis_list: List[float], ## Normalization constants
    n_target: int, 
    ga_data_dir: str,
    temps: List[int]=[1.0],
    depth: int = 0, ## Recursion depth
    higher_score_particle_field: str ="higher_score_particle",
    lower_score_particle_field: str ="lower_score_particle",
    higher_score_field:str ="higher_score",
    lower_score_field: str ="lower_score",
    proposal: str = None, ## Proposal distribution (safe or unconstrained), or None --> means running for filtering
    post_policy_control: bool = False, ## Whether calling post policy control (True <--> generating risk-controlled actions), or pre control (False <--> Generating proposals)
) -> str:

    n_models = len(model_dir_list)

    # accepted = [] ## List containing indicators for whether each considered proposal sample is accepted
    n_accepted = 0

    ## Initialize data frames for storing data for accepted samples
    unconstrained_lik_cols = [f'lik_r{i}' for i in range(n_models)]
    unconstrained_col_names = ['particle', 'score'] + unconstrained_lik_cols
    accepted_unconstrained_df = pd.DataFrame(columns=unconstrained_col_names)

    constrained_lik_cols = [f'con_lik_r{i}' for i in range(n_models)]
    constrained_col_names = ['particle', 'score'] + constrained_lik_cols
    accepted_constrained_df = pd.DataFrame(columns=constrained_col_names)

    call_idx = 0


    # if betas_list[-1] >= 1:
    if proposal == "unconstrained":
        ## If beta_t >= 1, then using unconstrained policy as proposal

        while n_accepted < n_target:

            accepted_curr = []

            temps_curr = temps if len(model_dir_list) > 1 else [cfg.temperature_init]


            ## Sample using unconstrained model as proposal
            _, iter_gen_outputs_list, hd = run_iterative_generation(
                cfg,
                fs,
                seeds_fp_list[-1], #combined_sft_dataset_fp,
                ga_data_dir,
                model_dir_list[-1], #sft_dir,
                output_dir,
                higher_score_particle_field=higher_score_particle_field,
                lower_score_particle_field=lower_score_particle_field,
                higher_score_field=higher_score_field,
                lower_score_field=lower_score_field,
                temps=temps_curr,
                model_idx = len(model_dir_list) - 1, ## Index for model being called for generation
                call_idx=call_idx, ## Index for times this generation has been called, including current, for same model directory
                proportion_of_target_n_accepted = n_accepted / n_target
            )
            call_idx += 1



            

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



            gen_liks_df = pd.read_json(gen_liks_fp, orient="records", lines=True)[unconstrained_col_names] #[unconstrained_col_names]
            gen_liks_mat = gen_liks_df[unconstrained_lik_cols].to_numpy() ## Shape (n_prop, n_models)
            # gen_liks_mat = gen_liks_df.to_numpy() ## Shape (n_prop, n_models)

            if cfg.conformal_policy_control.constrain_against == 'init':
                idx_safe_model = 0

                constrained_liks_mat = np.zeros(np.shape(gen_liks_mat))
                constrained_liks_mat[:,0] = gen_liks_mat[:,0]

                for c in range(1, len(unconstrained_lik_cols)):
                    prop_data_t0_safe_and_t_unconstrained_liks = gen_liks_mat[:, [0, c]]

                    constrained_liks_mat[:,c] = constrain_likelihoods(cfg, gen_liks_mat[:, [0, c]], [betas_list[0], betas_list[c]], [psis_list[0], psis_list[c]])[:,-1] ## Shape (n_prop, n_models)
            
                lik_ratios_unconstrained_over_safe = gen_liks_mat[:, -1] / constrained_liks_mat[:, idx_safe_model]

            else:
                idx_safe_model = -2

                constrained_liks_mat = constrain_likelihoods(cfg, gen_liks_mat, betas_list, psis_list) ## Shape (n_prop, n_models)

                if constrained_liks_mat.shape[1] > 1:
                    ## If is not original safe model, \pi_{\theta_0}
                    lik_ratios_unconstrained_over_safe = gen_liks_mat[:, -1] / constrained_liks_mat[:, -2]
                else:
                    ## Else is original safe model, \pi_{\theta_0}, so unconstrained and constrained likelihoods are the same
                    ## (Lik ratios should be == 1, and bound == inf, so should accept everything)
                    lik_ratios_unconstrained_over_safe = gen_liks_mat[:, -1] / constrained_liks_mat[:, -1]
            
            constrained_liks_df_ = pd.DataFrame(constrained_liks_mat, columns=constrained_lik_cols)
            constrained_liks_df = pd.concat([gen_liks_df[['particle', 'score']], constrained_liks_df_], axis=1)


            n_prop = len(gen_liks_df)


            
            # lik_ratios_unconstrained_over_safe = gen_liks_mat[:, -1] / constrained_liks_mat[:, idx_safe_model]


            ## Accept or reject each proposal
            # U = np.random.uniform(size=n_prop)
            for i in range(n_prop):
                u = np.random.uniform()
                
                if u < betas_list[-1]/lik_ratios_unconstrained_over_safe[i]:
                    accepted_curr.append(True)
                    n_accepted += 1

                    if n_accepted >= n_target:
                        break

                else:
                    accepted_curr.append(False)

            accepted_unconstrained_df = pd.concat([accepted_unconstrained_df, gen_liks_df.iloc[:len(accepted_curr)][accepted_curr]], ignore_index=True)
            accepted_constrained_df = pd.concat([accepted_constrained_df, constrained_liks_df[:len(accepted_curr)][accepted_curr]], ignore_index=True)

            ## Save accepted with unconstrained likelihoods
            # accepted_unconstrained_df = gen_liks_df[accepted]
            # accepted_unconstrained_gen_liks_fp = f"accepted_{gen_liks_fp}"
            # accepted_unconstrained_df.to_json(accepted_unconstrained_gen_liks_fp, orient="records", lines=True)

            ## Save accepted with constrained likelihoods
            # accepted_constrained_liks_df = constrained_liks_df[accepted]
            # accepted_constrained_gen_liks_fp = f"accepted_constrained_{gen_liks_fp}"
            # accepted_constrained_liks_df.to_json(accepted_constrained_gen_liks_fp, orient="records", lines=True)

    elif proposal == "safe":
        ## Else, beta_t < 1, then using safe policy as proposal

        while n_accepted < n_target:


            accepted_curr = []

            ## Choose num proposals to try to avoid multiple batch generations
            # n_target_safe = 1.1 * max(betas_list[-2], 1/betas_list[-2]) * n_target




                        # accepted_curr = []


            ## Sample using unconstrained model as proposal

            if cfg.conformal_policy_control.constrain_against == 'init':


                _, iter_gen_outputs_list, hd = run_iterative_generation(
                    cfg,
                    fs,
                    seeds_fp_list[0], #combined_sft_dataset_fp,
                    ga_data_dir,
                    model_dir_list[0], #sft_dir,
                    output_dir,
                    higher_score_particle_field=higher_score_particle_field,
                    lower_score_particle_field=lower_score_particle_field,
                    higher_score_field=higher_score_field,
                    lower_score_field=lower_score_field,
                    temps=[cfg.temperature_init],
                    model_idx = 0, #len(model_dir_list) - 1, ## Index for model being called for generation
                    call_idx=call_idx, ## Index for times this generation has been called, including current, for same model directory
                    proportion_of_target_n_accepted = n_accepted / n_target
                )


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


                gen_liks_df = pd.read_json(gen_liks_fp, orient="records", lines=True)[unconstrained_col_names] #[unconstrained_col_names]
                gen_liks_mat = gen_liks_df[unconstrained_lik_cols].to_numpy() 


                constrained_liks_mat = np.zeros(np.shape(gen_liks_mat))
                constrained_liks_mat[:,0] = gen_liks_mat[:,0]
                for c in range(1, len(unconstrained_lik_cols)):
                    prop_data_t0_safe_and_t_unconstrained_liks = gen_liks_mat[:, [0, c]]

                    constrained_liks_mat[:,c] = constrain_likelihoods(cfg, gen_liks_mat[:, [0, c]], [betas_list[0], betas_list[c]], [psis_list[0], psis_list[c]])[:,-1] ## Shape (n_prop, n_models)
                
                
                
                lik_ratios_unconstrained_over_safe = gen_liks_mat[:, -1] / constrained_liks_mat[:, 0]

                
                constrained_liks_df_ = pd.DataFrame(constrained_liks_mat, columns=constrained_lik_cols)
                constrained_liks_df = pd.concat([gen_liks_df[['particle', 'score']], constrained_liks_df_], axis=1)




            else:
                ## Shouldn't need this, handled in base case of recursion
                # temps_curr = temps if len(model_dir_list) - 1 > 1 else [cfg.temperature_init] ## If model list after removing one model is not just initial, use temps
                
                ## Sample using unconstrained model as proposal
                gen_liks_tmin1_df, gen_liks_tmin1_fp, constrained_gen_liks_tmin1_df, constrained_gen_liks_tmin1_fp = \
                                            accept_reject_sample_and_get_likelihoods(
                                                cfg,
                                                fs,
                                                model_dir_list[:-1],
                                                seeds_fp_list[:-1],
                                                output_dir,
                                                betas_list[:-1],
                                                psis_list[:-1], ## Normalization constants
                                                n_target, 
                                                ga_data_dir,
                                                temps,
                                                depth + 1,
                                                higher_score_particle_field=higher_score_particle_field,
                                                lower_score_particle_field=lower_score_particle_field,
                                                higher_score_field=higher_score_field,
                                                lower_score_field=lower_score_field,
                                                proposal='safe'
                                            )

                ## Compute unconstrained likelihoods for most recent model (not passed to recursion) on output proposal samples
                gen_liks_fp_list, hd = run_compute_liks_all_models_and_cal_data(
                    cfg, fs, 
                    seeds_fp_list=[seeds_fp_list[-1]],
                    prev_cal_data_fp_list=[],
                    model_dir_list=[model_dir_list[-1]],
                    target_fp=gen_liks_tmin1_fp, ## Should add a column for time t to gen_liks_tmin1_fp
                    model_indices = [len(model_dir_list)-1], ## Index for most recent model
                    temps=temps,
                )
                gen_liks_fp = gen_liks_fp_list[-1]



                gen_liks_df = pd.read_json(gen_liks_fp, orient="records", lines=True)[unconstrained_col_names] #[unconstrained_col_names]
                gen_liks_mat = gen_liks_df[unconstrained_lik_cols].to_numpy() ## Shape (n_prop, n_models)
                # gen_liks_mat = gen_liks_df.to_numpy() ## Shape (n_prop, n_models)

                gen_liks_df_t0_safe_and_t_unconstrained_mat = pd.concat([constrained_gen_liks_tmin1_df.iloc[:, -1], gen_liks_df.iloc[:,-1]], axis=1).to_numpy() ## Double check this

                constrained_liks_mat = constrain_likelihoods(cfg, gen_liks_df_t0_safe_and_t_unconstrained_mat, betas_list[-2:], psis_list[-2:])

                # constrained_liks_mat = constrain_likelihoods(cfg, gen_liks_df_t0_safe_and_t_unconstrained_mat, betas_list[-2:], psis_list[-2:])

                if constrained_liks_mat.shape[1] > 1:
                    ## If is not original safe model, \pi_{\theta_0}
                    lik_ratios_unconstrained_over_safe = gen_liks_mat[:, -1] / constrained_liks_mat[:, -2]
                else:
                    ## Else is original safe model, \pi_{\theta_0}, so unconstrained and constrained likelihoods are the same
                    ## (Lik ratios should be == 1, and bound == inf, so should accept everything)
                    lik_ratios_unconstrained_over_safe = gen_liks_mat[:, -1] / constrained_liks_mat[:, -1]

            
                constrained_liks_df_ = pd.DataFrame(constrained_liks_mat, columns=constrained_lik_cols[-2:])
                constrained_liks_df = pd.concat([constrained_gen_liks_tmin1_df, constrained_liks_df_.iloc[:,-1]], axis = 1)
                
                ### NOTE: This used to be
                ## lik_ratios_unconstrained_over_safe = gen_liks_mat[:, -1] / constrained_liks_mat[:, -2] 
                # lik_ratios_unconstrained_over_safe = gen_liks_mat[:, -1] / constrained_liks_mat[:, -1] 

        
            call_idx += 1

            n_prop = len(gen_liks_df)


            ## Accept or reject each proposal
            for i in range(n_prop):
                u = np.random.uniform()
                
                ## NOTE: Different acceptance criteria for different proposal
                if u < lik_ratios_unconstrained_over_safe[i]/betas_list[-1]:
                    accepted_curr.append(True)
                    n_accepted += 1

                    if n_accepted >= n_target:
                        break

                else:
                    accepted_curr.append(False)

            accepted_unconstrained_df = pd.concat([accepted_unconstrained_df, gen_liks_df[:len(accepted_curr)][accepted_curr]], ignore_index=True)
            accepted_constrained_df = pd.concat([accepted_constrained_df, constrained_liks_df[:len(accepted_curr)][accepted_curr]], ignore_index=True)
            

            # gen_liks_df = pd.read_json(gen_liks_fp, orient="records", lines=True) #[unconstrained_col_names]
            # # gen_liks_df = gen_liks_tmin1_df
            # gen_liks_mat = gen_liks_df[unconstrained_lik_cols].to_numpy() ## Shape (n_prop, n_models)
            
            # # constrained_liks_df = pd.read_json(constrained_gen_liks_fp, orient="records", lines=True) #[constrained_col_names]

            # gen_liks_df_tmin1_safe_and_t_unconstrained_mat = pd.concat([constrained_gen_liks_tmin1_df.iloc[:, -1], gen_liks_df.iloc[:,-1]], axis=1).to_numpy() ## Double check this

            # ## Compute constrained likelihoods, only starting from most recent safe likelihoods
            # constrained_liks_mat = constrain_likelihoods(gen_liks_df_tmin1_safe_and_t_unconstrained_mat, betas_list[-2:], psis_list[-2:])
            # constrained_liks_df_ = pd.DataFrame(constrained_liks_mat, columns=constrained_lik_cols[-2:])

            # if constrained_gen_liks_tmin1_df.iloc[0,-1] != constrained_liks_df_.iloc[0, 0]:
            #     raise ValueError(f"constrained_gen_liks_tmin1_df[0,-1] ({constrained_gen_liks_tmin1_df.iloc[0,-1]}) != ({constrained_liks_df_.iloc[0, 0]}) constrained_liks_df_tmp[0, 0].")


            # constrained_liks_df = pd.concat([constrained_gen_liks_tmin1_df, constrained_liks_df_.iloc[:,-1]], axis = 1)
            # # constrained_liks_df = pd.concat([gen_liks_df[['particle', 'score']], constrained_liks_df_], axis=1)
            # # constrained_liks_df = constrained_gen_liks_tmin1_df #[constrained_col_names]
            # # constrained_liks_mat = constrained_liks_df[constrained_lik_cols].to_numpy()


            # n_prop = len(gen_liks_df)
            # lik_ratios_unconstrained_over_safe = gen_liks_mat[:, -1] / constrained_liks_mat[:, -2]

            # ## Accept or reject each proposal
            # for i in range(n_prop):
            #     u = np.random.uniform()
                
            #     ## NOTE: Different acceptance criteria for different proposal
            #     if u < lik_ratios_unconstrained_over_safe[i]/betas_list[-1]:
            #         accepted_curr.append(True)
            #         n_accepted += 1

            #         if n_accepted >= n_target:
            #             break

            #     else:
            #         accepted_curr.append(False)

            # accepted_unconstrained_df = pd.concat([accepted_unconstrained_df, gen_liks_df[:len(accepted_curr)][accepted_curr]], ignore_index=True)
            # accepted_constrained_df = pd.concat([accepted_constrained_df, constrained_liks_df[:len(accepted_curr)][accepted_curr]], ignore_index=True)
    else:
        raise ValueError(f"Unknown proposal name : {proposal}")

    # ## Save accepted with unconstrained likelihoods
    # t = len(model_dir_list)-1
    # accepted_unconstrained_df = gen_liks_df[accepted]

    if depth == 0:

        if post_policy_control:
            ## Output filename for unconstrained likelihoods
            u_output_filename_prefix = f"accepted_uLiks_alpha{cfg.conformal_policy_control.alpha}_gens_likelihood_{cfg.iterative_generation.args.sample_size}sample_{cfg.iterative_generation.args.max_iterations}iter"
            c_output_filename_prefix = f"accepted_cLiks_alpha{cfg.conformal_policy_control.alpha}_gens_likelihood_{cfg.iterative_generation.args.sample_size}sample_{cfg.iterative_generation.args.max_iterations}iter"

            u_output_filename = f"{u_output_filename_prefix}_temp{temps[-1]}_{cfg.generation_sampling_num_return_sequences}seqs.jsonl"
            c_output_filename = f"{c_output_filename_prefix}_temp{temps[-1]}_{cfg.generation_sampling_num_return_sequences}seqs.jsonl"

            accepted_unconstrained_gen_liks_fp = os.path.join(os.path.dirname(gen_liks_fp), u_output_filename)
            accepted_constrained_gen_liks_fp = os.path.join(os.path.dirname(gen_liks_fp), c_output_filename)

        else:
            accepted_unconstrained_gen_liks_fp = os.path.join(os.path.dirname(gen_liks_fp), f"prop_{proposal}_beta{betas_list[-1]}_{depth}u_{os.path.basename(gen_liks_fp)}")
            accepted_constrained_gen_liks_fp = os.path.join(os.path.dirname(gen_liks_fp), f"prop_{proposal}_beta{betas_list[-1]}_{depth}c_{os.path.basename(gen_liks_fp)}")

    else:
        accepted_unconstrained_gen_liks_fp = os.path.join(os.path.dirname(gen_liks_fp), f"{depth}u_{os.path.basename(gen_liks_fp)}")
        accepted_constrained_gen_liks_fp = os.path.join(os.path.dirname(gen_liks_fp), f"{depth}c_{os.path.basename(gen_liks_fp)}")


    accepted_unconstrained_df.to_json(accepted_unconstrained_gen_liks_fp, orient="records", lines=True)
    accepted_constrained_df.to_json(accepted_constrained_gen_liks_fp, orient="records", lines=True)


    return accepted_unconstrained_df, accepted_unconstrained_gen_liks_fp, accepted_constrained_df, accepted_constrained_gen_liks_fp


def empirical_cdf(
    values, ## numpy array with values
    x ## value to evaluate empirical CDF at
) -> float:
    return len(values[values <= x])/len(values)


def run_conformal_policy_control(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    model_dir_list: List[str],
    seeds_fp_list: List[str],
    prev_cal_data_unconstrained_liks_fp_list: List[str], ## Should contain both cal data and *constrained* likelihoods, up to r{t}
    prev_cal_data_constrained_liks_fp_list: List[str], ## Should contain both cal data and *constrained* likelihoods, up to r{t-1}
    betas_list: List[float],
    psis_list: List[float], ## Normalization constants
    ga_data_dir: str,
    higher_score_particle_field: str ="higher_score_particle",
    lower_score_particle_field: str ="lower_score_particle",
    higher_score_field:str ="higher_score",
    lower_score_field: str ="lower_score",
    # target_fp: str,
    # particle_field: str = "higher_score_particle",
    # score_field: str = "score",
    # temps: List[float] = [1.0],
) -> str:
    """
    Runs conformal policy control.
    """
    opt_str = " ".join(
        get_all_strs_from_nested_dict(cfg["conformal_policy_control"]["args"])
    )
    ## Load calibration data into one dataframe
    n_cal_sets = len(prev_cal_data_constrained_liks_fp_list)

    if n_cal_sets != len(prev_cal_data_unconstrained_liks_fp_list):
        raise ValueError("Number of unconstrained and constrained cal sets must be the same")

    cal_data_constrained_all = pd.read_json(prev_cal_data_constrained_liks_fp_list[0], orient="records", lines=True)
    cal_data_unconstrained_all = pd.read_json(prev_cal_data_unconstrained_liks_fp_list[0], orient="records", lines=True)



    n_cal_per_model = [len(cal_data_constrained_all)]
    unconstrained_lik_cols = [f'lik_r{i}' for i in range(n_cal_sets)]
    constrained_lik_cols = [f'con_lik_r{i}' for i in range(n_cal_sets)]

    #pd.read_json(prev_cal_data_unconstrained_liks_fp_list[2], orient="records", lines=True)

    for i in range(1, n_cal_sets):
        cal_data_constrained_curr = pd.read_json(prev_cal_data_constrained_liks_fp_list[i], orient="records", lines=True)
        cal_data_unconstrained_curr = pd.read_json(prev_cal_data_unconstrained_liks_fp_list[i], orient="records", lines=True)

        if (len(cal_data_constrained_curr) != len(cal_data_unconstrained_curr)):
            raise ValueError("Num samples in constrained and constrained cal sets should be same (ie, same particles)!")

        n_cal_per_model.append(len(cal_data_constrained_curr))

        ## Check that columns are the same
        if cal_data_constrained_all.columns.equals(cal_data_constrained_curr.columns) and cal_data_unconstrained_all.columns.equals(cal_data_unconstrained_curr.columns):
            cal_data_constrained_all = pd.concat([cal_data_constrained_all, cal_data_constrained_curr], ignore_index=True)
            cal_data_unconstrained_all = pd.concat([cal_data_unconstrained_all, cal_data_unconstrained_curr], ignore_index=True)

        else:
            breakpoint()
            logger.info(f"cal_data_constrained_all.columns : {cal_data_constrained_all.columns}")
            logger.info(f"cal_data_constrained_curr.columns : {cal_data_constrained_curr.columns}")
            logger.info(f"cal_data_unconstrained_all.columns : {cal_data_unconstrained_all.columns}")
            logger.info(f"cal_data_unconstrained_curr.columns : {cal_data_unconstrained_curr.columns}")
            raise ValueError(f"Error: cal_data_constrained_all.columns. equals(cal_data_constrained_curr.columns) : {cal_data_constrained_all.columns.equals(cal_data_constrained_curr.columns)}")

    check_col_names(cal_data_constrained_all)
    check_col_names(cal_data_unconstrained_all)

    ## Prep cal data safe & unconstrained liks: Need most recent safe likelihoods (safe at t-1) and current unconstrained likelihoods (unconstrained at t) in loop
    # cal_data_tmin1_safe_and_t_unconstrained_liks = pd.concat([cal_data_constrained_all[constrained_lik_cols[-2]], cal_data_unconstrained_all[unconstrained_lik_cols[-1]]], axis=1).to_numpy()
    # cal_data_tmin1_safe_and_t_unconstrained_liks = pd.concat([cal_data_constrained_all.iloc[:,-2], cal_data_unconstrained_all.iloc[:,-1]], axis=1).to_numpy()
    cal_data_t0_safe_and_t_unconstrained_liks = pd.concat([cal_data_constrained_all[constrained_lik_cols[0]], cal_data_unconstrained_all.iloc[:,-1]], axis=1).to_numpy()  ## Double check this

    ## For cal data: Use constrained likelihoods to compute mixture distribution
    cal_data_constrained_all_liks = cal_data_constrained_all[constrained_lik_cols].to_numpy()
    mixture_weights = np.array(n_cal_per_model)
    cal_mixture_constrained_density = mixture_pdf_from_densities_mat(cal_data_constrained_all_liks, mixture_weights)



    if cfg.conformal_policy_control.alpha >= 1.0:
        policy_names = ['unconstrained']
        adjusted_alpha = 1.0
    elif cfg.conformal_policy_control.num_starts_beta_search == 2:
        policy_names = ['unconstrained', 'safe']
        adjusted_alpha = cfg.conformal_policy_control.alpha / 2 ## Multistart correction
    else:
        policy_names = ['safe', 'unconstrained']
        adjusted_alpha = cfg.conformal_policy_control.alpha ## Fixed sequence testing, no multistart correction


    lik_ratios_unconstrained_over_safe_cal_and_prop_dict = {}
    lik_ratios_unconstrained_over_safe_dict = {}

    unconstrained_liks_dict, safe_liks_dict = {}, {}

    unconstrained_df_dict, unconstrained_gen_liks_fp_dict, constrained_liks_df_dict, constrained_gen_liks_fp_dict = {}, {}, {}, {}
    prop_data_t0_safe_and_t_unconstrained_liks_dict = {}
    betas_list_tmp_dict, psis_list_tmp_dict = {}, {}

    for i, proposal in enumerate(policy_names):
        

        
        if proposal == 'safe':
            ## Use constrained/safe policy as the proposal
            betas_list_tmp = betas_list + [sys.float_info.min]
            psis_list_tmp = psis_list + [sys.float_info.min]
            # model_dir_list_tmp = model_dir_list[:-1]
            # seeds_fp_list_tmp = seeds_fp_list[:-1]
            # model_fp_tmp = model_dir_list[-2]

        else:
            ## Else, use unconstrained policy as the proposal
            betas_list_tmp = betas_list + [np.inf]
            psis_list_tmp = psis_list + [1.0]
            # model_dir_list_tmp = model_dir_list
            # seeds_fp_list_tmp = seeds_fp_list
            # model_fp_tmp = model_dir_list[-1]
        betas_list_tmp_dict[proposal] = betas_list_tmp
        psis_list_tmp_dict[proposal] = psis_list_tmp

        ## Get proposal samples, unconstrained likelihoods, and constrained likelihoods
        unconstrained_df, unconstrained_gen_liks_fp, constrained_liks_df, constrained_gen_liks_fp \
            = accept_reject_sample_and_get_likelihoods(cfg, fs, model_dir_list, seeds_fp_list, model_dir_list[-1],\
                                                       betas_list_tmp, psis_list_tmp, \
                                                       cfg.conformal_policy_control.accept_reject.n_target,\
                                                       ga_data_dir, higher_score_particle_field=higher_score_particle_field,\
                                                       lower_score_particle_field=lower_score_particle_field,
                                                       higher_score_field=higher_score_field,
                                                       lower_score_field=lower_score_field, proposal=proposal)

        unconstrained_df_dict[proposal] = unconstrained_df
        unconstrained_gen_liks_fp_dict[proposal] = unconstrained_gen_liks_fp
        constrained_liks_df_dict[proposal] = constrained_liks_df
        constrained_gen_liks_fp_dict[proposal] = constrained_gen_liks_fp


        ## NOTE/Warning: for proposal == 'unconstrained', should have unconstrained_df == constrained_liks_df (identical); else, should have constrained_liks_df.iloc[:,-1] == constrained_liks_df.iloc[:,-2] (fully constrained)
        check_col_names(unconstrained_df)
        check_col_names(constrained_liks_df)


        if cfg.conformal_policy_control.constrain_against == 'init':
            unconstrained_liks = unconstrained_df.iloc[:, -1]
            safe_liks = constrained_liks_df['con_lik_r0']
            lik_ratios_unconstrained_over_safe = unconstrained_df.iloc[:, -1] / constrained_liks_df['con_lik_r0']
            prop_data_t0_safe_and_t_unconstrained_liks = pd.concat([constrained_liks_df['con_lik_r0'], unconstrained_df.iloc[:, -1]], axis=1).to_numpy()
            lik_ratios_unconstrained_over_safe_cal_and_prop = np.concatenate((np.array(lik_ratios_unconstrained_over_safe), np.array(cal_data_unconstrained_all.iloc[:, -1] / cal_data_constrained_all['con_lik_r0'])))

        else:
            unconstrained_liks = unconstrained_df.iloc[:, -1]
            safe_liks = constrained_liks_df.iloc[:, -2]
            lik_ratios_unconstrained_over_safe = unconstrained_df.iloc[:, -1] / constrained_liks_df.iloc[:, -2]
            prop_data_t0_safe_and_t_unconstrained_liks = pd.concat([constrained_liks_df.iloc[:, -2], unconstrained_df.iloc[:, -1]], axis=1).to_numpy()
            lik_ratios_unconstrained_over_safe_cal_and_prop = np.concatenate((np.array(lik_ratios_unconstrained_over_safe), np.array(cal_data_unconstrained_all.iloc[:, -1] / cal_data_unconstrained_all.iloc[:, -2])))

        prop_data_t0_safe_and_t_unconstrained_liks_dict[proposal] = prop_data_t0_safe_and_t_unconstrained_liks
        lik_ratios_unconstrained_over_safe_cal_and_prop_dict[proposal] = lik_ratios_unconstrained_over_safe_cal_and_prop
        lik_ratios_unconstrained_over_safe_dict[proposal] = lik_ratios_unconstrained_over_safe

        unconstrained_liks_dict[proposal] = unconstrained_liks
        safe_liks_dict[proposal] = safe_liks



    lik_ratios_unconstrained_over_safe_cal_and_prop_arr = np.array(lik_ratios_unconstrained_over_safe_cal_and_prop_dict[policy_names[0]])
    lik_ratios_unconstrained_over_safe_arr = np.array(lik_ratios_unconstrained_over_safe_dict[policy_names[0]])

    if len(policy_names) > 1:
        lik_ratios_unconstrained_over_safe_cal_and_prop_arr = np.concatenate((lik_ratios_unconstrained_over_safe_cal_and_prop_arr, lik_ratios_unconstrained_over_safe_cal_and_prop_dict[policy_names[1]]))
        lik_ratios_unconstrained_over_safe_arr = np.concatenate((lik_ratios_unconstrained_over_safe_arr, lik_ratios_unconstrained_over_safe_dict[policy_names[1]]))

        # breakpoint()
        
        G = prepare_grid(cfg, lik_ratios_unconstrained_over_safe_arr, #lik_ratios_unconstrained_over_safe, 
                            n_grid = cfg.conformal_policy_control.args.n_grid,
                            proposal = "mixed")

    else:
        if cfg.conformal_policy_control.alpha >= 1.0:
            G = [np.inf]
        else:
            if proposal == 'unconstrained':
                G = prepare_grid(cfg, lik_ratios_unconstrained_over_safe_cal_and_prop, #lik_ratios_unconstrained_over_safe, 
                            n_grid = cfg.conformal_policy_control.args.n_grid,
                            proposal = proposal
                            )

            elif proposal == 'safe':
                G = prepare_grid(cfg, lik_ratios_unconstrained_over_safe_cal_and_prop, #lik_ratios_unconstrained_over_safe, 
                            n_grid = cfg.conformal_policy_control.args.n_grid,
                            proposal = proposal
                            )
            else:
                raise ValueError(f"Unrecognized proposal name : {proposal}")


    ## Search over grid for largest bound that satisfies conformal constraint
    beta_hat_t_curr = sys.float_info.min if adjusted_alpha < 1.0 else np.inf ## Currently selected beta_t is initially smallest float value


    for i, proposal in enumerate(policy_names):
        
        unconstrained_df = unconstrained_df_dict[proposal]
        unconstrained_gen_liks_fp = unconstrained_gen_liks_fp_dict[proposal]
        constrained_liks_df = constrained_liks_df_dict[proposal]
        constrained_gen_liks_fp = constrained_gen_liks_fp_dict[proposal]
        betas_list_tmp = betas_list_tmp_dict[proposal]
        psis_list_tmp = psis_list_tmp_dict[proposal]

        prop_data_t0_safe_and_t_unconstrained_liks = prop_data_t0_safe_and_t_unconstrained_liks_dict[proposal]
        lik_ratios_unconstrained_over_safe_cal_and_prop = lik_ratios_unconstrained_over_safe_cal_and_prop_dict[proposal]
        lik_ratios_unconstrained_over_safe = lik_ratios_unconstrained_over_safe_dict[proposal]
        unconstrained_liks = unconstrained_liks_dict[proposal]
        safe_liks = safe_liks_dict[proposal]

    
        ## For proposal data: Use constrained likelihoods to compute mixture distribution
        prop_data_constrained_all = constrained_liks_df
        prop_data_constrained_prev_liks = prop_data_constrained_all[constrained_lik_cols].to_numpy()
        mixture_weights = np.array(n_cal_per_model)

        prop_mixture_constrained_density = mixture_pdf_from_densities_mat(prop_data_constrained_prev_liks, mixture_weights)



        ## Get infeasibility indicators for calibration data
        cal_scores = cal_data_constrained_all['score'].to_numpy()
        cal_infeasible_indicators = np.isnan(cal_scores) | np.isinf(cal_scores)

        
        # n_safe_actions = get_num_safe_actions(cfg, cal_infeasible_indicators, cal_data_constrained_all.iloc[:,-1].to_numpy(), cal_mixture_constrained_density, constrained_liks_df.iloc[:, -2].to_numpy(), prop_mixture_constrained_density, cfg.conformal_policy_control.accept_reject.n_target)
        # n_safe_actions_uncon = get_num_safe_actions(cfg, cal_infeasible_indicators, cal_data_unconstrained_all.iloc[:,-1].to_numpy(), cal_mixture_constrained_density, unconstrained_df.iloc[:, -2].to_numpy(), prop_mixture_constrained_density, cfg.conformal_policy_control.accept_reject.n_target)
        # # for steps_back_to_safe in range(1, len(cal_data_constrained_all.columns)+1):
        # #     n_safe_actions = get_num_safe_actions(cfg, cal_infeasible_indicators, cal_data_constrained_all.iloc[:,-steps_back_to_safe].to_numpy(), cal_mixture_constrained_density, constrained_liks_df.iloc[:, -(steps_back_to_safe+1)].to_numpy(), prop_mixture_constrained_density, cfg.conformal_policy_control.accept_reject.n_target)
        # #     # n_safe_actions_uncon = get_num_safe_actions(cfg, cal_infeasible_indicators, cal_data_unconstrained_all.iloc[:,-steps_back_to_safe].to_numpy(), cal_mixture_constrained_density, unconstrained_df.iloc[:, -(steps_back_to_safe+1)].to_numpy(), prop_mixture_constrained_density, cfg.conformal_policy_control.accept_reject.n_target)
        

        # # n_safe_actions_curr = n_safe_actions

        # n_safe_actions_curr = (n_safe_actions + n_safe_actions_uncon) / 2 if n_safe_actions_uncon < n_safe_actions else n_safe_actions

        # if n_safe_actions == 0:
        #     ## If cannot take any actions under the safe policy, then that's the best can do and return with safe policy
        #     beta_t = sys.float_info.min
        #     psi_hat_t = importance_weighted_monte_carlo_integration(lik_ratios_unconstrained_over_safe, beta_t)
        #                 ## Save proposals with cpc-constrained likelihoods
            
        #     constrained_liks_df_beta_hat = pd.concat([constrained_liks_df.iloc[:, :-1], pd.DataFrame({f'con_lik_r{n_cal_sets}' : prop_constrained_liks_curr[:,-1]})], axis=1)
        #     constrained_liks_df_beta_hat_fp = os.path.join(os.path.dirname(constrained_gen_liks_fp), f"cpc_prop_alpha{cfg.conformal_policy_control.alpha}_beta{betas_list[-1]:.3g}_{constrained_gen_liks_fp}")
        #     constrained_liks_df_beta_hat.to_json(constrained_liks_df_beta_hat_fp, orient="records", lines=True)

        #     ## Also save proposals with unconstrained likelihoods
        #     unconstrained_liks_df_beta_hat_fp = os.path.join(os.path.dirname(unconstrained_gen_liks_fp), f"cpc_prop_alpha{cfg.conformal_policy_control.alpha}_beta{betas_list[-1]:.3g}_{unconstrained_gen_liks_fp}")
        #     unconstrained_df.to_json(unconstrained_liks_df_beta_hat_fp, orient="records", lines=True)

        #     check_col_names(constrained_liks_df_beta_hat)
        #     check_col_names(unconstrained_df)
            
        #     return beta_t, psi_hat_t, n_safe_actions, constrained_liks_df_beta_hat, constrained_liks_df_beta_hat_fp, unconstrained_df, unconstrained_liks_df_beta_hat_fp



        # n_safe_actions_curr = n_safe_actions
        # G = np.concatenate((G, [sys.float_info.min]))

        # k = max(int(n_safe_actions/ cfg.conformal_policy_control.args.n_grid_safe_actions), 1)

        # G_n_safe_actions = list(range(n_safe_actions))[::-k]


        # for n_, n_safe_actions_curr in enumerate(G_n_safe_actions):

        #     if n_ == len(G_n_safe_actions) - 1:

        #         G = np.concatenate((G, [sys.float_info.min]))



        for b, beta_t in enumerate(G):

            ## Estimate normalization constant via IWMCI
            psi_hat_t = importance_weighted_monte_carlo_integration(lik_ratios_unconstrained_over_safe, beta_t, proposal)



            
            psi_hat_t_unconstrained = importance_weighted_monte_carlo_integration(lik_ratios_unconstrained_over_safe_dict["unconstrained"], beta_t, "unconstrained")

            psi_hat_intersection_unconstrained = iwmci_intersection_est(
                LRs_unconstrained_over_safe=lik_ratios_unconstrained_over_safe_dict["unconstrained"], ## 1D numpy array
                unconstrained_liks=unconstrained_liks_dict["unconstrained"],
                safe_liks=safe_liks_dict["unconstrained"],
                beta_t=beta_t, ## float
                psi_t=psi_hat_t_unconstrained,
                proposal="unconstrained"
            )

            if len(policy_names) > 1:
                psi_hat_t_safe = importance_weighted_monte_carlo_integration(lik_ratios_unconstrained_over_safe_dict["safe"], beta_t, "safe")

                ## Estimated density under both the minimum of the proposal and constrained policies
                psi_hat_intersection_safe = iwmci_intersection_est(
                    LRs_unconstrained_over_safe=lik_ratios_unconstrained_over_safe_dict["safe"], ## 1D numpy array
                    unconstrained_liks=unconstrained_liks_dict["safe"],
                    safe_liks=safe_liks_dict["safe"],
                    beta_t=beta_t, ## float
                    psi_t=psi_hat_t_safe,
                    proposal="safe"
                )
            else:
                psi_hat_intersection_safe = None
                psi_hat_t_safe = sys.float_info.min

            # iwmci_intersection_est(LRs_unconstrained_over_safe=lik_ratios_unconstrained_over_safe, unconstrained_liks=unconstrained_liks_dict["safe"], safe_liks=safe_liks_dict["safe"], beta_t=np.quantile(G, 0.25), psi_t=importance_weighted_monte_carlo_integration(lik_ratios_unconstrained_over_safe_dict["safe"], np.quantile(G, 0.25), "safe"), proposal="safe")
            # iwmci_intersection_est(LRs_unconstrained_over_safe=lik_ratios_unconstrained_over_safe, unconstrained_liks=unconstrained_liks_dict["unconstrained"], safe_liks=safe_liks_dict["unconstrained"], beta_t=np.quantile(G, 0.25), psi_t=importance_weighted_monte_carlo_integration(lik_ratios_unconstrained_over_safe_dict["unconstrained"], np.quantile(G, 0.25), "unconstrained"), proposal="unconstrained")


            # psi_hat_intersection_safe = empirical_cdf(lik_ratios_unconstrained_over_safe_dict["safe"], beta_t)
            # psi_hat_intersection_unconstrained = empirical_cdf(lik_ratios_unconstrained_over_safe_dict["unconstrained"], beta_t)


            psi_hat_intersection = psi_hat_intersection_safe if proposal == "safe" else psi_hat_intersection_unconstrained
            psi_hat_intersection_non_prop = psi_hat_intersection_safe if proposal != "safe" else psi_hat_intersection_unconstrained



            if i == 0 and len(policy_names) > 1 and 2 * psi_hat_intersection < psi_hat_intersection_non_prop:
                G = G[b:]
                break
                    

            ## Compute constrained likelihoods for cal data on current candidate bound, beta_t
            if cfg.conformal_policy_control.constrain_against == 'init':
                test_pt_factor = 1
                cal_constrained_liks_curr = constrain_likelihoods(cfg, cal_data_t0_safe_and_t_unconstrained_liks, [betas_list[0], beta_t], [psis_list[0], psi_hat_t])
                prop_constrained_liks_curr = constrain_likelihoods(cfg, prop_data_t0_safe_and_t_unconstrained_liks, [betas_list[0], beta_t], [psis_list[0], psi_hat_t])
            else:
                test_pt_factor = 2
                cal_constrained_liks_curr = constrain_likelihoods(cfg, cal_data_t0_safe_and_t_unconstrained_liks, [betas_list[-1]] + [beta_t], [psis_list[-1]] + [psi_hat_t])
                prop_constrained_liks_curr = constrain_likelihoods(cfg, prop_data_t0_safe_and_t_unconstrained_liks, [betas_list[-1]] + [beta_t], [psis_list[-1]] + [psi_hat_t])

            ## Compute (unnormalized) CP weights for cal data: current constrained likelihoods over mixture density
            w_cal = cal_constrained_liks_curr[:,-1].flatten() / cal_mixture_constrained_density

            ## Compute estimated test point weight as the expectation of the ratio, with probabilities in the expectation given by prop_constrained_liks_curr[:,-1]
            prop_constrained_liks_curr_t = prop_constrained_liks_curr[:,-1].flatten()

            w_test = np.max(prop_constrained_liks_curr_t / prop_mixture_constrained_density)

                
            w_test *= test_pt_factor

            ## Concatenate and normalize
            sum_w_cal_test = np.sum(w_cal) + w_test
            w_cal_normalized = w_cal / sum_w_cal_test
            w_test_normalized = w_test / sum_w_cal_test
            w_infeasible_normalized = np.sum(w_cal_normalized[cal_infeasible_indicators]) + w_test_normalized



            # ## If using mean rather than max test point weight
            # w_test_mean = np.mean(prop_constrained_liks_curr_t / prop_mixture_constrained_density)
            # w_test_mean *= test_pt_factor
            # ## Concatenate and normalize
            # sum_w_cal_test_mean = np.sum(w_cal) + w_test_mean
            # w_cal_normalized_mean = w_cal / sum_w_cal_test_mean
            # w_test_normalized_mean = w_test_mean / sum_w_cal_test_mean
            # w_infeasible_normalized_mean = np.sum(w_cal_normalized_mean[cal_infeasible_indicators]) + w_test_normalized_mean

            # if proposal == 'unconstrained':
            #     breakpoint()


            ## If (Accepting null & either searching through unconstrained with previously rejected or is the last proposal):
            # if (np.sum(w_cal_normalized[cal_infeasible_indicators]) + w_test_normalized > adjusted_alpha or adjusted_alpha == 1.0): #  and ((proposal == 'unconstrained' and b > 0) or i > 0)
            if (np.sum(w_cal_normalized[cal_infeasible_indicators]) + w_test_normalized > adjusted_alpha or adjusted_alpha == 1.0): #  and ((proposal == 'unconstrained' and b > 0) or i > 0)
                ## If accepting null and previously rejected, then return most recently rejected beta_t


                # beta_t = G[b-1] if b > 0 else G[b] ## Upon first acceptance, set beta_t to most recent member of rejection set
                beta_t = beta_hat_t_curr

                psi_hat_t = importance_weighted_monte_carlo_integration(lik_ratios_unconstrained_over_safe, beta_t, proposal)

                logger.info(f"Selected beta_t = {beta_t}, psi_hat_t = {psi_hat_t}")
                logger.info(f"cal weights normalized sum : {np.sum(w_cal_normalized[cal_infeasible_indicators])}")
                logger.info(f"w_test_normalized sum : {w_test_normalized}")


                ## Compute constrained likelihoods for cal data on current candidate bound, beta_t
                # cal_constrained_liks_curr = constrain_likelihoods(cal_data_tmin1_safe_and_t_unconstrained_liks,[betas_list[-1]] + [beta_t], [psis_list[-1]] + [psi_hat_t])



                # prop_constrained_liks_curr = constrain_likelihoods(prop_data_tmin1_safe_and_t_unconstrained_liks,[betas_list[-1]] + [beta_t], [psis_list[-1]] + [psi_hat_t])


                ## Save proposals with cpc-constrained likelihoods
                constrained_liks_df_beta_hat = pd.concat([constrained_liks_df.iloc[:, :-1], pd.DataFrame({f'con_lik_r{n_cal_sets}' : prop_constrained_liks_curr[:,-1]})], axis=1)
                constrained_liks_df_beta_hat_fp = os.path.join(os.path.dirname(constrained_gen_liks_fp), f"prop_likBeta{beta_t:.3g}_interDens{psi_hat_intersection}_{os.path.basename(constrained_gen_liks_fp)}")
                constrained_liks_df_beta_hat.to_json(constrained_liks_df_beta_hat_fp, orient="records", lines=True)

                ## Also save proposals with unconstrained likelihoods
                # unconstrained_liks_df_beta_hat_fp = os.path.join(os.path.dirname(unconstrained_gen_liks_fp), f"cpc_prop_alpha{cfg.conformal_policy_control.alpha}_beta{beta_t:.3g}_{os.path.basename(unconstrained_gen_liks_fp)}")
                # unconstrained_df.to_json(unconstrained_liks_df_beta_hat_fp, orient="records", lines=True)

                check_col_names(constrained_liks_df_beta_hat)
                check_col_names(unconstrained_df)

                
                return beta_t, psi_hat_t, constrained_liks_df_beta_hat, constrained_liks_df_beta_hat_fp, unconstrained_df, unconstrained_gen_liks_fp, proposal, psi_hat_intersection #unconstrained_df, unconstrained_liks_df_beta_hat_fp
            
            else:
                ## Reject null hypothesis for beta_t, record it as the current candidate
                beta_hat_t_curr = beta_t
            # elif (np.sum(w_cal_normalized[cal_infeasible_indicators]) + w_test_normalized > adjusted_alpha or beta_t == np.inf and b == 0 and proposal == 'unconstrained' and cfg.conformal_policy_control.num_starts_beta_search == 2):

            # ## If accepting null and this is the first tested beta_t in the subsequence
            # elif (np.sum(w_cal_normalized[cal_infeasible_indicators]) + w_test_normalized > adjusted_alpha and (proposal == 'unconstrained' and b == 0)):
            #     ## If (accepting null for the first beta_t == 1 ie b == 0) and (the proposal is 'unconstrained') and (multistart==2) then go to testing with safe proposal
            #     break


            
    ## If does not find a risk-controlling policy:
    logger.info(f"WARNING : Conformal Policy Control could not control risk at desired risk level {cfg.conformal_policy_control.alpha}, with the provided safe policy, returning with what safe policy could provide.")

    psi_hat_t = importance_weighted_monte_carlo_integration(lik_ratios_unconstrained_over_safe, beta_t, proposal)

    constrained_liks_df_beta_hat = pd.concat([constrained_liks_df.iloc[:, :-1], pd.DataFrame({f'con_lik_r{n_cal_sets}' : prop_constrained_liks_curr[:,-1]})], axis=1)
    constrained_liks_df_beta_hat_fp = os.path.join(os.path.dirname(constrained_gen_liks_fp), f"prop_alpha{cfg.conformal_policy_control.alpha}_uncontrolled_beta{betas_list[-1]:.3g}_{os.path.basename(constrained_gen_liks_fp)}")
    constrained_liks_df_beta_hat.to_json(constrained_liks_df_beta_hat_fp, orient="records", lines=True)

    ## Also save proposals with unconstrained likelihoods
    # unconstrained_liks_df_beta_hat_fp = os.path.join(os.path.dirname(unconstrained_gen_liks_fp), f"prop_alpha{cfg.conformal_policy_control.alpha}_uncontrolled_beta{betas_list[-1]:.3g}_{os.path.basename(unconstrained_gen_liks_fp)}")
    # unconstrained_df.to_json(unconstrained_liks_df_beta_hat_fp, orient="records", lines=True)

    check_col_names(constrained_liks_df_beta_hat)
    check_col_names(unconstrained_df)
    
    return beta_t, psi_hat_t, constrained_liks_df_beta_hat, constrained_liks_df_beta_hat_fp, unconstrained_df, unconstrained_gen_liks_fp, proposal, psi_hat_intersection # unconstrained_liks_df_beta_hat_fp


    #     ## Temporary
    #     beta_hat = betas_list_tmp[-1]
    #     psi_hat = psis_list_tmp[-1]

    # return G, beta_hat, psi_hat
    # # return output_filepaths, hd






def run_compute_liks_all_models_and_cal_data(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    seeds_fp_list: List[str],
    prev_cal_data_fp_list: List[str],
    # data_dir: str,
    model_dir_list: List[str],
    target_fp: str,
    # particle_field: str = "higher_score_particle",
    # score_field: str = "score",
    model_indices: List[int] = [],
    temps: List[float] = [1.0],
) -> str:
    """
    Runs compute_likelihoods_all_models jobs.
    """
    opt_str = " ".join(
        get_all_strs_from_nested_dict(cfg["compute_likelihooods_all_models"]["args"])
    )


    ## Format lists of strings into a long string that python and hydra can interpret
    seeds_fp_list_str = f"\\['{seeds_fp_list[0]}'"
    model_dir_list_str = f"\\['{model_dir_list[0]}'"
    if len(prev_cal_data_fp_list) > 0:
        prev_cal_data_fp_list_str = f"\\['{prev_cal_data_fp_list[0]}'"
    else:
        prev_cal_data_fp_list_str = "\\["
    for i in range(1, len(seeds_fp_list)):
        seeds_fp_list_str += f",'{seeds_fp_list[i]}'"
        model_dir_list_str += f",'{model_dir_list[i]}'"
        if i < len(prev_cal_data_fp_list):
            prev_cal_data_fp_list_str += f",'{prev_cal_data_fp_list[i]}'"
    seeds_fp_list_str += "\\]"
    model_dir_list_str += "\\]"
    prev_cal_data_fp_list_str += "\\]"

    ## Indices of models, only if not running from beginning with r0
    if len(model_indices) > 0:
        model_indices_str = f"\\['{model_indices[0]}'"
        for i in range(1, len(model_indices)):
            model_indices_str += f",'{model_indices[i]}'"
        model_indices_str += "\\]"
    else:
        model_indices_str = "\\[\\]"
        model_indices = [i for i in range(len(model_dir_list))]


    output_dir = os.path.dirname(target_fp)


    args = f"{opt_str} input_data_path_list={seeds_fp_list_str} target_data_path={target_fp} prev_target_data_path_list={prev_cal_data_fp_list_str} model_name_or_path_list={model_dir_list_str} output_dir={output_dir} "
    # args += f"model_indices={model_indices_str}"

    args += f"model_indices={model_indices_str} overwrite_cmp_lik_all={cfg.overwrite_cmp_lik_all} "
    
    # args += f"test_fn_fp={data_dir}/ehrlich.jsonl "
    # args += f"particle_field={particle_field} "
    # args += f"score_field={score_field} "
    args += f"generation_config.max_new_tokens={cfg.compute_likelihooods_all_models.args.generation_config.max_new_tokens} "
    args += f"sanity_check={cfg.sanity_check} "

    # output_filename_prefix = f"cal_gens_all_likelihoods"
    # greedy_decoding_gen_args = f"generation_config.do_sample=False generation_config.num_beams=1 batch_size={cfg.greedy_gen_batch_size}"
    temp_sampling_gen_args = [f"generation_config.temperature={temp} " for temp in temps]

    all_gen_args = temp_sampling_gen_args
    # output_filenames = [f"{output_filename_prefix}_temp{temp}.jsonl" for temp in temps]

    # output_filepaths = [f"{model_dir_list[-1]}/{output_fn}" for output_fn in output_filenames]
    output_filepaths = [target_fp]
    output_filenames = [os.path.basename(target_fp)]
    # combined_outputs_fp = f"{model_dir_list[-1]}/{output_filename_prefix}.jsonl"
    slurm_dump_dir = f"{cfg.local_output_dir}/slurm_logs"
    os.makedirs(slurm_dump_dir, exist_ok=True)
    hd = None
    if cfg.run_compute_liks_all_models_and_cal_data:

        all_python_commands = []

        for gen_args, output_fn in zip(all_gen_args, output_filenames):
        # for gen_args in all_gen_args:
            if len(target_fp) > 0:
                ## In this condition, run compute_liks_all_models_one_target
                all_args_one_target = []


                # file_exists = fs.exists(f"{model_dir_list[-1]}/{output_fn}")
                file_exists = fs.exists(target_fp)
                target_df = pd.read_json(target_fp, orient="records", lines=True)

                if file_exists:
                    ## If file exists, check if already has all likelihoods computed for models trying to compute for
                    all_liks_computed = True
                    for m in model_indices:
                        all_liks_computed = all_liks_computed and f"lik_r{m}" in target_df.columns

                ## If not overwriting, file exists, and contains updated likelihoods (for most recent model), then don't overwrite
                if not cfg.overwrite_cmp_lik_all and file_exists and all_liks_computed:
                    logger.info(f"target_fp {target_fp} already exists with likelihoods computed. Skipping likelihoods computation...")
                else:
                    logger.info(f"Running compute_likelihoods all models...")
                    all_args_one_target.append(f"{args} {gen_args} output_filename={output_fn} ")
                ## Can only run script for updating curr data if provided non-empty filepath
                ## (so, providing empty `target_fp` allows skipping this)
                all_python_commands.extend([f"export CUDA_LAUNCH_BLOCKING=1 \n python -m compute_liks_all_models_one_target {a}" for a in all_args_one_target])


            if len(prev_cal_data_fp_list) > 0:
                all_args_prev_cal = []

                ## In this condition, run compute_likelihoods_one_model_all_data

                            # file_exists = fs.exists(f"{model_dir_list[-1]}/{output_fn}")
                all_prev_cal_files_exist = True
                all_prev_cal_has_lik_col = True
                for prev_cal_data_fp in prev_cal_data_fp_list:
                    all_prev_cal_files_exist = all_prev_cal_files_exist and fs.exists(prev_cal_data_fp)
                    if not all_prev_cal_files_exist:
                        break
                    cal_curr_df = pd.read_json(prev_cal_data_fp, orient="records", lines=True)

                    ## Check if cal file already has all of the likelihood columns trying to add now
                    for m in model_indices:
                        all_prev_cal_has_lik_col = all_prev_cal_has_lik_col and f"lik_r{m}" in cal_curr_df.columns

                ## If not overwriting, all prev cal files exist, and all contain updated likelihoods (for most recent model), then don't overwrite
                if not cfg.overwrite_cmp_lik_all and all_prev_cal_files_exist and all_prev_cal_has_lik_col:
                    logger.info(f"target_fp {target_fp} already exists with likelihoods computed. Skipping likelihoods computation...")
                else:
                    logger.info(f"Running compute_likelihoods all models...")
                    all_args_prev_cal.append(f"{args} {gen_args} output_filename={output_fn}")

                ## Can only run script for updating prev data with curr model likelihoods if provided paths to prev data
                ## (so, providing empty `prev_cal_data_fp_list` allows skipping this)
                all_python_commands.extend([f"export CUDA_LAUNCH_BLOCKING=1 \n python -m compute_likelihoods_one_model_all_data {a}" for a in all_args_prev_cal])

        
        slurm_kwargs = OmegaConf.to_container(cfg.compute_likelihooods_all_models.slurm_args)
        slurm_kwargs["job_name"] = "comp_lik_all_models"
        job_submissions = [
            submit_cmd_to_slurm(
                py_cmd,
                slurm_dump_dir,
                blocking=False,
                path_to_repo=cfg.path_to_repo,
                **slurm_kwargs,
            )
            for py_cmd in all_python_commands
        ]
        wait_for_slurm_jobs_to_complete(job_submissions)
        # hd = combine_datasets(cfg, fs, output_filepaths, combined_outputs_fp)
    return output_filepaths, hd



def get_temperatures(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    model_dir: str,
    prev_hd: Optional[float],
) -> List[float]:
    # check if previous temperatures file already exists
    temps_fp = f"{model_dir}/temperatures.json"
    if fs.exists(temps_fp):
        logger.info(f"Loading generation temperatures from {temps_fp}.")
        temps = json.load(fs.open(temps_fp))
    else:
        # if not already existing, dynamically compute and write to file
        temps = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
        # temps = [1.0]
        if cfg.temperature_scaling and prev_hd is not None:
            if prev_hd < 0.075:
                temps = [1.2, 1.4, 1.6, 1.8, 2.0, 2.2]
            elif prev_hd < 0.1:
                temps = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
            elif prev_hd < 0.15:
                temps = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
        with fs.open(temps_fp, "w") as f:
            f.write(json.dumps(temps))
    return temps


@hydra.main(config_path="config", config_name="pipeline")
def main(cfg: DictConfig):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(cfg.log_level.upper())
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    use_s3 = cfg.parent_output_dir is not None
    file_client_args = {"init_s3": use_s3}
    if use_s3:
        file_client_args["anon"] = False
        file_client_args["use_listings_cache"] = False
    file_client = LocalOrS3Client(**file_client_args)

    # Save full config to file!
    cfg_fp = (
        f"{cfg.parent_output_dir}/{cfg.run_name}/config.json"
        if use_s3
        else f"{cfg.local_output_dir}/{cfg.run_name}/config.json"
    )
    if not use_s3:
        os.makedirs(f"{cfg.local_output_dir}/{cfg.run_name}", exist_ok=True)
    if not cfg.overwrite and file_client.exists(cfg_fp):
        logger.info(f"{cfg_fp} already exists. Not overwiting.")
    else:
        cfg_dicts = pd.DataFrame([OmegaConf.to_container(cfg)])
        cfg_dicts.to_json(cfg_fp, orient="records", lines=True)


    ## Genetic algorithm to gather initial training data
    ga_data_dir = generate_ga_dataset(cfg, file_client)
    logger.info(f"GA dataset dir: {ga_data_dir}")

    ## Running list of all model paths
    all_model_paths = []

    '''If run_gpt: Pretrain model on unpaired sequences from genetic algorithm'''
    train_from_scratch = hasattr(cfg, "initial_model_config")
    if cfg.run_gpt:
        gpt_dir = train_gpt(
            cfg,
            file_client,
            ga_data_dir,
            gpt_run_name=f"{cfg.run_name}_alpha{cfg.conformal_policy_control.alpha}_gpt",
            model_dir=cfg.initial_model,
            train_from_scratch=train_from_scratch,
        )
        all_model_paths.append(gpt_dir)
        logger.info(f"GPT model dir: {all_model_paths[-1]}")
    else:
        all_model_paths.append(cfg.initial_model)
        logger.info(f"Initial model dir: {all_model_paths[-1]}")

    
    ## (Abandoned this step, due to issues with unconditional sampling) 
    ## Sample conformal calibration sequences unconditionally from GPT model
    # uncon_gen_outputs, hd = run_unconditional_generation(
    #     cfg,
    #     file_client,
    #     # combined_marge_dataset_fp,
    #     ga_data_dir,
    #     ga_data_dir,
    #     model_dir=all_model_paths[-1],
    #     particle_field="particle",
    #     score_field="score",
    #     temps=[1.0],
    # )


    ## Get name of policy improvement optimizer(s)
    setting = "" ## Setting: optimizer and constrain_against
    pi_optimizer_name = ""
    if cfg.num_sft_rounds > 0:
        pi_optimizer_name += "sft"
        setting += f"sft_CA{cfg.conformal_policy_control.constrain_against}_{cfg.setting_str}"
    elif cfg.num_dpo_rounds > 0:
        pi_optimizer_name += "dpo"
        setting += f"dpo_CA{cfg.conformal_policy_control.constrain_against}{cfg.setting_str}"
        # setting += f"dpo_CA{cfg.conformal_policy_control.constrain_against}_{cfg.conformal_policy_control.num_starts_beta_search}_ep{cfg.dpo.args.dpo_config.num_train_epochs}"
    elif cfg.num_marge_rounds > 0:
        pi_optimizer_name += "marge"
        setting += f"marge_CA{cfg.conformal_policy_control.constrain_against}_{cfg.setting_str}"
    else:
        raise ValueError("Must have at least one optimization round")


    '''Initialization: SFT pre-training, where generation has uniformly selected seeds to improve pretrained model'''
    all_prev_sft_datasets = []
    prev_round_outputs_fp = f"{ga_data_dir}/plain_pairs.jsonl" ## TO DO: Update this to samples from GPT model
    prev_hd = None

    for i in tqdm(range(cfg.num_init_sft_rounds), desc="SFT Initialization Iterations"):
        n = cfg.num_labels_after_first_round if i > 0 else None
        sft_dataset_fp = create_propen_sft_dataset(
            cfg, file_client, prev_round_outputs_fp, filename_prefix=f"sft_init_r{i}_", n=n, initial_sft=True
        )
        combined_sft_dataset_fp = combine_new_with_old_datasets(
            cfg, file_client, all_prev_sft_datasets, sft_dataset_fp
        )
        logger.info(f"SFT dataset path: {combined_sft_dataset_fp}")
        all_prev_sft_datasets.append(sft_dataset_fp)

        train_from_scratch = all_model_paths[-1] == cfg.initial_model and hasattr(
            cfg, "initial_model_config"
        )
        sft_dir = train_initial_sft(
            cfg,
            file_client,
            combined_sft_dataset_fp,
            ga_data_dir,
            initial_sft_run_name=f"{cfg.run_name}_sft_init_r{i}",
            model_dir=all_model_paths[-1],
            train_from_scratch=train_from_scratch,
        )
        

        logger.info(f"Trained initial SFT model: {sft_dir}")
        all_model_paths.append(sft_dir)

        

        seeds_fp = get_seeds_from_training_data(
            cfg, file_client,
            training_data_fp=combined_sft_dataset_fp,
            output_dir=sft_dir,
            sample_size=cfg.iterative_generation.init_args.sample_size,
            sampling_method=cfg.iterative_generation.init_args.sampling_method,
            pi_optimizer_name=pi_optimizer_name,
            setting=setting,
            random_seed = cfg.iterative_generation.init_args.seed,
        )




        # Take best checkpoint of trained model and get calibrated best likelihood range

        # if cfg.temperature_scaling:
        #     temps = get_temperatures(cfg, file_client, sft_dir, prev_hd)
        # elif i < cfg.num_init_sft_rounds - 1:
            ## At all iterations except the last, use a range of temperatures to stabilize pre-training
        # temps = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
        # init_gen_outputs_combined, init_gen_outputs_list, hd, seeds_filepaths = run_initial_generation(
        #     cfg,
        #     file_client,
        #     combined_sft_dataset_fp,
        #     ga_data_dir,
        #     sft_dir,
        #     higher_score_particle_field="higher_score_particle",
        #     lower_score_particle_field="lower_score_particle",
        #     higher_score_field="higher_score",
        #     lower_score_field="lower_score",
        #     temps=temps,
        # )

        # else:
            ## Last iteration of initialization SFT/generation will be treated as the first iterative generation/policy improvement iteration, which is "safer"
            # temps = [1.0] ## At last init iteration, use only temp=1.0 for sampling initial calibration data


    if pi_optimizer_name == "dpo":
        higher_score_particle_field="prompt"
        lower_score_particle_field="chosen"
        higher_score_field="prompt_score"
        lower_score_field="chosen_score"
    else:
        higher_score_particle_field="higher_score_particle"
        lower_score_particle_field="lower_score_particle"
        higher_score_field="higher_score"
        lower_score_field="lower_score"


    ## Generate examples from initial safe policy
    iter_gen_outputs_combined, iter_gen_outputs_list, hd = run_iterative_generation(
        cfg,
        file_client,
        seeds_fp, #combined_sft_dataset_fp,
        ga_data_dir,
        sft_dir,
        higher_score_particle_field=higher_score_particle_field,
        lower_score_particle_field=lower_score_particle_field,
        higher_score_field=higher_score_field,
        lower_score_field=lower_score_field,
        temps=[cfg.temperature_init],
        first_iter = True
    )


    ## Check that initial model is empirically safe
    init_gen_outputs_df = pd.read_json(iter_gen_outputs_list[0], orient="records", lines=True)
    init_gen_scores = init_gen_outputs_df['score'].to_numpy()
    cal_infeasible_indicators = np.isnan(init_gen_scores) | np.isinf(init_gen_scores)


    ## Initialize lists of models and seeds for policy improvement loop
    pi_model_fp_list = [all_model_paths[-1]]
    pi_seeds_filepaths_list = [seeds_fp]


    # ## Compute likelihoods for all initial generated data
    # gen_liks_fp, hd = run_compute_liks_all_models_and_cal_data(
    #     cfg,
    #     file_client,
    #     seeds_fp_list=pi_seeds_filepaths_list,
    #     prev_cal_data_fp_list=[],
    #     model_dir_list=pi_model_fp_list,
    #     target_fp=iter_gen_outputs_list[-1],
    #     temps=[cfg.temperature],
    # )
    # gen_liks_df = pd.read_json(gen_liks_fp, orient="records", lines=True)


    '''Split last batch of generated outputs into training and calibration data'''
    cal_df, cal_unconstrained_output_path, train_df, train_output_path = \
        train_cal_split_gen_outputs(cfg, file_client, iter_gen_outputs_list[0], sft_dir, first_iter=True, setting=setting) #, sample_num_cal=cfg.num_cal_per_step, sample_num_train=cfg.num_train_per_step)
    prev_round_outputs_fp = train_output_path ## Hereon, prev_round_outputs_fp will only contain training data
    # cal_data_fp_list.append(cal_output_path)
    logger.info(f"cal_r0 (n_cal{i}={len(cal_df)}) output path: {cal_unconstrained_output_path}")
    logger.info(f"train_r0 (n_tr{i}={len(train_df)}) output path: {train_output_path}")

    ## Compute likelihoods for all initial generated data
    cal_unconstrained_output_path_list, hd = run_compute_liks_all_models_and_cal_data(
        cfg,
        file_client,
        seeds_fp_list=pi_seeds_filepaths_list,
        prev_cal_data_fp_list=[],
        model_dir_list=pi_model_fp_list,
        target_fp=cal_unconstrained_output_path,
        temps=[cfg.temperature],
    )
    cal_unconstrained_output_path = cal_unconstrained_output_path_list[0]

    ## Keep track of calibration data with *unconstrained* liklihoods
    cal_data_unconstrained_fp_list = [cal_unconstrained_output_path]
    
    cal_df = pd.read_json(cal_unconstrained_output_path_list[0], orient="records", lines=True)
    ## Save initial calibration data with constrained likelihoods  
    cal_constrained_liks_df = cal_df.copy(deep=True)
    cal_constrained_liks_df = cal_constrained_liks_df[['particle', 'score', 'lik_r0']]
    cal_constrained_liks_df = cal_constrained_liks_df.rename(columns={'lik_r0' : 'con_lik_r0'})
    cal_constrained_output_path = os.path.join(os.path.dirname(cal_unconstrained_output_path), f'constrained_{os.path.basename(cal_unconstrained_output_path)}')
    cal_constrained_liks_df.to_json(cal_constrained_output_path, orient="records", lines=True)

    ## Keep track of calibration data with *constrained* liklihoods
    cal_data_constrained_fp_list = [cal_constrained_output_path]



    betas_list = [np.inf]
    psis_list = [1.0]
    proposals_list = ["safe"]
    intersection_psis_list = [1.0]



    '''SFT Policy Improvement Outer Loop, with Policy Control Inner Loop'''
    for i in tqdm(range(1, cfg.num_sft_rounds), desc="SFT Policy Improvement Iterations"):
        # n = cfg.num_labels_after_first_round if i > 0 else None
        n = cfg.num_labels_after_first_round


        ## TRAINING
        ## Format data
        sft_dataset_fp = create_propen_sft_dataset(
            cfg, file_client, prev_round_outputs_fp, filename_prefix=f"{setting}_r{i}", n=n
        )
        combined_sft_dataset_fp = combine_new_with_old_datasets(
            cfg, file_client, all_prev_sft_datasets, sft_dataset_fp
        )
        logger.info(f"SFT dataset path: {combined_sft_dataset_fp}")
        all_prev_sft_datasets.append(sft_dataset_fp)

        train_from_scratch = all_model_paths[-1] == cfg.initial_model and hasattr(
            cfg, "initial_model_config"
        )

        ## Train new model
        sft_dir = train_sft(
            cfg,
            file_client,
            combined_sft_dataset_fp,
            ga_data_dir,
            sft_run_name=f"{cfg.run_name}_alpha{cfg.conformal_policy_control.alpha}_{setting}_r{i}",
            model_dir=all_model_paths[-1],
            train_from_scratch=train_from_scratch,
        )
        
        ## Add new trained model to list
        logger.info(f"Trained SFT model: {sft_dir}")
        all_model_paths.append(sft_dir)
        pi_model_fp_list.append(sft_dir)



        ## SELECT PROMPTS: Select new prompts/seeds from recent training data
        seeds_fp = get_seeds_from_training_data(
            cfg, file_client,
            training_data_fp=combined_sft_dataset_fp,
            output_dir=sft_dir,
            sample_size=cfg.iterative_generation.args.sample_size,
            sampling_method=cfg.iterative_generation.args.sampling_method,
            random_seed = cfg.iterative_generation.args.seed,
        )
        pi_seeds_filepaths_list.append(seeds_fp)


        if cfg.temperature_scaling:
            temps = get_temperatures(cfg, file_client, sft_dir, prev_hd)
        else:
            # temps = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
            temps = [cfg.temperature]
        logger.info(f"temps: {temps}")

        ## Add curr model unconstrained likelihoods to previously collected calibration data
        cal_all_liks_fp, hd = run_compute_liks_all_models_and_cal_data(
            cfg,
            file_client,
            seeds_fp_list=pi_seeds_filepaths_list,
            prev_cal_data_fp_list=cal_data_unconstrained_fp_list,
            # data_dir=ga_data_dir,
            model_dir_list=pi_model_fp_list,
            target_fp='', ## Empty because this should already be updated #cal_output_path,
            # particle_field= "higher_score_particle",
            # score_field= "score",
            temps=[cfg.temperature],
        )
        logger.info(f"cal_all_liks_fp : {cal_all_liks_fp}")





        # ## CONFORMAL POLICY CONTROL
        beta_t, psi_hat_t, constrained_liks_df_beta_hat, constrained_liks_df_beta_hat_fp, unconstrained_df, unconstrained_liks_df_beta_hat_fp, proposal, psi_hat_intersection = run_conformal_policy_control(
            cfg,
            file_client,
            model_dir_list=pi_model_fp_list,
            seeds_fp_list=pi_seeds_filepaths_list,
            prev_cal_data_unconstrained_liks_fp_list=cal_data_unconstrained_fp_list, ## Should contain both cal data and *constrained* likelihoods
            prev_cal_data_constrained_liks_fp_list=cal_data_constrained_fp_list, ## Should contain both cal data and *constrained* likelihoods
            betas_list=betas_list,
            psis_list=psis_list, ## Normalization constants
            ga_data_dir=ga_data_dir
        )
        betas_list.append(beta_t)
        psis_list.append(psi_hat_t)
        proposals_list.append(proposal)
        intersection_psis_list.append(psi_hat_intersection)

        ## Save selected hyperparameters
        selected_hyperparams = {'beta_hats': betas_list,
                'psi_hats': psis_list,
                'proposals': proposals_list,
                'intersection_psis' : intersection_psis_list}
        selected_hyperparams_df = pd.DataFrame(selected_hyperparams)
        selected_hyperparams_df.to_json(os.path.join(pi_model_fp_list[-1], 'selected_hyperparams.json'), orient="records", lines=True)


        if constrained_liks_df_beta_hat.iloc[0,0] != unconstrained_df.iloc[0,0]:
            ## Sanity check
            raise ValueError(f"constrained_liks_df_beta_hat.iloc[0,0] ({constrained_liks_df_beta_hat.iloc[0,0]}) != ({unconstrained_df.iloc[0,0]}) unconstrained_df.iloc[0,0]")
        


        ## Add constrained likelihoods for the current model to previous calibration data
        for c_i, cal_dat_constrained_fp in enumerate(cal_data_constrained_fp_list):

            ## Load currently available constrained (0:t-1) and unconstrained (0:t) likelihoods
            cal_data_constrained_curr   = pd.read_json(cal_data_constrained_fp, orient="records", lines=True)
            cal_data_unconstrained_curr = pd.read_json(cal_data_unconstrained_fp_list[c_i], orient="records", lines=True)

            ## Compute constrained likelihoods for (t), which only requires constrained from (t-1) and unconstrained for (t)
            if cfg.conformal_policy_control.constrain_against == 'init':
                cal_liks_df_t0_safe_and_t_unconstrained_mat = pd.concat([cal_data_constrained_curr['con_lik_r0'], cal_data_unconstrained_curr.iloc[:,-1]], axis=1).to_numpy() ## Double check this
                cal_constrained_t_curr = constrain_likelihoods(cfg, cal_liks_df_t0_safe_and_t_unconstrained_mat, [betas_list[0], betas_list[-1]], [psis_list[0], psis_list[-1]])
            else:
                cal_liks_df_t0_safe_and_t_unconstrained_mat = pd.concat([cal_data_constrained_curr.iloc[:, -1], cal_data_unconstrained_curr.iloc[:,-1]], axis=1).to_numpy()
                cal_constrained_t_curr = constrain_likelihoods(cfg, cal_liks_df_t0_safe_and_t_unconstrained_mat, betas_list, psis_list)
            
            

            ## Add recently computed constrained likelihoods for (t) to the previously computed (0:t-1) values
            constrained_liks_df_beta_hat = pd.concat([cal_data_constrained_curr, pd.DataFrame({f'con_lik_r{i}' : cal_constrained_t_curr[:,-1]})], axis=1)
            # constrained_liks_df_beta_hat_fp = os.path.join(os.path.dirname(constrained_gen_liks_fp), f"cpc_prop_{constrained_gen_liks_fp}")
            constrained_liks_df_beta_hat.to_json(cal_dat_constrained_fp, orient="records", lines=True)


        # # first_iter = True if i == 0 else False ## bool: whether is first iteration (if so, will select seeds uniformly for a safe initial policy)
        # iter_gen_outputs_combined, iter_gen_outputs_list, hd = run_iterative_generation(
        #     cfg,
        #     file_client,
        #     seeds_fp, #combined_sft_dataset_fp,
        #     ga_data_dir,
        #     sft_dir,
        #     higher_score_particle_field="higher_score_particle",
        #     lower_score_particle_field="lower_score_particle",
        #     higher_score_field="higher_score",
        #     lower_score_field="lower_score",
        #     temps=temps
        # )
        # prev_hd = hd

        # logger.info(f"iter_gen_outputs_combined: {iter_gen_outputs_combined}")
        # logger.info(f"iter_gen_outputs_list: {iter_gen_outputs_list}")



        # # pi_model_dirs_list.append(sft_dir)



        ## Compute likelihoods for all initial generated data
        gen_liks_fp, hd = run_compute_liks_all_models_and_cal_data(
            cfg,
            file_client,
            seeds_fp_list=pi_seeds_filepaths_list,
            prev_cal_data_fp_list=[], ## Empty because not updating previous cal data likelihoods here
            model_dir_list=pi_model_fp_list,
            target_fp=iter_gen_outputs_list[-1],
            temps=[cfg.temperature],
        )


        # ## Contrastive generation to get test point weight
        # contrast_gen_outputs, hd = run_contrastive_generation(
        #     cfg,
        #     file_client,
        #     data_fp_list=pi_seeds_filepaths_list,
        #     data_dir=ga_data_dir,
        #     model_dir_list=pi_model_fp_list,
        #     particle_field= "higher_score_particle",
        #     score_field= "score",
        #     temps=[1.0],
        # )

    

        '''Split last batch of generated outputs into training and calibration data'''
        cal_df, cal_unconstrained_output_path, train_df, train_output_path = \
            train_cal_split_gen_outputs(cfg, file_client, iter_gen_outputs_list[0], sft_dir, setting=setting) #, sample_num_cal=cfg.num_cal_per_step, sample_num_train=cfg.num_train_per_step)
        prev_round_outputs_fp = train_output_path ## Hereon, prev_round_outputs_fp will only contain training data
        cal_data_unconstrained_fp_list.append(cal_unconstrained_output_path)
        logger.info(f"cal_r0 (n_cal{i}={len(cal_df)}) output path: {cal_unconstrained_output_path}")
        logger.info(f"train_r0 (n_tr{i}={len(train_df)}) output path: {train_output_path}")



        ## Save new calibration data with constrained likelihoods  
        cal_constrained_liks_df = constrained_liks_df_beta_hat.loc[cal_df.index]
        cal_constrained_liks_df = cal_constrained_liks_df.rename(columns={'lik_r0' : 'con_lik_r0'})
        cal_constrained_output_path = os.path.join(os.path.dirname(cal_unconstrained_output_path), f'cpc_constrained_{os.path.basename(cal_unconstrained_output_path)}')
        cal_constrained_liks_df.to_json(cal_constrained_output_path, orient="records", lines=True)

        ## Keep track of calibration data with *constrained* liklihoods
        cal_data_constrained_fp_list.append(cal_constrained_output_path)




        # prev_round_outputs_fp = iter_gen_outputs


    all_prev_dpo_datasets = []
    '''DPO Policy Improvement Outer Loop, with Policy Control Inner Loop'''
    for i in tqdm(range(1, cfg.num_dpo_rounds), desc="DPO Policy Improvement Iterations"):
        # n = cfg.num_labels_after_first_round if i > 0 else None


        ## TRAINING
        ## Format data
        n = cfg.num_labels_after_first_round
        dpo_dataset_fp = create_propen_preference_dataset(
            cfg, file_client, prev_round_outputs_fp, filename_prefix=f"alpha{cfg.conformal_policy_control.alpha}_{setting}_r{i}", n=n
        )
        combined_dpo_dataset_fp = combine_new_with_old_datasets(
            cfg, file_client, all_prev_dpo_datasets, dpo_dataset_fp
        )
        logger.info(f"DPO dataset path: {combined_dpo_dataset_fp}")
        all_prev_dpo_datasets.append(dpo_dataset_fp)

        train_from_scratch = all_model_paths[-1] == cfg.initial_model and hasattr(
            cfg, "initial_model_config"
        )



        ## Train new model
        dpo_dir = train_dpo(
            cfg,
            file_client,
            data_fp=combined_dpo_dataset_fp,
            ga_data_dir=ga_data_dir,
            run_name=f"{cfg.run_name}_alpha{cfg.conformal_policy_control.alpha}_{setting}_r{i}",
            ref_model_path=all_model_paths[-1],
            # train_from_scratch=train_from_scratch,
        )


        
        ## Add new trained model to list
        logger.info(f"Trained SFT model: {dpo_dir}")
        all_model_paths.append(dpo_dir)
        pi_model_fp_list.append(dpo_dir)




        ## SELECT PROMPTS: Select new prompts/seeds from recent training data
        seeds_fp = get_seeds_from_training_data(
            cfg, file_client,
            training_data_fp=combined_dpo_dataset_fp,
            output_dir=dpo_dir,
            sample_size=cfg.iterative_generation.args.sample_size,
            sampling_method=cfg.iterative_generation.args.sampling_method,
            higher_score_particle_field="prompt",
            lower_score_particle_field="chosen",
            higher_score_field="prompt_score",
            lower_score_field="chosen_score",
            random_seed = cfg.iterative_generation.args.seed,
        )
        pi_seeds_filepaths_list.append(seeds_fp)



        if cfg.temperature_scaling:
            temps = get_temperatures(cfg, file_client, dpo_dir, prev_hd)
        else:
            # temps = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
            temps = [cfg.temperature]
        logger.info(f"temps: {temps}")


        ## Add curr model unconstrained likelihoods to previously collected calibration data
        cal_all_liks_fp, hd = run_compute_liks_all_models_and_cal_data(
            cfg,
            file_client,
            seeds_fp_list=pi_seeds_filepaths_list,
            prev_cal_data_fp_list=cal_data_unconstrained_fp_list,
            # data_dir=ga_data_dir,
            model_dir_list=pi_model_fp_list,
            target_fp='', ## Empty because this should already be updated #cal_output_path,
            # particle_field= "higher_score_particle",
            # score_field= "score",
            temps=[cfg.temperature],
        )
        logger.info(f"cal_all_liks_fp : {cal_all_liks_fp}")


        # ## CONFORMAL POLICY CONTROL
        beta_t, psi_hat_t, constrained_liks_df_beta_hat, constrained_liks_df_beta_hat_fp, unconstrained_df, unconstrained_liks_df_beta_hat_fp, proposal, psi_hat_intersection \
        = run_conformal_policy_control(
            cfg,
            file_client,
            model_dir_list=pi_model_fp_list,
            seeds_fp_list=pi_seeds_filepaths_list,
            prev_cal_data_unconstrained_liks_fp_list=cal_data_unconstrained_fp_list, ## Should contain both cal data and *constrained* likelihoods
            prev_cal_data_constrained_liks_fp_list=cal_data_constrained_fp_list, ## Should contain both cal data and *constrained* likelihoods
            betas_list=betas_list,
            psis_list=psis_list, ## Normalization constants
            ga_data_dir=ga_data_dir,
            higher_score_particle_field="prompt",
            lower_score_particle_field="chosen",
            higher_score_field="prompt_score",
            lower_score_field="chosen_score"
        )

       ## For now, just dealing with this edge case by continuing to next step with one action
        # n_safe_actions = max(1, n_safe_actions)


        betas_list.append(beta_t)
        psis_list.append(psi_hat_t)
        proposals_list.append(proposal)
        intersection_psis_list.append(psi_hat_intersection)

        ## Save selected hyperparameters
        selected_hyperparams = {'beta_hats': betas_list,
                'psi_hats': psis_list,
                'proposals': proposals_list,
                'intersection_psis' : intersection_psis_list}
        selected_hyperparams_df = pd.DataFrame(selected_hyperparams)
        selected_hyperparams_df.to_json(os.path.join(pi_model_fp_list[-1], 'selected_hyperparams.json'), orient="records", lines=True)


        if constrained_liks_df_beta_hat.iloc[0,0] != unconstrained_df.iloc[0,0]:
            ## Sanity check
            raise ValueError(f"constrained_liks_df_beta_hat.iloc[0,0] ({constrained_liks_df_beta_hat.iloc[0,0]}) != ({unconstrained_df.iloc[0,0]}) unconstrained_df.iloc[0,0]")
        



        check_col_names(constrained_liks_df_beta_hat)
        check_col_names(unconstrained_df)
        

        ## Add constrained likelihoods for the current model to previous calibration data
        for c_i, cal_data_constrained_fp in enumerate(cal_data_constrained_fp_list):
            cal_data_constrained_curr   = pd.read_json(cal_data_constrained_fp, orient="records", lines=True)
            cal_data_unconstrained_curr = pd.read_json(cal_data_unconstrained_fp_list[c_i], orient="records", lines=True)

            check_col_names(cal_data_constrained_curr)
            check_col_names(cal_data_unconstrained_curr)

            if cfg.conformal_policy_control.constrain_against == 'init':
                cal_liks_df_t0_safe_and_t_unconstrained_mat = pd.concat([cal_data_constrained_curr['con_lik_r0'], cal_data_unconstrained_curr.iloc[:,-1]], axis=1).to_numpy() ## Double check this
            else:
                cal_liks_df_t0_safe_and_t_unconstrained_mat = pd.concat([cal_data_constrained_curr.iloc[:, -1], cal_data_unconstrained_curr.iloc[:,-1]], axis=1).to_numpy()
            
            ## Compute constrained likelihoods, only starting from most recent safe likelihoods
            cal_constrained_t_curr = constrain_likelihoods(cfg, cal_liks_df_t0_safe_and_t_unconstrained_mat, [betas_list[0], betas_list[-1]], [psis_list[0], psis_list[-1]])

            cal_constrained_liks_df_beta_hat = pd.concat([cal_data_constrained_curr, pd.DataFrame({f'con_lik_r{i}' : cal_constrained_t_curr[:,-1]})], axis=1)
            check_col_names(cal_constrained_liks_df_beta_hat)

            # constrained_liks_df_beta_hat_fp = os.path.join(os.path.dirname(constrained_gen_liks_fp), f"cpc_prop_{constrained_gen_liks_fp}")
            cal_constrained_liks_df_beta_hat.to_json(cal_data_constrained_fp, orient="records", lines=True)
        

        ## Sample with conformal policy control
        unconstrained_df, unconstrained_gen_liks_fp, constrained_liks_df, constrained_gen_liks_fp\
            = accept_reject_sample_and_get_likelihoods(cfg, file_client, pi_model_fp_list, pi_seeds_filepaths_list, dpo_dir,\
                                                       betas_list, psis_list, \
                                                       cfg.conformal_policy_control.accept_reject.n_target,\
                                                       ga_data_dir, higher_score_particle_field="prompt",
                                                       lower_score_particle_field="chosen",
                                                       higher_score_field="prompt_score",
                                                       lower_score_field="chosen_score", proposal = proposal, post_policy_control=True)


        '''Split last batch of generated outputs into training and calibration data'''
        cal_df, cal_unconstrained_output_path, train_df, train_output_path = \
            train_cal_split_gen_outputs(cfg, file_client, unconstrained_gen_liks_fp, dpo_dir, setting=setting) #, sample_num_cal=cfg.num_cal_per_step, sample_num_train=cfg.num_train_per_step)
        prev_round_outputs_fp = train_output_path ## Hereon, prev_round_outputs_fp will only contain training data
        cal_data_unconstrained_fp_list.append(cal_unconstrained_output_path)
        logger.info(f"cal_r0 (n_cal{i}={len(cal_df)}) output path: {cal_unconstrained_output_path}")
        logger.info(f"train_r0 (n_tr{i}={len(train_df)}) output path: {train_output_path}")


        ## Save new calibration data with constrained likelihoods  
        # cal_constrained_liks_df = constrained_liks_df_beta_hat.loc[cal_df.index]
        cal_constrained_liks_df = constrained_liks_df.loc[cal_df.index]
        cal_constrained_liks_df = cal_constrained_liks_df.rename(columns={'lik_r0' : 'con_lik_r0'})
        cal_constrained_output_path = os.path.join(os.path.dirname(cal_unconstrained_output_path), f'cpc_constrained_{os.path.basename(cal_unconstrained_output_path)}')
        cal_constrained_liks_df.to_json(cal_constrained_output_path, orient="records", lines=True)

        ## Keep track of calibration data with *constrained* liklihoods
        cal_data_constrained_fp_list.append(cal_constrained_output_path)




        # ## Add constrained likelihoods for the current model to previous calibration data
        # for c_i, cal_dat_constrained_fp in enumerate(cal_data_constrained_fp_list):
        #     cal_data_constrained_curr   = pd.read_json(cal_data_constrained_fp, orient="records", lines=True)
        #     cal_data_unconstrained_curr = pd.read_json(cal_data_unconstrained_fp_list[c_i], orient="records", lines=True)

        #     cal_liks_df_tmin1_safe_and_t_unconstrained_mat = pd.concat([cal_data_constrained_curr.iloc[:, -1], cal_data_unconstrained_curr.iloc[:,-1]], axis=1).to_numpy() ## Double check this

        #     ## Compute constrained likelihoods, only starting from most recent safe likelihoods
        #     cal_constrained_t_curr = constrain_likelihoods(cal_liks_df_tmin1_safe_and_t_unconstrained_mat, betas_list, psis_list)


        #     constrained_liks_df_beta_hat = pd.concat([cal_data_constrained_curr, pd.DataFrame({f'con_lik_r{i}' : cal_constrained_t_curr[:,-1]})], axis=1)
        #     # constrained_liks_df_beta_hat_fp = os.path.join(os.path.dirname(constrained_gen_liks_fp), f"cpc_prop_{constrained_gen_liks_fp}")
        #     constrained_liks_df_beta_hat.to_json(cal_dat_constrained_fp, orient="records", lines=True)

 

        # ## Compute likelihoods for all initial generated data
        # gen_liks_fp, hd = run_compute_liks_all_models_and_cal_data(
        #     cfg,
        #     file_client,
        #     seeds_fp_list=pi_seeds_filepaths_list,
        #     prev_cal_data_fp_list=[], ## Empty because not updating previous cal data likelihoods here
        #     model_dir_list=pi_model_fp_list,
        #     target_fp=iter_gen_outputs_list[-1],
        #     temps=[cfg.temperature],
        # )


    

        # '''Split last batch of generated outputs into training and calibration data'''
        # cal_df, cal_unconstrained_output_path, train_df, train_output_path = \
        #     train_cal_split_gen_outputs(cfg, file_client, unconstrained_liks_df_beta_hat_fp, dpo_dir) #, sample_num_cal=cfg.num_cal_per_step, sample_num_train=cfg.num_train_per_step)
        # prev_round_outputs_fp = train_output_path ## Hereon, prev_round_outputs_fp will only contain training data
        # cal_data_unconstrained_fp_list.append(cal_unconstrained_output_path)
        # logger.info(f"cal_r0 (n_cal{i}={len(cal_df)}) output path: {cal_unconstrained_output_path}")
        # logger.info(f"train_r0 (n_tr{i}={len(train_df)}) output path: {train_output_path}")

        # ## Save new calibration data with constrained likelihoods  
        # cal_constrained_liks_df = constrained_liks_df_beta_hat.loc[cal_df.index]
        # cal_constrained_liks_df = cal_constrained_liks_df.rename(columns={'lik_r0' : 'con_lik_r0'})
        # cal_constrained_output_path = os.path.join(os.path.dirname(cal_unconstrained_output_path), f'cpc_constrained_{os.path.basename(cal_unconstrained_output_path)}')
        # cal_constrained_liks_df.to_json(cal_constrained_output_path, orient="records", lines=True)

        # ## Keep track of calibration data with *constrained* liklihoods
        # cal_data_constrained_fp_list.append(cal_constrained_output_path)








    all_prev_marge_datasets = []
    '''MARGE Policy Improvement Outer Loop, with Policy Control Inner Loop'''
    for i in tqdm(range(1, cfg.num_marge_rounds), desc="MargE Iterations"):
        # n = cfg.num_labels_after_first_round if i > 0 else None

        ## TRAINING
        ## Format data
        n = cfg.num_labels_after_first_round
        marge_dataset_fp = create_propen_sft_dataset(
            cfg, file_client, prev_round_outputs_fp, filename_prefix=f"alpha{cfg.conformal_policy_control.alpha}_{setting}__r{i}", n=n
        )
        combined_marge_dataset_fp = combine_new_with_old_datasets(
            cfg, file_client, all_prev_marge_datasets, marge_dataset_fp
        )
        logger.info(f"MARGE dataset path: {combined_marge_dataset_fp}")
        all_prev_marge_datasets.append(marge_dataset_fp)

        train_from_scratch = all_model_paths[-1] == cfg.initial_model and hasattr(
            cfg, "initial_model_config"
        )

        ## Train new model
        marge_dir = train_marge(
            cfg,
            file_client,
            data_fp=combined_marge_dataset_fp,
            ga_data_dir=ga_data_dir,
            run_name=f"{cfg.run_name}_alpha{cfg.conformal_policy_control.alpha}_{setting}_r{i}",
            ref_model_path=all_model_paths[-1],
            # train_from_scratch=train_from_scratch,
        )


        ## Add new trained model to list
        logger.info(f"Trained MARGE model: {marge_dir}")
        all_model_paths.append(marge_dir)
        pi_model_fp_list.append(marge_dir)


        ## SELECT PROMPTS: Select new prompts/seeds from recent training data
        seeds_fp = get_seeds_from_training_data(
            cfg, file_client,
            training_data_fp=combined_marge_dataset_fp,
            output_dir=marge_dir,
            sample_size=cfg.iterative_generation.args.sample_size,
            sampling_method=cfg.iterative_generation.args.sampling_method,
            random_seed = cfg.iterative_generation.args.seed,
        )
        pi_seeds_filepaths_list.append(seeds_fp)



        if cfg.temperature_scaling:
            temps = get_temperatures(cfg, file_client, marge_dir, prev_hd)
        else:
            # temps = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
            temps = [cfg.temperature]
        logger.info(f"temps: {temps}")


        ## Add curr model unconstrained likelihoods to previously collected calibration data
        cal_all_liks_fp, hd = run_compute_liks_all_models_and_cal_data(
            cfg,
            file_client,
            seeds_fp_list=pi_seeds_filepaths_list,
            prev_cal_data_fp_list=cal_data_unconstrained_fp_list,
            # data_dir=ga_data_dir,
            model_dir_list=pi_model_fp_list,
            target_fp='', ## Empty because this should already be updated #cal_output_path,
            # particle_field= "higher_score_particle",
            # score_field= "score",
            temps=[cfg.temperature],
        )
        logger.info(f"cal_all_liks_fp : {cal_all_liks_fp}")



        ### CONFORMAL POLICY CONTROL
        beta_t, psi_hat_t, constrained_liks_df_beta_hat, constrained_liks_df_beta_hat_fp, unconstrained_df, unconstrained_liks_df_beta_hat_fp, proposal, psi_hat_intersection \
        = run_conformal_policy_control(
            cfg,
            file_client,
            model_dir_list=pi_model_fp_list,
            seeds_fp_list=pi_seeds_filepaths_list,
            prev_cal_data_unconstrained_liks_fp_list=cal_data_unconstrained_fp_list, ## Should contain both cal data and *constrained* likelihoods
            prev_cal_data_constrained_liks_fp_list=cal_data_constrained_fp_list, ## Should contain both cal data and *constrained* likelihoods
            betas_list=betas_list,
            psis_list=psis_list, ## Normalization constants
            ga_data_dir=ga_data_dir
        )

        ## For now, just dealing with this edge case by continuing to next step with one action
        # n_safe_actions = max(1, n_safe_actions)


        betas_list.append(beta_t)
        psis_list.append(psi_hat_t)
        proposals_list.append(proposal)
        intersection_psis_list.append(psi_hat_intersection)

        ## Save selected hyperparameters
        selected_hyperparams = {'beta_hats': betas_list,
                'psi_hats': psis_list,
                'proposals': proposals_list,
                'intersection_psis' : intersection_psis_list}
        selected_hyperparams_df = pd.DataFrame(selected_hyperparams)
        selected_hyperparams_df.to_json(os.path.join(pi_model_fp_list[-1], 'selected_hyperparams.json'), orient="records", lines=True)


        if constrained_liks_df_beta_hat.iloc[0,0] != unconstrained_df.iloc[0,0]:
            ## Sanity check
            raise ValueError(f"constrained_liks_df_beta_hat.iloc[0,0] ({constrained_liks_df_beta_hat.iloc[0,0]}) != ({unconstrained_df.iloc[0,0]}) unconstrained_df.iloc[0,0]")



        check_col_names(constrained_liks_df_beta_hat)
        check_col_names(unconstrained_df)
        

        ## Add constrained likelihoods for the current model to previous calibration data
        for c_i, cal_data_constrained_fp in enumerate(cal_data_constrained_fp_list):
            cal_data_constrained_curr   = pd.read_json(cal_data_constrained_fp, orient="records", lines=True)
            cal_data_unconstrained_curr = pd.read_json(cal_data_unconstrained_fp_list[c_i], orient="records", lines=True)

            check_col_names(cal_data_constrained_curr)
            check_col_names(cal_data_unconstrained_curr)

            if cfg.conformal_policy_control.constrain_against=='init':
                cal_liks_df_t0_safe_and_t_unconstrained_mat = pd.concat([cal_data_constrained_curr['con_lik_r0'], cal_data_unconstrained_curr.iloc[:,-1]], axis=1).to_numpy() ## Double check this
                ## Compute constrained likelihoods, only starting from most recent safe likelihoods
                cal_constrained_t_curr = constrain_likelihoods(cfg, cal_liks_df_t0_safe_and_t_unconstrained_mat, [betas_list[0], betas_list[-1]], [psis_list[0], psis_list[-1]])
            else:
                cal_liks_df_t0_safe_and_t_unconstrained_mat = pd.concat([cal_data_constrained_curr.iloc[:, -1], cal_data_unconstrained_curr.iloc[:,-1]], axis=1).to_numpy()
                cal_constrained_t_curr = constrain_likelihoods(cfg, cal_liks_df_tmin1_safe_and_t_unconstrained_mat, betas_list, psis_list)
            

            cal_constrained_liks_df_beta_hat = pd.concat([cal_data_constrained_curr, pd.DataFrame({f'con_lik_r{i}' : cal_constrained_t_curr[:,-1]})], axis=1)
            check_col_names(cal_constrained_liks_df_beta_hat)

            # constrained_liks_df_beta_hat_fp = os.path.join(os.path.dirname(constrained_gen_liks_fp), f"cpc_prop_{constrained_gen_liks_fp}")
            cal_constrained_liks_df_beta_hat.to_json(cal_data_constrained_fp, orient="records", lines=True)
        


        ## Sample with conformal policy control
        unconstrained_df, unconstrained_gen_liks_fp, constrained_liks_df, constrained_gen_liks_fp\
            = accept_reject_sample_and_get_likelihoods(cfg, file_client, pi_model_fp_list, pi_seeds_filepaths_list, marge_dir,\
                                                       betas_list, psis_list, \
                                                       cfg.conformal_policy_control.accept_reject.n_target,\
                                                       ga_data_dir, proposal = proposal, post_policy_control=True)


        '''Split last batch of generated outputs into training and calibration data'''
        cal_df, cal_unconstrained_output_path, train_df, train_output_path = \
            train_cal_split_gen_outputs(cfg, file_client, unconstrained_gen_liks_fp, marge_dir, setting=setting) #, sample_num_cal=cfg.num_cal_per_step, sample_num_train=cfg.num_train_per_step)
            # train_cal_split_gen_outputs(cfg, file_client, unconstrained_liks_df_beta_hat_fp, marge_dir)
        prev_round_outputs_fp = train_output_path ## Hereon, prev_round_outputs_fp will only contain training data
        cal_data_unconstrained_fp_list.append(cal_unconstrained_output_path)
        logger.info(f"cal_r0 (n_cal{i}={len(cal_df)}) output path: {cal_unconstrained_output_path}")
        logger.info(f"train_r0 (n_tr{i}={len(train_df)}) output path: {train_output_path}")


        ## Save new calibration data with constrained likelihoods  
        # cal_constrained_liks_df = constrained_liks_df_beta_hat.loc[cal_df.index]
        cal_constrained_liks_df = constrained_liks_df.loc[cal_df.index]
        cal_constrained_liks_df = cal_constrained_liks_df.rename(columns={'lik_r0' : 'con_lik_r0'})
        cal_constrained_output_path = os.path.join(os.path.dirname(cal_unconstrained_output_path), f'cpc_constrained_{os.path.basename(cal_unconstrained_output_path)}')
        cal_constrained_liks_df.to_json(cal_constrained_output_path, orient="records", lines=True)

        ## Keep track of calibration data with *constrained* liklihoods
        cal_data_constrained_fp_list.append(cal_constrained_output_path)







if __name__ == "__main__":
    main()
