from typing import Any
from omegaconf import DictConfig
import numpy as np
import pandas as pd


def constrain_likelihoods(
    cfg: DictConfig,
    likelihoods_mat : Any, ## 2-D np array, shape (n_prop, *); flexible num columns, equal to n_models total from safe model to curr
    betas : Any, ## 1-D np array or list of lik-ratio bounds
    psis : Any ## 1-D np array or list of normalization constants
) -> Any:  
    '''Process matrix of unconstrained likelihoods into constrained likelihoods'''
    n_prop, n_models = np.shape(likelihoods_mat)

    if n_models > 2 and cfg.conformal_policy_control.constrain_against == 'init':
        ## If constraining against initial safe policy, only want first model and current model
        raise ValueError("Modified to only constrain likelihoods relative to original safe policy")

    constrained_likelihoods_mat = np.zeros((n_prop, n_models))

    ## First col of likelihoods_mat should already be safe/constrained
    constrained_likelihoods_mat[:, 0] = likelihoods_mat[:, 0]  


    ## Compute constrained likelihoods for each subsequent policy and bound
    if cfg.conformal_policy_control.constrain_against == 'init':
        constrained_likelihoods_mat[:, 1] = np.where(likelihoods_mat[:, 1] / constrained_likelihoods_mat[:, 0] < betas[1], likelihoods_mat[:, 1] / psis[1], constrained_likelihoods_mat[:, 0] * (betas[1] / psis[1]))
    else:
        for i in range(1, n_models):
            constrained_likelihoods_mat[:, i] = np.where(likelihoods_mat[:, i] / constrained_likelihoods_mat[:, i-1] < betas[i], likelihoods_mat[:, i] / psis[i], constrained_likelihoods_mat[:, i-1] * (betas[i] / psis[i]))
        
    return constrained_likelihoods_mat




def mixture_pdf_from_densities_mat(
    constrained_densities_all_steps : Any, 
    mixture_weights : Any
) -> Any:
    '''
    constrained_densities_cal_test_all_steps : dim (n_samples, n_models) Note: columns correspond to t=0, ..., T-1
    mixture_weights         : dim (T), array of relative weights to put on each of *prior* distributions, from t=0, ..., T-1
                       Note : mixture_weights[i] = n_cal_model_i
    '''
    mixture_weights_normed = mixture_weights / np.sum(mixture_weights)

    mixture_pdfs = constrained_densities_all_steps @ mixture_weights_normed

    return mixture_pdfs



def check_col_names(df : Any):
    '''
    Sanity check that likelihood column names are increasing in pandas likelihoods dataframe.
    '''
    lik_cols = []
    for c in df.columns:
        if (c[0] == 'l' and c[1]=='i') or c[0] == 'c':
            lik_cols.append(c)

    col_indices = [int(c.split('_')[-1][1:]) for c in lik_cols]
    for i in range(len(col_indices)):

        if i > 0 and col_indices[i] - col_indices[i-1] != 1:
            raise ValueError(f"col indices not increasing {df.columns}")


