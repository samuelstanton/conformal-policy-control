"""
Given a set of (particle, score), generate a dataset for a particular format.
"""

import hydra
import json
import logging
import numpy as np
import pandas as pd
import pprint
import random
import torch

from itertools import chain, product

import scipy.sparse
from omegaconf import DictConfig, OmegaConf
from pynndescent import PyNNDescentTransformer
from tqdm import tqdm
from typing import Any, Dict, List, Tuple

from ..data_contracts import (
    CHOSEN,
    CHOSEN_LOGLIKELIHOOD,
    CHOSEN_SCORE,
    HIGHER_PARTICLE_LOGLIKELIHOOD,
    HIGHER_SCORE,
    HIGHER_SCORE_PARTICLE,
    LIKELIHOOD,
    LOGLIKELIHOOD,
    LOWER_PARTICLE_LOGLIKELIHOOD,
    LOWER_SCORE,
    LOWER_SCORE_PARTICLE,
    PARTICLE,
    PROMPT,
    PROMPT_LOGLIKELIHOOD,
    PROMPT_SCORE,
    REJECTED,
    REJECTED_LOGLIKELIHOOD,
    REJECTED_SCORE,
    SCORE,
)

logging.basicConfig(level="INFO", force=True)


def find_minimal_edit_pairs(cfg: DictConfig, df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Given a dataset of particles and scores, find minimal pairs that have different scores.
    """
    logging.info(f"Length of original df: {len(df)}")
    df = df.drop_duplicates(subset=[PARTICLE])
    logging.info(f"Length of df after removing duplicates: {len(df)}")
    X = np.array([p for p in df[PARTICLE]])
    scores = np.array([s for s in df[SCORE]])
    if cfg.score_lower_threshold is not None:
        X, scores = filter_by_score(X, scores, cfg.score_lower_threshold)
    pynn_transformer = PyNNDescentTransformer(
        n_neighbors=cfg.n_neighbors, metric=cfg.distance_metric
    ).fit(X)
    transformed = pynn_transformer.transform(X)

    # loop through rows instead of doing matrix computation to save memory
    idx_pairs = []
    for i in tqdm(range(transformed.shape[0])):
        # find elements that are nearest neighbors but have different score
        # from the original particle
        curr_score = scores[i]
        nearest_neighbor_idxs = set(transformed.getrow(i).nonzero()[1])
        diff_score_idxs = set(np.nonzero(scores != curr_score)[0])
        # now find the intersection
        eligible_elem_idxs = nearest_neighbor_idxs & diff_score_idxs
        if len(eligible_elem_idxs) == 0:
            logging.warning("No eligible neighbors found.")
            continue
        eligible_elem_idxs = list(eligible_elem_idxs)
        closest_eligible_nbor_idx = eligible_elem_idxs[
            np.argmin(
                np.array(
                    transformed.getrow(i)[
                        [0 for _ in range(len(eligible_elem_idxs))], eligible_elem_idxs
                    ]
                )[0]
            )
        ]
        idx_pairs.append((i, closest_eligible_nbor_idx))

    if cfg.n is not None:
        random.seed(cfg.seed)
        idx_pairs = random.sample(idx_pairs, k=min(cfg.n, len(idx_pairs)))
    outputs = get_outputs_from_idx_pairs(idx_pairs, X, scores)
    if cfg.max_proportion_infeasible is not None:
        outputs = filter_infeasible_examples(cfg, outputs)
    return outputs


def filter_by_score(
    library: torch.Tensor, scores: torch.Tensor, score_threshold: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    idxs = torch.where(scores > score_threshold)
    return library[idxs], scores[idxs]


def filter_by_score_df(df: pd.DataFrame, score_threshold: float) -> pd.DataFrame:
    return df[df[SCORE] > score_threshold]


def abs_subtract_replace_infs(x: float, y: float) -> float:
    """Replace the infinities with a score of 0.0."""
    if np.isinf(x) or np.isnan(x):
        x = 0.0
    if np.isinf(y) or np.isnan(y):
        y = 0.0
    return np.abs(x - y)


def filter_by_likelihood_range(
    df: pd.DataFrame, likelihood_quantile_range: Tuple[float, float]
) -> pd.DataFrame:
    orig_size = len(df)
    if LIKELIHOOD not in df.columns and LOGLIKELIHOOD not in df.columns:
        logging.error(
            f"Neither '{LIKELIHOOD}' nor '{LOGLIKELIHOOD}' are in DF columns. Not filtering by likelihood range."
        )
        return df
    elif LIKELIHOOD not in df.columns:
        df[LIKELIHOOD] = df[LOGLIKELIHOOD].map(lambda x: np.exp(x))
    lower_ll_bound = df[LIKELIHOOD].quantile(likelihood_quantile_range[0])
    upper_ll_bound = df[LIKELIHOOD].quantile(likelihood_quantile_range[1])
    df = df[(df[LIKELIHOOD] >= lower_ll_bound) & (df[LIKELIHOOD] <= upper_ll_bound)]
    logging.info(
        f"Filtered dataset from {orig_size} down to {len(df)} examples after filtering by likelihood range."
    )
    return df


def filter_by_likelihood_lower_threshold(
    df: pd.DataFrame, likelihood_quantile_threshold: float
) -> pd.DataFrame:
    orig_size = len(df)
    if LIKELIHOOD not in df.columns and LOGLIKELIHOOD not in df.columns:
        logging.error(
            f"Neither '{LIKELIHOOD}' nor '{LOGLIKELIHOOD}' are in DF columns. Not filtering by likelihood range."
        )
        return df
    elif LIKELIHOOD not in df.columns:
        df[LIKELIHOOD] = df[LOGLIKELIHOOD].map(lambda x: np.exp(x))
    likelihood_lower_threshold = df[LIKELIHOOD].quantile(likelihood_quantile_threshold)
    df = df[(df[LIKELIHOOD] >= likelihood_lower_threshold)]
    logging.info(
        f"Filtered dataset from {orig_size} down to {len(df)} examples after filtering by lower likelihood threshold."
    )
    return df


def _get_neighbors_within_threshold(
    transformed: scipy.sparse.spmatrix,
    row_idx: int,
    threshold: float,
) -> np.ndarray:
    """Return column indices of neighbors within distance threshold.

    Args:
        transformed: Sparse distance matrix from PyNNDescentTransformer.
        row_idx: Row index to extract neighbors for.
        threshold: Maximum distance threshold (inclusive).

    Returns:
        Array of column indices for neighbors within the threshold.
    """
    # All stored distances are positive for PyNNDescent output,
    # so .indices matches nonzero()[1] but avoids extra indexing.
    row_csr = transformed.getrow(row_idx)
    return row_csr.indices[row_csr.data <= threshold]


def find_dense_pairs(
    cfg: DictConfig, df: pd.DataFrame, allow_same_score_pair: bool = False
) -> List[Dict[str, Any]]:
    """Adapted from PropEn -- find all pairs within a specific edit distance of each other"""
    df = df.drop_duplicates(subset=[PARTICLE])
    if cfg.n is not None:
        if cfg.filter_by_likelihood and len(df) > cfg.n:
            df = filter_by_likelihood_lower_threshold(
                df, cfg.likelihood_quantile_threshold
            )
        elif cfg.filter_by_likelihood_range and len(df) > cfg.n:
            df = filter_by_likelihood_range(df, tuple(cfg.likelihood_quantile_range))
        df = df.sample(n=min(len(df), cfg.n), random_state=cfg.seed)
    if cfg.score_lower_threshold is not None:
        df = filter_by_score_df(df, cfg.score_lower_threshold)
    library = torch.stack([torch.LongTensor(p) for p in df[PARTICLE]])
    ranking_scores = torch.FloatTensor([x for x in df[SCORE]])
    filtered = library
    filtered_scores = ranking_scores
    filtered = filtered.numpy()
    pynn_transformer = PyNNDescentTransformer(
        n_neighbors=cfg.n_neighbors, metric=cfg.distance_metric
    ).fit(filtered)
    transformed = pynn_transformer.transform(filtered)
    # loop through rows instead of doing matrix computation to save memory
    idx_pairs = set([])
    for i in tqdm(range(transformed.shape[0]), desc="Finding pairs"):
        # find elements that are nearest neighbors but have different score
        # from the original particle and are within a bounded distance
        within_x_distance_idxs = _get_neighbors_within_threshold(
            transformed, i, cfg.dist_x_threshold
        )
        if cfg.dist_y_threshold is not None:
            selected_idxs = [
                idx
                for idx, score in zip(
                    within_x_distance_idxs, filtered_scores[within_x_distance_idxs]
                )
                if abs_subtract_replace_infs(score, filtered_scores[i])
                <= cfg.dist_y_threshold
            ]
        else:
            selected_idxs = within_x_distance_idxs
        for idx in selected_idxs:
            if i < idx:
                idx_pairs.add((i, idx))
            else:
                idx_pairs.add((idx, i))
    idx_pairs = list(idx_pairs)
    logging.info(f"{len(idx_pairs)} index pairs satisfied conditions.")
    outputs = get_outputs_from_idx_pairs(
        idx_pairs,
        filtered,
        filtered_scores,
        allow_same_score_pair=allow_same_score_pair,
        df=df,
    )
    if cfg.max_proportion_infeasible is not None:
        outputs = filter_infeasible_examples(cfg, outputs)
    return outputs


def get_score_pairs_df(
    i: int,
    j: int,
    scores: torch.Tensor | np.ndarray,
    library: torch.Tensor | np.ndarray,
    allow_same_score_pair: bool = False,
    loglikelihood_values: np.ndarray | None = None,
) -> Dict[str, Any] | None:
    particle_i = library[i]
    particle_j = library[j]
    # Convert to Python floats early for fast comparison and formatting
    score_i = float(scores[i])
    score_j = float(scores[j])
    if np.isnan(score_i):
        score_i = float("inf")
    if np.isnan(score_j):
        score_j = float("inf")

    if score_i > score_j:
        higher_score = score_i
        lower_score = score_j
        higher_score_p = particle_i
        lower_score_p = particle_j
        higher_idx = i
        lower_idx = j
    elif score_j > score_i:
        higher_score = score_j
        lower_score = score_i
        higher_score_p = particle_j
        lower_score_p = particle_i
        higher_idx = j
        lower_idx = i
    elif not allow_same_score_pair:
        return None
    else:
        higher_score = score_i
        lower_score = score_j
        higher_score_p = particle_i
        lower_score_p = particle_j
        higher_idx = i
        lower_idx = j

    output_dict = {
        LOWER_SCORE_PARTICLE: lower_score_p.tolist(),
        LOWER_SCORE: f"{lower_score:.3f}",
        HIGHER_SCORE_PARTICLE: higher_score_p.tolist(),
        HIGHER_SCORE: f"{higher_score:.3f}",  # higher score is worse! can be inf
    }
    if loglikelihood_values is not None:
        output_dict[LOWER_PARTICLE_LOGLIKELIHOOD] = loglikelihood_values[lower_idx]
        output_dict[HIGHER_PARTICLE_LOGLIKELIHOOD] = loglikelihood_values[higher_idx]
    return output_dict


def get_score_pairs(
    score_i: torch.Tensor | np.ndarray,
    score_j: torch.Tensor | np.ndarray,
    particle_i: torch.Tensor | np.ndarray,
    particle_j: torch.Tensor | np.ndarray,
    allow_same_score_pair: bool = False,
) -> Dict[str, Any] | None:
    # Convert to Python floats early for fast comparison and formatting
    score_i_f = float(score_i)
    score_j_f = float(score_j)
    if np.isnan(score_i_f):
        score_i_f = float("inf")
    if np.isnan(score_j_f):
        score_j_f = float("inf")

    if score_i_f > score_j_f:
        higher_score = score_i_f
        lower_score = score_j_f
        higher_score_p = particle_i
        lower_score_p = particle_j
    elif score_j_f > score_i_f:
        higher_score = score_j_f
        lower_score = score_i_f
        higher_score_p = particle_j
        lower_score_p = particle_i
    elif not allow_same_score_pair:
        return None
    else:
        higher_score = score_i_f
        lower_score = score_j_f
        higher_score_p = particle_i
        lower_score_p = particle_j

    output_dict = {
        LOWER_SCORE_PARTICLE: lower_score_p.tolist(),
        LOWER_SCORE: f"{lower_score:.3f}",
        HIGHER_SCORE_PARTICLE: higher_score_p.tolist(),
        HIGHER_SCORE: f"{higher_score:.3f}",  # higher score is worse! can be inf
    }
    return output_dict


def get_outputs_from_idx_pairs(
    idx_pairs: List[Tuple[int, int]],
    library: torch.Tensor | np.ndarray,
    scores: torch.Tensor | np.ndarray,
    allow_same_score_pair: bool = False,
    df: pd.DataFrame | None = None,
) -> List[Dict[str, Any]]:
    # Pre-extract loglikelihood values to avoid repeated df.iloc calls
    loglikelihood_values: np.ndarray | None = None
    if df is not None and LOGLIKELIHOOD in df.columns:
        loglikelihood_values = df[LOGLIKELIHOOD].values

    outputs = []
    same_score_count = 0
    for i, j in tqdm(idx_pairs, desc="Creating output records from index pairs"):
        output_dict = get_score_pairs_df(
            i,
            j,
            scores,
            library,
            allow_same_score_pair=allow_same_score_pair,
            loglikelihood_values=loglikelihood_values,
        )
        if output_dict is not None:
            outputs.append(output_dict)
        elif not allow_same_score_pair:
            same_score_count += 1
    if same_score_count > 0:
        logging.warning(f"Skipped {same_score_count} pairs with identical scores.")
    logging.info(f"{len(outputs)} output records.")
    return outputs


def filter_infeasible_examples(
    cfg: DictConfig,
    curr_examples: List[Dict[str, Any]],
    score_field: str = HIGHER_SCORE,
) -> List[Dict[str, Any]]:
    no_infeasible_seq = [
        i for i, ex in enumerate(curr_examples) if ex[score_field] != "inf"
    ]
    w_infeasible_seq = [
        i for i, ex in enumerate(curr_examples) if ex[score_field] == "inf"
    ]
    num_feasible = len(no_infeasible_seq)
    max_infeasible_examples = (
        num_feasible
        * cfg.max_proportion_infeasible
        / (1 - cfg.max_proportion_infeasible)
    )
    max_infeasible_examples = int(np.round(max_infeasible_examples))
    random.seed(cfg.seed)
    w_infeasible_seq = random.sample(
        w_infeasible_seq, k=min(len(w_infeasible_seq), max_infeasible_examples)
    )
    data_idxs = no_infeasible_seq + w_infeasible_seq
    random.shuffle(data_idxs)
    output_data = [curr_examples[i] for i in data_idxs]
    logging.info(
        f"Filtered dataset from {len(curr_examples)} to {len(output_data)} examples after downsampling infeasible examples."
    )
    return output_data


def find_preference_pairs(cfg: DictConfig, df: pd.DataFrame) -> List[Dict[str, Any]]:
    random.seed(cfg.seed)
    df = df.drop_duplicates(subset=[PARTICLE])
    if cfg.n is not None:
        if cfg.filter_by_likelihood and len(df) > cfg.n:
            df = filter_by_likelihood_lower_threshold(
                df, cfg.likelihood_quantile_threshold
            )
        elif cfg.filter_by_likelihood_range and len(df) > cfg.n:
            df = filter_by_likelihood_range(df, tuple(cfg.likelihood_quantile_range))
        df = df.sample(n=min(len(df), cfg.n), random_state=cfg.seed)

    if cfg.score_lower_threshold is not None:
        df = filter_by_score_df(df, cfg.score_lower_threshold)
    if cfg.max_proportion_infeasible is not None:
        data = filter_infeasible_examples(cfg, df.to_dict("records"), score_field=SCORE)
    else:
        data = df.to_dict("records")
    library = torch.stack([torch.LongTensor(p[PARTICLE]) for p in data])
    ranking_scores = torch.FloatTensor([x[SCORE] for x in data])
    filtered = library
    filtered_scores = ranking_scores
    filtered = filtered.numpy()
    # Pre-convert scores to numpy for fast scalar access in triple-finding loop
    scores_np = filtered_scores.numpy()
    pynn_transformer = PyNNDescentTransformer(
        n_neighbors=cfg.n_neighbors, metric=cfg.distance_metric
    ).fit(filtered)
    transformed = pynn_transformer.transform(filtered)
    # loop through rows instead of doing matrix computation to save memory
    idx_triples = set([])
    for i in tqdm(range(transformed.shape[0]), desc="Finding pairs"):
        # find elements that are nearest neighbors
        # and are within a bounded distance
        within_x_distance_idxs = _get_neighbors_within_threshold(
            transformed, i, cfg.dist_x_threshold
        )
        # get all possible pairs of (y1, y2), plus pairs with index i
        all_pair_idxs = chain(
            product(within_x_distance_idxs, within_x_distance_idxs),
            ((i, k) for k in within_x_distance_idxs),
        )
        curr_score = scores_np[i]
        # loop through all possible triples of (x, y1, y2) and check if valid
        for pair_idx in all_pair_idxs:
            first_score = scores_np[pair_idx[0]]
            second_score = scores_np[pair_idx[1]]
            if first_score < curr_score and second_score >= curr_score:
                # y1 is yw, y2 is yl
                idx_triples.add((i, pair_idx[0], pair_idx[1]))
            elif second_score < curr_score and first_score >= curr_score:
                idx_triples.add((i, pair_idx[1], pair_idx[0]))
    idx_triples = list(idx_triples)
    # Pre-format all scores as strings to avoid repeated Tensor.__format__ calls
    formatted_scores = [f"{x:.3f}" for x in scores_np]
    outputs = []
    for x_idx, yw_idx, yl_idx in tqdm(
        idx_triples, desc="Creating preference output records"
    ):
        output_dict = {
            PROMPT: filtered[x_idx].tolist(),
            PROMPT_SCORE: formatted_scores[x_idx],
            CHOSEN: filtered[yw_idx].tolist(),
            CHOSEN_SCORE: formatted_scores[yw_idx],
            REJECTED: filtered[yl_idx].tolist(),
            REJECTED_SCORE: formatted_scores[yl_idx],
        }
        if LOGLIKELIHOOD in data[0].keys():
            output_dict[PROMPT_LOGLIKELIHOOD] = data[x_idx][LOGLIKELIHOOD]
            output_dict[CHOSEN_LOGLIKELIHOOD] = data[yw_idx][LOGLIKELIHOOD]
            output_dict[REJECTED_LOGLIKELIHOOD] = data[yl_idx][LOGLIKELIHOOD]
        outputs.append(output_dict)
    return outputs


@hydra.main(config_path="../../../config", config_name="dataset_edit_pairs")
def main(cfg: DictConfig):
    logging.basicConfig(level=cfg.log_level.upper(), force=True)
    logging.info(f"Config:\n{pprint.pformat(OmegaConf.to_container(cfg))}")
    df = pd.read_json(cfg.source_dataset_path, orient="records", lines=True)
    df[PARTICLE] = df[PARTICLE].map(
        lambda input_str: (
            [int(x) for x in json.loads(input_str)]
            if isinstance(input_str, str)
            else [int(x) for x in input_str]
        )
    )
    if cfg.format == "minimal_edit_pairs":
        outputs = find_minimal_edit_pairs(cfg, df)
    elif cfg.format == "dense_neighborhood_pairs":
        outputs = find_dense_pairs(
            cfg, df, allow_same_score_pair=cfg.allow_same_score_pair
        )
    elif cfg.format == "dense_preference_pairs":
        outputs = find_preference_pairs(cfg, df)
    else:
        raise ValueError(f"Unknown format: '{cfg.format}'.")
    outputs = pd.DataFrame(outputs)
    outputs.to_json(cfg.output_path, orient="records", lines=True)


if __name__ == "__main__":
    main()
