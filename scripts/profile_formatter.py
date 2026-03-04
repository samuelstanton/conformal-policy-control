"""Profile the dataset formatter to establish baseline performance.

Usage:
    uv run python scripts/profile_formatter.py
    uv run python scripts/profile_formatter.py --n 10000
    uv run python scripts/profile_formatter.py --detailed   # cProfile breakdown
"""

import argparse
import cProfile
import io
import logging
import pstats
import time

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

# Silence noisy logs during profiling
logging.disable(logging.WARNING)


def make_synthetic_data(
    n_particles: int = 5000,
    dim: int = 32,
    num_states: int = 32,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic particle/score data with cluster structure.

    Creates data resembling GA output: particles cluster around a few
    "motif" templates with small mutations, so nearby particles exist
    at realistic hamming distances.
    """
    rng = np.random.RandomState(seed)

    # Create clustered data (like GA output with shared motifs)
    n_templates = 20
    templates = rng.randint(0, num_states, size=(n_templates, dim))
    particles = np.empty((n_particles, dim), dtype=int)
    template_ids = rng.randint(0, n_templates, size=n_particles)
    for i in range(n_particles):
        p = templates[template_ids[i]].copy()
        # Mutate ~10-30% of positions
        n_mutations = rng.randint(dim // 10, dim // 3)
        mut_pos = rng.choice(dim, size=n_mutations, replace=False)
        p[mut_pos] = rng.randint(0, num_states, size=n_mutations)
        particles[i] = p

    # Scores: correlated with template (so nearby particles have different scores)
    base_scores = rng.exponential(scale=2.0, size=n_templates)
    scores = base_scores[template_ids] + rng.normal(0, 0.5, size=n_particles)
    scores = np.maximum(scores, 0.01)
    # ~5% infeasible
    n_inf = int(0.05 * n_particles)
    inf_idxs = rng.choice(n_particles, size=n_inf, replace=False)
    scores[inf_idxs] = np.inf
    # Loglikelihood column
    loglikelihoods = rng.normal(-5.0, 1.0, size=n_particles)

    return pd.DataFrame(
        {
            "particle": particles.tolist(),
            "score": scores,
            "loglikelihood": loglikelihoods,
        }
    )


def make_cfg(**overrides):
    """Build an OmegaConf config matching dataset_edit_pairs.yaml defaults."""
    defaults = {
        "n": None,
        "dist_x_threshold": 0.3,
        "dist_y_threshold": None,
        "score_lower_threshold": None,
        "max_proportion_infeasible": 0.1,
        "distance_metric": "hamming",
        "seed": 0,
        "n_neighbors": 30,
        "filter_by_likelihood": False,  # skip for profiling (not the bottleneck)
        "filter_by_likelihood_range": False,
        "likelihood_quantile_threshold": 0.6,
        "likelihood_quantile_range": [0.25, 0.5],
        "allow_same_score_pair": False,
    }
    defaults.update(overrides)
    return OmegaConf.create(defaults)


def time_phase(label: str, fn, *args, **kwargs):
    """Time a function and print results. Returns (elapsed_seconds, result)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    print(f"  {label}: {elapsed:.3f}s")
    return elapsed, result


def profile_find_dense_pairs(df: pd.DataFrame, cfg, label: str = ""):
    """Profile find_dense_pairs with manual phase timing."""
    import torch
    from pynndescent import PyNNDescentTransformer

    from cpc_llm.data.synthetic_dataset_formatter import (
        filter_infeasible_examples,
        get_outputs_from_idx_pairs,
    )

    print(f"\n{'=' * 60}")
    print(f"find_dense_pairs {label}(n={len(df)} particles)")
    print(f"{'=' * 60}")

    # Phase 1: data prep
    t0 = time.perf_counter()
    df_dedup = df.drop_duplicates(subset=["particle"])
    library = torch.stack([torch.LongTensor(p) for p in df_dedup["particle"]])
    scores = torch.FloatTensor([x for x in df_dedup["score"]])
    filtered = library.numpy()
    print(f"  Data prep: {time.perf_counter() - t0:.3f}s  ({len(df_dedup)} unique)")

    # Phase 2: PyNNDescent fit + transform
    t0 = time.perf_counter()
    pynn = PyNNDescentTransformer(
        n_neighbors=cfg.n_neighbors, metric=cfg.distance_metric
    ).fit(filtered)
    print(f"  PyNNDescent fit: {time.perf_counter() - t0:.3f}s")

    t0 = time.perf_counter()
    transformed = pynn.transform(filtered)
    print(f"  PyNNDescent transform: {time.perf_counter() - t0:.3f}s")

    # Phase 3: inner loop (pair finding) — intentionally uses OLD implementation
    # to measure baseline time for comparison with the optimized version.
    t0 = time.perf_counter()
    idx_pairs = set()
    for i in range(transformed.shape[0]):
        row = transformed.getrow(i)
        nearest_neighbor_idxs = row.nonzero()[1]
        within_x_distance_idxs = [
            idx
            for idx, nn_dist in zip(
                nearest_neighbor_idxs,
                np.array(
                    row[
                        [0 for _ in range(nearest_neighbor_idxs.shape[-1])],
                        nearest_neighbor_idxs,
                    ]
                )[0].tolist(),
            )
            if nn_dist <= cfg.dist_x_threshold
        ]
        for idx in within_x_distance_idxs:
            if i < idx:
                idx_pairs.add((i, idx))
            else:
                idx_pairs.add((idx, i))
    idx_pairs = list(idx_pairs)
    print(
        f"  Pair finding loop: {time.perf_counter() - t0:.3f}s  ({len(idx_pairs)} pairs)"
    )

    # Phase 4: output construction
    t0 = time.perf_counter()
    outputs = get_outputs_from_idx_pairs(
        idx_pairs, filtered, scores, allow_same_score_pair=False, df=df_dedup
    )
    print(
        f"  Output construction: {time.perf_counter() - t0:.3f}s  ({len(outputs)} records)"
    )

    # Phase 5: infeasible filtering
    t0 = time.perf_counter()
    outputs = filter_infeasible_examples(cfg, outputs)
    print(
        f"  Infeasible filtering: {time.perf_counter() - t0:.3f}s  ({len(outputs)} final)"
    )


def profile_find_preference_pairs(df: pd.DataFrame, cfg, label: str = ""):
    """Profile find_preference_pairs with manual phase timing."""
    import random
    import torch
    from pynndescent import PyNNDescentTransformer

    print(f"\n{'=' * 60}")
    print(f"find_preference_pairs {label}(n={len(df)} particles)")
    print(f"{'=' * 60}")

    random.seed(cfg.seed)

    # Phase 1: data prep
    t0 = time.perf_counter()
    df_dedup = df.drop_duplicates(subset=["particle"])
    data = df_dedup.to_dict("records")
    library = torch.stack([torch.LongTensor(p["particle"]) for p in data])
    scores = torch.FloatTensor([x["score"] for x in data])
    filtered = library.numpy()
    filtered_scores = scores
    print(f"  Data prep: {time.perf_counter() - t0:.3f}s  ({len(df_dedup)} unique)")

    # Phase 2: PyNNDescent
    t0 = time.perf_counter()
    pynn = PyNNDescentTransformer(
        n_neighbors=cfg.n_neighbors, metric=cfg.distance_metric
    ).fit(filtered)
    print(f"  PyNNDescent fit: {time.perf_counter() - t0:.3f}s")

    t0 = time.perf_counter()
    transformed = pynn.transform(filtered)
    print(f"  PyNNDescent transform: {time.perf_counter() - t0:.3f}s")

    # Phase 3: triple finding loop — intentionally uses OLD implementation
    # to measure baseline time for comparison with the optimized version.
    t0 = time.perf_counter()
    idx_triples = set()
    for i in range(transformed.shape[0]):
        row = transformed.getrow(i)
        nearest_neighbor_idxs = row.nonzero()[1]
        within_x_distance_idxs = [
            idx
            for idx, nn_dist in zip(
                nearest_neighbor_idxs,
                np.array(
                    row[
                        [0 for _ in range(nearest_neighbor_idxs.shape[-1])],
                        nearest_neighbor_idxs,
                    ]
                )[0].tolist(),
            )
            if nn_dist <= cfg.dist_x_threshold
        ]
        all_pair_idxs = [
            (k, j) for k in within_x_distance_idxs for j in within_x_distance_idxs
        ]
        all_pair_idxs.extend([(i, k) for k in within_x_distance_idxs])
        curr_score = filtered_scores[i].item()
        for pair_idx in all_pair_idxs:
            first_score = filtered_scores[pair_idx[0]].item()
            second_score = filtered_scores[pair_idx[1]].item()
            if first_score < curr_score and second_score >= curr_score:
                idx_triples.add((i, pair_idx[0], pair_idx[1]))
            elif second_score < curr_score and first_score >= curr_score:
                idx_triples.add((i, pair_idx[1], pair_idx[0]))
    idx_triples = list(idx_triples)
    print(
        f"  Triple finding loop: {time.perf_counter() - t0:.3f}s  ({len(idx_triples)} triples)"
    )

    # Phase 4: output construction
    t0 = time.perf_counter()
    outputs = []
    for x_idx, yw_idx, yl_idx in idx_triples:
        output_dict = {
            "prompt": filtered[x_idx].tolist(),
            "prompt_score": f"{filtered_scores[x_idx]:.3f}",
            "chosen": filtered[yw_idx].tolist(),
            "chosen_score": f"{filtered_scores[yw_idx]:.3f}",
            "rejected": filtered[yl_idx].tolist(),
            "rejected_score": f"{filtered_scores[yl_idx]:.3f}",
        }
        outputs.append(output_dict)
    print(
        f"  Output construction: {time.perf_counter() - t0:.3f}s  ({len(outputs)} records)"
    )


def profile_with_cprofile(df: pd.DataFrame, cfg):
    """Run cProfile on the full functions for detailed breakdown."""
    from cpc_llm.data.synthetic_dataset_formatter import (
        find_dense_pairs,
        find_preference_pairs,
    )

    print(f"\n{'=' * 60}")
    print("cProfile: find_dense_pairs")
    print(f"{'=' * 60}")
    pr = cProfile.Profile()
    pr.enable()
    find_dense_pairs(cfg, df.copy())
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(20)
    print(s.getvalue())

    print(f"\n{'=' * 60}")
    print("cProfile: find_preference_pairs")
    print(f"{'=' * 60}")
    pr = cProfile.Profile()
    pr.enable()
    find_preference_pairs(cfg, df.copy())
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(20)
    print(s.getvalue())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n", type=int, default=5000, help="Number of synthetic particles"
    )
    parser.add_argument(
        "--detailed", action="store_true", help="Run cProfile breakdown"
    )
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--dist-x-threshold", type=float, default=0.3)
    args = parser.parse_args()

    cfg = make_cfg(
        n_neighbors=args.n_neighbors,
        dist_x_threshold=args.dist_x_threshold,
    )

    print(f"Generating synthetic data: {args.n} particles, dim={args.dim}")
    df = make_synthetic_data(n_particles=args.n, dim=args.dim)
    n_finite = np.isfinite(df["score"].values).sum()
    print(
        f"  {len(df)} rows, {n_finite} finite scores, {len(df) - n_finite} inf scores"
    )

    profile_find_dense_pairs(df, cfg)
    profile_find_preference_pairs(df, cfg)

    if args.detailed:
        # Re-enable logging for cProfile (captures logging overhead too)
        logging.disable(logging.NOTSET)
        profile_with_cprofile(df, cfg)


if __name__ == "__main__":
    main()
