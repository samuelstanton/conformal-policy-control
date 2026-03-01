# core/

Model inference and likelihood computation. Wraps HuggingFace transformers for sequence generation and log-likelihood evaluation.

## Key files

- **model_client.py** — `ModelClient` class: loads a HuggingFace causal LM, handles CUDA initialization with exponential backoff, computes log-likelihoods via `compute_likelihoods_avg()`, and generates sequences via `generate()`.
- **compute_likelihoods_one_model_all_data.py** — Computes likelihoods from the latest model across all historical calibration datasets. Backfills likelihood columns for previously sampled data.
- **compute_liks_all_models_one_target.py** — `compute_likelihoods_inmemory()`: core function that accepts DataFrames and returns a DataFrame with likelihood columns appended. `compute_likelihoods_all_models_one_target()`: file-I/O wrapper for subprocess execution. Builds up the (n_particles x n_models) likelihood matrix needed by CPC.
- **score.py / score2.py** — Standalone scoring utilities for evaluating model outputs on input-target pairs.

## Data structures

Likelihood matrices are (n_particles, n_models) NumPy arrays stored as JSONL DataFrames with columns `lik_r0, lik_r1, ...` (one per model checkpoint).
