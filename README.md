# Conformal Policy Control

Code for "Conformal Decision Theory for AI Agents" — a framework for enabling AI agents to automatically determine their zone of competence using conformal risk-control guarantees.

By Drew Prinster, Clara Fannjiang, Ji Won Park, Anqi Liu, Suchi Saria, and Samuel Stanton.

## Overview

This project develops **Conformal Policy Control (CPC)**: a method for iteratively improving a language model policy while maintaining formal guarantees on the risk (e.g., rate of infeasible outputs) over time. The key idea is to constrain the policy's likelihood ratios relative to a safe baseline, with the constraint level calibrated via conformal prediction so that risk stays below a user-specified level alpha.

The repository contains three sets of experiments:

- **`cpc_llm/`** — The main CPC pipeline for LLMs, applied to the Ehrlich protein motif discovery task. Iteratively generates sequences, scores them, calibrates policy bounds, and trains improved policies (SFT, DPO, or MARGE).
- **`constrained_AL/`** — Conformal prediction for constrained active learning with Gaussian process surrogates, applied to tabular regression benchmarks.
- **`QA_expts/`** — Generalized conformal risk control for LLM fact-checking, controlling false discovery rate on medical QA subclaim factuality.

## Setup

Requires Python >= 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group dev
```

This installs all dependencies (including the `cpc-llm` package in editable mode) and dev tools (pytest).

## Running the CPC-LLM pipeline

The pipeline is configured via [Hydra](https://hydra.cc/). Configs live in `cpc_llm/config/`.

```bash
# Sanity check (MARGE variant, local storage only)
cpc-llm --config-name=pipeline_sanity_check_no_s3 local_output_dir=/path/to/output

# Full run with S3 storage
cpc-llm --config-name=pipeline_marge_f2 \
  local_output_dir=/path/to/local \
  parent_output_dir=s3://bucket/path
```

### Key config parameters

| Parameter | Description |
|-----------|-------------|
| `conformal_policy_control.alpha` | Risk level (e.g., 0.1 for 10% constraint violation rate) |
| `num_sft_rounds` / `num_dpo_rounds` / `num_marge_rounds` | Number of training iterations per method |
| `parent_output_dir` | S3 path for outputs (set to `null` for local-only) |
| `local_output_dir` | Local path for outputs and model checkpoints |

### Important notes

- **Storage**: The pipeline supports both local and S3 storage. Model checkpoints are copied to S3 and deleted locally after training. Set `parent_output_dir: "null"` to disable S3.
- **SLURM**: Training, generation, and scoring jobs are launched as SLURM jobs. Configure via `slurm_args` sections in the config.
- **Resuming**: The pipeline automatically resumes prior runs if launched with the same config. Use `--overwrite=True` to start fresh.
- **GPU requirements**: Training uses DDP (single-node multi-GPU). You need ~4x the model size in GPU RAM for full-precision training.

## Running tests

```bash
uv run pytest tests/ -v
```

42 unit tests covering the core computational functions. Runs in <5s with no GPU required.

## Project structure

```
cpc_llm/                  # Main CPC-LLM package (installable)
  config/                 # Hydra configs for pipeline variants
  src/cpc_llm/
    main.py               # Entry point
    calibrate/            # CPC algorithm (beta search, likelihood constraining)
    core/                 # Model inference, likelihood computation
    data/                 # Dataset generation, formatting, splitting
    infer/                # Sequence generation, acceptance-rejection sampling
    infrastructure/       # File handling (local/S3), orchestration, SLURM
    train/                # SFT, DPO, MARGE training
    test_functions/       # Ehrlich benchmark utilities
constrained_AL/           # Active learning experiments
QA_expts/                 # Medical QA experiments
notebooks/                # Visualization notebooks
tests/                    # Unit tests
```

## License

See [LICENSE](LICENSE).
