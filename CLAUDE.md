# Conformal Policy Control

Research codebase for "Conformal Decision Theory for AI Agents" — a framework for enabling AI agents to automatically determine their zone of competence with statistical risk-control guarantees.

## Repository structure

- `cpc_llm/` — Main package: Conformal Policy Control for LLMs. Iterative policy training with risk guarantees. See `cpc_llm/CLAUDE.md` for details.
- `constrained_AL/` — Constrained active learning experiments. Applies conformal prediction with Gaussian process surrogates to active learning benchmarks (airfoil, communities, MEPS datasets).
- `QA_expts/` — Medical QA experiments. Applies generalized conformal risk control (GCRC) to LLM fact-checking, controlling false discovery rate on subclaim factuality.
- `notebooks/` — Visualization notebooks for acceptance-rejection sampling, non-monotonic losses, and piecewise constraints.
- `tests/` — Unit tests for pure computational functions across all modules.

## Development

```bash
uv sync --group dev     # install all deps including pytest
uv run pytest tests/ -v # run tests (<5s, no GPU needed)
```

Pre-commit hooks enforce ruff linting/formatting and nbstripout on all commits.

## Code style

- Type hints on all function signatures (parameters and return types)
- Google-style docstrings on public functions and classes
- Specific exception handling — no bare `except:` clauses
- Use `X | None` instead of `Optional[X]`

## Modal (GPU execution)

```bash
uv pip install modal                          # install modal client
uv run modal run modal_runner.py --test       # env test (verifies GPU, imports)
uv run modal run modal_runner.py --smoke      # smoke test (pythia-14m, 1 round)
uv run modal run modal_runner.py --smoke --cache  # reuse outputs from previous smoke run
uv run modal run modal_runner.py --check-progress  # tail latest subprocess log
uv run modal run modal_runner.py              # full run (default config)
```

Modal automatically forces `job_submission_system=direct` and `parent_output_dir=null`.

## Project layout

```
pyproject.toml              # workspace root, shared deps, ruff config, pytest config
cpc_llm/
  pyproject.toml            # publishable package with ML dependencies
  config/                   # hydra configs for pipeline variants
  src/cpc_llm/              # package source
    main.py                 # entry point (hydra)
    calibrate/              # CPC algorithm
    core/                   # model inference, likelihood computation
    data/                   # dataset generation, formatting, splitting
    infer/                  # sequence generation, acceptance-rejection sampling
    infrastructure/         # file handling (local/S3), orchestration, SLURM
    train/                  # SFT, DPO, MARGE training
    test_functions/         # Ehrlich benchmark utilities
constrained_AL/             # active learning experiments (standalone scripts)
QA_expts/                   # medical QA experiments (standalone scripts)
notebooks/                  # visualization notebooks
tests/                      # unit tests
```
