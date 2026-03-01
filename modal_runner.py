"""
Modal script for running CPC-LLM experiments on GPU.

Usage:
    # Environment test (verifies GPU, imports, configs)
    modal run modal_runner.py --test

    # Smoke test (tiny model, 1 round, ~5 min)
    modal run modal_runner.py --smoke

    # Smoke test with cached outputs (skips completed stages on re-run)
    modal run modal_runner.py --smoke --cache

    # Sanity check (MARGE, local storage, ~30 min)
    modal run modal_runner.py --config-name pipeline_sanity_check_no_s3

    # With cached outputs (skips completed stages on re-run)
    modal run modal_runner.py --config-name pipeline_sanity_check_no_s3 --cache

    # With overrides
    modal run modal_runner.py --config-name pipeline_sanity_check_no_s3 \
        --overrides "num_marge_rounds=1"
"""

from pathlib import Path

import modal

app = modal.App("cpc-llm")

script_dir = Path(__file__).parent.absolute()

LOCAL_DIR_IGNORE = [
    "**/__pycache__/**",
    "**/.git/**",
    "**/.pytest_cache/**",
    "**/*.pyc",
    "**/.DS_Store",
    "**/*.log",
    "**/outputs/**",
    "**/wandb/**",
    "**/.wandb/**",
    "**/.ruff_cache/**",
    "**/.venv/**",
    "**/notebooks/**",
]

# Persistent volume for HuggingFace model cache
hf_cache_volume = modal.Volume.from_name("cpc-llm-hf-cache", create_if_missing=True)
HF_CACHE_PATH = "/vol/hf-cache"

# Persistent volume for pipeline outputs (training checkpoints, generated data)
outputs_volume = modal.Volume.from_name("cpc-llm-outputs", create_if_missing=True)
OUTPUTS_PATH = "/vol/outputs"

# Base dependencies (cached across code changes)
# Use uv for fast dependency resolution (pip backtracking on s3fs/aiobotocore is brutal)
_base_deps = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "build-essential")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system --compile-bytecode "
        "'torch>=2.2,<2.9' "
        "transformers>=4.41 "
        "datasets>=2.20 "
        "accelerate>=0.31 "
        "peft>=0.11 "
        "trl>=0.9 "
        "botorch>=0.11 "
        "pytorch-holo>=0.0.1 "
        "numpy>=1.26 "
        "pandas>=2.2 "
        "scipy>=1.13 "
        "scikit-learn>=1.5 "
        "pynndescent>=0.5 "
        "hydra-core>=1.3 "
        "omegaconf>=2.3 "
        "tqdm>=4.66 "
        "s3fs>=2024.5 "
        "boto3>=1.34 "
        "backoff>=2.2 "
        "wandb>=0.17 "
    )
)

# Image with local code (add_local_dir must be last for cache efficiency)
image = _base_deps.add_local_dir(
    str(script_dir),
    remote_path="/app/cpc",
    ignore=LOCAL_DIR_IGNORE,
)


def _setup_env():
    """Common environment setup for remote functions."""
    import os
    import sys

    app_path = "/app/cpc"
    sys.path.insert(0, app_path)
    sys.path.insert(0, f"{app_path}/cpc_llm/src")
    os.chdir(app_path)

    # Point HuggingFace cache at the persistent volume
    os.environ["HF_HOME"] = HF_CACHE_PATH
    os.environ["TRANSFORMERS_CACHE"] = f"{HF_CACHE_PATH}/hub"

    # Ensure subprocesses can find cpc_llm
    python_path = os.environ.get("PYTHONPATH", "")
    new_paths = f"{app_path}:{app_path}/cpc_llm/src"
    if new_paths not in python_path:
        os.environ["PYTHONPATH"] = (
            f"{new_paths}:{python_path}" if python_path else new_paths
        )

    return app_path


@app.function(
    image=image,
    gpu="A100",
    memory=32768,
    timeout=14400,  # 4 hours
    secrets=[modal.Secret.from_name("wandb")],
    volumes={HF_CACHE_PATH: hf_cache_volume, OUTPUTS_PATH: outputs_volume},
)
def run_experiment_remote(
    config_name: str = "pipeline_sanity_check_no_s3",
    overrides: list | None = None,
    cache: bool = False,
):
    """Run CPC-LLM pipeline with the specified Hydra config.

    Args:
        config_name: Hydra config to use.
        overrides: List of Hydra overrides.
        cache: If True, reuse pipeline outputs (checkpoints, generated data)
            from previous runs, skipping completed stages. If False (default),
            clear the outputs volume first. Either way, outputs are persisted
            to the volume for future --cache runs.
    """
    import logging
    import shutil

    app_path = _setup_env()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting CPC-LLM with config: {config_name}")

    import torch

    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Outputs always persist to the volume; --no-cache empties it first.
    output_dir = OUTPUTS_PATH
    if not cache:
        output_path = Path(output_dir)
        if output_path.exists():
            for child in output_path.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            outputs_volume.commit()
            logger.info("Cleared outputs volume (--no-cache)")
    else:
        logger.info(f"Reusing cached outputs at {output_dir}")

    try:
        import hydra
        from omegaconf import OmegaConf

        override_list = overrides or []
        # Force direct execution (no SLURM) and local storage
        modal_overrides = [
            "job_submission_system=direct",
            "parent_output_dir=null",
            f"local_output_dir={output_dir}",
            f"path_to_repo={app_path}",
        ]
        override_list.extend(modal_overrides)

        config_path = f"{app_path}/cpc_llm/config"
        with hydra.initialize_config_dir(config_dir=config_path, version_base="1.1"):
            cfg = hydra.compose(config_name=config_name, overrides=override_list)
            logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

            from cpc_llm.main import run_pipeline

            run_pipeline(cfg)

        # Persist any newly downloaded models and pipeline outputs to volumes
        hf_cache_volume.commit()
        outputs_volume.commit()

        logger.info("CPC-LLM pipeline completed!")
        return "Success"

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        outputs_volume.commit()  # persist partial work so --cache can resume
        raise


@app.function(
    image=image,
    gpu="A100",
    memory=16384,
    timeout=300,
    volumes={HF_CACHE_PATH: hf_cache_volume},
)
def test_environment():
    """Verify GPU, imports, configs, and HF cache."""
    _setup_env()

    import torch

    results = []
    results.append(f"Python: {__import__('sys').version}")
    results.append(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        results.append(f"CUDA device: {torch.cuda.get_device_name(0)}")
    results.append(f"PyTorch: {torch.__version__}")

    import transformers

    results.append(f"Transformers: {transformers.__version__}")

    from pathlib import Path

    config_path = Path("cpc_llm/config")
    if config_path.exists():
        config_files = sorted(f.name for f in config_path.glob("*.yaml"))
        results.append(f"Configs: {config_files}")

    import cpc_llm  # noqa: F401

    results.append("cpc_llm: imported OK")

    hf_cache = Path(HF_CACHE_PATH)
    cached_models = list(hf_cache.glob("hub/models--*"))
    results.append(f"HF cache: {len(cached_models)} models cached")

    for line in results:
        print(line)

    return "Environment test passed!"


@app.local_entrypoint()
def main_entrypoint(
    config_name: str = "pipeline_sanity_check_no_s3",
    test: bool = False,
    smoke: bool = False,
    cache: bool = False,
    overrides: str = None,
):
    """
    Local entrypoint for running CPC-LLM on Modal.

    Usage:
        # Environment test
        modal run modal_runner.py --test

        # Smoke test (tiny model, ~5 min)
        modal run modal_runner.py --smoke

        # Smoke test with cached outputs (skips completed stages)
        modal run modal_runner.py --smoke --cache

        # Sanity check run
        modal run modal_runner.py --config-name pipeline_sanity_check_no_s3

        # With cached outputs (skips completed stages)
        modal run modal_runner.py --config-name pipeline_sanity_check_no_s3 --cache

        # With overrides
        modal run modal_runner.py --config-name pipeline_sanity_check_no_s3 \
            --overrides "num_marge_rounds=1"
    """
    if test:
        print("Running environment test...")
        result = test_environment.remote()
        print(f"Result: {result}")
    else:
        override_list = []
        if overrides:
            override_list = [o.strip() for o in overrides.split(",")]
        if smoke:
            config_name = "smoke"
            print("Running smoke test (pythia-14m, 1 round)...")
        else:
            print(f"Running CPC-LLM with config: {config_name}")
        if cache:
            print("Using cached outputs volume")
        result = run_experiment_remote.remote(config_name, override_list, cache=cache)
        print(f"Result: {result}")
