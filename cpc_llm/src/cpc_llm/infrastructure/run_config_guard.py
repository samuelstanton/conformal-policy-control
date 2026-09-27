"""Detect a run directory being reused by a differently-configured pipeline run.

Every pipeline stage caches its outputs by path and skips itself when that path
already exists (unless the matching ``overwrite_*`` flag is set). Most of those
paths encode only ``run_name``, the seed, the round and a few formatting
parameters -- not the settings that actually produced the artifact. The initial
SFT model, for instance, always lives at ``{run_name}/sft_init_seed{seed}_r0``,
whatever GA dataset or number of epochs it was trained with. Pointing a run at
output directories already populated by a differently-configured run (such as
the smoke test) therefore silently reuses that run's models and datasets, and
typically fails several stages later with a symptom far from the cause (e.g. an
empty DPO preference dataset).

``run_pipeline`` records its config in ``{run_name}/config.json`` on first use;
:func:`check_config_matches_record` compares the settings that shape cached
artifacts against that record so a mismatch is reported before any stage runs.
"""

import json
import logging
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from .file_handler import LocalOrS3Client

logger = logging.getLogger(__name__)

# Config sections whose values determine the contents of cached artifacts.
# Deliberately excludes settings that are expected to vary between runs sharing
# a run directory: seeds and alpha (sweeps share the initial stages across
# them), round counts (a run may be extended), output dirs, overwrite/run flags,
# and job-submission settings.
CACHE_DEFINING_KEYS = (
    "initial_model",
    "evol_dataset_gen.args",
    "propen_dataset_formatting_initial_sft.args",
    "initial_sft.args",
    "iterative_generation",
    "split",
    "propen_dataset_formatting_sft.args",
    "propen_dataset_formatting_preference.args",
    "sft.args",
    "dpo.args",
    "marge.args",
)

# Leaf settings inside the sections above that affect logging or parallelism
# but not the artifacts themselves.
IGNORED_LEAF_KEYS = frozenset(
    {
        "log_level",
        "log_interval",
        "wandb_mode",
        "project_name",
        "exp_name",
        "job_name",
        "num_jobs",
    }
)

_MAX_REPORTED_MISMATCHES = 20


class RunConfigMismatchError(RuntimeError):
    """Raised when a run's config conflicts with the one recorded in its run directory."""


def config_as_record(cfg: DictConfig) -> dict:
    """Encode ``cfg`` the same way ``run_pipeline`` writes it to ``config.json``.

    Round-tripping through the same pandas JSON writer keeps float formatting and
    unresolved interpolations identical on both sides of a comparison.

    Args:
        cfg: Pipeline config.

    Returns:
        The config as the plain dict that would be read back from ``config.json``.
    """
    record = pd.DataFrame([OmegaConf.to_container(cfg)]).to_json(
        orient="records", lines=True
    )
    return json.loads(record)


def read_recorded_config(file_client: LocalOrS3Client, cfg_fp: str) -> dict | None:
    """Read the config recorded by a previous run.

    Args:
        file_client: Client for local or S3 paths.
        cfg_fp: Path to the recorded ``config.json``.

    Returns:
        The recorded config, or None if the file is missing, empty or unparsable
        (e.g. still being written by a concurrent sweep job).
    """
    if not file_client.exists(cfg_fp):
        return None
    with file_client.open(cfg_fp, "r") as f:
        text = f.read().strip()
    if not text:
        return None
    try:
        return json.loads(text.splitlines()[0])
    except json.JSONDecodeError as exc:
        logger.warning(
            f"Could not parse recorded config {cfg_fp} ({exc}); skipping config check."
        )
        return None


def _diff(recorded: Any, current: Any, path: str) -> list[tuple[str, Any, Any]]:
    """Recursively list leaf differences, ignoring keys missing from either side."""
    if isinstance(recorded, dict) and isinstance(current, dict):
        mismatches = []
        for key in recorded.keys() & current.keys():
            if key in IGNORED_LEAF_KEYS:
                continue
            mismatches.extend(_diff(recorded[key], current[key], f"{path}.{key}"))
        return mismatches
    return [] if recorded == current else [(path, recorded, current)]


def find_config_mismatches(recorded: dict, current: dict) -> list[tuple[str, Any, Any]]:
    """Compare the cache-defining settings of two config records.

    Keys present in only one record are ignored, so configs recorded before a
    setting was added to the pipeline remain compatible.

    Args:
        recorded: Config previously recorded in the run directory.
        current: Config of the run about to start, as from :func:`config_as_record`.

    Returns:
        Sorted ``(dotted_key, recorded_value, current_value)`` for every differing setting.
    """
    mismatches = []
    for dotted_key in CACHE_DEFINING_KEYS:
        recorded_value, current_value = recorded, current
        for part in dotted_key.split("."):
            if not isinstance(recorded_value, dict) or not isinstance(
                current_value, dict
            ):
                break
            if part not in recorded_value or part not in current_value:
                break
            recorded_value, current_value = recorded_value[part], current_value[part]
        else:
            mismatches.extend(_diff(recorded_value, current_value, dotted_key))
    return sorted(mismatches, key=lambda m: m[0])


def check_config_matches_record(
    cfg: DictConfig, file_client: LocalOrS3Client, cfg_fp: str
) -> None:
    """Fail fast if ``cfg`` would reuse artifacts produced under a different config.

    Args:
        cfg: Config of the run about to start.
        file_client: Client for local or S3 paths.
        cfg_fp: Path to the run directory's recorded ``config.json``.

    Raises:
        RunConfigMismatchError: If any cache-defining setting differs from the record.
    """
    recorded = read_recorded_config(file_client, cfg_fp)
    if recorded is None:
        return
    mismatches = find_config_mismatches(recorded, config_as_record(cfg))
    if not mismatches:
        return

    lines = [f"  {key}: {old!r} -> {new!r}" for key, old, new in mismatches]
    if len(lines) > _MAX_REPORTED_MISMATCHES:
        hidden = len(lines) - _MAX_REPORTED_MISMATCHES
        lines = lines[:_MAX_REPORTED_MISMATCHES] + [f"  ... and {hidden} more"]
    raise RunConfigMismatchError(
        f"The outputs for run_name '{cfg.run_name}' were produced with a different "
        f"configuration (recorded in {cfg_fp}). Pipeline stages are cached by path, so "
        "this run would silently reuse those models and datasets instead of producing "
        "its own. Differing settings (recorded -> current):\n"
        + "\n".join(lines)
        + "\nUse a different run_name, local_output_dir or parent_output_dir for this "
        "configuration, or remove the existing run directories. If reusing the existing "
        f"outputs is intended, delete {cfg_fp} to accept the current configuration."
    )
