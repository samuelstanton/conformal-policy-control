"""Optional Weights & Biases setup shared by all pipeline entry points.

Weights & Biases is an optional dependency of the pipeline: runs must work on
machines with no ``wandb`` credentials. ``wandb.login`` raises
``UsageError`` when no API key is configured and stdin is not a tty, so it is
only called when credentials are actually available and an online run was
requested. Every other mode (and every failure) degrades to a local/offline or
disabled run instead of killing the job.
"""

import logging
import os

import wandb
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

logger = logging.getLogger(__name__)

VALID_WANDB_MODES = ("online", "offline", "disabled")


def _cfg_get(cfg: DictConfig, key: str, default: str) -> str:
    """Return ``cfg[key]`` if present and not None, else ``default``."""
    value = cfg[key] if key in cfg else None
    return default if value is None else value


def resolve_wandb_mode(cfg: DictConfig) -> str:
    """Resolve the effective wandb mode for this run.

    Falls back to ``"offline"`` when an online run is requested but no API key
    is available (via ``WANDB_API_KEY``, ``~/.netrc`` or a previous
    ``wandb login``), so that missing credentials never fail the job.

    Args:
        cfg: Job config, optionally carrying a ``wandb_mode`` entry.

    Returns:
        One of ``"online"``, ``"offline"`` or ``"disabled"``.
    """
    mode = str(_cfg_get(cfg, "wandb_mode", "offline")).lower()
    if mode not in VALID_WANDB_MODES:
        logger.warning(
            f"Unrecognized wandb_mode {mode!r}; expected one of {VALID_WANDB_MODES}. "
            "Falling back to 'disabled'."
        )
        return "disabled"

    if mode == "online" and not _has_api_key():
        logger.warning(
            "wandb_mode='online' but no wandb API key was found "
            "(set WANDB_API_KEY or run `wandb login`). Logging offline instead."
        )
        return "offline"

    return mode


def _has_api_key() -> bool:
    """Check whether wandb credentials are available without prompting."""
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        return wandb.api.api_key is not None
    except (wandb.errors.Error, OSError) as exc:
        logger.warning(f"Could not read wandb credentials: {exc}")
        return False


def _config_snapshot(cfg: DictConfig) -> dict:
    """Snapshot a config for logging, without letting it break the job.

    Resolving is best-effort: a config carrying an interpolation that cannot be
    resolved is still worth logging verbatim, and must never take down a
    training run that would otherwise have succeeded.

    Args:
        cfg: Job config.

    Returns:
        The config as a plain dict, resolved where possible.
    """
    try:
        return OmegaConf.to_container(cfg, resolve=True)
    except OmegaConfBaseException as exc:
        logger.warning(f"Could not resolve config for wandb ({exc}); logging as-is.")
        return OmegaConf.to_container(cfg, resolve=False)


def _apply_mode(cfg: DictConfig, mode: str) -> str:
    """Record the resolved mode on the config and in the environment.

    wandb internals and the HuggingFace ``WandbCallback`` read ``WANDB_MODE``
    rather than our config, and a wandb session started while probing for
    credentials would otherwise ignore the updated variable, so any such session
    is torn down here before it is re-read.

    Args:
        cfg: Job config to update in place.
        mode: Resolved wandb mode.

    Returns:
        The mode that was applied.
    """
    cfg["wandb_mode"] = mode
    os.environ["WANDB_MODE"] = mode
    try:
        wandb.teardown()
    except (wandb.errors.Error, OSError) as exc:
        logger.debug(f"wandb teardown before init failed: {exc}")
    return mode


def wandb_setup(cfg: DictConfig, default_project: str = "cpc_llm") -> str:
    """Initialize wandb logging, tolerating missing credentials.

    Runs ``wandb.login`` only for online runs with credentials available, then
    ``wandb.init`` in the resolved mode. The values in ``cfg`` are logged to the
    wandb run. ``WANDB_MODE`` is exported so downstream loggers (e.g. the
    HuggingFace ``WandbCallback`` used when ``report_to='wandb'``) pick up the
    same mode.

    Args:
        cfg: Job config. Recognized entries: ``wandb_host``, ``wandb_mode``,
            ``project_name``, ``exp_name``, ``job_name``.
        default_project: Project name used when ``cfg`` has no ``project_name``.

    Returns:
        The mode wandb was actually initialized in (``"disabled"`` if init failed).
    """
    host = _cfg_get(cfg, "wandb_host", "https://api.wandb.ai")
    project = _cfg_get(cfg, "project_name", default_project)
    group = _cfg_get(cfg, "exp_name", "default_group")
    name = cfg["job_name"] if "job_name" in cfg else None

    cfg["wandb_host"] = host
    mode = _apply_mode(cfg, resolve_wandb_mode(cfg))

    if mode == "disabled":
        logger.info("wandb logging disabled (wandb_mode='disabled').")
        return mode

    if mode == "online":
        try:
            wandb.login(host=host)
        except (wandb.errors.Error, OSError) as exc:
            logger.warning(f"wandb login failed ({exc}); logging offline instead.")
            mode = _apply_mode(cfg, "offline")

    try:
        wandb.init(
            project=project,
            mode=mode,
            group=group,
            name=name,
            config=_config_snapshot(cfg),
        )
    except (wandb.errors.Error, OSError) as exc:
        logger.warning(f"wandb init failed ({exc}); continuing without wandb logging.")
        mode = _apply_mode(cfg, "disabled")

    return mode


def wandb_log(metrics: dict, **kwargs) -> None:
    """Log metrics to wandb if a run is active, otherwise do nothing.

    Args:
        metrics: Metric name to value mapping.
        **kwargs: Forwarded to ``wandb.log`` (e.g. ``step``).
    """
    if wandb.run is None:
        return
    wandb.log(metrics, **kwargs)
