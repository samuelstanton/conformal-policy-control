from pathlib import Path

import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from cpc_llm.infrastructure.file_handler import LocalOrS3Client
from cpc_llm.infrastructure.run_config_guard import (
    RunConfigMismatchError,
    check_config_matches_record,
    config_as_record,
    find_config_mismatches,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "cpc_llm" / "config"


def compose_config(config_name, overrides=()):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name=config_name, overrides=list(overrides))


def write_record(cfg, cfg_fp):
    """Record a config exactly as run_pipeline does."""
    pd.DataFrame([OmegaConf.to_container(cfg)]).to_json(
        cfg_fp, orient="records", lines=True
    )


@pytest.fixture
def base_cfg():
    return OmegaConf.create(
        {
            "run_name": "run",
            "initial_seed": 0,
            "last_seed": 1,
            "num_dpo_rounds": 3,
            "overwrite": False,
            "initial_model": "EleutherAI/pythia-14m",
            "conformal_policy_control": {"alpha": 0.8},
            "initial_sft": {
                "args": {
                    "training_args": {"num_train_epochs": 10, "learning_rate": 1e-6},
                    "wandb_mode": "${wandb_mode}",
                    "log_level": "info",
                }
            },
            "iterative_generation": {
                "num_jobs": 6,
                "min_rel_len_feasible_particle": 1.0,
            },
            "dpo": {"args": {"dpo_config": {"learning_rate": 1.5e-7, "beta": 0.1}}},
        }
    )


class TestFindConfigMismatches:
    def test_identical_configs_match(self, base_cfg):
        record = config_as_record(base_cfg)
        assert find_config_mismatches(record, config_as_record(base_cfg)) == []

    def test_per_run_settings_are_ignored(self, base_cfg):
        current = base_cfg.copy()
        current.initial_seed = 3
        current.last_seed = 3
        current.num_dpo_rounds = 20
        current.overwrite = True
        current.conformal_policy_control.alpha = 0.1
        assert (
            find_config_mismatches(
                config_as_record(base_cfg), config_as_record(current)
            )
            == []
        )

    def test_logging_and_parallelism_leaves_are_ignored(self, base_cfg):
        current = base_cfg.copy()
        current.initial_sft.args.log_level = "debug"
        current.initial_sft.args.wandb_mode = "disabled"
        current.iterative_generation.num_jobs = 1
        assert (
            find_config_mismatches(
                config_as_record(base_cfg), config_as_record(current)
            )
            == []
        )

    def test_reports_differing_leaf_with_both_values(self, base_cfg):
        current = base_cfg.copy()
        current.initial_sft.args.training_args.num_train_epochs = 1
        current.iterative_generation.min_rel_len_feasible_particle = 0.1
        mismatches = find_config_mismatches(
            config_as_record(base_cfg), config_as_record(current)
        )
        assert mismatches == [
            ("initial_sft.args.training_args.num_train_epochs", 10, 1),
            ("iterative_generation.min_rel_len_feasible_particle", 1.0, 0.1),
        ]

    def test_keys_absent_from_either_record_are_ignored(self, base_cfg):
        current = base_cfg.copy()
        current.dpo.args.dpo_config.max_steps = 20
        current.initial_sft.args.model_config = {"attn_implementation": "eager"}
        assert (
            find_config_mismatches(
                config_as_record(base_cfg), config_as_record(current)
            )
            == []
        )


class TestCheckConfigMatchesRecord:
    def test_no_record_passes(self, base_cfg, tmp_path):
        check_config_matches_record(
            base_cfg, LocalOrS3Client(), str(tmp_path / "config.json")
        )

    def test_round_tripped_record_matches(self, base_cfg, tmp_path):
        cfg_fp = str(tmp_path / "config.json")
        write_record(base_cfg, cfg_fp)
        check_config_matches_record(base_cfg, LocalOrS3Client(), cfg_fp)

    def test_mismatched_record_raises(self, base_cfg, tmp_path):
        cfg_fp = str(tmp_path / "config.json")
        write_record(base_cfg, cfg_fp)
        current = base_cfg.copy()
        current.dpo.args.dpo_config.learning_rate = 1e-4
        with pytest.raises(
            RunConfigMismatchError, match="dpo.args.dpo_config.learning_rate"
        ) as exc:
            check_config_matches_record(current, LocalOrS3Client(), cfg_fp)
        assert cfg_fp in str(exc.value)

    @pytest.mark.parametrize("contents", ["", "{not json"])
    def test_unreadable_record_passes(self, base_cfg, tmp_path, contents):
        cfg_fp = tmp_path / "config.json"
        cfg_fp.write_text(contents)
        check_config_matches_record(base_cfg, LocalOrS3Client(), str(cfg_fp))


class TestShippedConfigs:
    def test_smoke_outputs_are_not_reused_by_full_pipeline(self, tmp_path):
        cfg_fp = str(tmp_path / "config.json")
        write_record(compose_config("smoke"), cfg_fp)
        full = compose_config("cpc_llm", ["run_name=smoke_smaller_pythia"])
        with pytest.raises(
            RunConfigMismatchError, match="initial_sft.args.training_args"
        ):
            check_config_matches_record(full, LocalOrS3Client(), cfg_fp)

    def test_sweep_overrides_share_run_directory(self, tmp_path):
        cfg_fp = str(tmp_path / "config.json")
        write_record(compose_config("cpc_llm"), cfg_fp)
        sweep_job = compose_config(
            "cpc_llm",
            [
                "initial_seed=2",
                "last_seed=2",
                "conformal_policy_control.alpha=0.4",
                "job_submission_system=direct",
                "parent_output_dir=null",
            ],
        )
        check_config_matches_record(sweep_job, LocalOrS3Client(), cfg_fp)
