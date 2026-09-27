import os

import pytest
from omegaconf import OmegaConf

from cpc_llm.infrastructure import wandb_utils
from cpc_llm.infrastructure.wandb_utils import (
    resolve_wandb_mode,
    wandb_log,
    wandb_setup,
)


@pytest.fixture(autouse=True)
def clean_wandb_env(monkeypatch):
    """Keep the ambient wandb environment out of these tests."""
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)


class FakeRun:
    def __init__(self):
        self.logged = []

    def log(self, metrics, **kwargs):
        self.logged.append((metrics, kwargs))


class FakeWandb:
    """Minimal stand-in for the parts of the wandb API we use."""

    class errors:
        class Error(Exception):
            pass

    def __init__(self, api_key=None, login_exc=None, init_exc=None):
        self.api = type("Api", (), {"api_key": api_key})()
        self.run = None
        self.login_exc = login_exc
        self.init_exc = init_exc
        self.login_calls = []
        self.init_kwargs = None
        self.teardown_calls = 0

    def login(self, host=None):
        self.login_calls.append(host)
        if self.login_exc is not None:
            raise self.login_exc
        return True

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        if self.init_exc is not None:
            raise self.init_exc
        self.run = FakeRun()
        return self.run

    def log(self, metrics, **kwargs):
        self.run.log(metrics, **kwargs)

    def teardown(self):
        self.teardown_calls += 1


@pytest.fixture
def fake_wandb(monkeypatch):
    def _install(**kwargs):
        fake = FakeWandb(**kwargs)
        monkeypatch.setattr(wandb_utils, "wandb", fake)
        return fake

    return _install


class TestResolveWandbMode:
    def test_defaults_to_offline_when_unset(self):
        assert resolve_wandb_mode(OmegaConf.create({})) == "offline"

    def test_none_falls_back_to_offline(self):
        assert resolve_wandb_mode(OmegaConf.create({"wandb_mode": None})) == "offline"

    def test_unrecognized_mode_disables(self):
        cfg = OmegaConf.create({"wandb_mode": "nonsense"})
        assert resolve_wandb_mode(cfg) == "disabled"

    def test_online_downgrades_without_api_key(self, fake_wandb):
        fake_wandb(api_key=None)
        cfg = OmegaConf.create({"wandb_mode": "online"})
        assert resolve_wandb_mode(cfg) == "offline"

    def test_online_kept_with_api_key(self, fake_wandb):
        fake_wandb(api_key="secret")
        cfg = OmegaConf.create({"wandb_mode": "online"})
        assert resolve_wandb_mode(cfg) == "online"

    def test_online_kept_with_api_key_env_var(self, fake_wandb, monkeypatch):
        fake_wandb(api_key=None)
        monkeypatch.setenv("WANDB_API_KEY", "secret")
        cfg = OmegaConf.create({"wandb_mode": "online"})
        assert resolve_wandb_mode(cfg) == "online"


class TestWandbSetup:
    def test_disabled_skips_login_and_init(self, fake_wandb):
        fake = fake_wandb()
        cfg = OmegaConf.create({"wandb_mode": "disabled"})
        assert wandb_setup(cfg) == "disabled"
        assert fake.login_calls == []
        assert fake.init_kwargs is None
        assert os.environ["WANDB_MODE"] == "disabled"

    def test_offline_skips_login_but_inits(self, fake_wandb):
        fake = fake_wandb()
        cfg = OmegaConf.create(
            {
                "wandb_mode": "offline",
                "project_name": "p",
                "exp_name": "g",
                "job_name": "j",
            }
        )
        assert wandb_setup(cfg) == "offline"
        assert fake.login_calls == []
        assert fake.init_kwargs["mode"] == "offline"
        assert fake.init_kwargs["project"] == "p"
        assert fake.init_kwargs["group"] == "g"
        assert fake.init_kwargs["name"] == "j"

    def test_online_without_key_does_not_login(self, fake_wandb):
        fake = fake_wandb(api_key=None)
        cfg = OmegaConf.create({"wandb_mode": "online"})
        assert wandb_setup(cfg) == "offline"
        assert fake.login_calls == []
        assert fake.init_kwargs["mode"] == "offline"
        assert cfg.wandb_mode == "offline"

    def test_online_with_key_logs_in(self, fake_wandb):
        fake = fake_wandb(api_key="secret")
        cfg = OmegaConf.create({"wandb_mode": "online", "wandb_host": "https://host"})
        assert wandb_setup(cfg) == "online"
        assert fake.login_calls == ["https://host"]
        assert fake.init_kwargs["mode"] == "online"

    def test_login_failure_downgrades_to_offline(self, fake_wandb):
        fake = fake_wandb(api_key="secret", login_exc=FakeWandb.errors.Error("boom"))
        cfg = OmegaConf.create({"wandb_mode": "online"})
        assert wandb_setup(cfg) == "offline"
        assert fake.init_kwargs["mode"] == "offline"

    def test_init_failure_is_not_fatal(self, fake_wandb):
        fake_wandb(init_exc=FakeWandb.errors.Error("boom"))
        cfg = OmegaConf.create({"wandb_mode": "offline"})
        assert wandb_setup(cfg) == "disabled"
        assert cfg.wandb_mode == "disabled"
        assert os.environ["WANDB_MODE"] == "disabled"

    def test_defaults_fill_in_missing_entries(self, fake_wandb):
        fake = fake_wandb()
        cfg = OmegaConf.create({"wandb_mode": "offline"})
        wandb_setup(cfg, default_project="my_project")
        assert fake.init_kwargs["project"] == "my_project"
        assert fake.init_kwargs["group"] == "default_group"
        assert fake.init_kwargs["name"] is None
        assert cfg.wandb_host == "https://api.wandb.ai"


class TestWandbLog:
    def test_no_run_is_a_noop(self, fake_wandb):
        fake_wandb()
        wandb_log({"loss": 1.0})  # must not raise

    def test_logs_when_run_active(self, fake_wandb):
        fake = fake_wandb()
        wandb_setup(OmegaConf.create({"wandb_mode": "offline"}))
        wandb_log({"loss": 1.0}, step=3)
        assert fake.run.logged == [({"loss": 1.0}, {"step": 3})]
