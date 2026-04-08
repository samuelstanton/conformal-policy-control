"""Retroactively tag output dirs with sweep names on the Modal volume."""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("cpc-llm-tag")
outputs_volume = modal.Volume.from_name("cpc-llm-outputs")

OUTPUTS_PATH = "/vol/outputs"


@app.function(
    volumes={OUTPUTS_PATH: outputs_volume},
    image=modal.Image.debian_slim(),
    timeout=60,
)
def tag_dirs(config_hashes: list[str], sweep_name: str) -> int:
    outputs_volume.reload()
    root = Path(OUTPUTS_PATH)
    tagged = 0
    for h in config_hashes:
        d = root / f"cpc_llm_{h}"
        if d.exists():
            (d / ".sweep_name").write_text(sweep_name)
            tagged += 1
            print(f"Tagged {d.name} -> {sweep_name}")
        else:
            print(f"Not found: cpc_llm_{h}")
    outputs_volume.commit()
    return tagged


@app.local_entrypoint()
def main() -> None:
    new_hparams_hashes = [
        "0cc1b3a5a475",
        "d6b08a7967cd",
        "f996f4908b3b",
        "f52ac8a22dd9",
        "8d38c49f584a",
        "8c80c88f8d21",
        "eb42eee5efd3",
        "a49c9fd3bd92",
    ]
    n = tag_dirs.remote(new_hparams_hashes, "unconstrained_new_hparams")
    print(f"Tagged {n} dirs")
