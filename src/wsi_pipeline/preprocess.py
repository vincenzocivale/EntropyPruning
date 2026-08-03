from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from tqdm.auto import tqdm


@dataclass(frozen=True)
class TridentConfig:
    trident_repo: Path
    wsi_dir: Path
    job_dir: Path
    gpus: tuple[int, ...] = (0,)
    segmenter: str = "hest"
    mag: int = 20
    patch_size: int = 512
    overlap: int = 0
    custom_list_of_wsis: Path | None = None
    search_nested: bool = True
    remove_artifacts: bool = False
    remove_penmarks: bool = False


def _base_command(config: TridentConfig) -> list[str]:
    command = [
        "python",
        str(config.trident_repo / "run_batch_of_slides.py"),
        "--wsi_dir",
        str(config.wsi_dir),
        "--job_dir",
        str(config.job_dir),
        "--gpus",
        *[str(gpu) for gpu in config.gpus],
    ]
    if config.custom_list_of_wsis is not None:
        command += ["--custom_list_of_wsis", str(config.custom_list_of_wsis)]
    if config.search_nested:
        command.append("--search_nested")
    return command


def build_trident_commands(config: TridentConfig, stages: Sequence[str]) -> list[list[str]]:
    commands: list[list[str]] = []
    for stage in stages:
        command = _base_command(config) + ["--task", stage]
        if stage == "seg":
            command += ["--segmenter", config.segmenter]
            if config.remove_artifacts:
                command.append("--remove_artifacts")
            elif config.remove_penmarks:
                command.append("--remove_penmarks")
        elif stage == "coords":
            command += [
                "--mag",
                str(config.mag),
                "--patch_size",
                str(config.patch_size),
                "--overlap",
                str(config.overlap),
            ]
        else:
            raise ValueError(f"Unsupported preprocessing stage: {stage}")
        commands.append(command)
    return commands


def run_trident_preprocessing(config: TridentConfig, stages: Sequence[str], *, dry_run: bool = False) -> None:
    config.job_dir.mkdir(parents=True, exist_ok=True)
    commands = build_trident_commands(config, stages)
    for command in tqdm(commands, desc="TRIDENT stages", unit="stage"):
        printable = shlex.join(command)
        print(f"[eaf-wsi] {printable}", flush=True)
        if not dry_run:
            subprocess.run(command, cwd=config.trident_repo, check=True)
