from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
BATCH_ROOT = REPO_ROOT / "logs" / "stage1_batch"
RUN_ROOT = REPO_ROOT / "logs" / "rl_games" / "inspire_hand"


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def append(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def run_name(clip: str) -> str:
    obj, start, end, subject, sequence = clip.split("-", 4)
    return (
        f"inspire-stage1_grid_1k_{obj}{start}-{end}-{subject}-{sequence}"
        "_B128_hybrid_ho4_imi0.3_imv0.1_bc0.3"
        "_gtobj_golap20.0_gth0.001_stage1semantic"
    )


def checkpoint_epoch(path: Path) -> int:
    match = re.search(r"_ep_(\d+)_", path.name)
    return int(match.group(1)) if match else -1


def latest_checkpoint(run_dir: Path, final_only: bool = False) -> Path | None:
    checkpoints = list((run_dir / "nn").glob("last_inspire_hand_ep_*.pth"))
    if final_only:
        checkpoints = [path for path in checkpoints if checkpoint_epoch(path) == 1000]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda path: (checkpoint_epoch(path), path.stat().st_mtime_ns))


def validate_checkpoint(path: Path) -> None:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in data:
        raise ValueError(f"Checkpoint has no model state: {path}")


def command_for(clip: str, checkpoint: Path | None) -> list[str]:
    command = [
        PYTHON,
        "-u",
        "stage1/train_stage1.py",
        "--hand",
        "inspire_hand",
        "--clip",
        clip,
        "-rn",
        "para",
        "-B",
        "128",
        "-am",
        "hybrid",
        "-imi",
        "0.3",
        "-imv",
        "0.1",
        "-bc",
        "0.3",
        "-ho",
        "4",
        "-me",
        "1000",
        "-sf",
        "1000",
        "--overlap_geom_penalty",
        "20",
        "--overlap_geom_threshold",
        "0.001",
        "--overlap_inside_backend",
        "grid",
        "--overlap_grid_pitch",
        "0.002",
        "-exp",
        "stage1_grid_1k",
    ]
    if checkpoint is not None:
        command.extend(["-ck", str(checkpoint)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable per-trajectory Stage1 worker.")
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with args.manifest.open(encoding="utf-8", newline="") as handle:
        jobs = list(csv.reader(handle, delimiter="\t"))
    status = BATCH_ROOT / f"{args.worker}.status.log"
    failed = BATCH_ROOT / "failed" / f"{args.worker}.tsv"

    while True:
        pending = []
        made_progress = False
        for clip, subject, stem, frames in jobs:
            done = BATCH_ROOT / "done" / subject / f"{stem}.done"
            if done.exists():
                continue
            retarget_done = REPO_ROOT / "logs" / "retarget_batch" / "done" / subject / f"{stem}.done"
            if not retarget_done.exists():
                pending.append((clip, subject, stem, frames))
                continue

            current_run = RUN_ROOT / run_name(clip)
            final_checkpoint = latest_checkpoint(current_run, final_only=True)
            if final_checkpoint is not None:
                validate_checkpoint(final_checkpoint)
                done.parent.mkdir(parents=True, exist_ok=True)
                done.write_text(f"checkpoint={final_checkpoint}\n", encoding="utf-8")
                made_progress = True
                continue

            resume = latest_checkpoint(current_run)
            task_log = BATCH_ROOT / "task_logs" / subject / f"{stem}.log"
            task_log.parent.mkdir(parents=True, exist_ok=True)
            command = command_for(clip, resume)
            if args.dry_run:
                print(" ".join(command))
                return

            append(status, f"{timestamp()} START gpu={args.gpu} clip={clip} resume={resume or 'none'}")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            env["PYTHONPATH"] = str(REPO_ROOT) + ((":" + env["PYTHONPATH"]) if env.get("PYTHONPATH") else "")
            env["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"
            env["WANDB_MODE"] = "offline"
            env["DEX_STAGE1_OVERLAP_TIMING"] = "0"
            started = time.time()
            with task_log.open("a", encoding="utf-8") as log_handle:
                result = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
            elapsed = int(time.time() - started)
            if result.returncode != 0:
                message = f"{timestamp()} FAIL gpu={args.gpu} clip={clip} code={result.returncode} elapsed={elapsed} log={task_log}"
                append(status, message)
                append(failed, message)
                made_progress = True
                continue

            final_checkpoint = latest_checkpoint(current_run, final_only=True)
            if final_checkpoint is None:
                message = f"{timestamp()} FAIL gpu={args.gpu} clip={clip} code=no_ep1000 elapsed={elapsed} log={task_log}"
                append(status, message)
                append(failed, message)
                made_progress = True
                continue
            validate_checkpoint(final_checkpoint)
            done.parent.mkdir(parents=True, exist_ok=True)
            done.write_text(f"checkpoint={final_checkpoint}\nelapsed_seconds={elapsed}\n", encoding="utf-8")
            append(status, f"{timestamp()} OK gpu={args.gpu} clip={clip} elapsed={elapsed} checkpoint={final_checkpoint}")
            made_progress = True

        if not pending:
            append(status, f"{timestamp()} WORKER_DONE gpu={args.gpu}")
            return
        if not made_progress:
            time.sleep(max(1, args.poll_seconds))


if __name__ == "__main__":
    main()
