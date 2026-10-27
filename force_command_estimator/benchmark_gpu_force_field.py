from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class BenchmarkResult:
    batch_size: int
    latency_ms: float
    environments_per_second: float
    peak_memory_mb: float
    relative_residual: float


def build_wrench_map(anchor_positions: torch.Tensor) -> torch.Tensor:
    """Map per-anchor 3D forces to resultant force and torque."""
    batch_size, num_anchors, _ = anchor_positions.shape
    identity = torch.eye(3, device=anchor_positions.device, dtype=anchor_positions.dtype)
    force_map = identity.expand(batch_size, num_anchors, 3, 3)
    x, y, z = anchor_positions.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    torque_map = torch.stack(
        (zeros, -z, y, z, zeros, -x, -y, x, zeros), dim=-1
    ).reshape(batch_size, num_anchors, 3, 3)
    return torch.cat((force_map, torque_map), dim=-2)


def project_coulomb_cone(
    forces: torch.Tensor,
    normals: torch.Tensor,
    friction: float,
    normal_force_limit: float,
) -> torch.Tensor:
    normal_force = (forces * normals).sum(dim=-1, keepdim=True).clamp_(0.0, normal_force_limit)
    tangent_force = forces - normal_force * normals
    tangent_norm = torch.linalg.vector_norm(tangent_force, dim=-1, keepdim=True).clamp_min_(1e-8)
    tangent_limit = friction * normal_force
    tangent_force = tangent_force * torch.clamp(tangent_limit / tangent_norm, max=1.0)
    return normal_force * normals + tangent_force


@torch.inference_mode()
def solve_force_field(
    wrench_map: torch.Tensor,
    normals: torch.Tensor,
    target_wrench: torch.Tensor,
    iterations: int,
    friction: float,
    regularization: float,
    normal_force_limit: float,
) -> torch.Tensor:
    batch_size, num_anchors = normals.shape[:2]
    forces = torch.zeros(
        (batch_size, num_anchors, 3), device=normals.device, dtype=normals.dtype
    )
    lipschitz_bound = wrench_map.square().sum(dim=(-1, -2, -3))
    step_size = (1.0 / (lipschitz_bound + regularization)).view(-1, 1, 1)
    for _ in range(iterations):
        achieved = torch.einsum("bnij,bnj->bi", wrench_map, forces)
        residual = achieved - target_wrench
        gradient = torch.einsum("bnij,bi->bnj", wrench_map, residual)
        forces = project_coulomb_cone(
            forces - step_size * (gradient + regularization * forces),
            normals,
            friction,
            normal_force_limit,
        )
    return forces


def relative_residual(
    wrench_map: torch.Tensor,
    forces: torch.Tensor,
    target_wrench: torch.Tensor,
) -> torch.Tensor:
    achieved = torch.einsum("bnij,bnj->bi", wrench_map, forces)
    error = torch.linalg.vector_norm(achieved - target_wrench, dim=-1)
    target_norm = torch.linalg.vector_norm(target_wrench, dim=-1).clamp_min(1e-8)
    return (error / target_norm).mean()


def load_batch(
    data: np.lib.npyio.NpzFile,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frame_indices = np.arange(batch_size) % len(data["frame_index"])
    positions = torch.as_tensor(data["anchor_position_base"][frame_indices], device=device)
    normals = torch.as_tensor(data["anchor_normal_base"][frame_indices], device=device)
    target = torch.as_tensor(data["wrench_cmd_base"][frame_indices], device=device)
    normals = torch.nn.functional.normalize(normals.float(), dim=-1)
    return positions.float(), normals, target.float()


def benchmark_batch(
    data: np.lib.npyio.NpzFile,
    batch_size: int,
    device: torch.device,
    args: argparse.Namespace,
) -> BenchmarkResult:
    positions, normals, target = load_batch(data, batch_size, device)
    wrench_map = build_wrench_map(positions)
    for _ in range(args.warmup):
        solve_force_field(
            wrench_map, normals, target, args.iterations, args.friction,
            args.regularization, args.normal_force_limit,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    forces = None
    for _ in range(args.repeats):
        forces = solve_force_field(
            wrench_map, normals, target, args.iterations, args.friction,
            args.regularization, args.normal_force_limit,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    assert forces is not None
    latency_ms = elapsed * 1000.0 / args.repeats
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024.0**2)
        if device.type == "cuda" else math.nan
    )
    return BenchmarkResult(
        batch_size=batch_size,
        latency_ms=latency_ms,
        environments_per_second=batch_size * 1000.0 / latency_ms,
        peak_memory_mb=peak_memory_mb,
        relative_residual=float(relative_residual(wrench_map, forces, target).item()),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark batched GPU projected-gradient spatial force-field estimation."
    )
    parser.add_argument(
        "--data", type=Path,
        default=Path("force_command_estimator/data/box_gt_global_force_commands_full.npz"),
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 64, 256, 1024, 4096])
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--friction", type=float, default=0.5)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--normal-force-limit", type=float, default=100.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iterations <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise ValueError("iterations and repeats must be positive; warmup must be non-negative")
    if any(batch_size <= 0 for batch_size in args.batch_sizes):
        raise ValueError("batch sizes must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    with np.load(args.data, allow_pickle=False) as data:
        required = {
            "frame_index", "anchor_position_base", "anchor_normal_base", "wrench_cmd_base"
        }
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"dataset is missing arrays: {sorted(missing)}")
        print(f"device={device} frames={len(data['frame_index'])} anchors={data['anchor_position_base'].shape[1]}")
        print(
            f"iterations={args.iterations} warmup={args.warmup} repeats={args.repeats} "
            f"friction={args.friction:g}"
        )
        print("batch  latency_ms     env/s  peak_mem_mb  rel_residual")
        for batch_size in args.batch_sizes:
            result = benchmark_batch(data, batch_size, device, args)
            print(
                f"{result.batch_size:5d}  {result.latency_ms:10.3f}  "
                f"{result.environments_per_second:8.0f}  {result.peak_memory_mb:11.1f}  "
                f"{result.relative_residual:12.5f}"
            )


if __name__ == "__main__":
    main()
