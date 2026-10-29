from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from benchmark_gpu_force_field import build_wrench_map, solve_force_field


def build_field_data(
    commands_path: Path,
    retarget_path: Path,
    out_path: Path,
    method: str,
    radius: float = 0.15,
    active_top_k: int = 60,
    prefit_iterations: int = 8,
    solver_iterations: int = 32,
    friction: float = 0.5,
    regularization: float = 1e-4,
    normal_force_limit: float = 100.0,
    max_frames: int | None = None,
) -> dict[str, float]:
    if method != "hard15" or not np.isclose(radius, 0.15):
        raise ValueError("This release supports only method='hard15' with radius=0.15m.")
    with np.load(commands_path, allow_pickle=False) as data:
        frame_index = np.asarray(data["frame_index"], dtype=np.int64)
        positions = np.asarray(data["anchor_position_base"], dtype=np.float32)
        normals = np.asarray(data["anchor_normal_base"], dtype=np.float32)
        target = np.asarray(data["generalized_cmd_base"], dtype=np.float32)
        part_ids = np.asarray(data["part_ids"], dtype=np.int64)
    if max_frames is not None:
        frame_index = frame_index[:max_frames]
        positions = positions[:max_frames]
        normals = normals[:max_frames]
        target = target[:max_frames]
    retarget = torch.load(retarget_path, map_location="cpu", weights_only=False)
    demo = retarget["demo_data"]
    hand_tracks = retarget["retarget_data"]
    obj_pos = np.asarray(demo["obj_pos"], dtype=np.float32)
    obj_quat = np.asarray(demo["obj_quat"], dtype=np.float32)  # wxyz, as consumed by Genesis
    hand_points = np.concatenate(
        [np.asarray(hand_tracks[side]["kpt_pos"], dtype=np.float32) for side in ("left", "right")],
        axis=1,
    )
    if frame_index.min() < 0 or frame_index.max() >= len(obj_pos) or len(obj_pos) != len(hand_points):
        raise ValueError("Command frame indices and retarget hand/object trajectories do not align")

    # Match dexmachina.envs.math_utils.matrix_from_quat (w, x, y, z).
    quat = obj_quat[frame_index]
    w, x, y, z = np.moveaxis(quat, -1, 0)
    rotations = np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ], axis=-1,
    ).reshape(-1, 3, 3)
    anchor_world = obj_pos[frame_index, None, :] + np.einsum("tij,taj->tai", rotations, positions)
    hands = hand_points[frame_index]
    # Chunk anchors to keep memory bounded; nearest distance is over the same retarget keypoints
    # used by the Stage2 runtime (rather than the sparse MANO visualization samples).
    distances = np.empty(positions.shape[:2], dtype=np.float32)
    for start in range(0, len(frame_index), 64):
        delta = anchor_world[start:start + 64, :, None, :] - hands[start:start + 64, None, :, :]
        distances[start:start + 64] = np.sqrt(np.sum(delta * delta, axis=-1)).min(axis=-1)

    device = torch.device("cpu")
    pos_t = torch.from_numpy(positions)
    norm_t = torch.nn.functional.normalize(torch.from_numpy(normals), dim=-1)
    target_t = torch.from_numpy(target[:, :6])
    wrench_map = build_wrench_map(pos_t)
    distance_t = torch.from_numpy(distances)
    force_rows = []
    active_counts = []
    residuals = []
    for i in range(len(frame_index)):
        pos_i = pos_t[i:i + 1]
        normal_i = norm_t[i:i + 1]
        target_i = target_t[i:i + 1]
        map_i = wrench_map[i:i + 1]
        dist_i = distance_t[i]
        if method == "hard15":
            candidate = dist_i <= radius
            if not bool(candidate.any()):
                candidate = torch.ones_like(candidate)
            prefit = solve_force_field(
                map_i, normal_i * candidate[None, :, None], target_i,
                prefit_iterations, friction, regularization, normal_force_limit,
            )
            scores = prefit.norm(dim=-1)[0].masked_fill(~candidate, -torch.inf)
            selected = torch.zeros_like(candidate)
            quota = min(active_top_k, int(candidate.sum())) if active_top_k > 0 else int(candidate.sum())
            selected[scores.topk(quota).indices] = True
            force = solve_force_field(
                map_i, normal_i * selected[None, :, None], target_i,
                solver_iterations, friction, regularization, normal_force_limit,
            )
        row = force[0].numpy()
        force_rows.append(row)
        active_counts.append(int(np.count_nonzero(np.linalg.norm(row, axis=-1) > 1e-4)))
        achieved = torch.einsum("bnij,bnj->bi", map_i, force)[0]
        residuals.append(float(torch.linalg.vector_norm(achieved - target_i[0]) /
                               torch.linalg.vector_norm(target_i[0]).clamp_min(1e-8)))

    force_array = np.asarray(force_rows, dtype=np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        frame_index=frame_index.astype(np.int32),
        anchor_position_base=positions,
        anchor_normal_base=normals,
        anchor_force_base=force_array,
        generalized_cmd_base=target,
        part_ids=part_ids.astype(np.int32),
        hand_distance=distances,
        active_counts=np.asarray(active_counts, dtype=np.int32),
        relative_wrench_residual=np.asarray(residuals, dtype=np.float32),
        method=np.asarray(method), radius=np.asarray(radius),
        active_top_k=np.asarray(active_top_k),
        prefit_iterations=np.asarray(prefit_iterations), solver_iterations=np.asarray(solver_iterations),
        friction=np.asarray(friction), regularization=np.asarray(regularization),
        normal_force_limit=np.asarray(normal_force_limit),
        retarget_name=np.asarray(Path(retarget_path).stem.rsplit("_vector_", 1)[-1]),
        distance_source=np.asarray("retarget_data.left/right.kpt_pos; nearest of current Stage2 hand keypoints"),
        source_commands=np.asarray(str(commands_path)), source_retarget=np.asarray(str(retarget_path)),
    )
    return {
        "frames": float(len(frame_index)),
        "mean_candidates_20cm": float(np.mean(np.sum(distances <= 0.20, axis=1))),
        "mean_candidates_within_radius": float(np.mean(np.sum(distances <= radius, axis=1))),
        "mean_active_anchors": float(np.mean(active_counts)),
        "mean_force_norm": float(np.mean(np.linalg.norm(force_array, axis=-1).sum(axis=-1))),
        "mean_relative_wrench_residual": float(np.mean(residuals)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate method-matched force-field data for encoder training.")
    parser.add_argument("--commands", type=Path, required=True)
    parser.add_argument("--retarget", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--method", choices=("hard15",), default="hard15")
    parser.add_argument("--radius", type=float, default=0.15)
    parser.add_argument("--active-top-k", type=int, default=60)
    parser.add_argument("--prefit-iterations", type=int, default=8)
    parser.add_argument("--solver-iterations", type=int, default=32)
    parser.add_argument("--friction", type=float, default=0.5)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--normal-force-limit", type=float, default=100.0)
    parser.add_argument("--max-frames", type=int, default=0, help="Optional short CPU smoke-test limit; 0 uses the full sequence.")
    args = parser.parse_args()
    if not np.isclose(args.radius, 0.15) or args.active_top_k < 0:
        parser.error("this release requires radius=0.15; active-top-k must be non-negative")
    metrics = build_field_data(
        args.commands, args.retarget, args.out, args.method, args.radius,
        args.active_top_k, args.prefit_iterations, args.solver_iterations,
        args.friction, args.regularization, args.normal_force_limit,
        None if args.max_frames <= 0 else args.max_frames,
    )
    print(f"saved={args.out} method={args.method} metrics={metrics}", flush=True)


if __name__ == "__main__":
    main()
