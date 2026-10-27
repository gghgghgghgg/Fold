from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from field_encoder import FieldAutoencoder


class FieldDataset(Dataset):
    def __init__(
        self,
        positions: np.ndarray,
        normals: np.ndarray,
        forces: np.ndarray,
        wrench: np.ndarray,
        indices: np.ndarray,
        position_scale: float,
        force_scale: float,
        wrench_scale: np.ndarray,
    ):
        self.positions = positions
        self.normals = normals
        self.forces = forces
        self.wrench = wrench
        self.indices = indices
        self.position_scale = float(position_scale)
        self.force_scale = float(force_scale)
        self.wrench_scale = wrench_scale

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, ...]:
        index = int(self.indices[item])
        positions = self.positions[index] / self.position_scale
        forces = self.forces[index] / self.force_scale
        wrench = self.wrench[index] / self.wrench_scale
        active = np.linalg.norm(self.forces[index], axis=1) > max(self.force_scale * 1e-4, 1e-8)
        return (
            torch.from_numpy(positions.astype(np.float32)),
            torch.from_numpy(self.normals[index].astype(np.float32)),
            torch.from_numpy(forces.astype(np.float32)),
            torch.from_numpy(active),
            torch.from_numpy(wrench.astype(np.float32)),
            torch.tensor(index, dtype=torch.long),
        )


def field_loss(
    force_pred: torch.Tensor,
    force_target: torch.Tensor,
    wrench_pred: torch.Tensor,
    wrench_target: torch.Tensor,
    active: torch.Tensor,
    wrench_weight: float,
    direction_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    per_anchor = F.smooth_l1_loss(force_pred, force_target, beta=0.1, reduction="none").mean(dim=-1)
    anchor_weight = 0.2 + 0.8 * active.to(force_pred.dtype)
    force_term = (per_anchor * anchor_weight).sum() / anchor_weight.sum().clamp_min(1.0)
    wrench_term = F.smooth_l1_loss(wrench_pred, wrench_target, beta=0.1)

    valid = active & (torch.linalg.vector_norm(force_target, dim=-1) > 1e-5)
    cosine = F.cosine_similarity(force_pred, force_target, dim=-1, eps=1e-8)
    direction_term = (1.0 - cosine[valid]).mean() if valid.any() else force_pred.new_zeros(())
    total = force_term + float(wrench_weight) * wrench_term + float(direction_weight) * direction_term
    return total, {
        "force": force_term.detach(),
        "wrench": wrench_term.detach(),
        "direction": direction_term.detach(),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    part_ids: torch.Tensor,
    device: torch.device,
    force_scale: float,
    wrench_scale: np.ndarray,
    top_k: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    model.eval()
    force_predictions = []
    force_targets = []
    wrench_predictions = []
    wrench_targets = []
    frame_indices = []
    with torch.no_grad():
        for positions, normals, forces, active, wrench, index in loader:
            positions = positions.to(device)
            normals = normals.to(device)
            forces_device = forces.to(device)
            active_device = active.to(device)
            force_pred, wrench_pred, _latent = model(
                positions, normals, forces_device, part_ids, active_device
            )
            force_predictions.append(force_pred.cpu().numpy() * force_scale)
            force_targets.append(forces.numpy() * force_scale)
            wrench_predictions.append(wrench_pred.cpu().numpy() * wrench_scale)
            wrench_targets.append(wrench.numpy() * wrench_scale)
            frame_indices.append(index.numpy())

    force_pred = np.concatenate(force_predictions)
    force_gt = np.concatenate(force_targets)
    wrench_pred = np.concatenate(wrench_predictions)
    wrench_gt = np.concatenate(wrench_targets)
    frames = np.concatenate(frame_indices)
    force_error = force_pred - force_gt
    force_norm = np.linalg.norm(force_gt, axis=-1)
    pred_norm = np.linalg.norm(force_pred, axis=-1)
    active = force_norm > max(force_scale * 1e-4, 1e-8)
    cosine = np.sum(force_pred * force_gt, axis=-1) / np.maximum(pred_norm * force_norm, 1e-8)

    overlaps = []
    k = min(int(top_k), force_gt.shape[1])
    for predicted_magnitude, target_magnitude in zip(pred_norm, force_norm):
        pred_top = set(np.argpartition(predicted_magnitude, -k)[-k:].tolist())
        target_top = set(np.argpartition(target_magnitude, -k)[-k:].tolist())
        overlaps.append(len(pred_top & target_top) / max(k, 1))

    wrench_error = wrench_pred - wrench_gt
    metrics = {
        "field_vector_rmse_all_n": float(np.sqrt(np.mean(np.sum(force_error * force_error, axis=-1)))),
        "field_vector_rmse_active_n": float(
            np.sqrt(np.mean(np.sum(force_error[active] * force_error[active], axis=-1)))
        )
        if np.any(active)
        else float("nan"),
        "field_direction_cosine_active": float(np.mean(cosine[active])) if np.any(active) else float("nan"),
        "top_k_overlap": float(np.mean(overlaps)),
        "force_cmd_vector_rmse_n": float(
            np.sqrt(np.mean(np.sum(wrench_error[:, :3] * wrench_error[:, :3], axis=-1)))
        ),
        "torque_cmd_vector_rmse_nm": float(
            np.sqrt(np.mean(np.sum(wrench_error[:, 3:6] * wrench_error[:, 3:6], axis=-1)))
        ),
        "hinge_tau_mae_nm": float(np.mean(np.abs(wrench_error[:, 6]))),
        "wrench_component_mae": float(np.mean(np.abs(wrench_error))),
        "active_fraction": float(np.mean(active)),
    }
    arrays = {
        "frame_index": frames,
        "force_pred": force_pred,
        "force_gt": force_gt,
        "wrench_pred": wrench_pred,
        "wrench_gt": wrench_gt,
    }
    return metrics, arrays


def permutation_error(
    model: nn.Module,
    dataset: FieldDataset,
    part_ids: torch.Tensor,
    device: torch.device,
    batch_size: int = 16,
) -> float:
    count = min(batch_size, len(dataset))
    samples = [dataset[i] for i in range(count)]
    positions = torch.stack([x[0] for x in samples]).to(device)
    normals = torch.stack([x[1] for x in samples]).to(device)
    forces = torch.stack([x[2] for x in samples]).to(device)
    active = torch.stack([x[3] for x in samples]).to(device)
    permutation = torch.randperm(positions.shape[1], device=device)
    model.eval()
    with torch.no_grad():
        _fp, _wp, latent = model(positions, normals, forces, part_ids, active)
        _fp2, _wp2, latent_permuted = model(
            positions[:, permutation],
            normals[:, permutation],
            forces[:, permutation],
            part_ids[permutation],
            active[:, permutation],
        )
    return float(torch.max(torch.abs(latent - latent_permuted)).item())


def benchmark(
    model: nn.Module,
    dataset: FieldDataset,
    part_ids: torch.Tensor,
    device: torch.device,
    batch_sizes: list[int],
) -> dict[str, float]:
    sample = dataset[0]
    results = {}
    model.eval()
    for batch_size in batch_sizes:
        positions = sample[0].unsqueeze(0).expand(batch_size, -1, -1).to(device)
        normals = sample[1].unsqueeze(0).expand(batch_size, -1, -1).to(device)
        forces = sample[2].unsqueeze(0).expand(batch_size, -1, -1).to(device)
        active = sample[3].unsqueeze(0).expand(batch_size, -1).to(device)
        try:
            with torch.no_grad():
                for _ in range(5):
                    model.encoder(positions, normals, forces, part_ids, active)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                repeats = 30 if batch_size <= 64 else 10
                for _ in range(repeats):
                    model.encoder(positions, normals, forces, part_ids, active)
                if device.type == "cuda":
                    torch.cuda.synchronize()
            results[str(batch_size)] = (time.perf_counter() - start) * 1000.0 / repeats
        except torch.cuda.OutOfMemoryError:
            results[str(batch_size)] = float("nan")
            torch.cuda.empty_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the PointNet global-force-field encoder.")
    parser.add_argument("--data", default="force_command_estimator/data/box_gt_global_force_commands_full.npz")
    parser.add_argument("--out-dir", default="force_command_estimator/results/field_encoder/box_pointnet")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wrench-weight", type=float, default=1.0)
    parser.add_argument("--direction-weight", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip the inference benchmark (including large batch sizes) after training.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device
        if args.device != "auto"
        else "cpu"
    )

    data = np.load(args.data, allow_pickle=False)
    required = (
        "anchor_position_base",
        "anchor_normal_base",
        "anchor_force_base",
        "generalized_cmd_base",
        "part_ids",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"dataset is missing {missing}; regenerate commands with generate_gt_commands.py")
    positions = np.asarray(data["anchor_position_base"], dtype=np.float32)
    normals = np.asarray(data["anchor_normal_base"], dtype=np.float32)
    forces = np.asarray(data["anchor_force_base"], dtype=np.float32)
    wrench = np.asarray(data["generalized_cmd_base"], dtype=np.float32)
    part_ids_np = np.asarray(data["part_ids"], dtype=np.int64)
    num_parts = int(np.max(part_ids_np)) + 1
    length = len(forces)
    train_end = int(0.70 * length)
    val_end = int(0.85 * length)
    indices = {
        "train": np.arange(0, train_end, dtype=np.int64),
        "val": np.arange(train_end, val_end, dtype=np.int64),
        "test": np.arange(val_end, length, dtype=np.int64),
    }
    if min(len(x) for x in indices.values()) == 0:
        raise ValueError(f"not enough frames for chronological split: {length}")

    position_scale = float(np.percentile(np.linalg.norm(positions[:train_end], axis=-1), 95))
    force_scale = float(np.percentile(np.linalg.norm(forces[:train_end], axis=-1), 95))
    position_scale = max(position_scale, 1e-3)
    force_scale = max(force_scale, 1.0)
    wrench_scale = np.percentile(np.abs(wrench[:train_end]), 95, axis=0).astype(np.float32)
    wrench_scale = np.maximum(wrench_scale, 1e-3)
    datasets = {
        name: FieldDataset(
            positions,
            normals,
            forces,
            wrench,
            split,
            position_scale,
            force_scale,
            wrench_scale,
        )
        for name, split in indices.items()
    }
    loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=name == "train", num_workers=0)
        for name, ds in datasets.items()
    }

    model = FieldAutoencoder(num_parts=num_parts, latent_dim=128).to(device)
    part_ids = torch.from_numpy(part_ids_np).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    best_val = float("inf")
    best_state = None
    loss_log = {"train": [], "val": []}
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0.0
        for positions_b, normals_b, forces_b, active_b, wrench_b, _index in loaders["train"]:
            positions_b = positions_b.to(device)
            normals_b = normals_b.to(device)
            forces_b = forces_b.to(device)
            active_b = active_b.to(device)
            wrench_b = wrench_b.to(device)
            optimizer.zero_grad(set_to_none=True)
            force_pred, wrench_pred, _latent = model(
                positions_b, normals_b, forces_b, part_ids, active_b
            )
            loss, _terms = field_loss(
                force_pred,
                forces_b,
                wrench_pred,
                wrench_b,
                active_b,
                args.wrench_weight,
                args.direction_weight,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_total += float(loss.item()) * len(positions_b)
        train_loss = train_total / len(datasets["train"])

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for positions_b, normals_b, forces_b, active_b, wrench_b, _index in loaders["val"]:
                positions_b = positions_b.to(device)
                normals_b = normals_b.to(device)
                forces_b = forces_b.to(device)
                active_b = active_b.to(device)
                wrench_b = wrench_b.to(device)
                force_pred, wrench_pred, _latent = model(
                    positions_b, normals_b, forces_b, part_ids, active_b
                )
                loss, _terms = field_loss(
                    force_pred,
                    forces_b,
                    wrench_pred,
                    wrench_b,
                    active_b,
                    args.wrench_weight,
                    args.direction_weight,
                )
                val_total += float(loss.item()) * len(positions_b)
        val_loss = val_total / len(datasets["val"])
        loss_log["train"].append(train_loss)
        loss_log["val"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            print(f"epoch={epoch:04d} train={train_loss:.6f} val={val_loss:.6f}")

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    metrics, arrays = evaluate(
        model,
        loaders["test"],
        part_ids,
        device,
        force_scale,
        wrench_scale,
        args.top_k,
    )
    metrics["permutation_latent_max_abs"] = permutation_error(
        model, datasets["test"], part_ids, device
    )
    metrics["encoder_time_ms"] = (
        {}
        if args.skip_benchmark
        else benchmark(model, datasets["test"], part_ids, device, [1, 64, 1024, 4096])
    )
    metrics["best_val_loss"] = float(best_val)
    metrics["split_samples"] = {name: len(split) for name, split in indices.items()}
    metrics["position_scale"] = position_scale
    metrics["force_scale"] = force_scale
    metrics["wrench_scale"] = wrench_scale.tolist()
    metrics["device"] = str(device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "num_parts": num_parts,
            "latent_dim": 128,
            "position_scale": position_scale,
            "force_scale": force_scale,
            "wrench_scale": wrench_scale,
        },
        out_dir / "best_field_encoder.pt",
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / "test_predictions.npz", **arrays)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    frames = arrays["frame_index"]
    force_gt_plot = arrays["wrench_gt"][:, :3]
    force_pred_plot = arrays["wrench_pred"][:, :3]
    for axis_index, axis_name in enumerate(("Fx", "Fy", "Fz")):
        axes[axis_index].plot(frames, force_gt_plot[:, axis_index], label="GT", linewidth=1.5)
        axes[axis_index].plot(frames, force_pred_plot[:, axis_index], label="pred", linewidth=1.0)
        axes[axis_index].set_ylabel(f"{axis_name} [N]")
        axes[axis_index].grid(alpha=0.25)
    axes[0].legend()
    axes[-1].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(out_dir / "wrench_reconstruction.png", dpi=180)
    plt.close(fig)

    print("\n--- PointNet Field Encoder Test ---")
    print(f"device                   : {device}")
    print(f"frames                   : {length} ({len(indices['test'])} test)")
    print(f"field active RMSE        : {metrics['field_vector_rmse_active_n']:.4f} N")
    print(f"field active cosine      : {metrics['field_direction_cosine_active']:.4f}")
    print(f"top-{args.top_k} overlap          : {metrics['top_k_overlap']:.4f}")
    print(f"force wrench RMSE        : {metrics['force_cmd_vector_rmse_n']:.4f} N")
    print(f"torque wrench RMSE       : {metrics['torque_cmd_vector_rmse_nm']:.4f} Nm")
    print(f"hinge tau MAE            : {metrics['hinge_tau_mae_nm']:.4f} Nm")
    print(f"permutation latent error : {metrics['permutation_latent_max_abs']:.3e}")
    print(f"encoder times [ms]       : {metrics['encoder_time_ms']}")
    print(f"Saved                    : {out_dir}")


if __name__ == "__main__":
    main()
