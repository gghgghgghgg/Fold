from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from force_command_estimator.benchmark_gpu_force_field import build_wrench_map, solve_force_field
from force_command_estimator.field_encoder import FieldAutoencoder
from stage2.realtime_wrench import RealtimeWrenchCommand, RealtimeWrenchConfig


def _build_frame_to_row(
    frame_index: np.ndarray,
    frame_offset: int,
    clip_length: int,
) -> tuple[int, np.ndarray]:
    """Build a dense lookup for a contiguous command clip with absolute frame ids."""
    frame_index = np.asarray(frame_index, dtype=np.int64)
    if frame_index.ndim != 1 or frame_index.size == 0:
        raise ValueError("Force-field data must contain at least one frame index.")
    if clip_length <= 0:
        raise ValueError(f"Demo clip length must be positive; got {clip_length}.")

    frame_min = int(frame_index.min())
    frame_max = int(frame_index.max())
    expected = np.arange(frame_min, frame_max + 1, dtype=np.int64)
    if frame_index.size != expected.size or not np.array_equal(np.sort(frame_index), expected):
        raise ValueError(
            "Force-field frame indices must be unique and contiguous; "
            f"got {frame_index.size} rows spanning [{frame_min}, {frame_max}]."
        )

    requested_start = int(frame_offset)
    requested_end = requested_start + int(clip_length) - 1
    if requested_start < frame_min or requested_end > frame_max:
        raise ValueError(
            "Force-field data does not cover the requested demo clip: "
            f"requested [{requested_start}, {requested_end}], "
            f"available [{frame_min}, {frame_max}]."
        )

    frame_to_row = np.full(expected.size, -1, dtype=np.int64)
    frame_to_row[frame_index - frame_min] = np.arange(frame_index.size, dtype=np.int64)
    return frame_min, frame_to_row


@dataclass(frozen=True)
class RealtimeForceFieldConfig:
    data_path: str
    encoder_checkpoint: str
    frame_offset: int = 0
    prefit_iterations: int = 8
    active_top_k: int = 60
    iterations: int = 32
    friction: float = 0.5
    regularization: float = 1e-4
    normal_force_limit: float = 100.0
    wrench: RealtimeWrenchConfig = RealtimeWrenchConfig()


class RealtimeForceFieldCommand:
    """Solve and encode a batched spatial field from a realtime feedback wrench."""

    def __init__(self, env, config: RealtimeForceFieldConfig):
        self.env = env
        self.config = config
        self.wrench_command = RealtimeWrenchCommand(env, config.wrench)

        with np.load(config.data_path, allow_pickle=False) as data:
            frame_index = np.asarray(data["frame_index"], dtype=np.int64)
            positions = np.asarray(data["anchor_position_base"], dtype=np.float32)
            normals = np.asarray(data["anchor_normal_base"], dtype=np.float32)
            part_ids = np.asarray(data["part_ids"], dtype=np.int64)
        if (
            positions.ndim != 3
            or positions.shape != normals.shape
            or positions.shape[1:] != (len(part_ids), 3)
            or positions.shape[0] != frame_index.size
        ):
            raise ValueError("Force-field anchor position, normal, and part shapes do not match.")

        demo_positions = self.env.reward_module.demo_tensors["obj_pos"]
        self.frame_min, frame_to_row = _build_frame_to_row(
            frame_index,
            config.frame_offset,
            int(demo_positions.shape[0]),
        )

        self.positions = torch.from_numpy(positions).to(env.device)
        self.normals = torch.nn.functional.normalize(torch.from_numpy(normals).to(env.device), dim=-1)
        self.part_ids = torch.from_numpy(part_ids).to(env.device)
        self.frame_to_row = torch.from_numpy(frame_to_row).to(env.device)

        checkpoint = torch.load(config.encoder_checkpoint, map_location="cpu", weights_only=False)
        model = FieldAutoencoder(
            num_parts=int(checkpoint["num_parts"]),
            latent_dim=int(checkpoint["latent_dim"]),
        )
        model.load_state_dict(checkpoint["model"])
        self.encoder = model.encoder.to(env.device).eval().requires_grad_(False)
        self.position_scale = float(checkpoint["position_scale"])
        self.force_scale = float(checkpoint["force_scale"])
        self.latent_dim = int(checkpoint["latent_dim"])

    @torch.inference_mode()
    def compute(self) -> torch.Tensor:
        reference_frame = self.env.episode_length_buf.long() + int(self.config.frame_offset)
        reference_row = reference_frame - self.frame_min
        reference_row = reference_row.clamp(0, self.frame_to_row.shape[0] - 1)
        rows = self.frame_to_row[reference_row]
        positions = self.positions[rows]
        normals = self.normals[rows]

        target_wrench = self.wrench_command.compute_physical()
        limits = target_wrench.new_tensor(
            [self.config.wrench.force_scale] * 3 + [self.config.wrench.torque_scale] * 3
        ) * self.config.wrench.command_clip
        target_wrench = target_wrench.clamp(-limits, limits)
        wrench_map = build_wrench_map(positions)
        solve_normals = normals
        use_top_k = 0 < self.config.active_top_k < normals.shape[1]
        if use_top_k:
            prefit_forces = solve_force_field(
                wrench_map,
                normals,
                target_wrench,
                self.config.prefit_iterations,
                self.config.friction,
                self.config.regularization,
                self.config.normal_force_limit,
            )
            scores = prefit_forces.norm(dim=-1)
            selected = scores.topk(self.config.active_top_k, dim=-1).indices
            selected_mask = torch.zeros_like(normals[..., 0], dtype=torch.bool)
            selected_mask.scatter_(1, selected, True)
            solve_normals = normals * selected_mask.unsqueeze(-1)
        forces = solve_force_field(
            wrench_map,
            solve_normals,
            target_wrench,
            self.config.iterations,
            self.config.friction,
            self.config.regularization,
            self.config.normal_force_limit,
        )
        active = torch.linalg.vector_norm(forces, dim=-1) > 1e-4
        latent = self.encoder(
            positions / self.position_scale,
            normals,
            forces / self.force_scale,
            self.part_ids,
            active,
        )
        return torch.nan_to_num(latent)


def install_realtime_force_field_observation(env, config: RealtimeForceFieldConfig) -> None:
    """Append a 128D realtime field latent after the frozen base observation prefix."""
    original_get_observations = env.get_observations
    command = RealtimeForceFieldCommand(env, config)
    env.obs_dim = int(env.obs_dim) + command.latent_dim
    env.num_obs = env.obs_dim
    env.obs_buf = torch.zeros((env.num_envs, env.obs_dim), device=env.device)
    env.obs_sum = torch.zeros_like(env.obs_buf)
    env.realtime_force_field_command = command

    def get_observations_with_force_field():
        observations = original_get_observations()
        field_latent = command.compute()
        if isinstance(observations, dict):
            observations["policy"] = torch.cat([observations["policy"], field_latent], dim=-1)
            observations["critic"] = torch.cat([observations["critic"], field_latent], dim=-1)
            observations["itemized"]["realtime_force_field_cmd"] = field_latent
            return observations
        return torch.cat([observations, field_latent], dim=-1)

    env.get_observations = get_observations_with_force_field
