from __future__ import annotations

from dataclasses import dataclass

import torch

from fold.envs.math_utils import matrix_from_quat, quat_conjugate, quat_mul


@dataclass(frozen=True)
class RealtimeWrenchConfig:
    position_kp: float = 200.0
    position_kd: float = 20.0
    rotation_kp: float = 20.0
    rotation_kd: float = 2.0
    force_scale: float = 25.0
    torque_scale: float = 3.0
    command_clip: float = 5.0


def _rotation_vector(current_quat: torch.Tensor, target_quat: torch.Tensor) -> torch.Tensor:
    relative = quat_mul(target_quat, quat_conjugate(current_quat))
    relative = relative / torch.linalg.vector_norm(relative, dim=-1, keepdim=True).clamp_min(1e-8)
    relative = torch.where(relative[:, :1] < 0.0, -relative, relative)
    vector = relative[:, 1:]
    vector_norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vector_norm, relative[:, :1].clamp_min(1e-8))
    return vector * (angle / vector_norm.clamp_min(1e-8))


class RealtimeWrenchCommand:
    """Compute a batched object-local feedback wrench from the next demo state."""

    def __init__(self, env, config: RealtimeWrenchConfig):
        if len(env.object_names) != 1:
            raise ValueError("Realtime wrench command currently requires exactly one object.")
        self.env = env
        self.config = config
        self.object = env.objects[env.object_names[0]]
        self.demo_pos = env.reward_module.demo_tensors["obj_pos"]
        self.demo_quat = env.reward_module.demo_tensors["obj_quat"]
        self.demo_length = int(self.demo_pos.shape[0])
        self.dt = float(env.dt)

    def compute_physical(self) -> torch.Tensor:
        frame = self.env.episode_length_buf.long().clamp(0, self.demo_length - 1)
        next_frame = (frame + 1).clamp_max(self.demo_length - 1)
        target_pos = self.demo_pos[next_frame]
        target_quat = self.demo_quat[next_frame]
        desired_lin_vel = (target_pos - self.demo_pos[frame]) / self.dt
        desired_ang_vel = _rotation_vector(self.demo_quat[frame], target_quat) / self.dt

        current_quat = self.object.root_quat
        world_to_object = matrix_from_quat(current_quat).transpose(1, 2)
        force_world = (
            self.config.position_kp * (target_pos - self.object.root_pos)
            + self.config.position_kd * (desired_lin_vel - self.object.root_lin_vel)
        )
        torque_world = (
            self.config.rotation_kp * _rotation_vector(current_quat, target_quat)
            + self.config.rotation_kd * (desired_ang_vel - self.object.root_ang_vel)
        )
        force_object = torch.einsum("bij,bj->bi", world_to_object, force_world)
        torque_object = torch.einsum("bij,bj->bi", world_to_object, torque_world)
        return torch.nan_to_num(torch.cat([force_object, torque_object], dim=-1))

    def compute(self) -> torch.Tensor:
        wrench = self.compute_physical()
        scale = wrench.new_tensor(
            [self.config.force_scale] * 3 + [self.config.torque_scale] * 3
        )
        return (wrench / scale).clamp(-self.config.command_clip, self.config.command_clip)


def install_realtime_wrench_observation(env, config: RealtimeWrenchConfig) -> None:
    """Append a 6D command while preserving the frozen base-policy observation prefix."""
    original_get_observations = env.get_observations
    command = RealtimeWrenchCommand(env, config)
    env.obs_dim = int(env.obs_dim) + 6
    env.num_obs = env.obs_dim
    env.obs_buf = torch.zeros((env.num_envs, env.obs_dim), device=env.device)
    env.obs_sum = torch.zeros_like(env.obs_buf)
    env.realtime_wrench_command = command

    def get_observations_with_wrench():
        observations = original_get_observations()
        wrench = command.compute()
        if isinstance(observations, dict):
            observations["policy"] = torch.cat([observations["policy"], wrench], dim=-1)
            observations["critic"] = torch.cat([observations["critic"], wrench], dim=-1)
            observations["itemized"]["realtime_wrench_cmd"] = wrench
            return observations
        return torch.cat([observations, wrench], dim=-1)

    env.get_observations = get_observations_with_wrench
