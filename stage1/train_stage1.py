from __future__ import annotations

import argparse
import math
import os
import pickle
import time

import genesis as gs
import numpy as np
import torch
import wandb
import yaml
from rl_games.algos_torch.model_builder import register_network
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from fold.asset_utils import get_rl_config_path
from fold.envs.base_env import BaseEnv
from fold.envs.contacts import get_filtered_contacts
from fold.envs.constructors import get_all_env_cfg, get_common_argparser, parse_clip_string
from fold.envs.math_utils import matrix_from_quat
from fold.envs.reward_utils import rotation_distance
from fold.envs.rewards import RewardModule
from fold.rl.rl_games_wrapper import RlGamesGpuEnv, RlGamesVecEnvWrapper
from stage1.network_builder import Stage1SemanticBuilder
from stage1.obs_spec import build_stage1_obs_spec


def dump_yaml(filename: str, data: dict | object, sort_keys: bool = False):
    if not filename.endswith("yaml"):
        filename += ".yaml"
    parent = os.path.dirname(filename)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    def to_list(obj):
        if isinstance(obj, dict):
            return {k: to_list(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_list(v) for v in obj]
        if isinstance(obj, tuple):
            return [to_list(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(filename, "w", encoding="utf-8") as handle:
        yaml.dump(to_list(data), handle, default_flow_style=False, sort_keys=sort_keys)


def build_parser() -> argparse.ArgumentParser:
    parser = get_common_argparser()
    parser.add_argument(
        "--imi_vel_rew_weight",
        "-imv",
        type=float,
        default=0.1,
        help="Extra stage1 imitation reward weight for matching demo velocities.",
    )
    parser.add_argument(
        "--overlap_geom_penalty",
        type=float,
        default=20.0,
        help="Penalty weight for hand keypoints getting too close to the GT-driven object during stage1 training.",
    )
    parser.add_argument(
        "--overlap_geom_threshold",
        type=float,
        default=0.12,
        help="Distance margin in meters below which the geometric anti-overlap penalty activates.",
    )
    parser.add_argument(
        "--overlap_inside_backend",
        choices=["trimesh", "fwn", "compare", "grid", "grid_compare"],
        default="trimesh",
        help="Backend used for object inside tests in the geometric anti-overlap penalty.",
    )
    parser.add_argument(
        "--overlap_grid_pitch",
        type=float,
        default=0.002,
        help="Voxel pitch in meters for the cached winding grid backend.",
    )
    parser.add_argument(
        "--disable_gt_object_train",
        action="store_true",
        help="Disable forcing the object back to the GT trajectory during stage1 training.",
    )
    parser.add_argument("--exp_name", "-exp", type=str, default="stage1", help="Experiment name.")
    parser.add_argument("--horizon", "-ho", type=int, default=16, help="Number of steps per environment.")
    parser.add_argument("--checkpoint", "-ck", type=str, default=None, help="Checkpoint file to load.")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.0003, help="Learning rate.")
    parser.add_argument("--wandb_project", "-wp", type=str, default="fold-stage1", help="WandB project.")
    parser.add_argument("--save_freq", "-sf", type=int, default=1000)
    parser.add_argument(
        "--stage1_architecture",
        choices=["semantic", "maniptrans"],
        default="semantic",
        help="Actor-critic encoder: semantic field branches or the flat ManipTrans MLP.",
    )
    parser.add_argument(
        "--stage1_maniptrans_units",
        type=int,
        nargs="+",
        default=[512, 512, 256, 128],
        help="Hidden sizes for the flat ManipTrans imitator MLP.",
    )
    parser.add_argument(
        "--stage1_field_units",
        type=int,
        nargs="+",
        default=[128, 64],
        help="Hidden sizes used by each semantic field encoder.",
    )
    parser.add_argument(
        "--stage1_trunk_units",
        type=int,
        nargs="+",
        default=[512, 256, 128],
        help="Hidden sizes used by the PPO trunk after semantic encoding.",
    )
    parser.add_argument(
        "--stage1_encoder_activation",
        type=str,
        default="elu",
        choices=["elu", "relu", "tanh", "silu", "identity"],
        help="Activation used by semantic field encoders.",
    )
    parser.add_argument(
        "--stage1_disable_layer_norm",
        action="store_true",
        help="Disable LayerNorm inside semantic field encoders.",
    )
    return parser


def make_stage1_exp_name(args) -> str:
    obj_name, start, end, subject_name, use_clip = parse_clip_string(args.clip)
    hand_prefix = str(args.hand).split("_")[0]
    exp_name = f"{hand_prefix}-{args.exp_name}"
    exp_name += f"_{obj_name}{start}-{end}-{subject_name}-{use_clip}_B{args.num_envs}"
    exp_name += f"_{args.action_mode}"
    if args.stage1_architecture != "semantic":
        exp_name += f"_{args.stage1_architecture}arch"
    exp_name += f"_ho{args.horizon}"
    exp_name += f"_imi{args.imi_rew_weight}"
    if args.imi_vel_rew_weight > 0:
        exp_name += f"_imv{args.imi_vel_rew_weight}"
    if args.bc_rew_weight > 0:
        exp_name += f"_bc{args.bc_rew_weight}"
    if args.contact_rew_weight > 0:
        exp_name += f"_con{args.contact_rew_weight}"
    if not args.disable_gt_object_train:
        exp_name += "_gtobj"
    if args.overlap_geom_penalty > 0:
        exp_name += f"_golap{args.overlap_geom_penalty}_gth{args.overlap_geom_threshold}"
    exp_name += "_stage1semantic"
    return exp_name


def disable_curriculum_for_stage1(env_kwargs: dict):
    env_kwargs["curriculum_cfg"] = {}
    for obj_cfg in env_kwargs["object_cfgs"].values():
        obj_cfg["actuated"] = False
        obj_cfg["kp"] = 0.0
        obj_cfg["kv"] = 0.0
        obj_cfg["force_range"] = 0.0

    reward_cfg = env_kwargs["reward_cfg"]
    reward_cfg["task_rew_weight"] = 0.0
    reward_cfg["contact_rew_weight"] = 0.0
    reward_cfg["force_wrench_rew_weight"] = 0.0
    reward_cfg["force_anchor_rew_weight"] = 0.0

    env_cfg = env_kwargs["env_cfg"]
    env_cfg["observe_tip_dist"] = False
    env_cfg["observe_contact_force"] = False
    env_cfg["use_contact_reward"] = False
    env_cfg["early_reset_threshold"] = 0.0
    env_cfg["early_reset_aux_thres"] = dict(con=0.0, imi=0.0, bc=0.0)


def _install_stage1_velocity_reward_patch():
    if getattr(RewardModule, "_stage1_velocity_patch_installed", False):
        return

    original_load_demo = RewardModule.load_demo
    original_compute_imitation_reward = RewardModule.compute_imitation_reward

    def _finite_difference(values: torch.Tensor) -> torch.Tensor:
        diffs = torch.zeros_like(values)
        if values.shape[0] > 1:
            diffs[1:] = values[1:] - values[:-1]
        return diffs

    def _quat_step_distance(quats: torch.Tensor) -> torch.Tensor:
        diffs = torch.zeros(quats.shape[0], dtype=quats.dtype, device=quats.device)
        if quats.shape[0] > 1:
            diffs[1:] = rotation_distance(quats[1:], quats[:-1])
        return diffs

    def patched_load_demo(self, demo_data, retarget_data, device):
        original_load_demo(self, demo_data, retarget_data, device)
        self.imi_vel_rew_weight = float(self.cfg.get("imi_vel_rew_weight", 0.0))
        self.use_imi_vel_rew = self.imi_vel_rew_weight > 0.0
        self.prev_policy_kpts_left = None
        self.prev_policy_kpts_right = None
        self.prev_policy_wrist_left = None
        self.prev_policy_wrist_right = None
        if not self.use_imi_vel_rew:
            return

        self.demo_tensors["kpts_vel_left"] = _finite_difference(self.demo_tensors["kpts_left"])
        self.demo_tensors["kpts_vel_right"] = _finite_difference(self.demo_tensors["kpts_right"])
        self.demo_tensors["wrist_pos_vel_left"] = _finite_difference(self.demo_tensors["wrist_pose_left"][:, :3])
        self.demo_tensors["wrist_pos_vel_right"] = _finite_difference(self.demo_tensors["wrist_pose_right"][:, :3])
        self.demo_tensors["wrist_rot_vel_left"] = _quat_step_distance(self.demo_tensors["wrist_pose_left"][:, 3:])
        self.demo_tensors["wrist_rot_vel_right"] = _quat_step_distance(self.demo_tensors["wrist_pose_right"][:, 3:])

    def patched_compute_imitation_reward(self, wrist_pose_left, wrist_pose_right, kpts_left, kpts_right, episode_length_buf):
        imi_rew, rew_dict = original_compute_imitation_reward(
            self,
            wrist_pose_left,
            wrist_pose_right,
            kpts_left,
            kpts_right,
            episode_length_buf,
        )
        if not getattr(self, "use_imi_vel_rew", False):
            return imi_rew, rew_dict

        if self.prev_policy_kpts_left is None:
            self.prev_policy_kpts_left = kpts_left.clone()
            self.prev_policy_kpts_right = kpts_right.clone()
            self.prev_policy_wrist_left = wrist_pose_left.clone()
            self.prev_policy_wrist_right = wrist_pose_right.clone()

        reset_mask = (episode_length_buf <= 1).view(-1, 1)
        reset_mask_kpt = reset_mask.unsqueeze(-1)

        prev_kpts_left = torch.where(reset_mask_kpt, kpts_left, self.prev_policy_kpts_left)
        prev_kpts_right = torch.where(reset_mask_kpt, kpts_right, self.prev_policy_kpts_right)
        prev_wrist_left = torch.where(reset_mask.expand(-1, wrist_pose_left.shape[-1]), wrist_pose_left, self.prev_policy_wrist_left)
        prev_wrist_right = torch.where(reset_mask.expand(-1, wrist_pose_right.shape[-1]), wrist_pose_right, self.prev_policy_wrist_right)

        curr_kpt_vel_left = kpts_left - prev_kpts_left
        curr_kpt_vel_right = kpts_right - prev_kpts_right
        demo_kpt_vel_left = self.match_demo_state("kpts_vel_left", episode_length_buf)
        demo_kpt_vel_right = self.match_demo_state("kpts_vel_right", episode_length_buf)
        kpt_vel_dist_left = torch.norm(curr_kpt_vel_left - demo_kpt_vel_left, p=2, dim=-1)
        kpt_vel_dist_right = torch.norm(curr_kpt_vel_right - demo_kpt_vel_right, p=2, dim=-1)
        kpt_vel_dist = torch.mean((kpt_vel_dist_left + kpt_vel_dist_right) / 2.0, dim=-1)
        kpt_vel_beta = float(self.cfg.get("imi_vel_kpt_beta", 40.0))
        kpt_vel_rew = torch.exp(-kpt_vel_beta * kpt_vel_dist)

        curr_wrist_pos_vel_left = wrist_pose_left[:, :3] - prev_wrist_left[:, :3]
        curr_wrist_pos_vel_right = wrist_pose_right[:, :3] - prev_wrist_right[:, :3]
        demo_wrist_pos_vel_left = self.match_demo_state("wrist_pos_vel_left", episode_length_buf)
        demo_wrist_pos_vel_right = self.match_demo_state("wrist_pos_vel_right", episode_length_buf)
        wrist_pos_vel_dist_left = torch.norm(curr_wrist_pos_vel_left - demo_wrist_pos_vel_left, p=2, dim=-1)
        wrist_pos_vel_dist_right = torch.norm(curr_wrist_pos_vel_right - demo_wrist_pos_vel_right, p=2, dim=-1)
        wrist_pos_vel_beta = float(self.cfg.get("imi_vel_wrist_pos_beta", 40.0))
        wrist_pos_vel_rew_left = torch.exp(-wrist_pos_vel_beta * wrist_pos_vel_dist_left)
        wrist_pos_vel_rew_right = torch.exp(-wrist_pos_vel_beta * wrist_pos_vel_dist_right)

        curr_wrist_rot_vel_left = rotation_distance(wrist_pose_left[:, 3:], prev_wrist_left[:, 3:])
        curr_wrist_rot_vel_right = rotation_distance(wrist_pose_right[:, 3:], prev_wrist_right[:, 3:])
        demo_wrist_rot_vel_left = self.match_demo_state("wrist_rot_vel_left", episode_length_buf)
        demo_wrist_rot_vel_right = self.match_demo_state("wrist_rot_vel_right", episode_length_buf)
        wrist_rot_vel_dist_left = torch.abs(curr_wrist_rot_vel_left - demo_wrist_rot_vel_left)
        wrist_rot_vel_dist_right = torch.abs(curr_wrist_rot_vel_right - demo_wrist_rot_vel_right)
        wrist_rot_vel_beta = float(self.cfg.get("imi_vel_wrist_rot_beta", 4.0))
        wrist_rot_vel_rew_left = torch.exp(-wrist_rot_vel_beta * wrist_rot_vel_dist_left)
        wrist_rot_vel_rew_right = torch.exp(-wrist_rot_vel_beta * wrist_rot_vel_dist_right)

        wrist_vel_rew_left = 0.5 * (wrist_pos_vel_rew_left + wrist_rot_vel_rew_left)
        wrist_vel_rew_right = 0.5 * (wrist_pos_vel_rew_right + wrist_rot_vel_rew_right)
        wrist_vel_rew = 0.5 * (wrist_vel_rew_left + wrist_vel_rew_right)
        wrist_mix = float(self.cfg.get("imi_vel_wrist_weight", 0.5))
        imi_vel_rew = self.imi_vel_rew_weight * (wrist_mix * wrist_vel_rew + (1.0 - wrist_mix) * kpt_vel_rew)

        total_imi_rew = imi_rew + imi_vel_rew
        rew_dict["imi_pose_rew"] = imi_rew
        rew_dict["imi_vel_rew"] = imi_vel_rew
        rew_dict["imi_rew"] = total_imi_rew
        rew_dict["kpt_vel_dist"] = kpt_vel_dist
        rew_dict["wrist_vel_dist"] = 0.5 * (wrist_pos_vel_dist_left + wrist_pos_vel_dist_right)
        rew_dict["wrist_ang_vel_dist"] = 0.5 * (wrist_rot_vel_dist_left + wrist_rot_vel_dist_right)

        self.prev_policy_kpts_left = kpts_left.clone()
        self.prev_policy_kpts_right = kpts_right.clone()
        self.prev_policy_wrist_left = wrist_pose_left.clone()
        self.prev_policy_wrist_right = wrist_pose_right.clone()
        return total_imi_rew, rew_dict

    RewardModule.load_demo = patched_load_demo
    RewardModule.compute_imitation_reward = patched_compute_imitation_reward
    RewardModule._stage1_velocity_patch_installed = True


def _install_stage1_gt_object_overlap_patch():
    if getattr(BaseEnv, "_stage1_gt_object_patch_installed", False):
        return

    original_get_rewards = BaseEnv._get_rewards
    original_compute_intermediate_values = BaseEnv._compute_intermediate_values

    def _ensure_overlap_timing_state(self):
        if hasattr(self, "stage1_overlap_timing_enabled"):
            return
        self.stage1_overlap_timing_enabled = os.environ.get("DEX_STAGE1_OVERLAP_TIMING", "1").lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        self.stage1_overlap_timing_every = max(1, int(os.environ.get("DEX_STAGE1_OVERLAP_TIMING_EVERY", "50")))
        self.stage1_overlap_timing_calls = 0
        self.stage1_overlap_timing_contains_calls = 0

    def _should_log_overlap_timing(call_idx, every):
        return call_idx <= 8 or call_idx % every == 0

    def _ensure_overlap_geom_assets(self):
        _ensure_overlap_timing_state(self)
        if hasattr(self, "stage1_overlap_obj_verts") and hasattr(self, "stage1_overlap_meshes"):
            return
        asset_start = time.perf_counter()
        import trimesh

        obj = self.objects[self.object_names[0]]
        self.stage1_overlap_obj_verts = {
            part: obj.sample_mesh_vertices(300, part).to(self.device)
            for part in ["top", "bottom"]
        }
        self.stage1_overlap_meshes = {
            part: trimesh.load(obj.cfg[f"{part}_mesh_fname"])
            for part in ["top", "bottom"]
        }
        backend = getattr(self, "stage1_overlap_inside_backend", "trimesh")
        if backend in {"fwn", "compare", "grid", "grid_compare"}:
            import igl

            self.stage1_overlap_igl = igl
            self.stage1_overlap_fwn_meshes = {
                part: (
                    np.asarray(mesh.vertices, dtype=np.float64),
                    np.asarray(mesh.faces, dtype=np.int32),
                )
                for part, mesh in self.stage1_overlap_meshes.items()
            }
        if backend in {"grid", "grid_compare"}:
            pitch = float(getattr(self, "stage1_overlap_grid_pitch", 0.002))
            if pitch <= 0.0:
                raise ValueError("overlap_grid_pitch must be positive")
            pitch_mm = int(round(pitch * 1000.0))
            if not np.isclose(pitch, pitch_mm / 1000.0):
                raise ValueError("overlap_grid_pitch must be an integer number of millimeters")
            self.stage1_overlap_fwn_grids = {}
            for part, mesh in self.stage1_overlap_meshes.items():
                cache_path = obj.cfg[f"{part}_mesh_fname"] + f".fwn_grid_{pitch_mm}mm.npz"
                if os.path.exists(cache_path):
                    with np.load(cache_path, allow_pickle=False) as cached:
                        origin = cached["origin"].astype(np.float32)
                        values = cached["values"].astype(np.float32)
                else:
                    origin = (mesh.bounds[0] - 0.004).astype(np.float32)
                    upper = (mesh.bounds[1] + 0.004).astype(np.float32)
                    shape = np.ceil((upper - origin) / pitch).astype(np.int64) + 1
                    values = np.empty(tuple(shape), dtype=np.float32)
                    vertices, faces = self.stage1_overlap_fwn_meshes[part]
                    flat_values = values.reshape(-1)
                    yz = int(shape[1] * shape[2])
                    for start in range(0, flat_values.size, 100000):
                        stop = min(start + 100000, flat_values.size)
                        linear = np.arange(start, stop, dtype=np.int64)
                        ix = linear // yz
                        remainder = linear % yz
                        iy = remainder // int(shape[2])
                        iz = remainder % int(shape[2])
                        points = origin + pitch * np.stack([ix, iy, iz], axis=1)
                        flat_values[start:stop] = self.stage1_overlap_igl.fast_winding_number_for_meshes(
                            vertices,
                            faces,
                            np.asarray(points, dtype=np.float64),
                        )
                    np.savez_compressed(cache_path, origin=origin, pitch=np.float32(pitch), values=values)
                upper = origin + pitch * (np.asarray(values.shape, dtype=np.float32) - 1.0)
                volume = torch.from_numpy(values.transpose(2, 1, 0).copy())[None, None].to(self.device)
                self.stage1_overlap_fwn_grids[part] = (
                    volume,
                    torch.as_tensor(origin, device=self.device),
                    torch.as_tensor(upper, device=self.device),
                )
        if self.stage1_overlap_timing_enabled:
            asset_ms = (time.perf_counter() - asset_start) * 1000.0
            print(f"[stage1 overlap timing] asset_init_ms={asset_ms:.2f}", flush=True)

    def _compute_inside_mask(self, keypoint_pos, pose, mesh, side, part):
        _ensure_overlap_timing_state(self)
        quat = pose[:, 3:7]
        rot_mats = matrix_from_quat(quat).transpose(1, 2)
        keypoints_local_gpu = torch.einsum(
            "nij,nkj->nki",
            rot_mats,
            keypoint_pos - pose[:, :3].unsqueeze(1),
        )
        backend = getattr(self, "stage1_overlap_inside_backend", "trimesh")
        contains_start = time.perf_counter()
        if backend in {"grid", "grid_compare"}:
            volume, origin, upper = self.stage1_overlap_fwn_grids[part]
            normalized = 2.0 * (keypoints_local_gpu - origin) / (upper - origin) - 1.0
            sample_grid = normalized.reshape(1, -1, 1, 1, 3)
            winding = torch.nn.functional.grid_sample(
                volume,
                sample_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).reshape(keypoint_pos.shape[0], keypoint_pos.shape[1])
            grid_inside = winding.abs() > 0.5
            if backend == "grid_compare":
                if not hasattr(self, "stage1_overlap_grid_compare_masks"):
                    self.stage1_overlap_grid_compare_masks = {}
                self.stage1_overlap_grid_compare_masks[(side, part)] = grid_inside
            if backend == "grid":
                inside_mask = grid_inside
        if backend != "grid":
            flat_points = keypoints_local_gpu.detach().cpu().reshape(-1, 3).numpy()
            finite_mask = np.isfinite(flat_points).all(axis=1)
            inside_flat = np.zeros(flat_points.shape[0], dtype=bool)
            if finite_mask.any():
                if backend in {"fwn", "compare"}:
                    vertices, faces = self.stage1_overlap_fwn_meshes[part]
                    winding = self.stage1_overlap_igl.fast_winding_number_for_meshes(
                        vertices,
                        faces,
                        np.asarray(flat_points[finite_mask], dtype=np.float64),
                    )
                    fwn_inside = np.abs(winding) > 0.5
                if backend == "fwn":
                    inside_flat[finite_mask] = fwn_inside
                else:
                    exact_inside = mesh.contains(flat_points[finite_mask])
                    inside_flat[finite_mask] = exact_inside
                    if backend in {"compare", "grid_compare"}:
                        candidate = fwn_inside if backend == "compare" else grid_inside.detach().cpu().reshape(-1).numpy()[finite_mask]
                        mismatch = int(np.count_nonzero(candidate != exact_inside))
                        print(
                            "[stage1 overlap compare] "
                            f"backend={backend} side={side} part={part} points={len(exact_inside)} mismatch={mismatch} "
                            f"agreement={1.0 - mismatch / max(1, len(exact_inside)):.8f}",
                            flush=True,
                        )
            inside_mask = torch.from_numpy(inside_flat.reshape(keypoint_pos.shape[0], keypoint_pos.shape[1])).to(self.device)
        contains_ms = (time.perf_counter() - contains_start) * 1000.0
        if self.stage1_overlap_timing_enabled:
            self.stage1_overlap_timing_contains_calls += 1
            contains_call_idx = self.stage1_overlap_timing_contains_calls
            if _should_log_overlap_timing(contains_call_idx, self.stage1_overlap_timing_every):
                print(
                    "[stage1 overlap timing] "
                    f"contains_call={contains_call_idx} side={side} part={part} "
                    f"envs={keypoint_pos.shape[0]} points={keypoint_pos.numel() // 3} "
                    f"contains_ms={contains_ms:.2f}",
                    flush=True,
                )
        return inside_mask, contains_ms

    def _force_object_to_demo(self):
        if not getattr(self, "stage1_gt_object_train", False) or self.n_objects != 1:
            return
        obj = self.objects[self.object_names[0]]
        demo_t = torch.clamp(self.episode_length_buf, 0, obj.demo_states.shape[0] - 1)
        demo_state = obj.demo_states[demo_t]
        obj.set_object_state(
            root_pos=demo_state[:, :3],
            root_quat=demo_state[:, 3:7],
            joint_qpos=demo_state[:, 7:8],
            env_idxs=torch.arange(self.num_envs, dtype=torch.int32, device=self.device),
        )

    def patched_compute_intermediate_values(self):
        _ensure_overlap_timing_state(self)
        self.stage1_overlap_timing_calls += 1
        overlap_call_idx = self.stage1_overlap_timing_calls
        overlap_start = time.perf_counter()
        original_compute_intermediate_values(self)
        if not getattr(self, "stage1_gt_object_train", False) or self.n_objects != 1:
            return
        _force_object_to_demo(self)
        original_compute_intermediate_values(self)
        _ensure_overlap_geom_assets(self)
        obj = self.objects[self.object_names[0]]
        unsigned_by_side = []
        inside_by_side = []
        contains_ms_total = 0.0
        for side in ["left", "right"]:
            robot = self.robots[side]
            part_dists = []
            part_inside = []
            for part in ["top", "bottom"]:
                part_pose = obj.get_part_pose(part)
                part_dists.append(
                    self.compute_closest_vertice_dist_single(
                        self.stage1_overlap_obj_verts[part],
                        robot.kpt_pos,
                        part_pose,
                    )
                )
                inside_mask_part, contains_ms = _compute_inside_mask(
                    self,
                    robot.kpt_pos,
                    part_pose,
                    self.stage1_overlap_meshes[part],
                    side,
                    part,
                )
                part_inside.append(inside_mask_part)
                contains_ms_total += contains_ms
            unsigned_by_side.append(torch.stack(part_dists, dim=-1).min(dim=-1).values)
            inside_by_side.append(torch.stack(part_inside, dim=-1).any(dim=-1))
        unsigned_dists = torch.cat(unsigned_by_side, dim=1)
        inside_mask = torch.cat(inside_by_side, dim=1)
        signed_like_dists = torch.where(inside_mask, -unsigned_dists, unsigned_dists)
        self.stage1_overlap_geom_dists = unsigned_dists
        self.stage1_overlap_geom_inside = inside_mask
        self.stage1_overlap_geom_signed = signed_like_dists
        if getattr(self, "stage1_overlap_inside_backend", "trimesh") == "grid_compare":
            candidate_inside = torch.cat(
                [
                    torch.stack(
                        [self.stage1_overlap_grid_compare_masks[(side, part)] for part in ["top", "bottom"]],
                        dim=-1,
                    ).any(dim=-1)
                    for side in ["left", "right"]
                ],
                dim=1,
            )
            candidate_signed = torch.where(candidate_inside, -unsigned_dists, unsigned_dists)
            margin = float(getattr(self, "stage1_overlap_geom_threshold", 0.0))
            weight = float(getattr(self, "stage1_overlap_geom_penalty", 0.0))
            exact_penalty = weight * torch.clamp(margin - signed_like_dists, min=0.0).mean(dim=1)
            candidate_penalty = weight * torch.clamp(margin - candidate_signed, min=0.0).mean(dim=1)
            penalty_delta = (candidate_penalty - exact_penalty).abs()
            print(
                "[stage1 overlap reward compare] "
                f"call={overlap_call_idx} mean_abs_delta={penalty_delta.mean().item():.10f} "
                f"max_abs_delta={penalty_delta.max().item():.10f}",
                flush=True,
            )
        if self.stage1_overlap_timing_enabled and _should_log_overlap_timing(overlap_call_idx, self.stage1_overlap_timing_every):
            overlap_total_ms = (time.perf_counter() - overlap_start) * 1000.0
            mean_episode_step = float(self.episode_length_buf.float().mean().item())
            print(
                "[stage1 overlap timing] "
                f"call={overlap_call_idx} num_envs={self.num_envs} episode_step_mean={mean_episode_step:.2f} "
                f"contains_total_ms={contains_ms_total:.2f} overlap_total_ms={overlap_total_ms:.2f}",
                flush=True,
            )

    def patched_get_rewards(self):
        rew_dict = original_get_rewards(self)
        signed_like_dists = getattr(self, "stage1_overlap_geom_signed", None)
        unsigned_dists = getattr(self, "stage1_overlap_geom_dists", None)
        inside_mask = getattr(self, "stage1_overlap_geom_inside", None)
        penalty_weight = getattr(self, "stage1_overlap_geom_penalty", 0.0)
        if penalty_weight > 0.0 and signed_like_dists is not None:
            dist_margin = getattr(self, "stage1_overlap_geom_threshold", 0.0)
            overlap_gap = torch.clamp(dist_margin - signed_like_dists, min=0.0)
            overlap_penalty = penalty_weight * overlap_gap.mean(dim=1)
            self.rew_buf[:] -= overlap_penalty
            rew_dict["overlap_penalty"] = overlap_penalty
            rew_dict["overlap_geom_penalty"] = overlap_penalty
            rew_dict["overlap_geom_dist_min"] = unsigned_dists.min(dim=1).values
            rew_dict["overlap_geom_dist_mean"] = unsigned_dists.mean(dim=1)
            rew_dict["overlap_geom_signed_min"] = signed_like_dists.min(dim=1).values
            rew_dict["overlap_geom_inside_ratio"] = inside_mask.float().mean(dim=1)
        return rew_dict

    BaseEnv._compute_intermediate_values = patched_compute_intermediate_values
    BaseEnv._get_rewards = patched_get_rewards
    BaseEnv._stage1_gt_object_patch_installed = True


def main():
    parser = build_parser()
    args = parser.parse_args()

    obj_name, start, end, subject_name, use_clip = parse_clip_string(args.clip)
    args.arctic_object = obj_name
    args.frame_start = start
    args.frame_end = end

    env_kwargs = get_all_env_cfg(args, device="cuda:0")
    disable_curriculum_for_stage1(env_kwargs)
    env_kwargs["reward_cfg"]["imi_vel_rew_weight"] = float(args.imi_vel_rew_weight)
    env_kwargs["reward_cfg"].setdefault("imi_vel_kpt_beta", 40.0)
    env_kwargs["reward_cfg"].setdefault("imi_vel_wrist_pos_beta", 40.0)
    env_kwargs["reward_cfg"].setdefault("imi_vel_wrist_rot_beta", 4.0)
    env_kwargs["reward_cfg"].setdefault("imi_vel_wrist_weight", 0.5)
    env_kwargs["env_cfg"]["use_rl_games"] = True

    _install_stage1_velocity_reward_patch()
    _install_stage1_gt_object_overlap_patch()
    register_network("stage1_actor_critic", Stage1SemanticBuilder)

    gs.init(backend=gs.gpu, logging_level="warning")
    env = BaseEnv(**env_kwargs)
    env.stage1_gt_object_train = not args.disable_gt_object_train
    env.stage1_overlap_geom_penalty = float(args.overlap_geom_penalty)
    env.stage1_overlap_geom_threshold = float(args.overlap_geom_threshold)
    env.stage1_overlap_inside_backend = args.overlap_inside_backend
    env.stage1_overlap_grid_pitch = float(args.overlap_grid_pitch)
    obs_spec = build_stage1_obs_spec(env)

    with open(get_rl_config_path("rl_games_ppo_cfg"), encoding="utf-8") as handle:
        agent_cfg = yaml.full_load(handle)

    exp_name = make_stage1_exp_name(args)
    agent_cfg["params"]["seed"] = args.seed
    agent_cfg["params"]["model"]["name"] = "continuous_a2c_logstd"
    agent_cfg["params"]["network"]["name"] = "stage1_actor_critic"
    agent_cfg["params"]["network"]["stage1_architecture"] = args.stage1_architecture
    agent_cfg["params"]["network"]["stage1_obs_spec"] = obs_spec
    agent_cfg["params"]["network"]["stage1_field_units"] = list(args.stage1_field_units)
    agent_cfg["params"]["network"]["stage1_encoder_activation"] = args.stage1_encoder_activation
    agent_cfg["params"]["network"]["stage1_use_layer_norm"] = not args.stage1_disable_layer_norm
    if args.stage1_architecture == "maniptrans":
        agent_cfg["params"]["network"]["mlp"]["units"] = list(args.stage1_maniptrans_units)
    else:
        agent_cfg["params"]["network"]["mlp"]["units"] = list(args.stage1_trunk_units)
    agent_cfg["params"]["config"]["name"] = args.hand

    log_root_path = os.path.abspath(os.path.join("logs", "rl_games", args.hand))
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = exp_name
    agent_cfg["params"]["config"]["max_epochs"] = int(args.max_epochs)
    agent_cfg["params"]["config"]["save_frequency"] = int(args.save_freq)
    agent_cfg["params"]["config"]["num_steps_per_env"] = args.horizon
    agent_cfg["params"]["config"]["learning_rate"] = args.learning_rate

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    wrapped_env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, use_sil=False)
    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: wrapped_env})

    agent_cfg["params"]["config"]["num_actors"] = wrapped_env.unwrapped.num_envs
    agent_cfg["params"]["config"]["minibatch_size"] = int(args.num_envs * 8)
    agent_cfg["params"]["config"]["mini_epochs"] = max(1, int(args.num_envs / 4096 * 5))

    env_save_kwargs = env_kwargs.copy()
    env_save_kwargs.pop("demo_data")
    env_save_kwargs.pop("retarget_data")
    env_save_kwargs["stage1_obs_spec"] = obs_spec

    wandb_cfg = agent_cfg.copy()
    wandb_cfg["env_kwargs"] = env_save_kwargs
    wandb_cfg["clip"] = f"{obj_name}{start}-{end}-{subject_name}-{use_clip}"
    wandb_cfg["hand"] = args.hand
    wandb_cfg["stage1_obs_spec"] = obs_spec

    run = wandb.init(
        project=args.wandb_project,
        config=wandb_cfg,
        monitor_gym=True,
        save_code=True,
        name=exp_name,
    )

    env_save_kwargs["wandb"] = {"run_name": run.name, "run_id": run.id}

    params_dir = os.path.join(log_root_path, exp_name, "params")
    dump_yaml(os.path.join(params_dir, "env.yaml"), env_save_kwargs)
    dump_yaml(os.path.join(params_dir, "agent.yaml"), agent_cfg)
    dump_yaml(os.path.join(params_dir, "stage1_obs_spec.yaml"), obs_spec)
    pickle.dump(env_kwargs, open(os.path.join(params_dir, "env.pkl"), "wb"))

    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()

    runner_args = {"train": True, "play": False, "sigma": None}
    if args.checkpoint is not None:
        runner_args["checkpoint"] = os.path.abspath(args.checkpoint)
    runner.run(runner_args)


if __name__ == "__main__":
    main()
