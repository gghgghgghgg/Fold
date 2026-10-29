import os
import yaml
import genesis as gs
import numpy as np
import torch
import argparse
import math
import inspect
import os
from os.path import join
import re
import yaml
import json
import pickle
import hashlib
import random
import shlex
import sys
from datetime import datetime

from fold.asset_utils import get_rl_config_path
from fold.envs.base_env import BaseEnv
from fold.envs.contacts import get_contact_marker_cfgs
from fold.envs.constructors import get_common_argparser, parse_clip_string
from fold.envs.demo_data import load_genesis_retarget_data
from fold.rl.rl_games_wrapper import RlGamesVecEnvWrapper, RlGamesGpuEnv


from collections import defaultdict
import shutil
import moviepy
from rl_games.algos_torch.model_builder import register_network
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner


def gather_object_state_tensor(demo_data):
    """ demo data should already be sliced since it's loaded from env kwargs """
    obj_pos = demo_data["obj_pos"]
    obj_quat = demo_data["obj_quat"]
    obj_arti = demo_data["obj_arti"] # need to reshape to (T, 1) instead of (T,)
    if len(obj_arti.shape) == 1:
        obj_arti = obj_arti[:, None]
    arr = np.concatenate([obj_pos, obj_quat, obj_arti], axis=1)
    return torch.tensor(arr).float()

def _append_recorded_frame(uenv):
    uenv.scene._visualizer.update(force=True)
    frame, depth_arr, seg_arr, normal_arr = uenv._floating_camera.render(segmentation=uenv.render_segmentation)
    if uenv.render_segmentation:
        frame = np.concatenate([frame, seg_arr[:, :, None]], axis=-1)
    uenv._recorded_frames.append(frame)


def _disable_object_collisions_runtime(env):
    for obj in env.objects.values():
        solver = obj.entity._solver
        for geom in obj.entity.geoms:
            solver.geoms_info.contype[geom.idx] = 0
            solver.geoms_info.conaffinity[geom.idx] = 0


def eval_one_episode(env, agent, obj_state_tensor, print_rew=False, record_video=False, show_reference=False, gt_object_trajectory=False, max_steps=None, zero_object_gains=True):
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()
    uenv = env.unwrapped
    ep_len = int(max_steps) if max_steps is not None else uenv.max_episode_length
    num_envs = uenv.num_envs
    device = torch.device('cuda:0')
    manual_gt_object_render = bool(record_video and (gt_object_trajectory or show_reference))
    if record_video:
        uenv.start_recording()
        uenv.max_video_frames = int(ep_len)
        max_frames = uenv.max_video_frames
        if manual_gt_object_render:
            uenv.record_video = False

    obj = None
    if len(uenv.objects) > 0:
        obj = uenv.objects[uenv.object_names[0]]
        if obj.actuated and zero_object_gains:
            print("Setting eval time obj gains to 0.0")
            obj.set_joint_gains(0.0, 0.0, force_range=0.0)
    assert obj is not None, "No object found in the environment"
    left_hand = uenv.robots["left"]
    right_hand = uenv.robots["right"]
    joint_target_left = left_hand.residual_qpos
    joint_target_right = right_hand.residual_qpos
    initial_state = {
        "initial_obj_state": torch.cat([obj.root_pos, obj.root_quat, obj.dof_pos], dim=-1).cpu().numpy(),
        "initial_left_hand_qpos": left_hand.dof_pos.cpu().numpy(),
        "initial_right_hand_qpos": right_hand.dof_pos.cpu().numpy(),
    }

    eval_data = defaultdict(list)
    episode_returns = torch.zeros(num_envs, dtype=torch.float32, device=device)
    episode_lengths = torch.full((num_envs,), ep_len, dtype=torch.int32, device=device)
    active_episodes = torch.ones(num_envs, dtype=torch.bool, device=device)
    for i in range(ep_len):
        with torch.inference_mode():
            env_step = i if show_reference else uenv.episode_length_buf.cpu().numpy()[0]
            env_step = min(env_step, len(obj_state_tensor) - 1)
            demo_state = obj_state_tensor[env_step]

            if show_reference:
                obj.set_object_state(
                    root_pos=demo_state[:3][None].to(device),
                    root_quat=demo_state[3:7][None].to(device),
                    joint_qpos=demo_state[7:8][None].to(device),
                    env_idxs=torch.tensor([0], dtype=torch.int32, device=device),
                )
                for robot, joints in zip([left_hand, right_hand], [joint_target_left, joint_target_right]):
                    robot.set_joint_position(joint_targets=joints[env_step][None], env_idxs=[0])
                if record_video:
                    _append_recorded_frame(uenv)
                obj_state = torch.cat([obj.root_pos, obj.root_quat, obj.dof_pos], dim=-1)
                eval_data["obj_state"].append(obj_state.cpu().numpy())
                eval_data["demo_state"].append(demo_state.cpu().numpy())
                eval_data["left_hand_qpos"].append(left_hand.dof_pos.cpu().numpy())
                eval_data["left_hand_demo_qpos"].append(joint_target_left[env_step].cpu().numpy())
                eval_data["right_hand_qpos"].append(right_hand.dof_pos.cpu().numpy())
                eval_data["right_hand_demo_qpos"].append(joint_target_right[env_step].cpu().numpy())
                continue

            # get actions from the agent
            actions = agent.get_action(obs, is_deterministic=True)
            obs, rew, dones, infos = env.step(actions)
            # Keep the exact normalized policy command together with the
            # translated targets and realized state.  This makes an eval
            # rollout usable as a traceable sim-to-real replay source instead
            # of requiring the video to be reverse engineered.
            eval_data["policy_action"].append(actions.detach().cpu().numpy())
            reward_values = rew.reshape(-1)
            eval_data["reward"].append(reward_values.detach().cpu().numpy())
            episode_returns[active_episodes] += reward_values[active_episodes]
            done_mask = dones.reshape(-1).bool() & active_episodes
            episode_lengths[done_mask] = i + 1
            active_episodes[done_mask] = False
            if gt_object_trajectory:
                obj.set_object_state(
                    root_pos=demo_state[:3][None].to(device),
                    root_quat=demo_state[3:7][None].to(device),
                    joint_qpos=demo_state[7:8][None].to(device),
                    env_idxs=torch.tensor([0], dtype=torch.int32, device=device),
                )
                if manual_gt_object_render:
                    _append_recorded_frame(uenv)
            obj_pos, obj_quat, obj_arti = obj.root_pos, obj.root_quat, obj.dof_pos
            # print(f"Step {env_step}: Obj pos: {obj_pos.cpu().numpy()}")
            obj_state = torch.cat([obj_pos, obj_quat, obj_arti], dim=-1)
            eval_data["obj_state"].append(obj_state.cpu().numpy())
            eval_data["demo_state"].append(demo_state.cpu().numpy())
            # Preserve the realized robot trajectory so deterministic policy
            # rollouts can be reproduced by the offline mesh renderer.
            eval_data["left_hand_qpos"].append(left_hand.dof_pos.cpu().numpy())
            eval_data["right_hand_qpos"].append(right_hand.dof_pos.cpu().numpy())
            eval_data["left_hand_target_qpos"].append(left_hand.curr_targets.cpu().numpy())
            eval_data["right_hand_target_qpos"].append(right_hand.curr_targets.cpu().numpy())
            eval_data["left_hand_qvel"].append(left_hand.dof_vel.cpu().numpy())
            eval_data["right_hand_qvel"].append(right_hand.dof_vel.cpu().numpy())
            eval_data["left_wrist_pose"].append(left_hand.wrist_pose.cpu().numpy())
            eval_data["right_wrist_pose"].append(right_hand.wrist_pose.cpu().numpy())
            eval_data["left_hand_keypoints"].append(left_hand.kpt_pos.cpu().numpy())
            eval_data["right_hand_keypoints"].append(right_hand.kpt_pos.cpu().numpy())

            rew_dict = uenv.rew_dict
            for key in ['task_rew', 'con_rew', 'pos_dist', 'rot_dist', 'arti_dist']:
                if key in rew_dict:
                    eval_data[key].append(rew_dict[key].cpu().numpy())
            contact_link_valid = getattr(uenv, "contact_link_valid", None)
            if contact_link_valid is not None:
                contact_active = contact_link_valid.reshape(num_envs, -1).any(dim=1)
                eval_data["contact_active"].append(contact_active.cpu().numpy())
            if len(dones) > 0:
                # reset rnn state for terminated episodes
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0

        if print_rew:
            print(f"Step {env_step}: Reward: {rew.cpu().numpy()}")
            # perform operations for terminated episodes
            rew_dict = uenv.rew_dict
            for k, v in rew_dict.items():
                if 'con' in k:
                    print(f"Step {env_step}: {k}: {v.cpu().numpy()}")
    eval_data = {k: np.stack(v) for k, v in eval_data.items()}
    eval_data.update(initial_state)
    # Runtime joint ordering is simulator-defined and must travel with the
    # numeric arrays; URDF XML ordering is not a safe replay contract.
    eval_data["left_hand_qpos_names"] = np.asarray(left_hand.actuated_dof_names)
    eval_data["right_hand_qpos_names"] = np.asarray(right_hand.actuated_dof_names)
    eval_data["left_hand_action_from_idxs"] = np.asarray(left_hand.action_from_idxs, dtype=np.int32)
    eval_data["right_hand_action_from_idxs"] = np.asarray(right_hand.action_from_idxs, dtype=np.int32)
    eval_data["left_hand_joint_from_idxs"] = np.asarray(left_hand.joint_from_idxs, dtype=np.int32)
    eval_data["right_hand_joint_from_idxs"] = np.asarray(right_hand.joint_from_idxs, dtype=np.int32)
    eval_data["left_hand_joint_multipliers"] = left_hand.joint_multipliers.cpu().numpy()
    eval_data["right_hand_joint_multipliers"] = right_hand.joint_multipliers.cpu().numpy()
    eval_data["episode_return"] = episode_returns.cpu().numpy()
    eval_data["episode_length"] = episode_lengths.cpu().numpy()
    if manual_gt_object_render:
        uenv.record_video = True
    frames = uenv.get_recorded_frames()
    return frames, eval_data



def load_agent_cfg(ckpt_path):
    saved_agent_cfg_fname = os.path.join(ckpt_path, "params", "agent.yaml")
    if os.path.exists(saved_agent_cfg_fname):
        with open(saved_agent_cfg_fname, encoding="utf-8") as f:
            agent_cfg = yaml.full_load(f)
    else:
        agent_cfg_fname = get_rl_config_path("rl_games_ppo_cfg")
        with open(agent_cfg_fname, encoding="utf-8") as f:
            agent_cfg = yaml.full_load(f)

    network_name = agent_cfg.get("params", {}).get("network", {}).get("name")
    if network_name == "stage1_actor_critic":
        from stage1.network_builder import Stage1SemanticBuilder

        register_network("stage1_actor_critic", Stage1SemanticBuilder)
    elif network_name == "stage2_residual_actor_critic":
        from stage2.network_builder import Stage2ResidualBuilder

        register_network("stage2_residual_actor_critic", Stage2ResidualBuilder)

    return agent_cfg


def install_stage2_observation_extensions(env, agent_cfg, retarget_name_override=None):
    obs_spec = agent_cfg.get("params", {}).get("network", {}).get("stage2_obs_spec")
    if not obs_spec:
        return
    field_names = {field.get("field") for field in obs_spec.get("fields", [])}
    if "realtime_wrench" in field_names:
        from stage2.realtime_wrench import RealtimeWrenchConfig, install_realtime_wrench_observation

        install_realtime_wrench_observation(env, RealtimeWrenchConfig())
    if "realtime_force_field" in field_names:
        from stage2.realtime_force_field import (
            install_realtime_force_field_observation,
            realtime_force_field_config_from_payload,
        )

        network_cfg = agent_cfg.get("params", {}).get("network", {})
        saved_cfg = network_cfg.get("realtime_force_field_config")
        if saved_cfg is None:
            saved_cfg = {
                "data_path": "force_command_estimator/data/box_gt_global_force_commands_full.npz",
                "encoder_checkpoint": "force_command_estimator/results/field_encoder/box_pointnet_300e/best_field_encoder.pt",
                "frame_offset": 30,
                "wrench": {},
            }
        if retarget_name_override is not None:
            saved_cfg = dict(saved_cfg)
            saved_cfg["retarget_name"] = retarget_name_override
        install_realtime_force_field_observation(
            env,
            realtime_force_field_config_from_payload(saved_cfg),
        )


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _first_env_series(eval_data, key):
    values = eval_data.get(key)
    if values is None:
        return None
    values = np.asarray(values)
    if values.ndim >= 2:
        values = values[:, 0]
    return values.reshape(-1)


def build_strict_metrics_summary(
    eval_data,
    args,
    checkpoint_epoch,
    checkpoint_sha256,
    env_params_path,
    effective_object_cfgs,
):
    reward = _first_env_series(eval_data, "reward")
    con_rew = _first_env_series(eval_data, "con_rew")
    contact_active = _first_env_series(eval_data, "contact_active")
    pos_error = _first_env_series(eval_data, "pos_dist")
    rot_error = _first_env_series(eval_data, "rot_dist")
    arti_dist = _first_env_series(eval_data, "arti_dist")
    if reward is None or pos_error is None or rot_error is None:
        raise RuntimeError("Strict metrics require reward, pos_dist, and rot_dist arrays")

    frame_count = int(reward.shape[0])
    if pos_error.shape[0] != frame_count or rot_error.shape[0] != frame_count:
        raise RuntimeError("Strict metric arrays have inconsistent frame counts")
    position_pass = pos_error <= float(args.position_success_threshold)
    rotation_threshold_rad = math.radians(float(args.rotation_success_threshold_deg))
    rotation_pass = rot_error <= rotation_threshold_rad
    frame_pass = position_pass & rotation_pass
    failed_steps = np.flatnonzero(~frame_pass)
    first_failed_step = int(failed_steps[0]) if failed_steps.size else None
    completion_length_ratio = 1.0 if first_failed_step is None else float(first_failed_step / frame_count)
    success = bool(first_failed_step is None)
    if success:
        exit_reason = "completed_full_clip_with_all_frames_passing"
    else:
        reasons = []
        if not position_pass[first_failed_step]:
            reasons.append("position_threshold_exceeded")
        if not rotation_pass[first_failed_step]:
            reasons.append("rotation_threshold_exceeded")
        exit_reason = "+".join(reasons)

    articulation_error = None
    if arti_dist is not None:
        # The environment stores a single articulated-object error as squared_error / 2.
        articulation_error = np.sqrt(np.maximum(2.0 * arti_dist, 0.0))

    initial_arrays = [
        np.asarray(eval_data[key])
        for key in ("initial_obj_state", "initial_left_hand_qpos", "initial_right_hand_qpos")
    ]
    initial_digest = hashlib.sha256()
    for arr in initial_arrays:
        initial_digest.update(np.ascontiguousarray(arr).tobytes())

    return {
        "schema_version": 1,
        "task": args.eval_task_name,
        "method": args.eval_method,
        "checkpoint_role": args.checkpoint_role,
        "checkpoint": {
            "path": os.path.abspath(args.checkpoint),
            "filename": os.path.basename(args.checkpoint),
            "epoch": checkpoint_epoch,
            "sha256": checkpoint_sha256,
        },
        "environment": {
            "params_path": os.path.abspath(env_params_path),
            "params_sha256": _sha256_file(env_params_path),
            "seed": int(args.seed),
            "num_envs": int(args.num_envs),
            "randomization": False,
            "rand_init_ratio": 0.0,
            "early_reset_threshold": 0.0,
            "preserve_terminal_state": bool(args.preserve_terminal_state),
            "object_control_mode": args.object_control_mode,
            "object_control": effective_object_cfgs,
            "deterministic_action": True,
            "initial_state_sha256": initial_digest.hexdigest(),
        },
        "thresholds": {
            "position_m": float(args.position_success_threshold),
            "rotation_deg": float(args.rotation_success_threshold_deg),
        },
        "metrics": {
            "reward_mean": float(reward.mean()),
            "contact_reward_mean": float(con_rew.mean()) if con_rew is not None else None,
            "contact_active_ratio": float(contact_active.astype(np.float64).mean()) if contact_active is not None else None,
            "position_error_mean_m": float(pos_error.mean()),
            "position_error_p90_m": float(np.percentile(pos_error, 90)),
            "rotation_error_mean_rad": float(rot_error.mean()),
            "rotation_error_mean_deg": float(np.degrees(rot_error).mean()),
            "rotation_error_p90_deg": float(np.degrees(np.percentile(rot_error, 90))),
            "articulation_error_mean_rad": float(articulation_error.mean()) if articulation_error is not None else None,
            "articulation_error_p90_rad": float(np.percentile(articulation_error, 90)) if articulation_error is not None else None,
            "sr": 1 if success else 0,
            "completion_length_ratio": completion_length_ratio,
            "frame_pass_ratio": float(frame_pass.astype(np.float64).mean()),
            "first_failed_step": first_failed_step,
            "frame_count": frame_count,
        },
        "exit_reason": exit_reason,
        "actual_command": shlex.join(sys.argv),
    }


def main():

    parser = get_common_argparser()
    parser.add_argument('--checkpoint', '-ck', type=str, default="inspire_hand")
    parser.add_argument('--eval_episodes', '-ne', type=int, default=1)
    parser.add_argument('--print_rew', '-pr', action='store_true')
    parser.add_argument('--show_reference', '-ref', action='store_true') # if not ture, don't show the retargeted reference
    parser.add_argument('--output_render', '-or', action='store_true') # if not ture, don't show the retargeted reference
    parser.add_argument('--render_dir', '-out', type=str, default="rendered") # if not provided, save in the same folder as the checkpoint
    parser.add_argument('--video_fname', '-of', type=str, default="-eval.mp4") # if not provided, save in the same folder as the checkpoint
    parser.add_argument('--eval_data_dir', type=str, default=None, help='Optional directory for eval NPY/NPZ/JSON outputs')
    parser.add_argument('--preserve_terminal_state', action='store_true', help='Delay environment timeout by one step so the final requested post-step state is not an auto-reset state')
    parser.add_argument('--env_params_from', type=str, default=None, help='Use trajectory, physics, reward, and initialization from this canonical env.pkl while preserving policy-compatible observation switches')
    parser.add_argument('--object_control_mode', choices=['zero', 'saved'], default='zero', help='Use zero object gains (legacy default) or the gains stored in env.pkl')
    parser.add_argument('--eval_task_name', type=str, default=None)
    parser.add_argument('--eval_method', choices=['baseline', 'ours'], default=None)
    parser.add_argument('--checkpoint_role', choices=['best', 'final'], default=None)
    parser.add_argument('--position_success_threshold', type=float, default=0.15)
    parser.add_argument('--rotation_success_threshold_deg', type=float, default=40.0)
    parser.add_argument('--disable_object_collision', action='store_true', help='Keep the object visible but disable its collisions during evaluation')
    parser.add_argument('--gt_object_trajectory', action='store_true', help='Force the rendered object to follow the demo trajectory every frame')
    parser.add_argument('--retarget_name_override', type=str, default=None, help='Reload retarget data for --clip instead of using checkpoint data')

    args = parser.parse_args()

    args.checkpoint = os.path.abspath(args.checkpoint)
    ckpt_path = os.path.dirname(os.path.dirname(args.checkpoint))
    saved_cfg_fname = os.path.join(ckpt_path, "params", "env.pkl")
    env_params_fname = os.path.abspath(args.env_params_from) if args.env_params_from else saved_cfg_fname

    ckpt_name = f"{args.checkpoint.split('/')[-1]}" # inpsire_ep1000 etc
    run_name = f"{args.checkpoint.split('/')[-3]}" # inpsire_ep1000 etc
    ckpt_data_folder = os.path.join(ckpt_path, ckpt_name.replace(".pth", "_eval"))
    if args.eval_data_dir is not None:
        ckpt_data_folder = os.path.abspath(args.eval_data_dir)
    os.makedirs(ckpt_data_folder, exist_ok=True)

    video_fname = join(ckpt_data_folder, f"video.mp4")
    if args.output_render:
        render_dir = os.path.join(args.render_dir, run_name)
        print('Saving video to a different folder')
        video_fname = os.path.join(render_dir, ckpt_name.split(".")[0] + args.video_fname)
        os.makedirs(render_dir, exist_ok=True)

    assert os.path.exists(saved_cfg_fname), f"File {saved_cfg_fname} does not exist"
    assert os.path.exists(env_params_fname), f"File {env_params_fname} does not exist"
    # Load the checkpoint's own environment first so policy-specific observation
    # switches remain compatible with the saved network input dimension.
    with open(saved_cfg_fname, "rb") as f:
        env_kwargs = pickle.load(f)
    if args.env_params_from:
        policy_observation_cfg = {
            key: env_kwargs['env_cfg'].get(key)
            for key in ('observe_tip_dist', 'observe_contact_force')
        }
        with open(env_params_fname, "rb") as f:
            env_kwargs = pickle.load(f)
        env_kwargs['env_cfg'].update(policy_observation_cfg)
        print(
            f"Using canonical trajectory/physics/reward params from {env_params_fname}; "
            f"preserved policy observation switches {policy_observation_cfg}"
        )

    if args.retarget_name_override is not None:
        obj_name, frame_start, frame_end, subject, use_clip = parse_clip_string(args.clip)
        demo_data, retarget_data = load_genesis_retarget_data(
            obj_name=obj_name,
            hand_name=args.hand,
            frame_start=frame_start,
            frame_end=frame_end,
            save_name=args.retarget_name_override,
            use_clip=use_clip,
            subject_name=subject,
        )
        env_kwargs['demo_data'] = demo_data
        env_kwargs['retarget_data'] = retarget_data
        env_kwargs['env_cfg']['episode_length'] = frame_end - frame_start
        print(f"Overriding checkpoint retarget data with {args.retarget_name_override} for {args.clip}")

    requested_eval_steps = int(env_kwargs['env_cfg']['episode_length'])
    if args.preserve_terminal_state:
        env_kwargs['env_cfg']['episode_length'] = requested_eval_steps + 1
        print(f"Delaying timeout to preserve post-step state for {requested_eval_steps} eval steps")

    assert env_kwargs['env_cfg']['use_rl_games'], "The saved environment is not from rl-games"

    if args.raytrace and args.record_video:
        env_kwargs['env_cfg']['scene_kwargs']['raytrace'] = True

    env_kwargs['env_cfg']['early_reset_threshold'] = 0.0
    print("WARNING: setting env.is_eval to True")
    env_kwargs['env_cfg']['is_eval'] = True
    if args.vis:
        # NOTE still needs this because kwargs are loaded from saved env
        env_kwargs['env_cfg']['scene_kwargs']['use_visualizer'] = True
        env_kwargs['env_cfg']['scene_kwargs']['show_viewer'] = True
    print("Not applying external forces during eval time")
    env_kwargs['rand_cfg']['randomize'] = False
    env_kwargs['rand_cfg']['external_force'] = False

    if args.show_markers:
        marker_cfgs = get_contact_marker_cfgs(
                num_vis_contacts=16,
                sources=['demo'],
                obj_parts=['top', 'bottom'],
                hand_sides=['left', 'right'],
            )
        env_kwargs['contact_marker_cfgs'] = marker_cfgs
        print('Setting visualize contact to True but observe contact force to False')
        env_kwargs['env_cfg']['scene_kwargs']['visualize_contact'] = True

    env_kwargs['env_cfg']['num_envs'] = args.num_envs
    env_kwargs['env_cfg']['rand_init_ratio'] = 0.0
    if args.record_video:
        env_kwargs['env_cfg']['scene_kwargs']['use_visualizer'] = True
        env_kwargs['env_cfg']['record_video'] = True
        print(f"Setting render resolution to 512")
        # env_kwargs['camera_res'] = (2048, 2048)
        env_kwargs['env_cfg']['camera_kwargs']['front'] = dict(
            res=(512, 512),
            fov=40,
            pos=(0.5, -1.5, 1.2),
            lookat=(0.0, -1.58, 2.0),
        )
    for name, cfg in env_kwargs['object_cfgs'].items():
        if args.object_control_mode == 'zero':
            print("Setting eval time object control to actuated=false, kp=0, kv=0, force_range=0")
            cfg['actuated'] = False
            cfg['kp'] = 0.0
            cfg['kv'] = 0.0
            cfg['force_range'] = 0.0
        else:
            print(
                f"Keeping saved object control for {name}: actuated={cfg.get('actuated')} "
                f"kp={cfg.get('kp')} kv={cfg.get('kv')} force_range={cfg.get('force_range')}"
            )

    if args.overlay:
        env_kwargs['env_cfg']['env_spacing'] = (0.0, 0.0)
    if args.object_control_mode == 'zero':
        print("Removing curriculum config for zero-gain eval")
        env_kwargs.pop("curriculum_cfg", None)
    else:
        print("Keeping saved curriculum initialization so actuated object gains remain at original params")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    checkpoint_data = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    checkpoint_epoch = int(checkpoint_data.get('epoch')) if isinstance(checkpoint_data, dict) and checkpoint_data.get('epoch') is not None else None
    del checkpoint_data
    checkpoint_sha256 = _sha256_file(args.checkpoint)
    effective_params = {
        "policy_env_params": saved_cfg_fname,
        "policy_env_params_sha256": _sha256_file(saved_cfg_fname),
        "source_env_params": env_params_fname,
        "source_env_params_sha256": _sha256_file(env_params_fname),
        "seed": int(args.seed),
        "env_cfg": {
            key: _jsonable(env_kwargs['env_cfg'].get(key))
            for key in (
                'episode_length', 'dt', 'early_reset_threshold', 'early_reset_aux_thres',
                'early_reset_interval', 'rand_init_ratio', 'use_contact_reward',
            )
        },
        "rand_cfg": _jsonable(env_kwargs.get('rand_cfg')),
        "object_cfgs": {
            name: {
                key: _jsonable(cfg.get(key))
                for key in (
                    'name', 'base_init_pos', 'base_init_quat', 'base_init_qpos', 'fixed',
                    'actuated', 'kp', 'kv', 'force_range', 'urdf_path',
                )
            }
            for name, cfg in env_kwargs['object_cfgs'].items()
        },
        "reward_cfg": _jsonable(env_kwargs.get('reward_cfg')),
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_sha256": checkpoint_sha256,
        "actual_command": shlex.join(sys.argv),
    }
    with open(os.path.join(ckpt_data_folder, "eval_params.json"), "w", encoding="utf-8") as f:
        json.dump(effective_params, f, indent=2, ensure_ascii=False)
    with open(os.path.join(ckpt_data_folder, "eval_command.txt"), "w", encoding="utf-8") as f:
        f.write(shlex.join(sys.argv) + "\n")

    device = torch.device('cuda:0')
    import genesis as gs
    gs.init(seed=args.seed, backend=gs.gpu, logging_level='warning')
    env = BaseEnv(
         **env_kwargs
    )
    if args.disable_object_collision:
        print("Disabling object collisions at runtime during eval")
        _disable_object_collisions_runtime(env)
    agent_cfg = load_agent_cfg(ckpt_path)
    agent_cfg["params"]["seed"] = int(args.seed)
    install_stage2_observation_extensions(env, agent_cfg, args.retarget_name_override)
    demo_data = env_kwargs['demo_data']
    obj_state_tensor = gather_object_state_tensor(demo_data)

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )

    uenv = env.unwrapped
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = uenv.num_envs

    # create runner from rl-games
    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    resume_path = os.path.abspath(args.checkpoint)
    agent.restore(resume_path)
    agent.reset()

    for eps in range(args.eval_episodes):
        frames, eval_data = eval_one_episode(
            env, agent, obj_state_tensor, args.print_rew, args.record_video, args.show_reference,
            args.gt_object_trajectory,
            max_steps=requested_eval_steps if args.preserve_terminal_state else None,
            zero_object_gains=args.object_control_mode == 'zero',
            )
        ckpt_eval_fname = os.path.join(ckpt_data_folder, f"eval_ep{eps}.npy")
        np.save(ckpt_eval_fname, eval_data)
        print(f"Saved eval data to {ckpt_eval_fname}")
        ckpt_eval_npz_fname = os.path.join(ckpt_data_folder, f"eval_ep{eps}.npz")
        safe_eval_data = {
            key: np.asarray(value)
            for key, value in eval_data.items()
            if np.asarray(value).dtype != object
        }
        np.savez_compressed(ckpt_eval_npz_fname, **safe_eval_data)
        print(f"Saved pickle-free eval data to {ckpt_eval_npz_fname}")
        reward_values = eval_data.get("reward")
        task_rew_values = eval_data.get("task_rew")
        con_rew_values = eval_data.get("con_rew")
        contact_values = eval_data.get("contact_active")
        contact_steps = (
            np.flatnonzero(contact_values.reshape(-1))
            if contact_values is not None
            else np.empty(0, dtype=np.int64)
        )
        eval_summary = {
            "reward_mean": float(reward_values.mean()) if reward_values is not None else None,
            "reward_sum": float(reward_values.sum()) if reward_values is not None else None,
            "task_rew_mean": float(task_rew_values.mean()) if task_rew_values is not None else None,
            "con_rew_mean": float(con_rew_values.mean()) if con_rew_values is not None else None,
            "last_contact_step": int(contact_steps[-1]) if contact_steps.size else None,
        }
        ckpt_summary_fname = os.path.join(ckpt_data_folder, f"eval_summary_ep{eps}.json")
        with open(ckpt_summary_fname, "w", encoding="utf-8") as f:
            json.dump(eval_summary, f, indent=2)
        print(f"Eval summary: {eval_summary}")
        print(f"Saved eval summary to {ckpt_summary_fname}")
        # Stage1 imitation/GT-object QC intentionally has no manipulation
        # pos_dist/rot_dist arrays.  Strict summaries are part of the named
        # Stage2 comparison protocol, so only require them for an explicitly
        # named eval task.
        if args.eval_task_name is not None:
            strict_summary = build_strict_metrics_summary(
                eval_data,
                args,
                checkpoint_epoch=checkpoint_epoch,
                checkpoint_sha256=checkpoint_sha256,
                env_params_path=env_params_fname,
                effective_object_cfgs=effective_params["object_cfgs"],
            )
            strict_summary_fname = os.path.join(
                ckpt_data_folder,
                "metrics_summary.json" if eps == 0 else f"metrics_summary_ep{eps}.json",
            )
            with open(strict_summary_fname, "w", encoding="utf-8") as f:
                json.dump(strict_summary, f, indent=2, ensure_ascii=False)
            print(f"Strict metrics summary: {strict_summary['metrics']}")
            print(f"Saved strict metrics summary to {strict_summary_fname}")
        # try loading the data
        # eval_data = np.load(ckpt_eval_fname, allow_pickle=True).item()
        if args.record_video:
            # save video with moviepy
            from moviepy.editor import ImageSequenceClip
            clip = ImageSequenceClip(frames, fps=int(1/uenv.dt/2))
            clip.write_videofile(video_fname)
            print(f"Saved video to {video_fname}")

    print("Done evaluating")


if __name__ == '__main__':
    main()
