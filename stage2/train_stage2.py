from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
from copy import deepcopy
from pathlib import Path

import genesis as gs
import yaml
import wandb
from rl_games.algos_torch.model_builder import register_network
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fold.asset_utils import get_rl_config_path
from fold.envs.base_env import BaseEnv
from fold.envs.constructors import get_all_env_cfg, get_common_argparser, parse_clip_string
from fold.rl.ppo_config import configure_ppo_rollout
from fold.rl.rl_games_wrapper import RlGamesGpuEnv, RlGamesVecEnvWrapper
from stage1.train_stage1 import (
    _install_stage1_velocity_reward_patch,
    dump_yaml,
)
from stage2.network_builder import Stage2ResidualBuilder
from stage2.obs_spec import build_stage2_obs_spec
from stage2.realtime_force_field import (
    RealtimeForceFieldConfig,
    install_realtime_force_field_observation,
)
from stage2.realtime_wrench import RealtimeWrenchConfig, install_realtime_wrench_observation


def build_parser() -> argparse.ArgumentParser:
    parser = get_common_argparser()
    parser.add_argument(
        "--imi_vel_rew_weight",
        "-imv",
        type=float,
        default=0.1,
        help="Extra stage2 imitation reward weight for matching demo velocities.",
    )
    parser.add_argument("--exp_name", "-exp", type=str, default="stage2_residual", help="Experiment name.")
    parser.add_argument("--horizon", "-ho", type=int, default=16, help="Number of steps per environment.")
    parser.add_argument("--checkpoint", "-ck", type=str, default=None, help="Stage2 checkpoint file to resume.")
    parser.add_argument(
        "--base_checkpoint",
        type=str,
        default=None,
        help="Frozen base-policy checkpoint used to compose the residual policy.",
    )
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.0003, help="Learning rate.")
    parser.add_argument(
        "--mini_epochs",
        type=int,
        default=None,
        help="PPO optimization epochs per rollout; defaults to scaling with num_envs.",
    )
    parser.add_argument("--wandb_project", "-wp", type=str, default="fold-stage2", help="WandB project.")
    parser.add_argument("--save_freq", "-sf", type=int, default=1000)
    parser.add_argument(
        "--disable_low_reward_early_stop",
        action="store_true",
        help="Keep training non-curriculum runs below the legacy reward-35 early-stop threshold.",
    )
    parser.add_argument(
        "--stage2_architecture",
        choices=["semantic", "maniptrans"],
        default="semantic",
        help="Residual encoder: semantic field branches or the flat ManipTrans MLP.",
    )
    parser.add_argument(
        "--stage2_maniptrans_units",
        type=int,
        nargs="+",
        default=[512, 512, 256, 128],
        help="Hidden sizes for the flat ManipTrans residual MLP.",
    )
    parser.add_argument(
        "--use_curriculum",
        action="store_true",
        help="Enable the standard object-actuator Curriculum during residual training.",
    )
    parser.add_argument(
        "--stage2_field_units",
        type=int,
        nargs="+",
        default=[128, 64],
        help="Hidden sizes used by each stage2 semantic field encoder.",
    )
    parser.add_argument(
        "--stage2_trunk_units",
        type=int,
        nargs="+",
        default=[512, 256, 128],
        help="Hidden sizes used by the PPO trunk after stage2 semantic encoding.",
    )
    parser.add_argument(
        "--stage2_encoder_activation",
        type=str,
        default="elu",
        choices=["elu", "relu", "tanh", "silu", "identity"],
        help="Activation used by stage2 semantic field encoders.",
    )
    parser.add_argument(
        "--stage2_disable_layer_norm",
        action="store_true",
        help="Disable LayerNorm inside stage2 semantic field encoders.",
    )
    parser.add_argument(
        "--stage2_residual_scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the learned residual action before adding it to the frozen base action.",
    )
    parser.add_argument("--realtime_wrench_cmd", action="store_true")
    parser.add_argument("--realtime_force_field_cmd", action="store_true")
    parser.add_argument(
        "--field_command_data",
        default="force_command_estimator/data/box_gt_global_force_commands_full.npz",
    )
    parser.add_argument(
        "--field_encoder_checkpoint",
        default="force_command_estimator/results/field_encoder/box_pointnet_300e/best_field_encoder.pt",
    )
    parser.add_argument("--field_solver_iterations", type=int, default=32)
    parser.add_argument("--field_prefit_iterations", type=int, default=8)
    parser.add_argument("--field_active_top_k", type=int, default=60)
    parser.add_argument("--field_friction", type=float, default=0.5)
    parser.add_argument("--field_regularization", type=float, default=1e-4)
    parser.add_argument("--field_normal_force_limit", type=float, default=100.0)
    parser.add_argument("--wrench_position_kp", type=float, default=200.0)
    parser.add_argument("--wrench_position_kd", type=float, default=20.0)
    parser.add_argument("--wrench_rotation_kp", type=float, default=20.0)
    parser.add_argument("--wrench_rotation_kd", type=float, default=2.0)
    parser.add_argument("--wrench_force_scale", type=float, default=25.0)
    parser.add_argument("--wrench_torque_scale", type=float, default=3.0)
    parser.add_argument("--wrench_command_clip", type=float, default=5.0)
    return parser


def make_stage2_exp_name(args) -> str:
    obj_name, start, end, subject_name, use_clip = parse_clip_string(args.clip)
    hand_prefix = str(args.hand).split("_")[0]
    exp_name = f"{hand_prefix}-{args.exp_name}"
    exp_name += f"_{obj_name}{start}-{end}-{subject_name}-{use_clip}_B{args.num_envs}"
    exp_name += f"_{args.action_mode}"
    if args.stage2_architecture != "semantic":
        exp_name += f"_{args.stage2_architecture}arch"
    if args.use_curriculum:
        exp_name += "_curriculum"
    exp_name += f"_ho{args.horizon}"
    exp_name += f"_imi{args.imi_rew_weight}"
    if args.imi_vel_rew_weight > 0:
        exp_name += f"_imv{args.imi_vel_rew_weight}"
    if args.bc_rew_weight > 0:
        exp_name += f"_bc{args.bc_rew_weight}"
    if args.contact_rew_weight > 0:
        exp_name += f"_con{args.contact_rew_weight}"
    if args.realtime_wrench_cmd:
        exp_name += "_rtw6"
    if args.realtime_force_field_cmd:
        exp_name += "_rtfield128"
    exp_name += f"_s2res{args.stage2_residual_scale}"
    exp_name += "_stage2semantic"
    return exp_name


def _load_agent_cfg_from_checkpoint(checkpoint_path: str) -> dict:
    ckpt_path = Path(checkpoint_path).resolve()
    agent_cfg_path = ckpt_path.parents[1] / "params" / "agent.yaml"
    if not agent_cfg_path.exists():
        raise FileNotFoundError(f"Could not find base agent config next to checkpoint: {agent_cfg_path}")
    with open(agent_cfg_path, encoding="utf-8") as handle:
        return yaml.full_load(handle)


def _resolve_base_checkpoint(args) -> str:
    if args.base_checkpoint is not None:
        return os.path.abspath(args.base_checkpoint)
    if args.checkpoint is None:
        raise ValueError("Stage2 requires --base_checkpoint for fresh runs or a resumable stage2 --checkpoint.")

    resume_agent_cfg = _load_agent_cfg_from_checkpoint(args.checkpoint)
    resume_base_checkpoint = resume_agent_cfg.get("params", {}).get("network", {}).get("base_checkpoint")
    if not resume_base_checkpoint:
        raise ValueError("Could not recover base_checkpoint from the resumed stage2 checkpoint config.")
    return os.path.abspath(resume_base_checkpoint)


def _resolve_wandb_mode() -> str | None:
    if os.environ.get("WANDB_MODE"):
        return None
    if os.environ.get("WANDB_API_KEY"):
        return None
    if sys.stdin.isatty():
        return None
    print("WANDB_MODE not set in non-interactive session; defaulting to offline wandb logging.")
    return "offline"


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.horizon <= 0:
        parser.error("--horizon must be positive")
    if args.use_curriculum:
        args.actuate_object = True
    if args.realtime_wrench_cmd and args.realtime_force_field_cmd:
        parser.error("--realtime_wrench_cmd and --realtime_force_field_cmd are mutually exclusive")
    if args.field_solver_iterations <= 0:
        parser.error("--field_solver_iterations must be positive")
    if args.field_prefit_iterations <= 0:
        parser.error("--field_prefit_iterations must be positive")
    if args.field_active_top_k < 0:
        parser.error("--field_active_top_k must be non-negative")

    obj_name, start, end, subject_name, use_clip = parse_clip_string(args.clip)
    args.arctic_object = obj_name
    args.frame_start = start
    args.frame_end = end

    base_checkpoint = _resolve_base_checkpoint(args)
    base_agent_cfg = _load_agent_cfg_from_checkpoint(base_checkpoint)

    env_kwargs = get_all_env_cfg(args, device="cuda:0")
    env_kwargs["reward_cfg"]["imi_vel_rew_weight"] = float(args.imi_vel_rew_weight)
    env_kwargs["reward_cfg"].setdefault("imi_vel_kpt_beta", 40.0)
    env_kwargs["reward_cfg"].setdefault("imi_vel_wrist_pos_beta", 40.0)
    env_kwargs["reward_cfg"].setdefault("imi_vel_wrist_rot_beta", 4.0)
    env_kwargs["reward_cfg"].setdefault("imi_vel_wrist_weight", 0.5)
    env_kwargs["env_cfg"]["use_rl_games"] = True

    _install_stage1_velocity_reward_patch()
    register_network("stage2_residual_actor_critic", Stage2ResidualBuilder)

    gs.init(backend=gs.gpu, logging_level="warning")
    env = BaseEnv(**env_kwargs)
    field_config_payload = None
    if args.realtime_wrench_cmd:
        wrench_config = RealtimeWrenchConfig(
            position_kp=args.wrench_position_kp,
            position_kd=args.wrench_position_kd,
            rotation_kp=args.wrench_rotation_kp,
            rotation_kd=args.wrench_rotation_kd,
            force_scale=args.wrench_force_scale,
            torque_scale=args.wrench_torque_scale,
            command_clip=args.wrench_command_clip,
        )
        install_realtime_wrench_observation(env, wrench_config)
    if args.realtime_force_field_cmd:
        wrench_config = RealtimeWrenchConfig(
            position_kp=args.wrench_position_kp,
            position_kd=args.wrench_position_kd,
            rotation_kp=args.wrench_rotation_kp,
            rotation_kd=args.wrench_rotation_kd,
            force_scale=args.wrench_force_scale,
            torque_scale=args.wrench_torque_scale,
            command_clip=args.wrench_command_clip,
        )
        field_config = RealtimeForceFieldConfig(
            data_path=args.field_command_data,
            encoder_checkpoint=args.field_encoder_checkpoint,
            frame_offset=args.frame_start,
            prefit_iterations=args.field_prefit_iterations,
            active_top_k=args.field_active_top_k,
            iterations=args.field_solver_iterations,
            friction=args.field_friction,
            regularization=args.field_regularization,
            normal_force_limit=args.field_normal_force_limit,
            wrench=wrench_config,
        )
        field_config_payload = {
            "data_path": os.path.abspath(args.field_command_data),
            "encoder_checkpoint": os.path.abspath(args.field_encoder_checkpoint),
            "frame_offset": int(args.frame_start),
            "prefit_iterations": int(args.field_prefit_iterations),
            "active_top_k": int(args.field_active_top_k),
            "iterations": int(args.field_solver_iterations),
            "friction": float(args.field_friction),
            "regularization": float(args.field_regularization),
            "normal_force_limit": float(args.field_normal_force_limit),
            "wrench": {
                "position_kp": float(args.wrench_position_kp),
                "position_kd": float(args.wrench_position_kd),
                "rotation_kp": float(args.wrench_rotation_kp),
                "rotation_kd": float(args.wrench_rotation_kd),
                "force_scale": float(args.wrench_force_scale),
                "torque_scale": float(args.wrench_torque_scale),
                "command_clip": float(args.wrench_command_clip),
            },
        }
        install_realtime_force_field_observation(env, field_config)
    obs_spec = build_stage2_obs_spec(env)

    with open(get_rl_config_path("rl_games_ppo_cfg"), encoding="utf-8") as handle:
        agent_cfg = yaml.full_load(handle)

    exp_name = make_stage2_exp_name(args)
    agent_cfg["params"]["seed"] = args.seed
    agent_cfg["params"]["model"]["name"] = "continuous_a2c_logstd"
    agent_cfg["params"]["network"]["name"] = "stage2_residual_actor_critic"
    agent_cfg["params"]["network"]["stage2_architecture"] = args.stage2_architecture
    agent_cfg["params"]["network"]["stage2_obs_spec"] = obs_spec
    agent_cfg["params"]["network"]["stage2_field_units"] = list(args.stage2_field_units)
    agent_cfg["params"]["network"]["stage2_encoder_activation"] = args.stage2_encoder_activation
    agent_cfg["params"]["network"]["stage2_use_layer_norm"] = not args.stage2_disable_layer_norm
    agent_cfg["params"]["network"]["stage2_residual_scale"] = float(args.stage2_residual_scale)
    agent_cfg["params"]["network"]["base_checkpoint"] = base_checkpoint
    agent_cfg["params"]["network"]["base_network_config"] = deepcopy(base_agent_cfg["params"]["network"])
    if field_config_payload is not None:
        agent_cfg["params"]["network"]["realtime_force_field_config"] = field_config_payload
    if args.stage2_architecture == "maniptrans":
        agent_cfg["params"]["network"]["mlp"]["units"] = list(args.stage2_maniptrans_units)
    else:
        agent_cfg["params"]["network"]["mlp"]["units"] = list(args.stage2_trunk_units)
    agent_cfg["params"]["config"]["name"] = args.hand

    log_root_path = os.path.abspath(os.path.join("logs", "rl_games", args.hand))
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = exp_name
    agent_cfg["params"]["config"]["max_epochs"] = int(args.max_epochs)
    agent_cfg["params"]["config"]["save_frequency"] = int(args.save_freq)
    agent_cfg["params"]["config"]["learning_rate"] = args.learning_rate
    agent_cfg["params"]["config"]["disable_low_reward_early_stop"] = bool(args.disable_low_reward_early_stop)
    agent_cfg["params"]["config"]["curriculum_zero_stop_epochs"] = int(args.curriculum_zero_stop_epochs)

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    wrapped_env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, use_sil=False)
    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: wrapped_env})

    num_actors = int(wrapped_env.unwrapped.num_envs)
    agent_cfg["params"]["config"]["num_actors"] = num_actors
    configure_ppo_rollout(agent_cfg["params"]["config"], num_actors, args.horizon)
    agent_cfg["params"]["config"]["mini_epochs"] = (
        args.mini_epochs if args.mini_epochs is not None else max(1, int(args.num_envs / 4096 * 5))
    )

    env_save_kwargs = env_kwargs.copy()
    env_save_kwargs.pop("demo_data")
    env_save_kwargs.pop("retarget_data")
    env_save_kwargs["stage2_obs_spec"] = obs_spec
    env_save_kwargs["base_checkpoint"] = base_checkpoint

    wandb_cfg = deepcopy(agent_cfg)
    wandb_cfg["env_kwargs"] = env_save_kwargs
    wandb_cfg["clip"] = f"{obj_name}{start}-{end}-{subject_name}-{use_clip}"
    wandb_cfg["hand"] = args.hand
    wandb_cfg["stage2_obs_spec"] = obs_spec

    wandb_init_kwargs = dict(
        project=args.wandb_project,
        config=wandb_cfg,
        monitor_gym=True,
        save_code=True,
        name=exp_name,
    )
    wandb_mode = _resolve_wandb_mode()
    if wandb_mode is not None:
        wandb_init_kwargs["mode"] = wandb_mode

    run = wandb.init(**wandb_init_kwargs)

    env_save_kwargs["wandb"] = {"run_name": run.name, "run_id": run.id}

    params_dir = os.path.join(log_root_path, exp_name, "params")
    dump_yaml(os.path.join(params_dir, "env.yaml"), env_save_kwargs)
    dump_yaml(os.path.join(params_dir, "agent.yaml"), agent_cfg)
    dump_yaml(os.path.join(params_dir, "stage2_obs_spec.yaml"), obs_spec)
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
