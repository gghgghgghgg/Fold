import os
import math
import yaml
import torch
import wandb
import shutil
import pickle
import argparse
import numpy as np
import genesis as gs
from datetime import datetime

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from fold.asset_utils import get_rl_config_path
from fold.envs.base_env import BaseEnv
from fold.envs.constructors import get_common_argparser, get_all_env_cfg, parse_clip_string
from fold.rl.rl_games_wrapper import RlGamesVecEnvWrapper, RlGamesGpuEnv


def dump_yaml(filename: str, data: dict | object, sort_keys: bool = False):
    """Saves data into a YAML file safely.

    Note:
        The function creates any missing directory along the file's path.

    Args:
        filename: The path to save the file at.
        data: The data to save either a dictionary or class object.
        sort_keys: Whether to sort the keys in the output file. Defaults to False.
    """
    # check ending
    if not filename.endswith("yaml"):
        filename += ".yaml"
    # create directory
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    # make all the numpy arrays into lists, recursively
    def to_list(d):
        for k, v in d.items():
            if isinstance(v, dict):
                to_list(v)
            elif isinstance(v, np.ndarray):
                d[k] = v.tolist()

    # save data
    with open(filename, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=sort_keys)

def main():
    parser = get_common_argparser()
    # now add RL training args
    parser.add_argument("--exp_name", "-exp", type=str, default="inspire", help="Experiment name.")
    parser.add_argument("--horizon", '-ho', type=int, default=16, help="Number of steps per environment.")
    parser.add_argument("--checkpoint", '-ck', type=str, default=None, help="Checkpoint file to load.")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.0003, help="Learning rate for the agent.")
    parser.add_argument("--wandb_project", "-wp", type=str, default="fold", help="WandB project name.")
    parser.add_argument("--save_freq", "-sf", type=int, default=1000)
    args = parser.parse_args()


    obj_name, start, end, subject_name, use_clip = parse_clip_string(args.clip)
    args.arctic_object = obj_name
    args.frame_start = start
    args.frame_end = end

    hand_prefix = str(args.hand).split("_")[0]
    exp_name = hand_prefix + "-" + args.exp_name
    exp_name += f"_{obj_name}{start}-{end}-{subject_name}-{use_clip}_B{args.num_envs}"
    exp_name += "_"+args.action_mode
    exp_name += f"_thres{args.early_reset_threshold}"
    exp_name += f"_ho{args.horizon}"
    exp_name += f"_imi{args.imi_rew_weight}"
    if args.contact_rew_weight > 0:
        exp_name += f"_con{args.contact_rew_weight}"
    if args.rand_init_ratio > 0:
        exp_name += f"_rand{args.rand_init_ratio}"
    if args.bc_rew_weight > 0:
        exp_name += f"_bc{args.bc_rew_weight}"


    num_envs = args.num_envs
    env_kwargs = get_all_env_cfg(args, device='cuda:0')
    env_kwargs['env_cfg']['use_rl_games'] = True
    device = torch.device('cuda:0')
    import genesis as gs
    gs.init(backend=gs.gpu, logging_level='warning')
    env = BaseEnv(
         **env_kwargs
    )
    agent_cfg_fname = get_rl_config_path("rl_games_ppo_cfg")
    with open(agent_cfg_fname, encoding="utf-8") as f:
        agent_cfg = yaml.full_load(f)
    agent_cfg["params"]["seed"] = args.seed
    agent_cfg["params"]["config"]["name"] = args.hand

    log_root_path = os.path.join("logs", "rl_games", args.hand)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs
    # logging directory path: <train_dir>/<full_experiment_name>
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = exp_name
    agent_cfg["params"]["config"]["max_epochs"] = int(args.max_epochs)
    agent_cfg["params"]["config"]["save_frequency"] = int(args.save_freq)

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    if args.checkpoint is not None:
        assert os.path.exists(args.checkpoint), f"Checkpoint file not found: {args.checkpoint}"
        # agent_cfg["params"]["load_checkpoint"] = True
        # agent_cfg["params"]["load_path"] = args.checkpoint

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, use_sil=False)
    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )

    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    agent_cfg["params"]["config"]["minibatch_size"] = int(args.num_envs * 8)
    agent_cfg["params"]["config"]["mini_epochs"] = max(1, int(args.num_envs / 4096 * 5)) # 5 epochs per 4096 samples
    agent_cfg["params"]["config"]["num_steps_per_env"] = args.horizon
    agent_cfg["params"]["config"]["learning_rate"] = args.learning_rate


    env_save_kwargs = env_kwargs.copy()
    # pop the demo data and retargeted data
    env_save_kwargs.pop('demo_data')
    env_save_kwargs.pop('retarget_data')


    # convert agent_cfg to dict:
    wandb_cfg = agent_cfg.copy()
    wandb_cfg['env_kwargs'] = env_save_kwargs
    # also save args
    wandb_cfg['clip'] = f"{obj_name}{start}-{end}-{subject_name}-{use_clip}"
    wandb_cfg['hand'] = args.hand

    run = wandb.init(
        project=args.wandb_project,
        config=wandb_cfg,
        monitor_gym=True,
        save_code=True,
        name=exp_name,
    )

    # get wandb run name and id
    run_name = run.name
    run_id = run.id
    env_save_kwargs['wandb'] = dict(
        run_name=run_name,
        run_id=run_id,
    )
    dump_yaml(os.path.join(log_root_path, exp_name, "params", "env.yaml"), env_save_kwargs)
    dump_yaml(os.path.join(log_root_path, exp_name, "params", "agent.yaml"), agent_cfg)
    # also dump as pkl file
    pickle.dump(env_kwargs, open(os.path.join(log_root_path, exp_name, "params", "env.pkl"), "wb"))

    # create runner from rl-games
    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)

    # set seed of the env
    # env.seed(agent_cfg["params"]["seed"])
    # reset the agent and env
    runner.reset()
    # train the agent
    runner_args = {"train": True, "play": False, "sigma": None}
    if args.checkpoint is not None:
        runner_args["checkpoint"] = os.path.abspath(args.checkpoint)
    runner.run(runner_args)

    # close the simulator
    exit()


if __name__ == "__main__":
    # run the main function
    main()
    exit()
