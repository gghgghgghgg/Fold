import os
import re
import yaml
import torch
import numpy as np
import genesis as gs
from glob import glob
from os.path import join

from pathlib import Path
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.kinematics_adaptor import KinematicAdaptor, MimicJointKinematicAdaptor

from fold.hand_proc.hand_utils import common_hand_proc_args
from fold.envs.object import ArticulatedObject, get_arctic_object_cfg
from fold.envs.robot import get_hand_specific_cfg
from fold.retargeting.retarget_utils import get_demo_obj_tensors



"""
Here we tune gains by loading a clip of ARCTIC demo data and use dex-retargeting to get joint references for the hand.
Then we use the joint references to tune the gains of the hand.

python hand_processing/tune_gains.py --hand ability --clip box-40-400 --num_iters 10 --kp 50 10.0 --kv 0.1 1.0 --force_range 100.0 100.0 -v
"""
def create_scene(args, hand_urdfs, obj_name):
    gs.init(backend=gs.gpu)
    scene_cfg = dict(
        sim_options=gs.options.SimOptions(
            dt=1/60,
            substeps=2,
            gravity=(0, 0, -9.81),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=True,
            #shadow hand fingers basically doesn't move when True
        ),
        show_viewer=args.vis,
        use_visualizer=True,
        show_FPS=False,
        vis_options = gs.options.VisOptions(
            # visualize_contact=True,
            plane_reflection = True,
            ambient_light    = (0.6, 0.6, 0.6),
            lights = [
                {"type": "directional", "dir": (0, 0, -1), "color": (1.0, 1.0, 1.0), "intensity": 8.0},
            ]
        ),
    )
    if args.raytrace and args.record_video:
        scene_cfg['renderer'] = gs.renderers.RayTracer(
            env_surface=gs.surfaces.Emission(
                emissive_texture=gs.textures.ImageTexture(
                    image_path="textures/indoor_bright.png",
                ),
            ),
            env_radius=15.0,
            env_euler=(0, 0, 180),
            lights=[
                {"pos": (0.0, 0.0, 10.0), "radius": 3.0, "color": (15.0, 15.0, 15.0)},
            ],
        )
    scene = gs.Scene(**scene_cfg)
    device = torch.device('cuda:0')
    # plane = scene.add_entity(gs.morphs.URDF(file='urdf/plane/plane.urdf', fixed=True))
    cardbox_size = (0.2,0.2,0.1)
    if 'notebook' in obj_name:
        print("Adding a wider cardboard box for notebook")
        cardbox_size = (0.25, 0.2, 0.1)
    cardboard_box = scene.add_entity(
            gs.morphs.Box(
                pos=(0, -0.08, 0.90),
                size=cardbox_size,
                fixed=True,
            ),
            surface=gs.surfaces.Smooth(
                roughness=0.1,
            ),
        )
    if not args.skip_object:
        obj_cfg = get_arctic_object_cfg(name=obj_name)
        # print('Warning: setting object to fixed')
        obj_cfg['fixed'] = False
        obj_cfg['base_init_pos'] = (0.0, 0.0, 0.5)
        obj = ArticulatedObject(
            obj_cfg, device=device, scene=scene, num_envs=args.num_envs,
            )
    else:
        obj = None
    hand_entities = dict()
    for side, urdf_path in hand_urdfs.items():
        hand = scene.add_entity(
            gs.morphs.URDF(
                file=urdf_path,
                fixed=True,
                merge_fixed_links=False,
                recompute_inertia=True,
                collision=True,
            ),
            material=gs.materials.Rigid(
                gravity_compensation=0.8
                ),
            # vis_mode="collision", # NOTE: uncomment this to visualize collision shapes
        )
        hand_entities[side] = hand
    camera = None
    if args.record_video:
        camera = scene.add_camera(
            pos=(0.0, -2.3, 2.0),
            lookat=(0, -0.08, 0.90),
            res=(512, 512),
            fov=30,
            spp=512,
        )

    scene.build(n_envs=args.num_envs, env_spacing=(args.spacing, args.spacing))
    return scene, hand_entities, obj, camera

def control_set_hand_to_step(
    hand_entities, dof_idxs, tuned_dof_idxs, init_step,
    retar_data, step, device, env_idxs=None, control_joints=True
    ):
    for side in ['left', 'right']:
        hand = hand_entities[side]
        if env_idxs is None:
            env_idxs = [0]
        num_envs = len(env_idxs)
        hand_qpos, _, _= [retar_data[side][key][step] for key in ['hand_qpos', 'wrist_qpos', 'wrist_idxs']]
        init_hand_qpos, _, _= [retar_data[side][key][init_step] for key in ['hand_qpos', 'wrist_qpos', 'wrist_idxs']]
        all_dof_idxs = dof_idxs[side]
        tuned_idxs = tuned_dof_idxs[side]
        # only use the gains for the tuned joints
        for i, idx in enumerate(all_dof_idxs):
            if idx in tuned_idxs:
                continue
            hand_qpos[i] = init_hand_qpos[i]
        hand_qpos = torch.tensor(hand_qpos).to(device).unsqueeze(0).repeat(num_envs, 1)
        if control_joints:
            hand.control_dofs_position(
                position=hand_qpos,
                dofs_idx_local=all_dof_idxs,
                envs_idx=env_idxs,
            )
        else:
            hand.set_dofs_position(
                position=hand_qpos,
                dofs_idx_local=all_dof_idxs,
                zero_velocity=True,
                envs_idx=env_idxs,
            )
    return

def find_joints_in_group(entity, exprs):
    # a list of regex expressions to match joint names
    all_joints = entity.joints
    names, idxs = [], []
    all_act_idxs = []
    for joint in all_joints:
        if joint.type not in [gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC]:
            continue
        all_act_idxs.append(joint.dof_idx_local)
        matched = False
        jname = joint.name
        for expr in exprs: # e.g. expr is '*J*, jname is 'MFJ1'
            if bool(re.match(expr, jname)):
                matched = True
                break
        if matched:
            names.append(jname)
            idx = joint.dof_idx_local
            assert isinstance(idx, int), f"Expected int, got {type(idx)}"
            idxs.append(idx)
    if len(idxs) == 0:
        print(f"Warning: No joints found for {exprs}")
        breakpoint()
    return names, idxs, all_act_idxs

def interpolate_gain_val(vals, num_iters, num_joints, multiplier=1):
    assert len(vals) == 2, "Need upper and lower bounds"
    _interp = np.linspace(vals[0], vals[1], num_iters)
    exp_mult = [multiplier**i for i in range(num_iters)]
    _interp = _interp * exp_mult
    # repeat for all joints to get shape (num_envs, num_joints)
    interp = np.repeat(_interp[:, None], num_joints, axis=1)
    return torch.tensor(interp, dtype=torch.float32)

def set_joint_gains(args, hand_entities, dof_idxs, kp, kv, fr, env_idxs=None):
    # interpolate the hand joints
    num_envs = args.num_envs
    num_joints = len(dof_idxs['left'])
    if env_idxs is None:
        env_idxs = list(range(num_envs))
    for side in ['left', 'right']:
        entity = hand_entities[side]
        entity.set_dofs_kp(
            kp,
            dofs_idx_local=dof_idxs[side],
        )
        entity.set_dofs_kv(
            kv,
            dofs_idx_local=dof_idxs[side],
        )

        entity.set_dofs_force_range(
            -1 * fr,
            fr,
            dofs_idx_local=dof_idxs[side],
        )
    return

def set_gains_from_cfg(entity, actuator_cfgs):
    for joint_group, act_cfg in actuator_cfgs.items():
        joint_exprs = act_cfg['joint_exprs']
        kp = act_cfg['kp']
        kv = act_cfg['kv']
        fr = act_cfg['force_range']
        names, idxs, act_idxs = find_joints_in_group(entity, joint_exprs)
        print(f"Joint group: {joint_group} | Names: {names} | idxs: {idxs}")
        num_joints = len(idxs)
        kp = torch.tensor([kp] * num_joints, dtype=torch.float32)
        kv = torch.tensor([kv] * num_joints, dtype=torch.float32)
        fr = torch.tensor([fr] * num_joints, dtype=torch.float32)
        entity.set_dofs_kp(
            kp,
            dofs_idx_local=idxs,
        )
        entity.set_dofs_kv(
            kv,
            dofs_idx_local=idxs,
        )
        entity.set_dofs_force_range(
            -1 * fr,
            fr,
            dofs_idx_local=idxs,
        )
    return


def main(args):
    print("Hard-code num envs to 2")
    args.num_envs = 2
    hand = args.hand if ("_hand" in args.hand or args.hand == "xhand") else args.hand + "_hand"
    device  = torch.device('cuda:0')
    robot_dir = f"assets/{hand}"
    config_path = join(robot_dir, "retarget_config.yaml")
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    with Path(config_path).open('r') as f:
        input_cfg = yaml.safe_load(f)
    # NOTE the URDF paths here are taken from the config yaml file!!
    urdfs = dict()
    for side in ['left', 'right']:
        urdf_path = input_cfg[side]['urdf_path']
        urdfs[side] = join(robot_dir, urdf_path)

    clip = str(args.clip)
    retargeted_fname = join(args.retarget_dir, hand, f"{clip}.pt")
    assert os.path.exists(retargeted_fname), f"Does not exist: {retargeted_fname}"
    retar_data = torch.load(retargeted_fname)
    print(f"Loaded retargeted data from {retargeted_fname}")

    obj_name = clip.split('-')[0]
    start = int(clip.split('-')[1])
    end = int(clip.split('-')[2])
    demo_fname = f"assets/arctic/processed/s01/{obj_name}_use_01.npy"
    loaded = np.load(demo_fname, allow_pickle=True).item()
    world_data = loaded["world_coord"]
    obj_init_state = get_demo_obj_tensors(loaded, 0, device)

    scene, hand_entities, obj, camera = create_scene(args, urdfs, obj_name)
    if not args.freespace and obj is not None:
        obj.set_object_state(
            root_pos=obj_init_state[0][None].repeat(args.num_envs, 1),
            root_quat=obj_init_state[1][None].repeat(args.num_envs, 1),
            joint_qpos=obj_init_state[2][None].repeat(args.num_envs, 1),
            env_idxs=[0,1]
            )

    hand_cfg = get_hand_specific_cfg(name=hand)

    # first, set the gains from config default values
    for side, hand in hand_entities.items():
        actuator_cfgs = hand_cfg[side]['actuators']
        set_gains_from_cfg(hand, actuator_cfgs)
    # Do gain tuning for only one joint group
    tuned_dof_idxs = dict()
    all_act_idxs = dict()
    for side in urdfs.keys():
        actuator_cfgs = hand_cfg[side]['actuators']
        joint_exprs = actuator_cfgs[args.joint_group]['joint_exprs']
        entity = hand_entities[side]
        names, idxs, act_idxs = find_joints_in_group(entity, joint_exprs)
        tuned_dof_idxs[side] = idxs
        all_act_idxs[side] = act_idxs
    print(tuned_dof_idxs)
    main_env_idxs = [0]
    reference_env_idxs = [1]
    # set the joint gains
    num_joints = len(tuned_dof_idxs['left'])
    num_steps = len(retar_data['left']['hand_qpos'])
    kp = interpolate_gain_val(args.kp, args.num_iters, num_joints)
    kv = interpolate_gain_val(args.kv, args.num_iters, num_joints, multiplier=args.kv_multiplier)
    fr = interpolate_gain_val(args.force_range, args.num_iters, num_joints)
    frames = []
    init_step = 0
    if args.step_response:
        init_step = 20 # try setting to more midair step
        assert args.target_tstep < num_steps, "Target time step should be less than total steps"
    for itr in range(args.num_iters):
        print(f"Iteration {itr+1}/{args.num_iters}")
        print(f"Gains: kp={kp[itr]}, kv={kv[itr]}, force_range={fr[itr]}")
        set_joint_gains(args, hand_entities, tuned_dof_idxs, kp[itr], kv[itr], fr[itr])
        control_set_hand_to_step(
            hand_entities, all_act_idxs, tuned_dof_idxs, init_step,
            retar_data, init_step, device,
            env_idxs=reference_env_idxs+main_env_idxs,
            control_joints=False,
            )
        for step in range(0, num_steps, args.step_interp):
            if args.step_response:
                t = args.target_tstep
                init_t = init_step
            else:
                t = step
                init_t = step # show the full traj!

            control_set_hand_to_step(
                hand_entities, all_act_idxs, tuned_dof_idxs, init_t,
                retar_data, t, device, env_idxs=reference_env_idxs, control_joints=False
                )
            control_set_hand_to_step(
                hand_entities, all_act_idxs, tuned_dof_idxs, init_t,
                retar_data, t, device, env_idxs=main_env_idxs, control_joints=True
                )
            scene.step()
            side = 'right'
            control_force = hand_entities[side].get_dofs_control_force(dofs_idx_local=all_act_idxs[side])[0]
            # round it to 2 decimal places
            control_force = np.round(control_force.cpu().numpy(), 1)
            # only print if any is bigger than 10:
            if np.any(np.abs(control_force) > 100):
                print(f"Control force: {control_force} for side {side}")
            # control errors
            # control_error = hand_entities[side].get_dofs_position() -
            # breakpoint()

            if args.record_video:
                frame, _, _, _ = camera.render()
                frames.append(frame)

        if not args.freespace and obj is not None:
            obj.set_object_state(
                root_pos=obj_init_state[0][None].repeat(args.num_envs, 1),
                root_quat=obj_init_state[1][None].repeat(args.num_envs, 1),
                joint_qpos=obj_init_state[2][None].repeat(args.num_envs, 1),
                env_idxs=[0,1]
                )
        if not args.record_video:
            print(f"Current gains: \n kp={kp[itr]}, kv={kv[itr]}, force_range={fr[itr]}")
            breakpoint()
        scene.reset()
    if args.record_video:
        video_path = f"{args.hand}_tuned_gains.mp4"
        from moviepy.editor import ImageSequenceClip
        clip = ImageSequenceClip(frames, fps=20)
        clip.write_videofile(video_path)
        print(f"Exported video to {video_path}")
    breakpoint()



if __name__ == '__main__':
    parser = common_hand_proc_args()
    parser.add_argument('--num_iters', type=int, default=10)
    parser.add_argument('--kp', type=float, nargs='+', default=[1.0, 10.0])
    parser.add_argument('--kv', type=float, nargs='+', default=[0.1, 1.0])
    parser.add_argument('--kv_multiplier', type=float, default=1.0)
    parser.add_argument('--force_range', '-fr', type=float, nargs='+', default=[100.0, 100.0])
    parser.add_argument('--joint_group', type=str, default='wrist_trans')
    parser.add_argument('--spacing', type=float, default=0.0)
    parser.add_argument('--retarget_dir', type=str, default='hand_proc/retargeted')
    parser.add_argument('--freespace', '-fs', action='store_true')
    parser.add_argument('--step_interp', type=int, default=1)
    parser.add_argument('--step_response', action='store_true') # if true, only do one step and set a fixed target for multiple time steps
    # step_response is true, set the target time step
    parser.add_argument('--target_tstep', type=int, default=50)
    parser.add_argument('--skip_object', action='store_true')
    args = parser.parse_args()
    main(args)