import os
import cv2
import yaml
import torch
import argparse
import numpy as np
from glob import glob
from os.path import join
from copy import deepcopy
import genesis as gs

from pathlib import Path
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.kinematics_adaptor import KinematicAdaptor, MimicJointKinematicAdaptor

from fold.hand_proc.hand_utils import common_hand_proc_args, parse_clip_string
from fold.retargeting.retarget_utils import compose_retarget_config, retarget_all_steps

"""
Quick and dirty script to get retargeted hand joints from a short clip, use for gain tuning
python hand_processing/retarget_hands.py --hand shadow --clip box-0-200 --overwrite -v
"""
MANO_IDXS = [0, 16, 17, 18, 19, 20]
# MANO_IDXS = [0]

def get_entity_info(entity):
    all_joints = entity.joints
    actuated_joints = [joint for joint in all_joints if joint.type in [gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC]]
    actuated_dof_names = [joint.name for joint in actuated_joints]
    actuated_dof_idxs = [joint.dof_idx_local for joint in actuated_joints]

    qposes = np.concatenate([joint.init_qpos for joint in actuated_joints]) # shape (ndof,)
    hand_init_pos = torch.tensor(qposes, dtype=torch.float32)[None, :]
    return hand_init_pos, actuated_dof_names, actuated_dof_idxs

def create_scene(args, hand_urdfs, obj_name):
    gs.init(backend=gs.gpu)
    scene_cfg = dict(
        sim_options=gs.options.SimOptions(
            dt=1/60,
            substeps=2,
            gravity=(0, 0,0),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=True,
            enable_joint_limit=True,
        ),
        show_viewer=args.vis,
        use_visualizer=(args.vis or args.record_video),
        show_FPS=False,
        vis_options = gs.options.VisOptions(
            plane_reflection = True,
            ambient_light    = (0.4, 0.4, 0.4),
            lights = [
                {"type": "directional", "dir": (0, 0, -1), "color": (1.0, 1.0, 1.0), "intensity": 2.0},
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
            env_radius=10.0,
            env_euler=(0, 0, 180),
            lights=[
                {"pos": (0.0, 0.0, 10.0), "radius": 1.0, "color": (15.0, 15.0, 15.0)},
            ],
        )
    scene = gs.Scene(**scene_cfg)
    device = torch.device('cuda:0')
    plane = scene.add_entity(
        # gs.morphs.URDF(file='assets/plane/plane_custom.urdf', fixed=True)
        gs.morphs.URDF(file='urdf/plane/plane.urdf', fixed=True)
    )
    obj = None
    if args.show_object:
        cardboard_box = scene.add_entity(
            gs.morphs.Box(
                pos=(0, -0.08, 0.90),
                size=(0.2,0.2,0.1),
                fixed=True,
            ),
            surface=gs.surfaces.Smooth(
                roughness=0.1,
            ),
        )
        obj_urdf = f"assets/arctic/{obj_name}/decomp/{obj_name}_decomp.urdf"
        obj = scene.add_entity(
            gs.morphs.URDF(
                pos=(0,0,1.20),
                file=obj_urdf,
                fixed=False,
                merge_fixed_links=False,
                recompute_inertia=True,
                collision=True,
            ),
        )
    hand_entities = dict()
    for side, urdf_path in hand_urdfs.items():
        hand = scene.add_entity(
            gs.morphs.URDF(
                file=urdf_path,
                fixed=True,
                merge_fixed_links=False,
                recompute_inertia=True,
                # collision=True,
                collision=False,
            ),
            material=gs.materials.Rigid(
                gravity_compensation=0.8
                ),

        )
        hand_entities[side] = hand
    markers = []
    if args.show_mano:
        # show the mano fingertips + wrist kpt, 6 spheres per hand
        for i in range(len(MANO_IDXS) * 2):
            marker = scene.add_entity(
                gs.morphs.Sphere(
                    radius=0.009,
                    pos=(0, 0, 0),
                    fixed=False,
                    collision=False,
                ),
                surface=gs.surfaces.Smooth(
                    color=((0.0, 1.0, 0.0) if i < 6 else (1.0, 0.0, 0.0)),
                ),
            )
            markers.append(marker)

    camera = None
    if args.vis or args.record_video:
        camera = scene.add_camera(
            pos=(1.8,-1.8, 2.0),
            lookat=(0, -0.1, 1.1),
            res=(1024, 1024),
            fov=12,
            spp=512,
        )
    scene.build(n_envs=1, env_spacing=(2.0, 2.0))
    return scene, hand_entities, camera, obj, markers


def set_object_state(obj, params, step):
    obj_trans, obj_quat, obj_arti = [params[key][step] for key in ['obj_trans', 'obj_quat', 'obj_arti']]
    obj_trans = torch.tensor(obj_trans).unsqueeze(0)
    obj_quat = torch.tensor(obj_quat).unsqueeze(0)
    obj.set_pos(pos=obj_trans)
    obj.set_quat(quat=obj_quat)
    # obj.set_dofs_position(po)
    return

def main(args):
    hand = args.hand if ("_hand" in args.hand or args.hand == "xhand") else args.hand + "_hand"
    device  = torch.device('cuda:0')
    robot_dir = f"assets/{hand}"
    config_path = join(robot_dir, "retarget_config.yaml")
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    with Path(config_path).open('r') as f:
        input_cfg = yaml.safe_load(f)
    # NOTE the URDF paths here are taken from the config yaml file!!
    print(f"Start retargeting with config {config_path}")

    if args.retarget_type == 'vector' and hand == 'shadow_hand':
        print("warning: shadow hand vector retargeting does not work well")
        # breakpoint()
    retarget_type = args.retarget_type
    retargeters = dict()
    urdfs = dict()
    for side in ['left', 'right']:
        config_dict = compose_retarget_config(
            input_cfg[side],
            retarget_type,
            input_cfg.get("low_pass_alpha", 1.0),
            input_cfg.get("low_pass_alpha_vel", 1.0),
            ignore_mimic_joint=False,
            add_dummy_free_joint=False,
        )
        retarget_cfg = RetargetingConfig.from_dict(
            deepcopy(config_dict)
        )
        retargeters[side] = retarget_cfg.build()
        urdf_path = config_dict['urdf_path']
        urdfs[side] = join(robot_dir, urdf_path)
    clip = str(args.clip)
    obj_name, start, end, subject_name, use_clip = parse_clip_string(clip)

    save_dir = join(args.save_dir, hand)
    os.makedirs(save_dir, exist_ok=True)
    save_fname = join(save_dir, f"{clip}.pt")
    if os.path.exists(save_fname) and not args.overwrite:
        print(f"Already exists: {save_fname}, add --overwrite tag to overwrite")
        return

    demo_fname = f"assets/arctic/processed/{subject_name}/{obj_name}_use_{use_clip}.npy"
    if not os.path.exists(demo_fname):
        print(f"File {demo_fname} not found")
        return
    loaded = np.load(demo_fname, allow_pickle=True).item()
    world_data = loaded["world_coord"]
    params = loaded["params"] # has keys obj_trans, obj_rot, obj_arti
    scene, hand_entities, camera, obj, markers = create_scene(args, urdfs, obj_name)

    retar_data = dict()
    dof_idxs = dict()
    for side in ['left', 'right']:
        entity = hand_entities[side]
        hand_init_pos, actuated_dof_names, actuated_dof_idxs = get_entity_info(
            entity
        )
        dof_limits = entity.get_dofs_limit(actuated_dof_idxs) # (lower, upper) tuple
        dof_idxs[side] = actuated_dof_idxs
        retargeted_vals = retarget_all_steps(
            dof_limits,
            hand_init_pos,
            actuated_dof_names,
            actuated_dof_idxs,
            retargeters[side],
            int(end-start),
            world_data[f"joints.{side}"],
            retarget_type,
            frame_start=start,
        )
        retar_data[side] = retargeted_vals
    print(f"Retargeted data for {clip} is ready")
    # breakpoint()
    torch.save(retar_data, save_fname)
    print(f"Saved to {save_fname}")
    step = 0
    max_steps = end - start
    frames = []
    all_mano_kpts = np.concatenate([world_data[f"joints.{side}"][:, MANO_IDXS]   for side in ['left', 'right']], axis=1) # (T, 6, 3) -> (T, 12, 3)
    # take only frame start end
    all_mano_kpts = all_mano_kpts[start:end]
    all_mano_kpts = torch.tensor(all_mano_kpts).to(device)
    while True:
        for side, entity in hand_entities.items():
            hand_qpos, wrist_qpos, wrist_idxs = [retar_data[side][key][step] for key in ['hand_qpos', 'wrist_qpos', 'wrist_idxs']]
            hand_qpos = torch.tensor(hand_qpos).to(device).unsqueeze(0)

            if True: #step == 0:
                entity.set_dofs_position(
                    position=hand_qpos,
                    dofs_idx_local=dof_idxs[side],
                    # envs_idx=env_idxs,
                    zero_velocity=True,
                )
            entity.control_dofs_position(
                position=hand_qpos,
                dofs_idx_local=dof_idxs[side],
            )
        if args.show_object:
            set_object_state(obj, params, step)

        if args.show_mano:
            mano_kpts = all_mano_kpts[step]
            for i, marker in enumerate(markers):
                marker.set_pos(pos=mano_kpts[i][None])

        scene.step()
        if args.record_video:
            frame, depth_arr, seg_arr, normal_arr = camera.render(segmentation=True)
            # frame = frame.copy()
            # frame[seg_arr==0, :] = 255 # make background white
            frames.append(frame)
            fname = join(save_dir, f"{clip}_{step:04d}.png")
            # cv2 has rgb format issue:
            cv2.imwrite(fname, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            print(fname)
            breakpoint()

        step += 1
        if step == max_steps:
            step = 0
            # breakpoint()
        if step % 10 == 0:
            print(f"Step {step}/{max_steps}")
            for side in ['right', 'left']:
                hand_qpos = retar_data[side]['hand_qpos'][step]
                print(f"{side} wrist xyz: {hand_qpos[:3].round(decimals=2)}")
                print(f"{side} wrist quat: {hand_qpos[3:6].round(decimals=2)}")
            # breakpoint()



if __name__ == '__main__':
    parser = common_hand_proc_args()
    parser.add_argument('--save_dir', type=str, default='hand_proc/retargeted')
    parser.add_argument('--show_object', action='store_true')
    parser.add_argument('--retarget_type', type=str, default='vector')
    parser.add_argument('--show_mano', '-sm', action='store_true') # show the MANO keypoints as spheres
    args = parser.parse_args()
    main(args)