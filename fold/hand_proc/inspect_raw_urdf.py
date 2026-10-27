"""
Load and inspect the raw URDF files into a minimal Genesis scene

- Iterate all the rotations to find proper camera
URDF=filename
python hand_proc/inspect_raw_urdf.py --record_video --render_image --urdf_path $URDF --skip_wrist_interp  --group_collisions --interp_step 30 --iterate_quat
"""
import os
import cv2
import torch
import argparse
import numpy as np
import genesis as gs
from glob import glob
from os.path import join
from scipy.spatial.transform import Rotation
from fold.hand_proc.hand_utils import generate_90degree_rotation_quaternions

def render_image(camera, transparent=True):
    """Render an image from the camera"""
    img, depth_arr, seg_arr, normal_arr = camera.render(segmentation=True)
    rgb_img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # only do this for vis!
    if transparent: # return 4-channel
        max_id = seg_arr.max() # assuming this is background
        channel = np.ones(img.shape[0:2], dtype=np.uint8) * 255
        channel[seg_arr == max_id] = 0
        channel[seg_arr == -1] = 0
        img = np.concatenate([img, channel[:, :, None]], axis=-1)
        # full the background in rgb_img with 255
        rgb_img[seg_arr == max_id] = 255
        rgb_img[seg_arr == -1] = 255
    return img, rgb_img

def get_base_rotation(name):
    """ ASSUME ALL LEFT HANDS"""
    if 'ability' in name and 'left' in name:
        return (0.7071, 0, 0, -0.7071)
    if 'allegro' in name and 'left' in name:
        return (1, 0, 0, 0)
    if 'dex' in name and 'left' in name:
        return (1, 0, 0, 0)
    if 'inspire' in name and 'left' in name:
        return (0, 0, 0.7071, -0.7071)
    if 'mimic' in name and 'left' in name:
        return (0.5, 0.5, 0.5, 0.5)
    if 'schunk' in name and 'left' in name:
        return (0.7071, 0, 0, -0.7071)
    if 'xhand' in name and 'left' in name:
        return (0, 0.7071, -0.7071, 0)
    if 'shadow' in name and 'left' in name:
        return (1, 0, 0, 0)


def interpolate_hand_joints(hand, n_steps=200, skip_wrist=False):
    all_joints = hand.joints
    actuated_joints = [joint for joint in all_joints if joint.type in [gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC]]
    act_idxs = [joint.dof_idx_local for joint in actuated_joints]
    limits = []
    for joint in actuated_joints:
        if joint.type in [gs.JOINT_TYPE.REVOLUTE]:
            limits.append(joint.dofs_limit[0])
        else:
            limits.append((-0.05, 0.05))
    # overwrite rot limits
    limits[3:6] = [
        (-1.8, 0),
        # (0.0, 0.1),
        # (3.0, 3.2),
        (0.68, 5),
        # (3.6, 3.7),
        # (1.68, 1.8),
        (0, 4.14)
    ]
    lower_bounds = [limit[0] for limit in limits]
    upper_bounds = [limit[1] for limit in limits]
    if skip_wrist:
        # still actuate it but just set limit to 0
        lower_bounds[:6] = [0] * 6
        upper_bounds[:6] = [0] * 6
    # interpolate between the lower and upper bounds
    interp_joints = []
    for idx, (lower, upper) in enumerate(zip(lower_bounds, upper_bounds)):
        values1 = np.linspace(lower, upper, n_steps)
        # then iterate from upper to lower again
        values2 = np.linspace(upper, lower, n_steps)
        interp_joints.append(
            np.concatenate([values1, values2])
        )
    interp_joints = np.stack(interp_joints, axis=1) # shape (n_steps, n_joints)
    return interp_joints, act_idxs

def interpolate_wrist_finger_separately(hand, n_steps=50, skip_wrist=False, wrist_only=False, wrist_rot_only=False):
    """
    Interpolate only one joint at a time for the first 6 wrist joints, then the finger joints
    returns 2 * n_steps * 7 interpolated joint values
    """
    all_joints = hand.joints
    actuated_joints = [joint for joint in all_joints if joint.type in [gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC]]
    if skip_wrist:
        actuated_joints = actuated_joints[6:]
    if wrist_only:
        actuated_joints = actuated_joints[:6]
    if wrist_rot_only:
        actuated_joints = actuated_joints[3:6]
    act_idxs = [joint.dof_idx_local for joint in actuated_joints]
    limits = []
    for joint in actuated_joints:
        if joint.type in [gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC]:
            limits.append(joint.dofs_limit[0])
        # else:
        #     limits.append((-0.08, 0.08))
    if wrist_rot_only:
        # mimic left hand
        # limits = [
        #     (-0.1, 0.1),
        #     (2.8, 2.9),
        #     (1.2, 1.3),
        # ]
        # mimic right:
        limits = [
            (-0.3, -0.2),
            (1.8, 1.9),
            (0, 0.9),
        ]
    lower_bounds = [limit[0] for limit in limits]
    upper_bounds = [limit[1] for limit in limits]
    # default_vals = [(lower + upper) / 2 for lower, upper in zip(lower_bounds, upper_bounds)]
    # default_vals = np.array(default_vals)
    num_joints = len(actuated_joints)
    # interpolate between the lower and upper bounds
    interp_joints = []
    for i in range(len(lower_bounds)):
        lower, upper = lower_bounds[i], upper_bounds[i]
        print(lower, upper)
        values1 = np.linspace(lower, upper, n_steps)
        # then iterate from upper to lower again
        values2 = np.linspace(upper, lower, n_steps)
        joints = np.zeros((n_steps * 2, num_joints))
        joints[:, i] = np.concatenate([values1, values2])
        # fill other joints with the lower bound limit value:
        for j in range(num_joints):
            if j != i:
                joints[:, j] = lower_bounds[j]
        interp_joints.append(joints)

    # now interpolate the finger joints
    joints = np.zeros((n_steps * 2, num_joints))

    for i in range(6, num_joints):
        lower, upper = lower_bounds[i], upper_bounds[i]
        values1 = np.linspace(lower, upper, n_steps)
        # then iterate from upper to lower again
        values2 = np.linspace(upper, lower, n_steps)
        joints[:, i] = np.concatenate([values1, values2])

    interp_joints.append(joints)
    interp_joints = np.concatenate(interp_joints, axis=0) # shape (n_steps, n_joints)
    return interp_joints, act_idxs

def export_video(frames, path, fps=15):
    if len(frames) == 0:
        return
    path = path + ".mp4" if not path.endswith(".mp4") else path
    from moviepy.editor import ImageSequenceClip
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(path)

def gather_geom_link_groups(hand):
    # for each geom on the hand entity, get its link's global and local indx and name
    info = dict()
    for geom in hand.geoms:
        geom_id = geom.idx
        link = geom.link
        link_local_idx = link.idx_local
        link_name = link.name
        info[geom_id] = (link_local_idx, link_name)
    return info

def find_all_parents(link, all_links):
    """recursively find all the parents global_idx of a link"""
    parents = []
    # print('current link:', link.name, link.idx, link.parent_idx)
    # if link.name != 'base_link':
    #     breakpoint()
    if link.idx == -1 or link.parent_idx == -1: # root link for the entity has -1 parent
        return parents
    else:
        return find_all_parents(all_links[link.parent_idx], all_links) + [link.parent_idx]

def divide_hand_groups(hand, palm_link_name='base_link'):
    """
    for each hand, we have 2 kinds of collision groups for each geom
    1. palm link group: the geom's link local idx <= the palm link idx (assume the wrist & palm are the first few links)
    2. finger groups: each group has one link whose parent is the palm link (this should use the global idx for palm)
    """
    palm_links = [link for link in hand.links if link.name == palm_link_name]
    assert len(palm_links) == 1, f"Found {len(palm_links)} palm links"
    palm_idx_local = palm_links[0].idx_local
    palm_idx_global = palm_links[0].idx
    finger_groups = {geom.link.idx_local: 0 for geom in hand.geoms if geom.link.idx_local <= palm_idx_local} # {geom's link local idx: group_id}

    first_links = []
    for link in hand.links:
        if link.parent_idx == palm_idx_global:
            group_id = len(first_links) + 1
            first_links.append(
                (link.idx, link.idx_local, link.name, group_id)
            )
    print("Found first links:", first_links)
    all_links = hand.solver.links # this should be all global links!! stuck in infinite recursion if use only entity links
    for geom in hand.geoms:
        link = geom.link
        parent_global_idxs = find_all_parents(link, all_links)
        for first_link in first_links:
            idx, idx_local, name, group_id = first_link
            if idx in parent_global_idxs:
                finger_groups[geom.link.idx_local] = group_id
                break
    return finger_groups

def main(args):
    hands_list = args.hands_list
    hand_names = []
    for hand in hands_list:
        if '_hand' in hand or 'xhand' == hand:
            hand_names.append(hand)
        else:
            hand_names.append(hand + '_hand')
    num_hands = len(hand_names)
    urdfs = dict()
    if len(args.urdf_path) > 0:
        assert os.path.exists(args.urdf_path), f"URDF path {args.urdf_path} does not exist"
        hand = args.urdf_path.split('assets/')[-1].split('/')[0]
        hand += '_left' if 'left' in args.urdf_path else '_right'
        urdfs[hand] = args.urdf_path
        hand_names = [hand]
    else:
        for hand in hand_names:
            urdf_paths = glob(join(args.asset_path, hand, "*.urdf")) + glob(join(args.asset_path, hand, "*", "*.urdf")) + glob(join(args.asset_path, hand, "*", "*", "*.urdf"))
            if 'barrett' not in hand:
                urdf_paths = [path for path in urdf_paths if args.side in path]
            # if args.vis_6dof:
            #     # filter only fnames that contain 6dof wrist joints
            #     urdf_paths = [path for path in urdf_paths if '6dof' in path]
            # else:
            #     urdf_paths = [path for path in urdf_paths if '6dof' not in path]

            # urdf_paths = glob(join(args.asset_path, hand,  "*copy.urdf"))

            assert len(urdf_paths) >= 1, f"Found {len(urdf_paths)} URDFs for {hand} - {args.side}"
            _path = urdf_paths[0]
            urdfs[hand] = _path

    gs.init(backend=gs.gpu)
    scene_cfg = dict(
        sim_options=gs.options.SimOptions(
            dt=1/60,
            substeps=2,
            gravity=(0, 0, 0),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=True if (args.group_collisions or args.enable_self_collision) else False,
            enable_joint_limit=True,
            # enable_joint_limit=False, # disable this might lead to unreasonable joint movements!
        ),
        show_viewer=args.vis,
        use_visualizer=True,
        # show_FPS=True,
        # renderer=gs.renderers.RayTracer()
        vis_options = gs.options.VisOptions(
            plane_reflection = True,
            ambient_light    = (0.6, 0.6, 0.6),
            lights = [
                {"type": "directional", "dir": (0, 0, -1), "color": (1.0, 1.0, 1.0), "intensity": 8.0},
            ]
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.5, 1.5, 1.8), # looking from behind
            camera_lookat=(0.0, -0.15, 1.0),
            camera_fov=10,
        ),
    )
    if args.raytrace:
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

    if args.group_collisions:
        # add link grouping info to rigid options
        scene_cfg['rigid_options'].self_collision_group_filter = True
        if 'allegro' in str(args.urdf_path):
            mapping = {8: 0, 13: 1, 14: 2, 15: 3, 16: 4, 17: 1, 18: 2, 19: 3, 20: 4, 21: 1, 22: 2, 23: 3, 24: 4, 25: 1, 26: 2, 27: 3, 28: 4}
        elif 'schunk' in str(args.urdf_path):
            mapping = {7: 0, 8: 0, 13: 1, 14: 1, 15: 2, 16: 3, 17: 4, 18: 1, 19: 1, 20: 2, 21: 3, 22: 4, 23: 1, 24: 1, 25: 2, 26: 3, 27: 4, 28: 1, 29: 1}
        elif 'shadow' in str(args.urdf_path):
            mapping  = {7: 0, 8: 0, 9: 0, 17: 3, 18: 4, 19: 5, 20: 6, 21: 7, 22: 3, 23: 4, 24: 5, 25: 6, 26: 7, 27: 3, 28: 4, 29: 5, 30: 6, 31: 7, 35: 6, 36: 7}
        elif 'xhand' in str(args.urdf_path):
            mapping = {7: 0, 13: 1, 14: 2, 15: 3, 16: 4, 17: 5, 18: 1, 19: 2}
        elif 'ability' in str(args.urdf_path):
            mapping = {7: 0, 8: 0, 14: 1, 15: 2, 16: 3, 17: 4, 18: 5}
        elif 'inspire' in str(args.urdf_path):
            mapping = {7: 0, 13: 1, 14: 2, 15: 3, 16: 4, 17: 5, 18: 1, 23: 1}
        elif 'mimic' in str(args.urdf_path):
            mapping = {7: 0, 14: 2, 15: 3, 16: 4, 17: 5, 18: 6, 19: 2, 20: 3, 21: 4, 22: 5, 23: 6, 24: 2, 25: 3, 26: 4, 27: 5, 28: 6}
        else:
            raise NotImplementedError(f"Link grouping not implemented for {args.urdf_path}")
        scene_cfg['rigid_options'].link_group_mapping = mapping
    scene = gs.Scene(**scene_cfg)

    hand_entities = dict()
    for idx, name in enumerate(hand_names):
        urdf_path = urdfs[name]
        pos = np.array([idx * 0.6, 0.0, 1.0])
        hand = scene.add_entity(
            gs.morphs.URDF(
                    file=urdf_path,
                    pos=pos,
                    quat=(get_base_rotation(name) if not args.iterate_quat else (1, 0, 0, 0)),
                    fixed=True,
                    convexify=True,
                    merge_fixed_links=False, # NOTE: need to keep this to track the fingertip links
                    recompute_inertia=True,
                ),
                visualize_contact=True,
                vis_mode="collision" if args.vis_collision else "visual",
        )
        hand_entities[name] = hand
    plane = scene.add_entity(gs.morphs.URDF(pos=(0, 0, -0.5), file='urdf/plane/plane.urdf', fixed=True))

    floating_camera = scene.add_camera(
            # pos=(0.0, -1.2, 2.5), # side view
            pos=(1.0, 0.0, 1.2),
            lookat=(0.0, 0.0, 1.1),
            res=(1024, 1024),
            fov=30,
            GUI=False,
        )
    # interpolate the hand joints
    scene.build(n_envs=args.num_envs, env_spacing=(2.0, 2.0))
    scene.reset()
    if args.print_joints:# print joint names:
        for name, hand in hand_entities.items():
            all_joints = hand.joints
            actuated_joints = [joint for joint in all_joints if joint.type in [gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC]]
            print(f"Actuated joints for {name}:")
            for joint in actuated_joints:
                print(joint.name)
    if args.gather_geoms:
        for name, hand in hand_entities.items():
            print(f"Gathered geom-link groups for {name}:")
            # print(gather_geom_link_groups(hand))
            # base_link = 'base_link' # allegro
            base_link = args.base_link_name
            groups = divide_hand_groups(hand, base_link)
            for geom_id, group_id in groups.items():
                print(f"Geom's link {geom_id} in group {group_id}")
        print("Geom link groups for all hands:")
        print(groups)
        print("Link names for all collison links")
        print(
            [l.name for l in hand.links if len(l.geoms)>0]
        )
        breakpoint()

    n_steps = args.interp_steps
    # get all the actuated joints and their dof limits
    hand_joint_vals = dict()
    for hand_name, hand in hand_entities.items():
        if args.interp_joints_separately or args.wrist_only or args.wrist_rot_only:
            interp_joints, act_idxs = interpolate_wrist_finger_separately(
                hand, n_steps,
                skip_wrist=args.skip_wrist_interp, wrist_only=args.wrist_only, wrist_rot_only=args.wrist_rot_only
            )
        else:
            interp_joints, act_idxs = interpolate_hand_joints(hand, n_steps, skip_wrist=args.skip_wrist_interp)
        hand_joint_vals[hand_name] = (interp_joints, act_idxs)

    # set the joint values and step the scene
    step = 0
    max_steps = interp_joints.shape[0]
    frames = []
    video_path = f"rendered/hands/{hand_names[-1]}"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    frame_path = f"{video_path}/frames"
    if args.raytrace:
        frame_path = f"{video_path}/raytrace"
    os.makedirs(frame_path, exist_ok=True)
    quat_combos = None
    if args.iterate_quat:
        quat_combos = generate_90degree_rotation_quaternions()
    # set the camera z axis (both pos and lookat) to align with the hand's first link
    hand = hand_entities[hand_names[-1]]
    first_link = [link for link in hand.links if len(link.geoms) > 0][0]
    link_z = first_link.get_pos()[0][2] + 0.12
    if 'shadow' in hand_names[-1]:
        link_z = first_link.get_pos()[0][2] + 0.15
        floating_camera.set_params(fov=40)
    if 'allegro' in hand_names[-1]:
        link_z = first_link.get_pos()[0][2] + 0.05
    camera_pos = floating_camera.pos
    camera_pos[2] = link_z
    camera_lookat = floating_camera.lookat
    camera_lookat[2] = link_z
    floating_camera.set_pose(pos=camera_pos, lookat=camera_lookat)
    scene.step()
    while True:
        if args.record_video or (args.render_image and step == 0):
            img, rgb_frame = render_image(floating_camera, transparent=args.transparent)
            frames.append(rgb_frame)
            img_fname = join(frame_path, f"{hand_names[-1]}_{step:03d}.png")
            if args.iterate_quat:
                quat_str = f"{new_quat[0]:.2f}_{new_quat[1]:.2f}_{new_quat[2]:.2f}_{new_quat[3]:.2f}"
                img_fname = join(frame_path, f"{hand_names[-1]}_{step:03d}_{quat_str}.png")
            cv2.imwrite(img_fname, img)

        for name, hand in hand_entities.items():
            interp_joints, act_idxs = hand_joint_vals[name]
            jpos = torch.tensor(interp_joints[step], dtype=torch.float32)[None].repeat(args.num_envs, 1)
            # print(f"Setting joint positions for {name}: {jpos}")
            hand.set_dofs_position(
                position=jpos,
                dofs_idx_local=act_idxs,
                zero_velocity=True,
            )
            if args.iterate_quat:
                result = quat_combos[step % len(quat_combos)]
                new_quat = np.array(result['quaternion'])
                desp = result['description']
                hand.set_quat(new_quat[None])
                print(f"Setting quaternion for {name}: {new_quat} - {desp}")

        scene.step()

        step += 1
        if step == max_steps:
            step = 0
            if args.record_video:
                export_video(frames, join(video_path, f"{hand_names[-1]}_render.mp4"), fps=10)
                print(f"Exported video to {video_path}")
                exit()

    breakpoint()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis', '-v', action='store_true')
    parser.add_argument('--hand', type=str, default='inspire')
    # take in multiple hand names into a list of args
    parser.add_argument('--hands_list', type=str, nargs='+', default=['inspire'])
    parser.add_argument('--urdf_path', '-u', type=str, default='') # if provided, use this urdf path and vis only one hand
    parser.add_argument('--group_collisions', action='store_true')
    parser.add_argument('--side', type=str, default='left')
    parser.add_argument('--asset_path', type=str, default='/home/mandi/chiral/assets')
    parser.add_argument('--vis_collision', '-vc', action='store_true')
    parser.add_argument('--vis_6dof', action='store_true', help='Visualize the scene')
    parser.add_argument('--record_video', action='store_true', help='Render a video')
    parser.add_argument('--render_image', action='store_true', help='Render images')
    parser.add_argument('--raytrace', action='store_true', help='Use raytracer')
    parser.add_argument('--transparent', '-tr', action='store_true', help='Use transparent rendering')
    parser.add_argument('--interp_steps', type=int, default=50, help='Number of interpolation steps')
    parser.add_argument('--interp_joints_separately', action='store_true', help='Interpolate wrist and finger joints separately')
    parser.add_argument('--skip_wrist_interp', action='store_true', help='Skip interpolating wrist joints')
    parser.add_argument('--wrist_only', action='store_true', help='Only interpolate wrist joints')
    parser.add_argument('--wrist_rot_only', action='store_true', help='Only interpolate wrist rotation joints')
    parser.add_argument('--num_envs', type=int, default=1, help='Number of environments to build')
    parser.add_argument('--print_joints', action='store_true', help='Print joint names')
    parser.add_argument('--gather_geoms', action='store_true', help='Gather geoms')
    parser.add_argument('--base_link_name', type=str, default='base_link', help='Name of the base link')
    parser.add_argument('--enable_self_collision', action='store_true', help='Enable selfcollision')
    parser.add_argument('--iterate_quat', action='store_true', help='Iterate through quaternion combinations')
    args = parser.parse_args()

    main(args)