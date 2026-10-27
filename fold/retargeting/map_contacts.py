import os
import cv2
import yaml
import numpy as np
import pickle
import inspect
import xml.etree.ElementTree as ET

from os.path import join
import genesis as gs
import torch

import argparse
from sklearn.neighbors import KDTree
from copy import deepcopy
from collections import defaultdict
from fold.envs.object import ArticulatedObject, get_arctic_object_cfg

from fold.asset_utils import get_asset_path
"""

python retargeting/map_contacts.py --hand allegro_hand --show_object  --num_markers 100  --record_video # --raytrace

Go from mesh vertice contacts (on object surface) to robot hand links
- use "contacts.left", "valid_contacts.left": shape (T, N=50, 4)
- use raw retargeted hand poses, no object no collision, and use AABB to approximate center positions for all collision links
- for each raw contact pos: find the closest link center pos
- for each link: find all the matched contact points -> average them?

Render only grouped contacts with raytracing
python retargeting/map_contacts.py --hand ${HAND} --record_video  --show_object  --num_markers 15 --load_fname assets/arctic/processed/${job} --render_only --raytrace --show_grouped_contact_only
"""


def infer_object_name_from_sequence_path(load_fname):
    """Recover object names, including names containing underscores, from *_use_XX.npy."""
    sequence_stem = os.path.splitext(os.path.basename(load_fname))[0]
    if "_use_" in sequence_stem:
        return sequence_stem.rsplit("_use_", 1)[0]
    return sequence_stem.split("_", 1)[0]


def get_retargeter_result_path(load_fname, hand_name, subject_name, retarget_type, retarget_name=None):
    suffix = f"_{retarget_type}"
    if retarget_name:
        suffix += f"_{retarget_name}"
    filename = os.path.basename(load_fname).replace(".npy", f"{suffix}.npy")
    return join(get_asset_path("retargeter_results"), hand_name, subject_name, filename)

def show_contact_plt(contact_links):
    # contact_links: shape (T, num_mano_links, 4) assign each link a color and show all frames
    import matplotlib
    matplotlib.use('tkAgg')
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # set fixed xyz limits
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.set_zlim(0.8, 1.4)
    scatter = None
    num_links = contact_links.shape[1]
    colors = plt.cm.jet(np.linspace(0, 1, num_links))

    t = 0
    max_t = contact_links.shape[0]
    while True:
        points = contact_links[t, :, :3]
        sizes = 50.0 * np.ones(points.shape[0])
        if scatter is None:
            scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=sizes)
        else:
            scatter._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
            scatter.set_color(colors)
            scatter.set_sizes(sizes)
        plt.pause(0.05)
        t += 1
        if t == max_t:
            t = 0

def render_transparent_img(cam):
    img, _, seg_arr, _ = cam.render(segmentation=True)
    rgb_img = img.copy() # do this BEFOFE
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # do this first!! do channel after
    max_id = seg_arr.max() # assuming this is background
    channel = np.ones(img.shape[0:2], dtype=np.uint8) * 255
    channel[seg_arr == max_id] = 0
    # make a 3 channel rgb image but the background is all white

    rgb_img[seg_arr == max_id] = 255
    img = np.concatenate([img, channel[:, :, None]], axis=-1)
    return img, rgb_img

def ensure_urdf_inertial_compat(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is not None:
            link.remove(inertial)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(inertial, "mass", value="1e-6")
        ET.SubElement(
            inertial,
            "inertia",
            ixx="1e-9",
            ixy="0",
            ixz="0",
            iyy="1e-9",
            iyz="0",
            izz="1e-9",
        )
    compat_path = urdf_path.replace('.urdf', '_genesis_compat.urdf')
    tree.write(compat_path, encoding='utf-8')
    return compat_path

def patch_entity_inertial_fields(entity):
    for link in entity.links:
        if getattr(link, "inertial_quat", None) is None:
            link._inertial_quat = np.array([1.0, 0.0, 0.0, 0.0])
        if getattr(link, "inertial_pos", None) is None:
            link._inertial_pos = np.array([0.0, 0.0, 0.0])
        if getattr(link, "inertial_mass", None) is None:
            link._inertial_mass = 1e-6
        if getattr(link, "inertial_i", None) is None:
            link._inertial_i = np.eye(3) * 1e-9

def create_scene(args, object_name, urdfs, num_raw_contact_markers=50, num_grouped_contact_markers=50):
    import genesis as gs
    gs.init(backend=gs.gpu)
    scene_cfg = dict(
        sim_options=gs.options.SimOptions(
            dt=1/60,
            substeps=2,
            gravity=(0, 0,0),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=False,
            enable_joint_limit=True,
        ),
        show_viewer=args.vis_scene,
        use_visualizer=(args.vis_scene or args.record_video),
        show_FPS=False,
        vis_options = gs.options.VisOptions(
            plane_reflection = True,
            ambient_light    = (0.4, 0.4, 0.4),
            lights = [
                {"type": "directional", "dir": (0, 0, -1), "color": (1.0, 1.0, 1.0), "intensity": 2.0},
            ]
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 0.8, 2.1),
            camera_lookat=(0.0, -0.1, 1.1),
            camera_fov=25,
        ),
    )
    plane_urdf = 'urdf/plane/plane.urdf' # NOTE this is Genesis default plane
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
    scene_init_params = inspect.signature(gs.Scene.__init__).parameters
    scene_cfg = {k: v for k, v in scene_cfg.items() if k in scene_init_params}
    scene = gs.Scene(**scene_cfg)
    device = torch.device('cuda:0')

    cam = None
    if args.record_video:
        if args.raytrace:
            cam = scene.add_camera(
            pos=scene_cfg['viewer_options'].camera_pos, lookat=scene_cfg['viewer_options'].camera_lookat,
            res=(1024, 1024), fov=20, GUI=False)
        else:
            cam = scene.add_camera(
            pos=scene_cfg['viewer_options'].camera_pos, lookat=scene_cfg['viewer_options'].camera_lookat,
            res=(512, 512), fov=20, GUI=False)

    hand_entities = dict()
    for side, urdf_path in urdfs.items():
        compat_urdf_path = ensure_urdf_inertial_compat(urdf_path)
        hand = scene.add_entity(
            gs.morphs.URDF(
                file=compat_urdf_path,
                fixed=True,
                merge_fixed_links=False,
                recompute_inertia=True,
                collision=True, # has to be true for get_AABB to work
                # collision=False,
            ),
            material=gs.materials.Rigid(
                gravity_compensation=0.8
                ),
            # surface=gs.surfaces.Smooth(color=(0, 0, 0.8, 0.5)),
        )
        patch_entity_inertial_fields(hand)
        hand_entities[side] = hand
    obj = None
    if args.show_object:
        obj_cfg = get_arctic_object_cfg(object_name)
        obj_cfg['fixed'] = False
        obj_cfg['disable_collision'] = False
        obj_cfg['color'] = (1.0, 0.423, 0.039, 0.3)
        obj = ArticulatedObject(obj_cfg, device=device, scene=scene, num_envs=1)
    markers = dict()
    if args.show_grouped_contact_only:
        num_raw_contact_markers = 0
    for palette, marker_type, num_markers in zip(['rocket', 'crest'],['raw', 'grouped'], [num_raw_contact_markers, num_grouped_contact_markers]):
        if num_markers > 0:
            import seaborn as sns
            marker_colors = sns.color_palette(palette, 2)
            marker_colors = np.array(marker_colors)

            for i, part in enumerate(['top', 'bottom']):
                color = marker_colors[i]

                marker_ents = [
                    scene.add_entity(
                        gs.morphs.Sphere(
                            radius=0.008 if marker_type == 'raw' else 0.015,
                            fixed=False,
                            collision=False),
                            surface=gs.surfaces.Smooth(color=color),
                            ) for _ in range(num_markers)
                ]
                markers[f"{marker_type}_{part}"] = marker_ents
    # add this last for segmentation to work
    ground = scene.add_entity(gs.morphs.URDF(file=plane_urdf, fixed=True))
    patch_entity_inertial_fields(ground)
    scene.build(n_envs=1, env_spacing=(2.0, 2.0))
    return scene, hand_entities, markers, obj, cam

def show_hand_joints_links_plt(hand_entites):
    import matplotlib
    matplotlib.use('tkAgg')
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # plot two subplots
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # set fixed xyz limits
    for ax in [ax1, ax2]:
        ax.set_xlim(-0.1, 0.1)
        ax.set_ylim(-0.1, 0.1)
        ax.set_zlim(-0.1, 0.1)
    scatter = None

    for side, hand in hand_entities.items():
        link_pos = np.array([link.pos for link in hand.links])
        link_names = [link.name for link in hand.links]
        joint_pos = np.array([joint.pos for joint in hand.joints if 'forearm' not in joint.name])
        joint_names = [joint.name for joint in hand.joints if 'forearm' not in joint.name]
        ax = ax1 if side == 'left' else ax2
        link_colors = plt.cm.summer(np.linspace(0, 1, len(link_pos)))
        joint_colors = plt.cm.spring(np.linspace(0, 1, len(joint_pos)))
        sizes = 100.0 * np.ones(len(link_pos))
        joint_sizes = 200.0 * np.ones(len(joint_pos))
        positions = np.concatenate([link_pos, joint_pos], axis=0)
        colors = np.concatenate([link_colors, joint_colors], axis=0)
        sizes = np.concatenate([sizes, joint_sizes], axis=0)
        labels = link_names + joint_names
        scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colors, s=sizes)
        # use annotate
        for i, label in enumerate(labels):
            ax.text(positions[i, 0], positions[i, 1], positions[i, 2], label, size=8, zorder=1)
    plt.show()

def show_hand_kpts_scene(scene, hand_entites, markers):
    marker_offset = 0
    for side, hand in hand_entities.items():
        # link_pos = np.array([link.pos for link in hand.links])
        link_pos = np.array([link.inertial_pos + link.pos for link in hand.links if link.geoms])
        aabbs = [link.get_AABB()[0] for link in hand.links if link.geoms] #each is (2, 3)
        aabb_centers = [ 0.5 * (aabb[0] + aabb[1]) for aabb in aabbs]
        # link_names = [link.name for link in hand.links]
        joint_pos = np.array([joint.pos for joint in hand.joints if 'forearm' not in joint.name])
        joint_names = [joint.name for joint in hand.joints if 'forearm' not in joint.name]

        # positions = np.concatenate([link_pos, joint_pos], axis=0)
        positions = aabb_centers
        for i, pos in enumerate(positions):
            idx = marker_offset + i
            if idx >= len(markers):
                break
            markers[idx].set_pos(pos[None])
        marker_offset += len(positions)
    scene.step()
    breakpoint()

def group_contacts(links, raw_contacts, valids, num_obj_parts=2):
    """
    Grouping per-step contacts, input:
    - links: raw contacts shape (N=50, 4)
    NOTE in ARCTIC, part_id=2 is 'bottom' link, part_id=1 is 'top'
    returns:
    - grouped_contacts: shape (num_obj_parts=2, num_dex_links, 4)
    - grouped_valids: shape (num_obj_parts=2, num_dex_links)
    -> note here that to be consistent with environment contact readings, need a separate set of contacts for each obj part
    - target_positions: shape (N=50, 3) target positions for each raw contacts after grouping
    """
    aabbs = [link.get_AABB()[0].cpu().numpy() for link in links] # each is (2, 3)
    link_center_pos = np.array([0.5 * (aabb[0] + aabb[1]) for aabb in aabbs])
    # now for each raw_contact, find the closest link center pos
    kdtree = KDTree(link_center_pos)
    positions = raw_contacts[:, :3]
    distances, indices = kdtree.query(positions, k=1)
    # for the invalid raw contacts, set the index to -1
    indices[~valids] = -1
    nlinks = len(links)
    # now for each link, find all the matched contact points
    grouped_contacts = np.zeros((num_obj_parts, nlinks, 4))
    grouped_valids = np.zeros((num_obj_parts, nlinks,))
    target_positions = np.zeros(positions.shape)
    for i in range(len(links)):
        for j in range(num_obj_parts):
            pidx = j + 1 # valid index should be 1 or 2, NOT 0
            part_match = raw_contacts[:, 3] == pidx # (50,)
            has_valid = (indices == i).flatten() # (50,1)->(50,)
            part_valid = part_match & has_valid
            if np.sum(part_valid) > 0:
                mean_pos = np.mean(positions[part_valid], axis=0)
                grouped_contacts[j, i, :3] = mean_pos
                # part_ids = raw_contacts[part_valid, 3]
                # voted_part_id = np.argmax(np.bincount(part_ids.astype(int)))
                grouped_contacts[j, i, 3] = pidx # no voting needed
                grouped_valids[j, i] = 1
                target_positions[has_valid] = mean_pos
    return grouped_contacts, grouped_valids, target_positions

def set_entities_to_step(hand_entities, retargeter_results, step):
    for side, hand in hand_entities.items():
        hand_qpos = retargeter_results[side]["hand_qpos"][step]
        hand_qpos = torch.tensor(hand_qpos).to(device)[None]
        joint_idxs = [joint.dof_idx_local for joint in hand.joints if joint.type in [gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC]]
        hand.set_dofs_position(position=hand_qpos, dofs_idx_local=joint_idxs)
    return

def control_entities_to_step(hand_entities, retargeter_results, step):
    for side, hand in hand_entities.items():
        hand_qpos = retargeter_results[side]["hand_qpos"][step]
        hand_qpos = torch.tensor(hand_qpos).to(device)[None]
        joint_idxs = [joint.dof_idx_local for joint in hand.joints if joint.type in [gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC]]
        hand.control_dofs_position(hand_qpos, dofs_idx_local=joint_idxs)
    return

def set_object_to_step(obj, obj_states, step):
    obj.set_object_state(
        root_pos=obj_states["root_pos"][step][None],
        root_quat=obj_states["root_quat"][step][None],
        joint_qpos=obj_states["joint_qpos"][step][None],
    )
    return

def visualize_markers(markers_dict, raw_contacts, grouped_contacts):
    # first set all markers to 0! avoid delays in vis
    for k, markers in markers_dict.items():
        for marker in markers:
            marker.set_pos(np.zeros((1, 3)))
    for i, part in enumerate(['top', 'bottom']):
        pid = i + 1
        raw_markers = markers_dict.get(f"raw_{part}", [])
        part_mask = raw_contacts[:, 3] == pid
        raw_pos = raw_contacts[part_mask, :3]
        for k, pos in enumerate(raw_pos):
            if k >= len(raw_markers):
                break
            raw_markers[k].set_pos(pos[None])
        grouped_markers = markers_dict.get(f"grouped_{part}", [])
        grouped_pos = []
        for side, contacts in grouped_contacts.items():
            # contacts are shaped (2, num_links, 4)
            grouped_pos.append(contacts[i, :, :3])
        grouped_pos = np.concatenate(grouped_pos, axis=0)
        for k, pos in enumerate(grouped_pos):
            if k >= len(grouped_markers):
                break
            grouped_markers[k].set_pos(pos[None])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hand', type=str, default='xhand')
    parser.add_argument('--load_fname', '-lf', type=str, default='/home/mandi/chiral/assets/arctic/processed/s01/box_use_01.npy')
    parser.add_argument('--retarget_name', type=str, default=None,
                        help='Optional retargeter result suffix, for example temporal_v1')
    parser.add_argument('--save_dir', type=str, default='contact_retarget')
    parser.add_argument('--show_mano_plt', action='store_true', help='Whether to show the matplotlib plot')
    parser.add_argument('--show_hand_links', action='store_true', help='Whether to show the hand links')
    parser.add_argument('--vis_scene', '-v', action='store_true', help='Whether to visualize the scene')
    parser.add_argument('--show_object', action='store_true', help='Whether to show the object')
    parser.add_argument('--num_markers', type=int, default=0, help='Number of markers to show')
    parser.add_argument('--record_video', action='store_true', help='Whether to record video')
    parser.add_argument('--raytrace', action='store_true', help='Whether to use raytracer')
    parser.add_argument('--render_only', action='store_true', help='if true, skip saving data')
    parser.add_argument('--show_grouped_contact_only', action='store_true')
    parser.add_argument('--physics_rollout', action='store_true', help='Use physics-based rollout instead of teleporting states')
    args = parser.parse_args()

    assert os.path.exists(args.load_fname), f"load_fname={args.load_fname} does not exist"
    subject_name = args.load_fname.split("/")[-2]
    hand_name = args.hand if 'hand' in args.hand else f"{args.hand}_hand"
    retarget_type = 'position' if hand_name == 'shadow_hand' else 'vector'
    retarget_fname = get_retargeter_result_path(
        args.load_fname, hand_name, subject_name, retarget_type, args.retarget_name
    )
    assert os.path.exists(retarget_fname), f"retarget_fname={retarget_fname} does not exist"

    retargeter_results = np.load(retarget_fname, allow_pickle=True).item()
    object_name = infer_object_name_from_sequence_path(args.load_fname)

    full_save_dir = get_asset_path(args.save_dir)
    save_path = os.path.join(full_save_dir, hand_name, subject_name)
    os.makedirs(save_path, exist_ok=True)
    if args.record_video:
        frame_path = os.path.join(save_path, f"frames_{object_name}")
        os.makedirs(frame_path, exist_ok=True)
    save_fname = os.path.join(save_path, args.load_fname.split("/")[-1])
    loaded_data = np.load(args.load_fname, allow_pickle=True).item()

    obj_states = {
        "root_pos": loaded_data['params']['obj_trans'],
        "root_quat": loaded_data['params']['obj_quat'],
        "joint_qpos": loaded_data['params']['obj_arti'],
    }

    if args.show_mano_plt:
        contacts_left = loaded_data['world_coord']["contact_links_left"]
        contacts_right = loaded_data['world_coord']["contact_links_right"]
        show_contact_plt(
            np.concatenate([contacts_left, contacts_right], axis=1)
        )
        breakpoint()


    urdfs = dict()
    robot_dir = get_asset_path(args.hand)
    config_path = join(robot_dir, "retarget_config.yaml")
    for side in ['left', 'right']:
        config = yaml.safe_load(open(config_path, 'r'))
        urdf_path = config[side]['urdf_path']
        urdfs[side] = join(robot_dir, urdf_path)

    # use genesis to create hand entities
    scene, hand_entities, markers, obj, cam = create_scene(
        args, object_name, urdfs,
        num_raw_contact_markers=args.num_markers, num_grouped_contact_markers=args.num_markers,
    )
    device = torch.device('cuda:0')
    if args.show_hand_links:
        # show the hand links and joints
        # show_hand_joints_links_plt(hand_entities)
        show_hand_kpts_scene(scene, hand_entities, markers)

    collision_links = dict()
    for side, hand in hand_entities.items():
        links = [link for link in hand.links if link.geoms]
        collision_links[side] = links

    num_steps = retargeter_results['left']["hand_qpos"].shape[0]
    step = 0
    frames = []
    saved_vid = False
    clip_name = args.load_fname.split('/')[-1].replace('.npy', "")
    tosave = dict(
        left=defaultdict(list),
        right=defaultdict(list),
    )
    saved_contacts = False
    if args.physics_rollout and args.show_object:
        set_object_to_step(obj, obj_states, 0)
    while True:
        if args.physics_rollout:
            control_entities_to_step(hand_entities, retargeter_results, step)
            if args.show_object and step == 0:
                set_object_to_step(obj, obj_states, 0)
        else:
            set_entities_to_step(hand_entities, retargeter_results, step)
            if args.show_object:
                set_object_to_step(obj, obj_states, step)

        # raw_contacts = [loaded_data['world_coord'][f"contacts.{side}"][step] for side in hand_entities.keys()]
        # raw_positions = np.concatenate(raw_contacts, axis=0)[:, :3]

        grouped_contacts = dict()
        grouped_valids = dict()
        target_pos = dict()
        for side, hand in hand_entities.items():
            raw_contacts = loaded_data['world_coord'][f"contacts.{side}"]
            valid_contacts = loaded_data['world_coord'][f"valid_contacts.{side}"]
            hand_link_contacts, hand_link_valids, target_positions = group_contacts(
                collision_links[side], raw_contacts[step], valid_contacts[step]
            )
            grouped_contacts[side] = hand_link_contacts # shape (num_obj_parts, num_dex_links, 4)
            grouped_valids[side] = hand_link_valids
            tosave[side]['dexlink_contacts'].append(hand_link_contacts)
            tosave[side]['dexlink_valid_contacts'].append(hand_link_valids)
            target_pos[side] = target_positions

        if args.num_markers > 0:
            all_raw_contacts = np.concatenate([loaded_data['world_coord'][f"contacts.{side}"][step] for side in hand_entities.keys()], axis=0)
            visualize_markers(markers, all_raw_contacts, grouped_contacts)

            # set_entities_to_step(hand_entities, retargeter_results, step)
            # set_object_to_step(obj, obj_states, step)
        scene.step()
        if args.record_video and not saved_vid:
            if args.raytrace:
                # render segmentation
                transparent_img, white_bg_img = render_transparent_img(cam)
                img_fname = os.path.join(frame_path, f"{clip_name}_{step:04d}.png")
                cv2.imwrite(img_fname, transparent_img)
                frames.append(white_bg_img)
            else:
                img, _, _, _ = cam.render()
                frames.append(img)
            # os.makedirs(os.path.join(save_path, clip_name), exist_ok=True)
            # fname = os.path.join(save_path, clip_name, f"{step:04d}.png")
            # import cv2
            # cv2.imwrite(fname, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        step += 1
        if step >= num_steps:
            step = 0
            if not saved_contacts:
                for side, data in tosave.items():
                    for key, val in data.items():
                        val = np.stack(val, axis=0)
                        tosave[side][key] = val
                        print(f"Key={key}, val.shape={val.shape}")
                for side, links in collision_links.items():
                    link_names = [link.name for link in links]
                    link_local_idxs = [link.idx_local for link in links]
                    tosave[side]['collision_link_names'] = link_names
                    tosave[side]['collision_link_local_idxs'] = link_local_idxs

                if not args.render_only:
                    np.save(save_fname, tosave)
                    print(f"Saved contacts to {save_fname}")
                if not args.record_video:
                    break


            if args.record_video and not saved_vid:
                video_fname = os.path.join(save_path, f"{args.load_fname.split('/')[-1].replace('.npy', '.mp4')}")
                try:
                    from moviepy import ImageSequenceClip
                except ImportError:
                    from moviepy.editor import ImageSequenceClip
                clip = ImageSequenceClip(frames, fps=15)
                clip.write_videofile(video_fname)
                print(f"Saved video to {video_fname}")
                saved_vid = True
                break
    exit()
