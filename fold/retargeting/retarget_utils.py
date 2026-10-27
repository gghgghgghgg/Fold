import os
import torch
import numpy as np
from lxml import etree
from scipy.spatial.transform import Rotation
from copy import deepcopy
from collections import defaultdict

from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.kinematics_adaptor import KinematicAdaptor, MimicJointKinematicAdaptor
from fold.retargeting.temporal import repair_trajectory, trajectory_metrics

def get_link_names(urdf_path):
    assert os.path.exists(urdf_path), f"Does not exist: {urdf_path}"
    lxml_parser = etree.XMLParser(remove_comments=True, remove_blank_text=True)
    tree = etree.parse(urdf_path, parser=lxml_parser)
    robot_elem = tree.getroot()
    assert robot_elem.tag == "robot", "The first element should be a robot element."
    links = []
    for link_elem in robot_elem.findall("link"):
        if link_elem.find("collision") is not None and link_elem.attrib.get("name", None) is not None:
            links.append(link_elem.attrib["name"])
    return sorted(links)

def compose_retarget_config(
        input,
        retarget_type="vector",
        low_pass_alpha=1.0,
        scaling_factor=1.0,
        add_dummy_free_joint=False,
        ignore_mimic_joint=False,
        include_palm_vectors=False,
    ):
    """ write retargeting config given input type and offline cfgs"""
    assert retarget_type in ["position", "vector"], f"Unsupported: {retarget_type}"
    urdf_path = input.get("urdf_path", None)
    assert urdf_path is not None, f"Need input urdf path"

    finger_links = input.get('target_finger_links', [])
    human_finger_idxs = input.get('human_finger_idxs', [])
    assert len(finger_links) == len(human_finger_idxs) and len(finger_links) > 0
    origin_link = input.get('target_origin_link', None)
    human_origin_idx = input.get('human_origin_idx', None)
    assert origin_link is not None and human_origin_idx is not None
    origin_link = str(origin_link)
    human_origin_idx = int(human_origin_idx)
    cfg = dict(
        type=str(retarget_type),
        urdf_path=urdf_path,
        low_pass_alpha=low_pass_alpha,
        scaling_factor=scaling_factor,
        add_dummy_free_joint=add_dummy_free_joint,
        ignore_mimic_joint=ignore_mimic_joint,
    )

    if retarget_type == "position":
        cfg['target_link_names'] = finger_links #+ [origin_link]
        cfg['target_link_human_indices'] = [int(idx) for idx in human_finger_idxs] #+ [int(human_origin_idx)]
    elif retarget_type == "vector":
        if include_palm_vectors:
            palm_links = input.get('target_palm_links', [])
            human_palm_idxs = input.get('human_palm_idxs', [])
            if len(palm_links) != len(human_palm_idxs) or len(palm_links) < 2:
                raise ValueError(
                    "Palm-vector retargeting requires at least two matching "
                    "target_palm_links and human_palm_idxs"
                )
            finger_links = finger_links + palm_links
            human_finger_idxs = human_finger_idxs + human_palm_idxs
        num_fingers = len(finger_links)
        cfg['target_origin_link_names'] = [origin_link] * num_fingers
        cfg['target_task_link_names'] = finger_links
        cfg['target_link_human_indices'] = [
            [int(human_origin_idx) for _ in range(num_fingers)],
            [int(idx) for idx in human_finger_idxs]
        ]
    target_joints = input.get("target_joint_names", [])
    if len(target_joints) > 0:
        cfg['target_joint_names'] = target_joints
    return cfg


def install_consistent_temporal_regularizer(retargeter):
    """Make dex-retargeting's previous-qpos regularizer objective-consistent.

    dex-retargeting 0.5.0 adds the regularizer gradient without adding its
    scalar value to the SLSQP objective.  Wrap the instance locally instead of
    editing the environment's site-packages copy.
    """
    optimizer = retargeter.optimizer
    original_factory = optimizer.get_objective_function
    norm_delta = float(getattr(optimizer, 'norm_delta', 0.0))
    if norm_delta <= 0.0:
        return retargeter

    def consistent_factory(ref_value, fixed_qpos, last_qpos):
        base_objective = original_factory(ref_value, fixed_qpos, last_qpos)
        last_qpos_array = np.asarray(last_qpos, dtype=np.float64)

        def objective(x, grad):
            saved_delta = optimizer.norm_delta
            optimizer.norm_delta = 0.0
            try:
                value = base_objective(x, grad)
            finally:
                optimizer.norm_delta = saved_delta
            delta = np.asarray(x, dtype=np.float64) - last_qpos_array
            if grad.size > 0:
                grad[:] += 2.0 * norm_delta * delta
            return float(value + norm_delta * np.dot(delta, delta))

        return objective

    optimizer.get_objective_function = consistent_factory
    return retargeter


def align_wrist_orientation_to_demo(
        qpos,
        actuated_dof_names,
        wrist_quat_wxyz,
        anchor_frame=20,
    ):
    """Transfer MANO relative wrist rotation onto the robot wrist chain.

    The constant MANO-to-robot frame offset is calibrated at a stable anchor
    frame.  This constrains relative motion without assuming that MANO and the
    Inspire Hand URDF share the same local palm axes.
    """
    qpos = np.asarray(qpos, dtype=np.float64).copy()
    wrist_quat_wxyz = np.asarray(wrist_quat_wxyz, dtype=np.float64)
    if len(qpos) != len(wrist_quat_wxyz):
        raise ValueError(
            f"qpos/quaternion length mismatch: {len(qpos)} vs {len(wrist_quat_wxyz)}"
        )
    anchor_frame = int(np.clip(anchor_frame, 0, len(qpos) - 1))
    rotation_names = [
        name for name in actuated_dof_names
        if any(axis in name for axis in ('forearm_roll', 'forearm_pitch', 'forearm_yaw'))
    ]
    if len(rotation_names) != 3:
        raise ValueError(f"Expected three forearm rotation joints, got {rotation_names}")
    rotation_columns = [actuated_dof_names.index(name) for name in rotation_names]
    robot_angles = qpos[:, rotation_columns]

    def robot_rotation(angles):
        roll, pitch, yaw = angles
        return (
            Rotation.from_rotvec([0.0, 0.0, roll])
            * Rotation.from_rotvec([pitch, 0.0, 0.0])
            * Rotation.from_rotvec([0.0, -yaw, 0.0])
        )

    robot_anchor = robot_rotation(robot_angles[anchor_frame])
    mano_rotation = Rotation.from_quat(wrist_quat_wxyz[:, [1, 2, 3, 0]])
    mano_to_robot = mano_rotation[anchor_frame].inv() * robot_anchor
    desired_rotation = mano_rotation * mano_to_robot
    desired_zxy = desired_rotation.as_euler('ZXY')
    desired_angles = np.stack(
        [desired_zxy[:, 0], desired_zxy[:, 1], -desired_zxy[:, 2]], axis=1
    )
    desired_angles = np.unwrap(desired_angles, axis=0)
    desired_angles += 2.0 * np.pi * np.round(
        (robot_angles[anchor_frame] - desired_angles[anchor_frame]) / (2.0 * np.pi)
    )
    qpos[:, rotation_columns] = desired_angles
    return qpos.astype(np.float32), {
        'anchor_frame': anchor_frame,
        'rotation_joint_names': rotation_names,
    }


def refine_fingers_with_fixed_wrist(
        qpos,
        actuated_dof_names,
        joint_pos_sequence,
        finger_retargeter,
    ):
    """Re-optimize finger joints after imposing the MANO wrist trajectory."""
    qpos = np.asarray(qpos, dtype=np.float32).copy()
    robot_joint_names = finger_retargeter.optimizer.robot.dof_joint_names
    initial_robot_qpos = finger_retargeter.get_qpos()
    for name, value in zip(actuated_dof_names, qpos[0]):
        if name in robot_joint_names:
            initial_robot_qpos[robot_joint_names.index(name)] = value
    finger_retargeter.set_qpos(initial_robot_qpos)

    fixed_names = finger_retargeter.optimizer.fixed_joint_names
    fixed_columns = [actuated_dof_names.index(name) for name in fixed_names]
    indices = finger_retargeter.optimizer.target_link_human_indices
    for step, joint_pos in enumerate(joint_pos_sequence):
        ref_value = get_ref_val(joint_pos, indices)
        robot_qpos = finger_retargeter.retarget(
            ref_value,
            fixed_qpos=qpos[step, fixed_columns],
        )
        for column, name in enumerate(actuated_dof_names):
            if 'forearm' not in name and name in robot_joint_names:
                qpos[step, column] = robot_qpos[robot_joint_names.index(name)]
    return qpos

def get_ref_val(joint_pos, indices):
    if len(indices.shape) > 1:
        origin_ind = indices[0, :]
        task_ind = indices[1, :]
        return joint_pos[task_ind, :] - joint_pos[origin_ind, :]
    else: # shape is (num_joints,)
        return joint_pos[indices]

def get_demo_obj_tensors(loaded_data, step, device):
    obj_trans = loaded_data['params']['obj_trans'][step]
    obj_rot = loaded_data['params']['obj_quat'][step]
    obj_arti = loaded_data['params']['obj_arti'][step]
    obj_trans = torch.tensor(obj_trans, dtype=torch.float32, device=device)
    obj_rot = torch.tensor(obj_rot, dtype=torch.float32, device=device)
    obj_arti = torch.tensor(obj_arti, dtype=torch.float32, device=device)
    return obj_trans, obj_rot, obj_arti


def retarget_one_hand(
    wrist_pos,
    retarget_type,
    ref_value,
    retargeter,
    hand_init_qpos,
    actuated_dof_names,
    actuated_dof_idxs
):

    if retarget_type == "position":
        wrist_pos *= 0.0
    # wrist_pos *= 0.0
    qpos_retargeted = retargeter.retarget(
        ref_value,
        fixed_qpos=np.zeros(len(retargeter.optimizer.fixed_joint_names))
    )

    optimizer_joint_names = retargeter.optimizer.robot.dof_joint_names
    idx_pin2target = retargeter.optimizer.idx_pin2target.tolist()
    if isinstance(retargeter.optimizer.adaptor, MimicJointKinematicAdaptor):
        idx_pin2target.extend(retargeter.optimizer.adaptor.idx_pin2mimic.tolist())
    joint_vals = dict()
    # hand_qpos = hand.init_qpos.clone() * 0.0 # tensor of shape (num_joints,)
    hand_qpos = hand_init_qpos.clone()
    wrist_idxs = []
    wrist_qpos = []
    for idx in idx_pin2target:
        val = qpos_retargeted[idx]
        jname = optimizer_joint_names[idx]
        target_jname = jname
        if 'tx' in jname or "_x" in jname:
            val += wrist_pos[0]
        elif 'ty' in jname or "_y" in jname:
            val += wrist_pos[1]
        elif 'tz' in jname or "_z" in jname: # for mano hand
            val += wrist_pos[2]
        joint_idx = actuated_dof_names.index(target_jname)
        hand_qpos[:, joint_idx] = val
        joint_vals[target_jname] = val
        if 'forearm' in target_jname: # wrist qpos
            wrist_qpos.append(val)
            wrist_idxs.append(actuated_dof_idxs[joint_idx])
    return joint_vals, hand_qpos, wrist_qpos, wrist_idxs

def control_hand(hand, hand_qpos, wrist_qpos, wrist_idxs, set_finger=False, set_wrist=False):
    """
    NOTE the hand here is genesis entity
    """
    hand_qpos = torch.tensor(hand_qpos).to(hand.init_qpos.device)
    if set_finger and set_wrist:
        hand.set_joint_position(hand_qpos)
    elif set_wrist:
        hand.set_joint_position(
            torch.tensor(wrist_qpos)[None].to(hand_qpos.device),
            joint_idxs=wrist_idxs,
        )
    else:
        # pass
        hand.control_joint_position(hand_qpos)
    return hand_qpos

def retarget_all_steps(
    dof_limits, # (lower, upper) tuple
    hand_init_qpos,
    actuated_dof_names,
    actuated_dof_idxs,
    retargeter,
    num_steps,
    joint_pos_demo,
    retarget_type,
    frame_start=0,
    temporal_config=None,
    wrist_quat_demo=None,
    wrist_orientation_anchor=20,
    finger_retargeter=None,
):
    """ do retarget only once, since the retargeter optimizer seems to have some non-deterministic behavior """
    # hand_init_qpos = hand.init_qpos.clone()
    # actuated_dof_names = hand.actuated_dof_names
    # actuated_dof_idxs = hand.actuated_dof_idxs
    joint_pos_sequence = None
    if temporal_config is not None:
        joint_pos_sequence = np.asarray(
            joint_pos_demo[frame_start:frame_start + num_steps], dtype=np.float64
        )
        flat_joint_pos = joint_pos_sequence.reshape(num_steps, -1)
        joint_pos_sequence = repair_trajectory(
            flat_joint_pos, config=temporal_config
        ).reshape(joint_pos_sequence.shape)

    result_keys = ['hand_qpos', 'wrist_qpos', 'wrist_idxs']
    all_ret = {key: [] for key in result_keys}
    for step in range(num_steps):
        demo_step = step + frame_start
        joint_pos = (
            joint_pos_sequence[step]
            if joint_pos_sequence is not None
            else joint_pos_demo[demo_step]
        )
        wrist_pos = joint_pos[0]
        indices = retargeter.optimizer.target_link_human_indices
        ref_val = get_ref_val(joint_pos, indices)
        joint_vals, hand_qpos, wrist_qpos, wrist_idxs = retarget_one_hand(
            wrist_pos=wrist_pos,
            retarget_type=retarget_type,
            ref_value=ref_val,
            retargeter=retargeter,
            hand_init_qpos=hand_init_qpos,
            actuated_dof_names=actuated_dof_names,
            actuated_dof_idxs=actuated_dof_idxs,
        )
        # clamp
        hand_qpos = torch.clamp(hand_qpos, dof_limits[0].to(hand_qpos.device), dof_limits[1].to(hand_qpos.device))
        all_ret['hand_qpos'].append(hand_qpos[0].cpu().numpy()) # only save the first env!
        all_ret['wrist_qpos'].append(wrist_qpos)
        all_ret['wrist_idxs'].append(wrist_idxs)
    all_ret = {key: np.stack(val) for key, val in all_ret.items()}
    if wrist_quat_demo is not None:
        optimizer_qpos = all_ret['hand_qpos'].copy()
        wrist_quat_sequence = np.asarray(
            wrist_quat_demo[frame_start:frame_start + num_steps], dtype=np.float64
        )
        all_ret['hand_qpos'], orientation_cfg = align_wrist_orientation_to_demo(
            optimizer_qpos,
            actuated_dof_names,
            wrist_quat_sequence,
            anchor_frame=wrist_orientation_anchor,
        )
        all_ret['hand_qpos_optimizer_raw'] = optimizer_qpos
        all_ret['wrist_orientation_reference'] = orientation_cfg
        if finger_retargeter is not None:
            refinement_joint_pos = (
                joint_pos_sequence
                if joint_pos_sequence is not None
                else np.asarray(
                    joint_pos_demo[frame_start:frame_start + num_steps],
                    dtype=np.float64,
                )
            )
            all_ret['hand_qpos'] = refine_fingers_with_fixed_wrist(
                all_ret['hand_qpos'],
                actuated_dof_names,
                refinement_joint_pos,
                finger_retargeter,
            )
            all_ret['finger_refined_with_fixed_wrist'] = True
    if temporal_config is not None:
        raw_qpos = all_ret['hand_qpos'].copy()
        angular_indices = tuple(
            index
            for index, name in enumerate(actuated_dof_names)
            if not any(axis in name for axis in ('_tx_', '_ty_', '_tz_'))
        )
        translation_indices = tuple(
            index
            for index, name in enumerate(actuated_dof_names)
            if any(axis in name for axis in ('_tx_', '_ty_', '_tz_'))
        )
        repaired_qpos = repair_trajectory(
            raw_qpos,
            config=temporal_config,
            angular_indices=angular_indices,
            translation_indices=translation_indices,
            lower_limits=dof_limits[0].detach().cpu().numpy(),
            upper_limits=dof_limits[1].detach().cpu().numpy(),
        )
        wrist_columns = [
            index for index, name in enumerate(actuated_dof_names) if 'forearm' in name
        ]
        all_ret['hand_qpos_raw'] = raw_qpos
        all_ret['hand_qpos'] = repaired_qpos
        all_ret['wrist_qpos'] = repaired_qpos[:, wrist_columns]
        all_ret['temporal_metrics'] = {
            'before': trajectory_metrics(raw_qpos, angular_indices=angular_indices),
            'after': trajectory_metrics(repaired_qpos, angular_indices=angular_indices),
        }
        print(f"Temporal repair metrics: {all_ret['temporal_metrics']}")
    return all_ret
