import os
from os.path import join
from fold.asset_utils import get_urdf_path

schunk_asset_dir = "schunk_hand/"
left_rel_urdf = join(schunk_asset_dir, "schunk_hand_left_6dof.urdf")
right_rel_urdf = join(schunk_asset_dir, "schunk_hand_right_6dof.urdf")

SCHUNK_MIMIC_JOINTS_LEFT = {
    'left_hand_j5': ('left_hand_Thumb_Opposition', 1.0),
    'left_hand_j3': ('left_hand_Thumb_Flexion', 1.01511),
    'left_hand_j4': ('left_hand_Thumb_Flexion', 1.44889),
    'left_hand_j14': ('left_hand_Index_Finger_Distal', 1.045),
    'left_hand_j15': ('left_hand_Middle_Finger_Distal', 1.0454),
    'left_hand_j12': ('left_hand_Ring_Finger', 1.3588),
    'left_hand_j16': ('left_hand_Ring_Finger', 1.42093),
    'left_hand_j13': ('left_hand_Pinky', 1.3588),
    'left_hand_j17': ('left_hand_Pinky', 1.42307),
    'left_hand_index_spread': ('left_hand_Finger_Spread', 0.5),
    'left_hand_ring_spread': ('left_hand_Finger_Spread', 0.5),
}

SCHUNK_LEFT_CFG={
    "urdf_path": get_urdf_path(left_rel_urdf),
    "wrist_link_name": "left_hand_base_link",
    "kpt_link_names": ["thtip", "fftip", "mftip", "rftip", "lftip", "left_hand_b", "left_hand_p", "left_hand_o", "left_hand_n", "left_hand_i"],
    "mimic_joint_map": SCHUNK_MIMIC_JOINTS_LEFT.copy(),
    "actuators": {
        "finger": dict(
            joint_exprs=['.*hand.+'],
            kp=80,
            kv=4.5,
            force_range=80,
        ),
        "wrist_rot": dict(
            joint_exprs=[r'[LR]_forearm_(roll|pitch|yaw)_link_joint'],
            kp=80,
            kv=5,
            force_range=100,
        ),
        "wrist_trans": dict(
            joint_exprs=[r'[LR]_forearm_t[xyz]_link_joint'],
            kp=350,
            kv=8,
            force_range=100,
        ),
    },
    "collision_groups": {7: 0, 8: 0, 13: 1, 14: 1, 15: 2, 16: 3, 17: 4, 18: 1, 19: 1, 20: 2, 21: 3, 22: 4, 23: 1, 24: 1, 25: 2, 26: 3, 27: 4, 28: 1, 29: 1},
    "collision_palm_name": "left_hand_e1",
}

SCHUNK_MIMIC_JOINTS_RIGHT={
    'right_hand_j5': ('right_hand_Thumb_Opposition', 1.0),
    'right_hand_j3': ('right_hand_Thumb_Flexion', 1.01511),
    'right_hand_j4': ('right_hand_Thumb_Flexion', 1.44889),
    'right_hand_j14': ('right_hand_Index_Finger_Distal', 1.045),
    'right_hand_j15': ('right_hand_Middle_Finger_Distal', 1.0454),
    'right_hand_j12': ('right_hand_Ring_Finger', 1.3588),
    'right_hand_j16': ('right_hand_Ring_Finger', 1.42093),
    'right_hand_j13': ('right_hand_Pinky', 1.3588),
    'right_hand_j17': ('right_hand_Pinky', 1.42307),
    'right_hand_index_spread': ('right_hand_Finger_Spread', 0.5),
    'right_hand_ring_spread': ('right_hand_Finger_Spread', 0.5),
}

SCHUNK_RIGHT_CFG={
    "urdf_path": get_urdf_path(right_rel_urdf),
    "wrist_link_name": "right_hand_base_link",
    "kpt_link_names": ["thtip", "fftip", "mftip", "rftip", "lftip", "right_hand_b", "right_hand_p", "right_hand_o", "right_hand_n", "right_hand_i"],
    "mimic_joint_map": SCHUNK_MIMIC_JOINTS_RIGHT.copy(),
    "actuators": SCHUNK_LEFT_CFG["actuators"].copy(),
    "collision_groups": {7: 0, 8: 0, 13: 1, 14: 1, 15: 2, 16: 3, 17: 4, 18: 1, 19: 1, 20: 2, 21: 3, 22: 4, 23: 1, 24: 1, 25: 2, 26: 3, 27: 4, 28: 1, 29: 1},
    "collision_palm_name": "right_hand_e1",
}

SCHUNK_CFGs=dict(left=SCHUNK_LEFT_CFG, right=SCHUNK_RIGHT_CFG)