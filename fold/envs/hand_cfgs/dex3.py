import os
from os.path import join
from fold.asset_utils import get_urdf_path

dex3_asset_dir = "dex3_hand/"
left_rel_urdf = join(dex3_asset_dir, "dex3_left_6dof.urdf")
right_rel_urdf = join(dex3_asset_dir, "dex3_right_6dof.urdf")

DEX3_LEFT_CFG={
    "urdf_path": get_urdf_path(left_rel_urdf),
    "wrist_link_name": "left_hand_palm_link",
    "kpt_link_names": ['left_hand_palm_link', 'left_hand_thumb_0_link', 'left_hand_middle_0_link', 'left_hand_index_0_link', 'left_hand_thumb_1_link', 'left_hand_middle_1_link', 'left_hand_index_1_link', 'left_hand_thumb_2_link'],
    "actuators": {
      "finger": dict(
        joint_exprs=[".*0_joint", ".*1_joint", ".*2_joint"],
        kp=100.0,
        kv=6,
        force_range=50.0,
      ),
      "wrist_rot": dict(
            joint_exprs=[r'[LR]_forearm_(roll|pitch|yaw)_link_joint'],
            kp=200,
            kv=10,
            force_range=50.0,
        ),
        "wrist_trans": dict(
            joint_exprs=[r'[LR]_forearm_t[xyz]_link_joint'],
            kp=300,
            kv=5.0,
            force_range=50.0,
        ),
    },
    "collision_groups": {7: 0, 11: 1, 12: 2, 13: 3, 14: 1},
    "collision_palm_name": "left_hand_palm_link",
}

DEX3_RIGHT_CFG={
    "urdf_path": get_urdf_path(right_rel_urdf),
    "wrist_link_name": "right_hand_palm_link",
    "kpt_link_names": ['right_hand_palm_link', 'right_hand_thumb_0_link', 'right_hand_middle_0_link', 'right_hand_index_0_link', 'right_hand_thumb_1_link', 'right_hand_middle_1_link', 'right_hand_index_1_link', 'right_hand_thumb_2_link'],
    "collision_groups": {7: 0, 11: 1, 12: 2, 13: 3, 14: 1},
    "collision_palm_name": "right_hand_palm_link",
    "actuators": DEX3_LEFT_CFG["actuators"].copy(),
}

DEX3_CFGs=dict(
    left=DEX3_LEFT_CFG,
    right=DEX3_RIGHT_CFG,
)