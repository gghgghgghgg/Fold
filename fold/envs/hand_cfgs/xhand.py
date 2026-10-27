import os
from os.path import join
from fold.asset_utils import get_urdf_path

xhand_asset_dir = "xhand/"
left_rel_urdf = join(xhand_asset_dir, "xhand_left_6dof.urdf")
right_rel_urdf = join(xhand_asset_dir, "xhand_right_6dof.urdf")

XHAND_LEFT_CFG={
    "urdf_path": get_urdf_path(left_rel_urdf),
    "wrist_link_name": "left_hand_link",
    "kpt_link_names": ["left_hand_thumb_rota_tip", "left_hand_index_rota_tip", "left_hand_mid_tip", "left_hand_ring_tip", "left_hand_pinky_tip"],
    "actuators": {
        "finger": dict(
            joint_exprs=['.*joint.+'],
            kp=20.0,
            kv=1.5,
            force_range=100.0,
        ),
        "wrist_rot": dict(
            joint_exprs=[r'[LR]_forearm_(roll|pitch|yaw)_link_joint'],
            kp=60.0,
            kv=3.0,
            force_range=100,
        ),
        "wrist_trans": dict(
            joint_exprs=[r'[LR]_forearm_t[xyz]_link_joint'],
            kp=350.0,
            kv=15.0,
            force_range=100,
        ),
    },
    "collision_groups": {7: 0, 13: 1, 14: 2, 15: 3, 16: 4, 17: 5, 18: 1, 19: 2},
    "collision_palm_name": "left_hand_link",
}

XHAND_RIGHT_CFG={
    "urdf_path": get_urdf_path(right_rel_urdf),
    "wrist_link_name": "right_hand_link",
    "kpt_link_names": ["right_hand_thumb_rota_tip", "right_hand_index_rota_tip", "right_hand_mid_tip", "right_hand_ring_tip", "right_hand_pinky_tip"],
    "actuators": XHAND_LEFT_CFG["actuators"].copy(),
    "collision_groups": {7: 0, 13: 1, 14: 2, 15: 3, 16: 4, 17: 5, 18: 1, 19: 2},
    "collision_palm_name": "right_hand_link",
}

XHAND_CFGs=dict(
    left=XHAND_LEFT_CFG,
    right=XHAND_RIGHT_CFG,
)