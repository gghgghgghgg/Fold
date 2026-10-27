import os
from os.path import join
from fold.asset_utils import get_urdf_path

dex_asset_dir = "dexrobot_hand/"
left_rel_urdf = join(dex_asset_dir, "dexhand021_left_6dof.urdf")
right_rel_urdf = join(dex_asset_dir, "dexhand021_right_6dof.urdf")

DEX_LEFT_CFG={
    "urdf_path": get_urdf_path(left_rel_urdf),
    "wrist_link_name": "base_dummy",
    "kpt_link_names": ["l_f_link1_tip",  "l_f_link2_tip", "l_f_link3_tip", "l_f_link4_tip", "l_f_link5_tip"], # should be overwritten with retargeting
    "actuators": {
        "finger": dict(
            joint_exprs=['[lr]_f.*'],
            kp=50.0,
            kv=3,
            force_range=100.0,
        ),
        "wrist_rot": dict(
            joint_exprs=[r'[LR]_forearm_(roll|pitch|yaw)_link_joint'],
            kp=70,
            kv=4,
            force_range=100.0,
        ),
        "wrist_trans": dict(
            joint_exprs=[r'[LR]_forearm_t[xyz]_link_joint'],
            kp=300.0,
            kv=15.0,
            force_range=100.0,
        ),
    },
    "collision_groups": {7: 0, 14: 2, 15: 3, 16: 4, 17: 5, 18: 6, 19: 2, 20: 3, 21: 4, 22: 5, 23: 6, 24: 2, 25: 3, 26: 4, 27: 5, 28: 6, 29: 2, 30: 2, 31: 3, 32: 3, 33: 4, 34: 4, 35: 5, 36: 5, 37: 6, 38: 6},
    "collision_palm_name": "left_hand_base",
}

DEX_RIGHT_CFG={
    "urdf_path": get_urdf_path(right_rel_urdf),
    "wrist_link_name": "base_dummy",
    "kpt_link_names": ["r_f_link1_tip",  "r_f_link2_tip", "r_f_link3_tip", "r_f_link4_tip", "r_f_link5_tip"], # should be overwritten with retargeting
    "actuators": DEX_LEFT_CFG["actuators"].copy(),
    "collision_groups": {7: 0, 14: 2, 15: 3, 16: 4, 17: 5, 18: 6, 19: 2, 20: 3, 21: 4, 22: 5, 23: 6, 24: 2, 25: 3, 26: 4, 27: 5, 28: 6, 29: 2, 30: 2, 31: 3, 32: 3, 33: 4, 34: 4, 35: 5, 36: 5, 37: 6, 38: 6},
    "collision_palm_name": "right_hand_base",
}

DEX_CFGs=dict(
    left=DEX_LEFT_CFG,
    right=DEX_RIGHT_CFG,
)