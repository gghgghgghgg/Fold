import os
from os.path import join
from fold.asset_utils import get_urdf_path

allegro_asset_dir = "allegro_hand/"
left_rel_urdf = join(allegro_asset_dir, "allegro_hand_left_6dof.urdf")
right_rel_urdf = join(allegro_asset_dir, "allegro_hand_right_6dof.urdf")

ALLEGRO_LEFT_CFG={
    "urdf_path": get_urdf_path(left_rel_urdf),
    "wrist_link_name": "base_dummy_link",
    "kpt_link_names": ["link_15.0_tip", "link_11.0_tip", "link_7.0_tip", "link_3.0_tip" ],
    "actuators": {
        "finger": dict(
            joint_exprs=['.*.0.+'],
            kp=30.0,
            kv=2.0,
            force_range=100.0,
        ),
        "wrist_rot": dict(
            joint_exprs=[r'[LR]_forearm_(roll|pitch|yaw)_link_joint'],
            kp=60,
            kv=5.0,
            force_range=100.0,
        ),
        "wrist_trans": dict(
            joint_exprs=[r'[LR]_forearm_t[xyz]_link_joint'],
            kp=350.0,
            kv=20.0,
            force_range=100.0,
        ),
    },
    "collision_groups": {8: 0, 13: 1, 14: 2, 15: 3, 16: 4, 17: 1, 18: 2, 19: 3, 20: 4, 21: 1, 22: 2, 23: 3, 24: 4, 25: 1, 26: 2, 27: 3, 28: 4},
    "collision_palm_name": "base_link",
}

ALLEGRO_RIGHT_CFG={
    "urdf_path": get_urdf_path(right_rel_urdf),
    "wrist_link_name": "base_dummy_link",
    "kpt_link_names": ["link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip" ],
    "actuators": ALLEGRO_LEFT_CFG["actuators"].copy(),
    "collision_groups": {8: 0, 13: 1, 14: 2, 15: 3, 16: 4, 17: 1, 18: 2, 19: 3, 20: 4, 21: 1, 22: 2, 23: 3, 24: 4, 25: 1, 26: 2, 27: 3, 28: 4},
    "collision_palm_name": "base_link",
}

ALLEGRO_CFGs=dict(
    left=ALLEGRO_LEFT_CFG,
    right=ALLEGRO_RIGHT_CFG,
)
