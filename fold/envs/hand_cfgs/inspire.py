import os
from os.path import join
from fold.asset_utils import get_urdf_path

inspire_asset_dir = "inspire_hand/"
left_rel_urdf = join(inspire_asset_dir, "left_xyz_copy.urdf")
right_rel_urdf = join(inspire_asset_dir, "right_xyz_copy.urdf")

MIMIC_JOINTS={
    "TH_J3": ("TH_J2", 1.0),
    "TH_J4": ("TH_J2", 1.13),
    "FF_J2": ("FF_J1", 1.13),
    "MF_J2": ("MF_J1", 1.13),
    "RF_J2": ("RF_J1", 1.08),
    "LF_J2": ("LF_J1", 1.13),
}
INSPIRE_LEFT_MIMIC_JOINTS=dict()
for name, (parent, ratio) in MIMIC_JOINTS.items():
    INSPIRE_LEFT_MIMIC_JOINTS[f"L_{name}"] = (f"L_{parent}", ratio)
INSPIRE_RIGHT_MIMIC_JOINTS=dict()
for name, (parent, ratio) in MIMIC_JOINTS.items():
    INSPIRE_RIGHT_MIMIC_JOINTS[f"R_{name}"] = (f"R_{parent}", ratio)

INSPIRE_LINK_NAMES=["TH_Tip", "FF_Tip", "MF_Tip", "RF_Tip", "LF_Tip"]
INSPIRE_DEFAULT_QPOS={
    'left':  [0.25, 0.11, 1.0, 0.76, 0.18, 0.58, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
    'right': [-0.35, 0.1, 1.05, -0.37, 0.23,  2.64,  0.  ,  0.  ,  0.  , 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ]
}

INSPIRE_LEFT_CFG={
    "urdf_path": get_urdf_path(left_rel_urdf),
    "wrist_link_name": "base_link",
    "mimic_joint_map": INSPIRE_LEFT_MIMIC_JOINTS.copy(),
    "kpt_link_names": ['L_'+name for name in INSPIRE_LINK_NAMES],
    # NOTE: this gets reset by demo data
    "joint_limits": {
        "L_forearm_tx_link_joint": (-0.05, 0.3),
        "L_forearm_ty_link_joint": (-0.1, 0.2),
        "L_forearm_tz_link_joint": (1.0, 1.3),

        "L_forearm_roll_link_joint": (-0.25, 0.5),
        "L_forearm_pitch_link_joint": (0.2, 0.6),
        "L_forearm_yaw_link_joint": (-0.7, 0.4),
    },
    "default_qpos": INSPIRE_DEFAULT_QPOS['left'],
    "actuators": {
        "finger": dict(
            joint_exprs=['.*J1', '.*J2', '.*J3', '.*J4'],
            kp=20.0,
            kv=1.0,
            force_range=50.0,
        ),
        "wrist_rot": dict(
            joint_exprs=[r'[LR]_forearm_(roll|pitch|yaw)_link_joint'],
            kp=80,
            kv=5.0,
            force_range=50.0,
        ),
        "wrist_trans": dict(
            joint_exprs=[r'[LR]_forearm_t[xyz]_link_joint'],
            kp=350,
            kv=12.0,
            force_range=50.0,
        ),
    },
    "collision_groups": {7: 0, 13: 1, 14: 2, 15: 3, 16: 4, 17: 5, 18: 1, 23: 1},
    "collision_palm_name": "base_link",
}

INSPIRE_RIGHT_CFG={
    "urdf_path": get_urdf_path(right_rel_urdf),
    "wrist_link_name": "base_link",
    "mimic_joint_map": INSPIRE_RIGHT_MIMIC_JOINTS.copy(),
    "kpt_link_names": ['R_'+name for name in INSPIRE_LINK_NAMES],
    # NOTE: this gets reset by demo data
    "joint_limits": {
        "R_forearm_tx_link_joint": (-0.4, -0.1),
        "R_forearm_ty_link_joint": (-0.1, 0.2),
        "R_forearm_tz_link_joint": (1.0, 1.3),

        "R_forearm_roll_link_joint": (-0.4, 0.2),
        "R_forearm_pitch_link_joint": (0.0, 0.6),
        "R_forearm_yaw_link_joint": (2.5, 3.1),
    },
    "default_qpos": INSPIRE_DEFAULT_QPOS['right'],
    "actuators": INSPIRE_LEFT_CFG["actuators"].copy(),
    "collision_groups": INSPIRE_LEFT_CFG["collision_groups"].copy(),
    "collision_palm_name": INSPIRE_LEFT_CFG["collision_palm_name"],
}

INSPIRE_CFGs=dict(left=INSPIRE_LEFT_CFG, right=INSPIRE_RIGHT_CFG)
