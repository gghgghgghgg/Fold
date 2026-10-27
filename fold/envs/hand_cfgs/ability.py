import os
from os.path import join
from fold.asset_utils import get_urdf_path

ability_asset_dir = "ability_hand/"
left_rel_urdf = join(ability_asset_dir, "ability_hand_left_6dof.urdf")
right_rel_urdf = join(ability_asset_dir, "ability_hand_right_6dof.urdf")

ABILITY_MIMIC_JOINTS = {
    "index_q2": ("index_q1", 1.05851325),
    "middle_q2": ("middle_q1", 1.05851325),
    "ring_q2": ("ring_q1", 1.05851325),
    "pinky_q2": ("pinky_q1", 1.05851325),
}

ABILITY_ACT_FINGER_JOINTS = ["thumb_q1", "thumb_q2", "index_q1", "middle_q1", "ring_q1", "pinky_q1"]

ABILITY_MIMIC=dict(left=dict(), right=dict())
for name, (target, multiplier) in ABILITY_MIMIC_JOINTS.items():
    for side in ["left", "right"]:
        ABILITY_MIMIC[side][name] = (target, multiplier)

ABILITY_LINKS=["thumb_tip",  "index_tip", "middle_tip", "ring_tip", "pinky_tip", "thumb_L1", "index_L1", "middle_L1", "ring_L1", "pinky_L1"]

ABILITY_LEFT_CFG={
    "urdf_path": get_urdf_path(left_rel_urdf),
    "wrist_link_name": "base",
    "mimic_joint_map": ABILITY_MIMIC['left'].copy(),
    "kpt_link_names": ABILITY_LINKS.copy(),
    "joint_limits": {
        "index_q1": (0.26, 2.0943951),
        "index_q2": (0.26, 2.6586),
        "middle_q1": (0.26, 2.0943951),
        "middle_q2": (0.26, 2.6586),
        "ring_q1": (0.26, 2.0943951),
        "ring_q2": (0.26, 2.6586),
        "pinky_q1": (0.26, 2.0943951),
        "pinky_q2": (0.26, 2.6586),
    },
    "actuators": {
        "finger": dict(
            joint_exprs=['.*q1', '.*q2'],
            kp=40.0,
            kv=3.0,
            force_range=50.0,
        ),
        "wrist_rot": dict(
            joint_exprs=[r'[LR]_forearm_(roll|pitch|yaw)_link_joint'],
            kp=80,
            kv=4.5,
            force_range=50.0,
        ),
        "wrist_trans": dict(
            joint_exprs=[r'[LR]_forearm_t[xyz]_link_joint'],
            kp=300,
            kv=4.0,
            force_range=100,
        ),
    },
    "collision_groups": {7: 0, 8: 0, 14: 1, 15: 2, 16: 3, 17: 4, 18: 5},
    "collision_palm_name": "thumb_base",
}

ABILITY_RIGHT_CFG={
    "urdf_path": get_urdf_path(right_rel_urdf),
    "wrist_link_name": "base",
    "mimic_joint_map": ABILITY_MIMIC['right'].copy(),
    "kpt_link_names": ABILITY_LINKS.copy(),
    "joint_limits": {
        "index_q1": (0.26, 2.0943951),
        "index_q2": (0.26, 2.6586),
        "middle_q1": (0.26, 2.0943951),
        "middle_q2": (0.26, 2.6586),
        "ring_q1": (0.26, 2.0943951),
        "ring_q2": (0.26, 2.6586),
        "pinky_q1": (0.26, 2.0943951),
        "pinky_q2": (0.26, 2.6586),
    },
    "actuators": ABILITY_LEFT_CFG["actuators"].copy(),
    "collision_groups": {7: 0, 8: 0, 14: 1, 15: 2, 16: 3, 17: 4, 18: 5},
    "collision_palm_name": "thumb_base",
}

ABILITY_CFGs=dict(left=ABILITY_LEFT_CFG, right=ABILITY_RIGHT_CFG)