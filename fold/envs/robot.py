import os
import sys
from os.path import join
import numpy as np
import torch
import genesis as gs
from collections import defaultdict
from typing import Dict
import importlib
from fold.asset_utils import get_urdf_path

@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower + 1e-5)

def get_hand_specific_cfg(name="inspire_hand"):
    hand_prefix = name.replace("_hand", "") # xhand still stays xhand
    try:
        module_name = f"fold.envs.hand_cfgs.{hand_prefix}"
        hand_cfg_module = importlib.import_module(module_name)
        # Assume the hand cfgs all ends with *_CFGs inside the module
        cfg_attr = next(
            attr for attr in dir(hand_cfg_module)
            if attr.upper().endswith("_CFGS")
        )
        hand_cfgs = getattr(hand_cfg_module, cfg_attr)
        assert isinstance(hand_cfgs, dict), f"{cfg_attr} should be a dict"
        return hand_cfgs

    except (ImportError, StopIteration, AttributeError, KeyError) as e:
        raise ValueError(f"Invalid hand name or configuration: {name} - {e}")

def get_default_robot_cfg(name="inspire_hand", side="left", wrist_only=True, group_collisions=False):
    robot_cfg = {
        "name": f"{name}_{side}",
        "gravity_compensation": 0.8,
        "actuators": {
            "all": dict(
                joint_exprs=['.*'],
                kp=100.0,
                kv=10.0,
                force_range=100.0,
            )
        },
        "base_init_pos": [0.0, 0.0, 0.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "action_moving_avg": 1.0,
        "action_mode": "residual",
        'wrist_only': False,
        "kpt_link_names": [], # NOTE this can only contact non-collision links, the collision links would be added automatically
        "collect_data": False, # if True, save kpt pos and joint qpos after each episode
        "use_saved_targets": False, # if True, use saved targets for residual action, which is not always achievable
        "hybrid_scales": (0.04, 0.5),
        "res_cap": False, # if True, use hybrid_scales to also cap residual wrist actions
        "show_keypoints": False,
        "visualization": True,
    }
    assert side in ["left", "right"], f"Invalid side {side}"
    # NOTE robot_cfg['wrist_link_name'] should match retargeting results

    hand_cfg = get_hand_specific_cfg(name=name)
    assert hand_cfg is not None, f"Hand config for {name} not found"
    assert hand_cfg.get(side, None) is not None, f"Hand config for {name} {side} not found"
    robot_cfg.update(hand_cfg[side].copy())

    if group_collisions:
        assert robot_cfg.get("collision_groups", None) is not None, "Need to set collision_groups"

    urdf_path = robot_cfg["urdf_path"]
    assert os.path.exists(urdf_path), f"{urdf_path} does not exist"
    return robot_cfg


class BaseRobot:
    def __init__(
        self,
        robot_cfg,
        device,
        scene,
        num_envs,
        obs_scale={'dof_vel': 0.1, 'root_ang_vel': 0.1, 'contact_norm': 0.1},
        retarget_data=dict(),
        visualize_contact=False,
        is_eval=False,
        disable_collision=False,
    ):

        """
        NOTE need to set fixed=True so the root link is fixed and the 6 wrist joints are actuated
        """

        self.name = robot_cfg["name"]
        self.cfg = robot_cfg
        self.device = device
        self.initialized = False
        self.entity = None
        self.obs_scale = obs_scale # a dict of scales for each observation

        self.init_pos = torch.tensor(
            robot_cfg["base_init_pos"], dtype=torch.float32, device=self.device
        )
        self.init_quat = torch.tensor(
            robot_cfg["base_init_quat"], dtype=torch.float32, device=self.device
        )

        self.action_mode = robot_cfg.get("action_mode", "residual")
        assert self.action_mode in ["residual", "absolute", "relative", "hybrid", "kinematic"], f"Invalid action mode {self.action_mode}"
        self.hybrid_scales = robot_cfg.get("hybrid_scales", (0.04, 0.5))
        self.res_cap = robot_cfg.get("res_cap", False)
        self.num_envs = num_envs
        self.scene = scene
        assert not self.initialized, "Robot already initialized"
        self.entity = scene.add_entity(
            gs.morphs.URDF(
                file=self.cfg["urdf_path"],
                pos=self.init_pos.cpu().numpy(),
                quat=self.init_quat.cpu().numpy(),
                convexify=True,
                fixed=True,
                merge_fixed_links=False, # NOTE: need to keep this to track the fingertip links
                recompute_inertia=True,
                collision=(not disable_collision),
                visualization=self.cfg.get("visualization", True),
            ),
            material=gs.materials.Rigid(gravity_compensation=robot_cfg["gravity_compensation"]),
            visualize_contact=visualize_contact,
        )
        all_joints = self.entity.joints
        self.actuated_joints = [joint for joint in all_joints if joint.type in [gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC]]
        self.actuated_dof_names = [joint.name for joint in self.actuated_joints]
        self.actuated_dof_idxs = [joint.dof_idx_local for joint in self.actuated_joints]
        self.ndof = len(self.actuated_joints) # NOTE this is NOT necessarily action dim due to mimic joints
        self.wrist_only = robot_cfg.get("wrist_only", False)
        self.wrist_dof_idxs = [joint.dof_idx_local for joint in self.actuated_joints if 'forearm' in joint.name]
        self.finger_dof_idxs = [joint.dof_idx_local for joint in self.actuated_joints if 'forearm' not in joint.name]
        assert len(self.wrist_dof_idxs) == 6, f"Found {len(self.wrist_dof_idxs)} wrist dofs"
        # setup joint limits
        dof_limits = []
        custom_dof_limits = robot_cfg.get("joint_limits", dict()).copy()
        for joint in self.actuated_joints:
            if joint.name in custom_dof_limits:
                limit = custom_dof_limits.pop(joint.name)
                dof_limits.append(
                    np.array(limit)[None,:]# shape (1, 2)
                    )
            else:
                dof_limits.append(joint.dofs_limit)
        assert len(custom_dof_limits) == 0, f"Custom limits {custom_dof_limits} not used"
        dof_limits = np.concatenate(dof_limits) # shape (ndof, 2) # NOTE that joint.dofs_limit is already shape (1, 2)
        # check the joint range is not less than zero
        assert (dof_limits[:, 1] - dof_limits[:, 0] >= 0).all(), f"Joint limits have negative range: {dof_limits}"
        self.dof_limits = torch.tensor(dof_limits, dtype=torch.float32, device=self.device) # shape (num_joints, 2)
        self.dof_range = self.dof_limits[:, 1] - self.dof_limits[:, 0] # shape (num_joints,)

        self.action_moving_avg = self.cfg["action_moving_avg"]

        # setup mimic joint mapping, i.e. map the same action input index to multiple joints
        self.mimic_joint_map = self.cfg.get("mimic_joint_map", dict())
        self.setup_action_mapping(self.actuated_joints, self.mimic_joint_map)

        # setup default joint qpos based on mimic joint mapping
        qposes = np.concatenate([joint.init_qpos for joint in self.actuated_joints]) # shape (ndof,)
        default_qpos = robot_cfg.get("default_qpos", None)
        if default_qpos is not None:
            assert len(default_qpos) == len(qposes), f"len(default_qpos)={len(default_qpos)} != {self.ndof}"
            qposes[:] = np.array(default_qpos)
        qposes = torch.tensor(qposes, dtype=torch.float32, device=self.device)
        # repeat for each env
        self.init_qpos = qposes.unsqueeze(0).repeat(self.num_envs, 1)
        if 'init_qpos' in retarget_data:
            self.set_custom_init_qpos(retarget_data['init_qpos'])

        self.is_eval = is_eval
        if self.is_eval and self.num_envs > 1:
            print('WARNING: setting robot.is_eval to True, setting the init_qpos for first env to 0s')
            self.init_qpos[-1, :] = 0.0

        # only take collision geoms for contact forces!
        coll_idxs_local, coll_idxs_global = [], []
        coll_link_names = []
        for i, link in enumerate(self.entity.links):
            if len(link.geoms) > 0:
                coll_idxs_local.append(i)
                coll_idxs_global.append(link.idx) # NOTE this is global!!
                coll_link_names.append(link.name)
        self.coll_idxs_local = coll_idxs_local # NOTE this is local!!
        self.coll_idxs_global = coll_idxs_global
        self.n_coll_links = len(coll_idxs_local)
        self.coll_link_names = coll_link_names

        # setup link names to track
        link_names = self.cfg.get("kpt_link_names", [])
        link_names += self.coll_link_names

        if 'kpts_data' in retarget_data:
            print(f"Overwrite kpt_link_names with saved retarget data")
            link_names = retarget_data['kpts_data']['kpt_names']

        self.set_kpt_links(link_names)

        self.kpt_markers = []
        if self.cfg.get("show_keypoints", False):
            KPT_COLOR = [x/255.0 for x in (229, 152, 155)] + [1.0] # pink
            KPT_RADIUS = 0.007
            for _ in range(self.n_kpts):
                marker = self.scene.add_entity(
                    gs.morphs.Sphere(
                        radius=KPT_RADIUS, fixed=False, collision=False, # no collision
                        ),
                    surface=gs.surfaces.Rough(color=KPT_COLOR),
                )
                self.kpt_markers.append(marker)


        self.wrist_link_name = self.cfg.get("wrist_link_name", "base_link")
        wrist_link_idxs = [i for i, link in enumerate(self.entity.links) if link.name == self.wrist_link_name]
        assert len(wrist_link_idxs) == 1, f"Found {len(wrist_link_idxs)} wrist links"
        self.wrist_link_idx = wrist_link_idxs[0]

        self.residual_qpos = None
        self.residual_num_frames = None
        if 'residual_qpos' in retarget_data:
            qpos_targets = None
            if self.cfg.get("use_saved_targets", False):
                qpos_targets = retarget_data.get('qpos_targets', None)
                assert qpos_targets is not None, "Need to set qpos_targets for residual action mode"
            self.set_residual_qpos(
                num_frames=retarget_data['num_frames'],
                residual_qpos_dict=retarget_data['residual_qpos'],
                qpos_targets_dict=qpos_targets,
            )
        if self.action_mode == "relative":
            assert self.residual_qpos is not None, "Need to set residual qpos for relative action mode"
            self.set_relative_step_size(self.residual_qpos)

        if 'limits' in retarget_data:
            self.set_custom_joint_limits(retarget_data['limits'])

        self.n_links = self.entity.n_links

        self.initialized = True
        self.initialize_value_buffers()


        self.obs_dim, self.obs_dim_info = self.compute_obs_dim()

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)

        self.collect_data = self.cfg.get("collect_data", False)
        if self.collect_data:
            print(f"Collecting data for {self.name}")
        self.episode_data = defaultdict(list)

    def get_collision_groups(self):
        return self.cfg.get("collision_groups", dict())

    def set_kpt_links(self, kpt_link_names):
        self.kpt_link_names, self.kpt_link_idxs = [], []
        link_names = [link.name for link in self.entity.links]
        for name in kpt_link_names:
            if name in link_names:
                self.kpt_link_names.append(name)
                self.kpt_link_idxs.append(link_names.index(name))
            else:
                print(f"WARNING - Link {name} not found in the URDF")
        self.n_kpts = len(self.kpt_link_names)
        return

    def set_custom_init_qpos(self, joint_qpos: Dict):
        curr_qpos = self.init_qpos.clone() # shape (num_envs, ndof)
        for jname, qpos in joint_qpos.items():
            if not jname in self.actuated_dof_names:
                print(f"WARNING: {jname} not in actuated joints")
                continue
            idx = self.actuated_dof_names.index(jname)
            if not isinstance(qpos, torch.Tensor):
                qpos = torch.tensor(qpos, dtype=torch.float32, device=self.device)
            curr_qpos[:, idx] = qpos.clone().to(self.device)
        self.init_qpos = curr_qpos
        self.curr_targets = curr_qpos.clone()

    def set_residual_qpos(self, num_frames, residual_qpos_dict: Dict, qpos_targets_dict=None): # v.shape is (num_frames, 1)
        # assert self.initialized, "Robot not initialized"
        # need to set shape (num_frames, ndof), don't repeat for the env dim
        # init_qpos is shape (num_envs, ndof)!!
        residual_qpos = self.init_qpos[0].clone().unsqueeze(0).repeat(num_frames, 1)
        qpos_dict = residual_qpos_dict
        if qpos_targets_dict is not None:
            qpos_dict = qpos_targets_dict
        for jname, qpos in qpos_dict.items():
            assert qpos.shape[0] == num_frames, f"qpos.shape[0]={qpos.shape[0]} != {num_frames}"
            if not jname in self.actuated_dof_names:
                print(f"WARNING: {jname} not in actuated joints")
                breakpoint()
            dof_idx = self.actuated_dof_names.index(jname)
            if isinstance(qpos, np.ndarray):
                residual_qpos[:, dof_idx] = torch.tensor(qpos, dtype=torch.float32, device=self.device)
            else:
                residual_qpos[:, dof_idx] = qpos.to(self.device)
        self.residual_qpos = residual_qpos
        self.residual_num_frames = num_frames

    def smooth_residual_qpos(self, joint_idxs=None):
        """
        return a smoothed version of residual_qpos along time dimension
        """
        from scipy.ndimage import gaussian_filter1d
        def gaussian_smooth(joint_values):
            joint_values = joint_values.cpu().numpy()
            smoothed = gaussian_filter1d(joint_values, sigma=1.0, axis=0)
            return torch.tensor(smoothed, dtype=torch.float32, device=self.device)

        smoothed_qpos = self.residual_qpos.clone()
        if joint_idxs is None:
            joint_idxs = [i for i in range(smoothed_qpos.shape[1])]
        smoothed_qpos[:, joint_idxs] = gaussian_smooth(smoothed_qpos[:, joint_idxs])
        return smoothed_qpos

    def interpolate_residual_qpos(self, mutiplier=1.0):
        """ interpolates residual qpos by mutiplier times, so it goes from (T, njoints) -> (T*multiplier, njoints) """
        assert self.residual_qpos is not None, "Need to set residual qpos"
        from scipy.interpolate import interp1d
        new_qpos = []
        old_T = self.residual_num_frames
        new_T = int(self.residual_num_frames * mutiplier)
        for j in range(self.residual_qpos.shape[1]):
            qpos = self.residual_qpos[:, j].cpu().numpy() # shape (num_frames,)
            x = np.linspace(0, old_T-1, old_T)
            func = interp1d(x, qpos, kind='linear')
            qpos_int = func(np.linspace(0, old_T-1, new_T))
            new_qpos.append(qpos_int)
        new_qpos = torch.tensor(new_qpos, dtype=torch.float32, device=self.device).T # shape (new_T, ndof)
        assert new_qpos.shape[0] == new_T, f"new_qpos.shape[0]={new_qpos.shape[0]} != {new_T}"
        self.residual_qpos = new_qpos
        self.residual_num_frames = new_T

    def set_relative_step_size(self, residual_qpos):
        # take the max of absolute values of all the per-step joint value changes
        deltas = torch.abs(residual_qpos[1:] - residual_qpos[:-1])
        max_delta = torch.max(deltas, dim=0).values
        self.relative_step_size = max_delta * 2.0 # shape (ndof,)

    def set_custom_joint_limits(self, joint_limits_dict: Dict):
        """ NOTE: skip finger joints """
        joint_limits = self.dof_limits.clone()
        for jname, limit in joint_limits_dict.items():
            if not jname in self.actuated_dof_names:
                print(f"WARNING: {jname} not in actuated joints")
                continue
            if 'forearm' not in jname:
                # only setting wrist joints
                continue
            idx = self.actuated_dof_names.index(jname)
            joint_limits[idx] = torch.tensor(limit, dtype=torch.float32, device=self.device)
        self.dof_limits = joint_limits
        self.dof_range = self.dof_limits[:, 1] - self.dof_limits[:, 0] # shape (num_joints,)

    def setup_action_mapping(self, actuated_joints, mimic_joint_map):
        """Setup so that action translation happens like actions[from_idx] controls joint targets"""
        # if no mimic joint, each action idx maps to a single joint
        if len(mimic_joint_map) == 0:
            self.action_dim = len(actuated_joints)
            self.action_from_idxs = [i for i in range(self.action_dim)]
            self.joint_from_idxs = [i for i in range(self.action_dim)]
            self.joint_multipliers = torch.tensor(
                [1.0 for i in range(self.action_dim)],
                dtype=torch.float32,
                device=self.device,
            )
            return
        # if mimic joint, each action idx maps to multiple joints
        action_dim = 0
        action_from_idxs = []
        joint_from_idxs = []
        joint_multipliers = []
        joint_name_to_action_idx = dict()
        joint_name_to_dof_idx = dict()
        for joint in actuated_joints: # assume the underlying model is fully actuated
            if joint.name not in mimic_joint_map:
                dof_idx = joint.dof_idx_local
                action_from_idxs.append(action_dim)
                joint_from_idxs.append(dof_idx)
                joint_name_to_action_idx[joint.name] = action_dim
                joint_name_to_dof_idx[joint.name] = dof_idx
                joint_multipliers.append(1.0)
                action_dim += 1
        for joint in actuated_joints:
            if joint.name in mimic_joint_map:
                parent, ratio = mimic_joint_map[joint.name]
                action_from_idxs.append(
                    joint_name_to_action_idx[parent]
                )
                joint_from_idxs.append(
                    joint_name_to_dof_idx[parent]
                )
                joint_multipliers.append(ratio)
        self.action_dim = action_dim
        self.action_from_idxs = action_from_idxs
        self.joint_from_idxs = joint_from_idxs

        self.joint_multipliers = torch.tensor(
            joint_multipliers, dtype=torch.float32, device=self.device,
        )
        return

    def get_action_dim(self):
        assert self.initialized, "Robot not initialized"
        return self.action_dim

    def set_joint_gains(self, kp, kv, fr, joint_idxs=None):
        if joint_idxs is None:
            joint_idxs = self.actuated_dof_idxs
        num_joints = len(joint_idxs)
        gain_shape = (num_joints,)
        if self.entity._solver._options.batch_dofs_info:
            gain_shape = (self.num_envs, num_joints)
        kp_tensor = torch.full(
            gain_shape, float(kp), dtype=torch.float32, device=self.device
        )
        kv_tensor = torch.full(
            gain_shape, float(kv), dtype=torch.float32, device=self.device
        )

        self.entity.set_dofs_kp(
            kp_tensor,
            joint_idxs,
        )
        self.entity.set_dofs_kv(
            kv_tensor,
            joint_idxs,
        )
        fr_tensor = torch.full(
            gain_shape, float(fr), dtype=torch.float32, device=self.device
        )
        self.entity.set_dofs_force_range(
            -1.0 * fr_tensor,
            fr_tensor,
            joint_idxs,
        )

    def set_inspire_gains(self):
        """ set the tuned values """
        kp, kv = 300.0, 30.0
        # kp, kv = 5000.0, 50.0
        forearm_trans = [joint.dof_idx_local for joint in self.actuated_joints if 'forearm_t' in joint.name]
        self.set_kp_kv_joints(kp, kv, forearm_trans)

        kp, kv = 300.0, 30.0
        # kp, kv = 5000.0, 50.0
        forearm_rot = [joint.dof_idx_local for joint in self.actuated_joints if 'roll' in joint.name or 'pitch' in joint.name or 'yaw' in joint.name]
        self.set_kp_kv_joints(kp, kv, forearm_rot)

        kp, kv = 20.0, 2.0
        # kp, kv = 100.0, 10.0
        fingers = [joint.dof_idx_local for joint in self.actuated_joints if '_J1' in joint.name or '_J2' in joint.name or '_J3' in joint.name or '_J4' in joint.name]
        self.set_kp_kv_joints(kp, kv, fingers)

    def post_scene_build_setup(self):
        assert self.initialized, "Robot not initialized"
        self.set_dof_gains_by_group(self.cfg["actuators"])

    def find_joints_in_group(self, joint_exprs):
        import re
        joint_idxs = []
        for joint in self.actuated_joints:
            matched = False
            jname = joint.name
            for expr in joint_exprs:
                if re.match(expr, jname):
                    matched = True
                    break
            if matched:
                joint_idxs.append(joint.dof_idx_local)
        if len(joint_idxs) == 0:
            print(f"No joints found for {joint_exprs}")
            breakpoint()
        return joint_idxs

    def set_dof_gains_by_group(self, actuator_cfgs):
        """ pass in a list of joint_groups """
        for group_name, group in actuator_cfgs.items():
            joint_idxs = self.find_joints_in_group(group["joint_exprs"])
            kp = group.get("kp", 200.0)
            kv = group.get("kv", 20.0)
            fr = group.get("force_range", 50.0)
            self.set_joint_gains(kp, kv, fr, joint_idxs)

    def initialize_value_buffers(self):
        assert self.initialized, "Robot not initialized"
        self.dof_pos = self.init_qpos.clone()
        self.dof_vel = torch.zeros((self.num_envs, self.ndof), dtype=torch.float32, device=self.device)
        # repeat default init pos
        self.curr_targets = self.init_qpos.clone()
        self.prev_targets = self.init_qpos.clone()
        self.curr_res_qpos = self.init_qpos.clone()

        # track keypoint links
        self.kpt_pos = torch.zeros((self.num_envs, self.n_kpts, 3), dtype=torch.float32, device=self.device)
        self.wrist_pose = torch.zeros((self.num_envs, 7), dtype=torch.float32, device=self.device) # 4 for quat, 3 for pos
        # contact forces, 3dim per link -> no used anymore, main env thread gets obj-hand filtered contact
        # self.contact_forces = torch.zeros((self.num_envs, self.n_coll_links, 3), dtype=torch.float32, device=self.device)
        self.control_forces = torch.zeros((self.num_envs, self.ndof), dtype=torch.float32, device=self.device)

    def update_value_buffers(self):
        assert self.initialized, "Robot not initialized"
        entity = self.entity
        self.dof_pos[:] = entity.get_dofs_position(self.actuated_dof_idxs)
        self.dof_vel[:] = entity.get_dofs_velocity(self.actuated_dof_idxs)

        link_pos = entity.get_links_pos()
        self.kpt_pos[:] = link_pos[:, self.kpt_link_idxs, :]
        # self.contact_forces[:] = entity.get_links_net_contact_force()[:, self.coll_idxs_local, :]
        self.control_forces[:] = entity.get_dofs_control_force(self.actuated_dof_idxs)
        self.wrist_pose[:, :3] = link_pos[:, self.wrist_link_idx, :]
        self.wrist_pose[:, 3:] = entity.get_links_quat()[:, self.wrist_link_idx, :]
        if len(self.kpt_markers) > 0:
            # update the kpt pos
            kpt_pos = self.entity.get_links_pos()[:, self.kpt_link_idxs, :]
            for i, marker in enumerate(self.kpt_markers):
                marker.set_pos(kpt_pos[:, i, :])

    def get_nan_envs(self):
        """ check along the env dim if self.dof_pos, self.dof_vel, self.kpt_pos have NaNs """
        assert self.initialized, "Robot not initialized"
        nan_mask = torch.isnan(self.dof_pos).any(dim=-1)
        for values in [self.dof_vel, self.kpt_pos, self.wrist_pose, self.control_forces]:
            nan_mask |= torch.isnan(values.flatten(start_dim=1)).any(dim=-1)
        return nan_mask

    def get_observations(self):
        assert self.initialized, "Robot not initialized"
        target_pos_diff = self.curr_targets - self.dof_pos
        obs_dict = {
            "dof_target_pos": target_pos_diff,
            "dof_pos": unscale(
                self.dof_pos,
                self.dof_limits[:, 0],
                self.dof_limits[:, 1],
            ),
            "dof_vel": self.dof_vel,
            "kpt_pos": self.kpt_pos.view(self.num_envs, -1),
            "wrist_pose": self.wrist_pose,
        }

        for k, scale in self.obs_scale.items():
            if k in obs_dict:
                obs_dict[k] *= scale
        return obs_dict

    def compute_obs_dim(self):
        dims = dict(
            qpos_dim = self.ndof,
            qpos_target_dim = self.ndof,
            qvel_dim = self.ndof,
            kpt_dim = int(len(self.kpt_link_names) * 3),
            wrist_dim = 7,
        )
        return sum(dims.values()), dims

    def translate_actions(self, actions, episode_length_buf):
        assert self.initialized, "Robot not initialized"
        assert actions.shape[-1] == self.action_dim, f"actions.shape={actions.shape} != {self.action_dim}"
        # first map low-dim action to joint targets
        joint_actions = actions[:, self.action_from_idxs]  # shape (N,ndof) because self.action_from_idxs may contain repeated idxs
        # assume actions are in [-1, 1], scale based on default init pos and limits
        upper_limit = self.dof_limits[:, 1] # shape (n_envs,)
        lower_limit = self.dof_limits[:, 0] # shape shape (n_envs,)
        if self.residual_qpos is not None:
            demo_t = torch.where(
                episode_length_buf >= self.residual_num_frames,
                self.residual_num_frames - 1,
                episode_length_buf
                )
            res_qpos = self.residual_qpos[demo_t] # shape (n_envs, ndof)
            self.curr_res_qpos[:] = res_qpos

        if self.action_mode == "residual":
            assert self.residual_qpos is not None and self.residual_num_frames is not None, "Residual qpos not set"
            # NOTE there's an implicit broadcast here bc actions is shape (n_envs, ndof). init_qpos is also shape (ndof,)
            # scale action to add to centering default init pos
            upper_margin = upper_limit - res_qpos  # shape (n_envs, 1)
            lower_margin = res_qpos - lower_limit  # shape (n_envs, 1)
            if self.res_cap:
                scale_trans, scale_rot = self.hybrid_scales
                upper_margin[:, self.wrist_dof_idxs[:3]] = scale_trans
                upper_margin[:, self.wrist_dof_idxs[3:]] = scale_rot
                lower_margin[:, self.wrist_dof_idxs[:3]] = -scale_trans
                lower_margin[:, self.wrist_dof_idxs[3:]] = -scale_rot
            # joint_actions is -1, 1, make it center around init_qpos
            upper = joint_actions >= 0
            scaled = torch.where(upper, joint_actions * upper_margin, joint_actions * lower_margin) # joint_actions has sign +-1!!
            joint_targets = res_qpos + scaled

        elif self.action_mode == "kinematic": # just all zeros
            assert self.residual_qpos is not None and self.residual_num_frames is not None, "Residual qpos not set"
            joint_targets = res_qpos

        elif self.action_mode == "relative":
            # translates policy action to a delta, then add to previous target & clamp
            deltas = joint_actions * self.relative_step_size # shape (n_envs, ndof)
            joint_targets = self.dof_pos + deltas

        elif self.action_mode == "absolute":
            joint_targets = lower_limit + (upper_limit - lower_limit) * (joint_actions + 1) / 2 # shape (n_envs, ndof)

        elif self.action_mode == "hybrid": # only residual on wrist joints, absolute on finger joints, use self.wrist_dof_idxs and self.finger_dof_idxs
            assert self.residual_qpos is not None and self.residual_num_frames is not None, "Residual qpos not set"
            # from objdex paper: wrist delta actions ±4 centimeters for transition and ±0.5 radian for rotation
            wrist_actions = torch.clamp(joint_actions[:, self.wrist_dof_idxs], -1, 1) # shape (n_envs, 6)
            wrist_trans_actions = joint_actions[:, self.wrist_dof_idxs[:3]] # shape (n_envs, 3)
            wrist_rot_actions = joint_actions[:, self.wrist_dof_idxs[3:]] # shape (n_envs, 3)
            scale_trans, scale_rot = self.hybrid_scales
            wrist_trans = self.curr_res_qpos[:, self.wrist_dof_idxs[:3]] + scale_trans * wrist_actions[:, :3] # shape (n_envs, 3)
            wrist_rot = self.curr_res_qpos[:, self.wrist_dof_idxs[3:]] + scale_rot * wrist_actions[:, 3:6] # shape (n_envs, 3)

            finger_actions = joint_actions[:, self.finger_dof_idxs]
            finger_targets = lower_limit[self.finger_dof_idxs] + (upper_limit[self.finger_dof_idxs] - lower_limit[self.finger_dof_idxs]) * (finger_actions + 1) / 2

            joint_targets = torch.concatenate([wrist_trans, wrist_rot, finger_targets], dim=-1)

        else:
            raise NotImplementedError
        # ignore the mimic values!
        target_dof_pos = joint_targets[:, self.joint_from_idxs] * self.joint_multipliers # shape (n_envs, ndof),
        target_dof_pos = torch.clamp(target_dof_pos, lower_limit, upper_limit)
        new_targets = self.action_moving_avg * target_dof_pos + (1 - self.action_moving_avg) * self.curr_targets
        new_targets = torch.clamp(new_targets, lower_limit, upper_limit)
        self.prev_targets[:] = self.curr_targets
        self.curr_targets[:] = new_targets
        return new_targets

    def map_joint_targets_to_actions(self, joint_targets):
        """
        Translate joint targets to [-1, 1] normalized actions
        """
        assert self.initialized, "Robot not initialized"
        assert joint_targets.shape[-1] == self.ndof, f"joint_targets.shape={joint_targets.shape} != {self.ndof}"
        assert self.action_mode != 'residual', "Residual action not supported"
        upper_limit = self.dof_limits[:, 1]
        lower_limit = self.dof_limits[:, 0]
        # given the desired joint targets, properly scale and shift to map to correct actions
        joint_targets = torch.clamp(joint_targets, lower_limit, upper_limit)
        joint_targets = (joint_targets - lower_limit) / (upper_limit - lower_limit) * 2 - 1
        # map to actions
        actions = torch.zeros((self.num_envs, self.action_dim), dtype=torch.float32, device=self.device)
        for i, idx in enumerate(self.action_from_idxs):
            actions[:, idx] = joint_targets[:, i]
        return actions

    def reset_idx(self, env_idxs=None, episode_start=None):
        assert self.initialized, "Robot not initialized"
        if env_idxs is None:
            env_idxs = range(self.num_envs) # reset all!
        if isinstance(env_idxs, int):
            env_idxs = [env_idxs]
        if len(env_idxs) == 0:
            return
        if episode_start is not None:
            assert episode_start.shape[0] == len(env_idxs), f"episode_start.shape={episode_start.shape} != {len(env_idxs)}"
        # reset value buffers
        if episode_start is not None and self.action_mode in ['residual', 'kinematic'] and self.residual_qpos is not None:
            # reset to residual qpos
            init_qpos = self.residual_qpos[episode_start] # shape (n_envs, ndof)
        else:
            init_qpos = self.init_qpos[env_idxs]

        self.dof_pos[env_idxs, :] = init_qpos
        self.dof_vel[env_idxs, :] = 0.0
        if self.is_eval and self.num_envs > 1:
            self.dof_pos[-1, :] = 0.0

        self.curr_targets[env_idxs, :] = init_qpos.clone()
        self.prev_targets[env_idxs, :] = init_qpos.clone()
        self.curr_res_qpos[env_idxs, :] = init_qpos.clone()
        # avoid doing this it it might NaN
        self.entity.set_dofs_position(
            position=self.dof_pos[env_idxs],
            dofs_idx_local=self.actuated_dof_idxs,
            zero_velocity=True,
            envs_idx=env_idxs,
        )

        self.entity.zero_all_dofs_velocity(envs_idx=env_idxs)

        prev_kpt_pos = self.kpt_pos[env_idxs, :].clone()
        new_kpt_pos = self.entity.get_links_pos()[:, self.kpt_link_idxs, :]
        self.kpt_pos[env_idxs] = new_kpt_pos[env_idxs]
        # reset wrist pose
        self.wrist_pose[env_idxs, :3] = new_kpt_pos[env_idxs, self.wrist_link_idx, :]
        self.wrist_pose[env_idxs, 3:] = self.entity.get_links_quat()[env_idxs, self.wrist_link_idx, :]

        # self.contact_forces[env_idxs, :] = 0.0
        self.control_forces[env_idxs, :] = 0.0
        self.episode_length_buf[env_idxs] = 0
        if episode_start is not None:
            self.episode_length_buf[env_idxs] = episode_start
        # self.episode_data = defaultdict(list) NOTE: only clear this after flush is called

    def check_env_idxs(self, env_idxs):
        if env_idxs is None:
            env_idxs = range(self.num_envs)
        if isinstance(env_idxs, int):
            env_idxs = [env_idxs]
        return env_idxs

    def set_joint_position(self, joint_targets, joint_idxs=[], env_idxs=None):
        assert self.initialized, "Robot not initialized"
        env_idxs = self.check_env_idxs(env_idxs)
        if len(joint_idxs) == 0:
            assert joint_targets.shape[-1] == self.ndof, f"joint_targets.shape={joint_targets.shape} != {self.ndof}"
            joint_idxs = self.actuated_dof_idxs
        self.entity.set_dofs_position(
            position=joint_targets,
            dofs_idx_local=joint_idxs,
            zero_velocity=True,
            envs_idx=env_idxs,
        )
        self.dof_pos[env_idxs] = joint_targets
        self.prev_targets[env_idxs] = joint_targets
        self.curr_targets[env_idxs] = joint_targets #self.entity.get_dofs_position(envs_idx=env_idxs)
        self.update_value_buffers()

    def get_wrist_xyz_joints(self):
        # return joint_idxs for wrist joints xyz
        idxs = []
        for word in ['forearm_tx', 'forearm_ty', 'forearm_tz']:
            for joint in self.actuated_joints:
                if word in joint.name:
                    idxs.append(joint.dof_idx_local)
        return idxs

    def get_control_force(self):
        return self.entity.get_dofs_control_force(dofs_idx_local=self.actuated_dof_idxs)

    def control_joint_position(self, joint_targets, joint_idxs=[], env_idxs=None):
        assert self.initialized, "Robot not initialized"
        env_idxs = self.check_env_idxs(env_idxs)
        if len(joint_idxs) == 0:
            assert joint_targets.shape[-1] == self.ndof, f"joint_targets.shape={joint_targets.shape} != {self.ndof}"
            joint_idxs = self.actuated_dof_idxs
        upper_limit = self.dof_limits[:, 1] # shape (n_envs,)
        lower_limit = self.dof_limits[:, 0] # shape shape (n_envs,)
        joint_targets = torch.clamp(joint_targets, lower_limit, upper_limit)
        self.entity.control_dofs_position(
            joint_targets,
            dofs_idx_local=joint_idxs,
            envs_idx=env_idxs,
        )
        self.prev_targets[:] = self.curr_targets
        self.curr_targets[:] = joint_targets
        return

    def step(self, actions, env_idxs=None):
        assert self.initialized, "Robot not initialized"
        target_dof_pos = self.translate_actions(actions, self.episode_length_buf)
        if env_idxs is not None:
            target_dof_pos = target_dof_pos[env_idxs]

        if self.wrist_only:
            self.entity.control_dofs_position(
                target_dof_pos[:, :6],
                self.wrist_dof_idxs,
                envs_idx=env_idxs
            )
        else:
            self.entity.control_dofs_position(
                target_dof_pos,
                self.actuated_dof_idxs,
                envs_idx=env_idxs
                )
        # NOTE: step the scene in the main thread
        self.episode_length_buf += 1
        return

    def get_bc_dist(self):
        # returns current joint target and residual qpos distance
        err = 0.5 * (self.curr_targets - self.curr_res_qpos).pow(2) # shape (n_envs, num_joints)
        # normalize with joint range
        err /= self.dof_range
        return err

    def flush_episode_data(self):
        if len(self.episode_data) == 0:
            return dict()
        jnames = self.actuated_dof_names
        kpt_names = self.kpt_link_names
        qpos_dict = dict()
        qpos_targets_dict = dict()
        for i, name in enumerate(jnames):
            joint_qpos = torch.stack(
                [step_data[:, i] for step_data in self.episode_data['qpos']], dim=0
            )
            if joint_qpos.shape[0] == 1: # shape (num_steps=1, num_envs)
                joint_qpos = joint_qpos[0]
            qpos_dict[name] = joint_qpos
            joint_target = torch.stack(
                [step_data[:, i] for step_data in self.episode_data['qpos_targets']], dim=0
            )
            if joint_target.shape[0] == 1: # shape (num_steps=1, num_envs)
                joint_target = joint_target[0]
            qpos_targets_dict[name] = joint_target # shape (num_step, num_envs)

        _data = dict(
            joint_qpos=qpos_dict,
            joint_targets=qpos_targets_dict,
            kpt_pos=torch.stack(self.episode_data['kpt_pos'], dim=0), # (num_frames, num_kpts, 3)
            kpt_names=kpt_names,
            wrist_pose=torch.stack(self.episode_data['wrist_pose'], dim=0), # (num_frames, 7)
            wrist_link_name=self.wrist_link_name,
        )
        self.episode_data = defaultdict(list)
        return _data

    def get_control_errors(self):
        err = torch.norm(self.curr_targets - self.dof_pos, p=2, dim=-1) # shape (n_envs,)
        return err

    def collect_data_step(self, collect_all_envs=False):
        if not self.collect_data:
            print("Not collecting data")
            return
        self.update_value_buffers()

        env_ids = [0]
        if collect_all_envs:
            env_ids = range(self.num_envs)
        # save torch tensors, should be more numerically accurate
        self.episode_data['qpos'].append(self.dof_pos[env_ids].clone())
        self.episode_data['qpos_targets'].append(self.curr_targets[env_ids].clone())
        self.episode_data['kpt_pos'].append(self.kpt_pos[env_ids].clone())
        self.episode_data['wrist_pose'].append(self.wrist_pose[env_ids].clone())