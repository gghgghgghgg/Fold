import os
import cv2
import torch
import numpy as np
import genesis as gs
from os.path import join
from collections import defaultdict

from fold.envs.math_utils import matrix_from_quat
from fold.asset_utils import get_asset_path


def get_arctic_object_cfg(name="box", convexify=True, decomp=True, texture_mesh=False):
    obj_cfg = {
        "name": name,
        "base_init_pos": [0.0, 0.0, 0.3], # make this above the ground to avoid startup delay
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "base_init_qpos": [0.0], # joint angle at rest
        "num_sample_vertics": 300,
        "convexify": convexify,
        "fixed": False,
        "actuated": False, # whether to control the object joint!
        "kp": 1000.0,
        "kv": 100.0,
        "force_range": 200.0,
        "collect_data": False,
        "offset_pos": [0.0, 0.0, 0.0], # useful for visualization
        "color": None,
        "show_link_frame": False,
    }

    data_dir = get_asset_path("arctic")
    urdf_path = join(data_dir, f"{name}/{name}.urdf")
    if decomp:
        urdf_path = join(data_dir, f"{name}/decomp/{name}_decomp.urdf")
        print(f"Using the decomp-ed urdf: {urdf_path}")
    assert os.path.exists(urdf_path), f"{urdf_path} does not exist"
    if texture_mesh:
        obj_cfg["texture_meshes"] = dict()
        for part in ["top", "bottom"]:
            mesh_fname = join(data_dir, f"{name}/{part}_textured.obj")
            mesh_tex = join(data_dir, f"{name}/{part}_texture.jpg")
            # mesh_tex = join(data_dir, f"{name}/decomp/material.jpg")
            obj_cfg["texture_meshes"][part] = dict(fname=mesh_fname, tex=mesh_tex)

    obj_cfg["urdf_path"] = urdf_path

    if name == "box":
        obj_cfg["base_init_pos"] = [0.0597, -0.2476,  1.0354]
        obj_cfg["base_init_quat"] = [-0.6413,  0.2875,  0.6467, -0.2964]

    obj_cfg['bottom_mesh_fname'] = join(data_dir, name, "bottom_watertight_tiny.stl")
    if not os.path.exists(obj_cfg['bottom_mesh_fname']):
        # try the obj file, some object don't have both stl and obj
        obj_cfg['bottom_mesh_fname'] = join(data_dir, name, "bottom_watertight_tiny.obj")
    obj_cfg['top_mesh_fname'] = join(data_dir, name, "top_watertight_tiny.stl")
    if not os.path.exists(obj_cfg['top_mesh_fname']):
        obj_cfg['top_mesh_fname'] = join(data_dir, name, "top_watertight_tiny.obj")
    assert os.path.exists(obj_cfg['bottom_mesh_fname']), f"{obj_cfg['bottom_mesh_fname']} does not exist"
    assert os.path.exists(obj_cfg['top_mesh_fname']), f"{obj_cfg['top_mesh_fname']} does not exist"
    return obj_cfg


class ArticulatedObject:
    """ Assume only one joint """
    def __init__(
        self,
        obj_cfg,
        device,
        scene,
        num_envs,
        obs_scale={'contact_norm': 0.05, 'root_lin_vel': 2.0, 'root_ang_vel': 0.25},
        demo_data=None,
        visualize_contact=False,
        disable_collision=False,
    ):
        self.name = obj_cfg["name"]
        self.cfg = obj_cfg
        self.kp = obj_cfg["kp"]
        self.kv = obj_cfg["kv"]
        self.force_range = obj_cfg["force_range"]
        self.current_kp = self.kp
        self.current_kv = self.kv
        self.current_force_range = self.force_range
        self.device = device
        self.initialized = False
        self.entity = None
        self.num_joints = 1
        self.dof_idxs = None
        self.obs_scale = obs_scale

        base_pos = obj_cfg["base_init_pos"]
        base_quat = obj_cfg["base_init_quat"]
        self.offset_pos = obj_cfg["offset_pos"]
        base_pos = [base_pos[i] + self.offset_pos[i] for i in range(3)]
        self.demo_states = None
        self.demo_dofs = None
        self.num_demo_frames = 0
        if demo_data is not None and demo_data != {}:
            # overwrite base pose
            base_pos, base_quat = self.set_demo_states(demo_data)
            self.num_demo_frames = self.demo_states.shape[0]

        self.init_pos = torch.tensor(
            base_pos, dtype=torch.float32, device=self.device
        )
        self.init_quat = torch.tensor(
            base_quat, dtype=torch.float32, device=self.device
        )

        self.init_qpos = torch.tensor(self.cfg["base_init_qpos"], dtype=torch.float32, device=self.device)
        if self.demo_states is not None:
            self.init_qpos = self.demo_states[0, 7:8].clone()

        self.num_envs = num_envs
        self.scene = scene

        entity = scene.add_entity(
            gs.morphs.URDF(
                fixed=self.cfg.get("fixed", False),
                file=self.cfg["urdf_path"],
                pos=self.init_pos.cpu().numpy(),
                quat=self.init_quat.cpu().numpy(),
                convexify=self.cfg.get("convexify", True),
                recompute_inertia=False,
                collision=(not disable_collision),
                visualization=(False if 'texture_meshes' in obj_cfg else True),
            ),
            visualize_contact=visualize_contact,
            # vis_mode="collision",
            surface=gs.surfaces.Smooth(color=obj_cfg.get("color")) if obj_cfg.get("color") is not None else None,
        )
        movable_joints = [joint for joint in entity.joints if joint.type in [gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC]]
        assert len(movable_joints) == 1, f"len(movable_joints)={len(movable_joints)}"

        self.texture_meshes = dict()
        if 'texture_meshes' in obj_cfg:
            # add two more meshes for top and bottom
            for part, info in obj_cfg["texture_meshes"].items():
                mesh = scene.add_entity(
                morph=gs.morphs.Mesh(
                    file=info["fname"],
                    fixed=False,
                    scale=0.001,
                    collision=False,
                ),
                surface=gs.surfaces.Default(
                        diffuse_texture=gs.textures.ImageTexture(
                            image_path=info["tex"],
                        ),
                    ),
                )
                self.texture_meshes[part] = mesh
        self.link_frames = dict()
        if obj_cfg.get("show_link_frame", False):
            for part in ["top", "bottom"]:
                for axis, color in zip(["x", "y", "z"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]):
                    mesh = scene.add_entity(
                        morph=gs.morphs.Mesh(
                            file=f"/home/mandi/chiral/assets/{axis}_axis.stl",
                            fixed=False,
                            scale=0.2,
                            collision=False,
                        ),
                        surface=gs.surfaces.Rough(color=color)
                    )
                    self.link_frames[f"{part}_{axis}"] = mesh



        self.n_links = len(entity.links)
        self.link_names = [link.name for link in entity.links] # this is ordered 'bottom', 'top'!!
        self.coll_idxs_global = [link.idx for link in entity.links if len(link.geoms) > 0]

        assert all([isinstance(joint.dof_idx_local, int) for joint in movable_joints]), "Only one dof per joint is supported"
        self.dof_idxs = [joint.dof_idx_local for joint in movable_joints]
        self.actuated = self.cfg.get("actuated", False)

        self.entity = entity
        self.initialize_value_buffers()
        self.initialized = True
        self.obs_dim, self.obs_dims = self.compute_obs_dim()
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)

        self.collect_data = self.cfg.get("collect_data", False)
        if self.collect_data:
            print("Collecting data from object")
        self.episode_data = defaultdict(list)
        self.post_built = False

    def post_scene_build_setup(self):
        obj_cfg = self.cfg
        if self.actuated:
            # set kp kv!
            self.set_joint_gains(self.kp, self.kv, self.force_range)
            demo_dofs = [] # use this to control actuated obj
            for i in range(self.demo_states.shape[0]):
                _state = self.demo_states[i]
                self.set_object_state(
                    root_pos=_state[:3][None].repeat(self.num_envs, 1),
                    root_quat=_state[3:7][None].repeat(self.num_envs, 1),
                    joint_qpos=_state[7:8][None].repeat(self.num_envs, 1),
                )
                dofs_pos = self.entity.get_dofs_position()
                demo_dofs.append(dofs_pos[0])
            self.demo_dofs = torch.stack(demo_dofs, dim=0)
        self.post_built = True

    def set_demo_states(self, demo_data):
        obj_pos = demo_data["obj_pos"]
        obj_quat = demo_data["obj_quat"]
        obj_arti = demo_data["obj_arti"]
        if len(obj_arti.shape) == 1:
            obj_arti = obj_arti[:, None]
        obj_pos += np.array(self.offset_pos)[None] # shift!!

        base_pos, base_quat = obj_pos[0], obj_quat[0]
        self.demo_states = np.concatenate([obj_pos, obj_quat, obj_arti], axis=1)
        self.demo_states = torch.tensor(self.demo_states, dtype=torch.float32, device=self.device)
        return base_pos, base_quat

    def set_to_demo_step(self, step=0):
        assert self.demo_states is not None, "demo_states is None"
        assert step < self.demo_states.shape[0], f"step={step} >= demo_states.shape[0]={self.demo_states.shape[0]}"
        demo_state = self.demo_states[step]
        self.set_object_state(
            root_pos=demo_state[:3][None].repeat(self.num_envs, 1),
            root_quat=demo_state[3:7][None].repeat(self.num_envs, 1),
            joint_qpos=demo_state[7:8][None].repeat(self.num_envs, 1),
        )
        return


    def fill_gain_tensor(self, val, num_dofs, num_envs, device):
        # shape is either 1d, num_dofs, or num_envs x num_dofs, fill in the missing dim
        if isinstance(val, (int, float)):
            val = torch.tensor([val], dtype=torch.float32, device=self.device)
            batched_val = val.repeat(num_dofs)[None].repeat(num_envs, 1)
        elif isinstance(val, torch.Tensor):
            if len(val.shape) == 1 and val.shape[0] == num_dofs:
                batched_val = val[None].repeat(num_envs, 1) # (num_dofs,) -> (1, num_dofs) -> (num_envs, num_dofs)
            elif len(val.shape) == 1 and val.shape[0] == 1:
                batched_val = val.repeat(num_dofs)[None].repeat(num_envs, 1) # (1,) -> (1, num_dofs) -> (num_envs, num_dofs)
            elif len(val.shape) == 1 and val.shape[0] == num_envs:
                batched_val = val[:, None].repeat(1, num_dofs) # (num_envs,) -> (num_envs, 1) -> (num_envs, num_dofs)
            else:
                assert val.shape == (num_envs, num_dofs), f"val.shape={val.shape}"
                batched_val = val
        else:
            raise not NotImplementedError
        if num_envs == 1:
            batched_val = batched_val[0] # remove first dim
        return batched_val.to(device)

    def interpolate_demo_states(self, multiplier=1.0):
        """ interpolate between demo states, shape (T, 8) -> (T*multiplier, 8) """
        assert self.demo_states is not None, "demo_states is None"
        old_T = self.demo_states.shape[0]
        if multiplier == 1.0:
            return self.demo_states

        from scipy.interpolate import interp1d
        from scipy.spatial.transform import Rotation, Slerp

        new_T = int(num_frames * multiplier)
        old_times = np.arange(old_T)
        target_times = np.linspace(0, old_T-1, new_T)
        # linear interpolation for position and articulation
        new_pos = torch.zeros((new_T, 3), dtype=torch.float32, device=self.device)
        func = interp1d(old_times, self.demo_states[:, :3].cpu().numpy(), axis=0)
        new_pos = func(target_times)

        # slerp for quaternion
        rotations = Rotation.from_quat(self.demo_states[:, 3:7].cpu().numpy())
        slerp = Slerp(old_times, rotations)
        new_quat = slerp(target_times).as_quat()

        # linear interpolation for articulation
        new_arti = torch.zeros((new_T, 1), dtype=torch.float32, device=self.device)
        func = interp1d(old_times, self.demo_states[:, 7:8].cpu().numpy(), axis=0)
        new_arti = func(target_times)

        new_states = torch.tensor(
            np.concatenate([new_pos, new_quat, new_arti], axis=1),
            dtype=torch.float32, device=self.device
        )
        assert new_states.shape[0] == new_T, f"new_states.shape={new_states.shape}, new_T={new_T}"
        return new_states

    def set_joint_gains(self, kp=None, kv=None, force_range=None, env_idxs=None):
        if kp is not None:
            self.current_kp = kp
        if kv is not None:
            self.current_kv = kv
        if force_range is not None:
            self.current_force_range = force_range

        num_envs = self.num_envs if env_idxs is None else len(env_idxs)
        dof_idxs = self.dof_idxs
        dof_idxs = [i for i in range(7)]
        num_dofs = len(dof_idxs)
        if kp is not None:
            batched_kp = self.fill_gain_tensor(kp, num_dofs, num_envs, self.device)
            self.entity.set_dofs_kp(batched_kp, dof_idxs, envs_idx=env_idxs)
        if kv is not None:
            batched_kv = self.fill_gain_tensor(kv, num_dofs, num_envs, self.device)
            self.entity.set_dofs_kv(batched_kv, dof_idxs, envs_idx=env_idxs)
        if force_range is not None:
            lower_f = self.fill_gain_tensor(-1 * force_range, num_dofs, num_envs, self.device)
            upper_f = self.fill_gain_tensor(force_range, num_dofs, num_envs, self.device)
            self.entity.set_dofs_force_range(lower_f, upper_f, dof_idxs, envs_idx=env_idxs)
        self.entity.zero_all_dofs_velocity(envs_idx=env_idxs)

    def sample_mesh_vertices(self, num_samples: int, part="top", seed=42) -> torch.Tensor:
        import trimesh
        mesh_fname = self.cfg[f"{part}_mesh_fname"]
        mesh = trimesh.load(mesh_fname)
        vertices = mesh.vertices
        num_vertices = vertices.shape[0]
        replace = num_samples > num_vertices
        # make this deterministic
        np.random.seed(seed)
        idxs = np.random.choice(num_vertices, num_samples, replace=replace)
        return torch.tensor(vertices[idxs], dtype=torch.float32)

    def initialize_value_buffers(self):
        self.root_pos = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.root_quat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)

        self.root_ang_vel = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.root_lin_vel = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

        self.part_pos = torch.zeros((self.num_envs, self.n_links, 3), dtype=torch.float32, device=self.device)
        self.part_quat = torch.zeros((self.num_envs, self.n_links, 4), dtype=torch.float32, device=self.device)

        self.dof_pos = torch.zeros((self.num_envs, self.num_joints), dtype=torch.float32, device=self.device)
        self.dof_vel = torch.zeros((self.num_envs, self.num_joints), dtype=torch.float32, device=self.device)

        self.contact_force = torch.zeros((self.num_envs, self.n_links, 3), dtype=torch.float32, device=self.device)
        self.state_diff = torch.zeros((self.num_envs, 8), dtype=torch.float32, device=self.device)

    def update_value_buffers(self):
        assert self.initialized, "Object not initialized"
        entity = self.entity
        assert self.dof_idxs is not None, "dof_idxs is None"
        self.root_pos[:] = entity.get_pos()
        self.root_quat[:] = entity.get_quat()
        self.root_ang_vel[:] = entity.get_ang()
        self.root_lin_vel[:] = entity.get_vel()

        self.part_pos[:] = entity.get_links_pos() # this contains both parts!
        self.part_quat[:] = entity.get_links_quat()

        self.dof_pos[:] = entity.get_dofs_position(self.dof_idxs)
        self.dof_vel[:] = entity.get_dofs_velocity(self.dof_idxs)
        self.contact_force[:] = entity.get_links_net_contact_force()
        if self.demo_states is not None:
            demo_goal_t = torch.where(
                self.episode_length_buf >= self.num_demo_frames - 1, self.num_demo_frames - 1, self.episode_length_buf + 1)
            self.state_diff[:] = self.demo_states[demo_goal_t] - torch.cat(
                [self.root_pos, self.root_quat, self.dof_pos], dim=-1)

    def get_nan_envs(self):
        """ check if obj state values has nan """
        nan_envs = torch.isnan(self.root_pos).any(dim=-1)
        for values in [self.root_quat, self.root_ang_vel, self.root_lin_vel, self.dof_pos, self.dof_vel]:
            nan_envs |= torch.isnan(values).any(dim=-1)
        return nan_envs

    def get_observations(self):
        assert self.initialized, "Object not initialized"
        obs_dict = {
            "parts_pos": self.part_pos.flatten(start_dim=1),
            "parts_quat": self.part_quat.flatten(start_dim=1),
            "dof_pos": self.dof_pos,
            "state_diff": self.state_diff,
            "root_ang_vel": self.root_ang_vel,
            "root_lin_vel": self.root_lin_vel,
        }
        for k, scale in self.obs_scale.items():
            if k in obs_dict:
                obs_dict[k] *= scale
        return obs_dict

    def get_part_pose(self, part='top'):
        idx = self.link_names.index(part)
        pose = torch.cat([self.part_pos[:, idx], self.part_quat[:, idx]], dim=-1)
        return pose

    def compute_obs_dim(self):
        dims = dict(
            parts_pose_dim=7*self.n_links,
            root_ang_vel_dim=3,
            root_lin_vel_dim=3,
            dof_pos_dim=self.num_joints,
            state_diff_dim=8,
        )
        return sum(dims.values()), dims

    def reset_idx(self, env_idxs=None, episode_start=None, reset_gains=False):
        assert self.initialized, "Object not initialized"
        if env_idxs is None:
            env_idxs = np.arange(self.num_envs)
        if isinstance(env_idxs, int):
            env_idxs = [env_idxs]
        if episode_start is not None:
            assert episode_start.shape[0] == len(env_idxs), f"reset_episode_start.shape={episode_start.shape}"

        init_qpos = self.init_qpos
        if episode_start is not None and torch.any(episode_start): # non-zero starts
            init_qpos = self.demo_states[episode_start, 7: 8].clone() # shape should be (N, 1)

        # try randomize the kp/kp
        if env_idxs is None:
            env_idxs = np.arange(self.num_envs)

        if self.actuated and reset_gains:
            dof_idxs = [i for i in range(7)]
            num_dofs = len(dof_idxs)
            rand_scale = torch.rand((len(env_idxs), 1)) * 0.3 + 0.7 # range [0.5, 1.0]
            kp = torch.ones(len(env_idxs), num_dofs) * self.kp * rand_scale
            kp = kp.to(self.device)
            kv = torch.ones(len(env_idxs), num_dofs) * self.kv * rand_scale
            kv = kv.to(self.device)
            self.entity.set_dofs_kp(kp, dof_idxs, envs_idx=env_idxs)
            self.entity.set_dofs_kv(kv, dof_idxs, envs_idx=env_idxs)


        self.dof_pos[env_idxs, :] = init_qpos
        self.dof_vel[env_idxs, :] = 0.0
        self.entity.set_dofs_position(
            position=self.dof_pos[env_idxs],
            dofs_idx_local=self.dof_idxs,
            zero_velocity=True,
            envs_idx=env_idxs,
        )
        init_pos = self.init_pos
        if episode_start is not None and torch.any(episode_start):
            init_pos = self.demo_states[episode_start, :3].clone()
        self.root_pos[env_idxs, :] = init_pos
        self.entity.set_pos(
            pos=self.root_pos[env_idxs],
            envs_idx=env_idxs,
        )

        init_quat = self.init_quat
        if episode_start is not None and torch.any(episode_start):
            init_quat = self.demo_states[episode_start, 3:7].clone()
        self.root_quat[env_idxs, :] = init_quat
        self.entity.set_quat(
            quat=self.root_quat[env_idxs],
            envs_idx=env_idxs,
        )

        self.root_ang_vel[env_idxs, :] = 0.0
        self.root_lin_vel[env_idxs, :] = 0.0
        self.entity.zero_all_dofs_velocity(envs_idx=env_idxs)

        self.part_pos[env_idxs, :, :] = 0.0
        self.part_quat[env_idxs, :, :] = 0.0

        self.contact_force[env_idxs, :] = 0.0
        self.episode_length_buf[env_idxs] = 0
        if episode_start is not None:
            self.episode_length_buf[env_idxs] = episode_start
        self.state_diff[env_idxs, :] = 0.0

    def reset(self):
        self.reset_idx()

    def set_object_state(self, root_pos, root_quat, joint_qpos, env_idxs=None):
        assert self.initialized, "Object not initialized"
        if root_pos.shape == (3,):
            root_pos = root_pos[None]

        self.entity.set_pos(
            pos=root_pos,
            envs_idx=env_idxs,
        )

        if root_quat.shape == (4,):
            root_quat = root_quat[None]
        self.entity.set_quat(
            quat=root_quat,
            envs_idx=env_idxs,
        )

        if len(joint_qpos.shape) == 1:
            joint_qpos = joint_qpos[None]
        assert joint_qpos.shape[-1] == self.num_joints, f"joint_qpos.shape={joint_qpos.shape}"
        self.entity.set_dofs_position(
            position=joint_qpos,
            dofs_idx_local=self.dof_idxs,
            zero_velocity=True,
            envs_idx=env_idxs,
        )
        self.entity.zero_all_dofs_velocity(envs_idx=env_idxs)
        self.update_value_buffers()

    def has_active_joint_gains(self):
        def any_positive(val):
            if val is None:
                return False
            if isinstance(val, torch.Tensor):
                return bool(torch.any(val > 0).item())
            return val > 0

        return any_positive(self.current_kp) and any_positive(self.current_kv)

    def step(self, env_idxs=None):
        assert self.initialized, "Object not initialized"
        assert self.post_built, "Must call post_scene_build_setup before stepping"
        if self.actuated and self.has_active_joint_gains():
            demo_goal_t = torch.where(
                self.episode_length_buf >= self.num_demo_frames - 1, self.num_demo_frames - 1, self.episode_length_buf + 1)
            targets = self.demo_dofs[demo_goal_t]
            self.entity.control_dofs_position(targets)
        if len(self.texture_meshes) > 0:
            for part, mesh in self.texture_meshes.items():
                pose = self.get_part_pose(part)
                mesh.set_pos(pose[:, :3])
                mesh.set_quat(pose[:, 3:7])
        if len(self.link_frames) > 0:
            # visualize the root frame on axis meshes:
             for part in ["top"]:
            # for part in ["top", "bottom"]:
                for axis in ["x", "y", "z"]:
                    mesh = self.link_frames[f"{part}_{axis}"]
                    pose = self.get_part_pose(part)
                    mesh.set_pos(pose[:, :3])
                    mesh.set_quat(pose[:, 3:7])

        self.episode_length_buf += 1
        return

    def flush_episode_data(self):
        if len(self.episode_data) == 0:
            return dict()
        _data = dict()
        for k in ['obj_pos', 'obj_quat', 'obj_arti']:
            # _data[k] = np.stack(self.episode_data[k], axis=0)
            _data[k] = torch.stack(self.episode_data[k], dim=0)
        self.episode_data = defaultdict(list)
        return _data

    def collect_data_step(self):
        if self.collect_data: # first env only
            self.update_value_buffers()
            self.episode_data["obj_pos"].append(self.root_pos[0])
            self.episode_data["obj_quat"].append(self.root_quat[0])
            self.episode_data["obj_arti"].append(self.dof_pos[0])

    def transform_part_vertices(self, mesh_verts, part="top"):
        """
        mesh_verts: (1, num_verts, 3)
        transform the mesh vertices to the current object pose
        """
        pose = self.get_part_pose(part)
        quat = pose[:, 3:7]
        matrices = matrix_from_quat(quat)
        offsets = pose[:, :3].unsqueeze(1)
        transformed = torch.einsum("nij,nkj->nki", matrices, mesh_verts) + offsets
        return transformed

    def get_part_vertices(self, part="top", num_verts=300):
        mesh_verts = self.sample_mesh_vertices(num_verts, part)[None].to(self.device)
        transformed = self.transform_part_vertices(mesh_verts, part)
        return transformed

    def get_object_vertices(self, num_verts=300):
        """ return:
        - current object's transformed vertices: (num_envs, num_verts * 2_parts, 3)
        - part ids: (num_envs, num_verts * 2_parts)
        """
        top_verts = self.get_part_vertices("top", num_verts)
        bottom_verts = self.get_part_vertices("bottom", num_verts)
        verts = torch.cat([top_verts, bottom_verts], dim=1)
        ids = torch.cat([torch.zeros(num_verts), torch.ones(num_verts)]).long()
        return verts, ids
