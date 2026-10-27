import os
import torch
import numpy as np
from os.path import join
from fold.envs.reward_utils import position_distance, rotation_distance, chamfer_distance, transform_contact
from fold.envs.math_utils import matrix_from_quat


def get_reward_cfg(last_n_frame=-1):
    reward_cfg = {
        "obj_pos_beta": 20.0,
        "obj_rot_beta": 5.0,
        "obj_arti_beta": 20.0,
        "obj_pos_weight": 2.0,
        "obj_rot_weight": 3.0,
        "obj_arti_weight": 5.0,

        "last_n_frame": last_n_frame,
        "multiply_task_rew": True,
        "multiply_all_rew": False,
        "task_rew_weight": 1.0,

        "exp_kpt_first": True,
        "imi_rew_weight":  0.0,
        "imi_wrist_weight": 0.0, # do a weighted avg between fingertip and wrist poses
        "imi_wrist_rot_beta": 3.0,
        "imi_wrist_pos_beta": 10.0,
        "imi_fingertip_beta": 20.0,

        "bc_rew_weight": 0.0,
        "bc_beta": 500.0,

        "contact_rew_weight": 0.0,
        "contact_rew_function": "exp", # exp or sigmoid
        "wrist_frame_contact": True,
        "contact_beta": 30.0,
        "contact_a": 100.0,
        "contact_b": 5.0,
        "multiply_frame_contact": True,
        "mask_zero_contact": True, # if both policy and demo has no contact, reward is 0 (1 if False)
        "contact_phase_penalty": 0,

        "mask_well_track": False,
        "scale_well_track": 1.0,
        "force_wrench_rew_weight": 0.0,
        "force_wrench_ref_path": "",
        "force_wrench_sigma_force": 25.0,
        "force_wrench_sigma_torque": 3.0,
        "force_wrench_sigma_distribution": 0.35,
        "force_wrench_torque_weight": 0.2,
        "force_wrench_decay_epochs": 0,
          "force_reward_curriculum_scale": False,
        "force_wrench_gate_mode": "smooth",
        "force_wrench_gate_pos_thresh": 0.02,
        "force_wrench_gate_rot_thresh": 0.3,
        "force_wrench_pose_gate": True,
        "force_anchor_rew_weight": 0.0,
        "force_anchor_sigma_pos": 0.03,
        "force_anchor_neighbor_k": 4,
        "force_anchor_direction_weight": 0.25,
        "force_anchor_use_direction": True,
        "force_anchor_normalize_reference": True,
        "force_anchor_eps": 1e-6,
        "force_reward_frame_start": 0,
        "force_reward_frame_end": -1,
        "force_penalty": 0.1,  # ~60 contact pairs in each env
        "action_penalty": 0.0,
        "objdex_baseline": False,
        "use_retarget_contact": False,
        "retarget_objframe": True, # if True, the contact is in the object frame, otherwise in the wrist frame

    }
    return reward_cfg

class RewardModule:
    def __init__(self, reward_cfg, demo_data, retarget_data, device):
        self.cfg = reward_cfg
        self.device = device
        self.last_n_frame = reward_cfg.get("last_n_frame", -1)
        self.exp_kpt_first = reward_cfg.get("exp_kpt_first", False)
        self.imi_rew_weight = reward_cfg["imi_rew_weight"]
        self.imi_wrist_weight = reward_cfg.get("imi_wrist_weight", 0.0)
        self.task_rew_weight = reward_cfg["task_rew_weight"]
        self.bc_rew_weight = reward_cfg.get("bc_rew_weight", 0.0)
        self.bc_wrist_weight = reward_cfg.get("bc_wrist_weight", 0.0)
        self.demo_data = demo_data
        self.multiply_task_rew = reward_cfg["multiply_task_rew"]
        self.multiply_all_rew = reward_cfg.get("multiply_all_rew", False)
        self.mask_well_track = reward_cfg.get("mask_well_track", False)
        self.scale_well_track = reward_cfg.get("scale_well_track", 1.0)
        self.obj_pos_beta = reward_cfg["obj_pos_beta"]
        self.obj_rot_beta = reward_cfg["obj_rot_beta"]
        self.obj_arti_beta = reward_cfg["obj_arti_beta"]
        if reward_cfg.get("objdex_baseline", False):
            print("Using the lambdas from ObjDex baseline")
            self.obj_pos_beta = 1.0
            self.obj_rot_beta = 20.0
            self.obj_arti_beta = 5.0
            self.imi_rew_weight = 0.0
            self.bc_rew_weight = 0.0
            self.task_rew_weight = 1.0
            self.contact_rew_weight = 0.0

        self.use_imi_rew = self.imi_rew_weight > 0.0
        self.use_bc_rew = self.bc_rew_weight > 0.0
        self.obj_pos_weight = reward_cfg["obj_pos_weight"] # not use because we multiply the rewards
        self.obj_rot_weight = reward_cfg["obj_rot_weight"]
        self.obj_arti_weight = reward_cfg["obj_arti_weight"]

        self.wrist_frame_contact = reward_cfg.get("wrist_frame_contact", False)
        self.use_retarget_contact = reward_cfg.get("use_retarget_contact", False) # if True, the loaded contact is from retargeted hands
        self.retarget_objframe = reward_cfg.get("retarget_objframe", True) # if True, the contact is in the object frame, otherwise in the wrist frame
        self.contact_rew_weight = reward_cfg.get("contact_rew_weight", 0.0)
        self.contact_beta = reward_cfg.get("contact_beta", 10.0)
        self.contact_a = reward_cfg.get("contact_a", 100.0)
        self.contact_b = reward_cfg.get("contact_b", 5.0)
        self.sigmoid_offset = torch.tensor(-1 / (1.0 + np.exp(-self.contact_b)), device=device) + 1.0
        self.contact_phase_penalty = reward_cfg.get("contact_phase_penalty", 0.0)
        self.multiply_frame_contact = reward_cfg.get("multiply_frame_contact", True)
        self.mask_zero_contact = reward_cfg.get("mask_zero_contact", True)
        self.contact_rew_function = reward_cfg.get("contact_rew_function", "exp")
        self.force_wrench_rew_weight = reward_cfg.get("force_wrench_rew_weight", 0.0)
        self.force_wrench_ref_path = reward_cfg.get("force_wrench_ref_path", "")
        self.force_wrench_sigma_force = reward_cfg.get("force_wrench_sigma_force", 25.0)
        self.force_wrench_sigma_torque = reward_cfg.get("force_wrench_sigma_torque", 3.0)
        self.force_wrench_sigma_distribution = reward_cfg.get("force_wrench_sigma_distribution", 0.35)
        self.force_wrench_torque_weight = reward_cfg.get("force_wrench_torque_weight", 0.2)
        self.force_wrench_decay_epochs = reward_cfg.get("force_wrench_decay_epochs", 0)
        self.force_wrench_gate_mode = reward_cfg.get("force_wrench_gate_mode", "smooth")
        self.force_wrench_gate_pos_thresh = reward_cfg.get("force_wrench_gate_pos_thresh", 0.02)
        self.force_wrench_gate_rot_thresh = reward_cfg.get("force_wrench_gate_rot_thresh", 0.3)
        self.force_wrench_pose_gate = reward_cfg.get("force_wrench_pose_gate", True)
        self.force_anchor_rew_weight = reward_cfg.get("force_anchor_rew_weight", 0.0)
        self.force_anchor_sigma_pos = reward_cfg.get("force_anchor_sigma_pos", 0.03)
        self.force_anchor_neighbor_k = max(int(reward_cfg.get("force_anchor_neighbor_k", 4)), 1)
        self.force_anchor_direction_weight = reward_cfg.get("force_anchor_direction_weight", 0.25)
        self.force_anchor_use_direction = reward_cfg.get("force_anchor_use_direction", True)
        self.force_anchor_normalize_reference = reward_cfg.get("force_anchor_normalize_reference", True)
        self.force_anchor_eps = reward_cfg.get("force_anchor_eps", 1e-6)
        self.force_reward_curriculum_scale = reward_cfg.get("force_reward_curriculum_scale", False)
        self.force_reward_curriculum_scale_value = 1.0
        self.force_wrench_epoch = 0
        self.left_contact_region_idx = None
        self.right_contact_region_idx = None
        self.anchor_positions_object = None
        self.anchor_force_object = None
        self.anchor_active_mask = None
        self.use_force_wrench_rew = self.force_wrench_rew_weight > 0.0
        self.use_force_anchor_rew = self.force_anchor_rew_weight > 0.0
        self.load_demo(demo_data, retarget_data, device)
        if self.use_force_wrench_rew or self.use_force_anchor_rew:
            self.load_force_reference(reward_cfg, device)

    def load_demo(self, demo_data, retarget_data, device):
        self.demo_tensors = dict()
        demo_keys = ["obj_pos", "obj_quat", "obj_arti"]
        if self.contact_rew_weight > 0.0:
            demo_keys += ["contact_links_left", "contact_links_right"]

        for key in demo_keys:
            assert key in demo_data, f"{key} not in demo_data"
            self.demo_tensors[key] = torch.tensor(
                demo_data[key], dtype=torch.float32, device=device
            )

        if self.use_imi_rew:
            for side in ['left', 'right']:
                key = f"kpts_{side}"
                self.demo_tensors[key] = torch.tensor(
                    retarget_data[side]['kpts_data']['kpt_pos'],
                    dtype=torch.float32, device=device
                )
        if self.contact_rew_weight > 0.0 or (self.use_imi_rew and self.imi_wrist_weight > 0.0):
            # load wrist pose for contact reward
            for side in ['left', 'right']:
                key = f"wrist_pose_{side}"
                if isinstance(retarget_data[side]['wrist_pose'], torch.Tensor):
                    self.demo_tensors[key] = retarget_data[side]['wrist_pose'].clone().to(device)
                else:
                    self.demo_tensors[key] = torch.tensor(
                        retarget_data[side]['wrist_pose'], dtype=torch.float32, device=device
                    )

        # check all the data have the same first dim size
        assert all(
            [self.demo_tensors[key].shape[0] == self.demo_tensors["obj_pos"].shape[0] for key in self.demo_tensors.keys()]
        ), f"First dim size mismatch: {[self.demo_tensors[key].shape[0] for key in self.demo_tensors.keys()]}"
        self.demo_length = self.demo_tensors["obj_pos"].shape[0]
        print(f"Loaded demo data with length {self.demo_length}")

    def get_demo_length(self):
        return self.demo_length

    def set_epoch(self, epoch_num):
        self.force_wrench_epoch = int(epoch_num)

    def set_curriculum_gains(self, curr_gains, init_gains):
        if not self.force_reward_curriculum_scale:
            self.force_reward_curriculum_scale_value = 1.0
            return
        curr_kp = curr_gains.get("kp", None) if curr_gains is not None else None
        init_kp = init_gains.get("kp", None) if init_gains is not None else None
        if curr_kp is None or init_kp is None or float(init_kp) <= 0.0:
            self.force_reward_curriculum_scale_value = 1.0
            return
        self.force_reward_curriculum_scale_value = float(np.clip(float(curr_kp) / float(init_kp), 0.0, 1.0))

    def set_contact_link_names(self, left_link_names, right_link_names):
        self.left_contact_region_idx = self._build_contact_region_index(left_link_names)
        self.right_contact_region_idx = self._build_contact_region_index(right_link_names)

    def _build_contact_region_index(self, link_names):
        region_idx = []
        for name in link_names:
            upper = name.upper()
            if "TH" in upper or "THUMB" in upper:
                region = 0
            elif "FF" in upper or "INDEX" in upper:
                region = 1
            elif "MF" in upper or "MIDDLE" in upper:
                region = 2
            elif "RF" in upper or "RING" in upper:
                region = 3
            elif "LF" in upper or "PINKY" in upper or "LITTLE" in upper:
                region = 4
            else:
                region = 5
            region_idx.append(region)
        return torch.tensor(region_idx, dtype=torch.long, device=self.device)

    def _load_region_force_reference(self, reward_cfg):
        seg_path = join(os.path.dirname(self.force_wrench_ref_path), "segmented_contacts_weighted.npy")
        if not os.path.exists(seg_path):
            return

        segmented_contacts = np.load(seg_path)
        max_frame = int(segmented_contacts[:, 0].max()) + 1
        region_force = np.zeros((max_frame, 2, 6), dtype=np.float32)
        frame_idx = segmented_contacts[:, 0].astype(np.int64)
        hand_idx = segmented_contacts[:, 1].astype(np.int64)
        region_idx = segmented_contacts[:, 2].astype(np.int64)
        force_mag = segmented_contacts[:, 6].astype(np.float32)
        valid = (hand_idx >= 0) & (hand_idx < 2) & (region_idx >= 0) & (region_idx < 6)
        np.add.at(region_force, (frame_idx[valid], hand_idx[valid], region_idx[valid]), force_mag[valid])

        start = reward_cfg.get("force_reward_frame_start", 0)
        end = reward_cfg.get("force_reward_frame_end", -1)
        if end is not None and end > 0:
            region_force = region_force[start:end]
        elif start > 0:
            region_force = region_force[start:]

        if region_force.shape[0] > self.demo_length:
            region_force = region_force[:self.demo_length]
        elif region_force.shape[0] < self.demo_length:
            pad = np.zeros((self.demo_length - region_force.shape[0], 2, 6), dtype=np.float32)
            region_force = np.concatenate([region_force, pad], axis=0)

        totals = region_force.sum(axis=(1, 2), keepdims=True)
        region_dist = np.divide(
            region_force,
            np.maximum(totals, 1e-6),
            out=np.zeros_like(region_force),
            where=totals > 1e-6,
        )
        self.demo_tensors["hand_region_force_dist"] = torch.tensor(
            region_dist, dtype=torch.float32, device=self.device
        )

    def load_force_reference(self, reward_cfg, device):
        ref_path = self.force_wrench_ref_path
        assert ref_path, "force_wrench_ref_path must be provided when a force-based reward is enabled"
        assert os.path.exists(ref_path), f"Force reference file not found: {ref_path}"
        ref_data = np.load(ref_path)
        start = reward_cfg.get("force_reward_frame_start", 0)
        end = reward_cfg.get("force_reward_frame_end", -1)

        def slice_frames(arr):
            if arr.shape[0] == self.demo_length:
                return arr
            if end is not None and end > 0:
                arr = arr[start:end]
            elif start > 0:
                arr = arr[start:]
            return arr

        ref_wrench = slice_frames(ref_data["hand_wrench_object"])
        self.demo_tensors["hand_wrench_object"] = torch.tensor(
            ref_wrench, dtype=torch.float32, device=device
        )
        assert self.demo_tensors["hand_wrench_object"].shape[0] == self.demo_length, (
            f"Force reference length mismatch: {self.demo_tensors['hand_wrench_object'].shape[0]} vs {self.demo_length}"
        )

        if self.use_force_anchor_rew:
            anchor_positions_object = ref_data["anchor_positions_object"]
            anchor_force_world = slice_frames(ref_data["anchor_force_world"])
            anchor_active_mask = slice_frames(ref_data["active_anchor_mask"])
            assert anchor_force_world.shape[0] == self.demo_length, (
                f"Anchor force reference length mismatch: {anchor_force_world.shape[0]} vs {self.demo_length}"
            )
            assert anchor_active_mask.shape[0] == self.demo_length, (
                f"Anchor mask reference length mismatch: {anchor_active_mask.shape[0]} vs {self.demo_length}"
            )
            anchor_force_world_t = torch.tensor(anchor_force_world, dtype=torch.float32, device=device)
            demo_quat = self.demo_tensors["obj_quat"]
            rot_t = matrix_from_quat(demo_quat).transpose(1, 2)
            self.anchor_positions_object = torch.tensor(anchor_positions_object, dtype=torch.float32, device=device)
            self.anchor_force_object = torch.einsum('fij,fkj->fki', rot_t, anchor_force_world_t)
            self.anchor_active_mask = torch.tensor(anchor_active_mask, dtype=torch.bool, device=device)

        self._load_region_force_reference(reward_cfg)

    def match_demo_state(self, demo_key, episode_length_buf):
        """ returns shape (num_envs, num_features) """
        assert demo_key in self.demo_tensors, f"Key {demo_key} not found in demo tensors"
        demo_t = torch.where(episode_length_buf >= self.demo_length, self.demo_length-1, episode_length_buf)
        # print(demo_t, episode_length_buf)
        return self.demo_tensors[demo_key][demo_t]

    def compute_task_reward(self, obj_pos, obj_quat, obj_arti, demo_pos, demo_quat, episode_length_buf):
        if obj_pos is None or obj_quat is None or obj_arti is None or self.task_rew_weight == 0.0:
            # dummy reward
            task_rew = torch.zeros(episode_length_buf.shape, device=episode_length_buf.device)
            return task_rew, dict(task_rew=task_rew)

        demo_arti = self.match_demo_state("obj_arti", episode_length_buf)
        pos_dist = position_distance(obj_pos, demo_pos)
        rot_dist = rotation_distance(obj_quat, demo_quat)
        # arti_dist = torch.mean((obj_arti - demo_arti)**2, dim=-1)

        if len(obj_arti.shape) > 1:
            obj_arti = obj_arti.flatten(start_dim=0)

        arti_dist = (obj_arti - demo_arti)**2 / 2.0

        # these should all be shape (num_envs, )!!
        obj_pos_rew = torch.exp(-self.obj_pos_beta * pos_dist)
        obj_rot_rew = torch.exp(-self.obj_rot_beta * rot_dist)
        obj_arti_rew = torch.exp(-self.obj_arti_beta * arti_dist)
        assert obj_pos_rew.shape == obj_rot_rew.shape == obj_arti_rew.shape, "Shape mismatch"
        # if joint is closed, don't compute arti reward
        if self.multiply_task_rew:
            task_rew = self.task_rew_weight * obj_pos_rew * obj_rot_rew * obj_arti_rew # scale is stil [0,1]
            # print(f"========= weighted task rew: {task_rew} | rew weight: {self.task_rew_weight * 10 } obj pos rew: {obj_pos_rew} | obj rot rew: {obj_rot_rew} | obj arti rew: {obj_arti_rew} =========")
        else:
            # obj_arti_rew = torch.where(demo_arti <= 0.0001, torch.zeros_like(obj_arti_rew), obj_arti_rew)
            task_rew = self.task_rew_weight * (
                self.obj_pos_weight * obj_pos_rew +
                self.obj_rot_weight * obj_rot_rew +
                self.obj_arti_weight * obj_arti_rew
                )

        # give a bonus if obj state is well tracked and articulation joint is open
        well_track = (pos_dist < 0.005) & (rot_dist < 0.1) & (arti_dist < 0.1)  & (demo_arti > 0.1)
        if self.scale_well_track > 1.0:
            task_rew = torch.where(well_track, task_rew * self.scale_well_track, task_rew)

        rew_dict = dict(
            pos_dist=pos_dist,
            rot_dist=rot_dist,
            arti_dist=arti_dist,
            obj_pos_rew=obj_pos_rew,
            obj_rot_rew=obj_rot_rew,
            obj_arti_rew=obj_arti_rew,
            task_rew=task_rew.clone(), # NOTE: otherwise it's the same tensor as the total reward
            well_track=well_track,
        )

        if self.last_n_frame > 0:
            # mask out rewards that are not from the last n frames
            tomask = torch.where(
                episode_length_buf < self.demo_length - self.last_n_frame,
                torch.zeros(task_rew.shape, device=task_rew.device, dtype=torch.bool),
                torch.ones(task_rew.shape, device=task_rew.device, dtype=torch.bool)
                )
            task_rew[tomask] = 0.0
        return task_rew, rew_dict

    def compute_keypoint_dist(self, keypoint_pos, episode_length_buf, left_hand=True):
        demo_key = "kpts_left" if left_hand else "kpts_right"
        demo_kpts = self.match_demo_state(demo_key, episode_length_buf)
        return position_distance(keypoint_pos, demo_kpts)

    def compute_wrist_reward(self, wrist_pose, episode_length_buf, side='left'):
        demo_wrist = self.match_demo_state(f"wrist_pose_{side}", episode_length_buf)
        wrist_rot_dist = rotation_distance(wrist_pose[:, 3:], demo_wrist[:, 3:])
        wrist_pos_dist = position_distance(wrist_pose[:, :3], demo_wrist[:, :3])
        rot_beta = self.cfg["imi_wrist_rot_beta"]
        pos_beta = self.cfg["imi_wrist_pos_beta"]
        wrist_rew = (torch.exp(-rot_beta * wrist_rot_dist) + torch.exp(-pos_beta * wrist_pos_dist)) / 2.0
        return wrist_rew, wrist_pos_dist, wrist_rot_dist

    def compute_imitation_reward(
        self,
        wrist_pose_left,
        wrist_pose_right,
        kpts_left: torch.Tensor,
        kpts_right: torch.Tensor,
        episode_length_buf: torch.Tensor,
    ):
        fingertip_dist_left = self.compute_keypoint_dist(kpts_left, episode_length_buf, left_hand=True)
        fingertip_dist_right = self.compute_keypoint_dist(kpts_right, episode_length_buf, left_hand=False)
        fingertip_dist = torch.mean( (fingertip_dist_left + fingertip_dist_right) / 2.0 , dim=-1) # (B, num_links) -> (B,)
        beta = self.cfg["imi_fingertip_beta"]
        if self.exp_kpt_first:
            fingertip_rew_left = torch.exp(- beta * fingertip_dist_left)
            fingertip_rew_right = torch.exp(- beta * fingertip_dist_right)
            fingertip_rew = torch.mean( (fingertip_rew_left + fingertip_rew_right) / 2.0 , dim=-1) # (B, num_links) -> (B,)
        else:
            fingertip_rew = torch.exp(-self.cfg["imi_fingertip_beta"] * fingertip_dist)
        #
        if self.imi_wrist_weight > 0.0:
            # do a rotation + position distance for wrist pose
            wrist_rew_left, pos_dist_left, rot_dist_left = self.compute_wrist_reward(wrist_pose_left, episode_length_buf, side='left')
            wrist_rew_right, pos_dist_right, rot_dist_right = self.compute_wrist_reward(wrist_pose_right, episode_length_buf, side='right')
            wrist_rew = (wrist_rew_left + wrist_rew_right) / 2.0
            imi_rew = self.imi_wrist_weight * wrist_rew + (1.0 - self.imi_wrist_weight) * fingertip_rew

            wrist_dist = torch.mean( (pos_dist_left + pos_dist_right) / 2.0 , dim=-1) # (B, num_links) -> (B,)
            keypoint_dist = self.imi_wrist_weight * wrist_dist + (1.0 - self.imi_wrist_weight) * fingertip_dist
        else:
            imi_rew = fingertip_rew
            keypoint_dist = fingertip_dist

        imi_rew *= self.imi_rew_weight

        rew_dict = dict(
            kpts_dist_left=fingertip_dist_left,
            kpts_dist_right=fingertip_dist_right,
            imi_rew=imi_rew,
            keypoint_dist=keypoint_dist,
        )
        if self.imi_wrist_weight > 0.0:
            rew_dict["wrist_pdist_left"] = pos_dist_left
            rew_dict["wrist_pdist_right"] = pos_dist_right
            rew_dict["wrist_rdist_left"] = rot_dist_left
            rew_dict["wrist_rdist_right"] = rot_dist_right
            rew_dict["wrist_rew_left"] = wrist_rew_left
            rew_dict["wrist_rew_right"] = wrist_rew_right
            rew_dict["fingertip_rew"] = fingertip_rew
            if not self.exp_kpt_first:
                rew_dict["fingertip_dist"] = fingertip_dist


        if self.last_n_frame > 0:
            # mask out rewards that are not from the last n frames
            tomask = torch.where(
                episode_length_buf < self.demo_length - self.last_n_frame,
                torch.zeros(imi_rew.shape, device=task_rew.device, dtype=torch.bool),
                torch.ones(imi_rew.shape, device=task_rew.device, dtype=torch.bool)
                )
            imi_rew[tomask] = 0.0
        return imi_rew, rew_dict

    def compute_force_wrench_reward(
        self,
        obj_pos,
        obj_quat,
        contact_forces,
        contact_link_pos_left,
        contact_link_valid_left,
        contact_link_pos_right,
        contact_link_valid_right,
        episode_length_buf,
        rew_dict,
    ):
        zero = torch.zeros(episode_length_buf.shape, device=episode_length_buf.device, dtype=torch.float32)
        if contact_forces is None or contact_link_pos_left is None or contact_link_pos_right is None:
            return zero, {"force_wrench_rew": zero}

        left_count = contact_link_pos_left.shape[2]
        right_count = contact_link_pos_right.shape[2]
        left_forces = contact_forces[:, :, :left_count]
        right_forces = contact_forces[:, :, left_count:left_count + right_count]
        left_mask = contact_link_valid_left.unsqueeze(-1).to(contact_forces.dtype)
        right_mask = contact_link_valid_right.unsqueeze(-1).to(contact_forces.dtype)

        rew_metrics = {}
        base_rew = None
        if (
            "hand_region_force_dist" in self.demo_tensors
            and self.left_contact_region_idx is not None
            and self.right_contact_region_idx is not None
            and self.left_contact_region_idx.numel() == left_count
            and self.right_contact_region_idx.numel() == right_count
        ):
            left_mag = torch.norm(left_forces, dim=-1) * contact_link_valid_left.to(contact_forces.dtype)
            right_mag = torch.norm(right_forces, dim=-1) * contact_link_valid_right.to(contact_forces.dtype)
            current_region_force = torch.zeros((contact_forces.shape[0], 2, 6), device=contact_forces.device, dtype=contact_forces.dtype)
            for region_idx in range(6):
                left_region_mask = (self.left_contact_region_idx == region_idx).to(contact_forces.dtype).view(1, 1, -1)
                right_region_mask = (self.right_contact_region_idx == region_idx).to(contact_forces.dtype).view(1, 1, -1)
                current_region_force[:, 0, region_idx] = (left_mag * left_region_mask).sum(dim=(1, 2))
                current_region_force[:, 1, region_idx] = (right_mag * right_region_mask).sum(dim=(1, 2))

            current_total = current_region_force.sum(dim=(1, 2), keepdim=True)
            current_region_dist = current_region_force / current_total.clamp_min(1e-6)
            ref_region_dist = self.match_demo_state("hand_region_force_dist", episode_length_buf)
            region_err = torch.abs(current_region_dist - ref_region_dist).sum(dim=(1, 2))
            sigma_distribution = max(self.force_wrench_sigma_distribution, 1e-6)
            base_rew = torch.exp(-torch.square(region_err / sigma_distribution))
            both_inactive = (current_total.squeeze(-1).squeeze(-1) <= 1e-6) & (ref_region_dist.sum(dim=(1, 2)) <= 1e-6)
            base_rew = torch.where(both_inactive, torch.ones_like(base_rew), base_rew)
            rew_metrics.update(
                force_wrench_region_err=region_err,
                force_wrench_current_force_sum=current_total.squeeze(-1).squeeze(-1),
                force_wrench_ref_force_sum=ref_region_dist.sum(dim=(1, 2)),
            )
        else:
            obj_origin = obj_pos[:, None, None, :]
            world_force = (left_forces * left_mask).sum(dim=(1, 2)) + (right_forces * right_mask).sum(dim=(1, 2))
            left_torque = torch.cross(contact_link_pos_left - obj_origin, left_forces, dim=-1) * left_mask
            right_torque = torch.cross(contact_link_pos_right - obj_origin, right_forces, dim=-1) * right_mask
            world_torque = left_torque.sum(dim=(1, 2)) + right_torque.sum(dim=(1, 2))

            rot_t = matrix_from_quat(obj_quat).transpose(1, 2)
            force_local = torch.einsum('bij,bj->bi', rot_t, world_force)
            torque_local = torch.einsum('bij,bj->bi', rot_t, world_torque)

            ref_wrench = self.match_demo_state("hand_wrench_object", episode_length_buf)
            force_err = torch.norm(force_local - ref_wrench[:, :3], dim=-1)
            torque_err = torch.norm(torque_local - ref_wrench[:, 3:], dim=-1)
            sigma_force = max(self.force_wrench_sigma_force, 1e-6)
            sigma_torque = max(self.force_wrench_sigma_torque, 1e-6)
            force_rew = torch.exp(-torch.square(force_err / sigma_force))
            torque_rew = torch.exp(-torch.square(torque_err / sigma_torque))
            torque_weight = float(np.clip(self.force_wrench_torque_weight, 0.0, 1.0))
            base_rew = (1.0 - torque_weight) * force_rew + torque_weight * torque_rew
            rew_metrics.update(
                force_wrench_force_err=force_err,
                force_wrench_torque_err=torque_err,
            )

        pose_gate = torch.ones_like(base_rew)
        if self.force_wrench_gate_mode == "smooth":
            if "obj_pos_rew" in rew_dict and "obj_rot_rew" in rew_dict:
                pose_gate = torch.clamp(rew_dict["obj_pos_rew"] * rew_dict["obj_rot_rew"], 0.0, 1.0)
        elif self.force_wrench_gate_mode == "hard":
            if "pos_dist" in rew_dict and "rot_dist" in rew_dict:
                pose_gate = (
                    (rew_dict["pos_dist"] <= self.force_wrench_gate_pos_thresh)
                    & (rew_dict["rot_dist"] <= self.force_wrench_gate_rot_thresh)
                ).float()

        decay_scale_value = 1.0
        if self.force_wrench_decay_epochs > 0:
            decay_scale_value = max(0.0, 1.0 - float(self.force_wrench_epoch) / float(self.force_wrench_decay_epochs))
        decay_scale = torch.full_like(base_rew, decay_scale_value)
        curriculum_scale = torch.full_like(base_rew, self.force_reward_curriculum_scale_value)

        wrench_rew = self.force_wrench_rew_weight * decay_scale * curriculum_scale * pose_gate * base_rew
        rew_metrics.update(
            force_wrench_rew=wrench_rew,
            force_wrench_pose_gate=pose_gate,
            force_wrench_decay_scale=decay_scale,
            force_wrench_curriculum_scale=curriculum_scale,
        )
        return wrench_rew, rew_metrics

    def compute_force_anchor_reward(
        self,
        obj_pos,
        obj_quat,
        contact_forces,
        contact_link_pos_left,
        contact_link_valid_left,
        contact_link_pos_right,
        contact_link_valid_right,
        episode_length_buf,
        rew_dict,
    ):
        zero = torch.zeros(episode_length_buf.shape, device=episode_length_buf.device, dtype=torch.float32)
        if (
            contact_forces is None
            or contact_link_pos_left is None
            or contact_link_pos_right is None
            or self.anchor_positions_object is None
            or self.anchor_force_object is None
            or self.anchor_active_mask is None
        ):
            return zero, {"force_anchor_rew": zero}

        left_count = contact_link_pos_left.shape[2]
        right_count = contact_link_pos_right.shape[2]
        all_pos_world = torch.cat([contact_link_pos_left, contact_link_pos_right], dim=2)
        all_valid = torch.cat([contact_link_valid_left, contact_link_valid_right], dim=2)
        all_force_world = torch.cat(
            [contact_forces[:, :, :left_count], contact_forces[:, :, left_count:left_count + right_count]],
            dim=2,
        )

        batch_size = all_pos_world.shape[0]
        flat_pos_world = all_pos_world.reshape(batch_size, -1, 3)
        flat_valid = all_valid.reshape(batch_size, -1).to(all_pos_world.dtype)
        flat_force_world = all_force_world.reshape(batch_size, -1, 3)

        obj_pose = torch.cat([obj_pos, obj_quat], dim=1)
        pos_object = transform_contact(flat_pos_world, obj_pose)
        rot_t = matrix_from_quat(obj_quat).transpose(1, 2)
        force_object = torch.einsum('bij,bmj->bmi', rot_t, flat_force_world)

        demo_t = torch.where(episode_length_buf >= self.demo_length, self.demo_length - 1, episode_length_buf)
        ref_anchor_force = self.anchor_force_object[demo_t]
        ref_active_mask = self.anchor_active_mask[demo_t].to(force_object.dtype)
        anchor_pos = self.anchor_positions_object.unsqueeze(0).expand(batch_size, -1, -1)

        rl_norm = torch.norm(force_object, dim=-1)
        ref_norm = torch.norm(ref_anchor_force, dim=-1)
        rl_mag = rl_norm * flat_valid
        ref_mag = ref_norm * ref_active_mask

        if self.force_anchor_normalize_reference:
            rl_mag = rl_mag / rl_mag.sum(dim=-1, keepdim=True).clamp_min(self.force_anchor_eps)
            ref_mag = ref_mag / ref_mag.sum(dim=-1, keepdim=True).clamp_min(self.force_anchor_eps)

        sigma = max(float(self.force_anchor_sigma_pos), self.force_anchor_eps)
        cross_dists = torch.cdist(pos_object, anchor_pos, p=2.0)
        cross_kernel = torch.exp(-torch.square(cross_dists / sigma))
        cross_kernel = cross_kernel * flat_valid.unsqueeze(-1) * ref_active_mask.unsqueeze(1)
        if self.force_anchor_neighbor_k > 0 and self.force_anchor_neighbor_k < anchor_pos.shape[1]:
            _, nn_idx = torch.topk(cross_dists, k=self.force_anchor_neighbor_k, dim=-1, largest=False)
            nn_mask = torch.zeros_like(cross_kernel)
            nn_mask.scatter_(2, nn_idx, 1.0)
            cross_kernel = cross_kernel * nn_mask

        cross_mag = torch.einsum('bma,bm,ba->b', cross_kernel, rl_mag, ref_mag)

        rl_self_dists = torch.cdist(pos_object, pos_object, p=2.0)
        rl_self_kernel = torch.exp(-torch.square(rl_self_dists / sigma))
        rl_self_kernel = rl_self_kernel * flat_valid.unsqueeze(-1) * flat_valid.unsqueeze(1)
        rl_self = torch.einsum('bmn,bm,bn->b', rl_self_kernel, rl_mag, rl_mag)

        ref_self_dists = torch.cdist(anchor_pos, anchor_pos, p=2.0)
        ref_self_kernel = torch.exp(-torch.square(ref_self_dists / sigma))
        ref_self_kernel = ref_self_kernel * ref_active_mask.unsqueeze(-1) * ref_active_mask.unsqueeze(1)
        ref_self = torch.einsum('ban,ba,bn->b', ref_self_kernel, ref_mag, ref_mag)

        mag_sim = cross_mag / torch.sqrt(rl_self * ref_self).clamp_min(self.force_anchor_eps)
        both_zero = (ref_mag.sum(dim=-1) <= self.force_anchor_eps) & (rl_mag.sum(dim=-1) <= self.force_anchor_eps)
        mag_sim = torch.where(both_zero, torch.ones_like(mag_sim), mag_sim)
        mag_sim = torch.clamp(mag_sim, 0.0, 1.0)

        base_rew = mag_sim
        dir_sim = torch.ones_like(base_rew)
        if self.force_anchor_use_direction:
            rl_unit = force_object / rl_norm.unsqueeze(-1).clamp_min(self.force_anchor_eps)
            ref_unit = ref_anchor_force / ref_norm.unsqueeze(-1).clamp_min(self.force_anchor_eps)
            pair_cos = torch.einsum('bmc,bac->bma', rl_unit, ref_unit)
            pair_cos = torch.clamp(pair_cos, -1.0, 1.0)
            pair_weights = cross_kernel * rl_mag.unsqueeze(-1) * ref_mag.unsqueeze(1)
            valid_dir = (rl_norm > self.force_anchor_eps).unsqueeze(-1) & (ref_norm > self.force_anchor_eps).unsqueeze(1)
            pair_weights = pair_weights * valid_dir.to(pair_weights.dtype)
            dir_num = (0.5 * (1.0 + pair_cos) * pair_weights).sum(dim=(1, 2))
            dir_den = pair_weights.sum(dim=(1, 2)).clamp_min(self.force_anchor_eps)
            dir_sim = dir_num / dir_den
            dir_sim = torch.where(both_zero, torch.ones_like(dir_sim), dir_sim)
            base_rew = (1.0 - self.force_anchor_direction_weight) * mag_sim + self.force_anchor_direction_weight * dir_sim

        pose_gate = torch.ones_like(base_rew)
        if self.force_wrench_gate_mode == "smooth":
            if "obj_pos_rew" in rew_dict and "obj_rot_rew" in rew_dict:
                pose_gate = torch.clamp(rew_dict["obj_pos_rew"] * rew_dict["obj_rot_rew"], 0.0, 1.0)
        elif self.force_wrench_gate_mode == "hard":
            if "pos_dist" in rew_dict and "rot_dist" in rew_dict:
                pose_gate = ((rew_dict["pos_dist"] <= self.force_wrench_gate_pos_thresh) & (rew_dict["rot_dist"] <= self.force_wrench_gate_rot_thresh)).float()

        curriculum_scale = torch.full_like(base_rew, self.force_reward_curriculum_scale_value)
        anchor_rew = self.force_anchor_rew_weight * curriculum_scale * pose_gate * base_rew
        return anchor_rew, {
            "force_anchor_rew": anchor_rew,
            "force_anchor_mag_sim": mag_sim,
            "force_anchor_dir_sim": dir_sim,
            "force_anchor_rl_sum": rl_mag.sum(dim=-1),
            "force_anchor_ref_sum": ref_mag.sum(dim=-1),
            "force_anchor_pose_gate": pose_gate,
            "force_anchor_curriculum_scale": curriculum_scale,
        }

    def contact_dist_to_rew(self, dist, function='exp'):
        if function == 'exp':
            return torch.exp(-self.contact_beta * dist)
        elif function == 'sigmoid':
            axb = self.contact_a * dist + self.contact_b
            rew = 1.0 / (1.0 + torch.exp(-axb)) + self.sigmoid_offset.to(dist.device)
            return rew
        else:
            raise ValueError(f"Unknown contact reward function: {function}")

    def compute_hand_contact_reward(
        self,
        contact_link_pos, # shape (N, num_obj_links * num_hand_links, 4)
        contact_link_valid, # shape (N, num_obj_links * num_hand_links, 1)
        wrist_pose, # shape (N, 7)
        obj_pose, # shape (N, 7)
        episode_length_buf,
        demo_obj_pose, # shape (N, 7)
        side='left',
    ):
        demo_wrist_pose = self.match_demo_state(f"wrist_pose_{side}", episode_length_buf)
        demo_contacts = self.match_demo_state(f"contact_links_{side}", episode_length_buf)
        # N, num_links * 2, 4 (last dim is contact pair ID) -> NOTE in ARCTIC, part_id=2 is 'bottom' link, part_id=1 is 'top'
        demo_positions = demo_contacts[:, :, :3]
        demo_valid_contact = demo_contacts[:, :, -1] > 0.0 # (part id is <= 0 if no contact)
        chamfer_dists = dict()
        contact_rewards = dict()

        positions = contact_link_pos[:, :, :3]
        valid_mask = contact_link_pos[:, :, -1] > 0.0
        for part_id in [1, 2]:
            # consider contact invalid if part_id is different
            demo_part_valid = (demo_contacts[:, :, -1] == part_id) & demo_valid_contact
            part_valid = (contact_link_pos[:, :, -1] == part_id ) & valid_mask

            for frame in ['obj', 'wrist']:
                if not self.wrist_frame_contact and frame == 'wrist':
                    continue
                demo_pose = demo_obj_pose if frame == 'obj' else demo_wrist_pose
                demo_in_frame = transform_contact(
                    demo_positions, demo_pose
                )
                pose = obj_pose if frame == 'obj' else wrist_pose
                in_frame = transform_contact(
                    positions, pose
                )
                dist = chamfer_distance(
                    in_frame, demo_in_frame, part_valid, demo_part_valid
                )
                chamfer_dists[f"CD_{side}_p{part_id}_frame_{frame}"] = dist
                rew = self.contact_dist_to_rew(dist, self.contact_rew_function)
                both_zero = (part_valid.sum(dim=-1) == 0) & (demo_part_valid.sum(dim=-1) == 0)
                if self.mask_zero_contact:
                    rew = torch.where(both_zero, torch.zeros_like(rew), rew)
                else:
                    rew = torch.where(both_zero, torch.ones_like(rew), rew)
                contact_rewards[f"conrew_{side}_p{part_id}_frame_{frame}"] = rew

        # per-part contact should be added, because sometimes full contact coverage is not feasible
        if self.multiply_frame_contact:
            # multiply the rew in obj & wrist frame such that one does not dominate the other
            contact_rews = []
            for part_id in [1, 2]:
                obj_rew = contact_rewards[f"conrew_{side}_p{part_id}_frame_obj"]
                if not self.wrist_frame_contact:
                    wrist_rew = 1.0
                else:
                    wrist_rew = contact_rewards[f"conrew_{side}_p{part_id}_frame_wrist"]
                multi_rew = obj_rew * wrist_rew
                contact_rewards[f"mul_conrew_{side}_p{part_id}"] = multi_rew
                contact_rews.append(multi_rew)
            contact_rew = torch.stack(contact_rews, dim=-1).mean(dim=-1)

        else: # sum & average all contact rewards
            contact_rew = sum(contact_rewards.values()) / len(contact_rewards)

        contact_dict = {**chamfer_dists, **contact_rewards, f"contact_rew_{side}": contact_rew}
        # add penalty for scenarios where demo has 0 contact but env has contact
        if self.contact_phase_penalty != 0.0:
            mismatch = torch.zeros_like(contact_rew).bool()
            num_mismatch = torch.zeros_like(contact_rew)
            for part_id in [1, 2]:
                demo_part_valid = (demo_contacts[:, :, -1] == part_id) & demo_valid_contact
                part_valid = (contact_link_pos[:, :, -1] == part_id ) & valid_mask
                part_mismatch = (demo_part_valid.sum(dim=-1) == 0) & (part_valid.sum(dim=-1) > 0)
                mismatch = mismatch | part_mismatch
                num_mismatch += part_mismatch.float()
            contact_rew[mismatch] += self.contact_phase_penalty
            contact_dict['num_mismatch'] = num_mismatch

        return contact_rew, contact_dict

    def compute_matched_contact_per_hand(
        self,
        contact_link_pos, # shape (N, num_obj_links, num_hand_links, 4)
        contact_link_valid, # shape (N, num_obj_links, num_hand_links, 1)
        episode_length_buf,
        # wrist_pose, # shape (N, 7)
        obj_pose, # shape (N, 7)
        demo_obj_pose,
        side='left',
        max_distance=1.0,
    ):
        """
        Instead of computing point cloud reward, here we assume demo contacts and policy contacts
        are from the same set of dex hand links and hence each contact point has a matched target position in the demo
        """
        demo_contacts = self.match_demo_state(f"contact_links_{side}", episode_length_buf)
        # NOTE the retargeted contacts are of shape (N, num_obj_parts=2, num_links, 4), first row is 'top' and second row is 'bottom'!
        # need to flip the order since policy contact has object link bottom first
        demo_contacts = demo_contacts.clone()[:, [1, 0], :, :] # (N, num_obj_links, num_hand_links, 4)
        assert demo_contacts.shape[1] == contact_link_pos.shape[1], f"Shape mismatch: {demo_contacts.shape} vs {contact_link_pos.shape}"
        assert demo_contacts.shape[2] == contact_link_pos.shape[2], f"Shape mismatch: {contact_link_pos.shape} vs {demo_contacts.shape}"
        demo_valids = demo_contacts[:, :, :, -1] > 0.0 # (part id is <= 0 if no contact)

        # need to reshape this to (N, num_obj_links * num_hand_links, 3) to do the transformation first
        if self.retarget_objframe:
            bsize, nparts, nlinks = demo_contacts.shape[:3]
            demo_pos_global = demo_contacts[:,:,:,:3].reshape(
                bsize, nparts * nlinks, 3
            ) # (N, num_obj_links * num_hand_links, 3)
            demo_pos = transform_contact(demo_pos_global, demo_obj_pose)
            demo_pos = demo_pos.reshape(bsize, nparts, nlinks, 3)

            policy_pos_global = contact_link_pos[:, :, :, :3].reshape(
                bsize, nparts * nlinks, 3
            ) # (N, num_obj_links, num_hand_links, 3)
            policy_pos = transform_contact(policy_pos_global, obj_pose)
            # then reshape it back!
            policy_pos = policy_pos.reshape(bsize, nparts, nlinks, 3)
        else:
            demo_pos = demo_contacts[:, :, :, :3]
            policy_pos = contact_link_pos[:, :, :, :3]


        both_invalid_mask = torch.logical_not(demo_valids) & torch.logical_not(contact_link_valid)
        # only one valid
        one_valid_mask = torch.logical_xor(demo_valids, contact_link_valid)
        # compute distance between demo and policy contact points
        dist = position_distance(demo_pos, policy_pos)
        if self.mask_zero_contact:
            # if both invalid, set distance to 0
            dist = torch.where(both_invalid_mask, torch.ones_like(dist) * max_distance, dist)
        else:
            # if both invalid, set distance to 0
            dist = torch.where(both_invalid_mask, torch.zeros_like(dist), dist)
        # if one valid, set distance to a large value
        dist = torch.where(one_valid_mask, torch.ones_like(dist) * max_distance, dist) # (B, 2, nlinks)

        # different ways to compute the distance:
        # if take mean, encourages all contacts to be close to the targets
        # if take min, encourages only one contact to be close to the target
        # part_dist = torch.mean(dist, dim=-1) # (N, num_obj_links=2)
        return dist

    def compute_matched_contact_reward(
        self,
        contacts_link_left, # shape (N, num_obj_links, num_hand_links)
        contacts_link_valid_left, # shape (N, num_obj_links, num_hand_links)
        contacts_link_right,
        contacts_link_valid_right,
        obj_pose,
        demo_obj_pose,
        episode_length_buf,
    ):
        rews = dict()
        contact_rew = 0
        for side, contacts, valids in zip(
            ['left', 'right'],
            [contacts_link_left, contacts_link_right],
            [contacts_link_valid_left, contacts_link_valid_right]
        ):
            part_dist = self.compute_matched_contact_per_hand(
                contacts, valids, episode_length_buf,
                obj_pose, demo_obj_pose, side=side
            )
            for i, part in enumerate(['bottom', 'top']):
                con_dist = part_dist[:, i]
                if self.exp_kpt_first:
                    con_rew = self.contact_dist_to_rew(con_dist, self.contact_rew_function).mean(dim=-1)
                else:
                    con_rew = self.contact_dist_to_rew(con_dist.mean(dim=-1), self.contact_rew_function)
                rews[f"conrew_{side}_{part}"] = con_rew
                rews[f"matched_condist_{side}_{part}"] = con_dist
                contact_rew += con_rew
        contact_rew /= 4.0
        contact_rew *= self.contact_rew_weight
        rews['con_rew'] = contact_rew
        return contact_rew, rews


    def compute_contact_reward(
        self,
        obj_pose,
        wrist_pose_left,
        wrist_pose_right,
        contacts_link_left,
        contacts_link_valid_left, # shape (N, num_obj_links, num_hand_links)
        contacts_link_right,
        contacts_link_valid_right,
        episode_length_buf,
    ):
        demo_obj_pos = self.match_demo_state("obj_pos", episode_length_buf)
        demo_obj_quat = self.match_demo_state("obj_quat", episode_length_buf)
        demo_obj_pose = torch.cat([demo_obj_pos, demo_obj_quat], dim=1)
        contact_rew_left, contact_dict_left = self.compute_hand_contact_reward(
            contacts_link_left, contacts_link_valid_left,
            wrist_pose_left, obj_pose, episode_length_buf, demo_obj_pose, side='left'
        )
        contact_rew_right, contact_dict_right = self.compute_hand_contact_reward(
            contacts_link_right, contacts_link_valid_right,
            wrist_pose_right, obj_pose, episode_length_buf, demo_obj_pose, side='right'
        )
        contact_rew = (contact_rew_left + contact_rew_right) / 2.0

        # add bonus to reward more contact points
        # num_points = torch.sum(contacts_link_valid_left, dim=-1) + torch.sum(contacts_link_valid_right, dim=-1)
        # more_points_rew = torch.mean(num_points.float() / 2.0, dim=-1) # average over num_links, then over batch # max is 1.0
        # contact_rew += more_points_rew * 0.01

        contact_rew *= self.contact_rew_weight
        contact_dict = {**contact_dict_left, **contact_dict_right} # merge the two dicts, should have no key overlap
        contact_dict['con_rew'] = contact_rew
        return contact_rew, contact_dict

    def reshape_contact_with_label(self, contact_link_pos, contact_link_valid):
        """
        Reshape the environment contact to better match with demos:
        In:
            contact_link_pos_left: (N, 2, num_hand_links, 3), 2 is each object part
            contact_link_valid_left: (N, 2, num_hand_links)
        Out:
            reshaped_contact_link_pos_left: (N, 2*num_hand_links, 4): 4 is (x, y, z, part_id), part_id is 1 or 2 or 0 if no valid
            reshaped contact_link_valid: (N, 2, num_hand_links) -> (N, 2*num_hand_links)
        """
        # first, add a part_id dim so (N, 2, num_links, 4) and last dim is part_id label
        labeled = torch.cat(
            [contact_link_pos, torch.zeros_like(contact_link_pos)[:, :, :, 0:1]], dim=-1
        ) # (N, 2, num_links, 4)
        # fill in the labels 1 or 2 if valid contact
        valid_reshaped = contact_link_valid[..., None] # (N, 2, num_links, 1)
        valid = torch.cat(
            [valid_reshaped, torch.zeros_like(valid_reshaped)], dim=-1
        ) # -> (N, 2, num_links, 2)
        for part_idx in [0, 1]: # bottom, top (sice obj urdf is flipped for parsing error)
            label_id = part_idx + 1
            # Need flipping, because in demos, id=2 is bottom
            label_id = 1 if label_id == 2 else 2
            labeled[:, part_idx, :, -1] = torch.where(
                contact_link_valid[:, part_idx, :],  label_id, 0
            )
            valid[:, part_idx, :, 1] = torch.where(
                contact_link_valid[:, part_idx, :], label_id, 0
            )

        # reshape to (N, num_links*2, 4)
        contact_link_reshaped = labeled.view(
            labeled.shape[0], -1, labeled.shape[-1]
        )
        contact_valid_reshaped = valid.view(
            valid.shape[0], -1, valid.shape[-1]
        )

        return contact_link_reshaped, contact_valid_reshaped

    def compute_reward(
        self,
        actions,
        bc_dist, # (num_envs, left+right hand num of joints)
        obj_pos,
        obj_quat,
        obj_arti,
        kpts_left,
        kpts_right,
        contact_link_pos_left,
        contact_link_valid_left,  # shape (N, num_obj_links, num_hand_links)
        contact_link_pos_right,
        contact_link_valid_right,
        wrist_pose_left,
        wrist_pose_right,
        contact_forces, # shape (N, num_obj_links, num_hand_links, 3)
        episode_length_buf,
    ):
        demo_pos = self.match_demo_state("obj_pos", episode_length_buf)
        demo_quat = self.match_demo_state("obj_quat", episode_length_buf)
        rew, rew_dict = self.compute_task_reward(
            obj_pos, obj_quat, obj_arti,
            demo_pos, demo_quat,
            episode_length_buf
        )
        if self.bc_rew_weight > 0.0:
            bc_rew = self.bc_rew_weight * torch.exp(-self.cfg["bc_beta"] * bc_dist)
            bc_rew = torch.mean(bc_rew, dim=-1)
            rew_dict["bc_dist"] = bc_dist.mean(dim=-1)
            rew_dict["bc_rew"] = bc_rew
            # rew += bc_rew

        if self.use_imi_rew:
            imi_rew, imi_rew_dict = self.compute_imitation_reward(
                wrist_pose_left, wrist_pose_right,
                kpts_left, kpts_right, episode_length_buf
            )
            if "well_track" in rew_dict and self.mask_well_track:
                # use well_track to mask out the imitation reward
                imi_rew = torch.where(rew_dict["well_track"], torch.zeros_like(imi_rew), imi_rew)
            # rew += imi_rew
            rew_dict.update(imi_rew_dict)

        if self.contact_rew_weight > 0.0:
            obj_pose = torch.cat([obj_pos, obj_quat], dim=1)
            demo_pose = torch.cat([demo_pos, demo_quat], dim=1)
            if self.use_retarget_contact:
                contact_rew, contact_dict = self.compute_matched_contact_reward(
                    contact_link_pos_left, contact_link_valid_left,
                    contact_link_pos_right, contact_link_valid_right,
                    obj_pose, demo_pose, episode_length_buf
                )
            else:
                left_reshaped, left_valid = self.reshape_contact_with_label(
                    contact_link_pos_left, contact_link_valid_left
                    )
                right_reshaped, right_valid = self.reshape_contact_with_label(
                    contact_link_pos_right, contact_link_valid_right
                    )
                contact_rew, contact_dict = self.compute_contact_reward(
                    obj_pose,
                    wrist_pose_left,
                    wrist_pose_right,
                    left_reshaped, left_valid,
                    right_reshaped, right_valid,
                    episode_length_buf
                )
            if "well_track" in rew_dict and self.mask_well_track:
                # use well_track to mask out the contact reward
                contact_rew = torch.where(rew_dict["well_track"], torch.zeros_like(contact_rew), contact_rew)

            # rew += contact_rew
            rew_dict.update(contact_dict)

        if self.use_force_wrench_rew:
            force_wrench_rew, force_wrench_dict = self.compute_force_wrench_reward(
                obj_pos, obj_quat,
                contact_forces,
                contact_link_pos_left, contact_link_valid_left,
                contact_link_pos_right, contact_link_valid_right,
                episode_length_buf,
                rew_dict,
            )
            rew_dict.update(force_wrench_dict)

        if self.use_force_anchor_rew:
            force_anchor_rew, force_anchor_dict = self.compute_force_anchor_reward(
                obj_pos, obj_quat,
                contact_forces,
                contact_link_pos_left, contact_link_valid_left,
                contact_link_pos_right, contact_link_valid_right,
                episode_length_buf,
                rew_dict,
            )
            rew_dict.update(force_anchor_dict)

        if self.multiply_all_rew:
            aux_rew = 1.0
            if self.use_imi_rew:
                aux_rew *= imi_rew
            if self.contact_rew_weight > 0.0:
                aux_rew *= contact_rew
            if self.bc_rew_weight > 0.0:
                aux_rew *= bc_rew
            if self.use_force_wrench_rew:
                aux_rew *= force_wrench_rew
            if self.use_force_anchor_rew:
                aux_rew *= force_anchor_rew
            rew += aux_rew * 0.5
        else:
            if self.use_imi_rew:
                rew += imi_rew
            if self.contact_rew_weight > 0.0:
                rew += contact_rew
            if self.bc_rew_weight > 0.0:
                rew += bc_rew
            if self.use_force_wrench_rew:
                rew += force_wrench_rew
            if self.use_force_anchor_rew:
                rew += force_anchor_rew

        if self.cfg["force_penalty"] > 0.0 and contact_forces is not None: # shape (B, 2, num_links*2, 3)
            force_norm = torch.norm(contact_forces, dim=-1).flatten(start_dim=1) # shape (B, 2*num_links*2) -> this goes to up to 5k
            high_force = torch.where(force_norm > 500.0, force_norm - 500.0, torch.zeros_like(force_norm))
            high_force = torch.mean(high_force, dim=-1) # shape (B, )
            force_penalty = self.cfg["force_penalty"] * high_force
            rew -= force_penalty
            rew_dict["force_penalty"] = force_penalty

        # rew *= 0.02
        # penalize action to stay close to 0
        if self.cfg["action_penalty"] > 0.0:
            action_penalty = torch.mean(actions**2, dim=-1) * self.cfg["action_penalty"]
            rew -= action_penalty
            rew_dict["action_penalty"] = action_penalty
        return rew, rew_dict

    def get_reward_keys(self):
        keys = []
        if self.task_rew_weight > 0:
            keys.append('task')
        if self.imi_rew_weight > 0:
            keys.append('imi')
        if self.contact_rew_weight > 0:
            keys.append('con')
        if self.bc_rew_weight > 0:
            keys.append('bc')
        return keys