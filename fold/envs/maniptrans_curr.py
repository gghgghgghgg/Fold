import os
import torch
import numpy as np
import genesis as gs
from collections import deque

def get_maniptrans_cfg(kwargs=dict()):
    curr_cfg = {
        "type": "maniptrans",
        "gravity_range": [9.81, 0.005], # NOTE we use this for decay, but set the gravity to (g - 9.81) in solver
        "friction_range": [5.0, 1.0], # default is 1.0
        "wait_epochs": 100,
        "eps_finger_range": [0.06, 0.04],
        "eps_object_Prange": [0.06, 0.02],
        "eps_object_Rrange": [1.57, 0.52],
        "mode": "fixed", # fixed or auto
        "schedule": "exp", # exp or linear
        # if schedule=fixed: hard-coded decay
        "interval": 500, # epoch count  for mode=fixed
        "exp_ratio": 0.9, # decay ratio for mode=auto
        # maybe keep deque to track task progress
        "deque_len": 30, # length of reward deques
        "deque_freq": 1, # epoch count
        "seed": 42,
        "rew_thresholds": dict(task=0.6, con=0, imi=0, bc=0),
        "zero_epoch": 30000, # set all zeros once beyond this hardstop
    }
    # update with kwargs
    curr_cfg.update(kwargs)
    return curr_cfg

class ManipTransCurriculum:
    """
    all curriculums should have:
    - post_scene_build_setup()
    - get_reward_grads()
    - get_current_gains()
    - update_progress()
    - set_curriculum()
    - decay_reward_weights()
    New to maniptrans: use the eps_ values to determine environment early termination
    """
    def __init__(
            self,
            curr_cfg,
            task_object,
            reward_keys,
            num_envs,
            achieved_length,
            max_episode_length,
            sim, # need this to decay gravity in solver
            rigid_solver,
            ):
        self.curr_cfg = curr_cfg
        self.num_envs = num_envs
        self.object = task_object
        self.gain_range = {
            "gravity": curr_cfg['gravity_range'],
            "friction": curr_cfg['friction_range'],
            "eps_finger": curr_cfg['eps_finger_range'],
            "eps_object_P": curr_cfg['eps_object_Prange'],
            "eps_object_R": curr_cfg['eps_object_Rrange'],
        }
        self.init_gains = {k: v[0] for k, v in self.gain_range.items()}
        self.final_gains = {k: v[1] for k, v in self.gain_range.items()}
        self.curr_gains = self.init_gains.copy()

        self.decay_terms = [k for k in self.curr_gains.keys()]
        self.wait_epochs = curr_cfg['wait_epochs']

        # next get the params for different schedules
        self.mode = curr_cfg['mode']
        assert self.mode in ['fixed', 'auto'], "Invalid mode"
        self.fixed_interval = curr_cfg.get('interval', 500)
        self.auto_ratio = curr_cfg.get('exp_ratio', 0.9)
        self.schedule = curr_cfg.get('schedule', 'exp')
        assert self.schedule in ['exp', 'linear'], "Invalid auto schedule"

        self.deque_appends = 0
        self.deque_append_freq = curr_cfg['deque_freq']
        self.rew_deques = dict()
        self.deque_len = curr_cfg['deque_len']
        for key in reward_keys:
            assert key in ["task", "imi", "bc", "con"], f"Invalid reward key: {key}"
            self.rew_deques[key] = deque(maxlen=self.deque_len)
        self.ep_lens = deque(maxlen=self.deque_len)

        self.reward_keys = reward_keys
        self.rew_thresholds = curr_cfg['rew_thresholds']
        self.max_episode_length = max_episode_length
        # self.achieved_length = achieved_length # need to update this from env!
        self.seed = curr_cfg['seed']
        self.num_epoch_since_last_decay = 0
        self.num_epoch_since_zero = 0
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.zero_epoch = curr_cfg.get('zero_epoch', 50000)
        self.sim = sim
        self.rigid_solver = rigid_solver

    def post_scene_build_setup(self):
        return # dont need this since we don't decay solver params

    def get_reward_grads(self):
        return dict()

    def decay_reward_weights(self):
        return False

    def get_current_gains(self):
        return self.curr_gains.copy()

    def update_progress(self, rewards, achieved_length):
        # this is same as update_progress in our curriculum
        self.deque_appends += 1
        if self.deque_appends % self.deque_append_freq == 0:
            for k, val in rewards.items():
                assert k in self.rew_deques, f"Invalid key: {k}"
                self.rew_deques[k].append(val)
            self.deque_appends = 0
        # self.achieved_length = achieved_length
        self.ep_lens.append(achieved_length)

    def set_curriculum(self, epoch_num):
        """
        need to return:
        1. if the params are decayed to final
        2. if some decay happened (hence needs to reset reward tracker)
        3. verbal reason for decay/no decay, for debugging
        """
        self.num_epoch_since_last_decay += 1
        learning_stablized, reason = self.determine_decay(epoch_num)
        final_gains = True
        for key in self.decay_terms:
            if self.curr_gains[key] != self.final_gains[key]:
                final_gains = False
                break
        if final_gains:
            self.num_epoch_since_zero += 1

        gains_decayed = False
        if epoch_num > self.zero_epoch:
            if not final_gains:
                for key in self.decay_terms:
                    self.curr_gains[key] = self.final_gains[key]
                reason += f"final gains reached: {self.final_gains} "
                gains_decayed = True
            return final_gains, gains_decayed, reason

        if final_gains or not learning_stablized:
            return final_gains, gains_decayed, reason

        # now decay the gains
        if self.mode == "fixed":
            gains_decayed, reason = self.set_fixed_decay(epoch_num)
        elif self.mode == "auto":
            gains_decayed, reason = self.set_auto_decay(epoch_num)

        # now update the params
        if gains_decayed:
            # print(f"Epoch {epoch_num}: {reason}, {self.curr_gains}")
            curr_g = self.curr_gains["gravity"]
            if curr_g <= 0.01:
                curr_g = 0
            new_gravity = (0, 0, curr_g - 9.81)
            self.sim._gravity[:] = np.array(new_gravity)
            self.rigid_solver.set_gravity(new_gravity)
            reason += f"gravity decayed to {new_gravity} "

            new_friction = self.curr_gains["friction"]
            friction_vals = new_friction * torch.ones(self.rigid_solver.n_geoms)
            self.rigid_solver.set_geoms_friction(
                friction_vals, torch.arange(self.rigid_solver.n_geoms)
            )
            reason += f"friction decayed to {new_friction} "
        return final_gains, gains_decayed, reason

    def determine_early_term(self, obj_pos_err, obj_rot_err, finger_pos_err):
        """
        determine if the episode should be terminated early based on the eps_ values
        each err should have shape (num_envs, )
        """
        need_reset = torch.zeros(self.num_envs, dtype=torch.bool).to(obj_pos_err.device)
        for key, err in zip(["eps_object_P", "eps_object_R", "eps_finger"], [obj_pos_err, obj_rot_err, finger_pos_err]):
            if key in self.curr_gains.keys():
                need_reset = torch.logical_or(need_reset, err > self.curr_gains[key])
        return need_reset

    def set_fixed_decay(self, epoch_num):
        # decay at fixed intervals
        if epoch_num % self.fixed_interval != 0:
            return False, "Not decay interval"
        # decay the gains
        decayed = False
        reason = ""
        for key, param_range in self.gain_range.items():
            decayed = self.get_fixed_decay_param(epoch_num, param_range, self.zero_epoch, self.schedule, fixed_interval=self.fixed_interval)
            if decayed != self.final_gains[key]:
                self.curr_gains[key] = decayed
                reason += f"{key} decayed to {decayed} "
                decayed = True
        return decayed, reason

    def set_auto_decay(self, epoch_num):
        # always decay assume learning_stablized
        decayed = False
        reason = ""
        for key, curr_gain in self.curr_gains.items():
            new_gain = curr_gain * self.auto_ratio
            if new_gain < self.final_gains[key]:
                new_gain = self.final_gains[key]
            if new_gain != curr_gain:
                self.curr_gains[key] = new_gain
                decayed = True
                reason += f"{key} decayed to {new_gain} "
        return

    def get_fixed_decay_param(self, epoch_num, range, zero_epoch, schedule, fixed_interval=100):
        """
        Calculate the decayed parameter value based on the current epoch number.
        """
        assert epoch_num % fixed_interval == 0, f"Epoch {epoch_num} is not a multiple of the fixed interval {fixed_interval}"
        init_value, final_value = range
        # If we've passed the zero_epoch, return the final value
        if epoch_num >= zero_epoch:
            return final_value
        if schedule == 'exp':
            # Exponential decay
            # Calculate how many decay steps we need to reach final value at zero_epoch
            # final_value = init_value * (exp_ratio ^ num_steps)
            # Therefore, num_steps = log(final_value / init_value) / log(exp_ratio)
            if init_value == 0 or final_value == 0:
                # Handle edge cases
                if init_value == 0:
                    return 0  # Already at minimum
                if final_value == 0:
                    # Calculate ratio that reaches exactly 0 at zero_epoch
                    steps_to_zero = zero_epoch // fixed_interval
                    if steps_to_zero == 0:
                        return final_value
                    # Cannot reach exactly 0 with exponential decay, so get very close
                    current_step = epoch_num // fixed_interval
                    return init_value * (pow(1e-10, 1/steps_to_zero) ** current_step)

            # Calculate effective decay ratio to reach final value at zero_epoch
            steps_to_zero = zero_epoch // fixed_interval
            if steps_to_zero == 0:
                return final_value

            effective_ratio = pow(final_value / init_value, 1/steps_to_zero)
            current_step = epoch_num // fixed_interval

            return init_value * (effective_ratio ** current_step)
        elif schedule == 'linear':
            # Linear decay
            # Calculate decay amount per update
            steps_to_zero = max(1, zero_epoch // fixed_interval)
            step_size = (final_value - init_value) / steps_to_zero

            current_step = min(epoch_num // fixed_interval, steps_to_zero)
            return init_value + step_size * current_step

        else:
            raise ValueError("Unknown schedule type. Use 'exp' or 'linear'.")


    def determine_decay(self, epoch_num):
        reduce_gains = True
        reason = f"Epoch {epoch_num}: "
        if epoch_num < self.wait_epochs:
            reason += f"waiting for {self.wait_epochs} epochs"
            reduce_gains = False
        if self.mode == "fixed": # always True
            return reduce_gains, reason

        for key in self.rew_deques.keys():
            if len(self.rew_deques[key]) < self.deque_len:
                reduce_gains = False
                reason += f"{key} deque too short: {len(self.rew_deques[key]) } "
                break
            rew_mean = np.mean(self.rew_deques[key])
            rew_thres = self.rew_thresholds.get(key, 0.01)
            if rew_mean < rew_thres:
                reason += f"{key} reward too low: {rew_mean} "
                reduce_gains = False
            if not reduce_gains:
                break # no need to check other terms

        if self.num_epoch_since_last_decay < 10:
            reason += f"too soon until last decay: {self.num_epoch_since_last_decay} epochs "
            reduce_gains = False

        if len(self.ep_lens) < self.deque_len:
            reason += f"ep lens deque too short: {len(self.ep_lens)} "
            reduce_gains = False
        else:
            achieved_len = np.mean(self.ep_lens)
            if achieved_len < self.max_episode_length - 2:
                reason += f"max achieved length: {achieved_len} too low"
                reduce_gains = False
        return reduce_gains, reason