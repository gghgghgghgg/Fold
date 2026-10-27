import os
import torch
import numpy as np
import genesis as gs
from collections import deque

def get_curriculum_cfg(kwargs=dict()):
    curr_cfg = {
        "kp_init": 1000.0,
        "kv_init": 10.0,
        "force_range_init": 50.0,
        "wait_epochs": 2000,
        "decay_rew": False,
        "schedule": "exp", # or exp or uniform
        "deque_len": 100, # length of reward deques
        "interval": 500, # epoch count
        "first_ratio": 0.3, # determines how much to reduce the gain at the end of firt stage
        "first_stop_iter": 7000, # epoch count
        "second_stop_iter": 16000, # epoch count
        "fixed_mode": "exp", # or exp

        # if schedule=exp: auto linear
        "deque_freq": 1, # epoch count
        "grad_threshold": 0.0001,
        "gain_mode": "all", # which terms to reduce gain

        # if schedule=uniform: auto uniform distribution of gains
        "uniform_mode": "fast", # or slow
        # "upper_ratio": 0.9, # upper = curr_upper * (upper_ratio)
        # "lower_ratio": 0.8, # if fast: lower=curr_lower * lower_ratio, if slow: lower=curr_upper * lower_ratio
        "upper_ratios": dict(kp=0.5, kv=0.9, fr=0.95),
        "lower_ratios": dict(kp=0.5, kv=0.8, fr=0.9),
        "staged_gain_decay": False,
        "staged_kp_thresholds": [40.0, 5.0],
        "staged_kp_ratios": [0.95, 0.85, 0.6],
        "staged_kp_lower_ratios": [0.9, 0.8, 0.7],
        "seed": 42,
        "decay_solimp": False,
        "solip_multiplier": 0.98,
        "d0_lower": 0.9,
        "dmid_lower": 0.95,
        "tconst_lower": 0.1,
        "tconst_upper": 0.15, # 0.2 is too soft
        "rew_thresholds": dict(task=0.5, con=0.05, imi=0.05, bc=0.05),
        "decay_ep_len_threshold": None,
        "decay_min_epochs": 40,
        "resample_every_epoch": -1, # if true and using uniform, re-sample gains every epoch
        "skip_grad": False, # if true, skip gradient check
        "zero_epoch": 30000, # set all zeros once beyond this hardstop
        "zero_trigger_kp_ratio": None, # if set, switch all gains to zero once curr_kp/init_kp falls below this ratio
        "dialback_ep_len": 50,
        "dialback_min_epochs": 500,
        "dialback_ratios": dict(kp=0.98, kv=0.98, fr=0.98), # don't fully reset to prev gains, instead decay by this ratio
        "enhanced_dialback": False,
        "enhanced_dialback_min_epochs": 100,
        "enhanced_dialback_ep_len": 160,
        "enhanced_dialback_ratio": 0.98,
        "dialback_task_rew_threshold": 0.4,
        "dialback_relative_ep_drop": 15.0,
        "dialback_relative_task_drop": 0.05,
        "dialback_stall_epochs": 500,
    }
    # update with kwargs
    curr_cfg.update(kwargs)
    return curr_cfg

SCHEDULE_OPTIONS=["fixed", "exp", "uniform"]

class Curriculum:
    def __init__(self, curr_cfg, task_object, reward_keys, num_envs, achieved_length, max_episode_length):
        self.curr_cfg = curr_cfg
        self.num_envs = num_envs
        self.object = task_object
        self.init_gains = {
            "kp": curr_cfg['kp_init'],
            "kv": curr_cfg['kv_init'],
            "fr": curr_cfg['force_range_init'],
        }
        upper_init = curr_cfg.get("curriculum_upper_init")
        lower_init = curr_cfg.get("curriculum_lower_init")
        self.curr_gains = upper_init.copy() if upper_init is not None else self.init_gains.copy()
        self.curr_gains_lower = lower_init.copy() if lower_init is not None else self.curr_gains.copy() # use for uniform mode
        for key in self.init_gains:
            assert 0.0 <= self.curr_gains_lower[key] <= self.curr_gains[key] <= self.init_gains[key], f"Invalid initial {key} bounds"
        previous_uppers = curr_cfg.get("curriculum_previous_upper_init")
        previous_lowers = curr_cfg.get("curriculum_previous_lower_init")
        assert (previous_uppers is None) == (previous_lowers is None)
        self.gain_history = [] # keep a list of gains for prev epochs
        if previous_uppers is not None:
            if isinstance(previous_uppers, dict):
                previous_uppers, previous_lowers = [previous_uppers], [previous_lowers]
            assert len(previous_uppers) == len(previous_lowers)
            for previous_upper, previous_lower in zip(previous_uppers, previous_lowers):
                for key in self.init_gains:
                    assert 0.0 <= previous_lower[key] <= previous_upper[key] <= self.init_gains[key]
                    assert previous_upper[key] >= self.curr_gains[key]
                self.gain_history.append(dict(gains=previous_upper.copy(), lower=previous_lower.copy(), metrics={}))
        self.gain_mode = curr_cfg['gain_mode']
        self.decay_rew = curr_cfg['decay_rew']
        # first figure out which terms to decay:
        decay_terms = []
        if self.gain_mode == "all":
            decay_terms = ["kp", "kv", "fr"]
        else:
            if "kp" in self.gain_mode:
                decay_terms.append("kp")
            if "kv" in self.gain_mode:
                decay_terms.append("kv")
            if "fr" in self.gain_mode:
                decay_terms.append("fr")
            if not decay_terms:
                raise ValueError("Invalid gain mode")
        self.decay_terms = decay_terms
        self.wait_epochs = curr_cfg['wait_epochs']

        # next get the params for different schedules
        self.schedule = curr_cfg['schedule']
        assert self.schedule in SCHEDULE_OPTIONS, "Invalid schedule"

        self.fixed_mode = curr_cfg['fixed_mode']
        assert self.fixed_mode in ['lin', 'exp', 'uniform'], "Invalid fixed mode"

        self.uniform_mode = curr_cfg['uniform_mode']
        self.obj_ndof = 7 # base + articulation
        assert self.uniform_mode in ['fast', 'slow'], "Invalid uniform mode"
        self.deque_appends = 0
        self.deque_append_freq = curr_cfg['deque_freq']

        self.decay_solimp = curr_cfg['decay_solimp']
        self.solip_multiplier = curr_cfg['solip_multiplier']
        self.d0_lower = curr_cfg['d0_lower']
        self.dmid_lower = curr_cfg['dmid_lower']
        self.tconst_lower = curr_cfg['tconst_lower']
        self.staged_gain_decay = curr_cfg.get('staged_gain_decay', False)
        self.staged_kp_thresholds = curr_cfg.get('staged_kp_thresholds', [40.0, 5.0])
        self.staged_kp_ratios = curr_cfg.get('staged_kp_ratios', [0.95, 0.85, 0.6])
        self.staged_kp_lower_ratios = curr_cfg.get('staged_kp_lower_ratios', [0.9, 0.8, 0.7])
        assert len(self.staged_kp_thresholds) == 2
        assert len(self.staged_kp_ratios) == len(self.staged_kp_lower_ratios) == 3
        assert self.staged_kp_thresholds[0] > self.staged_kp_thresholds[1] > 0
        self.tconst_upper = curr_cfg['tconst_upper']

        self.upper_ratios = curr_cfg.get('upper_ratios', dict(kp=0.9, kv=0.9, fr=0.9))
        self.lower_ratios = curr_cfg.get('lower_ratios', dict(kp=0.8, kv=0.8, fr=0.8))

        self.rew_deques = dict()
        self.rew_grads = dict()
        self.deque_len = curr_cfg['deque_len']
        for key in reward_keys:
            assert key in ["task", "imi", "bc", "con"], f"Invalid reward key: {key}"
            self.rew_deques[key] = deque(maxlen=self.deque_len)
            self.rew_grads[key] = 1.0
        self.ep_lens = deque(maxlen=self.deque_len)

        self.reward_keys = reward_keys
        self.grad_threshold = curr_cfg['grad_threshold']
        self.rew_thresholds = curr_cfg['rew_thresholds']
        self.decay_ep_len_threshold = curr_cfg.get('decay_ep_len_threshold')
        if self.decay_ep_len_threshold is None:
            self.decay_ep_len_threshold = max_episode_length - 2
        assert 0 < self.decay_ep_len_threshold <= max_episode_length
        self.decay_min_epochs = curr_cfg.get('decay_min_epochs', 40)
        assert self.decay_min_epochs > 0
        self.enhanced_dialback = curr_cfg.get('enhanced_dialback', False)
        self.last_dialback_reason = ""
        self.max_episode_length = max_episode_length
        # self.achieved_length = achieved_length # need to update this from env!
        self.seed = curr_cfg['seed']
        self.num_epoch_since_last_decay = 0
        self.num_epoch_since_zero = 0
        self.resample_every_epoch = curr_cfg['resample_every_epoch']
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.skip_grad = curr_cfg['skip_grad']
        self.zero_epoch = curr_cfg.get('zero_epoch', 50000)
        self.zero_trigger_kp_ratio = curr_cfg.get('zero_trigger_kp_ratio')

    def _set_zero_gains(self):
        for key in self.decay_terms:
            self.curr_gains[key] = 0.0
            self.curr_gains_lower[key] = 0.0

    def _should_trigger_zero_from_kp_ratio(self, epoch_num):
        if self.zero_trigger_kp_ratio is None:
            return False, ""
        if "kp" not in self.decay_terms:
            return False, ""
        init_kp = float(self.init_gains.get("kp", 0.0))
        if init_kp <= 0.0:
            return False, ""
        curr_ratio = float(self.curr_gains["kp"]) / init_kp
        if curr_ratio > float(self.zero_trigger_kp_ratio):
            return False, ""
        return True, f"Epoch {epoch_num}: Gains set to zero by kp ratio trigger ({curr_ratio:.4f} <= {float(self.zero_trigger_kp_ratio):.4f})"

    def post_scene_build_setup(self):
        if self.curr_cfg.get("curriculum_upper_init") is not None:
            self.reset_object_gains()
        if self.decay_solimp:
            new_params = torch.tensor([self.tconst_upper, 1.0, self.d0_lower, self.dmid_lower, 0.001, 0.5, 2.0])
            self.object.entity._solver.set_geom_sol_params(new_params)
        return

    def set_fixed_decay(self, epoch_num):
        """
        2 stage fast and slow linear decay, or 1 stage exponential decay, only depending on epoch_num
        """
        interval = self.curr_cfg['interval']
        if epoch_num % interval != 0 or epoch_num == 0:
            return False, "wrong interval"
        if self.fixed_mode == "exp" or self.fixed_mode == "uniform":
            for k in self.decay_terms:
                fixed_ratio = self.upper_ratios.get(k, 0.9)
                # if self.curr_gains[k] < 1 and k == 'kp':
                #     fixed_ratio = 0.96
                self.curr_gains[k] *= fixed_ratio
            # if uniform, decay lower bound as well
            if self.fixed_mode == "uniform":
                for k in self.decay_terms:
                    low_ratio = self.lower_ratios[k]
                    if self.uniform_mode == "fast":
                        self.curr_gains_lower[k] *= low_ratio
                    elif self.uniform_mode == "slow":
                        self.curr_gains_lower[k] = self.curr_gains[k] * low_ratio
        elif self.fixed_mode == "lin":
            # two stage linear decay
            first_stop = self.curr_cfg['first_stop_iter']
            second_stop = self.curr_cfg['second_stop_iter']
            if epoch_num < first_stop:
                mid_gains = {k: v * self.curr_cfg['first_ratio'] for k, v in self.init_gains.items()}
                frac = epoch_num / first_stop
                new_gains = {
                    "kp": self.init_gains['kp'] * (1 - frac) + mid_gains['kp'] * frac,
                    "kv": self.init_gains['kv'] * (1 - frac) + mid_gains['kv'] * frac,
                    "fr": self.init_gains['fr'] * (1 - frac) + mid_gains['fr'] * frac,
                }
            elif epoch_num < second_stop:
                frac = (epoch_num - first_stop) / (second_stop - first_stop)
                new_gains = {
                    "kp": mid_gains['kp'] * (1 - frac) + self.init_gains['kp'] * frac,
                    "kv": mid_gains['kv'] * (1 - frac) + self.init_gains['kv'] * frac,
                    "fr": mid_gains['fr'] * (1 - frac) + self.init_gains['fr'] * frac,
                }
            else:
                new_gains = {k: 0.0 for k in self.init_gains}

            for k in self.decay_terms:
                self.curr_gains[k] = new_gains[k]
        else:
            raise ValueError("Invalid fixed mode")
        return True, ""

    def determine_decay(self, epoch_num):
        reduce_gains = True
        reason = f"Epoch {epoch_num}: "
        if epoch_num < self.wait_epochs:
            reason += f"waiting for {self.wait_epochs} epochs"
            reduce_gains = False
        if self.schedule == "fixed": # always True
            return reduce_gains, reason
        for key in self.rew_deques.keys():
            if len(self.rew_deques[key]) < self.deque_len:
                reduce_gains = False
                reason += f"{key} deque too short: {len(self.rew_deques[key]) } "
                break
            grad = self.rew_grads[key]
            rew_mean = np.mean(self.rew_deques[key])
            rew_thres = self.rew_thresholds.get(key, 0.01)
            if rew_mean < rew_thres:
                reason += f"{key} reward too low: {rew_mean} "
                reduce_gains = False
            elif not self.skip_grad and abs(grad) > self.grad_threshold:
                reason += f"{key} grad too low/high: {grad} "
                reduce_gains = False
            if not reduce_gains:
                break # no need to check other terms

        if self.num_epoch_since_last_decay < self.decay_min_epochs:
            reason += f"too soon until last decay: {self.num_epoch_since_last_decay} epochs "
            reduce_gains = False

        if len(self.ep_lens) < self.deque_len:
            reason += f"ep lens deque too short: {len(self.ep_lens)} "
            reduce_gains = False
        else:
            achieved_len = np.mean(self.ep_lens)
            if achieved_len < self.decay_ep_len_threshold:
                reason += f"max achieved length: {achieved_len} too low"
                reduce_gains = False
        return reduce_gains, reason

    def get_progress_metrics(self):
        task_rew = None
        if "task" in self.rew_deques and len(self.rew_deques["task"]) > 0:
            task_rew = float(np.mean(self.rew_deques["task"]))
        episode_length = None
        if len(self.ep_lens) > 0:
            episode_length = float(np.mean(self.ep_lens))
        return dict(task=task_rew, episode_length=episode_length)

    def determine_dialback(self, epoch_num, learning_stabilized=False):
        """ if the acheived ep length has been too low for too long, dial back the gains """
        if self.enhanced_dialback:
            max_past = self.curr_cfg["enhanced_dialback_min_epochs"]
            min_ep_len = self.curr_cfg["enhanced_dialback_ep_len"]
            ratios = {key: self.curr_cfg["enhanced_dialback_ratio"] for key in self.decay_terms}
        else:
            max_past = self.curr_cfg["dialback_min_epochs"]
            min_ep_len = self.curr_cfg["dialback_ep_len"]
            ratios = self.curr_cfg["dialback_ratios"]
        dialed = False
        if len(self.ep_lens) < self.deque_len:
            return dialed
        achieved_len = np.mean(self.ep_lens)
        if self.enhanced_dialback:
            if learning_stabilized or not self.gain_history or self.num_epoch_since_last_decay < max_past:
                return dialed
            metrics = self.get_progress_metrics()
            previous_metrics = self.gain_history[-1].get("metrics", {})
            task_rew = metrics["task"]
            previous_task_rew = previous_metrics.get("task")
            previous_ep_len = previous_metrics.get("episode_length")
            severe_drop = achieved_len < min_ep_len
            if task_rew is not None:
                severe_drop = severe_drop or task_rew < self.curr_cfg["dialback_task_rew_threshold"]
            relative_ep_drop = previous_ep_len is not None and achieved_len < previous_ep_len - self.curr_cfg["dialback_relative_ep_drop"]
            relative_task_drop = task_rew is not None and previous_task_rew is not None and task_rew < previous_task_rew - self.curr_cfg["dialback_relative_task_drop"]
            if self.curr_cfg.get("dialback_relative_require_both", False):
                relative_drop = relative_ep_drop and relative_task_drop
            else:
                relative_drop = relative_ep_drop or relative_task_drop
            stalled = self.num_epoch_since_last_decay >= self.curr_cfg["dialback_stall_epochs"]
            if not (severe_drop or relative_drop or stalled):
                return dialed
            previous = self.gain_history.pop()
            self.curr_gains = {
                key: value * ratios.get(key, 1.0)
                for key, value in previous["gains"].items()
            }
            self.curr_gains_lower = {
                key: value * ratios.get(key, 1.0)
                for key, value in previous["lower"].items()
            }
            reasons = []
            if severe_drop:
                reasons.append("absolute performance drop")
            if relative_drop:
                reasons.append("relative performance drop")
            if stalled:
                reasons.append("curriculum stalled")
            self.last_dialback_reason = ", ".join(reasons)
            return True

        if achieved_len < min_ep_len:
            if self.num_epoch_since_last_decay > max_past:
                upper_gains = self.gain_history[-1]['gains']
                lower_gains = self.gain_history[-1]['lower']
                for k in self.decay_terms:
                    if k in upper_gains:
                        ratio = ratios.get(k, 1.0)
                        upper_gains[k] *= ratio
                        lower_gains[k] *= ratio
                self.curr_gains = upper_gains
                self.curr_gains_lower = lower_gains
                dialed = True
        return dialed

    def set_auto_exp_decay(self, epoch_num):
        new_gains = dict()
        for k, v in self.curr_gains.items():
            ratio = self.upper_ratios[k]
            new_gains[k] = v * ratio
        if "kp" in new_gains and new_gains['kp'] < 0.05 or "fr" in new_gains and new_gains['fr'] < 0.01:
            new_gains = {k: 0.0 for k in new_gains}
        for k in self.decay_terms:
            self.curr_gains[k] = new_gains[k]
        return True, ""

    def set_auto_uniform_decay(self, epoch_num):
        """ update an upper and lower bound for gains """
        staged_idx = None
        if self.staged_gain_decay and "kp" in self.decay_terms:
            if self.curr_gains["kp"] > self.staged_kp_thresholds[0]:
                staged_idx = 0
            elif self.curr_gains["kp"] > self.staged_kp_thresholds[1]:
                staged_idx = 1
            else:
                staged_idx = 2
        for k in self.decay_terms:
            upp_ratio = self.staged_kp_ratios[staged_idx] if staged_idx is not None and k == "kp" else self.upper_ratios[k]
            self.curr_gains[k] = self.curr_gains[k] * upp_ratio
        if self.uniform_mode == "fast":
            # use self.curr_gains as upper bound
            for k in self.decay_terms:
                low_ratio = self.lower_ratios[k]
                self.curr_gains_lower[k] = self.curr_gains_lower[k] * low_ratio
        elif self.uniform_mode == "slow":
            # use self.curr_gains as upper bound
            for k in self.decay_terms:
                low_ratio = self.staged_kp_lower_ratios[staged_idx] if staged_idx is not None and k == "kp" else self.lower_ratios[k]
                self.curr_gains_lower[k] = self.curr_gains[k] * low_ratio
        else:
            raise ValueError("Invalid uniform mode")

        if ('kp' in self.decay_terms and self.curr_gains['kp'] < 0.05) or ('fr' in self.decay_terms and self.curr_gains['fr'] < 0.01):
            for k in self.decay_terms:
                self.curr_gains[k] = 0.0
                self.curr_gains_lower[k] = 0.0
            self.low_ratio = 0.0
        # if the lower bound kp is within 0.1 thres, reduce it to 0
        if ('kp' in self.decay_terms and self.curr_gains_lower.get('kp', 1) < 0.1):
            for k in self.decay_terms:
                self.curr_gains_lower[k] = 0.0
            self.low_ratio = 0.0
        return True, ""

    def update_progress(self, rewards, achieved_length):
        self.deque_appends += 1
        if self.deque_appends % self.deque_append_freq == 0:
            for k, val in rewards.items():
                assert k in self.rew_deques, f"Invalid key: {k}"
                self.rew_deques[k].append(val)
            self.deque_appends = 0
        # self.achieved_length = achieved_length
        self.ep_lens.append(achieved_length)

    def get_current_gains(self):
        ret_gains = self.curr_gains.copy()
        if self.schedule == "uniform" or self.fixed_mode == "uniform":
            ret_gains.update(
                {f"{k}_lower": v for k, v in self.curr_gains_lower.items()}
            )
        if self.decay_solimp:
            ret_gains.update(
                # dict(d0_lower=self.d0_lower, dmid_lower=self.dmid_lower)
                dict(tconst_lower=self.tconst_lower, tconst_upper=self.tconst_upper)
            )
        return ret_gains

    def reset_object_gains(self):
        rand_gains = dict()
        if self.schedule == "uniform" or self.fixed_mode == "uniform":
            if any([self.curr_gains[key] == 0 for key in self.decay_terms]):
                rand_gains = {k: 0.0 for k in self.decay_terms}
            else:
                # sample for every joint dim and env dim! shape should be (num_envs, num_joints)
                for key in self.decay_terms:
                    lower = self.curr_gains_lower[key]
                    upper = self.curr_gains[key]
                    # uniform from lower to upper, sample shape (num_envs, num_joints)
                    rand_gains[key] = torch.rand((self.num_envs, self.obj_ndof), dtype=torch.float32) * (upper - lower) + lower
        else:
            # same params for all
            for key in self.decay_terms:
                rand_gains[key] = self.curr_gains[key]
        # this should handle missing shapes:
        self.object.set_joint_gains(
            kp=rand_gains.get("kp", None),
            kv=rand_gains.get("kv", None),
            force_range=rand_gains.get("fr", None),
            env_idxs=None,
        )

    def reset_solimp(self):
        self.tconst_upper = max(self.tconst_upper * self.solip_multiplier, 0.02)
        self.tconst_lower = max(self.tconst_lower * self.solip_multiplier, 0.02)
        if self.tconst_upper == 0.02:
            tconst = 0.02
        else:
            tconst = np.random.uniform(self.tconst_lower, self.tconst_upper)
        new_params = torch.tensor([tconst,1.0, self.d0_lower, self.dmid_lower, 0.001, 0.5, 2.0])
        self.object.entity._solver.set_geom_sol_params(new_params)
        return

    def get_reward_grads(self):
        return self.rew_grads

    def update_reward_grads(self):
        for key in self.rew_deques.keys():
            if len(self.rew_deques[key]) < self.deque_len:
                continue
            grad = np.gradient(self.rew_deques[key]).mean()
            self.rew_grads[key] = grad

    def decay_reward_weights(self, reward_module):
        decayed = False
        if not self.decay_rew:
            return decayed
        info = ""
        for attr in ['imi_rew_weight', 'bc_rew_weight', 'contact_rew_weight']:
            if hasattr(reward_module, attr):
                val = getattr(reward_module, attr)
                if val > 0.01:
                    setattr(reward_module, attr, val * 0.95)
                    info += f"{attr}: to {val * 0.95} "
                    decayed = True
        # also decay the rew thresholds
        for key in self.rew_thresholds.keys():
            self.rew_thresholds[key] = self.rew_thresholds[key] * 0.95
        if len(info) > 0:
            print(info)
        return decayed

    def set_curriculum(self, epoch_num):
        # if gains are already zero, no need to reset
        self.update_reward_grads()
        self.num_epoch_since_last_decay += 1
        learning_stablized, reason = self.determine_decay(epoch_num)
        zero_gains = True
        for key in self.decay_terms:
            if self.curr_gains[key] > 0.0:
                zero_gains = False
                break
        # add to the count of epochs since last zero gains
        if zero_gains:
            self.num_epoch_since_zero += 1
        gains_decayed = False
        if epoch_num > self.zero_epoch:
            if not zero_gains:
                # set all gains to zero
                for key in self.decay_terms:
                    self.curr_gains[key] = 0.0
                    self.curr_gains_lower[key] = 0.0
                reason += f"Epoch {epoch_num}: Gains set to zero"
                gains_decayed = True
                self.reset_object_gains()
            return True, gains_decayed, reason

        if not zero_gains:
            zero_triggered, zero_trigger_reason = self._should_trigger_zero_from_kp_ratio(epoch_num)
            if zero_triggered:
                self._set_zero_gains()
                reason += zero_trigger_reason
                gains_decayed = True
                self.reset_object_gains()
                return True, gains_decayed, reason

        if zero_gains:
            return zero_gains, gains_decayed, reason

        if not self.enhanced_dialback and not learning_stablized:
            return zero_gains, gains_decayed, reason

        # decide if need to dial back the gains
        dialed_back = self.determine_dialback(epoch_num, learning_stablized)
        if self.enhanced_dialback and not learning_stablized and not dialed_back:
            return zero_gains, gains_decayed, reason
        if dialed_back:
            dialback_reason = f" ({self.last_dialback_reason})" if self.last_dialback_reason else ""
            print(f"Epoch {epoch_num}: Dialed back gains{dialback_reason}")
            gains_decayed = True # if the gains are dialed back, don't append to the history
        else:
            history_entry = dict(gains=self.curr_gains.copy(), lower=self.curr_gains_lower.copy())
            if self.enhanced_dialback:
                history_entry["metrics"] = self.get_progress_metrics()
            self.gain_history.append(history_entry)

            if self.schedule == "fixed":
                gains_decayed, reason = self.set_fixed_decay(epoch_num)
            elif self.schedule == "exp":
                gains_decayed, reason = self.set_auto_exp_decay(epoch_num)
            elif self.schedule == "uniform":
                gains_decayed, reason = self.set_auto_uniform_decay(epoch_num)
            else:
                raise ValueError("Invalid schedule")

        if gains_decayed:
            new_gains = self.get_current_gains()
            print(f"Epoch {epoch_num}: New Gains: {new_gains}")
            self.reset_object_gains()
            # also clear the deques!!
            for key in self.rew_deques.keys():
                self.rew_deques[key].clear()
                self.rew_grads[key] = 1.0
            self.ep_lens.clear()
            if self.decay_solimp:
                self.reset_solimp()
            self.deque_appends = 0
            self.num_epoch_since_last_decay = 0
        elif self.resample_every_epoch > 0 and epoch_num % self.resample_every_epoch == 0:
            self.reset_object_gains()
        return zero_gains, gains_decayed, reason
