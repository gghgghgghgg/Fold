import random
import numpy as np
from rl_games.common.segment_tree import SumSegmentTree, MinSegmentTree
import torch
import copy

class ReplayBuffer(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, traj):

        if self._next_idx >= len(self._storage):
            self._storage.append(traj)
        else:
            self._storage[self._next_idx] = traj
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def sample(self):
        idx = random.randint(0, len(self._storage) - 1)
        return self._storage[idx]


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha > 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha
        return idx

    def _sample_proportional(self):
        mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
        idx = self._it_sum.find_prefixsum_idx(mass)
        return idx

    def sample(self, beta):

        idx = self._sample_proportional()

        if beta > 0:
            p_min = self._it_min.min() / self._it_sum.sum()
            max_weight = (p_min * len(self._storage)) ** (-beta)

            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weight = weight / max_weight
        else:
            weight = 1
        return self._storage[idx], idx
        #return tuple([self._storage[i], weight, idx])

    def update_priority(self, idx, priority):
        priority = max(priority, 1e-6)
        assert priority > 0
        assert 0 <= idx < len(self._storage)
        self._it_sum[idx] = priority ** self._alpha
        self._it_min[idx] = priority ** self._alpha

        self._max_priority = max(self._max_priority, priority)


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done) # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]


class SelfImitation(object):

    def __init__(self, max_trajs=1000, alpha=0.5, beta=1.0, gamma=0.99, num_actors=32):
        self.buffer = PrioritizedReplayBuffer(max_trajs, alpha)

        self.in_episodes = [[[], [], [], [], []] for _ in range(num_actors)]

        self.out_episodes = [[[], [], [], [], []] for _ in range(num_actors)]
        self.out_episodes_idx = [-1 for _ in range(num_actors)]
        self.out_samples_idx = [[] for _ in range(num_actors)]
        self.out_samples_adv = [[] for _ in range(num_actors)]

        self.num_actors = num_actors
        self.beta = beta
        self.gamma = gamma

        self.input_episode_rewards = []
        self.output_episode_rewards = []

    def clean_in_episodes(self):
        self.in_episodes = [[[], [], [], [], []] for _ in range(self.num_actors)]
        self.input_episode_rewards = []

    def step(self, obs, states, actions, rewards, dones):
        for i in range(self.num_actors):
            self.in_episodes[i][0].append(copy.deepcopy(obs[i]))
            self.in_episodes[i][1].append(copy.deepcopy(states[i]))
            self.in_episodes[i][2].append(copy.deepcopy(actions[i]))
            self.in_episodes[i][3].append(copy.deepcopy(rewards[i]))
            self.in_episodes[i][4].append(copy.deepcopy(dones[i]))
            if dones[i]:
                traj = copy.deepcopy(self.in_episodes[i])
                #returns = discount_with_dones(rewards=copy.deepcopy(traj[3]), dones=copy.deepcopy(traj[4]), gamma=self.gamma)
                #traj.append(returns)
                idx = self.buffer.add(traj)
                self.buffer.update_priority(idx, torch.cat(traj[3]).sum().item())
                #print('input return max', torch.cat(returns).max().item())
                #print('input return', [returns[k].item() for k in range(len(returns))])
                #print('input reward', [traj[3][k].item() for k in range(len(returns))])
                #if torch.cat(traj[3]).sum().item()>200:
                #    import ipdb; ipdb.set_trace()
                self.process_episode_reward(traj, self.input_episode_rewards)

                self.in_episodes[i] = [[], [], [], [], []]

    def process_episode_reward(self, traj, reward_list):
        episode_reward = torch.cat(traj[3]).sum().item()
        if len(reward_list)>=200:
            reward_list.pop(0)
        reward_list.append(episode_reward)

    def num_traj(self):
        return len(self.buffer)

    def cat_env(self, input_list):
        return torch.cat([x.unsqueeze(0) for x in input_list])

    def sample_batch(self):
        obs, states, dones, actions, rewards, new_dones, new_obs, new_states, returns = [],[],[],[],[],[],[],[],[]
        for i in range(self.num_actors):
            if len(self.out_episodes[i][0])==0:
                #put the newly sampled episode
                traj, idx = self.buffer.sample(beta=self.beta)
                self.out_episodes_idx[i] = idx
                self.process_episode_reward(traj, self.output_episode_rewards)
                self.out_episodes[i] = copy.deepcopy(traj)
                #old done is True
                dones.append(self.out_episodes[i][4][0]*0+1)
            else:
                #old done is False
                dones.append(self.out_episodes[i][4][0]*0)
            #observation in next step
            if len(self.out_episodes[i][0])>1:
                new_obs.append(self.out_episodes[i][0][1])
                new_states.append(self.out_episodes[i][1][1])
            else:
                #the last observtion in an episode is the same as the second last
                new_obs.append(self.out_episodes[i][0][0])
                new_states.append(self.out_episodes[i][1][0])
            obs.append(self.out_episodes[i][0].pop(0))
            states.append(self.out_episodes[i][1].pop(0))
            actions.append(self.out_episodes[i][2].pop(0))
            rewards.append(self.out_episodes[i][3].pop(0))
            new_dones.append(self.out_episodes[i][4].pop(0))
            #returns.append(self.out_episodes[i][5].pop(0))
            #record the trajectory idx in the sampled data
            self.out_samples_idx[i].append(self.out_episodes_idx[i])

        obs = {'obs':self.cat_env(obs), 'states':self.cat_env(states)}
        dones = self.cat_env(dones)
        actions = self.cat_env(actions)
        rewards = self.cat_env(rewards)
        new_dones = self.cat_env(new_dones)
        new_obs = {'obs':self.cat_env(new_obs), 'states':self.cat_env(new_states)}
        #returns = self.cat_env(returns)
        return obs, dones, actions, rewards, new_obs, new_dones #, returns

    def update_out_adv(self, advs):
        for i in range(self.num_actors):
            seq_adv = advs[:,i,0]
            self.out_samples_adv[i].extend([seq_adv[i] for i in range(seq_adv.size(0))])
            traj_idx = self.out_samples_idx[i][0]
            traj_len = len(self.buffer._storage[traj_idx][0])
            #print('env', i, 'traj len', traj_len, 'sample adv', len(self.out_samples_adv[i]))
            if traj_len<=len(self.out_samples_adv[i]):
                #the advantage for the whole trajectory is ready
                traj_adv = copy.deepcopy(self.out_samples_adv[i][:traj_len])
                print('out episode mean adv', torch.mean(torch.tensor(traj_adv)).item())
                self.buffer.update_priority(traj_idx, torch.mean(torch.tensor(traj_adv)).item())
                self.out_samples_idx[i] = copy.deepcopy(self.out_samples_idx[i][traj_len:])
                self.out_samples_adv[i] = copy.deepcopy(self.out_samples_adv[i][traj_len:])
