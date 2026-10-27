import os
from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.algos_torch import torch_ext

from rl_games.algos_torch import central_value
from rl_games.common import common_losses
from rl_games.common import datasets
from rl_games.common.experience import ExperienceBuffer

from torch import optim
import torch
from torch import nn
import numpy as np
import gym

import time
from rl_games.common.a2c_common import swap_and_flatten01, inverse_swap_and_flatten01
from rl_games.common.self_imitation import SelfImitation
import copy

class A2CSILAgent(A2CAgent):
    def __init__(self, base_name, params):
        A2CAgent.__init__(self, base_name, params)
        self.seq_len = self.seq_length
        if self.has_central_value:
            cv_config = {
                'state_shape' : self.state_shape,
                'value_size' : self.value_size,
                'ppo_device' : self.ppo_device,
                'num_agents' : self.num_agents,
                'horizon_length' : self.horizon_length,
                'num_actors' : self.num_actors,
                'num_actions' : self.actions_num,
                'seq_len' : self.seq_len,
                'normalize_value' : self.normalize_value,
                'network' : self.central_value_config['network'],
                'config' : self.central_value_config,
                'writter' : self.writer,
                'max_epochs' : self.max_epochs,
                'multi_gpu' : self.multi_gpu,
                'sil_coef': self.config['sil_coef'],
                'sil_clip': self.config['sil_clip'],
                'zero_rnn_on_done' : self.zero_rnn_on_done,
            }
            self.central_value_net = central_value.CentralValueSILTrain(**cv_config).to(self.ppo_device)

        if self.normalize_value:
            self.value_mean_std = self.central_value_net.model.value_mean_std if self.has_central_value else self.model.value_mean_std

        self.sil = SelfImitation(num_actors=self.num_actors, alpha=self.config['sil_alpha'], gamma=self.gamma)

        self.sil_rnn_states = None
        self.sil_dataset = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_len)

        self.sil_coef = self.config['sil_coef']
        self.sil_clip = self.config['sil_clip']
        self.sil_critic = self.config['sil_critic'] if 'sil_critic' in self.config else True
        self.sil_central_value = self.config['sil_central_value'] if 'sil_central_value' in self.config else False
        self.sil_update_adv = self.config['sil_update_adv'] if 'sil_update_adv' in self.config else False
        self.retrieve_demo_freq = self.config['retrieve_demo_freq'] if 'retrieve_demo_freq' in self.config else 100000
        self.retrieve_demo_num = self.config['retrieve_demo_num'] if 'retrieve_demo_num' in self.config else 100
        self.skill_mode = self.config['skill_mode'] if 'skill_mode' in self.config else 'select'

        #self.checkpoint = params['checkpoint']
        #self.checkpoints = params['checkpoints']
        if len(params['checkpoint'])==1:
            self.checkpoint = params['checkpoint'][0]
            self.checkpoints = None
        else:
            self.checkpoint = None
            self.checkpoints = [] # params['checkpoint']
        # self.load_mode = params['load_mode']
        self.rank_size = 1
        self.rank = 0
        self.has_soft_aug = False

    def init_tensors(self):
        super().init_tensors()

        algo_info = {
            'num_actors' : self.num_actors,
            'horizon_length' : self.horizon_length,
            'has_central_value' : self.has_central_value,
            'use_action_masks' : self.use_action_masks
        }
        self.sil_experience_buffer = ExperienceBuffer(self.env_info, algo_info, self.ppo_device)
        #self.sil_experience_buffer._init_from_aux_dict({'returns':(1,)})

        if self.is_rnn:
            self.sil_rnn_states = self.model.get_default_rnn_state()
            self.sil_rnn_states = [s.to(self.ppo_device) for s in self.sil_rnn_states]

            total_agents = self.num_agents * self.num_actors
            num_seqs = self.horizon_length // self.seq_len
            assert((self.horizon_length * total_agents // self.num_minibatches) % self.seq_len == 0)
            self.mb_sil_rnn_states = [torch.zeros((num_seqs, s.size()[0], total_agents, s.size()[2]), dtype = torch.float32, device=self.ppo_device) for s in self.sil_rnn_states]


    def play_steps_rnn(self):
        update_list = self.update_list
        mb_rnn_states = self.mb_rnn_states
        step_time = 0.0

        for n in range(self.horizon_length):
            if n % self.seq_len == 0:
                for s, mb_s in zip(self.rnn_states, mb_rnn_states):
                    mb_s[n // self.seq_len,:,:,:] = s

            if self.has_central_value:
                self.central_value_net.pre_step_rnn(n)

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)
            self.rnn_states = res_dict['rnn_states']
            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones.byte())

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            step_time_start = time.time()
            obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            step_time_end = time.time()

            step_time += (step_time_end - step_time_start)

            shaped_rewards = self.rewards_shaper(rewards)

            #assert(infos['time_outs'][0]==self.dones[0])
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()

            self.experience_buffer.update_data('rewards', n, shaped_rewards)

            #self.sil.step(self.obs['obs'], self.obs['states'], res_dict['actions'], shaped_rewards, self.dones)
            self.sil.step(self.obs['obs'], self.obs['states'], res_dict['actions'], rewards, self.dones)
            self.obs = obs

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = self.dones.view(self.num_actors, self.num_agents).all(dim=1).nonzero(as_tuple=False)
            if len(all_done_indices) > 0:
                for s in self.rnn_states:
                    s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0
                if self.has_central_value:
                    self.central_value_net.post_step_rnn(all_done_indices)

            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        last_values = self.get_values(self.obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values
        self.writer.add_scalar('info/ppo_return', torch.mean(mb_returns).item(), self.frame)
        #self.writer.add_scalar('info/sil_in_adv', torch.mean(mb_advs).item(), self.frame)

        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        states = []
        for mb_s in mb_rnn_states:
            t_size = mb_s.size()[0] * mb_s.size()[2]
            h_size = mb_s.size()[3]
            states.append(mb_s.permute(1,2,0,3).reshape(-1,t_size, h_size))
        batch_dict['rnn_states'] = states
        batch_dict['step_time'] = step_time
        return batch_dict


    def get_sil_action_values(self, obs, actions):
        processed_obs = self._preproc_obs(obs['obs'])
        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs' : processed_obs,
            'rnn_states' : self.sil_rnn_states
        }

        with torch.no_grad():
            input_dict['obs'] = self.model.norm_obs(input_dict['obs'])
            mu, logstd, value, states = self.model.a2c_network(input_dict)
            sigma = torch.exp(logstd)
            neglogp = self.model.neglogp(actions, mu, sigma, logstd)
            res_dict = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : self.model.denorm_value(value),
                    'actions' : actions,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }

            if self.has_central_value:
                states = obs['states']
                input_dict = {
                    'is_train': False,
                    'states' : states,
                }
                value = self.central_value_net.get_sil_value(input_dict)
                self.writer.add_scalar('info/sil_out_value_play', torch.mean(value).item(), self.frame)
                res_dict['values'] = value
        return res_dict

    def play_sil_steps_rnn(self):
        update_list = self.update_list
        mb_sil_rnn_states = self.mb_sil_rnn_states

        for n in range(self.horizon_length):
            if n % self.seq_len == 0:
                for s, mb_s in zip(self.sil_rnn_states, mb_sil_rnn_states):
                    mb_s[n // self.seq_len,:,:,:] = s

            if self.has_central_value:
                self.central_value_net.pre_sil_step_rnn(n)

            obs, dones, actions, rewards, new_obs, new_dones = self.sil.sample_batch()
            #obs, dones, actions, rewards, new_obs, new_dones, returns = self.sil.sample_batch()
            #print('returns', returns.max().item(), returns.min().item(), returns.mean().item())
            #if returns.mean().item()>35:
            #    import ipdb; ipdb.set_trace()
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_sil_masked_action_values(obs, masks)
            else:
                res_dict = self.get_sil_action_values(obs, actions)

            self.sil_rnn_states = res_dict['rnn_states']
            self.sil_experience_buffer.update_data('obses', n, obs['obs'])
            self.sil_experience_buffer.update_data('dones', n, dones.byte())

            for k in update_list:
                self.sil_experience_buffer.update_data(k, n, res_dict[k])
            if self.has_central_value:
                self.sil_experience_buffer.update_data('states', n, obs['states'])

            #shaped_rewards = self.rewards_shaper(rewards)

            #if self.value_bootstrap:
            #    shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(new_dones).unsqueeze(1).float()

            #self.sil_experience_buffer.update_data('rewards', n, shaped_rewards)
            self.sil_experience_buffer.update_data('rewards', n, rewards)
            #self.sil_experience_buffer.update_data('returns', n, returns)

            all_done_indices = new_dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[::self.num_agents]

            if len(all_done_indices) > 0:
                for s in self.sil_rnn_states:
                    s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0
                if self.has_central_value:
                    self.central_value_net.post_sil_step_rnn(all_done_indices)

        last_values = self.get_values(new_obs)

        fdones = new_dones.float()

        mb_fdones = self.sil_experience_buffer.tensor_dict['dones'].float()

        mb_values = self.sil_experience_buffer.tensor_dict['values']
        mb_rewards = self.sil_experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values
        #mb_returns = self.sil_experience_buffer.tensor_dict['returns']
        #mb_advs = mb_returns - mb_values
        self.writer.add_scalar('info/sil_return', torch.mean(mb_returns).item(), self.frame)
        self.writer.add_scalar('info/sil_out_adv_play', torch.mean(mb_advs).item(), self.frame)

        #print('mb return', mb_returns.max().item(), mb_returns.min().item())
        #print('!!!!!!!mb returns', mb_returns.mean().item(), '!!!!!! mb advs', mb_advs.mean().item(), '!!!!!! mb values',mb_values.mean().item())
        batch_dict = self.sil_experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)

        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        states = []
        for mb_s in mb_sil_rnn_states:
            t_size = mb_s.size()[0] * mb_s.size()[2]
            h_size = mb_s.size()[3]
            states.append(mb_s.permute(1,2,0,3).reshape(-1,t_size, h_size))

        batch_dict['rnn_states'] = states

        return batch_dict


    def prepare_sil_dataset(self, batch_dict):
        obses = batch_dict['obses']
        returns = batch_dict['returns']
        dones = batch_dict['dones']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        mus = batch_dict['mus']
        sigmas = batch_dict['sigmas']
        rnn_states = batch_dict.get('rnn_states', None)
        rnn_masks = batch_dict.get('rnn_masks', None)
        advantages = returns - values

        obses = batch_dict['obses']
        if self.normalize_value:
            self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()

        advantages = torch.sum(advantages, axis=1)

        if self.normalize_advantage:
            if self.is_rnn:
                if self.normalize_rms_advantage:
                    advantages = self.advantage_mean_std(advantages, mask=rnn_masks)
                else:
                    advantages = torch_ext.normalization_with_masks(advantages, rnn_masks)
            else:
                if self.normalize_rms_advantage:
                    advantages = self.advantage_mean_std(advantages)
                else:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['old_logp_actions'] = neglogpacs
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = returns
        dataset_dict['actions'] = actions
        dataset_dict['obs'] = obses
        dataset_dict['dones'] = dones
        dataset_dict['rnn_states'] = rnn_states
        dataset_dict['rnn_masks'] = rnn_masks
        dataset_dict['mu'] = mus
        dataset_dict['sigma'] = sigmas

        self.sil_dataset.update_values_dict(dataset_dict)

        if self.has_central_value:
            dataset_dict = {}
            dataset_dict['old_values'] = values
            dataset_dict['advantages'] = advantages
            dataset_dict['returns'] = returns
            dataset_dict['actions'] = actions
            dataset_dict['obs'] = batch_dict['states']
            dataset_dict['dones'] = dones
            dataset_dict['rnn_masks'] = rnn_masks
            self.central_value_net.update_sil_dataset(dataset_dict)

    def clean_rnn_states(self):
        self.dones = self.dones*0+1
        all_done_indices = self.dones.nonzero(as_tuple=False)
        for s in self.rnn_states:
            s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0
        if self.has_central_value:
            self.central_value_net.post_step_rnn(all_done_indices)

    def restore_agent(self, ckpt):
        # if self.load_mode=='actor':
        #     self.restore_actor(ckpt)
        # elif self.load_mode=='model':
        #     self.restore_model(ckpt)
        # else:
        #     self.restore(ckpt)
        if ckpt is None or ckpt=='':
            return
        self.restore(ckpt)

    def retrieve_skill_demo(self, ckpt):
        self.set_eval()
        self.save('tmp')
        self.restore_agent(ckpt)
        self.sil.clean_in_episodes()
        self.clean_rnn_states()
        rew = 0
        demo_num = 0
        with torch.no_grad():
            while demo_num < self.retrieve_demo_num:
                self.play_steps_rnn()
                rew += self.experience_buffer.tensor_dict['rewards'].sum()
                demo_num += self.experience_buffer.tensor_dict['dones'][1:].sum()
        self.restore_agent('tmp.pth')
        self.sil.clean_in_episodes()
        self.clean_rnn_states()
        return rew/demo_num

    def select_skill_ckpts(self):
        best_rew = -np.inf
        best_ckpt = None
        for ckpt in self.checkpoints:
            if ckpt is None or ckpt=='':
                continue
            rew = self.retrieve_skill_demo(ckpt)
            if rew > best_rew:
                best_rew = rew
                best_ckpt = ckpt
        self.checkpoint = best_ckpt
        self.restore_agent(self.checkpoint)

    def mix_skill_ckpts(self):
        loaded_params = []
        for ckpt in self.checkpoints:
            if ckpt is None or ckpt=='':
                continue
            checkpoint = torch_ext.load_checkpoint(ckpt)
            state_dict = checkpoint['model']
            loaded_params.append(copy.deepcopy(state_dict))
        mix_params = {}
        for k in state_dict.keys():
            v = [loaded_params[i][k].unsqueeze(0) for i in range(len(loaded_params))]
            v = torch.cat(v).mean(dim=0)
            mix_params[k] = copy.deepcopy(v)
        torch_ext.save_checkpoint('tmp', {'model':mix_params})
        self.restore_agent('tmp.pth')



    def train_epoch(self):
        self.vec_env.set_train_info(self.frame, self)

        self.set_eval()
        play_time_start = time.time()
        with torch.no_grad():
            sil_batch_dict = None
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
                if self.sil.num_traj()>=self.num_actors:
                    sil_batch_dict = self.play_sil_steps_rnn()
            else:
                batch_dict = self.play_steps()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)

        self.set_train()
        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        if sil_batch_dict is not None:
            self.prepare_sil_dataset(sil_batch_dict)

        self.algo_observer.after_steps()
        if self.has_central_value:
            self.train_central_value()
            if self.sil.num_traj()>=self.num_actors and self.sil_central_value:
                self.central_value_net.train_sil_net()

        a_losses = []
        c_losses = []
        b_losses = []
        entropies = []
        kls = []
        sil_a_losses = []
        sil_c_losses = []
        sil_entropies = []

        for mini_ep in range(0, self.mini_epochs_num):
            ep_kls = []
            if mini_ep==self.mini_epochs_num-1:
                sil_advs = []
            else:
                sil_advs = None
            for i in range(len(self.dataset)):
                data = self.dataset[i]
                if sil_batch_dict is not None:
                    sil_data = self.sil_dataset[i]
                    for k, v in sil_data.items():
                        data.update({'sil_'+k: v})
                a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss, sil_a_loss, sil_c_loss, sil_entropy, sil_adv = self.train_actor_critic(data)

                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)
                sil_a_losses.append(sil_a_loss)
                sil_c_losses.append(sil_c_loss)
                sil_entropies.append(sil_entropy)
                if (sil_advs is not None) and (sil_adv is not None):
                    sil_advs.append(sil_adv)

                self.dataset.update_mu_sigma(cmu, csigma)
                if self.schedule_type == 'legacy':
                    av_kls = kl
                    if self.multi_gpu:
                        dist.all_reduce(kl, op=dist.ReduceOp.SUM)
                        av_kls /= self.rank_size
                    self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                    self.update_lr(self.last_lr)

            av_kls = torch_ext.mean_list(ep_kls)
            if self.multi_gpu:
                dist.all_reduce(av_kls, op=dist.ReduceOp.SUM)
                av_kls /= self.rank_size
            if self.schedule_type == 'standard':
                self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                self.update_lr(self.last_lr)

            kls.append(av_kls)
            self.diagnostics.mini_epoch(self, mini_ep)
            if self.normalize_input:
                self.model.running_mean_std.eval() # don't need to update statstics more than one miniepoch

        if self.sil.num_traj()>=self.num_actors and self.sil_update_adv:
            sil_advs = torch.cat(sil_advs)
            sil_advs = inverse_swap_and_flatten01(sil_advs, s0=self.horizon_length, s1=self.num_actors)
            self.sil.update_out_adv(sil_advs)
            self.writer.add_scalar('info/sil_out_adv', torch.mean(sil_advs).item(), self.frame)

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return batch_dict['step_time'], play_time, update_time, total_time, a_losses, c_losses, b_losses, entropies, kls, sil_a_losses, sil_c_losses, sil_entropies, last_lr, lr_mul


    def calc_gradients(self, input_dict):
        super().calc_gradients(input_dict)
        if 'sil_old_values' not in input_dict:
            #we do not have sil sample batch in the input, skip sil loss
            loss_0 = self.train_result[-1]*0.0
            self.train_result += (loss_0, loss_0, loss_0, None)
            return
        value_preds_batch = input_dict['sil_old_values']
        old_action_log_probs_batch = input_dict['sil_old_logp_actions']
        advantage = input_dict['sil_advantages']
        old_mu_batch = input_dict['sil_mu']
        old_sigma_batch = input_dict['sil_sigma']
        return_batch = input_dict['sil_returns']
        actions_batch = input_dict['sil_actions']
        obs_batch = input_dict['sil_obs']
        obs_batch = self._preproc_obs(obs_batch)

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch,
            'obs' : obs_batch,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['sil_rnn_masks']
            batch_dict['rnn_states'] = input_dict['sil_rnn_states']
            batch_dict['seq_length'] = self.seq_len

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            a_loss = action_log_probs
            if self.sil_critic:
                a_loss *= torch.clamp(return_batch - values, 0.0, self.sil_clip).detach().squeeze()

            if self.has_value_loss:
                c_loss = common_losses.critic_sil_loss(values, return_batch, self.sil_clip)
            #print('!!!!!!return', return_batch.mean().item(), '!!!!!!!value', values.mean().item())
            advantage = (return_batch-values).detach().squeeze()
            mask = torch.where(advantage>0.0, advantage.clone()*0.0+1.0, advantage.clone()*0.0)
            entropy = entropy*mask

            losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss , entropy.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy = losses[0], losses[1], losses[2]

            loss = a_loss
            if self.sil_critic:
                loss += 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef

            loss = loss*self.sil_coef

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        #TODO: Refactor this ugliest code of they year
        self.trancate_gradients_and_step()
        self.train_result += (a_loss, c_loss, entropy, advantage)


    def train(self):
        self.init_tensors()
        self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        if self.multi_gpu:
            print("====================broadcasting parameters")
            model_params = [self.model.state_dict()]
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])

        if self.checkpoints is not None:
            if self.skill_mode == 'select':
                self.select_skill_ckpts()
            elif self.skill_mode == 'mix':
                self.mix_skill_ckpts()

        while True:
            epoch_num = self.update_epoch()
            if self.retrieve_demo_num>0 and epoch_num%self.retrieve_demo_freq==0 and epoch_num > 0:
                self.retrieve_skill_demo(self.checkpoint)

            step_time, play_time, update_time, sum_time, a_losses, c_losses, b_losses, entropies, kls, sil_a_losses, sil_c_losses, sil_entropies, last_lr, lr_mul = self.train_epoch()
            total_time += sum_time
            frame = self.frame // self.num_agents

            # cleaning memory to optimize space
            self.dataset.update_values_dict(None)
            self.sil_dataset.update_values_dict(None)
            should_exit = False

            if self.rank == 0:
                self.diagnostics.epoch(self, current_epoch=epoch_num)
                # do we need scaled_time?
                scaled_time = self.num_agents * sum_time
                scaled_play_time = self.num_agents * play_time
                curr_frames = self.curr_frames * self.rank_size if self.multi_gpu else self.curr_frames
                self.frame += curr_frames

                if self.print_stats:
                    step_time = max(step_time, 1e-6)
                    fps_step = curr_frames / step_time
                    fps_step_inference = curr_frames / scaled_play_time
                    fps_total = curr_frames / scaled_time
                    print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num}/{self.max_epochs}')

                self.write_stats(total_time, epoch_num, step_time, play_time, update_time, a_losses, c_losses, entropies, kls, last_lr, lr_mul, frame, scaled_time, scaled_play_time, curr_frames)
                if len(b_losses) > 0:
                    self.writer.add_scalar('losses/bounds_loss', torch_ext.mean_list(b_losses).item(), frame)

                if self.has_soft_aug:
                    self.writer.add_scalar('losses/aug_loss', np.mean(aug_losses), frame)

                self.writer.add_scalar('losses/sil_a_loss', torch_ext.mean_list(sil_a_losses).item(), frame)
                self.writer.add_scalar('losses/sil_c_loss', torch_ext.mean_list(sil_c_losses).item(), frame)
                self.writer.add_scalar('losses/sil_entropy', torch_ext.mean_list(sil_entropies).item(), frame)
                self.writer.add_scalar('info/sil_input_reward', np.mean(self.sil.input_episode_rewards))
                self.writer.add_scalar('info/sil_output_reward', np.mean(self.sil.output_episode_rewards))

                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()
                    self.mean_rewards = mean_rewards[0]

                    for i in range(self.value_size):
                        rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
                        self.writer.add_scalar(rewards_name + '/step'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar(rewards_name + '/iter'.format(i), mean_rewards[i], epoch_num)
                        self.writer.add_scalar(rewards_name + '/time'.format(i), mean_rewards[i], total_time)

                    self.writer.add_scalar('episode_lengths/step', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
                    self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                    checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) + '_rew_' + str(mean_rewards[0])

                    if self.save_freq > 0:
                        if (epoch_num % self.save_freq == 0) and (mean_rewards[0] <= self.last_mean_rewards):
                            self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))

                    if mean_rewards[0] > self.last_mean_rewards and epoch_num >= self.save_best_after:
                        print('saving next best rewards: ', mean_rewards)
                        self.last_mean_rewards = mean_rewards[0]
                        self.save(os.path.join(self.nn_dir, self.config['name']))

                        if 'score_to_win' in self.config:
                            if self.last_mean_rewards > self.config['score_to_win']:
                                print('Network won!')
                                self.save(os.path.join(self.nn_dir, checkpoint_name))
                                should_exit = True

                if epoch_num >= self.max_epochs:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max epochs reached before any env terminated at least once')
                        mean_rewards = -np.inf
                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + 'ep' + str(epoch_num) + 'rew' + str(mean_rewards)))
                    print('MAX EPOCHS NUM!')
                    should_exit = True

                update_time = 0

            if self.multi_gpu:
                should_exit_t = torch.tensor(should_exit, device=self.device).float()
                dist.broadcast(should_exit_t, 0)
                should_exit = should_exit_t.float().item()
            if should_exit:
                return self.last_mean_rewards, epoch_num

            if should_exit:
                return self.last_mean_rewards, epoch_num
