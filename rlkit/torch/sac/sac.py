import os
from turtle import position
import torch
import torch.optim as optim
import numpy as np
import rlkit.torch.pytorch_util as ptu
from torch import nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from itertools import product
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.rl_algorithm import OfflineMetaRLAlgorithm
from rlkit.torch.brac import divergences
from rlkit.torch.brac import utils
import pdb

# main algorithms
class CSROSoftActorCritic(OfflineMetaRLAlgorithm):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            latent_dim,
            nets,
            goal_radius=1,
            optimizer_class=optim.Adam,
            plotter=None,
            render_eval_paths=False,
            wandb_project_name=None,
            wandb_run_name=None,
            **kwargs
    ):
        super().__init__(
            env=env,
            agent=nets[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            goal_radius=goal_radius,
            wandb_project_name=wandb_project_name,
            wandb_run_name=wandb_run_name,
            **kwargs
        )

        self.latent_dim                     = latent_dim
        self.soft_target_tau                = kwargs['soft_target_tau']
        self.policy_mean_reg_weight         = kwargs['policy_mean_reg_weight']
        self.policy_std_reg_weight          = kwargs['policy_std_reg_weight']
        self.policy_pre_activation_weight   = kwargs['policy_pre_activation_weight']
        self.recurrent                      = kwargs['recurrent']
        self.kl_lambda                      = kwargs['kl_lambda']
        self._divergence_name               = kwargs['divergence_name']
        self.sparse_rewards                 = kwargs['sparse_rewards']
        self.use_next_obs_in_context        = kwargs['use_next_obs_in_context']
        self.use_brac                       = kwargs['use_brac']
        self.use_value_penalty              = kwargs['use_value_penalty']
        self.alpha_max                      = kwargs['alpha_max']
        self._c_iter                        = kwargs['c_iter']
        self.train_alpha                    = kwargs['train_alpha']
        self._target_divergence             = kwargs['target_divergence']
        self.alpha_init                     = kwargs['alpha_init']
        self.alpha_lr                       = kwargs['alpha_lr']
        self.policy_lr                      = kwargs['policy_lr']
        self.qf_lr                          = kwargs['qf_lr']
        self.vf_lr                          = kwargs['vf_lr']
        self.c_lr                           = kwargs['c_lr']
        self.context_lr                     = kwargs['context_lr']
        self.z_loss_weight                  = kwargs['z_loss_weight']
        self.max_entropy                    = kwargs['max_entropy']
        self.allow_backward_z               = kwargs['allow_backward_z']
        self.loss                           = {}
        self.plotter                        = plotter
        self.render_eval_paths              = render_eval_paths
        self.qf_criterion                   = nn.MSELoss()
        self.vf_criterion                   = nn.MSELoss()
        self.vib_criterion                  = nn.MSELoss()
        self.l2_reg_criterion               = nn.MSELoss()
        self.club_criterion                 = nn.MSELoss()
        self.cross_entropy_loss             = nn.CrossEntropyLoss()

        self.qf1, self.qf2, self.vf, self.c, self.club_model, self.reward_decoder, self.transition_decoder = nets[1:]
        self.target_vf                      = self.vf.copy()

        self.policy_optimizer               = optimizer_class(self.agent.policy.parameters(), lr=self.policy_lr)
        self.qf1_optimizer                  = optimizer_class(self.qf1.parameters(), lr=self.qf_lr)
        self.qf2_optimizer                  = optimizer_class(self.qf2.parameters(), lr=self.qf_lr)
        self.vf_optimizer                   = optimizer_class(self.vf.parameters(),  lr=self.vf_lr)
        self.c_optimizer                    = optimizer_class(self.c.parameters(),   lr=self.c_lr)
        self.context_optimizer              = optimizer_class(self.agent.context_encoder.parameters(), lr=self.context_lr)
        self.club_model_optimizer           = optimizer_class(self.club_model.parameters(), lr=self.context_lr)

        self._num_steps                     = 0
        self._visit_num_steps_train         = 10
        self._alpha_var                     = torch.tensor(1.)

        self.wandb_project_name = wandb_project_name
        self.wandb_run_name = wandb_run_name

        self.pred_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.VideoRLCS_loss = nn.SmoothL1Loss()

        self.reward_decoder_optimizer = optimizer_class(self.reward_decoder.parameters(), lr=self.qf_lr)
        self.transition_decoder_optimizer = optimizer_class(self.transition_decoder.parameters(), lr=self.qf_lr)

        for net in nets:
            self.print_networks(net)

    ###### Torch stuff #####
    @property
    def networks(self):
        return self.agent.networks + [self.qf1, self.qf2, self.vf, self.target_vf, self.c, self.club_model, self.reward_decoder, self.transition_decoder]

    @property
    def get_alpha(self):
        return utils.clip_v2(
            self._alpha_var, 0.0, self.alpha_max)

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)
        if self.train_alpha:
            self._alpha_var = torch.tensor(self.alpha_init, device=ptu.device, requires_grad=True)
        self._divergence = divergences.get_divergence(name=self._divergence_name, c=self.c, device=ptu.device)

    def print_networks(self, net):
        print('---------- Networks initialized -------------')
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        #print(net)
        print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
        print('-----------------------------------------------')

    ##### Data handling #####
    def unpack_batch(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if sparse_reward:
            r = batch['sparse_rewards'][None, ...]
        else:
            r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t]

    def sample_sac(self, indices, b_size=None):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense
        if b_size is None:
            batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size)) for idx in indices]
        else:
            batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=b_size)) for idx
                       in indices]
        unpacked = [self.unpack_batch(batch) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_context(self, indices):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=self.recurrent)) for idx in indices]
        context = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context] # 5 * self.meta_batch * self.embedding_batch_size * dim(o, a, r, no, t)
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        if self.use_next_obs_in_context:
            context = torch.cat(context[:-1], dim=2)
        else:
            context = torch.cat(context[:-2], dim=2)

        return context

    ##### Training #####
    def _do_training(self, indices):
        mb_size = self.embedding_mini_batch_size # NOTE: not meta batch!
        num_updates = self.embedding_batch_size // mb_size

        # sample context batch
        context_batch = self.sample_context(indices)

        # zero out context and hidden encoder state
        self.agent.clear_z(num_tasks=len(indices))

        z_means_lst = []
        z_vars_lst = []
        # do this in a loop so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):
            context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]
            self.loss['step'] = self._num_steps

            z_means, z_vars, wandb_stat = self._take_step(indices, context)
            self._num_steps += 1
            z_means_lst.append(z_means[None, ...])
            z_vars_lst.append(z_vars[None, ...])
            # stop backprop
            self.agent.detach_z()
        z_means = np.mean(np.concatenate(z_means_lst), axis=0)
        z_vars = np.mean(np.concatenate(z_vars_lst), axis=0)
        return z_means, z_vars, wandb_stat

    def _min_q(self, obs, actions, task_z):
        q1 = self.qf1(obs, actions, task_z.detach())
        q2 = self.qf2(obs, actions, task_z.detach())
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def _optimize_c(self, indices, context):
        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)

        # run inference in networks
        policy_outputs, task_z, task_z_vars = self.agent(obs, context, task_indices=indices)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # optimize for c network (which computes dual-form divergences)
        c_loss = self._divergence.dual_critic_loss(obs, new_actions.detach(), actions, task_z.detach())
        self.c_optimizer.zero_grad()
        c_loss.backward(retain_graph=True)
        self.c_optimizer.step()
        
    def FOCAL_z_loss(self, indices, task_z, task_z_vars, b, epsilon=1e-3, threshold=0.999):
        pos_z_loss = 0.
        neg_z_loss = 0.
        pos_cnt = 0
        neg_cnt = 0
        for i in range(len(indices)):
            idx_i = i * b # index in task * batch dim
            for j in range(i+1, len(indices)):
                idx_j = j * b # index in task * batch dim
                if indices[i] == indices[j]:
                    pos_z_loss += torch.sqrt(torch.mean((task_z[idx_i] - task_z[idx_j]) ** 2) + epsilon)
                    pos_cnt += 1
                else:
                    neg_z_loss += 1/(torch.mean((task_z[idx_i] - task_z[idx_j]) ** 2) + epsilon * 100)
                    neg_cnt += 1
        return pos_z_loss/(pos_cnt + epsilon) +  neg_z_loss/(neg_cnt + epsilon)


    def sample_trajs(self, indices):
        batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size, sequence=True)) for
                   idx in indices]

        unpacked = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]

        if self.use_next_obs_in_context:
            context = torch.cat(unpacked[:-1], dim=2)
        else:
            context = torch.cat(unpacked[:-2], dim=2)

        return unpacked, context

    def TACO_loss_vectorized(self, indices, K=8):
        trajs, trajs_context = self.sample_trajs(indices)

        context_embedding = self.agent.only_infer_context_embeddings_TACO(trajs_context)
        # print("context_embedding.shape: ", context_embedding.shape)
        task_num, seq_len, context_dim = context_embedding.shape
        device = context_embedding.device

        cross_entropy_loss = nn.CrossEntropyLoss()

        unfolded = context_embedding.unfold(dimension=1, size=K, step=1)  # [task_num, T-K+1, K, dim]
        unfolded = unfolded.permute(0, 1, 3, 2)
        T_new = unfolded.shape[1]

        prior_context = unfolded[:, :, :K - 1, :].mean(dim=2)  # [task_num, T-K+1, dim]
        last_context = unfolded[:, :, -1, :]  # [task_num, T-K+1, dim]
        first_context = unfolded[:, :, 0, :]  # [task_num, T-K+1, dim]
        sub_context = unfolded.mean(dim=2)  # [task_num, T-K+1, dim]
        later_context = unfolded[:, :, 1:, :].mean(dim=2)  # [task_num, T-K+1, dim]

        Wz = torch.matmul(self.agent.W, prior_context.permute(0, 2, 1))  # [dim, task_num, T-K+1]
        logits = torch.bmm(last_context, Wz)
        logits = logits - logits.max(dim=2, keepdim=True)[0]  # Stability

        labels = torch.arange(task_num, device=device).long()
        labels = labels[:, None].expand(-1, T_new).reshape(-1)
        loss = cross_entropy_loss(logits.reshape(-1, logits.shape[-1]), labels)

        return loss

    def bisimulation_loss_vectorized(self, indices):
        context_batch_1 = self.sample_context(indices)
        context_batch_2 = self.sample_context(indices)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices, b_size=8)
        t, b, _ = obs.size()

        obs = obs.reshape(t * b, -1)
        actions = actions.reshape(t * b, -1)

        context_embedding_1 = self.agent.only_infer_context_embeddings(context_batch_1)  # [task_num, context_dim]
        context_embedding_2 = self.agent.only_infer_context_embeddings(context_batch_2)  # [task_num, context_dim]

        task_num = context_embedding_1.shape[0]

        # Expand contexts
        context_A = context_embedding_1.unsqueeze(1).expand(task_num, task_num, -1)  # [task_num, task_num, context_dim]
        context_B = context_embedding_2.unsqueeze(0).expand(task_num, task_num, -1)  # [task_num, task_num, context_dim]

        # Flatten pairs
        context_A_flat = context_A.reshape(-1, context_embedding_1.shape[-1]).unsqueeze(1).expand(-1, t * b,
                                                                                                  -1).reshape(-1,
                                                                                                              context_embedding_1.shape[
                                                                                                                  -1])
        context_B_flat = context_B.reshape(-1, context_embedding_2.shape[-1]).unsqueeze(1).expand(-1, t * b,
                                                                                                  -1).reshape(-1,
                                                                                                              context_embedding_2.shape[
                                                                                                                  -1])

        obs_expand = obs.repeat(task_num * task_num, 1)
        actions_expand = actions.repeat(task_num * task_num, 1)

        pre_reward_A = self.reward_decoder(0, 0, context_A_flat, obs_expand, actions_expand)
        pre_reward_B = self.reward_decoder(0, 0, context_B_flat, obs_expand, actions_expand)
        reward_part = 0.5 * (pre_reward_A - pre_reward_B).pow(2)

        pre_transition_A = self.transition_decoder(0, 0, context_A_flat, obs_expand, actions_expand)
        pre_transition_B = self.transition_decoder(0, 0, context_B_flat, obs_expand, actions_expand)
        transition_part = 0.5 * (pre_transition_A - pre_transition_B).pow(2)

        context_part = (context_A_flat - context_B_flat).pow(2)

        loss = reward_part.mean().detach() + transition_part.mean().detach() - context_part.mean()
        return loss / (task_num * task_num)



    def _take_step(self, indices, context):

        wandb_stat = {}

        obs_dim = int(np.prod(self.env.observation_space.shape))
        action_dim = int(np.prod(self.env.action_space.shape))
        reward_in_context = context[:, :, obs_dim + action_dim].cpu().numpy()
        self.loss["non_sparse_ratio"] = len(reward_in_context[np.nonzero(reward_in_context)]) / np.size(reward_in_context)
        wandb_stat["non_sparse_ratio"] = len(reward_in_context[np.nonzero(reward_in_context)]) / np.size(reward_in_context)

        num_tasks = len(indices)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)

        policy_outputs, task_z, task_z_vars = self.agent(obs, context, task_indices=indices, for_update=True)


        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)


        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]


        if self.allow_backward_z:
            q1_pred = self.qf1(t, b, obs, actions, task_z)
            q2_pred = self.qf2(t, b, obs, actions, task_z)
            v_pred = self.vf(t, b, obs, task_z.detach())
        else:
            q1_pred = self.qf1(t, b, obs, actions, task_z.detach())
            q2_pred = self.qf2(t, b, obs, actions, task_z.detach())
            v_pred = self.vf(t, b, obs, task_z.detach())

        
        c_loss = self._divergence.dual_critic_loss(obs, new_actions.detach(), actions, task_z.detach())
        self.c_optimizer.zero_grad()
        c_loss.backward(retain_graph=True)
        self.c_optimizer.step()
        for _ in range(self._c_iter - 1):
            self._optimize_c(indices=indices, context=context)
        self.loss["c_loss"] = c_loss.item()
        wandb_stat["c_loss"] = c_loss.item()

        div_estimate = self._divergence.dual_estimate(
            obs, new_actions, actions, task_z.detach())
        self.loss["div_estimate"] = torch.mean(div_estimate).item()
        wandb_stat["div_estimate"] = torch.mean(div_estimate).item()

        with torch.no_grad():
            if self.use_brac and self.use_value_penalty:
                target_v_values = self.target_vf(t, b, next_obs, task_z) - self.get_alpha * div_estimate
            else:
                target_v_values = self.target_vf(t, b, next_obs, task_z)
        self.loss["target_v_values"] = torch.mean(target_v_values).item()
        wandb_stat["target_v_values"] = torch.mean(target_v_values).item()

        if self.use_club:
            self.club_model_optimizer.zero_grad()
            z_target = self.agent.encode_no_mean(context).detach()
            z_param = self.club_model(context[...,:self.club_model.input_size])

            z_mean = z_param[..., :self.latent_dim]
            z_var = F.softplus(z_param[..., self.latent_dim:])
            club_model_loss = self.club_model_loss_weight * ((z_target - z_mean)**2/(2*z_var) + torch.log(torch.sqrt(z_var))).mean()
            club_model_loss.backward()
            self.loss["club_model_loss"] = club_model_loss.item()
            wandb_stat["club_model_loss"] = club_model_loss.item()
            self.club_model_optimizer.step()

        self.context_optimizer.zero_grad()
        self.reward_decoder_optimizer.zero_grad()
        self.transition_decoder_optimizer.zero_grad()

        if self.use_club:
            z_target = self.agent.encode_no_mean(context)
            z_param = self.club_model(context[...,:self.club_model.input_size]).detach()
            z_mean = z_param[..., :self.latent_dim]
            z_var = F.softplus(z_param[..., self.latent_dim:])
            z_t, z_b, _ = z_mean.size()
            position = - ((z_target-z_mean)**2/z_var).mean()
            z_mean_expand = z_mean[:, :, None, :].expand(-1, -1, z_b, -1).reshape(z_t, z_b**2, -1)
            z_var_expand = z_var[:, :, None, :].expand(-1, -1, z_b, -1).reshape(z_t, z_b**2, -1)
            z_target_repeat = z_target.repeat(1, z_b, 1)
            negative = - ((z_target_repeat-z_mean_expand)**2/z_var_expand).mean()
            club_loss = self.club_loss_weight * (position - negative)
            club_loss.backward(retain_graph=True)
            self.loss["club_loss"] = club_loss.item()
            wandb_stat["club_loss"] = club_loss.item()

        if self.use_FOCAL_cl:
            z_loss = self.z_loss_weight * self.FOCAL_z_loss(indices=indices, task_z=task_z, task_z_vars=task_z_vars, b=b)
            z_loss.backward(retain_graph=True)
            self.loss["z_loss"] = z_loss.item()
            wandb_stat["z_loss"] = z_loss.item()

        pred_rewardss = rewards.view(self.batch_size * num_tasks, -1)
        rew_pred = self.reward_decoder.forward(0, 0, task_z, obs, actions)
        rew_loss = self.pred_loss(pred_rewardss, rew_pred) * 1.0
        self.loss["reward_prediction_loss"] = torch.mean(rew_loss).item()
        wandb_stat["reward_prediction_loss"] = torch.mean(rew_loss).item()

        pred_transitions = next_obs.view(self.batch_size * num_tasks, -1)
        transition_pred = self.transition_decoder.forward(0, 0, task_z, obs, actions)
        transition_loss = self.pred_loss(pred_transitions, transition_pred) * 1.0
        self.loss["transition_prediction_loss"] = torch.mean(transition_loss).item()
        wandb_stat["transition_prediction_loss"] = torch.mean(transition_loss).item()

        taco_loss = self.TACO_loss_vectorized(indices, K=8)
        self.loss["taco_loss"] = taco_loss.item()
        wandb_stat["taco_loss"] = taco_loss.item()
        taco_loss = taco_loss * 1.0
        sub_loss = taco_loss * 1.0 

        sub_loss.backward(retain_graph=True)

        bisimulation_loss = self.bisimulation_loss_vectorized(indices)
        bisimulation_loss = bisimulation_loss * 1.0
        self.loss["bisimulation_loss"] = bisimulation_loss.item()
        wandb_stat["bisimulation_loss"] = bisimulation_loss.item()

        bisimulation_loss.backward(retain_graph=True)


        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward(retain_graph=True)
        self.loss["qf_loss"] = qf_loss.item()
        wandb_stat["qf_loss"] = qf_loss.item()

        self.loss["q_target"] = torch.mean(q_target).item()
        wandb_stat["q_target"] = torch.mean(q_target).item()

        self.loss["q1_pred"] = torch.mean(q1_pred).item()
        wandb_stat["q1_pred"] = torch.mean(q1_pred).item()

        self.loss["q2_pred"] = torch.mean(q2_pred).item()
        wandb_stat["q2_pred"] = torch.mean(q2_pred).item()

        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.context_optimizer.step()

        self.reward_decoder_optimizer.step()
        self.transition_decoder_optimizer.step()

        # compute min Q on the new actions
        min_q_new_actions = torch.min(self.qf1(t, b, obs, new_actions, task_z.detach()),
                                        self.qf2(t, b, obs, new_actions, task_z.detach()))

        # vf update
        if self.max_entropy:
            v_target = min_q_new_actions - log_pi
        else:
            v_target = min_q_new_actions
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward(retain_graph=True)
        self.vf_optimizer.step()
        self._update_target_network()
        self.loss["vf_loss"] = vf_loss.item()
        wandb_stat["vf_loss"] = vf_loss.item()

        self.loss["v_target"] = torch.mean(v_target).item()
        wandb_stat["v_target"] = torch.mean(v_target).item()

        self.loss["v_pred"] = torch.mean(v_pred).item()
        wandb_stat["v_pred"] = torch.mean(v_pred).item()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        # BRAC:
        if self.use_brac:
            if self.max_entropy:
                policy_loss = (log_pi - log_policy_target + self.get_alpha.detach() * div_estimate).mean()
            else:
                policy_loss = (-log_policy_target + self.get_alpha.detach() * div_estimate).mean()
        else:
            if self.max_entropy:
                policy_loss = (log_pi - log_policy_target).mean()
            else:
                policy_loss = - log_policy_target.mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value**2).sum(dim=-1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optimizer.step()

        self.loss["policy_loss"] = policy_loss.item()
        wandb_stat["policy_loss"] = policy_loss.item()
        # optimize for c network (which computes dual-form divergences)
        # BRAC for training alpha:
        a_loss = -torch.mean(self._alpha_var * (div_estimate - self._target_divergence).detach())
        a_loss.backward()
        with torch.no_grad():
            self._alpha_var -= self.alpha_lr * self._alpha_var.grad
            # Manually zero the gradients after updating weights
            self._alpha_var.grad.zero_()
        self.loss["a_loss"] = a_loss.item()
        wandb_stat["a_loss"] = a_loss.item()

        if self._num_steps % self._visit_num_steps_train == 0:
            print(self.loss)
        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()

            # z_mean = np.mean(np.abs(ptu.get_numpy(self.agent.z_means[0])))
            for i in range(len(self.agent.z_means[0])):
                z_mean = ptu.get_numpy(self.agent.z_means[0][i])
                name = 'Z mean train' + str(i)
                self.eval_statistics[name] = z_mean
                
            #z_mean1 = ptu.get_numpy(self.agent.z_means[0][0])
            #z_mean2 = ptu.get_numpy(self.agent.z_means[0][1])
            #z_mean3 = ptu.get_numpy(self.agent.z_means[0][2])
            #z_mean4 = ptu.get_numpy(self.agent.z_means[0][3])
            #z_mean5 = ptu.get_numpy(self.agent.z_means[0][4])

            z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))
            #self.eval_statistics['Z mean train1'] = z_mean1
            #self.eval_statistics['Z mean train2'] = z_mean2
            #self.eval_statistics['Z mean train3'] = z_mean3
            #self.eval_statistics['Z mean train4'] = z_mean4
            #self.eval_statistics['Z mean train5'] = z_mean5

            self.eval_statistics['Z variance train'] = z_sig
            self.eval_statistics['task idx'] = indices[0]

            if self.use_club:
                self.eval_statistics['Club model Loss'] = ptu.get_numpy(club_model_loss)
                self.eval_statistics['Club Loss'] = ptu.get_numpy(club_loss)
            if self.use_FOCAL_cl:
                self.eval_statistics['Z Loss'] = ptu.get_numpy(z_loss)

            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            if self.use_brac:
                self.eval_statistics['Dual Critic Loss'] = np.mean(ptu.get_numpy(c_loss))
            self.eval_statistics.update(create_stats_ordered_dict('Q Predictions',  ptu.get_numpy(q1_pred)))
            self.eval_statistics.update(create_stats_ordered_dict('V Predictions',  ptu.get_numpy(v_pred)))
            self.eval_statistics.update(create_stats_ordered_dict('Log Pis',        ptu.get_numpy(log_pi)))
            self.eval_statistics.update(create_stats_ordered_dict('Policy mu',      ptu.get_numpy(policy_mean)))
            self.eval_statistics.update(create_stats_ordered_dict('Policy log std', ptu.get_numpy(policy_log_std)))
            self.eval_statistics.update(create_stats_ordered_dict('alpha',          ptu.get_numpy(self._alpha_var).reshape(-1)))
            self.eval_statistics.update(create_stats_ordered_dict('div_estimate',   ptu.get_numpy(div_estimate)))
        return ptu.get_numpy(self.agent.z_means), ptu.get_numpy(self.agent.z_vars), wandb_stat

    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            vf=self.vf.state_dict(),
            target_vf=self.target_vf.state_dict(),
            context_encoder=self.agent.context_encoder.state_dict(),
            club_model=self.club_model.state_dict(),
            c=self.c.state_dict(),
            )
        return snapshot
