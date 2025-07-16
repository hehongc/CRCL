"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import torch
from torch import nn as nn
from torch.nn import functional as F
from rlkit.policies.base import Policy
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.modules import LayerNorm
import pdb

import pywt
import numpy as np
import numpy

from torch.nn.utils import weight_norm
import torch.nn.init as init

# import rlkit.torch.transformer as transformer

def identity(x):
    return x

class Mlp(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            batch_attention=False,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            output_activation_half=False,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            use_dropout=False,
            layer_norm_kwargs=None,
    ):
        self.save_init_params(locals())
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        # when output is [mean, var], if output_activation_half is true ,just activate mean, not var
        self.output_activation_half = output_activation_half
        self.layer_norm = layer_norm
        self.use_dropout = use_dropout
        self.fcs = []
        self.layer_norms = []
        self.dropouts = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)
            
            if self.use_dropout:
                dropout_n = nn.Dropout(0.1)
                self.__setattr__("drop_out{}".format(i), dropout_n)
                self.dropouts.append(dropout_n)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)
        
        self.batch_attention = batch_attention
        self.transition_attention = transformer.BatchTransitionAttention(
            hidden=100,
            input_size=input_size,
            output_size=input_size,
            n_layers=3,
            attn_heads=1,
            dropout=0.1
        ) if self.batch_attention else None

    def forward(self, input, return_preactivations=False):
        if self.batch_attention:
            input = self.transition_attention(input)
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
            if self.use_dropout and i < len(self.fcs) - 1:
                h = self.dropouts[i](h)
        preactivation = self.last_fc(h)
        half_output_size = int(self.output_size/2)
        if self.output_activation_half:
            output =  torch.cat([self.output_activation(preactivation[..., :half_output_size]), preactivation[..., half_output_size:]], dim=-1)
        else:
            output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class FlattenMlp(Mlp):
    """
    if there are multiple inputs, concatenate along dim 1
    """

    def forward(self, meta_size=16, batch_size=256, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=-1)
        return super().forward(flat_inputs, **kwargs)

class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return self.eval_np(obs)


class TanhMlpPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """
    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        super().__init__(*args, output_activation=torch.tanh, **kwargs)


class MlpEncoder(Mlp):
    '''
    encode context via MLP
    '''

    def reset(self, num_tasks=1):
        pass


class RecurrentEncoder(FlattenMlp):
    '''
    encode context via recurrent network
    '''

    def __init__(self,
                 *args,
                 **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.hidden_dim = self.hidden_sizes[-1]
        self.register_buffer('hidden', torch.zeros(1, 1, self.hidden_dim))

        # input should be (task, seq, feat) and hidden should be (task, 1, feat)

        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True)

    def forward(self, in_, return_preactivations=False):
        # expects inputs of dimension (task, seq, feat)
        task, seq, feat = in_.size()
        out = in_.view(task * seq, feat)

        # embed with MLP
        for i, fc in enumerate(self.fcs):
            out = fc(out)
            out = self.hidden_activation(out)

        out = out.view(task, seq, -1)
        out, (hn, cn) = self.lstm(out, (self.hidden, torch.zeros(self.hidden.size()).to(ptu.device)))
        self.hidden = hn
        # take the last hidden state to predict z
        out = out[:, -1, :]

        # output layer
        preactivation = self.last_fc(out)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output

    def reset(self, num_tasks=1):
        self.hidden = self.hidden.new_full((1, num_tasks, self.hidden_dim), 0)



# CORRO Network Part
class SelfAttnEncoder(PyTorchModule):
    def __init__(self,
                 input_dim,
                 num_output_mlp=0,
                 task_gt_dim=5,
                 init_w=3e-3,
                 ):
        super(SelfAttnEncoder, self).__init__()

        self.input_dim = input_dim
        self.score_func = nn.Linear(input_dim, 1)

        self.num_output_mlp = num_output_mlp

        if num_output_mlp > 0:
            self.output_mlp = Mlp(
                input_size=input_dim,
                output_size=task_gt_dim,
                hidden_sizes=[200 for i in range(num_output_mlp - 1)]
            )

        self.score_func.weight.data.uniform_(-init_w, init_w)
        self.score_func.bias.data.uniform_(-init_w, init_w)


    def forward(self, input, z_mean):
        # b = task_num
        # N = trajectory_len
        b, N, dim = input.shape

        z_mean = [z.repeat(N, 1) for z in z_mean]
        z_mean = torch.cat(z_mean, dim=0)

        score_func_input_tuple_representations = input.reshape(-1, dim)
        score_func_input_z_mean = z_mean.reshape(score_func_input_tuple_representations.shape[0], -1)
        score_func_input = torch.cat([score_func_input_tuple_representations, score_func_input_z_mean], dim=-1)

        # input.reshape(-1, dim) -> [task_num * trajectory_len, dim]
        # scores.shape = [task_num, trajectory_len]
        # scores = self.score_func(input.reshape(-1, dim)).reshape(b, N)
        scores = self.score_func(score_func_input).reshape(b, N)
        scores_before_softmax = scores
        scores_sigmoid = F.sigmoid(scores)

        # scores.shape = [task_num, trajectory_len]
        scores = F.softmax(scores, dim=-1)

        reverse_scores = 1 - scores

        # scores.shape = [task_num, trajectory_len, 1]
        # context.shape = [task_num, trajectory_len, dim]
        context = scores.unsqueeze(-1).expand_as(input).mul(input)
        # context_sum.shape = [task_num, dim]
        context_sum = context.sum(1)

        # [task_num, trajectory_len, dim]
        reverse_context = reverse_scores.unsqueeze(-1).expand_as(input).mul(input)
        # context_sum.shape = [task_num, dim]
        reverse_context_sum = reverse_context.sum(1)

        # return context, context_sum, scores, scores_sigmoid, reverse_context, reverse_context_sum
        return context, context_sum, scores, scores_before_softmax, reverse_context, reverse_context_sum

    # def forward(self, input):
    #     # b = task_num
    #     # N = trajectory_len
    #     b, N, dim = input.shape
    #
    #     # input.reshape(-1, dim) -> [task_num * trajectory_len, dim]
    #     # scores.shape = [task_num, trajectory_len]
    #     scores = self.score_func(input.reshape(-1, dim)).reshape(b, N)
    #     # scores.shape = [task_num, trajectory_len]
    #     scores = F.softmax(scores, dim=-1)
    #
    #     # scores.shape = [task_num, trajectory_len, 1]
    #     # context.shape = [task_num, trajectory_len, dim]
    #     context = scores.unsqueeze(-1).expand_as(input).mul(input)
    #
    #     return context




