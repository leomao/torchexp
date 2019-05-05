import collections.abc
import math
import torch as th
from torch import nn
import torch.nn.functional as F

import torchexp as te


class NormSpec:
    def __init__(self, norm_type, *args, **kwargs):
        self.norm_type = norm_type
        self.args = args
        self.kws = kwargs

    def instantiate(self, in_features, nd=1):
        if self.norm_type == 'batch':
            if nd == 1:
                return nn.BatchNorm1d(in_features, *self.args, **self.kws)
            elif nd == 2:
                return nn.BatchNorm2d(in_features, *self.args, **self.kws)
            elif nd == 3:
                return nn.BatchNorm3d(in_features, *self.args, **self.kws)
        elif self.norm_type == 'instance':
            if nd == 1:
                return nn.InstanceNorm1d(in_features, *self.args, **self.kws)
            elif nd == 2:
                return nn.InstanceNorm2d(in_features, *self.args, **self.kws)
            elif nd == 3:
                return nn.InstanceNorm3d(in_features, *self.args, **self.kws)
        elif self.norm_type == 'group':
            return nn.GroupNorm(self.args[0], in_features,
                                *self.args[1:], **self.kws)
        else:
            raise ValueError(f'The norm type {self.norm_type} is not supported.')


def MLPStack(input_size, specs, bias_all=True):
    layers = []
    for spec in specs:
        if isinstance(spec, nn.Module):
            # `spec` must not change the number of features
            layers.append(spec)
        else:
            if isinstance(spec, int):
                next_size = spec
                bias = bias_all
            elif isinstance(spec, tuple) and len(spec) == 2:
                next_size, bias = spec
            else:
                raise ValueError(f'The spec `{spec}` is not acceptable'
                                 ' as a layer in MLPStack')
            layers.append(nn.Linear(input_size, spec, bias))
            input_size = spec
    return nn.Sequential(*layers)


class ConvSpec:
    def __init__(self, num_channels, kernel_size,
                 s=1, p=0, d=1, g=1, bias=True, **kwargs):
        '''
            s => stride
            p => padding
            d => dilation
            g => groups
            bias => bias
        '''
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = s
        self.padding = p
        self.dilation = d
        self.groups = g
        self.bias = bias
        for k, v in kwargs.items():
            for name in self.__dict__:
                if name.startwidth(k):
                    self.__setattr__(name, v)
                    break

    def conv_kwargs(self):
        return {
            'out_channels': self.num_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'dilation': self.dilation,
            'groups': self.groups,
            'bias': self.bias,
        }


def _ConvStack(in_channels, specs, nd):
    if nd == 1:
        _Conv = nn.Conv1d
    elif nd == 2:
        _Conv = nn.Conv2d
    elif nd == 3:
        _Conv = nn.Conv3d
    else:
        raise ValueError('`nd` should be 1, 2 or 3')
    prev_c = in_channels
    layers = []
    for spec in specs:
        if isinstance(spec, nn.Module):
            # `spec` must not change in_channel
            layers.append(spec)
        else:
            if isinstance(spec, collections.abc.Sequence):
                spec = ConvSpec(*spec)
            elif isinstance(spec, collections.abc.Mapping):
                spec = ConvSpec(**spec)

            if not isinstance(spec, ConvSpec):
                raise TypeError(f'type `{type(spec)}` is not acceptable'
                                f' as a layer in Conv{nd}dStack')

            layers.append(_Conv(prev_c, **spec.conv_kwargs()))
            prev_c = spec.num_channels
    return nn.Sequential(*layers)


def Conv1dStack(in_channels, specs):
    return _ConvStack(in_channels, specs, 1)


def Conv2dStack(in_channels, specs):
    return _ConvStack(in_channels, specs, 2)


def Conv3dStack(in_channels, specs):
    return _ConvStack(in_channels, specs, 3)


class Attention(nn.Module):
    def __init__(self, input_size, feat_size, attn_size):
        super(Attention, self).__init__()
        self.w_feat = nn.Linear(feat_size, attn_size, bias=False)
        self.w_input = nn.Linear(input_size, attn_size, bias=True)
        self.v_attn = nn.Parameter(th.empty(attn_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.attn_v.size(0))
        self.attn_v.data.normal_(-stdv, stdv)

    def forward(self, z, features):
        # features:      B * feat_size * N
        # permuted(tf):  B * N * feat_size -> B * N * attn_size
        # self.w_inp(z): B * attn_size
        tf = features.permute(0, 2, 1)
        attn = F.relu(self.w_feat(tf) + self.w_inp(z).unsqueeze(1))
        # att:   B * N * attn_size
        # alpha: B * N
        alpha = attn.matmul(self.v_attn).softmax(-1)
        # batch matrix-vector product
        # B * feat_size * N <dot> B * N * 1
        # ctx: B * feat_size
        ctx = te.bmv(features, alpha)
        return ctx, alpha


class CosineAttention(nn.Module):
    def __init__(self, input_size, feat_size, attn_size):
        super().__init__()
        self.w_feat = nn.Linear(feat_size, attn_size)
        self.w_inp = nn.Linear(input_size, attn_size)

    def forward(self, z, features):
        # features:      B * feat_size * N
        # permuted(tf):  B * N * feat_size -> B * N * attn_size
        # self.w_inp(z): B * attn_size
        tf = features.permute(0, 2, 1)
        sim = F.cosine_similarity(
            self.w_feat(tf),
            self.w_inp(z).unsqueeze(1),
            dim=-1
        ).squeeze()
        # sim, alpha:  B * N
        alpha = sim.softmax(-1)
        # batch matrix-vector product
        # B * feat_size * N <dot> B * N * 1
        # ctx: B * feat_size
        ctx = te.bmv(features, alpha)
        return ctx, alpha


# deprecated functions, for reference


def scaled_dot_product_attention(q, k, v, temperature=None, mask=None,
                                 dropout=None):
    '''
    Aargs:
        q: Tensor(B, M, sk)
        k: Tensor(B, N, sk)
        v: Tensor(B, N, sv)
    '''
    if temperature is None:
        temperature = q.size(-1) ** 0.5
    logits = th.bmm(q, k.transpose(-2, -1)) / temperature
    if mask is not None:
        logits.masked_fill(mask, -np.inf)
    weights = logits.softmax(dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    return th.bmm(weights, v)


class MultiHeadAttention(nn.Module):
    ''' Multi-head Attention '''

    def __init__(self, input_sizes, out_size, h, sk, sv=None, p_dropout=None,
                 transform_bias=False):
        '''
        Args:
            input_sizes (int or tuple): the sizes of queries, keys, values
                from input
            sk: size of keys and queries after linear transform
            sv: size of values after linear transform
            h:  number of heads
            out_size: size of output
        '''
        self.h = h
        self.sk = sk
        if sv is None:
            sv = sk
        self.sv = sv
        if type(input_sizes) is int:
            input_sq = input_sk = input_sv = input_sizes
        elif (isinstance(input_sizes, collections.abc.Sequence)
              and len(input_sizes) == 3):
            input_sq, input_sk, input_sv = input_sizes
        else:
            raise TypeError('the `input_sizes` should be int or 3-tuple of int')

        self.ws_q = nn.Linear(input_sq, h * sk, bias=transform_bias)
        self.ws_k = nn.Linear(input_sk, h * sk, bias=transform_bias)
        self.ws_v = nn.Linear(input_sv, h * sv, bias=transform_bias)
        self.w_out = nn.Linear(h * sv, out_size)

        self.dropout = None
        if p_dropout is not None and p_dropout > 0:
            self.dropout = nn.Dropout(p_dropout)

    @staticmethod
    def _heads_to_batch(t, size):
        '''
        Args:
            t: Tensor(B, N, h * sh)
        Returns:
            Tensor(B * h, N, sh)
        '''
        B, N, _ = t.size()
        # we need to use reshape since we cannot use view after transpose
        return t.view(B, N, -1, size).transpose(1, 2).reshape(-1, N, size)

    def forward(self, q, k, v, mask=None):
        '''
        Aargs:
            q: Tensor(B, M, input_sq)
            k: Tensor(B, N, input_sk)
            v: Tensor(B, N, input_sv)
        '''
        sk, sv = self.sk, self.sv
        if len(q.size()) == 2:
            q = q.unsqueeze(dim=1)
        B, M, _ = q.size()
        # merge batch_size and h for batch matrix-matrix product
        q = self._heads_to_batch(self.ws_q(q), sk)
        k = self._heads_to_batch(self.ws_k(k), sk)
        v = self._heads_to_batch(self.ws_v(v), sv)
        if mask is not None:
            mask = mask.repeat(self.h, -1, -1)
        out = scaled_dot_product_attention(q, k, v, mask=mask,
                                           dropout=self.dropout)
        # we need to use reshape since we cannot use view after transpose
        out = out.view(B, -1, M, sv).transpose(1, 2).reshape(B, M, -1)
        return self.w_out(out).squeeze(dim=1)


class SelfAttention(MultiHeadAttention):
    def __init__(self, input_size, out_size, h, sk, **kwargs):
        super().__init__(input_size, out_size, h, sk, **kwargs)

    def forward(self, x, mask=None):
        super().forward(x, x, x, mask=mask)


__all__ = [
    'NormSpec',
    'MLPStack',
    'ConvSpec',
    'Conv1dStack',
    'Conv2dStack',
    'Conv3dStack',
    'Attention',
    'CosineAttention',
]
