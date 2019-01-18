import collections.abc
import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F

import torchexp as te


def MLPStack(input_size, specs):
    layers = []
    for spec in specs:
        if isinstance(spec, nn.Module):
            # assume `spec` will not change the number of features
            layers.append(spec)
        else:
            layers.append(nn.Linear(input_size, spec))
            input_size = spec
    return nn.Sequential(*layers)


class ConvSpec:
    def __init__(self, num_channels, kernel_size, s=1, p=0, d=1, g=1, bn=True):
        '''
            s => stride
            p => padding
            d => dilation
            g => groups
            bn => batch_norm
        '''
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = s
        self.padding = p
        self.dilation = d
        self.groups = g
        self.batch_norm = bn

    def conv_kwargs(self):
        return {
            'out_channels': self.n_c,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'dilation': self.dilation,
            'groups': self.groups,
        }


def ConvStack(in_channels, specs):
    prev_c = in_channels
    layers = []
    for spec in specs:
        if isinstance(spec, nn.Module):
            # assume `spec` will not change in_channel
            layers.append(spec)
        else:
            if isinstance(spec, collections.abc.Sequence):
                spec = ConvSpec(*spec)
            elif isinstance(spec, collections.abc.Mapping):
                spec = ConvSpec(**spec)

            if not isinstance(spec, ConvSpec):
                raise TypeError(f'type `{type(spec)}` is not acceptable'
                                ' as a layer in ConvStack')

            layers.append(nn.Conv2d(prev_c, **spec.conv_kwargs()))
            if spec.batch_norm:
                layers.append(nn.BatchNorm2d(spec.num_channels))
            prev_c = spec.num_channels
    return nn.Sequential(*layers)


class Attention(nn.Module):
    def __init__(self, input_size, feat_size, atten_size):
        super(Attention, self).__init__()
        self.w_feat = nn.Linear(feat_size, atten_size, bias=False)
        self.w_input = nn.Linear(inp_size, atten_size, bias=True)
        self.v_atten = nn.Parameter(th.empty(atten_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.atten_v.size(0))
        self.atten_v.data.normal_(-stdv, stdv)

    def forward(self, z, features):
        # features:      B * feat_size * N
        # permuted(tf):  B * N * feat_size -> B * N * atten_size
        # self.w_inp(z): B * atten_size
        tf = features.permute(0, 2, 1)
        att = F.relu(self.w_feat(tf) + self.w_inp(z).unsqueeze(1))
        # att:   B * N * atten_size
        # alpha: B * N
        alpha = att.matmul(self.v_atten).softmax(-1)
        # batch matrix-vector product
        # B * feat_size * N <dot> B * N * 1
        # ctx: B * feat_size
        ctx = te.bmv(features, alpha)
        return ctx, alpha


class CosineAttention(nn.Module):
    def __init__(self, inp_size, feat_size, atten_size):
        super(Attention, self).__init__()
        self.w_feat = nn.Linear(feat_size, atten_size)
        self.w_inp = np.Linear(inp_size, atten_size)

    def forward(self, z, features):
        # features:      B * feat_size * N
        # permuted(tf):  B * N * feat_size -> B * N * atten_size
        # self.w_inp(z): B * atten_size
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
        ctx = tm.bmv(features, alpha)
        return ctx, alpha


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

    def __init__(self, input_sizes, out_size, h, sk, sv=None, p_dropout=None):
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

        self.ws_q = nn.Linear(input_sq, h * sk)
        self.ws_k = nn.Linear(input_sk, h * sk)
        self.ws_v = nn.Linear(input_sv, h * sv)
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
    def __init__(self, input_size, out_size, h, sk, sv=None, p_dropout=None):
        super().__init__(input_size, out_size, h, sk, sv, p_dropout)

    def forward(self, x, mask=None):
        super().forward(x, x, x, mask=mask)
