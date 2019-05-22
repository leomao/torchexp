import collections.abc
import numpy as np
import torch as th
import gin


def _apply_all(*val, fn):
    if len(val) == 1:
        val = val[0]

    if isinstance(val, collections.abc.Sequence):
        return [_apply_all(x, fn=fn) for x in val]
    elif isinstance(val, collections.abc.Mapping):
        return {k: _apply_all(x, fn=fn) for k, x in val.items()}
    else:
        return fn(val)


@gin.configurable(whitelist=['device'], module='torchexp')
def CC(args, device='cpu'):
    return _apply_all(args, fn=lambda x: x.to(device))


def np2tor(*args, requires_grad=False):
    '''
    note: this operation may not copy the data
    '''
    def _np2tor(arr):
        return th.from_numpy(arr).requires_grad_(requires_grad)
    return _apply_all(*args, fn=_np2tor)


def _tor2np(arr):
    nparr = arr.cpu().detach().numpy()
    if nparr.size == 1:
        return np.asscalar(nparr)
    else:
        return nparr


def tor2np(*args):
    '''
    note: this operation may not copy the data
    '''
    return _apply_all(*args, fn=_tor2np)


def one_hot(index, n, dtype=th.float32) -> th.Tensor:
    '''
    Args:
        index: Tensor(*sizes) - one hot index (zero based)
        n: int - the size of one-hot vectors
        dtype: torch.dtype - the data type of the return one-hot vectors,
            default is th.float32
    Returns:
        onehot: Tensor(*sizes, n) - converted one-hot vectors
    '''
    last_dim = len(index.size())
    onehot = index.new_zeros(*index.size(), n, dtype=dtype)
    onehot.scatter_(last_dim, index.data.unsqueeze(last_dim), 1.)
    return onehot


def length_to_mask(lens, max_len=None, dtype=None):
    if max_len is None:
        max_len = lens.max().item()
    seq = th.arange(max_len, device=lens.device, dtype=lens.dtype)
    mask = seq.expand(*lens.size(), -1) < lens.unsqueeze(-1)
    if dtype is not None:
        mask = mask.to(dtype)
    return mask


def mask_seqs(seqs, lens):
    '''
    Args:
        seqs: Tensor(*size, max_length)
        lens: Tensor(*size)
    Returns:
        masked_seqs: `seqs` masked with length `lens`
    '''
    mask = length_to_mask(lens, max_len=seqs.size(-1), dtype=th.float32)
    masked_seqs = mask * seqs
    return masked_seqs


# Computation shortcut
def bmv(bm: th.Tensor, bv: th.Tensor) -> th.Tensor:
    '''
    batch matrix-vector product
    Args:
        bm: Tensor(B, M, N)
        bv: Tensor(B, N)
    Returns:
        Tensor(B, M) - Batch matrix-vector product of bm and bv
    '''
    return th.bmm(bm, bv.unsqueeze(-1)).squeeze()


def infinite_iter(iterable):
    it = iter(iterable)
    while True:
        try:
            ret = next(it)
            yield ret
        except StopIteration:
            it = iter(iterable)
