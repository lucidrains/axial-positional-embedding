import torch
from torch import nn
from operator import mul
from functools import reduce

class AxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, axial_shape = ()):
        super().__init__()

        self.dim = dim
        self.shape = axial_shape
        self.max_seq_len = reduce(mul, axial_shape, 1)

        self.weights = ParameterList(self, 'weights', len(axial_shape))

        for ind, shape in enumerate(self.shape):
            ax_shape = [1] * len(self.shape)
            ax_shape[ind] = shape
            ax_shape = (1, *ax_shape, dim)
            ax_emb = nn.Parameter(torch.zeros(ax_shape).normal_(0, 1))
            self.weights.append(ax_emb)

    def forward(self, x):
        b, t, e = x.shape
        assert (t < self.max_seq_len), f'Sequence length ({t}) must be less than the maximum sequence length allowed ({self.max_seq_len})'
        embs = []

        for ax_emb in self.weights.to_list():
            expand_shape = (b, *self.shape, self.dim)
            emb = ax_emb.expand(expand_shape).reshape(b, self.max_seq_len, self.dim)
            embs.append(emb)

        pos_emb = sum(embs)
        return pos_emb[:, :t].to(x)

# a mock parameter list object until below issue is resolved
# https://github.com/pytorch/pytorch/issues/36035
class ParameterList(object):
    def __init__(self, kls, prefix, length):
        self.ind = 0
        self.kls = kls
        self.prefix = prefix
        self.length = length

    def _keyname(self, prefix, ind):
        return f'{prefix}_{ind}'

    def append(self, x):
        setattr(self.kls, self._keyname(self.prefix, self.ind), x)
        self.ind += 1

    def to_list(self):
        return [getattr(self.kls, self._keyname(self.prefix, i)) for i in range(self.length)]

