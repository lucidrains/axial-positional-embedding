import torch
from torch import nn
from operator import mul
from functools import reduce

class AxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, axial_shape, axial_dims = None):
        super().__init__()

        self.dim = dim
        self.shape = axial_shape
        self.max_seq_len = reduce(mul, axial_shape, 1)

        self.summed = axial_dims is None
        axial_dims = ((dim,) * len(axial_shape)) if self.summed else axial_dims

        assert len(self.shape) == len(axial_dims), 'number of axial dimensions must equal the number of dimensions in the shape'
        assert self.summed or not self.summed and sum(axial_dims) == dim, f'axial dimensions must sum up to the target dimension {dim}'

        self.weights = ParameterList(self, 'weights', len(axial_shape))

        for ind, (shape, axial_dim) in enumerate(zip(self.shape, axial_dims)):
            ax_shape = [1] * len(self.shape)
            ax_shape[ind] = shape
            ax_shape = (1, *ax_shape, axial_dim)
            ax_emb = nn.Parameter(torch.zeros(ax_shape).normal_(0, 1))
            self.weights.append(ax_emb)

    def forward(self, x):
        b, t, e = x.shape
        assert (t <= self.max_seq_len), f'Sequence length ({t}) must be less than the maximum sequence length allowed ({self.max_seq_len})'
        embs = []

        for ax_emb in self.weights.to_list():
            axial_dim = ax_emb.shape[-1]
            expand_shape = (b, *self.shape, axial_dim)
            emb = ax_emb.expand(expand_shape).reshape(b, self.max_seq_len, axial_dim)
            embs.append(emb)

        pos_emb = sum(embs) if self.summed else torch.cat(embs, dim=-1)
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

# Axial Positional Embedding for Images

class AxialPositionalEmbeddingImage(nn.Module):
    def __init__(self, dim, axial_shape, axial_dims = None):
        super().__init__()
        assert len(axial_shape) == 2, 'Axial shape must have 2 dimensions for images'
        self.pos_emb = AxialPositionalEmbedding(dim, axial_shape, axial_dims)

    def forward(self, img):
        b, c, h, w = img.shape
        img = img.permute(0, 2, 3, 1).reshape(b, h * w, c)
        pos_emb = self.pos_emb(img)
        return pos_emb.reshape(b, h, w, c).permute(0, 3, 1, 2)
