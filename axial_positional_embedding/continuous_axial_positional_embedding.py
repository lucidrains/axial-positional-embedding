from __future__ import annotations
from math import ceil
from functools import reduce
from operator import mul
from itertools import zip_longest

import torch
from torch import nn, tensor, Size, Tensor
from torch.nn import Module, ModuleList

from einops import rearrange

# helper methods

def exists(v):
    return v is not None

# mlp - continuously parameterizing each axial position

def MLP(
    dim_in,
    dim_out,
    depth = 2,
    expansion = 2
):
    curr_dim = dim_in
    dim_hidden = int(expansion * max(dim_in, dim_out))

    layers = []

    for _ in range(depth):
        layers.append(nn.Linear(curr_dim, dim_hidden))
        layers.append(nn.SiLU())

        curr_dim = dim_hidden

    layers.append(nn.Linear(curr_dim, dim_out))
    return nn.Sequential(*layers)

# main class

class ContinuousAxialPositionalEmbedding(Module):
    def __init__(
        self,
        dim,
        num_axial_dims,
        mlp_depth = 2,
        mlp_expansion = 2.
    ):
        super().__init__()
        self.num_axial_dims = num_axial_dims
        self.mlps = ModuleList([MLP(1, dim, depth = mlp_depth, expansion = mlp_expansion) for _ in range(num_axial_dims)])

        self.register_buffer('dummy', tensor(0), persistent = False)

    @property
    def device(self):
        return self.dummy.device

    def combine_factorized(
        self,
        axial_embeds: list[Tensor],
        axial_dims: tuple[int, ...] | None = None,
        flatten = False
    ):
        if not exists(axial_dims):
            axial_dims = tuple(axial_embed.shape[0] for axial_embed in axial_embeds)

        assert len(axial_dims) == len(axial_embeds)

        axial_embeds = [axial_embed[:axial_dim] for axial_embed, axial_dim in zip(axial_embeds, axial_dims)]

        axial_embed, *rest_axial_embeds = axial_embeds

        for rest_axial_embed in rest_axial_embeds:
            axial_embed = axial_embed[..., None, :] + rest_axial_embed

        assert axial_embed.shape[:-1] == axial_dims

        if flatten:
            axial_embed = rearrange(axial_embed, '... d -> (...) d')

        return axial_embed

    def forward_with_seq_len(
        self,
        seq_len: int,
        axial_dims: Tensor | Size | tuple[int, ...] = (),
    ):
        ndims = self.num_axial_dims
        assert len(axial_dims) in (ndims, ndims - 1)

        if len(axial_dims) == (ndims - 1):
            stride = reduce(mul, (*axial_dims,))

            outer_dim = ceil(seq_len / stride)
            axial_dims = (outer_dim, *axial_dims)

        axial_embeds = self.forward(axial_dims, flatten = True)

        return axial_embeds[:seq_len]

    def forward(
        self,
        axial_dims: Tensor | Size | tuple[int, ...],
        return_factorized = False,   # whether to return list[Tensor] of factorized axial positional embeddings
        flatten = False,             # whether to flatten axial dims
    ):
        axial_embeds = []

        for mlp, axial_dim in zip(self.mlps, axial_dims):
            seq = torch.arange(axial_dim, device = self.device, dtype = torch.float)
            axial_embed = mlp(rearrange(seq, 'n -> n 1'))

            axial_embeds.append(axial_embed)

        if return_factorized:
            assert not flatten

            # needed for Transfusion
            return axial_embeds

        return self.combine_factorized(axial_embeds, flatten = flatten)
