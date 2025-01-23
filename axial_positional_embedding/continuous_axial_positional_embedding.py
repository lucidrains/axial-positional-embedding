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

    @property
    def dtype(self):
        return next(self.mlps.parameters()).dtype

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

    def maybe_derive_outer_dim(
        self,
        max_seq_len,
        axial_dims: Tensor | Size | tuple[int, ...]
    ):
        ndims = self.num_axial_dims
        assert len(axial_dims) in (ndims, ndims - 1)

        if len(axial_dims) == ndims:
            return axial_dims

        stride = reduce(mul, (*axial_dims,))

        outer_dim = ceil(max_seq_len / stride)
        return (outer_dim, *axial_dims)

    def forward_with_seq_len(
        self,
        seq_len: int,
        axial_dims: Tensor | Size | tuple[int, ...] = (),
        *,
        factorized: list[Tensor] | None = None,
        return_factorized = False
    ):
        if not exists(factorized):
            axial_dims = self.maybe_derive_outer_dim(seq_len, axial_dims)
            factorized = self.forward(axial_dims, return_factorized = True)

        axial_embeds = self.combine_factorized(factorized, flatten = True)

        axial_embeds = axial_embeds[:seq_len]

        if not return_factorized:
            return axial_embeds

        return axial_embeds, factorized

    def forward_with_pos(
        self,
        pos: Tensor,
        axial_dims: Tensor | Size | tuple[int, ...] = (),
    ):
        assert pos.dtype in (torch.int, torch.long)

        max_pos = pos.amax().item() + 1
        axial_dims = self.maybe_derive_outer_dim(max_pos, axial_dims)
        indices = torch.unravel_index(pos, axial_dims)

        axial_embed = 0.

        for mlp, axial_index in zip(self.mlps, indices):
            axial_index = rearrange(axial_index, '... -> ... 1')
            axial_embed = axial_embed + mlp(axial_index.to(self.dtype))

        return axial_embed

    def forward(
        self,
        axial_dims: Tensor | Size | tuple[int, ...],
        return_factorized = False,   # whether to return list[Tensor] of factorized axial positional embeddings
        flatten = False,             # whether to flatten axial dims
    ):
        axial_embeds = []

        for mlp, axial_dim in zip(self.mlps, axial_dims):
            seq = torch.arange(axial_dim, device = self.device, dtype = self.dtype)
            axial_embed = mlp(rearrange(seq, 'n -> n 1'))

            axial_embeds.append(axial_embed)

        if return_factorized:
            assert not flatten

            # needed for Transfusion
            return axial_embeds

        return self.combine_factorized(axial_embeds, flatten = flatten)
