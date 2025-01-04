## Axial Positional Embedding

[![PyPI version](https://badge.fury.io/py/axial-positional-embedding.svg)](https://badge.fury.io/py/axial-positional-embedding)

A type of positional embedding that is very effective when working with attention networks on multi-dimensional data, or for language models in general.

## Install

```bash
$ pip install axial-positional-embedding
```

## Usage

```python
import torch
from axial_positional_embedding import AxialPositionalEmbedding

pos_emb = AxialPositionalEmbedding(
    dim = 512,
    axial_shape = (64, 64),          # axial shape will multiply up to the maximum sequence length allowed (64 * 64 = 4096)
    axial_dims = (256, 256)          # if not specified, dimensions will default to 'dim' for all axials and summed at the end. if specified, each axial will have the specified dimension and be concatted together. the concatted dimensions needs to sum up to the `dim` (256 + 256 = 512)
)

tokens = torch.randn(1, 1024, 512)  # assume are tokens
tokens = pos_emb(tokens) + tokens   # add positional embedding to token embeddings
```

A continuous version with better extrapolation ability (each axis parameterized by a 2 layer MLP)

```python
import torch
from axial_positional_embedding import ContinuousAxialPositionalEmbedding

pos_emb = ContinuousAxialPositionalEmbedding(
    dim = 512,
    num_axial_dims = 3
)

tokens = torch.randn(1, 8, 16, 32, 512) # say a video with 8 frames, 16 x 32 image dimension

axial_pos_emb = pos_emb((8, 16, 32)) # pass in the size from above

tokens = axial_pos_emb + tokens   # add positional embedding to token embeddings
```

## Citations

```bibtex
@inproceedings{kitaev2020reformer,
    title       = {Reformer: The Efficient Transformer},
    author      = {Nikita Kitaev and Lukasz Kaiser and Anselm Levskaya},
    booktitle   = {International Conference on Learning Representations},
    year        = {2020},
    url         = {https://openreview.net/forum?id=rkgNKkHtvB}
}
```

```bibtex
@misc{ho2019axial,
    title   = {Axial Attention in Multidimensional Transformers},
    author  = {Jonathan Ho and Nal Kalchbrenner and Dirk Weissenborn and Tim Salimans},
    year    = {2019},
    archivePrefix = {arXiv}
}
```
