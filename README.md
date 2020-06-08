# Axial Positional Embedding

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
    max_seq_len = 4096,
    axial_shape = (64, 64)          # axial shape must multiply up to the max_seq_len (64 * 64 = 4096)
)

tokens = torch.randn(1, 1024, 512)  # assume are tokens
tokens = pos_emb(tokens) + tokens   # add positional embedding to token embeddings
```

## Citations

```bibtex
@misc{ho2019axial,
    title = {Axial Attention in Multidimensional Transformers},
    author = {Jonathan Ho and Nal Kalchbrenner and Dirk Weissenborn and Tim Salimans},
    year = {2019},
    archivePrefix = {arXiv}
}
```
