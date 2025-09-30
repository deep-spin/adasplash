# AdaSplash: Adaptive Sparse Flash Attention

[![PyPI version](https://badge.fury.io/py/adasplash.svg)](https://badge.fury.io/py/adasplash)

AdaSplash, aka flash entmax attention, is an efficient adaptive sparse attention mechanism implemented in Triton.
Check out our paper: https://arxiv.org/abs/2502.12082.

## Installation

You can install AdaSplash via pip:

```bash
pip install adasplash
```

Alternatively, install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/deep-spin/adasplash.git
```

## Usage

AdaSplash provides three main functions, all available via `from adasplash import ...`:

### **Triton Entmax** (Optimized Entmax Activation)
```python
from adasplash import triton_entmax
import torch

x = torch.randn(128, 256).cuda()
y = triton_entmax(x, alpha=1.5, n_iter=10, fast_math=True)
```
- Uses **Halley's method + bisection** instead of pure bisection.
- Faster and more efficient than traditional Entmax implementations.

### **AdaSplash with Block Masking**
```python
from adasplash import adasplash

q = torch.randn(1, 8, 128, 64, device="cuda")
k = torch.randn(1, 8, 128, 64, device="cuda")
v = torch.randn(1, 8, 128, 64, device="cuda")

output = adasplash(q, k, v, alpha=1.5, niter=10, is_causal=True, varlen=None)
```
- Leverages **adaptive sparsity** for efficiency in both forward and backward passes.
- Requires **O(Tr × Tc) bits** of extra memory for storing a binary mask per block.

### **AdaSplash without Block Masking**
```python
from adasplash import adasplash_no_block_mask

output = adasplash_no_block_mask(q, k, v, alpha=1.5, niter=10, is_causal=True, varlen=None)
```
- Does **not** use block masking but still benefits from **tiling and fused ops** for efficiency.
- Requires **less memory** than the block-masked version.

### Key Features

Variable Length Sequences:
```python
varlen = torch.tensor([34, 128], device='cuda')  # Actual sequence lengths
output = adasplash(q, k, v, varlen=varlen)
```

Adaptive Sparsity Control:
```python
# Control sparsity via alpha parameter
output = adasplash(q, k, v, alpha=1.333)  # More dense
output = adasplash(q, k, v, alpha=2.0)  # More sparse
```

Causal and Non-causal Masking:
```python
output = adasplash(q, k, v, is_causal=True)  # Causal masking
output = adasplash(q, k, v, is_causal=False)  # Non-causal masking
```

## Benchmarks

### Efficiency
![Benchmark](benchmark.png)

### Single-vector retrieval

Check the [Sparse ModernBERT repo](https://github.com/deep-spin/SparseModernBERT).


## Testing
To ensure the library works as expected, install the development dependencies and run tests:

```bash
pip install -r requirements-dev.txt
pytest
```

## Citation
If you use AdaSplash in your research, please cite:

```
@inproceedings{goncalves2025adasplash,
    title={AdaSplash: Adaptive Sparse Flash Attention},
    author={Nuno Gon{\c{c}}alves and Marcos V Treviso and Andre Martins},
    booktitle={Forty-second International Conference on Machine Learning},
    year={2025},
    url={https://openreview.net/forum?id=OWIPDWhUcO}
}
```


## Acknowledgements

> We thank Vlad Niculae for his insightful and constructive comments throughout this work. We also thank the SARDINE Lab members for reviewing this paper and providing helpful feedback. This work was supported by the Portuguese Recovery and Resilience Plan through project C645008882-00000055 (Center for ResponsibleAI), by the EU’s Horizon Europe Research and Innovation Actions (UTTER, contract 101070631), by the project DECOLLAGE (ERC-2022-CoG 101088763), and by FCT/MECI through national funds and when applicable co-funded EU funds under UID/50008: Instituto de Telecomunicações.


## License
AdaSplash is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## Acknowledgements

> We thank Vlad Niculae for his insightful and constructive comments throughout this work. We also thank the
SARDINE Lab members for reviewing this paper and providing helpful feedback. This work was supported by the Portuguese Recovery and Resilience Plan through project C645008882-00000055 (Center for ResponsibleAI), by the EU’s Horizon Europe Research and Innovation Actions (UTTER, contract 101070631), by the project DECOLLAGE (ERC-2022-CoG 101088763), and by FCT/MECI through national funds and when applicable co-funded EU funds under UID/50008: Instituto de Telecomunicações.
