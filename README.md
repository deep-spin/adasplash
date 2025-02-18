# AdaSplash: Adaptive Sparse Flash Attention

[![PyPI version](https://badge.fury.io/py/adasplash.svg)](https://badge.fury.io/py/adasplash)

AdaSplash, aka flash entmax attention, is an efficient adaptive sparse attention mechanism implemented in Triton.

Paper: https://arxiv.org/abs/2502.12082

## Features
- **Optimized Entmax (triton_entmax):** A Triton-accelerated implementation of Entmax, using Halley's method with bisection instead of pure bisection.
- **AdaSplash with block masking (adasplash):** Uses adaptive sparsity to improve efficiency, requiring extra memory for storing a binary mask.
- **AdaSplash without block masking (adasplash_no_block_mask):** A more memory-efficient variant that maintains efficiency through tiling and fused operations.
- **Fully compatible with PyTorch** and optimized for GPU execution.

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

## Testing
To ensure the library works as expected, install the development dependencies and run tests:

```bash
pip install -r requirements-dev.txt
pytest
```

## Citation
If you use AdaSplash in your research, please cite:

```
@article{goncalves2025adasplash,
  title={AdaSplash: Adaptive Sparse Flash Attention},
  author={Nuno Gonçalves and Marcos Treviso and André F. T. Martins},
  journal={arXiv preprint arXiv:2502.12082},
  url={https://arxiv.org/abs/2502.12082},
  year={2025}
}
```

## License
AdaSplash is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

