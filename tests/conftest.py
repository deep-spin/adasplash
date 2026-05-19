import os
import sys
from pathlib import Path

import pytest


LOCAL_TRITON = Path(__file__).resolve().parents[1] / "triton" / "python"

if LOCAL_TRITON.exists():
    sys.path.insert(0, str(LOCAL_TRITON))

os.environ.setdefault("ADASPLASH_TEST_FAST_AUTOTUNE", "1")

import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="CUDA is required for Triton kernel tests")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
