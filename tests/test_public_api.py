import inspect
from pathlib import Path

import adasplash


def test_public_api_exports_are_lazy_and_versioned():
    assert adasplash.__version__ == "0.2.0"
    for name in [
        "adasplash",
        "adasplash_v1",
        "adasplash_v2",
        "adasplash_no_block_mask",
        "triton_entmax",
        "triton_entmax_v1",
        "triton_entmax_v2",
        "triton_sparsemax",
        "triton_entmax15",
        "entmax_attention",
    ]:
        assert name in adasplash.__all__
        assert callable(getattr(adasplash, name))


def test_dispatcher_signatures_are_stable():
    assert list(inspect.signature(adasplash.adasplash).parameters) == [
        "q",
        "k",
        "v",
        "alpha",
        "is_causal",
        "varlen",
        "niter",
    ]
    assert list(inspect.signature(adasplash.adasplash_v1).parameters) == [
        "q",
        "k",
        "v",
        "alpha",
        "is_causal",
        "varlen",
        "niter",
    ]
    assert list(inspect.signature(adasplash.adasplash_v2).parameters) == ["q", "k", "v", "niter", "varlen"]
    assert list(inspect.signature(adasplash.triton_entmax).parameters) == [
        "x",
        "alpha",
        "n_iter",
        "use_histogram",
        "fast_math",
    ]
    assert list(inspect.signature(adasplash.entmax_attention).parameters) == [
        "q",
        "k",
        "v",
        "alpha",
        "varlen",
        "is_causal",
        "padding",
        "niter",
        "alibi_slopes",
    ]


def test_package_source_allowlist():
    package_dir = Path(adasplash.__file__).resolve().parent
    allowed = {
        "__init__.py",
        "attention.py",
        "adasplash_block_mask.py",
        "adasplash_no_block_mask.py",
        "adasplash_v2.py",
        "triton_entmax.py",
        "triton_entmax_v2.py",
    }
    actual = {path.name for path in package_dir.glob("*.py")}
    assert actual == allowed


def test_v2_entmax_contains_histogram_hybrid_solver():
    source = (Path(adasplash.__file__).resolve().parent / "triton_entmax_v2.py").read_text()
    assert "hist_init_tau" in source
    assert "hybrid_update" in source
    assert "use_histogram" in source


def test_dispatcher_is_not_shadowed_by_v2_submodules():
    import adasplash.adasplash_v2  # noqa: F401
    import adasplash.triton_entmax_v2  # noqa: F401

    assert callable(adasplash.adasplash)
    assert callable(adasplash.triton_entmax)
