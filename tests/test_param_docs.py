"""Smoke tests for vector search parameter models.

No running Qdrant server is required.  These tests verify:
- Safe default values for SearchParams and related models.
- That the new Memory-enum migration paths are accepted by the validators.
- That all three quantization wrapper classes now carry a class docstring
  (regression guard for the documentation fix).
"""
import pytest
from qdrant_client.http.models import (
    BinaryQuantization,
    BinaryQuantizationConfig,
    CompressionRatio,
    HnswConfigDiff,
    Memory,
    ProductQuantization,
    ProductQuantizationConfig,
    QuantizationSearchParams,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    SearchParams,
)


# ---------------------------------------------------------------------------
# SearchParams
# ---------------------------------------------------------------------------


class TestSearchParams:
    def test_all_defaults_are_safe(self):
        p = SearchParams()
        assert p.hnsw_ef is None
        assert p.exact is False
        assert p.indexed_only is False
        assert p.quantization is None

    def test_hnsw_ef_accepted(self):
        assert SearchParams(hnsw_ef=128).hnsw_ef == 128

    def test_exact_true(self):
        assert SearchParams(exact=True).exact is True

    def test_indexed_only_true(self):
        assert SearchParams(indexed_only=True).indexed_only is True

    def test_quantization_rescore_true(self):
        q = QuantizationSearchParams(rescore=True)
        p = SearchParams(quantization=q)
        assert p.quantization.rescore is True

    def test_quantization_rescore_false_with_oversampling(self):
        q = QuantizationSearchParams(rescore=False, oversampling=2.4)
        assert q.oversampling == 2.4


# ---------------------------------------------------------------------------
# HnswConfigDiff — Memory migration path for on_disk=True
# ---------------------------------------------------------------------------


class TestHnswConfigDiff:
    def test_all_optional(self):
        cfg = HnswConfigDiff()
        assert cfg.m is None
        assert cfg.ef_construct is None
        assert cfg.on_disk is None
        assert cfg.memory is None

    def test_memory_cold_accepted(self):
        # Migration path: on_disk=True  →  memory=Memory.cold
        cfg = HnswConfigDiff(memory=Memory.cold)
        assert cfg.memory == Memory.cold

    def test_memory_cached_accepted(self):
        cfg = HnswConfigDiff(memory=Memory.cached)
        assert cfg.memory == Memory.cached


# ---------------------------------------------------------------------------
# ScalarQuantization — always_ram migration path
# ---------------------------------------------------------------------------


class TestScalarQuantization:
    def test_int8_type(self):
        q = ScalarQuantization(scalar=ScalarQuantizationConfig(type=ScalarType.INT8))
        assert q.scalar.type == ScalarType.INT8

    def test_quantile_range(self):
        cfg = ScalarQuantizationConfig(type=ScalarType.INT8, quantile=0.99)
        assert cfg.quantile == 0.99

    def test_memory_ram_accepted(self):
        # Migration path: always_ram=True  →  memory=Memory.ram
        cfg = ScalarQuantizationConfig(type=ScalarType.INT8, memory=Memory.ram)
        assert cfg.memory == Memory.ram

    def test_has_docstring(self):
        assert ScalarQuantization.__doc__ is not None
        assert len(ScalarQuantization.__doc__) > 20


# ---------------------------------------------------------------------------
# ProductQuantization — compression field + always_ram migration path
# ---------------------------------------------------------------------------


class TestProductQuantization:
    def test_compression_x4(self):
        q = ProductQuantization(product=ProductQuantizationConfig(compression=CompressionRatio.x4))
        assert q.product.compression == CompressionRatio.x4

    def test_compression_x64(self):
        q = ProductQuantization(product=ProductQuantizationConfig(compression=CompressionRatio.x64))
        assert q.product.compression == CompressionRatio.x64

    def test_memory_ram_accepted(self):
        # Migration path: always_ram=True  →  memory=Memory.ram
        cfg = ProductQuantizationConfig(compression=CompressionRatio.x4, memory=Memory.ram)
        assert cfg.memory == Memory.ram

    def test_has_docstring(self):
        assert ProductQuantization.__doc__ is not None
        assert len(ProductQuantization.__doc__) > 20


# ---------------------------------------------------------------------------
# BinaryQuantization
# ---------------------------------------------------------------------------


class TestBinaryQuantization:
    def test_instantiation(self):
        q = BinaryQuantization(binary=BinaryQuantizationConfig())
        assert q.binary is not None

    def test_memory_ram_accepted(self):
        cfg = BinaryQuantizationConfig(memory=Memory.ram)
        assert cfg.memory == Memory.ram

    def test_has_docstring(self):
        # Regression guard: class must have a docstring after the documentation fix.
        assert BinaryQuantization.__doc__ is not None
        assert len(BinaryQuantization.__doc__) > 20
