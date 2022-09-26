import pytest
import jax.numpy as jnp

from hippo_operator import (
    hippo_legs,
    hippo_legt,
    hippo_lmu,
    hippo_lagt,
    hippo_fru,
    hippo_fout,
    hippo_fourd,
)
from hippo_operator import (
    gu_hippo_legs,
    gu_hippo_legt,
    gu_hippo_lmu,
    gu_hippo_lagt,
    gu_hippo_fru,
    gu_hippo_fout,
    gu_hippo_fourd,
)
from trans_matrices import (
    legs_matrices,
    legt_matrices,
    legt_lmu_matrices,
    lagt_matrices,
    fru_matrices,
    fout_matrices,
    fourd_matrices,
)
from trans_matrices import (
    gu_legs_matrices,
    gu_legt_matrices,
    gu_legt_lmu_matrices,
    gu_lagt_matrices,
    gu_fru_matrices,
    gu_fout_matrices,
    gu_fourd_matrices,
)

# --- Test HiPPO Matrices --- #


def test_legt_matrices(legt_matrices, gu_legt_matrices):
    assert jnp.allclose(legt_matrices.A, gu_legt_matrices.A)
    assert jnp.allclose(legt_matrices.B, gu_legt_matrices.B)


def test_lmu_matrices(legt_lmu_matrices, gu_legt_lmu_matrices):
    assert jnp.allclose(legt_lmu_matrices.A, gu_legt_lmu_matrices.A)
    assert jnp.allclose(legt_lmu_matrices.B, gu_legt_lmu_matrices.B)


def test_lagt_matrices(lagt_matrices, gu_lagt_matrices):
    assert jnp.allclose(lagt_matrices.A, gu_lagt_matrices.A)
    assert jnp.allclose(lagt_matrices.B, gu_lagt_matrices.B)


def test_legs_matrices(legs_matrices, gu_legs_matrices):
    assert jnp.allclose(legs_matrices.A, gu_legs_matrices.A)
    assert jnp.allclose(legs_matrices.B, gu_legs_matrices.B)


def test_fru_matrices(fru_matrices, gu_fru_matrices):
    assert jnp.allclose(fru_matrices.A, gu_fru_matrices.A)
    assert jnp.allclose(fru_matrices.B, gu_fru_matrices.B)


def test_fout_matrices(fout_matrices, gu_fout_matrices):
    assert jnp.allclose(fout_matrices.A, gu_fout_matrices.A)
    assert jnp.allclose(fout_matrices.B, gu_fout_matrices.B)


def test_fourd_matrices(fourd_matrices, gu_fourd_matrices):
    assert jnp.allclose(fourd_matrices.A, gu_fourd_matrices.A)
    assert jnp.allclose(fourd_matrices.B, gu_fourd_matrices.B)


# --- Test HiPPO Operators --- #


def test_hippo_legt_operator(hippo_legt, gu_hippo_legt):
    assert hippo_legt.measure == "legt"
    assert hippo_legt.lambda_n == 1.0
    assert hippo_legt.N == hippo_legt.A.shape[0]
    assert hippo_legt.A.shape == (16, 16)
    assert hippo_legt.B.shape == (16,)
    assert hippo_legt.seq_L == 16
    assert hippo_legt.max_length == 16
    assert hippo_legt.step == 1.0 / 16
    assert hippo_legt.GBT_alpha == 0.5
    assert hippo_legt.N == 16
    assert jnp.allclose(hippo_legt.A, gu_hippo_legt.A)
    assert jnp.allclose(hippo_legt.B, gu_hippo_legt.B)


def test_hippo_lmu_operator(hippo_lmu, gu_hippo_lmu):
    assert hippo_lmu.measure == "legt"
    assert hippo_lmu.lambda_n == 2.0
    assert hippo_lmu.N == hippo_lmu.A.shape[0]
    assert hippo_lmu.A.shape == (16, 16)
    assert hippo_lmu.B.shape == (16,)
    assert hippo_lmu.seq_L == 16
    assert hippo_lmu.max_length == 16
    assert hippo_lmu.step == 1.0 / 16
    assert hippo_lmu.GBT_alpha == 0.5
    assert hippo_lmu.N == 16
    assert jnp.allclose(hippo_lmu.A, gu_hippo_lmu.A)
    assert jnp.allclose(hippo_lmu.B, gu_hippo_lmu.B)


def test_hippo_lagt_operator(hippo_lagt, gu_hippo_lagt):
    assert hippo_lagt.measure == "lagt"
    assert hippo_lagt.lambda_n == 1.0
    assert hippo_lagt.N == hippo_lagt.A.shape[0]
    assert hippo_lagt.A.shape == (16, 16)
    assert hippo_lagt.B.shape == (16,)
    assert hippo_lagt.seq_L == 16
    assert hippo_lagt.max_length == 16
    assert hippo_lagt.step == 1.0 / 16
    assert hippo_lagt.GBT_alpha == 0.5
    assert hippo_lagt.N == 16
    assert jnp.allclose(hippo_lagt.A, gu_hippo_lagt.A)
    assert jnp.allclose(hippo_lagt.B, gu_hippo_lagt.B)


def test_hippo_legs_operator(hippo_legs, gu_hippo_legs):
    assert hippo_legs.measure == "lagt"
    assert hippo_legs.lambda_n == 1.0
    assert hippo_legs.N == hippo_legs.A.shape[0]
    assert hippo_legs.A.shape == (16, 16)
    assert hippo_legs.B.shape == (16,)
    assert hippo_legs.seq_L == 16
    assert hippo_legs.max_length == 16
    assert hippo_legs.step == 1.0 / 16
    assert hippo_legs.GBT_alpha == 0.5
    assert hippo_legs.N == 16
    assert jnp.allclose(hippo_legs.A, gu_hippo_legs.A)
    assert jnp.allclose(hippo_legs.B, gu_hippo_legs.B)


def test_hippo_fru_operator(hippo_fru, gu_hippo_fru):
    assert hippo_fru.measure == "fru"
    assert hippo_fru.lambda_n == 1.0
    assert hippo_fru.N == hippo_fru.A.shape[0]
    assert hippo_fru.A.shape == (16, 16)
    assert hippo_fru.B.shape == (16,)
    assert hippo_fru.seq_L == 16
    assert hippo_fru.max_length == 16
    assert hippo_fru.step == 1.0 / 16
    assert hippo_fru.GBT_alpha == 0.5
    assert hippo_fru.N == 16
    assert jnp.allclose(hippo_fru.A, gu_hippo_fru.A)
    assert jnp.allclose(hippo_fru.B, gu_hippo_fru.B)


def test_hippo_fout_operator(hippo_fout, gu_hippo_fout):
    assert hippo_fout.measure == "fout"
    assert hippo_fout.lambda_n == 1.0
    assert hippo_fout.N == hippo_fout.A.shape[0]
    assert hippo_fout.A.shape == (16, 16)
    assert hippo_fout.B.shape == (16,)
    assert hippo_fout.seq_L == 16
    assert hippo_fout.max_length == 16
    assert hippo_fout.step == 1.0 / 16
    assert hippo_fout.GBT_alpha == 0.5
    assert hippo_fout.N == 16
    assert jnp.allclose(hippo_fout.A, gu_hippo_fout.A)
    assert jnp.allclose(hippo_fout.B, gu_hippo_fout.B)


def test_hippo_fourd_operator(hippo_fourd, gu_hippo_fourd):
    assert hippo_fourd.measure == "fourd"
    assert hippo_fourd.lambda_n == 1.0
    assert hippo_fourd.N == hippo_fourd.A.shape[0]
    assert hippo_fourd.A.shape == (16, 16)
    assert hippo_fourd.B.shape == (16,)
    assert hippo_fourd.seq_L == 16
    assert hippo_fourd.max_length == 16
    assert hippo_fourd.step == 1.0 / 16
    assert hippo_fourd.GBT_alpha == 0.5
    assert hippo_fourd.N == 16
    assert jnp.allclose(hippo_fourd.A, gu_hippo_fourd.A)
    assert jnp.allclose(hippo_fourd.B, gu_hippo_fourd.B)
