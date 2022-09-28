import pytest
import jax.numpy as jnp
import numpy as np
import torch

from src.tests.hippo_tests.hippo_operator import (  # fixtures for respective operators
    hippo_legs,
    hippo_legt,
    hippo_lmu,
    hippo_lagt,
    hippo_fru,
    hippo_fout,
    hippo_fourd,
)
from src.tests.hippo_tests.hippo_operator import (  # fixtures for respective operators made by Albert Gu
    gu_hippo_legs,
    gu_hippo_legt,
    gu_hippo_lmu,
    gu_hippo_lagt,
    gu_hippo_fru,
    gu_hippo_fout,
    gu_hippo_fourd,
)
from src.tests.hippo_tests.trans_matrices import (  # transition matrices A and B from respective polynomials
    legs_matrices,
    legt_matrices,
    legt_lmu_matrices,
    lagt_matrices,
    fru_matrices,
    fout_matrices,
    fourd_matrices,
)
from src.tests.hippo_tests.trans_matrices import (  # transition matrices A and B from respective polynomials made by Albert Gu
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
    A, B = legt_matrices
    gu_A, gu_B = gu_legt_matrices
    assert jnp.allclose(A, gu_A)
    assert jnp.allclose(B, gu_B)


def test_lmu_matrices(legt_lmu_matrices, gu_legt_lmu_matrices):
    A, B = legt_lmu_matrices
    gu_A, gu_B = gu_legt_lmu_matrices
    assert jnp.allclose(A, gu_A)
    assert jnp.allclose(B, gu_B)


def test_lagt_matrices(lagt_matrices, gu_lagt_matrices):
    A, B = lagt_matrices
    gu_A, gu_B = gu_lagt_matrices
    assert jnp.allclose(A, gu_A)
    assert jnp.allclose(B, gu_B)


def test_legs_matrices(legs_matrices, gu_legs_matrices):
    A, B = legs_matrices
    gu_A, gu_B = gu_legs_matrices
    assert jnp.allclose(A, gu_A)
    assert jnp.allclose(B, gu_B)


def test_fru_matrices(fru_matrices, gu_fru_matrices):
    A, B = fru_matrices
    gu_A, gu_B = gu_fru_matrices
    assert jnp.allclose(A, gu_A)
    assert jnp.allclose(B, gu_B)


def test_fout_matrices(fout_matrices, gu_fout_matrices):
    A, B = fout_matrices
    gu_A, gu_B = gu_fout_matrices
    assert jnp.allclose(A, gu_A)
    assert jnp.allclose(B, gu_B)


def test_fourd_matrices(fourd_matrices, gu_fourd_matrices):
    A, B = fourd_matrices
    gu_A, gu_B = gu_fourd_matrices
    assert jnp.allclose(A, gu_A)
    assert jnp.allclose(B, gu_B)


# --- Test HiPPO Operators --- #


def test_hippo_legt_operator(hippo_legt, gu_hippo_legt, random_input, legt_key):
    LTI_bool = True
    c_k = []
    i = 0
    params = hippo_legt.init(legt_key, f=random_input, init_state=c_k, t_step=i)
    c_k_list = hippo_legt.apply(params, f=random_input, t_step=(random_input.shape[0]))
    x = torch.tensor(random_input, dtype=torch.float32)
    for i, c_k in enumerate(c_k_list):
        GU_c_k = gu_hippo_legt(x)
        idx = i - 1
        g_c_k = GU_c_k[0][idx]
        gu = jnp.expand_dims(g_c_k, -1)
        assert jnp.allclose(c_k, gu)


def test_hippo_lmu_operator(hippo_lmu, gu_hippo_lmu, random_input, lmu_key):
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
    # assert jnp.allclose(hippo_lmu.A, gu_hippo_lmu.A)
    # assert jnp.allclose(hippo_lmu.B, gu_hippo_lmu.B)


def test_hippo_lagt_operator(hippo_lagt, gu_hippo_lagt, random_input, lagt_key):
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
    # assert jnp.allclose(hippo_lagt.A, gu_hippo_lagt.A)
    # assert jnp.allclose(hippo_lagt.B, gu_hippo_lagt.B)


def test_hippo_legs_operator(hippo_legs, gu_hippo_legs, random_input, legs_key):
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
    # assert jnp.allclose(hippo_legs.A, gu_hippo_legs.A)
    # assert jnp.allclose(hippo_legs.B, gu_hippo_legs.B)


def test_hippo_fru_operator(hippo_fru, gu_hippo_fru, random_input, fru_key):
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
    # assert jnp.allclose(hippo_fru.A, gu_hippo_fru.A)
    # assert jnp.allclose(hippo_fru.B, gu_hippo_fru.B)


def test_hippo_fout_operator(hippo_fout, gu_hippo_fout, random_input, fout_key):
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
    # assert jnp.allclose(hippo_fout.A, gu_hippo_fout.A)
    # assert jnp.allclose(hippo_fout.B, gu_hippo_fout.B)


def test_hippo_fourd_operator(hippo_fourd, gu_hippo_fourd, random_input, fourd_key):
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
    # assert jnp.allclose(hippo_fourd.A, gu_hippo_fourd.A)
    # assert jnp.allclose(hippo_fourd.B, gu_hippo_fourd.B)
