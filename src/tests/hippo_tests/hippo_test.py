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
from src.tests.hippo_tests.hippo_utils import (
    key_generator,
    legt_key,
    lmu_key,
    lagt_key,
    legs_key,
    fru_key,
    fout_key,
    fourd_key,
)
from src.tests.hippo_tests.hippo_utils import (
    random_input,
    ones_input,
    zeros_input,
    desc_input,
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
    print(f"\nA:\n{A}\n")
    print(f"B:\n{B}\n")
    print(f"Gu's A:\n{gu_A}\n")
    print(f"Gu's B:\n{gu_B}\n")
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


def test_hippo_legs_operator(hippo_legs, gu_hippo_legs, random_input, legs_key):
    i = 0
    x_tensor = torch.tensor(random_input, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array
    params = hippo_legs.init(legs_key, f=x_jnp, t_step=i)
    c_k_list = hippo_legs.apply(params, f=x_jnp, t_step=(x_jnp.shape[0]))
    GU_c_k = gu_hippo_legs(x_tensor)
    for i, c_k in enumerate(c_k_list):
        g_c_k = GU_c_k[i][0]
        gu = torch.unsqueeze(g_c_k, -1)
        gu_c = jnp.asarray(gu, dtype=jnp.float32)  # convert torch array to jax array
        assert jnp.allclose(c_k, gu_c, rtol=1e-04, atol=1e-06)


def test_hippo_legt_operator(hippo_legt, gu_hippo_legt, random_input, legt_key):
    i = 0
    x_tensor = torch.tensor(random_input, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array
    params = hippo_legt.init(legt_key, f=x_jnp, t_step=i)
    c_k_list = hippo_legt.apply(params, f=x_jnp, t_step=(x_jnp.shape[0]))
    GU_c_k = gu_hippo_legt(x_tensor)
    for i, c_k in enumerate(c_k_list):
        g_c_k = GU_c_k[i][0]
        gu = torch.unsqueeze(g_c_k, -1)
        gu_c = jnp.asarray(gu, dtype=jnp.float32)  # convert torch array to jax array
        assert jnp.allclose(c_k, gu_c, rtol=1e-04, atol=1e-06)


def test_hippo_lmu_operator(hippo_lmu, gu_hippo_lmu, random_input, lmu_key):
    i = 0
    x_tensor = torch.tensor(random_input, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array
    params = hippo_lmu.init(lmu_key, f=x_jnp, t_step=i)
    c_k_list = hippo_lmu.apply(params, f=x_jnp, t_step=(x_jnp.shape[0]))
    GU_c_k = gu_hippo_lmu(x_tensor)
    for i, c_k in enumerate(c_k_list):
        g_c_k = GU_c_k[i][0]
        gu = torch.unsqueeze(g_c_k, -1)
        gu_c = jnp.asarray(gu, dtype=jnp.float32)  # convert torch array to jax array
        assert jnp.allclose(c_k, gu_c, rtol=1e-04, atol=1e-06)


def test_hippo_lagt_operator(hippo_lagt, gu_hippo_lagt, random_input, lagt_key):
    i = 0
    x_tensor = torch.tensor(random_input, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array
    params = hippo_lagt.init(lagt_key, f=x_jnp, t_step=i)
    c_k_list = hippo_lagt.apply(params, f=x_jnp, t_step=(x_jnp.shape[0]))
    GU_c_k = gu_hippo_lagt(x_tensor)
    for i, c_k in enumerate(c_k_list):
        g_c_k = GU_c_k[i][0]
        gu = torch.unsqueeze(g_c_k, -1)
        gu_c = jnp.asarray(gu, dtype=jnp.float32)  # convert torch array to jax array
        assert jnp.allclose(c_k, gu_c, rtol=1e-04, atol=1e-06)


def test_hippo_fru_operator(hippo_fru, gu_hippo_fru, random_input, fru_key):
    i = 0
    x_tensor = torch.tensor(random_input, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array
    params = hippo_fru.init(fru_key, f=x_jnp, t_step=i)
    c_k_list = hippo_fru.apply(params, f=x_jnp, t_step=(x_jnp.shape[0]))
    GU_c_k = gu_hippo_fru(x_tensor)
    for i, c_k in enumerate(c_k_list):
        g_c_k = GU_c_k[i][0]
        gu = torch.unsqueeze(g_c_k, -1)
        gu_c = jnp.asarray(gu, dtype=jnp.float32)  # convert torch array to jax array
        assert jnp.allclose(c_k, gu_c, rtol=1e-04, atol=1e-06)


def test_hippo_fout_operator(hippo_fout, gu_hippo_fout, random_input, fout_key):
    i = 0
    x_tensor = torch.tensor(random_input, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array
    params = hippo_fout.init(fout_key, f=x_jnp, t_step=i)
    c_k_list = hippo_fout.apply(params, f=x_jnp, t_step=(x_jnp.shape[0]))
    GU_c_k = gu_hippo_fout(x_tensor)
    for i, c_k in enumerate(c_k_list):
        g_c_k = GU_c_k[i][0]
        gu = torch.unsqueeze(g_c_k, -1)
        gu_c = jnp.asarray(gu, dtype=jnp.float32)  # convert torch array to jax array
        assert jnp.allclose(c_k, gu_c, rtol=1e-04, atol=1e-06)


def test_hippo_fourd_operator(hippo_fourd, gu_hippo_fourd, random_input, fourd_key):
    i = 0
    x_tensor = torch.tensor(random_input, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array
    params = hippo_fourd.init(fourd_key, f=x_jnp, t_step=i)
    c_k_list = hippo_fourd.apply(params, f=x_jnp, t_step=(x_jnp.shape[0]))
    GU_c_k = gu_hippo_fourd(x_tensor)
    for i, c_k in enumerate(c_k_list):
        g_c_k = GU_c_k[i][0]
        gu = torch.unsqueeze(g_c_k, -1)
        gu_c = jnp.asarray(gu, dtype=jnp.float32)  # convert torch array to jax array
        assert jnp.allclose(c_k, gu_c, rtol=1e-04, atol=1e-06)
