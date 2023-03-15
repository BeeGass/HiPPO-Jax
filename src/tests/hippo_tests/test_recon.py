import einops
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

# implementation of HiPPO RECONSTRUCTIONs
# Gu's implementation of HiPPO RECONSTRUCTIONs
from src.tests.hippo_tests.hippo_operator import (
    hr_hippo_lsi_legs_be,
    hr_hippo_lsi_legs_bi,
    hr_hippo_lsi_legs_fe,
    hr_hippo_lsi_legs_zoh,
    hr_hippo_lti_foud_be,
    hr_hippo_lti_foud_bi,
    hr_hippo_lti_foud_fe,
    hr_hippo_lti_foud_zoh,
    hr_hippo_lti_fout_be,
    hr_hippo_lti_fout_bi,
    hr_hippo_lti_fout_fe,
    hr_hippo_lti_fout_zoh,
    hr_hippo_lti_fru_be,
    hr_hippo_lti_fru_bi,
    hr_hippo_lti_fru_fe,
    hr_hippo_lti_fru_zoh,
    hr_hippo_lti_lagt_be,
    hr_hippo_lti_lagt_bi,
    hr_hippo_lti_lagt_fe,
    hr_hippo_lti_lagt_zoh,
    hr_hippo_lti_legs_be,
    hr_hippo_lti_legs_bi,
    hr_hippo_lti_legs_fe,
    hr_hippo_lti_legs_zoh,
    hr_hippo_lti_legt_be,
    hr_hippo_lti_legt_bi,
    hr_hippo_lti_legt_fe,
    hr_hippo_lti_legt_zoh,
    hr_hippo_lti_lmu_be,
    hr_hippo_lti_lmu_bi,
    hr_hippo_lti_lmu_fe,
    hr_hippo_lti_lmu_zoh,
    hippo_lsi_legs_be,
    hippo_lsi_legs_bi,
    hippo_lsi_legs_fe,
    hippo_lsi_legs_zoh,
    hippo_lti_foud_be,
    hippo_lti_foud_bi,
    hippo_lti_foud_fe,
    hippo_lti_foud_zoh,
    hippo_lti_fout_be,
    hippo_lti_fout_bi,
    hippo_lti_fout_fe,
    hippo_lti_fout_zoh,
    hippo_lti_fru_be,
    hippo_lti_fru_bi,
    hippo_lti_fru_fe,
    hippo_lti_fru_zoh,
    hippo_lti_lagt_be,
    hippo_lti_lagt_bi,
    hippo_lti_lagt_fe,
    hippo_lti_lagt_zoh,
    hippo_lti_legs_be,
    hippo_lti_legs_bi,
    hippo_lti_legs_fe,
    hippo_lti_legs_zoh,
    hippo_lti_legt_be,
    hippo_lti_legt_bi,
    hippo_lti_legt_fe,
    hippo_lti_legt_zoh,
    hippo_lti_lmu_be,
    hippo_lti_lmu_bi,
    hippo_lti_lmu_fe,
    hippo_lti_lmu_zoh,
)
from src.tests.hippo_tests.hippo_utils import (
    foud_key,
    fout_key,
    fru_key,
    key_generator,
    lagt_key,
    legs_key,
    legt_key,
    lmu_key,
    random_1_input,
    random_16_input,
    random_32_input,
    random_64_input,
)

# transition dplr matrices from respective polynomials made by Albert Gu
# transition dplr matrices from respective polynomials
# transition nplr matrices from respective polynomials made by Albert Gu
# transition nplr matrices from respective polynomials
# transition matrices A and B from respective polynomials made by Albert Gu
# transition matrices A and B from respective polynomials
from src.tests.hippo_tests.trans_matrices import (
    dplr_foud,
    dplr_fout,
    dplr_fru,
    dplr_lagt,
    dplr_legs,
    dplr_legt,
    dplr_lmu,
    foud_matrices,
    fout_matrices,
    fru_matrices,
    hr_dplr_foud,
    hr_dplr_fout,
    hr_dplr_fru,
    hr_dplr_lagt,
    hr_dplr_legs,
    hr_dplr_legt,
    hr_dplr_lmu,
    hr_foud_matrices,
    hr_fout_matrices,
    hr_fru_matrices,
    hr_lagt_matrices,
    hr_legs_matrices,
    hr_legt_lmu_matrices,
    hr_legt_matrices,
    hr_nplr_foud,
    hr_nplr_fout,
    hr_nplr_fru,
    hr_nplr_lagt,
    hr_nplr_legs,
    hr_nplr_legt,
    hr_nplr_lmu,
    lagt_matrices,
    legs_matrices,
    legt_lmu_matrices,
    legt_matrices,
    nplr_foud,
    nplr_fout,
    nplr_fru,
    nplr_lagt,
    nplr_legs,
    nplr_legt,
    nplr_lmu,
)

# ----------------------------------------------------------------
# --------------------- Test HiPPO RECONSTRUCTIONs ---------------------
# ----------------------------------------------------------------

# --------------------
# -- Forward Euler --
# --------------------

# ----------
# -- legs --
# ----------


def test_hippo_legs_lti_fe_reconstruction(
    hippo_lti_legs_fe, hr_hippo_lti_legs_fe, random_16_input, legs_key
):
    print("\nHiPPO RECONSTRUCTION LEGS (LTI, FE)")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    assert x_jnp.shape == (16, 3000)  # batch_size, seq_len

    x_jnp = einops.rearrange(
        x_jnp, "batch seq_len -> batch seq_len 1"
    )  # add input_size dimension

    params = hippo_lti_legs_fe.init(legs_key, f=x_jnp)
    c, y = hippo_lti_legs_fe.apply(params, f=x_jnp)
    y = einops.rearrange(y, "batch seq_len -> batch 1 seq_len")

    x_tensor = einops.rearrange(x_tensor, "batch seq_len -> seq_len batch")
    _, hr_c = hr_hippo_lti_legs_fe(x_tensor, fast=False)
    hr_c = einops.rearrange(
        hr_c, "batch N -> batch 1 N"
    )  # add input_size dimension for comparison
    hr_y = hr_hippo_lti_legs_fe.reconstruct(hr_c)
    print(f"hr_y shape: {hr_y.shape}")
    hr_y = einops.rearrange(hr_y, "batch seq_len -> batch 1 seq_len")
    hr_c = jnp.asarray(hr_c, dtype=jnp.float32)  # convert torch array to jax array
    hr_y = jnp.asarray(hr_y, dtype=jnp.float32)  # convert torch array to jax array

    assert hr_c.shape == c.shape
    assert hr_y.shape == y.shape

    assert c.shape == (16, 1, 50)  # batch_size, input_size, N
    assert y.shape == (16, 1, 3000)  # batch_size, input_size, seq_len

    co_flag = True
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            co_check = jnp.allclose(c[i, j, :], hr_c[i, j, :], rtol=1e-03, atol=1e-03)
            if not co_check:
                co_flag = False
            assert jnp.allclose(y[i, j, :], hr_y[i, j, :], rtol=1e-03, atol=1e-03)
    print(f"coefficients tests: {co_flag}")


def test_hippo_legs_lsi_fe_reconstruction(
    hippo_lsi_legs_fe, hr_hippo_lsi_legs_fe, random_16_input, legs_key
):
    print("\nHiPPO RECONSTRUCTION LEGS (LSI, FE)")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    assert x_jnp.shape == (16, 3000)  # batch_size, seq_len

    x_jnp = einops.rearrange(
        x_jnp, "batch seq_len -> batch seq_len 1"
    )  # add input_size dimension

    params = hippo_lsi_legs_fe.init(legs_key, f=x_jnp)
    c, y = hippo_lsi_legs_fe.apply(params, f=x_jnp)

    y = einops.rearrange(
        y, "batch seq_len seq_len2 input_len -> batch seq_len input_len seq_len2"
    )

    x_tensor = einops.rearrange(x_tensor, "batch seq_len -> seq_len batch")
    hr_cs, _ = hr_hippo_lsi_legs_fe(x_tensor, fast=False)
    hr_c = einops.rearrange(
        hr_cs, "seq_len batch N -> batch seq_len 1 N"
    )  # add input_size and swap batch and seq_len dimension for comparison
    hr_y = hr_hippo_lsi_legs_fe.reconstruct(hr_c)
    print(f"hr_y shape: {hr_y.shape}")
    hr_y = einops.rearrange(
        hr_y, "seq_len batch seq_len2 input_len -> batch seq_len input_len seq_len2"
    )
    hr_c = jnp.asarray(hr_c, dtype=jnp.float32)  # convert torch array to jax array
    hr_y = jnp.asarray(hr_y, dtype=jnp.float32)  # convert torch array to jax array

    assert hr_c.shape == c.shape
    assert hr_y.shape == y.shape

    assert c.shape == (16, 3000, 1, 50)  # batch_size, seq_len, input_size, N
    assert y.shape == (16, 3000, 1, 3000)  # batch_size, seq_len, input_size, seq_len

    co_flag = True
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            co_check = jnp.allclose(
                c[i, j, :, :], hr_c[i, j, :, :], rtol=1e-03, atol=1e-03
            )
            if not co_check:
                co_flag = False
            assert jnp.allclose(y[i, j, :, :], hr_y[i, j, :, :], rtol=1e-03, atol=1e-03)
    print(f"coefficients tests: {co_flag}")


# ----------
# -- legt --
# ----------


def test_hippo_legt_lti_fe_reconstruction(
    hippo_lti_legt_fe, hr_hippo_lti_legt_fe, random_16_input, legt_key
):
    print("\nHiPPO RECONSTRUCTION LEGT (LTI, FE)")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    assert x_jnp.shape == (16, 3000)  # batch_size, seq_len

    x_jnp = einops.rearrange(
        x_jnp, "batch seq_len -> batch seq_len 1"
    )  # add input_size dimension

    params = hippo_lti_legt_fe.init(legt_key, f=x_jnp)
    c, y = hippo_lti_legt_fe.apply(params, f=x_jnp)
    y = einops.rearrange(y, "batch seq_len -> batch 1 seq_len")

    x_tensor = einops.rearrange(x_tensor, "batch seq_len -> seq_len batch")
    _, hr_c = hr_hippo_lti_legt_fe(x_tensor, fast=False)
    hr_c = einops.rearrange(
        hr_c, "batch N -> batch 1 N"
    )  # add input_size dimension for comparison
    hr_y = hr_hippo_lti_legt_fe.reconstruct(hr_c)
    print(f"hr_y shape: {hr_y.shape}")
    hr_y = einops.rearrange(hr_y, "batch seq_len -> batch 1 seq_len")
    hr_c = jnp.asarray(hr_c, dtype=jnp.float32)  # convert torch array to jax array
    hr_y = jnp.asarray(hr_y, dtype=jnp.float32)  # convert torch array to jax array

    assert hr_c.shape == c.shape
    assert hr_y.shape == y.shape

    assert c.shape == (16, 1, 50)  # batch_size, input_size, N
    assert y.shape == (16, 1, 3000)  # batch_size, input_size, seq_len

    co_flag = True
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            co_check = jnp.allclose(c[i, j, :], hr_c[i, j, :], rtol=1e-03, atol=1e-03)
            if not co_check:
                co_flag = False
            assert jnp.allclose(y[i, j, :], hr_y[i, j, :], rtol=1e-03, atol=1e-03)
    print(f"coefficients tests: {co_flag}")


# ----------
# -- lmu --
# ----------


def test_hippo_lmu_lti_fe_reconstruction(
    hippo_lti_lmu_fe, hr_hippo_lti_lmu_fe, random_16_input, lmu_key
):
    print("\nHiPPO RECONSTRUCTION LMU (LTI, FE)")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    assert x_jnp.shape == (16, 3000)  # batch_size, seq_len

    x_jnp = einops.rearrange(
        x_jnp, "batch seq_len -> batch seq_len 1"
    )  # add input_size dimension

    params = hippo_lti_lmu_fe.init(lmu_key, f=x_jnp)
    c, y = hippo_lti_lmu_fe.apply(params, f=x_jnp)
    y = einops.rearrange(y, "batch seq_len -> batch 1 seq_len")

    x_tensor = einops.rearrange(x_tensor, "batch seq_len -> seq_len batch")
    _, hr_c = hr_hippo_lti_lmu_fe(x_tensor, fast=False)
    hr_c = einops.rearrange(
        hr_c, "batch N -> batch 1 N"
    )  # add input_size dimension for comparison
    hr_y = hr_hippo_lti_lmu_fe.reconstruct(hr_c)
    print(f"hr_y shape: {hr_y.shape}")
    hr_y = einops.rearrange(hr_y, "batch seq_len -> batch 1 seq_len")
    hr_c = jnp.asarray(hr_c, dtype=jnp.float32)  # convert torch array to jax array
    hr_y = jnp.asarray(hr_y, dtype=jnp.float32)  # convert torch array to jax array

    assert hr_c.shape == c.shape
    assert hr_y.shape == y.shape

    assert c.shape == (16, 1, 50)  # batch_size, input_size, N
    assert y.shape == (16, 1, 3000)  # batch_size, input_size, seq_len

    co_flag = True
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            co_check = jnp.allclose(c[i, j, :], hr_c[i, j, :], rtol=1e-03, atol=1e-03)
            if not co_check:
                co_flag = False
            assert jnp.allclose(y[i, j, :], hr_y[i, j, :], rtol=1e-03, atol=1e-03)
    print(f"coefficients tests: {co_flag}")


# ----------
# -- lagt --
# ----------


def test_hippo_lagt_lti_fe_reconstruction(
    hippo_lti_lagt_fe, hr_hippo_lti_lagt_fe, random_16_input, lagt_key
):
    print("\nHiPPO RECONSTRUCTION LAGT (LTI, FE)")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    assert x_jnp.shape == (16, 3000)  # batch_size, seq_len

    x_jnp = einops.rearrange(
        x_jnp, "batch seq_len -> batch seq_len 1"
    )  # add input_size dimension

    params = hippo_lti_lagt_fe.init(lagt_key, f=x_jnp)
    c, y = hippo_lti_lagt_fe.apply(params, f=x_jnp)
    y = einops.rearrange(y, "batch seq_len -> batch 1 seq_len")

    x_tensor = einops.rearrange(x_tensor, "batch seq_len -> seq_len batch")
    _, hr_c = hr_hippo_lti_lagt_fe(x_tensor, fast=False)
    hr_c = einops.rearrange(
        hr_c, "batch N -> batch 1 N"
    )  # add input_size dimension for comparison
    hr_y = hr_hippo_lti_lagt_fe.reconstruct(hr_c)
    print(f"hr_y shape: {hr_y.shape}")
    hr_y = einops.rearrange(hr_y, "batch seq_len -> batch 1 seq_len")
    hr_c = jnp.asarray(hr_c, dtype=jnp.float32)  # convert torch array to jax array
    hr_y = jnp.asarray(hr_y, dtype=jnp.float32)  # convert torch array to jax array

    assert hr_c.shape == c.shape
    assert hr_y.shape == y.shape

    assert c.shape == (16, 1, 50)  # batch_size, input_size, N
    assert y.shape == (16, 1, 3000)  # batch_size, input_size, seq_len

    co_flag = True
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            co_check = jnp.allclose(c[i, j, :], hr_c[i, j, :], rtol=1e-03, atol=1e-03)
            if not co_check:
                co_flag = False
            assert jnp.allclose(y[i, j, :], hr_y[i, j, :], rtol=1e-03, atol=1e-03)
    print(f"coefficients tests: {co_flag}")


# ------------
# --- fout ---
# ------------


def test_hippo_fout_lti_fe_reconstruction(
    hippo_lti_fout_fe, hr_hippo_lti_fout_fe, random_16_input, fout_key
):
    print("\nHiPPO RECONSTRUCTION FOUT (LTI, FE)")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    assert x_jnp.shape == (16, 3000)  # batch_size, seq_len

    x_jnp = einops.rearrange(
        x_jnp, "batch seq_len -> batch seq_len 1"
    )  # add input_size dimension

    params = hippo_lti_fout_fe.init(fout_key, f=x_jnp)
    c, y = hippo_lti_fout_fe.apply(params, f=x_jnp)
    y = einops.rearrange(y, "batch seq_len -> batch 1 seq_len")

    x_tensor = einops.rearrange(x_tensor, "batch seq_len -> seq_len batch")
    _, hr_c = hr_hippo_lti_fout_fe(x_tensor, fast=False)
    hr_c = einops.rearrange(
        hr_c, "batch N -> batch 1 N"
    )  # add input_size dimension for comparison
    hr_y = hr_hippo_lti_fout_fe.reconstruct(hr_c)
    print(f"hr_y shape: {hr_y.shape}")
    hr_y = einops.rearrange(hr_y, "batch seq_len -> batch 1 seq_len")
    hr_c = jnp.asarray(hr_c, dtype=jnp.float32)  # convert torch array to jax array
    hr_y = jnp.asarray(hr_y, dtype=jnp.float32)  # convert torch array to jax array

    assert hr_c.shape == c.shape
    assert hr_y.shape == y.shape

    assert c.shape == (16, 1, 50)  # batch_size, input_size, N
    assert y.shape == (16, 1, 3000)  # batch_size, input_size, seq_len

    co_flag = True
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            co_check = jnp.allclose(c[i, j, :], hr_c[i, j, :], rtol=1e-03, atol=1e-03)
            if not co_check:
                co_flag = False
            assert jnp.allclose(y[i, j, :], hr_y[i, j, :], rtol=1e-03, atol=1e-03)
    print(f"coefficients tests: {co_flag}")


# --------------------
# -- Backward Euler --
# --------------------

# ----------
# -- legs --
# ----------


def test_hippo_legs_lti_be_reconstruction(
    hippo_lti_legs_be, hr_hippo_lti_legs_be, random_16_input, legs_key
):
    print("\nHiPPO RECONSTRUCTION LEGS (LTI, BE)")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    assert x_jnp.shape == (16, 3000)  # batch_size, seq_len

    x_jnp = einops.rearrange(
        x_jnp, "batch seq_len -> batch seq_len 1"
    )  # add input_size dimension

    params = hippo_lti_legs_be.init(legs_key, f=x_jnp)
    c, y = hippo_lti_legs_be.apply(params, f=x_jnp)
    y = einops.rearrange(y, "batch seq_len -> batch 1 seq_len")

    x_tensor = einops.rearrange(x_tensor, "batch seq_len -> seq_len batch")
    _, hr_c = hr_hippo_lti_legs_be(x_tensor, fast=False)
    hr_c = einops.rearrange(
        hr_c, "batch N -> batch 1 N"
    )  # add input_size dimension for comparison
    hr_y = hr_hippo_lti_legs_be.reconstruct(hr_c)
    print(f"hr_y shape: {hr_y.shape}")
    hr_y = einops.rearrange(hr_y, "batch seq_len -> batch 1 seq_len")
    hr_c = jnp.asarray(hr_c, dtype=jnp.float32)  # convert torch array to jax array
    hr_y = jnp.asarray(hr_y, dtype=jnp.float32)  # convert torch array to jax array

    assert hr_c.shape == c.shape
    assert hr_y.shape == y.shape

    assert c.shape == (16, 1, 50)  # batch_size, input_size, N
    assert y.shape == (16, 1, 3000)  # batch_size, input_size, seq_len

    co_flag = True
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            co_check = jnp.allclose(c[i, j, :], hr_c[i, j, :], rtol=1e-03, atol=1e-03)
            if not co_check:
                co_flag = False
            assert jnp.allclose(y[i, j, :], hr_y[i, j, :], rtol=1e-03, atol=1e-03)
    print(f"coefficients tests: {co_flag}")


def test_hippo_legs_lsi_be_reconstruction(
    hippo_lsi_legs_be, hr_hippo_lsi_legs_be, random_16_input, legs_key
):
    print("\nHiPPO RECONSTRUCTION LEGS (LSI, BE)")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    assert x_jnp.shape == (16, 3000)  # batch_size, seq_len

    x_jnp = einops.rearrange(
        x_jnp, "batch seq_len -> batch seq_len 1"
    )  # add input_size dimension

    params = hippo_lsi_legs_be.init(legs_key, f=x_jnp)
    c, y = hippo_lsi_legs_be.apply(params, f=x_jnp)

    y = einops.rearrange(
        y, "batch seq_len seq_len2 input_len -> batch seq_len input_len seq_len2"
    )

    x_tensor = einops.rearrange(x_tensor, "batch seq_len -> seq_len batch")
    hr_cs, _ = hr_hippo_lsi_legs_be(x_tensor, fast=False)
    hr_c = einops.rearrange(
        hr_cs, "seq_len batch N -> batch seq_len 1 N"
    )  # add input_size and swap batch and seq_len dimension for comparison
    hr_y = hr_hippo_lsi_legs_be.reconstruct(hr_c)
    print(f"hr_y shape: {hr_y.shape}")
    hr_y = einops.rearrange(
        hr_y, "seq_len batch seq_len2 input_len -> batch seq_len input_len seq_len2"
    )
    hr_c = jnp.asarray(hr_c, dtype=jnp.float32)  # convert torch array to jax array
    hr_y = jnp.asarray(hr_y, dtype=jnp.float32)  # convert torch array to jax array

    assert hr_c.shape == c.shape
    assert hr_y.shape == y.shape

    assert c.shape == (16, 3000, 1, 50)  # batch_size, seq_len, input_size, N
    assert y.shape == (16, 3000, 1, 3000)  # batch_size, seq_len, input_size, seq_len

    co_flag = True
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            co_check = jnp.allclose(
                c[i, j, :, :], hr_c[i, j, :, :], rtol=1e-03, atol=1e-03
            )
            if not co_check:
                co_flag = False
            assert jnp.allclose(y[i, j, :, :], hr_y[i, j, :, :], rtol=1e-03, atol=1e-03)
    print(f"coefficients tests: {co_flag}")


# ----------
# -- legt --
# ----------


def test_hippo_legt_lti_be_reconstruction(
    hippo_lti_legt_be, hr_hippo_lti_legt_be, random_16_input, legt_key
):
    print("\nHiPPO RECONSTRUCTION LEGT (LTI, BE)")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    assert x_jnp.shape == (16, 3000)  # batch_size, seq_len

    x_jnp = einops.rearrange(
        x_jnp, "batch seq_len -> batch seq_len 1"
    )  # add input_size dimension

    params = hippo_lti_legt_be.init(legt_key, f=x_jnp)
    c, y = hippo_lti_legt_be.apply(params, f=x_jnp)
    y = einops.rearrange(y, "batch seq_len -> batch 1 seq_len")

    x_tensor = einops.rearrange(x_tensor, "batch seq_len -> seq_len batch")
    _, hr_c = hr_hippo_lti_legt_be(x_tensor, fast=False)
    hr_c = einops.rearrange(
        hr_c, "batch N -> batch 1 N"
    )  # add input_size dimension for comparison
    hr_y = hr_hippo_lti_legt_be.reconstruct(hr_c)
    print(f"hr_y shape: {hr_y.shape}")
    hr_y = einops.rearrange(hr_y, "batch seq_len -> batch 1 seq_len")
    hr_c = jnp.asarray(hr_c, dtype=jnp.float32)  # convert torch array to jax array
    hr_y = jnp.asarray(hr_y, dtype=jnp.float32)  # convert torch array to jax array

    assert hr_c.shape == c.shape
    assert hr_y.shape == y.shape

    assert c.shape == (16, 1, 50)  # batch_size, input_size, N
    assert y.shape == (16, 1, 3000)  # batch_size, input_size, seq_len

    co_flag = True
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            co_check = jnp.allclose(c[i, j, :], hr_c[i, j, :], rtol=1e-03, atol=1e-03)
            if not co_check:
                co_flag = False
            assert jnp.allclose(y[i, j, :], hr_y[i, j, :], rtol=1e-03, atol=1e-03)
    print(f"coefficients tests: {co_flag}")


# ----------
# -- lmu --
# ----------


def test_hippo_lmu_lti_be_reconstruction(
    hippo_lti_lmu_be, hr_hippo_lti_lmu_be, random_16_input, lmu_key
):
    print("\nHiPPO RECONSTRUCTION LMU (LTI, BE)")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    assert x_jnp.shape == (16, 3000)  # batch_size, seq_len

    x_jnp = einops.rearrange(
        x_jnp, "batch seq_len -> batch seq_len 1"
    )  # add input_size dimension

    params = hippo_lti_lmu_be.init(lmu_key, f=x_jnp)
    c, y = hippo_lti_lmu_be.apply(params, f=x_jnp)
    y = einops.rearrange(y, "batch seq_len -> batch 1 seq_len")

    x_tensor = einops.rearrange(x_tensor, "batch seq_len -> seq_len batch")
    _, hr_c = hr_hippo_lti_lmu_be(x_tensor, fast=False)
    hr_c = einops.rearrange(
        hr_c, "batch N -> batch 1 N"
    )  # add input_size dimension for comparison
    hr_y = hr_hippo_lti_lmu_be.reconstruct(hr_c)
    print(f"hr_y shape: {hr_y.shape}")
    hr_y = einops.rearrange(hr_y, "batch seq_len -> batch 1 seq_len")
    hr_c = jnp.asarray(hr_c, dtype=jnp.float32)  # convert torch array to jax array
    hr_y = jnp.asarray(hr_y, dtype=jnp.float32)  # convert torch array to jax array

    assert hr_c.shape == c.shape
    assert hr_y.shape == y.shape

    assert c.shape == (16, 1, 50)  # batch_size, input_size, N
    assert y.shape == (16, 1, 3000)  # batch_size, input_size, seq_len

    co_flag = True
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            co_check = jnp.allclose(c[i, j, :], hr_c[i, j, :], rtol=1e-03, atol=1e-03)
            if not co_check:
                co_flag = False
            assert jnp.allclose(y[i, j, :], hr_y[i, j, :], rtol=1e-03, atol=1e-03)
    print(f"coefficients tests: {co_flag}")


# ----------
# -- lagt --
# ----------


def test_hippo_lagt_lti_be_reconstruction(
    hippo_lti_lagt_be, hr_hippo_lti_lagt_be, random_16_input, lagt_key
):
    print("\nHiPPO RECONSTRUCTION LAGT (LTI, BE)")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    assert x_jnp.shape == (16, 3000)  # batch_size, seq_len

    x_jnp = einops.rearrange(
        x_jnp, "batch seq_len -> batch seq_len 1"
    )  # add input_size dimension

    params = hippo_lti_lagt_be.init(lagt_key, f=x_jnp)
    c, y = hippo_lti_lagt_be.apply(params, f=x_jnp)
    y = einops.rearrange(y, "batch seq_len -> batch 1 seq_len")

    x_tensor = einops.rearrange(x_tensor, "batch seq_len -> seq_len batch")
    _, hr_c = hr_hippo_lti_lagt_be(x_tensor, fast=False)
    hr_c = einops.rearrange(
        hr_c, "batch N -> batch 1 N"
    )  # add input_size dimension for comparison
    hr_y = hr_hippo_lti_lagt_be.reconstruct(hr_c)
    print(f"hr_y shape: {hr_y.shape}")
    hr_y = einops.rearrange(hr_y, "batch seq_len -> batch 1 seq_len")
    hr_c = jnp.asarray(hr_c, dtype=jnp.float32)  # convert torch array to jax array
    hr_y = jnp.asarray(hr_y, dtype=jnp.float32)  # convert torch array to jax array

    assert hr_c.shape == c.shape
    assert hr_y.shape == y.shape

    assert c.shape == (16, 1, 50)  # batch_size, input_size, N
    assert y.shape == (16, 1, 3000)  # batch_size, input_size, seq_len

    co_flag = True
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            co_check = jnp.allclose(c[i, j, :], hr_c[i, j, :], rtol=1e-03, atol=1e-03)
            if not co_check:
                co_flag = False
            assert jnp.allclose(y[i, j, :], hr_y[i, j, :], rtol=1e-03, atol=1e-03)
    print(f"coefficients tests: {co_flag}")


# ------------
# --- fout ---
# ------------


def test_hippo_fout_lti_be_reconstruction(
    hippo_lti_fout_be, hr_hippo_lti_fout_be, random_16_input, fout_key
):
    print("\nHiPPO RECONSTRUCTION FOUT (LTI, BE)")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    assert x_jnp.shape == (16, 3000)  # batch_size, seq_len

    x_jnp = einops.rearrange(
        x_jnp, "batch seq_len -> batch seq_len 1"
    )  # add input_size dimension

    params = hippo_lti_fout_be.init(fout_key, f=x_jnp)
    c, y = hippo_lti_fout_be.apply(params, f=x_jnp)
    y = einops.rearrange(y, "batch seq_len -> batch 1 seq_len")

    x_tensor = einops.rearrange(x_tensor, "batch seq_len -> seq_len batch")
    _, hr_c = hr_hippo_lti_fout_be(x_tensor, fast=False)
    hr_c = einops.rearrange(
        hr_c, "batch N -> batch 1 N"
    )  # add input_size dimension for comparison
    hr_y = hr_hippo_lti_fout_be.reconstruct(hr_c)
    print(f"hr_y shape: {hr_y.shape}")
    hr_y = einops.rearrange(hr_y, "batch seq_len -> batch 1 seq_len")
    hr_c = jnp.asarray(hr_c, dtype=jnp.float32)  # convert torch array to jax array
    hr_y = jnp.asarray(hr_y, dtype=jnp.float32)  # convert torch array to jax array

    assert hr_c.shape == c.shape
    assert hr_y.shape == y.shape

    assert c.shape == (16, 1, 50)  # batch_size, input_size, N
    assert y.shape == (16, 1, 3000)  # batch_size, input_size, seq_len

    co_flag = True
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            co_check = jnp.allclose(c[i, j, :], hr_c[i, j, :], rtol=1e-03, atol=1e-03)
            if not co_check:
                co_flag = False
            assert jnp.allclose(y[i, j, :], hr_y[i, j, :], rtol=1e-03, atol=1e-03)
    print(f"coefficients tests: {co_flag}")


# --------------------
# ----- Bilinear -----
# --------------------
# ----------
# -- legs --
# ----------


def test_hippo_legs_lti_bi_reconstruction(
    hippo_lti_legs_bi, hr_hippo_lti_legs_bi, random_16_input, legs_key
):
    print("\nHiPPO RECONSTRUCTION LEGS (LTI, BI)")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    assert x_jnp.shape == (16, 3000)  # batch_size, seq_len

    x_jnp = einops.rearrange(
        x_jnp, "batch seq_len -> batch seq_len 1"
    )  # add input_size dimension

    params = hippo_lti_legs_bi.init(legs_key, f=x_jnp)
    c, y = hippo_lti_legs_bi.apply(params, f=x_jnp)
    y = einops.rearrange(y, "batch seq_len -> batch 1 seq_len")

    x_tensor = einops.rearrange(x_tensor, "batch seq_len -> seq_len batch")
    _, hr_c = hr_hippo_lti_legs_bi(x_tensor, fast=False)
    hr_c = einops.rearrange(
        hr_c, "batch N -> batch 1 N"
    )  # add input_size dimension for comparison
    hr_y = hr_hippo_lti_legs_bi.reconstruct(hr_c)
    print(f"hr_y shape: {hr_y.shape}")
    hr_y = einops.rearrange(hr_y, "batch seq_len -> batch 1 seq_len")
    hr_c = jnp.asarray(hr_c, dtype=jnp.float32)  # convert torch array to jax array
    hr_y = jnp.asarray(hr_y, dtype=jnp.float32)  # convert torch array to jax array

    assert hr_c.shape == c.shape
    assert hr_y.shape == y.shape

    assert c.shape == (16, 1, 50)  # batch_size, input_size, N
    assert y.shape == (16, 1, 3000)  # batch_size, input_size, seq_len

    co_flag = True
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            co_check = jnp.allclose(c[i, j, :], hr_c[i, j, :], rtol=1e-03, atol=1e-03)
            if not co_check:
                co_flag = False
            assert jnp.allclose(y[i, j, :], hr_y[i, j, :], rtol=1e-03, atol=1e-03)
    print(f"coefficients tests: {co_flag}")


def test_hippo_legs_lsi_bi_reconstruction(
    hippo_lsi_legs_bi, hr_hippo_lsi_legs_bi, random_16_input, legs_key
):
    print("\nHiPPO RECONSTRUCTION LEGS (LSI, BI)")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    assert x_jnp.shape == (16, 3000)  # batch_size, seq_len

    x_jnp = einops.rearrange(
        x_jnp, "batch seq_len -> batch seq_len 1"
    )  # add input_size dimension

    params = hippo_lsi_legs_bi.init(legs_key, f=x_jnp)
    c, y = hippo_lsi_legs_bi.apply(params, f=x_jnp)

    y = einops.rearrange(
        y, "batch seq_len seq_len2 input_len -> batch seq_len input_len seq_len2"
    )

    x_tensor = einops.rearrange(x_tensor, "batch seq_len -> seq_len batch")
    hr_cs, _ = hr_hippo_lsi_legs_bi(x_tensor, fast=False)
    hr_c = einops.rearrange(
        hr_cs, "seq_len batch N -> batch seq_len 1 N"
    )  # add input_size and swap batch and seq_len dimension for comparison
    hr_y = hr_hippo_lsi_legs_bi.reconstruct(hr_c)
    print(f"hr_y shape: {hr_y.shape}")
    hr_y = einops.rearrange(
        hr_y, "seq_len batch seq_len2 input_len -> batch seq_len input_len seq_len2"
    )
    hr_c = jnp.asarray(hr_c, dtype=jnp.float32)  # convert torch array to jax array
    hr_y = jnp.asarray(hr_y, dtype=jnp.float32)  # convert torch array to jax array

    assert hr_c.shape == c.shape
    assert hr_y.shape == y.shape

    assert c.shape == (16, 3000, 1, 50)  # batch_size, seq_len, input_size, N
    assert y.shape == (16, 3000, 1, 3000)  # batch_size, seq_len, input_size, seq_len

    co_flag = True
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            co_check = jnp.allclose(
                c[i, j, :, :], hr_c[i, j, :, :], rtol=1e-03, atol=1e-03
            )
            if not co_check:
                co_flag = False
            assert jnp.allclose(y[i, j, :, :], hr_y[i, j, :, :], rtol=1e-03, atol=1e-03)
    print(f"coefficients tests: {co_flag}")


# ----------
# -- legt --
# ----------


def test_hippo_legt_lti_bi_reconstruction(
    hippo_lti_legt_bi, hr_hippo_lti_legt_bi, random_16_input, legt_key
):
    print("\nHiPPO RECONSTRUCTION LEGT (LTI, BI)")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    assert x_jnp.shape == (16, 3000)  # batch_size, seq_len

    x_jnp = einops.rearrange(
        x_jnp, "batch seq_len -> batch seq_len 1"
    )  # add input_size dimension

    params = hippo_lti_legt_bi.init(legt_key, f=x_jnp)
    c, y = hippo_lti_legt_bi.apply(params, f=x_jnp)
    y = einops.rearrange(y, "batch seq_len -> batch 1 seq_len")

    x_tensor = einops.rearrange(x_tensor, "batch seq_len -> seq_len batch")
    _, hr_c = hr_hippo_lti_legt_bi(x_tensor, fast=False)
    hr_c = einops.rearrange(
        hr_c, "batch N -> batch 1 N"
    )  # add input_size dimension for comparison
    hr_y = hr_hippo_lti_legt_bi.reconstruct(hr_c)
    print(f"hr_y shape: {hr_y.shape}")
    hr_y = einops.rearrange(hr_y, "batch seq_len -> batch 1 seq_len")
    hr_c = jnp.asarray(hr_c, dtype=jnp.float32)  # convert torch array to jax array
    hr_y = jnp.asarray(hr_y, dtype=jnp.float32)  # convert torch array to jax array

    assert hr_c.shape == c.shape
    assert hr_y.shape == y.shape

    assert c.shape == (16, 1, 50)  # batch_size, input_size, N
    assert y.shape == (16, 1, 3000)  # batch_size, input_size, seq_len

    co_flag = True
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            co_check = jnp.allclose(c[i, j, :], hr_c[i, j, :], rtol=1e-03, atol=1e-03)
            if not co_check:
                co_flag = False
            assert jnp.allclose(y[i, j, :], hr_y[i, j, :], rtol=1e-03, atol=1e-03)
    print(f"coefficients tests: {co_flag}")


# ----------
# -- lmu --
# ----------


def test_hippo_lmu_lti_bi_reconstruction(
    hippo_lti_lmu_bi, hr_hippo_lti_lmu_bi, random_16_input, lmu_key
):
    print("\nHiPPO RECONSTRUCTION LMU (LTI, BI)")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    assert x_jnp.shape == (16, 3000)  # batch_size, seq_len

    x_jnp = einops.rearrange(
        x_jnp, "batch seq_len -> batch seq_len 1"
    )  # add input_size dimension

    params = hippo_lti_lmu_bi.init(lmu_key, f=x_jnp)
    c, y = hippo_lti_lmu_bi.apply(params, f=x_jnp)
    y = einops.rearrange(y, "batch seq_len -> batch 1 seq_len")

    x_tensor = einops.rearrange(x_tensor, "batch seq_len -> seq_len batch")
    _, hr_c = hr_hippo_lti_lmu_bi(x_tensor, fast=False)
    hr_c = einops.rearrange(
        hr_c, "batch N -> batch 1 N"
    )  # add input_size dimension for comparison
    hr_y = hr_hippo_lti_lmu_bi.reconstruct(hr_c)
    print(f"hr_y shape: {hr_y.shape}")
    hr_y = einops.rearrange(hr_y, "batch seq_len -> batch 1 seq_len")
    hr_c = jnp.asarray(hr_c, dtype=jnp.float32)  # convert torch array to jax array
    hr_y = jnp.asarray(hr_y, dtype=jnp.float32)  # convert torch array to jax array

    assert hr_c.shape == c.shape
    assert hr_y.shape == y.shape

    assert c.shape == (16, 1, 50)  # batch_size, input_size, N
    assert y.shape == (16, 1, 3000)  # batch_size, input_size, seq_len

    co_flag = True
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            co_check = jnp.allclose(c[i, j, :], hr_c[i, j, :], rtol=1e-03, atol=1e-03)
            if not co_check:
                co_flag = False
            assert jnp.allclose(y[i, j, :], hr_y[i, j, :], rtol=1e-03, atol=1e-03)
    print(f"coefficients tests: {co_flag}")


# ----------
# -- lagt --
# ----------


def test_hippo_lagt_lti_bi_reconstruction(
    hippo_lti_lagt_bi, hr_hippo_lti_lagt_bi, random_16_input, lagt_key
):
    print("\nHiPPO RECONSTRUCTION LAGT (LTI, BI)")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    assert x_jnp.shape == (16, 3000)  # batch_size, seq_len

    x_jnp = einops.rearrange(
        x_jnp, "batch seq_len -> batch seq_len 1"
    )  # add input_size dimension

    params = hippo_lti_lagt_bi.init(lagt_key, f=x_jnp)
    c, y = hippo_lti_lagt_bi.apply(params, f=x_jnp)
    y = einops.rearrange(y, "batch seq_len -> batch 1 seq_len")

    x_tensor = einops.rearrange(x_tensor, "batch seq_len -> seq_len batch")
    _, hr_c = hr_hippo_lti_lagt_bi(x_tensor, fast=False)
    hr_c = einops.rearrange(
        hr_c, "batch N -> batch 1 N"
    )  # add input_size dimension for comparison
    hr_y = hr_hippo_lti_lagt_bi.reconstruct(hr_c)
    print(f"hr_y shape: {hr_y.shape}")
    hr_y = einops.rearrange(hr_y, "batch seq_len -> batch 1 seq_len")
    hr_c = jnp.asarray(hr_c, dtype=jnp.float32)  # convert torch array to jax array
    hr_y = jnp.asarray(hr_y, dtype=jnp.float32)  # convert torch array to jax array

    assert hr_c.shape == c.shape
    assert hr_y.shape == y.shape

    assert c.shape == (16, 1, 50)  # batch_size, input_size, N
    assert y.shape == (16, 1, 3000)  # batch_size, input_size, seq_len

    co_flag = True
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            co_check = jnp.allclose(c[i, j, :], hr_c[i, j, :], rtol=1e-03, atol=1e-03)
            if not co_check:
                co_flag = False
            assert jnp.allclose(y[i, j, :], hr_y[i, j, :], rtol=1e-03, atol=1e-03)
    print(f"coefficients tests: {co_flag}")


# ------------
# --- fout ---
# ------------


def test_hippo_fout_lti_bi_reconstruction(
    hippo_lti_fout_bi, hr_hippo_lti_fout_bi, random_16_input, fout_key
):
    print("\nHiPPO RECONSTRUCTION FOUT (LTI, BI)")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    assert x_jnp.shape == (16, 3000)  # batch_size, seq_len

    x_jnp = einops.rearrange(
        x_jnp, "batch seq_len -> batch seq_len 1"
    )  # add input_size dimension

    params = hippo_lti_fout_bi.init(fout_key, f=x_jnp)
    c, y = hippo_lti_fout_bi.apply(params, f=x_jnp)
    y = einops.rearrange(y, "batch seq_len -> batch 1 seq_len")

    x_tensor = einops.rearrange(x_tensor, "batch seq_len -> seq_len batch")
    _, hr_c = hr_hippo_lti_fout_bi(x_tensor, fast=False)
    hr_c = einops.rearrange(
        hr_c, "batch N -> batch 1 N"
    )  # add input_size dimension for comparison
    hr_y = hr_hippo_lti_fout_bi.reconstruct(hr_c)
    print(f"hr_y shape: {hr_y.shape}")
    hr_y = einops.rearrange(hr_y, "batch seq_len -> batch 1 seq_len")
    hr_c = jnp.asarray(hr_c, dtype=jnp.float32)  # convert torch array to jax array
    hr_y = jnp.asarray(hr_y, dtype=jnp.float32)  # convert torch array to jax array

    assert hr_c.shape == c.shape
    assert hr_y.shape == y.shape

    assert c.shape == (16, 1, 50)  # batch_size, input_size, N
    assert y.shape == (16, 1, 3000)  # batch_size, input_size, seq_len

    co_flag = True
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            co_check = jnp.allclose(c[i, j, :], hr_c[i, j, :], rtol=1e-03, atol=1e-03)
            if not co_check:
                co_flag = False
            assert jnp.allclose(y[i, j, :], hr_y[i, j, :], rtol=1e-03, atol=1e-03)
    print(f"coefficients tests: {co_flag}")


# --------------------
# - Zero-order Hold --
# --------------------

# ----------
# -- legs --
# ----------


def test_hippo_legs_lti_zoh_reconstruction(
    hippo_lti_legs_zoh, hr_hippo_lti_legs_zoh, random_16_input, legs_key
):
    print("\nHiPPO RECONSTRUCTION LEGS (LTI, ZOH)")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    assert x_jnp.shape == (16, 3000)  # batch_size, seq_len

    x_jnp = einops.rearrange(
        x_jnp, "batch seq_len -> batch seq_len 1"
    )  # add input_size dimension

    params = hippo_lti_legs_zoh.init(legs_key, f=x_jnp)
    c, y = hippo_lti_legs_zoh.apply(params, f=x_jnp)
    y = einops.rearrange(y, "batch seq_len -> batch 1 seq_len")

    x_tensor = einops.rearrange(x_tensor, "batch seq_len -> seq_len batch")
    _, hr_c = hr_hippo_lti_legs_zoh(x_tensor, fast=False)
    hr_c = einops.rearrange(
        hr_c, "batch N -> batch 1 N"
    )  # add input_size dimension for comparison
    hr_y = hr_hippo_lti_legs_zoh.reconstruct(hr_c)
    print(f"hr_y shape: {hr_y.shape}")
    hr_y = einops.rearrange(hr_y, "batch seq_len -> batch 1 seq_len")
    hr_c = jnp.asarray(hr_c, dtype=jnp.float32)  # convert torch array to jax array
    hr_y = jnp.asarray(hr_y, dtype=jnp.float32)  # convert torch array to jax array

    assert hr_c.shape == c.shape
    assert hr_y.shape == y.shape

    assert c.shape == (16, 1, 50)  # batch_size, input_size, N
    assert y.shape == (16, 1, 3000)  # batch_size, input_size, seq_len

    co_flag = True
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            co_check = jnp.allclose(c[i, j, :], hr_c[i, j, :], rtol=1e-03, atol=1e-03)
            if not co_check:
                co_flag = False
            assert jnp.allclose(y[i, j, :], hr_y[i, j, :], rtol=1e-03, atol=1e-03)
    print(f"coefficients tests: {co_flag}")


def test_hippo_legs_lsi_zoh_reconstruction(
    hippo_lsi_legs_zoh, hr_hippo_lsi_legs_zoh, random_16_input, legs_key
):
    print("\nHiPPO RECONSTRUCTION LEGS (LSI, ZOH)")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    assert x_jnp.shape == (16, 3000)  # batch_size, seq_len

    x_jnp = einops.rearrange(
        x_jnp, "batch seq_len -> batch seq_len 1"
    )  # add input_size dimension

    params = hippo_lsi_legs_zoh.init(legs_key, f=x_jnp)
    c, y = hippo_lsi_legs_zoh.apply(params, f=x_jnp)

    y = einops.rearrange(
        y, "batch seq_len seq_len2 input_len -> batch seq_len input_len seq_len2"
    )

    x_tensor = einops.rearrange(x_tensor, "batch seq_len -> seq_len batch")
    hr_cs, _ = hr_hippo_lsi_legs_zoh(x_tensor, fast=False)
    hr_c = einops.rearrange(
        hr_cs, "seq_len batch N -> batch seq_len 1 N"
    )  # add input_size and swap batch and seq_len dimension for comparison
    hr_y = hr_hippo_lsi_legs_zoh.reconstruct(hr_c)
    print(f"hr_y shape: {hr_y.shape}")
    hr_y = einops.rearrange(
        hr_y, "seq_len batch seq_len2 input_len -> batch seq_len input_len seq_len2"
    )
    hr_c = jnp.asarray(hr_c, dtype=jnp.float32)  # convert torch array to jax array
    hr_y = jnp.asarray(hr_y, dtype=jnp.float32)  # convert torch array to jax array

    assert hr_c.shape == c.shape
    assert hr_y.shape == y.shape

    assert c.shape == (16, 3000, 1, 50)  # batch_size, seq_len, input_size, N
    assert y.shape == (16, 3000, 1, 3000)  # batch_size, seq_len, input_size, seq_len

    co_flag = True
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            co_check = jnp.allclose(
                c[i, j, :, :], hr_c[i, j, :, :], rtol=1e-03, atol=1e-03
            )
            if not co_check:
                co_flag = False
            assert jnp.allclose(y[i, j, :, :], hr_y[i, j, :, :], rtol=1e-03, atol=1e-03)
    print(f"coefficients tests: {co_flag}")


# ----------
# -- legt --
# ----------


def test_hippo_legt_lti_zoh_reconstruction(
    hippo_lti_legt_zoh, hr_hippo_lti_legt_zoh, random_16_input, legt_key
):
    print("\nHiPPO RECONSTRUCTION LEGT (LTI, ZOH)")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    assert x_jnp.shape == (16, 3000)  # batch_size, seq_len

    x_jnp = einops.rearrange(
        x_jnp, "batch seq_len -> batch seq_len 1"
    )  # add input_size dimension

    params = hippo_lti_legt_zoh.init(legt_key, f=x_jnp)
    c, y = hippo_lti_legt_zoh.apply(params, f=x_jnp)
    y = einops.rearrange(y, "batch seq_len -> batch 1 seq_len")

    x_tensor = einops.rearrange(x_tensor, "batch seq_len -> seq_len batch")
    _, hr_c = hr_hippo_lti_legt_zoh(x_tensor, fast=False)
    hr_c = einops.rearrange(
        hr_c, "batch N -> batch 1 N"
    )  # add input_size dimension for comparison
    hr_y = hr_hippo_lti_legt_zoh.reconstruct(hr_c)
    print(f"hr_y shape: {hr_y.shape}")
    hr_y = einops.rearrange(hr_y, "batch seq_len -> batch 1 seq_len")
    hr_c = jnp.asarray(hr_c, dtype=jnp.float32)  # convert torch array to jax array
    hr_y = jnp.asarray(hr_y, dtype=jnp.float32)  # convert torch array to jax array

    assert hr_c.shape == c.shape
    assert hr_y.shape == y.shape

    assert c.shape == (16, 1, 50)  # batch_size, input_size, N
    assert y.shape == (16, 1, 3000)  # batch_size, input_size, seq_len

    co_flag = True
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            co_check = jnp.allclose(c[i, j, :], hr_c[i, j, :], rtol=1e-03, atol=1e-03)
            if not co_check:
                co_flag = False
            assert jnp.allclose(y[i, j, :], hr_y[i, j, :], rtol=1e-03, atol=1e-03)
    print(f"coefficients tests: {co_flag}")


# ----------
# -- lmu --
# ----------


def test_hippo_lmu_lti_zoh_reconstruction(
    hippo_lti_lmu_zoh, hr_hippo_lti_lmu_zoh, random_16_input, lmu_key
):
    print("\nHiPPO RECONSTRUCTION LMU (LTI, ZOH)")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    assert x_jnp.shape == (16, 3000)  # batch_size, seq_len

    x_jnp = einops.rearrange(
        x_jnp, "batch seq_len -> batch seq_len 1"
    )  # add input_size dimension

    params = hippo_lti_lmu_zoh.init(lmu_key, f=x_jnp)
    c, y = hippo_lti_lmu_zoh.apply(params, f=x_jnp)
    y = einops.rearrange(y, "batch seq_len -> batch 1 seq_len")

    x_tensor = einops.rearrange(x_tensor, "batch seq_len -> seq_len batch")
    _, hr_c = hr_hippo_lti_lmu_zoh(x_tensor, fast=False)
    hr_c = einops.rearrange(
        hr_c, "batch N -> batch 1 N"
    )  # add input_size dimension for comparison
    hr_y = hr_hippo_lti_lmu_zoh.reconstruct(hr_c)
    print(f"hr_y shape: {hr_y.shape}")
    hr_y = einops.rearrange(hr_y, "batch seq_len -> batch 1 seq_len")
    hr_c = jnp.asarray(hr_c, dtype=jnp.float32)  # convert torch array to jax array
    hr_y = jnp.asarray(hr_y, dtype=jnp.float32)  # convert torch array to jax array

    assert hr_c.shape == c.shape
    assert hr_y.shape == y.shape

    assert c.shape == (16, 1, 50)  # batch_size, input_size, N
    assert y.shape == (16, 1, 3000)  # batch_size, input_size, seq_len

    co_flag = True
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            co_check = jnp.allclose(c[i, j, :], hr_c[i, j, :], rtol=1e-03, atol=1e-03)
            if not co_check:
                co_flag = False
            assert jnp.allclose(y[i, j, :], hr_y[i, j, :], rtol=1e-03, atol=1e-03)
    print(f"coefficients tests: {co_flag}")


# ----------
# -- lagt --
# ----------


def test_hippo_lagt_lti_zoh_reconstruction(
    hippo_lti_lagt_zoh, hr_hippo_lti_lagt_zoh, random_16_input, lagt_key
):
    print("\nHiPPO RECONSTRUCTION LAGT (LTI, ZOH)")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    assert x_jnp.shape == (16, 3000)  # batch_size, seq_len

    x_jnp = einops.rearrange(
        x_jnp, "batch seq_len -> batch seq_len 1"
    )  # add input_size dimension

    params = hippo_lti_lagt_zoh.init(lagt_key, f=x_jnp)
    c, y = hippo_lti_lagt_zoh.apply(params, f=x_jnp)
    y = einops.rearrange(y, "batch seq_len -> batch 1 seq_len")

    x_tensor = einops.rearrange(x_tensor, "batch seq_len -> seq_len batch")
    _, hr_c = hr_hippo_lti_lagt_zoh(x_tensor, fast=False)
    hr_c = einops.rearrange(
        hr_c, "batch N -> batch 1 N"
    )  # add input_size dimension for comparison
    hr_y = hr_hippo_lti_lagt_zoh.reconstruct(hr_c)
    print(f"hr_y shape: {hr_y.shape}")
    hr_y = einops.rearrange(hr_y, "batch seq_len -> batch 1 seq_len")
    hr_c = jnp.asarray(hr_c, dtype=jnp.float32)  # convert torch array to jax array
    hr_y = jnp.asarray(hr_y, dtype=jnp.float32)  # convert torch array to jax array

    assert hr_c.shape == c.shape
    assert hr_y.shape == y.shape

    assert c.shape == (16, 1, 50)  # batch_size, input_size, N
    assert y.shape == (16, 1, 3000)  # batch_size, input_size, seq_len

    co_flag = True
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            co_check = jnp.allclose(c[i, j, :], hr_c[i, j, :], rtol=1e-03, atol=1e-03)
            if not co_check:
                co_flag = False
            assert jnp.allclose(y[i, j, :], hr_y[i, j, :], rtol=1e-03, atol=1e-03)
    print(f"coefficients tests: {co_flag}")


# ------------
# --- fout ---
# ------------


def test_hippo_fout_lti_zoh_reconstruction(
    hippo_lti_fout_zoh, hr_hippo_lti_fout_zoh, random_16_input, fout_key
):
    print("\nHiPPO RECONSTRUCTION FOUT (LTI, ZOH)")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    assert x_jnp.shape == (16, 3000)  # batch_size, seq_len

    x_jnp = einops.rearrange(
        x_jnp, "batch seq_len -> batch seq_len 1"
    )  # add input_size dimension

    params = hippo_lti_fout_zoh.init(fout_key, f=x_jnp)
    c, y = hippo_lti_fout_zoh.apply(params, f=x_jnp)
    y = einops.rearrange(y, "batch seq_len -> batch 1 seq_len")

    x_tensor = einops.rearrange(x_tensor, "batch seq_len -> seq_len batch")
    _, hr_c = hr_hippo_lti_fout_zoh(x_tensor, fast=False)
    hr_c = einops.rearrange(
        hr_c, "batch N -> batch 1 N"
    )  # add input_size dimension for comparison
    hr_y = hr_hippo_lti_fout_zoh.reconstruct(hr_c)
    print(f"hr_y shape: {hr_y.shape}")
    hr_y = einops.rearrange(hr_y, "batch seq_len -> batch 1 seq_len")
    hr_c = jnp.asarray(hr_c, dtype=jnp.float32)  # convert torch array to jax array
    hr_y = jnp.asarray(hr_y, dtype=jnp.float32)  # convert torch array to jax array

    assert hr_c.shape == c.shape
    assert hr_y.shape == y.shape

    assert c.shape == (16, 1, 50)  # batch_size, input_size, N
    assert y.shape == (16, 1, 3000)  # batch_size, input_size, seq_len

    co_flag = True
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            co_check = jnp.allclose(c[i, j, :], hr_c[i, j, :], rtol=1e-03, atol=1e-03)
            if not co_check:
                co_flag = False
            assert jnp.allclose(y[i, j, :], hr_y[i, j, :], rtol=1e-03, atol=1e-03)
    print(f"coefficients tests: {co_flag}")
