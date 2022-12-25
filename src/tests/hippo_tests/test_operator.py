import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

# implementation of HiPPO Operators
# Gu's implementation of HiPPO Operators
from src.tests.hippo_tests.hippo_operator import (
    gu_hippo_lsi_legs_be,
    gu_hippo_lsi_legs_bi,
    gu_hippo_lsi_legs_fe,
    gu_hippo_lsi_legs_zoh,
    gu_hippo_lti_foud_be,
    gu_hippo_lti_foud_bi,
    gu_hippo_lti_foud_fe,
    gu_hippo_lti_foud_zoh,
    gu_hippo_lti_fout_be,
    gu_hippo_lti_fout_bi,
    gu_hippo_lti_fout_fe,
    gu_hippo_lti_fout_zoh,
    gu_hippo_lti_fru_be,
    gu_hippo_lti_fru_bi,
    gu_hippo_lti_fru_fe,
    gu_hippo_lti_fru_zoh,
    gu_hippo_lti_lagt_be,
    gu_hippo_lti_lagt_bi,
    gu_hippo_lti_lagt_fe,
    gu_hippo_lti_lagt_zoh,
    gu_hippo_lti_legs_be,
    gu_hippo_lti_legs_bi,
    gu_hippo_lti_legs_fe,
    gu_hippo_lti_legs_zoh,
    gu_hippo_lti_legt_be,
    gu_hippo_lti_legt_bi,
    gu_hippo_lti_legt_fe,
    gu_hippo_lti_legt_zoh,
    gu_hippo_lti_lmu_be,
    gu_hippo_lti_lmu_bi,
    gu_hippo_lti_lmu_fe,
    gu_hippo_lti_lmu_zoh,
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
    gu_dplr_foud,
    gu_dplr_fout,
    gu_dplr_fru,
    gu_dplr_lagt,
    gu_dplr_legs,
    gu_dplr_legt,
    gu_dplr_lmu,
    gu_foud_matrices,
    gu_fout_matrices,
    gu_fru_matrices,
    gu_lagt_matrices,
    gu_legs_matrices,
    gu_legt_lmu_matrices,
    gu_legt_matrices,
    gu_nplr_foud,
    gu_nplr_fout,
    gu_nplr_fru,
    gu_nplr_lagt,
    gu_nplr_legs,
    gu_nplr_legt,
    gu_nplr_lmu,
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
# --------------------- Test HiPPO Operators ---------------------
# ----------------------------------------------------------------

# --------------------
# -- Forward Euler --
# --------------------

# ----------
# -- legs --
# ----------


def test_hippo_legs_lti_fe_operator(
    hippo_lti_legs_fe, gu_hippo_lti_legs_fe, random_16_input, legs_key
):
    print("HiPPO OPERATOR LEGS")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    params = hippo_lti_legs_fe.init(legs_key, f=x_jnp)
    c_k, y_k_list = hippo_lti_legs_fe.apply(params, f=x_jnp)

    x_tensor = torch.moveaxis(x_tensor, 0, 1)
    GU_c_k = gu_hippo_lti_legs_fe(x_tensor, fast=False)
    gu_c = jnp.asarray(GU_c_k, dtype=jnp.float32)  # convert torch array to jax array

    assert gu_c.shape == c_k.shape
    assert c_k.shape[0] == 16
    assert c_k.shape[1] == 512
    assert c_k.shape[2] == 1
    assert c_k.shape[3] == 100

    for i in range(c_k.shape[0]):
        for j in range(c_k.shape[1]):
            assert jnp.allclose(
                c_k[i, j, :, :], gu_c[i, j, :, :], rtol=1e-03, atol=1e-03
            )


def test_hippo_legs_lsi_fe_operator(
    hippo_lsi_legs_fe, gu_hippo_lsi_legs_fe, random_16_input, legs_key
):
    print("HiPPO OPERATOR LEGS")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    params = hippo_lsi_legs_fe.init(legs_key, f=x_jnp)
    c_k, y_k_list = hippo_lsi_legs_fe.apply(params, f=x_jnp)

    x_tensor = torch.moveaxis(x_tensor, 0, 1)
    GU_c_k = gu_hippo_lsi_legs_fe(x_tensor, fast=False)
    gu_c = jnp.asarray(GU_c_k, dtype=jnp.float32)  # convert torch array to jax array

    c_k = jnp.moveaxis(c_k, 0, 1)
    gu_c = jnp.moveaxis(gu_c, 0, 1)

    assert gu_c.shape == c_k.shape
    assert c_k.shape[0] == 16
    assert c_k.shape[1] == 512
    assert c_k.shape[2] == 1
    assert c_k.shape[3] == 100

    for i in range(c_k.shape[0]):
        for j in range(c_k.shape[1]):
            assert jnp.allclose(
                c_k[i, j, :, :], gu_c[i, j, :, :], rtol=1e-03, atol=1e-03
            )


# ----------
# -- legt --
# ----------


def test_hippo_legt_lti_fe_operator(
    hippo_lti_legt_fe, gu_hippo_lti_legt_fe, random_16_input, legt_key
):
    print("HiPPO OPERATOR LEGT")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    params = hippo_lti_legt_fe.init(legt_key, f=x_jnp)
    c_k, y_k_list = hippo_lti_legt_fe.apply(params, f=x_jnp)

    x_tensor = torch.moveaxis(x_tensor, 0, 1)
    GU_c_k = gu_hippo_lti_legt_fe(x_tensor, fast=False)
    gu_c = jnp.asarray(GU_c_k, dtype=jnp.float32)  # convert torch array to jax array

    assert gu_c.shape == c_k.shape
    assert c_k.shape[0] == 16
    assert c_k.shape[1] == 512
    assert c_k.shape[2] == 1
    assert c_k.shape[3] == 100

    for i in range(c_k.shape[0]):
        for j in range(c_k.shape[1]):
            assert jnp.allclose(
                c_k[i, j, :, :], gu_c[i, j, :, :], rtol=1e-03, atol=1e-03
            )


# ----------
# -- lmu --
# ----------


def test_hippo_lmu_lti_fe_operator(
    hippo_lti_lmu_fe, gu_hippo_lti_lmu_fe, random_16_input, lmu_key
):
    print("HiPPO OPERATOR LMU")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    params = hippo_lti_lmu_fe.init(lmu_key, f=x_jnp)
    c_k, y_k_list = hippo_lti_lmu_fe.apply(params, f=x_jnp)

    x_tensor = torch.moveaxis(x_tensor, 0, 1)
    GU_c_k = gu_hippo_lti_lmu_fe(x_tensor, fast=False)
    gu_c = jnp.asarray(GU_c_k, dtype=jnp.float32)  # convert torch array to jax array

    assert gu_c.shape == c_k.shape
    assert c_k.shape[0] == 16
    assert c_k.shape[1] == 512
    assert c_k.shape[2] == 1
    assert c_k.shape[3] == 100

    for i in range(c_k.shape[0]):
        for j in range(c_k.shape[1]):
            assert jnp.allclose(
                c_k[i, j, :, :], gu_c[i, j, :, :], rtol=1e-03, atol=1e-03
            )


# ----------
# -- lagt --
# ----------


def test_hippo_lagt_lti_fe_operator(
    hippo_lti_lagt_fe, gu_hippo_lti_lagt_fe, random_16_input, lagt_key
):
    print("HiPPO OPERATOR LAGT")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    params = hippo_lti_lagt_fe.init(lagt_key, f=x_jnp)
    c_k, y_k_list = hippo_lti_lagt_fe.apply(params, f=x_jnp)

    x_tensor = torch.moveaxis(x_tensor, 0, 1)
    GU_c_k = gu_hippo_lti_lagt_fe(x_tensor, fast=False)
    gu_c = jnp.asarray(GU_c_k, dtype=jnp.float32)  # convert torch array to jax array

    assert gu_c.shape == c_k.shape
    assert c_k.shape[0] == 16
    assert c_k.shape[1] == 512
    assert c_k.shape[2] == 1
    assert c_k.shape[3] == 100

    for i in range(c_k.shape[0]):
        for j in range(c_k.shape[1]):
            assert jnp.allclose(
                c_k[i, j, :, :], gu_c[i, j, :, :], rtol=1e-03, atol=1e-03
            )


# ----------
# --- fru --
# ----------


def test_hippo_fru_lti_fe_operator(
    hippo_lti_fru_fe, gu_hippo_lti_fru_fe, random_16_input, fru_key
):
    print("HiPPO OPERATOR FRU")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    params = hippo_lti_fru_fe.init(fru_key, f=x_jnp)
    c_k, y_k_list = hippo_lti_fru_fe.apply(params, f=x_jnp)

    x_tensor = torch.moveaxis(x_tensor, 0, 1)
    GU_c_k = gu_hippo_lti_fru_fe(x_tensor, fast=False)
    gu_c = jnp.asarray(GU_c_k, dtype=jnp.float32)  # convert torch array to jax array

    assert gu_c.shape == c_k.shape
    assert c_k.shape[0] == 16
    assert c_k.shape[1] == 512
    assert c_k.shape[2] == 1
    assert c_k.shape[3] == 100

    for i in range(c_k.shape[0]):
        for j in range(c_k.shape[1]):
            assert jnp.allclose(
                c_k[i, j, :, :], gu_c[i, j, :, :], rtol=1e-03, atol=1e-03
            )


# ------------
# --- fout ---
# ------------


def test_hippo_fout_lti_fe_operator(
    hippo_lti_fout_fe, gu_hippo_lti_fout_fe, random_16_input, fout_key
):
    print("HiPPO OPERATOR FOUT")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    params = hippo_lti_fout_fe.init(fout_key, f=x_jnp)
    c_k, y_k_list = hippo_lti_fout_fe.apply(params, f=x_jnp)

    x_tensor = torch.moveaxis(x_tensor, 0, 1)
    GU_c_k = gu_hippo_lti_fout_fe(x_tensor, fast=False)
    gu_c = jnp.asarray(GU_c_k, dtype=jnp.float32)  # convert torch array to jax array

    assert gu_c.shape == c_k.shape
    assert c_k.shape[0] == 16
    assert c_k.shape[1] == 512
    assert c_k.shape[2] == 1
    assert c_k.shape[3] == 100

    for i in range(c_k.shape[0]):
        for j in range(c_k.shape[1]):
            assert jnp.allclose(
                c_k[i, j, :, :], gu_c[i, j, :, :], rtol=1e-03, atol=1e-03
            )


# ------------
# --- foud ---
# ------------


def test_hippo_foud_lti_fe_operator(
    hippo_lti_foud_fe, gu_hippo_lti_foud_fe, random_16_input, foud_key
):
    print("HiPPO OPERATOR FOUD")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    params = hippo_lti_foud_fe.init(foud_key, f=x_jnp)
    c_k, y_k_list = hippo_lti_foud_fe.apply(params, f=x_jnp)

    x_tensor = torch.moveaxis(x_tensor, 0, 1)
    GU_c_k = gu_hippo_lti_foud_fe(x_tensor, fast=False)
    gu_c = jnp.asarray(GU_c_k, dtype=jnp.float32)  # convert torch array to jax array

    assert gu_c.shape == c_k.shape
    assert c_k.shape[0] == 16
    assert c_k.shape[1] == 512
    assert c_k.shape[2] == 1
    assert c_k.shape[3] == 100

    for i in range(c_k.shape[0]):
        for j in range(c_k.shape[1]):
            assert jnp.allclose(
                c_k[i, j, :, :], gu_c[i, j, :, :], rtol=1e-03, atol=1e-03
            )


# --------------------
# -- Backward Euler --
# --------------------

# ----------
# -- legs --
# ----------


def test_hippo_legs_lti_be_operator(
    hippo_lti_legs_be, gu_hippo_lti_legs_be, random_16_input, legs_key
):
    print("HiPPO OPERATOR LEGS")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    params = hippo_lti_legs_be.init(legs_key, f=x_jnp)
    c_k, y_k_list = hippo_lti_legs_be.apply(params, f=x_jnp)

    x_tensor = torch.moveaxis(x_tensor, 0, 1)
    GU_c_k = gu_hippo_lti_legs_be(x_tensor, fast=False)
    gu_c = jnp.asarray(GU_c_k, dtype=jnp.float32)  # convert torch array to jax array

    assert gu_c.shape == c_k.shape
    assert c_k.shape[0] == 16
    assert c_k.shape[1] == 512
    assert c_k.shape[2] == 1
    assert c_k.shape[3] == 100

    for i in range(c_k.shape[0]):
        for j in range(c_k.shape[1]):
            assert jnp.allclose(
                c_k[i, j, :, :], gu_c[i, j, :, :], rtol=1e-03, atol=1e-03
            )


def test_hippo_legs_lsi_be_operator(
    hippo_lsi_legs_be, gu_hippo_lsi_legs_be, random_16_input, legs_key
):
    print("HiPPO OPERATOR LEGS")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    params = hippo_lsi_legs_be.init(legs_key, f=x_jnp)
    c_k, y_k_list = hippo_lsi_legs_be.apply(params, f=x_jnp)

    x_tensor = torch.moveaxis(x_tensor, 0, 1)
    GU_c_k = gu_hippo_lsi_legs_be(x_tensor, fast=False)
    gu_c = jnp.asarray(GU_c_k, dtype=jnp.float32)  # convert torch array to jax array

    c_k = jnp.moveaxis(c_k, 0, 1)
    gu_c = jnp.moveaxis(gu_c, 0, 1)

    assert gu_c.shape == c_k.shape
    assert c_k.shape[0] == 16
    assert c_k.shape[1] == 512
    assert c_k.shape[2] == 1
    assert c_k.shape[3] == 100

    for i in range(c_k.shape[0]):
        for j in range(c_k.shape[1]):
            assert jnp.allclose(
                c_k[i, j, :, :], gu_c[i, j, :, :], rtol=1e-03, atol=1e-03
            )


# ----------
# -- legt --
# ----------


def test_hippo_legt_lti_be_operator(
    hippo_lti_legt_be, gu_hippo_lti_legt_be, random_16_input, legt_key
):
    print("HiPPO OPERATOR LEGT")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    params = hippo_lti_legt_be.init(legt_key, f=x_jnp)
    c_k, y_k_list = hippo_lti_legt_be.apply(params, f=x_jnp)

    x_tensor = torch.moveaxis(x_tensor, 0, 1)
    GU_c_k = gu_hippo_lti_legt_be(x_tensor, fast=False)
    gu_c = jnp.asarray(GU_c_k, dtype=jnp.float32)  # convert torch array to jax array

    assert gu_c.shape == c_k.shape
    assert c_k.shape[0] == 16
    assert c_k.shape[1] == 512
    assert c_k.shape[2] == 1
    assert c_k.shape[3] == 100

    for i in range(c_k.shape[0]):
        for j in range(c_k.shape[1]):
            assert jnp.allclose(
                c_k[i, j, :, :], gu_c[i, j, :, :], rtol=1e-03, atol=1e-03
            )


# ----------
# -- lmu --
# ----------


def test_hippo_lmu_lti_be_operator(
    hippo_lti_lmu_be, gu_hippo_lti_lmu_be, random_16_input, lmu_key
):
    print("HiPPO OPERATOR LMU")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    params = hippo_lti_lmu_be.init(lmu_key, f=x_jnp)
    c_k, y_k_list = hippo_lti_lmu_be.apply(params, f=x_jnp)

    x_tensor = torch.moveaxis(x_tensor, 0, 1)
    GU_c_k = gu_hippo_lti_lmu_be(x_tensor, fast=False)
    gu_c = jnp.asarray(GU_c_k, dtype=jnp.float32)  # convert torch array to jax array

    assert gu_c.shape == c_k.shape
    assert c_k.shape[0] == 16
    assert c_k.shape[1] == 512
    assert c_k.shape[2] == 1
    assert c_k.shape[3] == 100

    for i in range(c_k.shape[0]):
        for j in range(c_k.shape[1]):
            assert jnp.allclose(
                c_k[i, j, :, :], gu_c[i, j, :, :], rtol=1e-03, atol=1e-03
            )


# ----------
# -- lagt --
# ----------


def test_hippo_lagt_lti_be_operator(
    hippo_lti_lagt_be, gu_hippo_lti_lagt_be, random_16_input, lagt_key
):
    print("HiPPO OPERATOR LAGT")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    params = hippo_lti_lagt_be.init(lagt_key, f=x_jnp)
    c_k, y_k_list = hippo_lti_lagt_be.apply(params, f=x_jnp)

    x_tensor = torch.moveaxis(x_tensor, 0, 1)
    GU_c_k = gu_hippo_lti_lagt_be(x_tensor, fast=False)
    gu_c = jnp.asarray(GU_c_k, dtype=jnp.float32)  # convert torch array to jax array

    assert gu_c.shape == c_k.shape
    assert c_k.shape[0] == 16
    assert c_k.shape[1] == 512
    assert c_k.shape[2] == 1
    assert c_k.shape[3] == 100

    for i in range(c_k.shape[0]):
        for j in range(c_k.shape[1]):
            assert jnp.allclose(
                c_k[i, j, :, :], gu_c[i, j, :, :], rtol=1e-03, atol=1e-03
            )


# ----------
# --- fru --
# ----------


def test_hippo_fru_lti_be_operator(
    hippo_lti_fru_be, gu_hippo_lti_fru_be, random_16_input, fru_key
):
    print("HiPPO OPERATOR FRU")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    params = hippo_lti_fru_be.init(fru_key, f=x_jnp)
    c_k, y_k_list = hippo_lti_fru_be.apply(params, f=x_jnp)

    x_tensor = torch.moveaxis(x_tensor, 0, 1)
    GU_c_k = gu_hippo_lti_fru_be(x_tensor, fast=False)
    gu_c = jnp.asarray(GU_c_k, dtype=jnp.float32)  # convert torch array to jax array

    assert gu_c.shape == c_k.shape
    assert c_k.shape[0] == 16
    assert c_k.shape[1] == 512
    assert c_k.shape[2] == 1
    assert c_k.shape[3] == 100

    for i in range(c_k.shape[0]):
        for j in range(c_k.shape[1]):
            assert jnp.allclose(
                c_k[i, j, :, :], gu_c[i, j, :, :], rtol=1e-03, atol=1e-03
            )


# ------------
# --- fout ---
# ------------


def test_hippo_fout_lti_be_operator(
    hippo_lti_fout_be, gu_hippo_lti_fout_be, random_16_input, fout_key
):
    print("HiPPO OPERATOR FOUT")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    params = hippo_lti_fout_be.init(fout_key, f=x_jnp)
    c_k, y_k_list = hippo_lti_fout_be.apply(params, f=x_jnp)

    x_tensor = torch.moveaxis(x_tensor, 0, 1)
    GU_c_k = gu_hippo_lti_fout_be(x_tensor, fast=False)
    gu_c = jnp.asarray(GU_c_k, dtype=jnp.float32)  # convert torch array to jax array

    assert gu_c.shape == c_k.shape
    assert c_k.shape[0] == 16
    assert c_k.shape[1] == 512
    assert c_k.shape[2] == 1
    assert c_k.shape[3] == 100

    for i in range(c_k.shape[0]):
        for j in range(c_k.shape[1]):
            assert jnp.allclose(
                c_k[i, j, :, :], gu_c[i, j, :, :], rtol=1e-03, atol=1e-03
            )


# ------------
# --- foud ---
# ------------


def test_hippo_foud_lti_be_operator(
    hippo_lti_foud_be, gu_hippo_lti_foud_be, random_16_input, foud_key
):
    print("HiPPO OPERATOR FOUD")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    params = hippo_lti_foud_be.init(foud_key, f=x_jnp)
    c_k, y_k_list = hippo_lti_foud_be.apply(params, f=x_jnp)

    x_tensor = torch.moveaxis(x_tensor, 0, 1)
    GU_c_k = gu_hippo_lti_foud_be(x_tensor, fast=False)
    gu_c = jnp.asarray(GU_c_k, dtype=jnp.float32)  # convert torch array to jax array

    assert gu_c.shape == c_k.shape
    assert c_k.shape[0] == 16
    assert c_k.shape[1] == 512
    assert c_k.shape[2] == 1
    assert c_k.shape[3] == 100

    for i in range(c_k.shape[0]):
        for j in range(c_k.shape[1]):
            assert jnp.allclose(
                c_k[i, j, :, :], gu_c[i, j, :, :], rtol=1e-03, atol=1e-03
            )


# --------------------
# ----- Bilinear -----
# --------------------
# ----------
# -- legs --
# ----------


def test_hippo_legs_lti_bi_operator(
    hippo_lti_legs_bi, gu_hippo_lti_legs_bi, random_16_input, legs_key
):
    print("HiPPO OPERATOR LEGS")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    params = hippo_lti_legs_bi.init(legs_key, f=x_jnp)
    c_k, y_k_list = hippo_lti_legs_bi.apply(params, f=x_jnp)

    x_tensor = torch.moveaxis(x_tensor, 0, 1)
    GU_c_k = gu_hippo_lti_legs_bi(x_tensor, fast=False)
    gu_c = jnp.asarray(GU_c_k, dtype=jnp.float32)  # convert torch array to jax array

    assert gu_c.shape == c_k.shape
    assert c_k.shape[0] == 16
    assert c_k.shape[1] == 512
    assert c_k.shape[2] == 1
    assert c_k.shape[3] == 100

    for i in range(c_k.shape[0]):
        for j in range(c_k.shape[1]):
            assert jnp.allclose(
                c_k[i, j, :, :], gu_c[i, j, :, :], rtol=1e-03, atol=1e-03
            )


def test_hippo_legs_lsi_bi_operator(
    hippo_lsi_legs_bi, gu_hippo_lsi_legs_bi, random_16_input, legs_key
):
    print("HiPPO OPERATOR LEGS")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    params = hippo_lsi_legs_bi.init(legs_key, f=x_jnp)
    c_k, y_k_list = hippo_lsi_legs_bi.apply(params, f=x_jnp)

    x_tensor = torch.moveaxis(x_tensor, 0, 1)
    GU_c_k = gu_hippo_lsi_legs_bi(x_tensor, fast=False)
    gu_c = jnp.asarray(GU_c_k, dtype=jnp.float32)  # convert torch array to jax array

    c_k = jnp.moveaxis(c_k, 0, 1)
    gu_c = jnp.moveaxis(gu_c, 0, 1)

    assert gu_c.shape == c_k.shape
    assert c_k.shape[0] == 16
    assert c_k.shape[1] == 512
    assert c_k.shape[2] == 1
    assert c_k.shape[3] == 100

    for i in range(c_k.shape[0]):
        for j in range(c_k.shape[1]):
            assert jnp.allclose(
                c_k[i, j, :, :], gu_c[i, j, :, :], rtol=1e-03, atol=1e-03
            )


# ----------
# -- legt --
# ----------


def test_hippo_legt_lti_bi_operator(
    hippo_lti_legt_bi, gu_hippo_lti_legt_bi, random_16_input, legt_key
):
    print("HiPPO OPERATOR LEGT")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    params = hippo_lti_legt_bi.init(legt_key, f=x_jnp)
    c_k, y_k_list = hippo_lti_legt_bi.apply(params, f=x_jnp)

    x_tensor = torch.moveaxis(x_tensor, 0, 1)
    GU_c_k = gu_hippo_lti_legt_bi(x_tensor, fast=False)
    gu_c = jnp.asarray(GU_c_k, dtype=jnp.float32)  # convert torch array to jax array

    assert gu_c.shape == c_k.shape
    assert c_k.shape[0] == 16
    assert c_k.shape[1] == 512
    assert c_k.shape[2] == 1
    assert c_k.shape[3] == 100

    for i in range(c_k.shape[0]):
        for j in range(c_k.shape[1]):
            assert jnp.allclose(
                c_k[i, j, :, :], gu_c[i, j, :, :], rtol=1e-03, atol=1e-03
            )


# ----------
# -- lmu --
# ----------


def test_hippo_lmu_lti_bi_operator(
    hippo_lti_lmu_bi, gu_hippo_lti_lmu_bi, random_16_input, lmu_key
):
    print("HiPPO OPERATOR LMU")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    params = hippo_lti_lmu_bi.init(lmu_key, f=x_jnp)
    c_k, y_k_list = hippo_lti_lmu_bi.apply(params, f=x_jnp)

    x_tensor = torch.moveaxis(x_tensor, 0, 1)
    GU_c_k = gu_hippo_lti_lmu_bi(x_tensor, fast=False)
    gu_c = jnp.asarray(GU_c_k, dtype=jnp.float32)  # convert torch array to jax array

    assert gu_c.shape == c_k.shape
    assert c_k.shape[0] == 16
    assert c_k.shape[1] == 512
    assert c_k.shape[2] == 1
    assert c_k.shape[3] == 100

    for i in range(c_k.shape[0]):
        for j in range(c_k.shape[1]):
            assert jnp.allclose(
                c_k[i, j, :, :], gu_c[i, j, :, :], rtol=1e-03, atol=1e-03
            )


# ----------
# -- lagt --
# ----------


def test_hippo_lagt_lti_bi_operator(
    hippo_lti_lagt_bi, gu_hippo_lti_lagt_bi, random_16_input, lagt_key
):
    print("HiPPO OPERATOR LAGT")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    params = hippo_lti_lagt_bi.init(lagt_key, f=x_jnp)
    c_k, y_k_list = hippo_lti_lagt_bi.apply(params, f=x_jnp)

    x_tensor = torch.moveaxis(x_tensor, 0, 1)
    GU_c_k = gu_hippo_lti_lagt_bi(x_tensor, fast=False)
    gu_c = jnp.asarray(GU_c_k, dtype=jnp.float32)  # convert torch array to jax array

    assert gu_c.shape == c_k.shape
    assert c_k.shape[0] == 16
    assert c_k.shape[1] == 512
    assert c_k.shape[2] == 1
    assert c_k.shape[3] == 100

    for i in range(c_k.shape[0]):
        for j in range(c_k.shape[1]):
            assert jnp.allclose(
                c_k[i, j, :, :], gu_c[i, j, :, :], rtol=1e-03, atol=1e-03
            )


# ----------
# --- fru --
# ----------


def test_hippo_fru_lti_bi_operator(
    hippo_lti_fru_bi, gu_hippo_lti_fru_bi, random_16_input, fru_key
):
    print("HiPPO OPERATOR FRU")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    params = hippo_lti_fru_bi.init(fru_key, f=x_jnp)
    c_k, y_k_list = hippo_lti_fru_bi.apply(params, f=x_jnp)

    x_tensor = torch.moveaxis(x_tensor, 0, 1)
    GU_c_k = gu_hippo_lti_fru_bi(x_tensor, fast=False)
    gu_c = jnp.asarray(GU_c_k, dtype=jnp.float32)  # convert torch array to jax array

    assert gu_c.shape == c_k.shape
    assert c_k.shape[0] == 16
    assert c_k.shape[1] == 512
    assert c_k.shape[2] == 1
    assert c_k.shape[3] == 100

    for i in range(c_k.shape[0]):
        for j in range(c_k.shape[1]):
            assert jnp.allclose(
                c_k[i, j, :, :], gu_c[i, j, :, :], rtol=1e-03, atol=1e-03
            )


# ------------
# --- fout ---
# ------------


def test_hippo_fout_lti_bi_operator(
    hippo_lti_fout_bi, gu_hippo_lti_fout_bi, random_16_input, fout_key
):
    print("HiPPO OPERATOR FOUT")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    params = hippo_lti_fout_bi.init(fout_key, f=x_jnp)
    c_k, y_k_list = hippo_lti_fout_bi.apply(params, f=x_jnp)

    x_tensor = torch.moveaxis(x_tensor, 0, 1)
    GU_c_k = gu_hippo_lti_fout_bi(x_tensor, fast=False)
    gu_c = jnp.asarray(GU_c_k, dtype=jnp.float32)  # convert torch array to jax array

    assert gu_c.shape == c_k.shape
    assert c_k.shape[0] == 16
    assert c_k.shape[1] == 512
    assert c_k.shape[2] == 1
    assert c_k.shape[3] == 100

    for i in range(c_k.shape[0]):
        for j in range(c_k.shape[1]):
            assert jnp.allclose(
                c_k[i, j, :, :], gu_c[i, j, :, :], rtol=1e-03, atol=1e-03
            )


# ------------
# --- foud ---
# ------------


def test_hippo_foud_lti_bi_operator(
    hippo_lti_foud_bi, gu_hippo_lti_foud_bi, random_16_input, foud_key
):
    print("HiPPO OPERATOR FOUD")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    params = hippo_lti_foud_bi.init(foud_key, f=x_jnp)
    c_k, y_k_list = hippo_lti_foud_bi.apply(params, f=x_jnp)

    x_tensor = torch.moveaxis(x_tensor, 0, 1)
    GU_c_k = gu_hippo_lti_foud_bi(x_tensor, fast=False)
    gu_c = jnp.asarray(GU_c_k, dtype=jnp.float32)  # convert torch array to jax array

    assert gu_c.shape == c_k.shape
    assert c_k.shape[0] == 16
    assert c_k.shape[1] == 512
    assert c_k.shape[2] == 1
    assert c_k.shape[3] == 100

    for i in range(c_k.shape[0]):
        for j in range(c_k.shape[1]):
            assert jnp.allclose(
                c_k[i, j, :, :], gu_c[i, j, :, :], rtol=1e-03, atol=1e-03
            )


# --------------------
# - Zero-order Hold --
# --------------------

# ----------
# -- legs --
# ----------


def test_hippo_legs_lti_zoh_operator(
    hippo_lti_legs_zoh, gu_hippo_lti_legs_zoh, random_16_input, legs_key
):
    print("HiPPO OPERATOR LEGS")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    params = hippo_lti_legs_zoh.init(legs_key, f=x_jnp)
    c_k, y_k_list = hippo_lti_legs_zoh.apply(params, f=x_jnp)

    x_tensor = torch.moveaxis(x_tensor, 0, 1)
    GU_c_k = gu_hippo_lti_legs_zoh(x_tensor, fast=False)
    gu_c = jnp.asarray(GU_c_k, dtype=jnp.float32)  # convert torch array to jax array

    assert gu_c.shape == c_k.shape
    assert c_k.shape[0] == 16
    assert c_k.shape[1] == 512
    assert c_k.shape[2] == 1
    assert c_k.shape[3] == 100

    for i in range(c_k.shape[0]):
        for j in range(c_k.shape[1]):
            assert jnp.allclose(
                c_k[i, j, :, :], gu_c[i, j, :, :], rtol=1e-03, atol=1e-03
            )


def test_hippo_legs_lsi_zoh_operator(
    hippo_lsi_legs_zoh, gu_hippo_lsi_legs_zoh, random_16_input, legs_key
):
    print("HiPPO OPERATOR LEGS")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    params = hippo_lsi_legs_zoh.init(legs_key, f=x_jnp)
    c_k, y_k_list = hippo_lsi_legs_zoh.apply(params, f=x_jnp)

    x_tensor = torch.moveaxis(x_tensor, 0, 1)
    GU_c_k = gu_hippo_lsi_legs_zoh(x_tensor, fast=False)
    gu_c = jnp.asarray(GU_c_k, dtype=jnp.float32)  # convert torch array to jax array

    c_k = jnp.moveaxis(c_k, 0, 1)
    gu_c = jnp.moveaxis(gu_c, 0, 1)

    assert gu_c.shape == c_k.shape
    assert c_k.shape[0] == 16
    assert c_k.shape[1] == 512
    assert c_k.shape[2] == 1
    assert c_k.shape[3] == 100

    for i in range(c_k.shape[0]):
        for j in range(c_k.shape[1]):
            assert jnp.allclose(
                c_k[i, j, :, :], gu_c[i, j, :, :], rtol=1e-03, atol=1e-03
            )


# ----------
# -- legt --
# ----------


def test_hippo_legt_lti_zoh_operator(
    hippo_lti_legt_zoh, gu_hippo_lti_legt_zoh, random_16_input, legt_key
):
    print("HiPPO OPERATOR LEGT")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    params = hippo_lti_legt_zoh.init(legt_key, f=x_jnp)
    c_k, y_k_list = hippo_lti_legt_zoh.apply(params, f=x_jnp)

    x_tensor = torch.moveaxis(x_tensor, 0, 1)
    GU_c_k = gu_hippo_lti_legt_zoh(x_tensor, fast=False)
    gu_c = jnp.asarray(GU_c_k, dtype=jnp.float32)  # convert torch array to jax array

    assert gu_c.shape == c_k.shape
    assert c_k.shape[0] == 16
    assert c_k.shape[1] == 512
    assert c_k.shape[2] == 1
    assert c_k.shape[3] == 100

    for i in range(c_k.shape[0]):
        for j in range(c_k.shape[1]):
            assert jnp.allclose(
                c_k[i, j, :, :], gu_c[i, j, :, :], rtol=1e-03, atol=1e-03
            )


# ----------
# -- lmu --
# ----------


def test_hippo_lmu_lti_zoh_operator(
    hippo_lti_lmu_zoh, gu_hippo_lti_lmu_zoh, random_16_input, lmu_key
):
    print("HiPPO OPERATOR LMU")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    params = hippo_lti_lmu_zoh.init(lmu_key, f=x_jnp)
    c_k, y_k_list = hippo_lti_lmu_zoh.apply(params, f=x_jnp)

    x_tensor = torch.moveaxis(x_tensor, 0, 1)
    GU_c_k = gu_hippo_lti_lmu_zoh(x_tensor, fast=False)
    gu_c = jnp.asarray(GU_c_k, dtype=jnp.float32)  # convert torch array to jax array

    assert gu_c.shape == c_k.shape
    assert c_k.shape[0] == 16
    assert c_k.shape[1] == 512
    assert c_k.shape[2] == 1
    assert c_k.shape[3] == 100

    for i in range(c_k.shape[0]):
        for j in range(c_k.shape[1]):
            assert jnp.allclose(
                c_k[i, j, :, :], gu_c[i, j, :, :], rtol=1e-03, atol=1e-03
            )


# ----------
# -- lagt --
# ----------


def test_hippo_lagt_lti_zoh_operator(
    hippo_lti_lagt_zoh, gu_hippo_lti_lagt_zoh, random_16_input, lagt_key
):
    print("HiPPO OPERATOR LAGT")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    params = hippo_lti_lagt_zoh.init(lagt_key, f=x_jnp)
    c_k, y_k_list = hippo_lti_lagt_zoh.apply(params, f=x_jnp)

    x_tensor = torch.moveaxis(x_tensor, 0, 1)
    GU_c_k = gu_hippo_lti_lagt_zoh(x_tensor, fast=False)
    gu_c = jnp.asarray(GU_c_k, dtype=jnp.float32)  # convert torch array to jax array

    assert gu_c.shape == c_k.shape
    assert c_k.shape[0] == 16
    assert c_k.shape[1] == 512
    assert c_k.shape[2] == 1
    assert c_k.shape[3] == 100

    for i in range(c_k.shape[0]):
        for j in range(c_k.shape[1]):
            assert jnp.allclose(
                c_k[i, j, :, :], gu_c[i, j, :, :], rtol=1e-03, atol=1e-03
            )


# ----------
# --- fru --
# ----------


def test_hippo_fru_lti_zoh_operator(
    hippo_lti_fru_zoh, gu_hippo_lti_fru_zoh, random_16_input, fru_key
):
    print("HiPPO OPERATOR FRU")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    params = hippo_lti_fru_zoh.init(fru_key, f=x_jnp)
    c_k, y_k_list = hippo_lti_fru_zoh.apply(params, f=x_jnp)

    x_tensor = torch.moveaxis(x_tensor, 0, 1)
    GU_c_k = gu_hippo_lti_fru_zoh(x_tensor, fast=False)
    gu_c = jnp.asarray(GU_c_k, dtype=jnp.float32)  # convert torch array to jax array

    assert gu_c.shape == c_k.shape
    assert c_k.shape[0] == 16
    assert c_k.shape[1] == 512
    assert c_k.shape[2] == 1
    assert c_k.shape[3] == 100

    for i in range(c_k.shape[0]):
        for j in range(c_k.shape[1]):
            assert jnp.allclose(
                c_k[i, j, :, :], gu_c[i, j, :, :], rtol=1e-03, atol=1e-03
            )


# ------------
# --- fout ---
# ------------


def test_hippo_fout_lti_zoh_operator(
    hippo_lti_fout_zoh, gu_hippo_lti_fout_zoh, random_16_input, fout_key
):
    print("HiPPO OPERATOR FOUT")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    params = hippo_lti_fout_zoh.init(fout_key, f=x_jnp)
    c_k, y_k_list = hippo_lti_fout_zoh.apply(params, f=x_jnp)

    x_tensor = torch.moveaxis(x_tensor, 0, 1)
    GU_c_k = gu_hippo_lti_fout_zoh(x_tensor, fast=False)
    gu_c = jnp.asarray(GU_c_k, dtype=jnp.float32)  # convert torch array to jax array

    assert gu_c.shape == c_k.shape
    assert c_k.shape[0] == 16
    assert c_k.shape[1] == 512
    assert c_k.shape[2] == 1
    assert c_k.shape[3] == 100

    for i in range(c_k.shape[0]):
        for j in range(c_k.shape[1]):
            assert jnp.allclose(
                c_k[i, j, :, :], gu_c[i, j, :, :], rtol=1e-03, atol=1e-03
            )


# ------------
# --- foud ---
# ------------


def test_hippo_foud_lti_zoh_operator(
    hippo_lti_foud_zoh, gu_hippo_lti_foud_zoh, random_16_input, foud_key
):
    print("HiPPO OPERATOR FOUD")
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    x_jnp = jnp.asarray(x_tensor, dtype=jnp.float32)  # convert torch array to jax array

    params = hippo_lti_foud_zoh.init(foud_key, f=x_jnp)
    c_k, y_k_list = hippo_lti_foud_zoh.apply(params, f=x_jnp)

    x_tensor = torch.moveaxis(x_tensor, 0, 1)
    GU_c_k = gu_hippo_lti_foud_zoh(x_tensor, fast=False)
    gu_c = jnp.asarray(GU_c_k, dtype=jnp.float32)  # convert torch array to jax array

    assert gu_c.shape == c_k.shape
    assert c_k.shape[0] == 16
    assert c_k.shape[1] == 512
    assert c_k.shape[2] == 1
    assert c_k.shape[3] == 100

    for i in range(c_k.shape[0]):
        for j in range(c_k.shape[1]):
            assert jnp.allclose(
                c_k[i, j, :, :], gu_c[i, j, :, :], rtol=1e-03, atol=1e-03
            )
