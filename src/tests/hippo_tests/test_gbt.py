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
# --------------- Test HiPPO Matrix Transformations --------------
# ----------------------------------------------------------------

# --------------------
# -- Forward Euler --
# --------------------

# ----------
# -- legs --
# ----------


def test_GBT_LSI_legs_FE(
    hippo_lsi_legs_fe, gu_hippo_lsi_legs_fe, legs_matrices, random_16_input
):
    print("\nHiPPO GBT LEGS")
    L = random_16_input.shape[1]
    alpha = 0.0
    A, B = legs_matrices

    assert random_16_input.shape[0] == 16
    assert random_16_input.shape[1] == 3000

    for i in range(1, L + 1):
        GBT_A, GBT_B = hippo_lsi_legs_fe.discretize(
            A, B, step=i, alpha=alpha, dtype=jnp.float32
        )
        gu_GBT_A, gu_GBT_B = (
            jnp.asarray(gu_hippo_lsi_legs_fe.A_stacked[i - 1], dtype=jnp.float32),
            jnp.expand_dims(
                jnp.asarray(gu_hippo_lsi_legs_fe.B_stacked[i - 1], dtype=jnp.float32),
                axis=1,
            ),
        )
        assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
        assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


def test_GBT_LTI_legs_FE(
    hippo_lti_legs_fe, gu_hippo_lti_legs_fe, legs_matrices, random_16_input
):
    print("\nHiPPO GBT LEGS")
    L = random_16_input.shape[1]
    alpha = 0.0
    A, B = legs_matrices

    GBT_A, GBT_B = hippo_lti_legs_fe.discretize(
        A, B, step=1.0, alpha=alpha, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_legs_fe.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_legs_fe.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert random_16_input.shape[0] == 16
    assert random_16_input.shape[1] == 512
    assert random_16_input.shape[2] == 1

    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ----------
# -- legt --
# ----------


def test_GBT_LTI_legt_FE(
    hippo_lti_legt_fe, gu_hippo_lti_legt_fe, legt_matrices, random_16_input
):
    print("\nHiPPO GBT LEGS")
    L = random_16_input.shape[1]
    alpha = 0.0
    A, B = legt_matrices

    GBT_A, GBT_B = hippo_lti_legt_fe.encoder.discretize(
        A, B, step=1.0, alpha=alpha, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_legt_fe.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_legt_fe.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert random_16_input.shape[0] == 16
    assert random_16_input.shape[1] == 512
    assert random_16_input.shape[2] == 1

    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ----------
# -- lmu --
# ----------


def test_GBT_LTI_lmu_FE(
    hippo_lti_lmu_fe, gu_hippo_lti_lmu_fe, legt_lmu_matrices, random_16_input
):
    print("\nHiPPO GBT LEGS")
    L = random_16_input.shape[1]
    alpha = 0.0
    A, B = legt_lmu_matrices

    GBT_A, GBT_B = hippo_lti_lmu_fe.discretize(
        A, B, step=1.0, alpha=alpha, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_lmu_fe.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_lmu_fe.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert random_16_input.shape[0] == 16
    assert random_16_input.shape[1] == 512
    assert random_16_input.shape[2] == 1

    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ----------
# -- lagt --
# ----------


def test_GBT_LTI_lagt_FE(
    hippo_lti_lagt_fe, gu_hippo_lti_lagt_fe, lagt_matrices, random_16_input
):
    print("\nHiPPO GBT LEGS")
    L = random_16_input.shape[1]
    alpha = 0.0
    A, B = lagt_matrices

    GBT_A, GBT_B = hippo_lti_lagt_fe.discretize(
        A, B, step=1.0, alpha=alpha, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_lagt_fe.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_lagt_fe.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert random_16_input.shape[0] == 16
    assert random_16_input.shape[1] == 512
    assert random_16_input.shape[2] == 1

    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ----------
# --- fru --
# ----------


def test_GBT_LTI_fru_FE(
    hippo_lti_fru_fe, gu_hippo_lti_fru_fe, fru_matrices, random_16_input
):
    print("\nHiPPO GBT LEGS")
    L = random_16_input.shape[1]
    alpha = 0.0
    A, B = fru_matrices

    GBT_A, GBT_B = hippo_lti_fru_fe.discretize(
        A, B, step=1.0, alpha=alpha, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_fru_fe.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_fru_fe.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert random_16_input.shape[0] == 16
    assert random_16_input.shape[1] == 512
    assert random_16_input.shape[2] == 1

    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ------------
# --- fout ---
# ------------


def test_GBT_LTI_fout_FE(
    hippo_lti_fout_fe, gu_hippo_lti_fout_fe, fout_matrices, random_16_input
):
    print("\nHiPPO GBT LEGS")
    L = random_16_input.shape[1]
    alpha = 0.0
    A, B = fout_matrices

    GBT_A, GBT_B = hippo_lti_fout_fe.discretize(
        A, B, step=1.0, alpha=alpha, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_fout_fe.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_fout_fe.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert random_16_input.shape[0] == 16
    assert random_16_input.shape[1] == 512
    assert random_16_input.shape[2] == 1

    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ------------
# --- foud ---
# ------------


def test_GBT_LTI_foud_FE(
    hippo_lti_foud_fe, gu_hippo_lti_foud_fe, foud_matrices, random_16_input
):
    print("\nHiPPO GBT LEGS")
    L = random_16_input.shape[1]
    alpha = 0.0
    A, B = foud_matrices

    GBT_A, GBT_B = hippo_lti_foud_fe.discretize(
        A, B, step=1.0, alpha=alpha, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_foud_fe.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_foud_fe.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert random_16_input.shape[0] == 16
    assert random_16_input.shape[1] == 512
    assert random_16_input.shape[2] == 1

    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# --------------------
# -- Backward Euler --
# --------------------

# ----------
# -- legs --
# ----------


def test_GBT_LSI_legs_BE(
    hippo_lsi_legs_be, gu_hippo_lsi_legs_be, legs_matrices, random_16_input
):
    print("\nHiPPO GBT LEGS")
    L = random_16_input.shape[1]
    alpha = 1.0
    A, B = legs_matrices

    assert random_16_input.shape[0] == 16
    assert random_16_input.shape[1] == 512
    assert random_16_input.shape[2] == 1

    for i in range(1, L + 1):
        GBT_A, GBT_B = hippo_lsi_legs_be.discretize(
            A, B, step=i, alpha=alpha, dtype=jnp.float32
        )
        gu_GBT_A, gu_GBT_B = (
            jnp.asarray(gu_hippo_lsi_legs_be.A_stacked[i - 1], dtype=jnp.float32),
            jnp.expand_dims(
                jnp.asarray(gu_hippo_lsi_legs_be.B_stacked[i - 1], dtype=jnp.float32),
                axis=1,
            ),
        )
        assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
        assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


def test_GBT_LTI_legs_BE(
    hippo_lti_legs_be, gu_hippo_lti_legs_be, legs_matrices, random_16_input
):
    print("\nHiPPO GBT LEGS")
    L = random_16_input.shape[1]
    alpha = 1.0
    A, B = legs_matrices

    GBT_A, GBT_B = hippo_lti_legs_be.discretize(
        A, B, step=1.0, alpha=alpha, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_legs_be.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_legs_be.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert random_16_input.shape[0] == 16
    assert random_16_input.shape[1] == 512
    assert random_16_input.shape[2] == 1

    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ----------
# -- legt --
# ----------


def test_GBT_LTI_legt_BE(
    hippo_lti_legt_be, gu_hippo_lti_legt_be, legt_matrices, random_16_input
):
    print("\nHiPPO GBT LEGS")
    L = random_16_input.shape[1]
    alpha = 1.0
    A, B = legt_matrices

    GBT_A, GBT_B = hippo_lti_legt_be.discretize(
        A, B, step=1.0, alpha=alpha, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_legt_be.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_legt_be.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert random_16_input.shape[0] == 16
    assert random_16_input.shape[1] == 512
    assert random_16_input.shape[2] == 1

    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ----------
# -- lmu --
# ----------


def test_GBT_LTI_lmu_BE(
    hippo_lti_lmu_be, gu_hippo_lti_lmu_be, legt_lmu_matrices, random_16_input
):
    print("\nHiPPO GBT LEGS")
    L = random_16_input.shape[1]
    alpha = 1.0
    A, B = legt_lmu_matrices

    GBT_A, GBT_B = hippo_lti_lmu_be.discretize(
        A, B, step=1.0, alpha=alpha, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_lmu_be.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_lmu_be.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert random_16_input.shape[0] == 16
    assert random_16_input.shape[1] == 512
    assert random_16_input.shape[2] == 1

    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ----------
# -- lagt --
# ----------


def test_GBT_LTI_lagt_BE(
    hippo_lti_lagt_be, gu_hippo_lti_lagt_be, lagt_matrices, random_16_input
):
    print("\nHiPPO GBT LEGS")
    L = random_16_input.shape[1]
    alpha = 1.0
    A, B = lagt_matrices

    GBT_A, GBT_B = hippo_lti_lagt_be.discretize(
        A, B, step=1.0, alpha=alpha, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_lagt_be.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_lagt_be.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert random_16_input.shape[0] == 16
    assert random_16_input.shape[1] == 512
    assert random_16_input.shape[2] == 1

    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ----------
# --- fru --
# ----------


def test_GBT_LTI_fru_BE(
    hippo_lti_fru_be, gu_hippo_lti_fru_be, fru_matrices, random_16_input
):
    print("\nHiPPO GBT LEGS")
    L = random_16_input.shape[1]
    alpha = 1.0
    A, B = fru_matrices

    GBT_A, GBT_B = hippo_lti_fru_be.discretize(
        A, B, step=1.0, alpha=alpha, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_fru_be.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_fru_be.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert random_16_input.shape[0] == 16
    assert random_16_input.shape[1] == 512
    assert random_16_input.shape[2] == 1

    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ------------
# --- fout ---
# ------------


def test_GBT_LTI_fout_BE(
    hippo_lti_fout_be, gu_hippo_lti_fout_be, fout_matrices, random_16_input
):
    print("\nHiPPO GBT LEGS")
    L = random_16_input.shape[1]
    alpha = 1.0
    A, B = fout_matrices

    GBT_A, GBT_B = hippo_lti_fout_be.discretize(
        A, B, step=1.0, alpha=alpha, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_fout_be.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_fout_be.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert random_16_input.shape[0] == 16
    assert random_16_input.shape[1] == 512
    assert random_16_input.shape[2] == 1

    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ------------
# --- foud ---
# ------------


def test_GBT_LTI_foud_BE(
    hippo_lti_foud_be, gu_hippo_lti_foud_be, foud_matrices, random_16_input
):
    print("\nHiPPO GBT LEGS")
    L = random_16_input.shape[1]
    alpha = 1.0
    A, B = foud_matrices

    GBT_A, GBT_B = hippo_lti_foud_be.discretize(
        A, B, step=1.0, alpha=alpha, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_foud_be.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_foud_be.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert random_16_input.shape[0] == 16
    assert random_16_input.shape[1] == 512
    assert random_16_input.shape[2] == 1

    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# --------------------
# ----- Bilinear -----
# --------------------

# ----------
# -- legs --
# ----------


def test_GBT_LSI_legs_BI(
    hippo_lsi_legs_bi, gu_hippo_lsi_legs_bi, legs_matrices, random_16_input
):
    print("\nHiPPO GBT LEGS")
    L = random_16_input.shape[1]
    alpha = 0.5
    A, B = legs_matrices

    assert random_16_input.shape[0] == 16
    assert random_16_input.shape[1] == 512
    assert random_16_input.shape[2] == 1

    for i in range(1, L + 1):
        GBT_A, GBT_B = hippo_lsi_legs_bi.discretize(
            A, B, step=i, alpha=alpha, dtype=jnp.float32
        )
        gu_GBT_A, gu_GBT_B = (
            jnp.asarray(gu_hippo_lsi_legs_bi.A_stacked[i - 1], dtype=jnp.float32),
            jnp.expand_dims(
                jnp.asarray(gu_hippo_lsi_legs_bi.B_stacked[i - 1], dtype=jnp.float32),
                axis=1,
            ),
        )
        assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
        assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


def test_GBT_LTI_legs_BI(
    hippo_lti_legs_bi, gu_hippo_lti_legs_bi, legs_matrices, random_16_input
):
    print("\nHiPPO GBT LEGS")
    L = random_16_input.shape[1]
    alpha = 0.5
    A, B = legs_matrices

    GBT_A, GBT_B = hippo_lti_legs_bi.discretize(
        A, B, step=1.0, alpha=alpha, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_legs_bi.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_legs_bi.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert random_16_input.shape[0] == 16
    assert random_16_input.shape[1] == 512
    assert random_16_input.shape[2] == 1

    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ----------
# -- legt --
# ----------


def test_GBT_LTI_legt_BI(
    hippo_lti_legt_bi, gu_hippo_lti_legt_bi, legt_matrices, random_16_input
):
    print("\nHiPPO GBT LEGS")
    L = random_16_input.shape[1]
    alpha = 0.5
    A, B = legt_matrices

    GBT_A, GBT_B = hippo_lti_legt_bi.discretize(
        A, B, step=1.0, alpha=alpha, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_legt_bi.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_legt_bi.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert random_16_input.shape[0] == 16
    assert random_16_input.shape[1] == 512
    assert random_16_input.shape[2] == 1

    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ----------
# -- lmu --
# ----------


def test_GBT_LTI_lmu_BI(
    hippo_lti_lmu_bi, gu_hippo_lti_lmu_bi, legt_lmu_matrices, random_16_input
):
    print("\nHiPPO GBT LEGS")
    L = random_16_input.shape[1]
    alpha = 0.5
    A, B = legt_lmu_matrices

    GBT_A, GBT_B = hippo_lti_lmu_bi.discretize(
        A, B, step=1.0, alpha=alpha, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_lmu_bi.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_lmu_bi.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert random_16_input.shape[0] == 16
    assert random_16_input.shape[1] == 512
    assert random_16_input.shape[2] == 1

    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ----------
# -- lagt --
# ----------


def test_GBT_LTI_lagt_BI(
    hippo_lti_lagt_bi, gu_hippo_lti_lagt_bi, lagt_matrices, random_16_input
):
    print("\nHiPPO GBT LEGS")
    L = random_16_input.shape[1]
    alpha = 0.5
    A, B = lagt_matrices

    GBT_A, GBT_B = hippo_lti_lagt_bi.discretize(
        A, B, step=1.0, alpha=alpha, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_lagt_bi.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_lagt_bi.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert random_16_input.shape[0] == 16
    assert random_16_input.shape[1] == 512
    assert random_16_input.shape[2] == 1

    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ----------
# --- fru --
# ----------


def test_GBT_LTI_fru_BI(
    hippo_lti_fru_bi, gu_hippo_lti_fru_bi, fru_matrices, random_16_input
):
    print("\nHiPPO GBT LEGS")
    L = random_16_input.shape[1]
    alpha = 0.5
    A, B = fru_matrices

    GBT_A, GBT_B = hippo_lti_fru_bi.discretize(
        A, B, step=1.0, alpha=alpha, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_fru_bi.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_fru_bi.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert random_16_input.shape[0] == 16
    assert random_16_input.shape[1] == 512
    assert random_16_input.shape[2] == 1

    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ------------
# --- fout ---
# ------------


def test_GBT_LTI_fout_BI(
    hippo_lti_fout_bi, gu_hippo_lti_fout_bi, fout_matrices, random_16_input
):
    print("\nHiPPO GBT LEGS")
    L = random_16_input.shape[1]
    alpha = 0.5
    A, B = fout_matrices

    GBT_A, GBT_B = hippo_lti_fout_bi.discretize(
        A, B, step=1.0, alpha=alpha, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_fout_bi.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_fout_bi.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert random_16_input.shape[0] == 16
    assert random_16_input.shape[1] == 512
    assert random_16_input.shape[2] == 1

    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ------------
# --- foud ---
# ------------


def test_GBT_LTI_foud_BI(
    hippo_lti_foud_bi, gu_hippo_lti_foud_bi, foud_matrices, random_16_input
):
    print("\nHiPPO GBT LEGS")
    L = random_16_input.shape[1]
    alpha = 0.5
    A, B = foud_matrices

    GBT_A, GBT_B = hippo_lti_foud_bi.discretize(
        A, B, step=1.0, alpha=alpha, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_foud_bi.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_foud_bi.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert random_16_input.shape[0] == 16
    assert random_16_input.shape[1] == 512
    assert random_16_input.shape[2] == 1

    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# --------------------
# - Zero-order Hold --
# --------------------

# ----------
# -- legs --
# ----------


def test_GBT_LSI_legs_ZOH(
    hippo_lsi_legs_zoh, gu_hippo_lsi_legs_zoh, legs_matrices, random_16_input
):
    print("\nHiPPO GBT LEGS")
    L = random_16_input.shape[1]
    alpha = 2.0
    A, B = legs_matrices

    assert random_16_input.shape[0] == 16
    assert random_16_input.shape[1] == 512
    assert random_16_input.shape[2] == 1

    for i in range(1, L + 1):
        GBT_A, GBT_B = hippo_lsi_legs_zoh.discretize(
            A, B, step=i, alpha=alpha, dtype=jnp.float32
        )
        gu_GBT_A, gu_GBT_B = (
            jnp.asarray(gu_hippo_lsi_legs_zoh.A_stacked[i - 1], dtype=jnp.float32),
            jnp.expand_dims(
                jnp.asarray(gu_hippo_lsi_legs_zoh.B_stacked[i - 1], dtype=jnp.float32),
                axis=1,
            ),
        )

        assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
        assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


def test_GBT_LTI_legs_ZOH(
    hippo_lti_legs_zoh, gu_hippo_lti_legs_zoh, legs_matrices, random_16_input
):
    print("\nHiPPO GBT LEGS")
    L = random_16_input.shape[1]
    alpha = 2.0
    A, B = legs_matrices

    GBT_A, GBT_B = hippo_lti_legs_zoh.discretize(
        A, B, step=1.0, alpha=alpha, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_legs_zoh.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_legs_zoh.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert random_16_input.shape[0] == 16
    assert random_16_input.shape[1] == 512
    assert random_16_input.shape[2] == 1

    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ----------
# -- legt --
# ----------


def test_GBT_LTI_legt_ZOH(
    hippo_lti_legt_zoh, gu_hippo_lti_legt_zoh, legt_matrices, random_16_input
):
    print("\nHiPPO GBT LEGS")
    L = random_16_input.shape[1]
    alpha = 2.0
    A, B = legt_matrices

    GBT_A, GBT_B = hippo_lti_legt_zoh.discretize(
        A, B, step=1.0, alpha=alpha, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_legt_zoh.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_legt_zoh.dB, dtype=jnp.float32),
            axis=1,
        ),
    )

    print(f"gu_GBT_A shape:{gu_GBT_A.shape}\n")
    print(f"GBT_A shape: {GBT_A.shape}\n")
    print(f"gu_GBT_B shape: {gu_GBT_B.shape}\n")
    print(f"GBT_B shape: {GBT_B.shape}")

    print(f"gu_GBT_A:\n{gu_GBT_A}\n")
    print(f"GBT_A:\n{GBT_A}\n")
    print(f"gu_GBT_B:\n{gu_GBT_B}\n")
    print(f"GBT_B:\n{GBT_B}")

    assert random_16_input.shape[0] == 16
    assert random_16_input.shape[1] == 512
    assert random_16_input.shape[2] == 1

    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ----------
# -- lmu --
# ----------


def test_GBT_LTI_lmu_ZOH(
    hippo_lti_lmu_zoh, gu_hippo_lti_lmu_zoh, legt_lmu_matrices, random_16_input
):
    print("\nHiPPO GBT LEGS")
    L = random_16_input.shape[1]
    alpha = 2.0
    A, B = legt_lmu_matrices

    GBT_A, GBT_B = hippo_lti_lmu_zoh.discretize(
        A, B, step=1.0, alpha=alpha, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_lmu_zoh.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_lmu_zoh.dB, dtype=jnp.float32),
            axis=1,
        ),
    )

    print(f"gu_GBT_A shape:{gu_GBT_A.shape}\n")
    print(f"GBT_A shape: {GBT_A.shape}\n")
    print(f"gu_GBT_B shape: {gu_GBT_B.shape}\n")
    print(f"GBT_B shape: {GBT_B.shape}")

    print(f"gu_GBT_A:\n{gu_GBT_A}\n")
    print(f"GBT_A:\n{GBT_A}\n")
    print(f"gu_GBT_B:\n{gu_GBT_B}\n")
    print(f"GBT_B:\n{GBT_B}")

    assert random_16_input.shape[0] == 16
    assert random_16_input.shape[1] == 512
    assert random_16_input.shape[2] == 1

    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ----------
# -- lagt --
# ----------


def test_GBT_LTI_lagt_ZOH(
    hippo_lti_lagt_zoh, gu_hippo_lti_lagt_zoh, lagt_matrices, random_16_input
):
    print("\nHiPPO GBT LEGS")
    L = random_16_input.shape[1]
    alpha = 2.0
    A, B = lagt_matrices

    GBT_A, GBT_B = hippo_lti_lagt_zoh.discretize(
        A, B, step=1.0, alpha=alpha, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_lagt_zoh.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_lagt_zoh.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert random_16_input.shape[0] == 16
    assert random_16_input.shape[1] == 512
    assert random_16_input.shape[2] == 1

    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ----------
# --- fru --
# ----------


def test_GBT_LTI_fru_ZOH(
    hippo_lti_fru_zoh, gu_hippo_lti_fru_zoh, fru_matrices, random_16_input
):
    print("\nHiPPO GBT LEGS")
    L = random_16_input.shape[1]
    alpha = 2.0
    A, B = fru_matrices

    GBT_A, GBT_B = hippo_lti_fru_zoh.discretize(
        A, B, step=1.0, alpha=alpha, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_fru_zoh.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_fru_zoh.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert random_16_input.shape[0] == 16
    assert random_16_input.shape[1] == 512
    assert random_16_input.shape[2] == 1

    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ------------
# --- fout ---
# ------------


def test_GBT_LTI_fout_ZOH(
    hippo_lti_fout_zoh, gu_hippo_lti_fout_zoh, fout_matrices, random_16_input
):
    print("\nHiPPO GBT LEGS")
    L = random_16_input.shape[1]
    alpha = 2.0
    A, B = fout_matrices

    GBT_A, GBT_B = hippo_lti_fout_zoh.discretize(
        A, B, step=1.0, alpha=alpha, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_fout_zoh.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_fout_zoh.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert random_16_input.shape[0] == 16
    assert random_16_input.shape[1] == 512
    assert random_16_input.shape[2] == 1

    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ------------
# --- foud ---
# ------------


def test_GBT_LTI_foud_ZOH(
    hippo_lti_foud_zoh, gu_hippo_lti_foud_zoh, foud_matrices, random_16_input
):
    print("\nHiPPO GBT LEGS")
    L = random_16_input.shape[1]
    alpha = 2.0
    A, B = foud_matrices

    GBT_A, GBT_B = hippo_lti_foud_zoh.discretize(
        A, B, step=1.0, alpha=alpha, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_foud_zoh.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_foud_zoh.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert random_16_input.shape[0] == 16
    assert random_16_input.shape[1] == 512
    assert random_16_input.shape[2] == 1

    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)
