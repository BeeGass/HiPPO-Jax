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
# ---------------------- Test HiPPO Matrices ---------------------
# ----------------------------------------------------------------

# ---------------
# --- Vanilla ---
# ---------------


def test_legt_matrices(legt_matrices, gu_legt_matrices):
    print("HiPPO LEGT")
    A, B = legt_matrices
    gu_A, gu_B = gu_legt_matrices
    print(f"\nA:\n{A}\n")
    print(f"B:\n{B}\n")
    print(f"Gu's A:\n{gu_A}\n")
    print(f"Gu's B:\n{gu_B}\n")
    assert jnp.allclose(A, gu_A)
    assert jnp.allclose(B, gu_B)


def test_lmu_matrices(legt_lmu_matrices, gu_legt_lmu_matrices):
    print("HiPPO LMU")
    A, B = legt_lmu_matrices
    gu_A, gu_B = gu_legt_lmu_matrices
    print(f"\nA:\n{A}\n")
    print(f"B:\n{B}\n")
    print(f"Gu's A:\n{gu_A}\n")
    print(f"Gu's B:\n{gu_B}\n")
    assert jnp.allclose(A, gu_A)
    assert jnp.allclose(B, gu_B)


def test_lagt_matrices(lagt_matrices, gu_lagt_matrices):
    print("HiPPO LAGT")
    A, B = lagt_matrices
    gu_A, gu_B = gu_lagt_matrices
    print(f"\nA:\n{A}\n")
    print(f"B:\n{B}\n")
    print(f"Gu's A:\n{gu_A}\n")
    print(f"Gu's B:\n{gu_B}\n")
    assert jnp.allclose(A, gu_A)
    assert jnp.allclose(B, gu_B)


def test_legs_matrices(legs_matrices, gu_legs_matrices):
    print("HiPPO LEGS")
    A, B = legs_matrices
    gu_A, gu_B = gu_legs_matrices
    print(f"\nA:\n{A}\n")
    print(f"B:\n{B}\n")
    print(f"Gu's A:\n{gu_A}\n")
    print(f"Gu's B:\n{gu_B}\n")
    assert jnp.allclose(A, gu_A)
    assert jnp.allclose(B, gu_B)


def test_fru_matrices(fru_matrices, gu_fru_matrices):
    print("HiPPO FRU")
    A, B = fru_matrices
    gu_A, gu_B = gu_fru_matrices
    print(f"\nA:\n{A}\n")
    print(f"B:\n{B}\n")
    print(f"Gu's A:\n{gu_A}\n")
    print(f"Gu's B:\n{gu_B}\n")
    assert jnp.allclose(A, gu_A)
    assert jnp.allclose(B, gu_B)


def test_fout_matrices(fout_matrices, gu_fout_matrices):
    print("HiPPO FOUT")
    A, B = fout_matrices
    gu_A, gu_B = gu_fout_matrices
    print(f"\nA:\n{A}\n")
    print(f"B:\n{B}\n")
    print(f"Gu's A:\n{gu_A}\n")
    print(f"Gu's B:\n{gu_B}\n")
    assert jnp.allclose(A, gu_A)
    assert jnp.allclose(B, gu_B)


def test_foud_matrices(foud_matrices, gu_foud_matrices):
    print("HiPPO FOUD")
    A, B = foud_matrices
    gu_A, gu_B = gu_foud_matrices
    print(f"\nA:\n{A}\n")
    print(f"B:\n{B}\n")
    print(f"Gu's A:\n{gu_A}\n")
    print(f"Gu's B:\n{gu_B}\n")
    assert jnp.allclose(A, gu_A)
    assert jnp.allclose(B, gu_B)
