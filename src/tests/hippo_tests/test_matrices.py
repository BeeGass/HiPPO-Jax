import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

# implementation of HiPPO Operators
# HR's implementation of HiPPO Operators
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
# ---------------------- Test HiPPO Matrices ---------------------
# ----------------------------------------------------------------

# ---------------
# --- Vanilla ---
# ---------------


def test_legt_matrices(legt_matrices, hr_legt_matrices):
    print("HiPPO LEGT")
    A, B = legt_matrices
    hr_A, hr_B = hr_legt_matrices
    print(f"\nA:\n{A}\n")
    print(f"B:\n{B}\n")
    print(f"HR's A:\n{hr_A}\n")
    print(f"HR's B:\n{hr_B}\n")
    assert jnp.allclose(A, hr_A)
    assert jnp.allclose(B, hr_B)


def test_lmu_matrices(legt_lmu_matrices, hr_legt_lmu_matrices):
    print("HiPPO LMU")
    A, B = legt_lmu_matrices
    hr_A, hr_B = hr_legt_lmu_matrices
    print(f"\nA:\n{A}\n")
    print(f"B:\n{B}\n")
    print(f"HR's A:\n{hr_A}\n")
    print(f"HR's B:\n{hr_B}\n")
    assert jnp.allclose(A, hr_A)
    assert jnp.allclose(B, hr_B)


def test_lagt_matrices(lagt_matrices, hr_lagt_matrices):
    print("HiPPO LAGT")
    A, B = lagt_matrices
    hr_A, hr_B = hr_lagt_matrices
    print(f"\nA:\n{A}\n")
    print(f"B:\n{B}\n")
    print(f"HR's A:\n{hr_A}\n")
    print(f"HR's B:\n{hr_B}\n")
    assert jnp.allclose(A, hr_A)
    assert jnp.allclose(B, hr_B)


def test_legs_matrices(legs_matrices, hr_legs_matrices):
    print("HiPPO LEGS")
    A, B = legs_matrices
    hr_A, hr_B = hr_legs_matrices
    print(f"\nA:\n{A}\n")
    print(f"B:\n{B}\n")
    print(f"HR's A:\n{hr_A}\n")
    print(f"HR's B:\n{hr_B}\n")
    assert jnp.allclose(A, hr_A)
    assert jnp.allclose(B, hr_B)


def test_fru_matrices(fru_matrices, hr_fru_matrices):
    print("HiPPO FRU")
    A, B = fru_matrices
    hr_A, hr_B = hr_fru_matrices
    print(f"\nA:\n{A}\n")
    print(f"B:\n{B}\n")
    print(f"HR's A:\n{hr_A}\n")
    print(f"HR's B:\n{hr_B}\n")
    assert jnp.allclose(A, hr_A)
    assert jnp.allclose(B, hr_B)


def test_fout_matrices(fout_matrices, hr_fout_matrices):
    print("HiPPO FOUT")
    A, B = fout_matrices
    hr_A, hr_B = hr_fout_matrices
    print(f"\nA:\n{A}\n")
    print(f"B:\n{B}\n")
    print(f"HR's A:\n{hr_A}\n")
    print(f"HR's B:\n{hr_B}\n")
    assert jnp.allclose(A, hr_A)
    assert jnp.allclose(B, hr_B)


def test_foud_matrices(foud_matrices, hr_foud_matrices):
    print("HiPPO FOUD")
    A, B = foud_matrices
    hr_A, hr_B = hr_foud_matrices
    print(f"\nA:\n{A}\n")
    print(f"B:\n{B}\n")
    print(f"HR's A:\n{hr_A}\n")
    print(f"HR's B:\n{hr_B}\n")
    assert jnp.allclose(A, hr_A)
    assert jnp.allclose(B, hr_B)
