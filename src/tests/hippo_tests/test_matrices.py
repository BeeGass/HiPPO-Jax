import pytest
import jax.numpy as jnp
import numpy as np
import torch

# Gu's implementation of HiPPO Operators
from src.tests.hippo_tests.hippo_operator import (
    gu_hippo_lti_legs_fe,
    gu_hippo_lsi_legs_fe,
    gu_hippo_lti_legt_fe,
    gu_hippo_lti_lmu_fe,
    gu_hippo_lti_lagt_fe,
    gu_hippo_lti_fru_fe,
    gu_hippo_lti_fout_fe,
    gu_hippo_lti_foud_fe,
)
from src.tests.hippo_tests.hippo_operator import (
    gu_hippo_lti_legs_be,
    gu_hippo_lsi_legs_be,
    gu_hippo_lti_legt_be,
    gu_hippo_lti_lmu_be,
    gu_hippo_lti_lagt_be,
    gu_hippo_lti_fru_be,
    gu_hippo_lti_fout_be,
    gu_hippo_lti_foud_be,
)
from src.tests.hippo_tests.hippo_operator import (
    gu_hippo_lti_legs_bi,
    gu_hippo_lsi_legs_bi,
    gu_hippo_lti_legt_bi,
    gu_hippo_lti_lmu_bi,
    gu_hippo_lti_lagt_bi,
    gu_hippo_lti_fru_bi,
    gu_hippo_lti_fout_bi,
    gu_hippo_lti_foud_bi,
)
from src.tests.hippo_tests.hippo_operator import (
    gu_hippo_lti_legs_zoh,
    gu_hippo_lsi_legs_zoh,
    gu_hippo_lti_legt_zoh,
    gu_hippo_lti_lmu_zoh,
    gu_hippo_lti_lagt_zoh,
    gu_hippo_lti_fru_zoh,
    gu_hippo_lti_fout_zoh,
    gu_hippo_lti_foud_zoh,
)

# implementation of HiPPO Operators
from src.tests.hippo_tests.hippo_operator import (
    hippo_lti_legs_fe,
    hippo_lsi_legs_fe,
    hippo_lti_legt_fe,
    hippo_lti_lmu_fe,
    hippo_lti_lagt_fe,
    hippo_lti_fru_fe,
    hippo_lti_fout_fe,
    hippo_lti_foud_fe,
)
from src.tests.hippo_tests.hippo_operator import (
    hippo_lti_legs_be,
    hippo_lsi_legs_be,
    hippo_lti_legt_be,
    hippo_lti_lmu_be,
    hippo_lti_lagt_be,
    hippo_lti_fru_be,
    hippo_lti_fout_be,
    hippo_lti_foud_be,
)
from src.tests.hippo_tests.hippo_operator import (
    hippo_lti_legs_bi,
    hippo_lsi_legs_bi,
    hippo_lti_legt_bi,
    hippo_lti_lmu_bi,
    hippo_lti_lagt_bi,
    hippo_lti_fru_bi,
    hippo_lti_fout_bi,
    hippo_lti_foud_bi,
)
from src.tests.hippo_tests.hippo_operator import (
    hippo_lti_legs_zoh,
    hippo_lsi_legs_zoh,
    hippo_lti_legt_zoh,
    hippo_lti_lmu_zoh,
    hippo_lti_lagt_zoh,
    hippo_lti_fru_zoh,
    hippo_lti_fout_zoh,
    hippo_lti_foud_zoh,
)

# transition matrices A and B from respective polynomials
from src.tests.hippo_tests.trans_matrices import (
    legs_matrices,
    legt_matrices,
    legt_lmu_matrices,
    lagt_matrices,
    fru_matrices,
    fout_matrices,
    foud_matrices,
)

# transition matrices A and B from respective polynomials made by Albert Gu
from src.tests.hippo_tests.trans_matrices import (
    gu_legs_matrices,
    gu_legt_matrices,
    gu_legt_lmu_matrices,
    gu_lagt_matrices,
    gu_fru_matrices,
    gu_fout_matrices,
    gu_foud_matrices,
)

# transition nplr matrices from respective polynomials
from src.tests.hippo_tests.trans_matrices import (
    nplr_legs,
    nplr_legt,
    nplr_lmu,
    nplr_lagt,
    nplr_fru,
    nplr_fout,
    nplr_foud,
)

# transition nplr matrices from respective polynomials made by Albert Gu
from src.tests.hippo_tests.trans_matrices import (
    gu_nplr_legs,
    gu_nplr_legt,
    gu_nplr_lmu,
    gu_nplr_lagt,
    gu_nplr_fru,
    gu_nplr_fout,
    gu_nplr_foud,
)

# transition dplr matrices from respective polynomials
from src.tests.hippo_tests.trans_matrices import (
    dplr_legs,
    dplr_legt,
    dplr_lmu,
    dplr_lagt,
    dplr_fru,
    dplr_fout,
    dplr_foud,
)

# transition dplr matrices from respective polynomials made by Albert Gu
from src.tests.hippo_tests.trans_matrices import (
    gu_dplr_legs,
    gu_dplr_legt,
    gu_dplr_lmu,
    gu_dplr_lagt,
    gu_dplr_fru,
    gu_dplr_fout,
    gu_dplr_foud,
)
from src.tests.hippo_tests.hippo_utils import (
    key_generator,
    legt_key,
    lmu_key,
    lagt_key,
    legs_key,
    fru_key,
    fout_key,
    foud_key,
)
from src.tests.hippo_tests.hippo_utils import (
    random_1_input,
    random_16_input,
    random_32_input,
    random_64_input,
)
import jax

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
