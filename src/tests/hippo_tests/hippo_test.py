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


# ----------------------------------------------------------------
# ----------------------- Test NPLR Matrices ---------------------
# ----------------------------------------------------------------


def test_nplr_legt(nplr_legt, gu_nplr_legt):
    print("NPLR LEGT")
    A, B, P, S = nplr_legt.A, nplr_legt.B, nplr_legt.P, nplr_legt.S
    gu_A, gu_B, gu_P, gu_S = (
        jnp.asarray(gu_nplr_legt.A, dtype=jnp.float32),
        jnp.asarray(gu_nplr_legt.B, dtype=jnp.float32),
        jnp.asarray(gu_nplr_legt.P, dtype=jnp.float32),
        jnp.asarray(gu_nplr_legt.S, dtype=jnp.float32),
    )
    print(f"\nA:\n{A}\n")
    print(f"gu_A:\n{gu_A}\n")
    assert jnp.allclose(A, gu_A, rtol=1e-04, atol=1e-06)

    print(f"B:\n{B}\n")
    print(f"gu_B:\n{gu_B}\n")
    assert jnp.allclose(B, gu_B, rtol=1e-04, atol=1e-06)

    print(f"P:\n{P}\n")
    print(f"gu_P:\n{gu_P}\n")
    assert jnp.allclose(P, gu_P, rtol=1e-04, atol=1e-06)

    print(f"S:\n{S}\n")
    print(f"gu_S:\n{gu_S}\n")
    assert jnp.allclose(S, gu_S, rtol=1e-04, atol=1e-06)


def test_nplr_lmu(nplr_lmu, gu_nplr_lmu):
    print("NPLR LMU")
    A, B, P, S = nplr_lmu.A, nplr_lmu.B, nplr_lmu.P, nplr_lmu.S
    gu_A, gu_B, gu_P, gu_S = (
        jnp.asarray(gu_nplr_lmu.A, dtype=jnp.float32),
        jnp.asarray(gu_nplr_lmu.B, dtype=jnp.float32),
        jnp.asarray(gu_nplr_lmu.P, dtype=jnp.float32),
        jnp.asarray(gu_nplr_lmu.S, dtype=jnp.float32),
    )
    print(f"\nA:\n{A}\n")
    print(f"gu_A:\n{gu_A}\n")
    assert jnp.allclose(A, gu_A, rtol=1e-04, atol=1e-06)

    print(f"B:\n{B}\n")
    print(f"gu_B:\n{gu_B}\n")
    assert jnp.allclose(B, gu_B, rtol=1e-04, atol=1e-06)

    print(f"P:\n{P}\n")
    print(f"gu_P:\n{gu_P}\n")
    assert jnp.allclose(P, gu_P, rtol=1e-04, atol=1e-06)

    print(f"S:\n{S}\n")
    print(f"gu_S:\n{gu_S}\n")
    assert jnp.allclose(S, gu_S, rtol=1e-04, atol=1e-06)


def test_nplr_lagt(nplr_lagt, gu_nplr_lagt):
    print("NPLR LAGT")
    A, B, P, S = nplr_lagt.A, nplr_lagt.B, nplr_lagt.P, nplr_lagt.S
    gu_A, gu_B, gu_P, gu_S = (
        jnp.asarray(gu_nplr_lagt.A, dtype=jnp.float32),
        jnp.asarray(gu_nplr_lagt.B, dtype=jnp.float32),
        jnp.asarray(gu_nplr_lagt.P, dtype=jnp.float32),
        jnp.asarray(gu_nplr_lagt.S, dtype=jnp.float32),
    )
    print(f"\nA:\n{A}\n")
    print(f"gu_A:\n{gu_A}\n")
    assert jnp.allclose(A, gu_A, rtol=1e-04, atol=1e-06)

    print(f"B:\n{B}\n")
    print(f"gu_B:\n{gu_B}\n")
    assert jnp.allclose(B, gu_B, rtol=1e-04, atol=1e-06)

    print(f"P:\n{P}\n")
    print(f"gu_P:\n{gu_P}\n")
    assert jnp.allclose(P, gu_P, rtol=1e-04, atol=1e-06)

    print(f"S:\n{S}\n")
    print(f"gu_S:\n{gu_S}\n")
    assert jnp.allclose(S, gu_S, rtol=1e-04, atol=1e-06)


def test_nplr_legs(nplr_legs, gu_nplr_legs):
    print("NPLR LEGS")
    A, B, P, S = nplr_legs.A, nplr_legs.B, nplr_legs.P, nplr_legs.S
    gu_A, gu_B, gu_P, gu_S = (
        jnp.asarray(gu_nplr_legs.A, dtype=jnp.float32),
        jnp.asarray(gu_nplr_legs.B, dtype=jnp.float32),
        jnp.asarray(gu_nplr_legs.P, dtype=jnp.float32),
        jnp.asarray(gu_nplr_legs.S, dtype=jnp.float32),
    )
    print(f"\nA:\n{A}\n")
    print(f"gu_A:\n{gu_A}\n")
    assert jnp.allclose(A, gu_A, rtol=1e-04, atol=1e-06)

    print(f"B:\n{B}\n")
    print(f"gu_B:\n{gu_B}\n")
    assert jnp.allclose(B, gu_B, rtol=1e-04, atol=1e-06)

    print(f"P:\n{P}\n")
    print(f"gu_P:\n{gu_P}\n")
    assert jnp.allclose(P, gu_P, rtol=1e-04, atol=1e-06)

    print(f"S:\n{S}\n")
    print(f"gu_S:\n{gu_S}\n")
    assert jnp.allclose(S, gu_S, rtol=1e-04, atol=1e-06)


def test_nplr_fru(nplr_fru, gu_nplr_fru):
    print("NPLR FRU")
    A, B, P, S = nplr_fru.A, nplr_fru.B, nplr_fru.P, nplr_fru.S
    gu_A, gu_B, gu_P, gu_S = (
        jnp.asarray(gu_nplr_fru.A, dtype=jnp.float32),
        jnp.asarray(gu_nplr_fru.B, dtype=jnp.float32),
        jnp.asarray(gu_nplr_fru.P, dtype=jnp.float32),
        jnp.asarray(gu_nplr_fru.S, dtype=jnp.float32),
    )
    print(f"\nA:\n{A}\n")
    print(f"gu_A:\n{gu_A}\n")
    assert jnp.allclose(A, gu_A, rtol=1e-04, atol=1e-06)

    print(f"B:\n{B}\n")
    print(f"gu_B:\n{gu_B}\n")
    assert jnp.allclose(B, gu_B, rtol=1e-04, atol=1e-06)

    print(f"P:\n{P}\n")
    print(f"gu_P:\n{gu_P}\n")
    assert jnp.allclose(P, gu_P, rtol=1e-04, atol=1e-06)

    print(f"S:\n{S}\n")
    print(f"gu_S:\n{gu_S}\n")
    assert jnp.allclose(S, gu_S, rtol=1e-04, atol=1e-06)


def test_nplr_fout(nplr_fout, gu_nplr_fout):
    print("NPLR FOUT")
    A, B, P, S = nplr_fout.A, nplr_fout.B, nplr_fout.P, nplr_fout.S
    gu_A, gu_B, gu_P, gu_S = (
        jnp.asarray(gu_nplr_fout.A, dtype=jnp.float32),
        jnp.asarray(gu_nplr_fout.B, dtype=jnp.float32),
        jnp.asarray(gu_nplr_fout.P, dtype=jnp.float32),
        jnp.asarray(gu_nplr_fout.S, dtype=jnp.float32),
    )
    print(f"\nA:\n{A}\n")
    print(f"gu_A:\n{gu_A}\n")
    assert jnp.allclose(A, gu_A, rtol=1e-04, atol=1e-06)

    print(f"B:\n{B}\n")
    print(f"gu_B:\n{gu_B}\n")
    assert jnp.allclose(B, gu_B, rtol=1e-04, atol=1e-06)

    print(f"P:\n{P}\n")
    print(f"gu_P:\n{gu_P}\n")
    assert jnp.allclose(P, gu_P, rtol=1e-04, atol=1e-06)

    print(f"S:\n{S}\n")
    print(f"gu_S:\n{gu_S}\n")
    assert jnp.allclose(S, gu_S, rtol=1e-04, atol=1e-06)


def test_nplr_foud(nplr_foud, gu_nplr_foud):
    print("NPLR FOUD")
    A, B, P, S = nplr_foud.A, nplr_foud.B, nplr_foud.P, nplr_foud.S
    gu_A, gu_B, gu_P, gu_S = (
        jnp.asarray(gu_nplr_foud.A, dtype=jnp.float32),
        jnp.asarray(gu_nplr_foud.B, dtype=jnp.float32),
        jnp.asarray(gu_nplr_foud.P, dtype=jnp.float32),
        jnp.asarray(gu_nplr_foud.S, dtype=jnp.float32),
    )
    print(f"\nA:\n{A}\n")
    print(f"gu_A:\n{gu_A}\n")
    assert jnp.allclose(A, gu_A, rtol=1e-04, atol=1e-06)

    print(f"B:\n{B}\n")
    print(f"gu_B:\n{gu_B}\n")
    assert jnp.allclose(B, gu_B, rtol=1e-04, atol=1e-06)

    print(f"P:\n{P}\n")
    print(f"gu_P:\n{gu_P}\n")
    assert jnp.allclose(P, gu_P, rtol=1e-04, atol=1e-06)

    print(f"S:\n{S}\n")
    print(f"gu_S:\n{gu_S}\n")
    assert jnp.allclose(S, gu_S, rtol=1e-04, atol=1e-06)


# ----------------------------------------------------------------
# ---------------------- Test DPLR Matrices ----------------------
# ----------------------------------------------------------------


def test_dplr_legt(dplr_legt, gu_dplr_legt):
    print("DPLR LEGT")
    Lambda, P, B, V = dplr_legt.Lambda, dplr_legt.P, dplr_legt.B, dplr_legt.V
    gu_Lambda, gu_P, gu_B, gu_V = (
        jnp.asarray(gu_dplr_legt.Lambda, dtype=jnp.float32),
        jnp.asarray(gu_dplr_legt.P, dtype=jnp.float32),
        jnp.asarray(gu_dplr_legt.B, dtype=jnp.float32),
        jnp.asarray(gu_dplr_legt.V, dtype=jnp.float32),
    )
    print(f"\nLambda:\n{Lambda}\n")
    print(f"gu_Lambda:\n{gu_Lambda}\n")
    assert jnp.allclose(Lambda, gu_Lambda, rtol=1e-04, atol=1e-06)

    print(f"P:\n{P}\n")
    print(f"gu_P:\n{gu_P}\n")
    assert jnp.allclose(P, gu_P, rtol=1e-04, atol=1e-06)

    print(f"B:\n{B}\n")
    print(f"gu_B:\n{gu_B}\n")
    assert jnp.allclose(B, gu_B, rtol=1e-04, atol=1e-06)

    print(f"V:\n{V}\n")
    print(f"gu_V:\n{gu_V}\n")
    assert jnp.allclose(V, gu_V, rtol=1e-04, atol=1e-06)


def test_dplr_lmu(dplr_lmu, gu_dplr_lmu):
    print("DPLR LMU")
    Lambda, P, B, V = dplr_lmu.Lambda, dplr_lmu.P, dplr_lmu.B, dplr_lmu.V
    gu_Lambda, gu_P, gu_B, gu_V = (
        jnp.asarray(gu_dplr_lmu.Lambda, dtype=jnp.float32),
        jnp.asarray(gu_dplr_lmu.P, dtype=jnp.float32),
        jnp.asarray(gu_dplr_lmu.B, dtype=jnp.float32),
        jnp.asarray(gu_dplr_lmu.V, dtype=jnp.float32),
    )
    print(f"\nLambda:\n{Lambda}\n")
    print(f"gu_Lambda:\n{gu_Lambda}\n")
    assert jnp.allclose(Lambda, gu_Lambda, rtol=1e-04, atol=1e-06)

    print(f"P:\n{P}\n")
    print(f"gu_P:\n{gu_P}\n")
    assert jnp.allclose(P, gu_P, rtol=1e-04, atol=1e-06)

    print(f"B:\n{B}\n")
    print(f"gu_B:\n{gu_B}\n")
    assert jnp.allclose(B, gu_B, rtol=1e-04, atol=1e-06)

    print(f"V:\n{V}\n")
    print(f"gu_V:\n{gu_V}\n")
    assert jnp.allclose(V, gu_V, rtol=1e-04, atol=1e-06)


def test_dplr_lagt(dplr_lagt, gu_dplr_lagt):
    print("DPLR LAGT")
    Lambda, P, B, V = dplr_lagt.Lambda, dplr_lagt.P, dplr_lagt.B, dplr_lagt.V
    gu_Lambda, gu_P, gu_B, gu_V = (
        jnp.asarray(gu_dplr_lagt.Lambda, dtype=jnp.float32),
        jnp.asarray(gu_dplr_lagt.P, dtype=jnp.float32),
        jnp.asarray(gu_dplr_lagt.B, dtype=jnp.float32),
        jnp.asarray(gu_dplr_lagt.V, dtype=jnp.float32),
    )
    print(f"\nLambda:\n{Lambda}\n")
    print(f"gu_Lambda:\n{gu_Lambda}\n")
    assert jnp.allclose(Lambda, gu_Lambda, rtol=1e-04, atol=1e-06)

    print(f"P:\n{P}\n")
    print(f"gu_P:\n{gu_P}\n")
    assert jnp.allclose(P, gu_P, rtol=1e-04, atol=1e-06)

    print(f"B:\n{B}\n")
    print(f"gu_B:\n{gu_B}\n")
    assert jnp.allclose(B, gu_B, rtol=1e-04, atol=1e-06)

    print(f"V:\n{V}\n")
    print(f"gu_V:\n{gu_V}\n")
    assert jnp.allclose(V, gu_V, rtol=1e-04, atol=1e-06)


def test_dplr_legs(dplr_legs, gu_dplr_legs):
    print("DPLR LEGS")
    Lambda, P, B, V = dplr_legs.Lambda, dplr_legs.P, dplr_legs.B, dplr_legs.V
    gu_Lambda, gu_P, gu_B, gu_V = (
        jnp.asarray(gu_dplr_legs.Lambda, dtype=jnp.float32),
        jnp.asarray(gu_dplr_legs.P, dtype=jnp.float32),
        jnp.asarray(gu_dplr_legs.B, dtype=jnp.float32),
        jnp.asarray(gu_dplr_legs.V, dtype=jnp.float32),
    )
    print(f"\nLambda:\n{Lambda}\n")
    print(f"gu_Lambda:\n{gu_Lambda}\n")
    assert jnp.allclose(Lambda, gu_Lambda, rtol=1e-04, atol=1e-06)

    print(f"P:\n{P}\n")
    print(f"gu_P:\n{gu_P}\n")
    assert jnp.allclose(P, gu_P, rtol=1e-04, atol=1e-06)

    print(f"B:\n{B}\n")
    print(f"gu_B:\n{gu_B}\n")
    assert jnp.allclose(B, gu_B, rtol=1e-04, atol=1e-06)

    print(f"V:\n{V}\n")
    print(f"gu_V:\n{gu_V}\n")
    assert jnp.allclose(V, gu_V, rtol=1e-04, atol=1e-06)


def test_dplr_fru(dplr_fru, gu_dplr_fru):
    print("DPLR FRU")
    Lambda, P, B, V = dplr_fru.Lambda, dplr_fru.P, dplr_fru.B, dplr_fru.V
    gu_Lambda, gu_P, gu_B, gu_V = (
        jnp.asarray(gu_dplr_fru.Lambda, dtype=jnp.float32),
        jnp.asarray(gu_dplr_fru.P, dtype=jnp.float32),
        jnp.asarray(gu_dplr_fru.B, dtype=jnp.float32),
        jnp.asarray(gu_dplr_fru.V, dtype=jnp.float32),
    )
    print(f"\nLambda:\n{Lambda}\n")
    print(f"gu_Lambda:\n{gu_Lambda}\n")
    assert jnp.allclose(Lambda, gu_Lambda, rtol=1e-04, atol=1e-06)

    print(f"P:\n{P}\n")
    print(f"gu_P:\n{gu_P}\n")
    assert jnp.allclose(P, gu_P, rtol=1e-04, atol=1e-06)

    print(f"B:\n{B}\n")
    print(f"gu_B:\n{gu_B}\n")
    assert jnp.allclose(B, gu_B, rtol=1e-04, atol=1e-06)

    print(f"V:\n{V}\n")
    print(f"gu_V:\n{gu_V}\n")
    assert jnp.allclose(V, gu_V, rtol=1e-04, atol=1e-06)


def test_dplr_fout(dplr_fout, gu_dplr_fout):
    print("DPLR FOUT")
    Lambda, P, B, V = dplr_fout.Lambda, dplr_fout.P, dplr_fout.B, dplr_fout.V
    gu_Lambda, gu_P, gu_B, gu_V = (
        jnp.asarray(gu_dplr_fout.Lambda, dtype=jnp.float32),
        jnp.asarray(gu_dplr_fout.P, dtype=jnp.float32),
        jnp.asarray(gu_dplr_fout.B, dtype=jnp.float32),
        jnp.asarray(gu_dplr_fout.V, dtype=jnp.float32),
    )
    print(f"\nLambda:\n{Lambda}\n")
    print(f"gu_Lambda:\n{gu_Lambda}\n")
    assert jnp.allclose(Lambda, gu_Lambda, rtol=1e-04, atol=1e-06)

    print(f"P:\n{P}\n")
    print(f"gu_P:\n{gu_P}\n")
    assert jnp.allclose(P, gu_P, rtol=1e-04, atol=1e-06)

    print(f"B:\n{B}\n")
    print(f"gu_B:\n{gu_B}\n")
    assert jnp.allclose(B, gu_B, rtol=1e-04, atol=1e-06)

    print(f"V:\n{V}\n")
    print(f"gu_V:\n{gu_V}\n")
    assert jnp.allclose(V, gu_V, rtol=1e-04, atol=1e-06)


def test_dplr_foud(dplr_foud, gu_dplr_foud):
    print("DPLR FOUD")
    Lambda, P, B, V = dplr_foud.Lambda, dplr_foud.P, dplr_foud.B, dplr_foud.V
    gu_Lambda, gu_P, gu_B, gu_V = (
        jnp.asarray(gu_dplr_foud.Lambda, dtype=jnp.float32),
        jnp.asarray(gu_dplr_foud.P, dtype=jnp.float32),
        jnp.asarray(gu_dplr_foud.B, dtype=jnp.float32),
        jnp.asarray(gu_dplr_foud.V, dtype=jnp.float32),
    )
    print(f"\nLambda:\n{Lambda}\n")
    print(f"gu_Lambda:\n{gu_Lambda}\n")
    assert jnp.allclose(Lambda, gu_Lambda, rtol=1e-04, atol=1e-06)

    print(f"P:\n{P}\n")
    print(f"gu_P:\n{gu_P}\n")
    assert jnp.allclose(P, gu_P, rtol=1e-04, atol=1e-06)

    print(f"B:\n{B}\n")
    print(f"gu_B:\n{gu_B}\n")
    assert jnp.allclose(B, gu_B, rtol=1e-04, atol=1e-06)

    print(f"V:\n{V}\n")
    print(f"gu_V:\n{gu_V}\n")
    assert jnp.allclose(V, gu_V, rtol=1e-04, atol=1e-06)


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
    print("HiPPO GBT LEGS")
    L = random_16_input.shape[1]
    desc_val = 0.0
    A, B = legs_matrices

    for i in range(1, L + 1):
        GBT_A, GBT_B = hippo_lsi_legs_fe.discretize(
            A, B, step=i, alpha=desc_val, dtype=jnp.float32
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
    print("HiPPO GBT LEGS")
    L = random_16_input.shape[1]
    desc_val = 0.0
    A, B = legs_matrices

    GBT_A, GBT_B = hippo_lti_legs_fe.discretize(
        A, B, step=1.0, alpha=desc_val, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_legs_fe.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_legs_fe.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ----------
# -- legt --
# ----------


def test_GBT_LTI_legt_FE(
    hippo_lti_legt_fe, gu_hippo_lti_legt_fe, legt_matrices, random_16_input
):
    print("HiPPO GBT LEGS")
    L = random_16_input.shape[1]
    desc_val = 0.0
    A, B = legt_matrices

    GBT_A, GBT_B = hippo_lti_legt_fe.discretize(
        A, B, step=1.0, alpha=desc_val, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_legt_fe.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_legt_fe.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ----------
# -- lmu --
# ----------


def test_GBT_LTI_lmu_FE(
    hippo_lti_lmu_fe, gu_hippo_lti_lmu_fe, legt_lmu_matrices, random_16_input
):
    print("HiPPO GBT LEGS")
    L = random_16_input.shape[1]
    desc_val = 0.0
    A, B = legt_lmu_matrices

    GBT_A, GBT_B = hippo_lti_lmu_fe.discretize(
        A, B, step=1.0, alpha=desc_val, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_lmu_fe.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_lmu_fe.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ----------
# -- lagt --
# ----------


def test_GBT_LTI_lagt_FE(
    hippo_lti_lagt_fe, gu_hippo_lti_lagt_fe, lagt_matrices, random_16_input
):
    print("HiPPO GBT LEGS")
    L = random_16_input.shape[1]
    desc_val = 0.0
    A, B = lagt_matrices

    GBT_A, GBT_B = hippo_lti_lagt_fe.discretize(
        A, B, step=1.0, alpha=desc_val, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_lagt_fe.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_lagt_fe.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ----------
# --- fru --
# ----------


def test_GBT_LTI_fru_FE(
    hippo_lti_fru_fe, gu_hippo_lti_fru_fe, fru_matrices, random_16_input
):
    print("HiPPO GBT LEGS")
    L = random_16_input.shape[1]
    desc_val = 0.0
    A, B = fru_matrices

    GBT_A, GBT_B = hippo_lti_fru_fe.discretize(
        A, B, step=1.0, alpha=desc_val, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_fru_fe.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_fru_fe.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ------------
# --- fout ---
# ------------


def test_GBT_LTI_fout_FE(
    hippo_lti_fout_fe, gu_hippo_lti_fout_fe, fout_matrices, random_16_input
):
    print("HiPPO GBT LEGS")
    L = random_16_input.shape[1]
    desc_val = 0.0
    A, B = fout_matrices

    GBT_A, GBT_B = hippo_lti_fout_fe.discretize(
        A, B, step=1.0, alpha=desc_val, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_fout_fe.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_fout_fe.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ------------
# --- foud ---
# ------------


def test_GBT_LTI_fout_FE(
    hippo_lti_foud_fe, gu_hippo_lti_foud_fe, foud_matrices, random_16_input
):
    print("HiPPO GBT LEGS")
    L = random_16_input.shape[1]
    desc_val = 0.0
    A, B = foud_matrices

    GBT_A, GBT_B = hippo_lti_foud_fe.discretize(
        A, B, step=1.0, alpha=desc_val, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_foud_fe.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_foud_fe.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
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
    print("HiPPO GBT LEGS")
    L = random_16_input.shape[1]
    desc_val = 1.0
    A, B = legs_matrices

    for i in range(1, L + 1):
        GBT_A, GBT_B = hippo_lsi_legs_be.discretize(
            A, B, step=i, alpha=desc_val, dtype=jnp.float32
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
    print("HiPPO GBT LEGS")
    L = random_16_input.shape[1]
    desc_val = 1.0
    A, B = legs_matrices

    GBT_A, GBT_B = hippo_lti_legs_be.discretize(
        A, B, step=1.0, alpha=desc_val, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_legs_be.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_legs_be.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ----------
# -- legt --
# ----------


def test_GBT_LTI_legt_BE(
    hippo_lti_legt_be, gu_hippo_lti_legt_be, legt_matrices, random_16_input
):
    print("HiPPO GBT LEGS")
    L = random_16_input.shape[1]
    desc_val = 1.0
    A, B = legt_matrices

    GBT_A, GBT_B = hippo_lti_legt_be.discretize(
        A, B, step=1.0, alpha=desc_val, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_legt_be.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_legt_be.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ----------
# -- lmu --
# ----------


def test_GBT_LTI_lmu_BE(
    hippo_lti_lmu_be, gu_hippo_lti_lmu_be, legt_lmu_matrices, random_16_input
):
    print("HiPPO GBT LEGS")
    L = random_16_input.shape[1]
    desc_val = 1.0
    A, B = legt_lmu_matrices

    GBT_A, GBT_B = hippo_lti_lmu_be.discretize(
        A, B, step=1.0, alpha=desc_val, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_lmu_be.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_lmu_be.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ----------
# -- lagt --
# ----------


def test_GBT_LTI_lagt_BE(
    hippo_lti_lagt_be, gu_hippo_lti_lagt_be, lagt_matrices, random_16_input
):
    print("HiPPO GBT LEGS")
    L = random_16_input.shape[1]
    desc_val = 1.0
    A, B = lagt_matrices

    GBT_A, GBT_B = hippo_lti_lagt_be.discretize(
        A, B, step=1.0, alpha=desc_val, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_lagt_be.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_lagt_be.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ----------
# --- fru --
# ----------


def test_GBT_LTI_fru_BE(
    hippo_lti_fru_be, gu_hippo_lti_fru_be, fru_matrices, random_16_input
):
    print("HiPPO GBT LEGS")
    L = random_16_input.shape[1]
    desc_val = 1.0
    A, B = fru_matrices

    GBT_A, GBT_B = hippo_lti_fru_be.discretize(
        A, B, step=1.0, alpha=desc_val, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_fru_be.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_fru_be.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ------------
# --- fout ---
# ------------


def test_GBT_LTI_fout_BE(
    hippo_lti_fout_be, gu_hippo_lti_fout_be, fout_matrices, random_16_input
):
    print("HiPPO GBT LEGS")
    L = random_16_input.shape[1]
    desc_val = 1.0
    A, B = fout_matrices

    GBT_A, GBT_B = hippo_lti_fout_be.discretize(
        A, B, step=1.0, alpha=desc_val, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_fout_be.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_fout_be.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ------------
# --- foud ---
# ------------


def test_GBT_LTI_fout_BE(
    hippo_lti_foud_be, gu_hippo_lti_foud_be, foud_matrices, random_16_input
):
    print("HiPPO GBT LEGS")
    L = random_16_input.shape[1]
    desc_val = 1.0
    A, B = foud_matrices

    GBT_A, GBT_B = hippo_lti_foud_be.discretize(
        A, B, step=1.0, alpha=desc_val, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_foud_be.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_foud_be.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
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
    print("HiPPO GBT LEGS")
    L = random_16_input.shape[1]
    desc_val = 0.5
    A, B = legs_matrices

    for i in range(1, L + 1):
        GBT_A, GBT_B = hippo_lsi_legs_bi.discretize(
            A, B, step=i, alpha=desc_val, dtype=jnp.float32
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
    print("HiPPO GBT LEGS")
    L = random_16_input.shape[1]
    desc_val = 0.5
    A, B = legs_matrices

    GBT_A, GBT_B = hippo_lti_legs_bi.discretize(
        A, B, step=1.0, alpha=desc_val, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_legs_bi.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_legs_bi.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ----------
# -- legt --
# ----------


def test_GBT_LTI_legt_BI(
    hippo_lti_legt_bi, gu_hippo_lti_legt_bi, legt_matrices, random_16_input
):
    print("HiPPO GBT LEGS")
    L = random_16_input.shape[1]
    desc_val = 0.5
    A, B = legt_matrices

    GBT_A, GBT_B = hippo_lti_legt_bi.discretize(
        A, B, step=1.0, alpha=desc_val, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_legt_bi.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_legt_bi.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ----------
# -- lmu --
# ----------


def test_GBT_LTI_lmu_BI(
    hippo_lti_lmu_bi, gu_hippo_lti_lmu_bi, legt_lmu_matrices, random_16_input
):
    print("HiPPO GBT LEGS")
    L = random_16_input.shape[1]
    desc_val = 0.5
    A, B = legt_lmu_matrices

    GBT_A, GBT_B = hippo_lti_lmu_bi.discretize(
        A, B, step=1.0, alpha=desc_val, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_lmu_bi.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_lmu_bi.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ----------
# -- lagt --
# ----------


def test_GBT_LTI_lagt_BI(
    hippo_lti_lagt_bi, gu_hippo_lti_lagt_bi, lagt_matrices, random_16_input
):
    print("HiPPO GBT LEGS")
    L = random_16_input.shape[1]
    desc_val = 0.5
    A, B = lagt_matrices

    GBT_A, GBT_B = hippo_lti_lagt_bi.discretize(
        A, B, step=1.0, alpha=desc_val, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_lagt_bi.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_lagt_bi.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ----------
# --- fru --
# ----------


def test_GBT_LTI_fru_BI(
    hippo_lti_fru_bi, gu_hippo_lti_fru_bi, fru_matrices, random_16_input
):
    print("HiPPO GBT LEGS")
    L = random_16_input.shape[1]
    desc_val = 0.5
    A, B = fru_matrices

    GBT_A, GBT_B = hippo_lti_fru_bi.discretize(
        A, B, step=1.0, alpha=desc_val, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_fru_bi.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_fru_bi.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ------------
# --- fout ---
# ------------


def test_GBT_LTI_fout_BI(
    hippo_lti_fout_bi, gu_hippo_lti_fout_bi, fout_matrices, random_16_input
):
    print("HiPPO GBT LEGS")
    L = random_16_input.shape[1]
    desc_val = 0.5
    A, B = fout_matrices

    GBT_A, GBT_B = hippo_lti_fout_bi.discretize(
        A, B, step=1.0, alpha=desc_val, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_fout_bi.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_fout_bi.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ------------
# --- foud ---
# ------------


def test_GBT_LTI_fout_BI(
    hippo_lti_foud_bi, gu_hippo_lti_foud_bi, foud_matrices, random_16_input
):
    print("HiPPO GBT LEGS")
    L = random_16_input.shape[1]
    desc_val = 0.5
    A, B = foud_matrices

    GBT_A, GBT_B = hippo_lti_foud_bi.discretize(
        A, B, step=1.0, alpha=desc_val, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_foud_bi.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_foud_bi.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
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
    print("HiPPO GBT LEGS")
    L = random_16_input.shape[1]
    desc_val = 2.0
    A, B = legs_matrices

    for i in range(1, L + 1):
        GBT_A, GBT_B = hippo_lsi_legs_zoh.discretize(
            A, B, step=i, alpha=desc_val, dtype=jnp.float32
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
    print("HiPPO GBT LEGS")
    L = random_16_input.shape[1]
    desc_val = 2.0
    A, B = legs_matrices

    GBT_A, GBT_B = hippo_lti_legs_zoh.discretize(
        A, B, step=1.0, alpha=desc_val, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_legs_zoh.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_legs_zoh.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ----------
# -- legt --
# ----------


def test_GBT_LTI_legt_ZOH(
    hippo_lti_legt_zoh, gu_hippo_lti_legt_zoh, legt_matrices, random_16_input
):
    print("HiPPO GBT LEGS")
    L = random_16_input.shape[1]
    desc_val = 2.0
    A, B = legt_matrices

    GBT_A, GBT_B = hippo_lti_legt_zoh.discretize(
        A, B, step=1.0, alpha=desc_val, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_legt_zoh.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_legt_zoh.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ----------
# -- lmu --
# ----------


def test_GBT_LTI_lmu_ZOH(
    hippo_lti_lmu_zoh, gu_hippo_lti_lmu_zoh, legt_lmu_matrices, random_16_input
):
    print("HiPPO GBT LEGS")
    L = random_16_input.shape[1]
    desc_val = 2.0
    A, B = legt_lmu_matrices

    GBT_A, GBT_B = hippo_lti_lmu_zoh.discretize(
        A, B, step=1.0, alpha=desc_val, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_lmu_zoh.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_lmu_zoh.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ----------
# -- lagt --
# ----------


def test_GBT_LTI_lagt_ZOH(
    hippo_lti_lagt_zoh, gu_hippo_lti_lagt_zoh, lagt_matrices, random_16_input
):
    print("HiPPO GBT LEGS")
    L = random_16_input.shape[1]
    desc_val = 2.0
    A, B = lagt_matrices

    GBT_A, GBT_B = hippo_lti_lagt_zoh.discretize(
        A, B, step=1.0, alpha=desc_val, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_lagt_zoh.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_lagt_zoh.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ----------
# --- fru --
# ----------


def test_GBT_LTI_fru_ZOH(
    hippo_lti_fru_zoh, gu_hippo_lti_fru_zoh, legs_matrices, random_16_input
):
    print("HiPPO GBT LEGS")
    L = random_16_input.shape[1]
    desc_val = 2.0
    A, B = legs_matrices

    GBT_A, GBT_B = hippo_lti_fru_zoh.discretize(
        A, B, step=1.0, alpha=desc_val, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_fru_zoh.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_fru_zoh.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ------------
# --- fout ---
# ------------


def test_GBT_LTI_fout_ZOH(
    hippo_lti_fout_zoh, gu_hippo_lti_fout_zoh, fout_matrices, random_16_input
):
    print("HiPPO GBT LEGS")
    L = random_16_input.shape[1]
    desc_val = 2.0
    A, B = fout_matrices

    GBT_A, GBT_B = hippo_lti_fout_zoh.discretize(
        A, B, step=1.0, alpha=desc_val, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_fout_zoh.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_fout_zoh.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


# ------------
# --- foud ---
# ------------


def test_GBT_LTI_fout_ZOH(
    hippo_lti_foud_zoh, gu_hippo_lti_foud_zoh, foud_matrices, random_16_input
):
    print("HiPPO GBT LEGS")
    L = random_16_input.shape[1]
    desc_val = 2.0
    A, B = foud_matrices

    GBT_A, GBT_B = hippo_lti_foud_zoh.discretize(
        A, B, step=1.0, alpha=desc_val, dtype=jnp.float32
    )
    gu_GBT_A, gu_GBT_B = (
        jnp.asarray(gu_hippo_lti_foud_zoh.dA, dtype=jnp.float32),
        jnp.expand_dims(
            jnp.asarray(gu_hippo_lti_foud_zoh.dB, dtype=jnp.float32),
            axis=1,
        ),
    )
    assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-04)
    assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-04)


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
