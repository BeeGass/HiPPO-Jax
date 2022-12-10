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
    hippo_foud,
)
from src.tests.hippo_tests.hippo_operator import (  # fixtures for respective operators made by Albert Gu
    gu_hippo_legs,
    gu_hippo_legt,
    gu_hippo_lmu,
    gu_hippo_lagt,
    gu_hippo_fru,
    gu_hippo_fout,
    gu_hippo_foud,
)
from src.tests.hippo_tests.trans_matrices import (  # transition matrices A and B from respective polynomials
    legs_matrices,
    legt_matrices,
    legt_lmu_matrices,
    lagt_matrices,
    fru_matrices,
    fout_matrices,
    foud_matrices,
)
from src.tests.hippo_tests.trans_matrices import (  # transition matrices A and B from respective polynomials made by Albert Gu
    gu_legs_matrices,
    gu_legt_matrices,
    gu_legt_lmu_matrices,
    gu_lagt_matrices,
    gu_fru_matrices,
    gu_fout_matrices,
    gu_foud_matrices,
)
from src.tests.hippo_tests.trans_matrices import (  # transition nplr matrices from respective polynomials
    nplr_legs,
    nplr_legt,
    nplr_lmu,
    nplr_lagt,
    nplr_fru,
    nplr_fout,
    nplr_foud,
)
from src.tests.hippo_tests.trans_matrices import (  # transition nplr matrices from respective polynomials made by Albert Gu
    gu_nplr_legs,
    gu_nplr_legt,
    gu_nplr_lmu,
    gu_nplr_lagt,
    gu_nplr_fru,
    gu_nplr_fout,
    gu_nplr_foud,
)
from src.tests.hippo_tests.trans_matrices import (  # transition dplr matrices from respective polynomials
    dplr_legs,
    dplr_legt,
    dplr_lmu,
    dplr_lagt,
    dplr_fru,
    dplr_fout,
    dplr_foud,
)
from src.tests.hippo_tests.trans_matrices import (  # transition dplr matrices from respective polynomials made by Albert Gu
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

# ------------------------------------------------ #
# --------------- Test HiPPO Matrices ------------ #
# ------------------------------------------------ #

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


# ---------------
# ----- NPLR ----
# ---------------


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


# ---------------
# ----- DPLR ----
# ---------------


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


# ------------------------------------------------ #
# -------------- Test HiPPO Operators ------------ #
# ------------------------------------------------ #


def test_hippo_legs_GBT(hippo_legs, gu_hippo_legs):
    print("HiPPO OPERATOR LEGS")
    i = 0
    for i, (GBT_A, GBT_B) in enumerate(
        zip(hippo_legs.GBT_A_list, hippo_legs.GBT_B_list)
    ):
        gu_GBT_A, gu_GBT_B = (
            jnp.asarray(gu_hippo_legs.A_stacked[i], dtype=jnp.float32),
            jnp.asarray(gu_hippo_legs.B_stacked[i], dtype=jnp.float32),
        )
        print(f"GBT_A:\n{GBT_A}\n")
        print(f"gu_GBT_A:\n{gu_GBT_A}\n")

        print(f"GBT_B:\n{GBT_B}\n")
        print(f"gu_GBT_B:\n{gu_GBT_B}\n")
        assert jnp.allclose(GBT_B, gu_GBT_B, rtol=1e-04, atol=1e-06)
        assert jnp.allclose(GBT_A, gu_GBT_A, rtol=1e-04, atol=1e-06)


def test_hippo_legs_operator(hippo_legs, gu_hippo_legs, random_16_input, legs_key):
    print("HiPPO OPERATOR LEGS")
    i = 0
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)

    params = hippo_legs.init(
        legs_key, f=random_16_input, t_step=random_16_input.shape[1]
    )
    c_k_list, y_k_list, GBT_A_list, GBT_B_list = hippo_legs.apply(
        params, f=random_16_input, t_step=random_16_input.shape[1]
    )
    c_k = jnp.stack(c_k_list, axis=0)
    c_k = jnp.moveaxis(c_k, 0, 1)

    GU_c_k = gu_hippo_legs(x_tensor)
    gu_c = jnp.asarray(GU_c_k, dtype=jnp.float32)  # convert torch array to jax array
    gu_c = jnp.moveaxis(gu_c, -1, -2)

    assert gu_c.shape == c_k.shape
    assert c_k.shape[0] == 16
    assert c_k.shape[1] == 512
    assert c_k.shape[2] == 8
    assert c_k.shape[3] == 1

    for i in range(c_k.shape[0]):
        for j in range(c_k.shape[1]):
            # print(
            #     f"batch {i} on trajectory {j} compare : {jnp.allclose(c_k[i,j,:,:], gu_c[i,j,:,:], rtol=1e-04, atol=1e-04)}"
            # )
            assert jnp.allclose(
                c_k[i, j, :, :], gu_c[i, j, :, :], rtol=1e-04, atol=1e-04
            ), f"batch {i} on trajectory {j} compare: {jnp.allclose(c_k[i,j,:,:], gu_c[i,j,:,:], rtol=1e-04, atol=1e-04)}"


def test_hippo_legt_operator(hippo_legt, gu_hippo_legt, random_16_input, legt_key):
    print("HiPPO OPERATOR LEGT")
    i = 0
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)

    params = hippo_legt.init(
        legt_key, f=random_16_input, t_step=random_16_input.shape[1]
    )
    c_k_list, y_k_list, GBT_A_list, GBT_B_list = hippo_legt.apply(
        params, f=random_16_input, t_step=random_16_input.shape[1]
    )
    GU_c_k = gu_hippo_legt(x_tensor)
    for i, c_k in enumerate(c_k_list):
        g_c_k = GU_c_k[i][0]
        gu = torch.unsqueeze(g_c_k, -1)
        gu_c = jnp.asarray(gu, dtype=jnp.float32)  # convert torch array to jax array
        assert jnp.allclose(c_k, gu_c, rtol=1e-04, atol=1e-06)


def test_hippo_lmu_operator(hippo_lmu, gu_hippo_lmu, random_16_input, lmu_key):
    print("HiPPO OPERATOR LMU")
    i = 0
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)

    params = hippo_lmu.init(lmu_key, f=random_16_input, t_step=random_16_input.shape[1])
    c_k_list, y_k_list, GBT_A_list, GBT_B_list = hippo_lmu.apply(
        params, f=random_16_input, t_step=random_16_input.shape[1]
    )
    GU_c_k = gu_hippo_lmu(x_tensor)
    for i, c_k in enumerate(c_k_list):
        g_c_k = GU_c_k[i][0]
        gu = torch.unsqueeze(g_c_k, -1)
        gu_c = jnp.asarray(gu, dtype=jnp.float32)  # convert torch array to jax array
        assert jnp.allclose(c_k, gu_c, rtol=1e-04, atol=1e-06)


def test_hippo_lagt_operator(hippo_lagt, gu_hippo_lagt, random_16_input, lagt_key):
    print("HiPPO OPERATOR LAGT")
    i = 0
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)

    params = hippo_lagt.init(
        lagt_key, f=random_16_input, t_step=random_16_input.shape[1]
    )
    c_k_list, y_k_list, GBT_A_list, GBT_B_list = hippo_lagt.apply(
        params, f=random_16_input, t_step=random_16_input.shape[1]
    )
    GU_c_k = gu_hippo_lagt(x_tensor)
    for i, c_k in enumerate(c_k_list):
        g_c_k = GU_c_k[i][0]
        gu = torch.unsqueeze(g_c_k, -1)
        gu_c = jnp.asarray(gu, dtype=jnp.float32)  # convert torch array to jax array
        assert jnp.allclose(c_k, gu_c, rtol=1e-04, atol=1e-06)


def test_hippo_fru_operator(hippo_fru, gu_hippo_fru, random_16_input, fru_key):
    print("HiPPO OPERATOR FRU")
    i = 0
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)

    params = hippo_fru.init(fru_key, f=random_16_input, t_step=random_16_input.shape[1])
    c_k_list, y_k_list, GBT_A_list, GBT_B_list = hippo_fru.apply(
        params, f=random_16_input, t_step=random_16_input.shape[1]
    )
    GU_c_k = gu_hippo_fru(x_tensor)
    for i, c_k in enumerate(c_k_list):
        g_c_k = GU_c_k[i][0]
        gu = torch.unsqueeze(g_c_k, -1)
        gu_c = jnp.asarray(gu, dtype=jnp.float32)  # convert torch array to jax array
        assert jnp.allclose(c_k, gu_c, rtol=1e-04, atol=1e-06)


def test_hippo_fout_operator(hippo_fout, gu_hippo_fout, random_16_input, fout_key):
    print("HiPPO OPERATOR FOUT")
    i = 0
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)

    params = hippo_fout.init(
        fout_key, f=random_16_input, t_step=random_16_input.shape[1]
    )
    c_k_list, y_k_list, GBT_A_list, GBT_B_list = hippo_fout.apply(
        params, f=random_16_input, t_step=random_16_input.shape[1]
    )
    GU_c_k = gu_hippo_fout(x_tensor)
    for i, c_k in enumerate(c_k_list):
        g_c_k = GU_c_k[i][0]
        gu = torch.unsqueeze(g_c_k, -1)
        gu_c = jnp.asarray(gu, dtype=jnp.float32)  # convert torch array to jax array
        assert jnp.allclose(c_k, gu_c, rtol=1e-04, atol=1e-06)


def test_hippo_foud_operator(hippo_foud, gu_hippo_foud, random_16_input, foud_key):
    print("HiPPO OPERATOR FOUD")
    i = 0
    x_np = np.asarray(random_16_input, dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)

    params = hippo_foud.init(
        foud_key, f=random_16_input, t_step=random_16_input.shape[1]
    )
    c_k_list, y_k_list, GBT_A_list, GBT_B_list = hippo_foud.apply(
        params, f=random_16_input, t_step=random_16_input.shape[1]
    )
    GU_c_k = gu_hippo_foud(x_tensor)
    for i, c_k in enumerate(c_k_list):
        g_c_k = GU_c_k[i][0]
        gu = torch.unsqueeze(g_c_k, -1)
        gu_c = jnp.asarray(gu, dtype=jnp.float32)  # convert torch array to jax array
        assert jnp.allclose(c_k, gu_c, rtol=1e-04, atol=1e-06)
