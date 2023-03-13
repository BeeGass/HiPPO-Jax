import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

# implementation of HiPPO Operators
# Gu's implementation of HiPPO Operators
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
# ----------------------- Test NPLR Matrices ---------------------
# ----------------------------------------------------------------


def test_nplr_legt(nplr_legt, hr_nplr_legt):
    print("NPLR LEGT")
    A, B, P, S = nplr_legt.A, nplr_legt.B, nplr_legt.P, nplr_legt.S
    hr_A, hr_B, hr_P, hr_S = (
        jnp.asarray(hr_nplr_legt.A, dtype=jnp.float32),
        jnp.asarray(hr_nplr_legt.B, dtype=jnp.float32),
        jnp.asarray(hr_nplr_legt.P, dtype=jnp.float32),
        jnp.asarray(hr_nplr_legt.S, dtype=jnp.float32),
    )
    print(f"\nA:\n{A}\n")
    print(f"hr_A:\n{hr_A}\n")
    assert jnp.allclose(A, hr_A, rtol=1e-04, atol=1e-06)

    print(f"B:\n{B}\n")
    print(f"hr_B:\n{hr_B}\n")
    assert jnp.allclose(B, hr_B, rtol=1e-04, atol=1e-06)

    print(f"P:\n{P}\n")
    print(f"hr_P:\n{hr_P}\n")
    assert jnp.allclose(P, hr_P, rtol=1e-04, atol=1e-06)

    print(f"S:\n{S}\n")
    print(f"hr_S:\n{hr_S}\n")
    assert jnp.allclose(S, hr_S, rtol=1e-04, atol=1e-06)


def test_nplr_lmu(nplr_lmu, hr_nplr_lmu):
    print("NPLR LMU")
    A, B, P, S = nplr_lmu.A, nplr_lmu.B, nplr_lmu.P, nplr_lmu.S
    hr_A, hr_B, hr_P, hr_S = (
        jnp.asarray(hr_nplr_lmu.A, dtype=jnp.float32),
        jnp.asarray(hr_nplr_lmu.B, dtype=jnp.float32),
        jnp.asarray(hr_nplr_lmu.P, dtype=jnp.float32),
        jnp.asarray(hr_nplr_lmu.S, dtype=jnp.float32),
    )
    print(f"\nA:\n{A}\n")
    print(f"hr_A:\n{hr_A}\n")
    assert jnp.allclose(A, hr_A, rtol=1e-04, atol=1e-06)

    print(f"B:\n{B}\n")
    print(f"hr_B:\n{hr_B}\n")
    assert jnp.allclose(B, hr_B, rtol=1e-04, atol=1e-06)

    print(f"P:\n{P}\n")
    print(f"hr_P:\n{hr_P}\n")
    assert jnp.allclose(P, hr_P, rtol=1e-04, atol=1e-06)

    print(f"S:\n{S}\n")
    print(f"hr_S:\n{hr_S}\n")
    assert jnp.allclose(S, hr_S, rtol=1e-04, atol=1e-06)


def test_nplr_lagt(nplr_lagt, hr_nplr_lagt):
    print("NPLR LAGT")
    A, B, P, S = nplr_lagt.A, nplr_lagt.B, nplr_lagt.P, nplr_lagt.S
    hr_A, hr_B, hr_P, hr_S = (
        jnp.asarray(hr_nplr_lagt.A, dtype=jnp.float32),
        jnp.asarray(hr_nplr_lagt.B, dtype=jnp.float32),
        jnp.asarray(hr_nplr_lagt.P, dtype=jnp.float32),
        jnp.asarray(hr_nplr_lagt.S, dtype=jnp.float32),
    )
    print(f"\nA:\n{A}\n")
    print(f"hr_A:\n{hr_A}\n")
    assert jnp.allclose(A, hr_A, rtol=1e-04, atol=1e-06)

    print(f"B:\n{B}\n")
    print(f"hr_B:\n{hr_B}\n")
    assert jnp.allclose(B, hr_B, rtol=1e-04, atol=1e-06)

    print(f"P:\n{P}\n")
    print(f"hr_P:\n{hr_P}\n")
    assert jnp.allclose(P, hr_P, rtol=1e-04, atol=1e-06)

    print(f"S:\n{S}\n")
    print(f"hr_S:\n{hr_S}\n")
    assert jnp.allclose(S, hr_S, rtol=1e-04, atol=1e-06)


def test_nplr_legs(nplr_legs, hr_nplr_legs):
    print("NPLR LEGS")
    A, B, P, S = nplr_legs.A, nplr_legs.B, nplr_legs.P, nplr_legs.S
    hr_A, hr_B, hr_P, hr_S = (
        jnp.asarray(hr_nplr_legs.A, dtype=jnp.float32),
        jnp.asarray(hr_nplr_legs.B, dtype=jnp.float32),
        jnp.asarray(hr_nplr_legs.P, dtype=jnp.float32),
        jnp.asarray(hr_nplr_legs.S, dtype=jnp.float32),
    )
    print(f"\nA:\n{A}\n")
    print(f"hr_A:\n{hr_A}\n")
    assert jnp.allclose(A, hr_A, rtol=1e-04, atol=1e-06)

    print(f"B:\n{B}\n")
    print(f"hr_B:\n{hr_B}\n")
    assert jnp.allclose(B, hr_B, rtol=1e-04, atol=1e-06)

    print(f"P:\n{P}\n")
    print(f"hr_P:\n{hr_P}\n")
    assert jnp.allclose(P, hr_P, rtol=1e-04, atol=1e-06)

    print(f"S:\n{S}\n")
    print(f"hr_S:\n{hr_S}\n")
    assert jnp.allclose(S, hr_S, rtol=1e-04, atol=1e-06)


def test_nplr_fru(nplr_fru, hr_nplr_fru):
    print("NPLR FRU")
    A, B, P, S = nplr_fru.A, nplr_fru.B, nplr_fru.P, nplr_fru.S
    hr_A, hr_B, hr_P, hr_S = (
        jnp.asarray(hr_nplr_fru.A, dtype=jnp.float32),
        jnp.asarray(hr_nplr_fru.B, dtype=jnp.float32),
        jnp.asarray(hr_nplr_fru.P, dtype=jnp.float32),
        jnp.asarray(hr_nplr_fru.S, dtype=jnp.float32),
    )
    print(f"\nA:\n{A}\n")
    print(f"hr_A:\n{hr_A}\n")
    assert jnp.allclose(A, hr_A, rtol=1e-04, atol=1e-06)

    print(f"B:\n{B}\n")
    print(f"hr_B:\n{hr_B}\n")
    assert jnp.allclose(B, hr_B, rtol=1e-04, atol=1e-06)

    print(f"P:\n{P}\n")
    print(f"hr_P:\n{hr_P}\n")
    assert jnp.allclose(P, hr_P, rtol=1e-04, atol=1e-06)

    print(f"S:\n{S}\n")
    print(f"hr_S:\n{hr_S}\n")
    assert jnp.allclose(S, hr_S, rtol=1e-04, atol=1e-06)


def test_nplr_fout(nplr_fout, hr_nplr_fout):
    print("NPLR FOUT")
    A, B, P, S = nplr_fout.A, nplr_fout.B, nplr_fout.P, nplr_fout.S
    hr_A, hr_B, hr_P, hr_S = (
        jnp.asarray(hr_nplr_fout.A, dtype=jnp.float32),
        jnp.asarray(hr_nplr_fout.B, dtype=jnp.float32),
        jnp.asarray(hr_nplr_fout.P, dtype=jnp.float32),
        jnp.asarray(hr_nplr_fout.S, dtype=jnp.float32),
    )
    print(f"\nA:\n{A}\n")
    print(f"hr_A:\n{hr_A}\n")
    assert jnp.allclose(A, hr_A, rtol=1e-04, atol=1e-06)

    print(f"B:\n{B}\n")
    print(f"hr_B:\n{hr_B}\n")
    assert jnp.allclose(B, hr_B, rtol=1e-04, atol=1e-06)

    print(f"P:\n{P}\n")
    print(f"hr_P:\n{hr_P}\n")
    assert jnp.allclose(P, hr_P, rtol=1e-04, atol=1e-06)

    print(f"S:\n{S}\n")
    print(f"hr_S:\n{hr_S}\n")
    assert jnp.allclose(S, hr_S, rtol=1e-04, atol=1e-06)


def test_nplr_foud(nplr_foud, hr_nplr_foud):
    print("NPLR FOUD")
    A, B, P, S = nplr_foud.A, nplr_foud.B, nplr_foud.P, nplr_foud.S
    hr_A, hr_B, hr_P, hr_S = (
        jnp.asarray(hr_nplr_foud.A, dtype=jnp.float32),
        jnp.asarray(hr_nplr_foud.B, dtype=jnp.float32),
        jnp.asarray(hr_nplr_foud.P, dtype=jnp.float32),
        jnp.asarray(hr_nplr_foud.S, dtype=jnp.float32),
    )
    print(f"\nA:\n{A}\n")
    print(f"hr_A:\n{hr_A}\n")
    assert jnp.allclose(A, hr_A, rtol=1e-04, atol=1e-06)

    print(f"B:\n{B}\n")
    print(f"hr_B:\n{hr_B}\n")
    assert jnp.allclose(B, hr_B, rtol=1e-04, atol=1e-06)

    print(f"P:\n{P}\n")
    print(f"hr_P:\n{hr_P}\n")
    assert jnp.allclose(P, hr_P, rtol=1e-04, atol=1e-06)

    print(f"S:\n{S}\n")
    print(f"hr_S:\n{hr_S}\n")
    assert jnp.allclose(S, hr_S, rtol=1e-04, atol=1e-06)
