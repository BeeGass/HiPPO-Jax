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
# ---------------------- Test DPLR Matrices ----------------------
# ----------------------------------------------------------------


def test_dplr_legt(dplr_legt, hr_dplr_legt):
    print("DPLR LEGT")
    Lambda, P, B, V = dplr_legt.Lambda, dplr_legt.P, dplr_legt.B, dplr_legt.V
    hr_Lambda, hr_P, hr_B, hr_V = (
        jnp.asarray(hr_dplr_legt.Lambda, dtype=jnp.float32),
        jnp.asarray(hr_dplr_legt.P, dtype=jnp.float32),
        jnp.asarray(hr_dplr_legt.B, dtype=jnp.float32),
        jnp.asarray(hr_dplr_legt.V, dtype=jnp.float32),
    )
    print(f"\nLambda:\n{Lambda}\n")
    print(f"hr_Lambda:\n{hr_Lambda}\n")
    assert jnp.allclose(Lambda, hr_Lambda, rtol=1e-04, atol=1e-06)

    print(f"P:\n{P}\n")
    print(f"hr_P:\n{hr_P}\n")
    assert jnp.allclose(P, hr_P, rtol=1e-04, atol=1e-06)

    print(f"B:\n{B}\n")
    print(f"hr_B:\n{hr_B}\n")
    assert jnp.allclose(B, hr_B, rtol=1e-04, atol=1e-06)

    print(f"V:\n{V}\n")
    print(f"hr_V:\n{hr_V}\n")
    assert jnp.allclose(V, hr_V, rtol=1e-04, atol=1e-06)


def test_dplr_lmu(dplr_lmu, hr_dplr_lmu):
    print("DPLR LMU")
    Lambda, P, B, V = dplr_lmu.Lambda, dplr_lmu.P, dplr_lmu.B, dplr_lmu.V
    hr_Lambda, hr_P, hr_B, hr_V = (
        jnp.asarray(hr_dplr_lmu.Lambda, dtype=jnp.float32),
        jnp.asarray(hr_dplr_lmu.P, dtype=jnp.float32),
        jnp.asarray(hr_dplr_lmu.B, dtype=jnp.float32),
        jnp.asarray(hr_dplr_lmu.V, dtype=jnp.float32),
    )
    print(f"\nLambda:\n{Lambda}\n")
    print(f"hr_Lambda:\n{hr_Lambda}\n")
    assert jnp.allclose(Lambda, hr_Lambda, rtol=1e-04, atol=1e-06)

    print(f"P:\n{P}\n")
    print(f"hr_P:\n{hr_P}\n")
    assert jnp.allclose(P, hr_P, rtol=1e-04, atol=1e-06)

    print(f"B:\n{B}\n")
    print(f"hr_B:\n{hr_B}\n")
    assert jnp.allclose(B, hr_B, rtol=1e-04, atol=1e-06)

    print(f"V:\n{V}\n")
    print(f"hr_V:\n{hr_V}\n")
    assert jnp.allclose(V, hr_V, rtol=1e-04, atol=1e-06)


def test_dplr_lagt(dplr_lagt, hr_dplr_lagt):
    print("DPLR LAGT")
    Lambda, P, B, V = dplr_lagt.Lambda, dplr_lagt.P, dplr_lagt.B, dplr_lagt.V
    hr_Lambda, hr_P, hr_B, hr_V = (
        jnp.asarray(hr_dplr_lagt.Lambda, dtype=jnp.float32),
        jnp.asarray(hr_dplr_lagt.P, dtype=jnp.float32),
        jnp.asarray(hr_dplr_lagt.B, dtype=jnp.float32),
        jnp.asarray(hr_dplr_lagt.V, dtype=jnp.float32),
    )
    print(f"\nLambda:\n{Lambda}\n")
    print(f"hr_Lambda:\n{hr_Lambda}\n")
    assert jnp.allclose(Lambda, hr_Lambda, rtol=1e-04, atol=1e-06)

    print(f"P:\n{P}\n")
    print(f"hr_P:\n{hr_P}\n")
    assert jnp.allclose(P, hr_P, rtol=1e-04, atol=1e-06)

    print(f"B:\n{B}\n")
    print(f"hr_B:\n{hr_B}\n")
    assert jnp.allclose(B, hr_B, rtol=1e-04, atol=1e-06)

    print(f"V:\n{V}\n")
    print(f"hr_V:\n{hr_V}\n")
    assert jnp.allclose(V, hr_V, rtol=1e-04, atol=1e-06)


def test_dplr_legs(dplr_legs, hr_dplr_legs):
    print("DPLR LEGS")
    Lambda, P, B, V = dplr_legs.Lambda, dplr_legs.P, dplr_legs.B, dplr_legs.V
    hr_Lambda, hr_P, hr_B, hr_V = (
        jnp.asarray(hr_dplr_legs.Lambda, dtype=jnp.float32),
        jnp.asarray(hr_dplr_legs.P, dtype=jnp.float32),
        jnp.asarray(hr_dplr_legs.B, dtype=jnp.float32),
        jnp.asarray(hr_dplr_legs.V, dtype=jnp.float32),
    )
    print(f"\nLambda:\n{Lambda}\n")
    print(f"hr_Lambda:\n{hr_Lambda}\n")
    assert jnp.allclose(Lambda, hr_Lambda, rtol=1e-04, atol=1e-06)

    print(f"P:\n{P}\n")
    print(f"hr_P:\n{hr_P}\n")
    assert jnp.allclose(P, hr_P, rtol=1e-04, atol=1e-06)

    print(f"B:\n{B}\n")
    print(f"hr_B:\n{hr_B}\n")
    assert jnp.allclose(B, hr_B, rtol=1e-04, atol=1e-06)

    print(f"V:\n{V}\n")
    print(f"hr_V:\n{hr_V}\n")
    assert jnp.allclose(V, hr_V, rtol=1e-04, atol=1e-06)


def test_dplr_fru(dplr_fru, hr_dplr_fru):
    print("DPLR FRU")
    Lambda, P, B, V = dplr_fru.Lambda, dplr_fru.P, dplr_fru.B, dplr_fru.V
    hr_Lambda, hr_P, hr_B, hr_V = (
        jnp.asarray(hr_dplr_fru.Lambda, dtype=jnp.float32),
        jnp.asarray(hr_dplr_fru.P, dtype=jnp.float32),
        jnp.asarray(hr_dplr_fru.B, dtype=jnp.float32),
        jnp.asarray(hr_dplr_fru.V, dtype=jnp.float32),
    )
    print(f"\nLambda:\n{Lambda}\n")
    print(f"hr_Lambda:\n{hr_Lambda}\n")
    assert jnp.allclose(Lambda, hr_Lambda, rtol=1e-04, atol=1e-06)

    print(f"P:\n{P}\n")
    print(f"hr_P:\n{hr_P}\n")
    assert jnp.allclose(P, hr_P, rtol=1e-04, atol=1e-06)

    print(f"B:\n{B}\n")
    print(f"hr_B:\n{hr_B}\n")
    assert jnp.allclose(B, hr_B, rtol=1e-04, atol=1e-06)

    print(f"V:\n{V}\n")
    print(f"hr_V:\n{hr_V}\n")
    assert jnp.allclose(V, hr_V, rtol=1e-04, atol=1e-06)


def test_dplr_fout(dplr_fout, hr_dplr_fout):
    print("DPLR FOUT")
    Lambda, P, B, V = dplr_fout.Lambda, dplr_fout.P, dplr_fout.B, dplr_fout.V
    hr_Lambda, hr_P, hr_B, hr_V = (
        jnp.asarray(hr_dplr_fout.Lambda, dtype=jnp.float32),
        jnp.asarray(hr_dplr_fout.P, dtype=jnp.float32),
        jnp.asarray(hr_dplr_fout.B, dtype=jnp.float32),
        jnp.asarray(hr_dplr_fout.V, dtype=jnp.float32),
    )
    print(f"\nLambda:\n{Lambda}\n")
    print(f"hr_Lambda:\n{hr_Lambda}\n")
    assert jnp.allclose(Lambda, hr_Lambda, rtol=1e-04, atol=1e-06)

    print(f"P:\n{P}\n")
    print(f"hr_P:\n{hr_P}\n")
    assert jnp.allclose(P, hr_P, rtol=1e-04, atol=1e-06)

    print(f"B:\n{B}\n")
    print(f"hr_B:\n{hr_B}\n")
    assert jnp.allclose(B, hr_B, rtol=1e-04, atol=1e-06)

    print(f"V:\n{V}\n")
    print(f"hr_V:\n{hr_V}\n")
    assert jnp.allclose(V, hr_V, rtol=1e-04, atol=1e-06)


def test_dplr_foud(dplr_foud, hr_dplr_foud):
    print("DPLR FOUD")
    Lambda, P, B, V = dplr_foud.Lambda, dplr_foud.P, dplr_foud.B, dplr_foud.V
    hr_Lambda, hr_P, hr_B, hr_V = (
        jnp.asarray(hr_dplr_foud.Lambda, dtype=jnp.float32),
        jnp.asarray(hr_dplr_foud.P, dtype=jnp.float32),
        jnp.asarray(hr_dplr_foud.B, dtype=jnp.float32),
        jnp.asarray(hr_dplr_foud.V, dtype=jnp.float32),
    )
    print(f"\nLambda:\n{Lambda}\n")
    print(f"hr_Lambda:\n{hr_Lambda}\n")
    assert jnp.allclose(Lambda, hr_Lambda, rtol=1e-04, atol=1e-06)

    print(f"P:\n{P}\n")
    print(f"hr_P:\n{hr_P}\n")
    assert jnp.allclose(P, hr_P, rtol=1e-04, atol=1e-06)

    print(f"B:\n{B}\n")
    print(f"hr_B:\n{hr_B}\n")
    assert jnp.allclose(B, hr_B, rtol=1e-04, atol=1e-06)

    print(f"V:\n{V}\n")
    print(f"hr_V:\n{hr_V}\n")
    assert jnp.allclose(V, hr_V, rtol=1e-04, atol=1e-06)
