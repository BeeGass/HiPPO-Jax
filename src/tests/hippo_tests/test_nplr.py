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
