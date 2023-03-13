import jax.numpy as jnp
import pytest

from src.models.hippo.hr_hippo import HRHiPPO_LSI, HRHiPPO_LTI
from src.models.hippo.hippo import HiPPOLTI, HiPPOLSI

# ----------------------------------------------------------------
# ------------------------ HiPPO operators -----------------------
# ----------------------------------------------------------------

# --------------------
# -- Forward Euler --
# --------------------

# ----------
# -- legs --
# ----------


@pytest.fixture
def hippo_lti_legs_fe():
    N = 50
    T = 3
    step = 1e-3
    measure = "legs"
    desc_val = 0.0

    return HiPPOLTI(
        N=N,
        step_size=step,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        basis_size=T,
        dtype=jnp.float32,
        unroll=False,
    )


@pytest.fixture
def hippo_lsi_legs_fe():
    N = 50
    T = 3
    step = 1e-3
    L = int(T / step)
    measure = "legs"
    desc_val = 0.0
    return HiPPOLSI(
        N=N,
        max_length=L,
        step_size=step,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        dtype=jnp.float32,
        unroll=True,
    )


# ----------
# -- legt --
# ----------


@pytest.fixture
def hippo_lti_legt_fe():
    N = 50
    T = 3
    step = 1e-3
    measure = "legt"
    desc_val = 0.0
    return HiPPOLTI(
        N=N,
        step_size=step,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        basis_size=T,
        dtype=jnp.float32,
        unroll=False,
    )


# ----------
# -- lmu --
# ----------


@pytest.fixture
def hippo_lti_lmu_fe():
    N = 50
    T = 3
    step = 1e-3
    measure = "lmu"
    desc_val = 0.0
    return HiPPOLTI(
        N=N,
        step_size=step,
        lambda_n=2.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        basis_size=T,
        dtype=jnp.float32,
        unroll=False,
    )


# ----------
# -- lagt --
# ----------


@pytest.fixture
def hippo_lti_lagt_fe():
    N = 50
    T = 3
    step = 1e-3
    measure = "lagt"
    desc_val = 0.0
    return HiPPOLTI(
        N=N,
        step_size=step,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        basis_size=T,
        dtype=jnp.float32,
        unroll=False,
    )


# ----------
# --- fru --
# ----------


@pytest.fixture
def hippo_lti_fru_fe():
    N = 50
    T = 3
    step = 1e-3
    measure = "fru"
    desc_val = 0.0
    return HiPPOLTI(
        N=N,
        step_size=step,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        basis_size=T,
        dtype=jnp.float32,
        unroll=False,
    )


# ------------
# --- fout ---
# ------------


@pytest.fixture
def hippo_lti_fout_fe():
    N = 50
    T = 3
    step = 1e-3
    measure = "fout"
    desc_val = 0.0
    return HiPPOLTI(
        N=N,
        step_size=step,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        basis_size=T,
        dtype=jnp.float32,
        unroll=False,
    )


# ------------
# --- foud ---
# ------------


@pytest.fixture
def hippo_lti_foud_fe():
    N = 50
    T = 3
    step = 1e-3
    measure = "foud"
    desc_val = 0.0
    return HiPPOLTI(
        N=N,
        step_size=step,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        basis_size=T,
        dtype=jnp.float32,
        unroll=False,
    )


# --------------------
# -- Backward Euler --
# --------------------

# ----------
# -- legs --
# ----------


@pytest.fixture
def hippo_lti_legs_be():
    N = 50
    T = 3
    step = 1e-3
    measure = "legs"
    desc_val = 1.0
    return HiPPOLTI(
        N=N,
        step_size=step,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        basis_size=T,
        dtype=jnp.float32,
        unroll=False,
    )


@pytest.fixture
def hippo_lsi_legs_be():
    N = 50
    T = 3
    step = 1e-3
    L = int(T / step)
    measure = "legs"
    desc_val = 1.0
    return HiPPOLSI(
        N=N,
        max_length=L,
        step_size=step,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        dtype=jnp.float32,
        unroll=True,
    )


# ----------
# -- legt --
# ----------


@pytest.fixture
def hippo_lti_legt_be():
    N = 50
    T = 3
    step = 1e-3
    measure = "legt"
    desc_val = 1.0
    return HiPPOLTI(
        N=N,
        step_size=step,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        basis_size=T,
        dtype=jnp.float32,
        unroll=False,
    )


# ----------
# -- lmu --
# ----------


@pytest.fixture
def hippo_lti_lmu_be():
    N = 50
    T = 3
    step = 1e-3
    measure = "lmu"
    desc_val = 1.0
    return HiPPOLTI(
        N=N,
        step_size=step,
        lambda_n=2.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        basis_size=T,
        dtype=jnp.float32,
        unroll=False,
    )


# ----------
# -- lagt --
# ----------


@pytest.fixture
def hippo_lti_lagt_be():
    N = 50
    T = 3
    step = 1e-3
    measure = "lagt"
    desc_val = 1.0
    return HiPPOLTI(
        N=N,
        step_size=step,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        basis_size=T,
        dtype=jnp.float32,
        unroll=False,
    )


# ----------
# --- fru --
# ----------


@pytest.fixture
def hippo_lti_fru_be():
    N = 50
    T = 3
    step = 1e-3
    measure = "fru"
    desc_val = 1.0
    return HiPPOLTI(
        N=N,
        step_size=step,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        basis_size=T,
        dtype=jnp.float32,
        unroll=False,
    )


# ------------
# --- fout ---
# ------------


@pytest.fixture
def hippo_lti_fout_be():
    N = 50
    T = 3
    step = 1e-3
    measure = "fout"
    desc_val = 1.0
    return HiPPOLTI(
        N=N,
        step_size=step,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        basis_size=T,
        dtype=jnp.float32,
        unroll=False,
    )


# ------------
# --- foud ---
# ------------


@pytest.fixture
def hippo_lti_foud_be():
    N = 50
    T = 3
    step = 1e-3
    measure = "foud"
    desc_val = 1.0
    return HiPPOLTI(
        N=N,
        step_size=step,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        basis_size=T,
        dtype=jnp.float32,
        unroll=False,
    )


# --------------------
# ----- Bilinear -----
# --------------------

# ----------
# -- legs --
# ----------


@pytest.fixture
def hippo_lti_legs_bi():
    N = 50
    T = 3
    step = 1e-3
    measure = "legs"
    desc_val = 0.5
    return HiPPOLTI(
        N=N,
        step_size=step,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        basis_size=T,
        dtype=jnp.float32,
        unroll=False,
    )


@pytest.fixture
def hippo_lsi_legs_bi():
    N = 50
    T = 3
    step = 1e-3
    L = int(T / step)
    measure = "legs"
    desc_val = 0.5
    return HiPPOLSI(
        N=N,
        max_length=L,
        step_size=step,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        dtype=jnp.float32,
        unroll=True,
    )


# ----------
# -- legt --
# ----------


@pytest.fixture
def hippo_lti_legt_bi():
    N = 50
    T = 3
    step = 1e-3
    measure = "legt"
    desc_val = 0.5
    return HiPPOLTI(
        N=N,
        step_size=step,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        basis_size=T,
        dtype=jnp.float32,
        unroll=False,
    )


# ----------
# -- lmu --
# ----------


@pytest.fixture
def hippo_lti_lmu_bi():
    N = 50
    T = 3
    step = 1e-3
    measure = "lmu"
    desc_val = 0.5
    return HiPPOLTI(
        N=N,
        step_size=step,
        lambda_n=2.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        basis_size=T,
        dtype=jnp.float32,
        unroll=False,
    )


# ----------
# -- lagt --
# ----------


@pytest.fixture
def hippo_lti_lagt_bi():
    N = 50
    T = 3
    step = 1e-3
    measure = "lagt"
    desc_val = 0.5
    return HiPPOLTI(
        N=N,
        step_size=step,
        lambda_n=2.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        basis_size=T,
        dtype=jnp.float32,
        unroll=False,
    )


# ----------
# --- fru --
# ----------


@pytest.fixture
def hippo_lti_fru_bi():
    N = 50
    T = 3
    step = 1e-3
    measure = "fru"
    desc_val = 0.5
    return HiPPOLTI(
        N=N,
        step_size=step,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        basis_size=T,
        dtype=jnp.float32,
        unroll=False,
    )


# ------------
# --- fout ---
# ------------


@pytest.fixture
def hippo_lti_fout_bi():
    N = 50
    T = 3
    step = 1e-3
    measure = "fout"
    desc_val = 0.5
    return HiPPOLTI(
        N=N,
        step_size=step,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        basis_size=T,
        dtype=jnp.float32,
        unroll=False,
    )


# ------------
# --- foud ---
# ------------


@pytest.fixture
def hippo_lti_foud_bi():
    N = 50
    T = 3
    step = 1e-3
    measure = "foud"
    desc_val = 0.5
    return HiPPOLTI(
        N=N,
        step_size=step,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        basis_size=T,
        dtype=jnp.float32,
        unroll=False,
    )


# --------------------
# - Zero-order Hold --
# --------------------

# ----------
# -- legs --
# ----------


@pytest.fixture
def hippo_lti_legs_zoh():
    N = 50
    T = 3
    step = 1e-3
    measure = "legs"
    desc_val = 2.0
    return HiPPOLTI(
        N=N,
        step_size=step,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        basis_size=T,
        dtype=jnp.float32,
        unroll=False,
    )


@pytest.fixture
def hippo_lsi_legs_zoh():
    N = 50
    T = 3
    step = 1e-3
    L = int(T / step)
    measure = "legs"
    desc_val = 2.0
    return HiPPOLSI(
        N=N,
        max_length=L,
        step_size=step,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        dtype=jnp.float32,
        unroll=True,
    )


# ----------
# -- legt --
# ----------


@pytest.fixture
def hippo_lti_legt_zoh():
    N = 50
    T = 3
    step = 1e-3
    measure = "legt"
    desc_val = 2.0
    return HiPPOLTI(
        N=N,
        step_size=step,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        basis_size=T,
        dtype=jnp.float32,
        unroll=False,
    )


# ----------
# -- lmu --
# ----------


@pytest.fixture
def hippo_lti_lmu_zoh():
    N = 50
    T = 3
    step = 1e-3
    measure = "lmu"
    desc_val = 2.0
    return HiPPOLTI(
        N=N,
        step_size=step,
        lambda_n=2.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        basis_size=T,
        dtype=jnp.float32,
        unroll=False,
    )


# ----------
# -- lagt --
# ----------


@pytest.fixture
def hippo_lti_lagt_zoh():
    N = 50
    T = 3
    step = 1e-3
    measure = "lagt"
    desc_val = 2.0
    return HiPPOLTI(
        N=N,
        step_size=step,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        basis_size=T,
        dtype=jnp.float32,
        unroll=False,
    )


# ----------
# --- fru --
# ----------


@pytest.fixture
def hippo_lti_fru_zoh():
    N = 50
    T = 3
    step = 1e-3
    measure = "fru"
    desc_val = 2.0
    return HiPPOLTI(
        N=N,
        step_size=step,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        basis_size=T,
        dtype=jnp.float32,
        unroll=False,
    )


# ------------
# --- fout ---
# ------------


@pytest.fixture
def hippo_lti_fout_zoh():
    N = 50
    T = 3
    step = 1e-3
    measure = "fout"
    desc_val = 2.0
    return HiPPOLTI(
        N=N,
        step_size=step,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        basis_size=T,
        dtype=jnp.float32,
        unroll=False,
    )


# ------------
# --- foud ---
# ------------


@pytest.fixture
def hippo_lti_foud_zoh():
    N = 50
    T = 3
    step = 1e-3
    measure = "foud"
    desc_val = 2.0
    return HiPPOLTI(
        N=N,
        step_size=step,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        basis_size=T,
        dtype=jnp.float32,
        unroll=False,
    )


# ----------------------------------------------------------------
# --------------------- Gu's Implementations ---------------------
# ----------------------------------------------------------------

# --------------------
# -- Forward Euler --
# --------------------

# ----------
# -- legs --
# ----------


@pytest.fixture
def hr_hippo_lti_legs_fe():
    N = 50
    T = 3
    step = 1e-3
    measure = "legs"
    desc_val = 0.0
    return HRHiPPO_LTI(
        N=N,
        method=measure,
        dt=step,
        T=T,
        discretization=desc_val,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        c=0.0,
    )


@pytest.fixture
def hr_hippo_lsi_legs_fe():
    N = 50
    T = 3
    step = 1e-3
    L = int(T / step)
    measure = "legs"
    desc_val = 0.0
    return HRHiPPO_LSI(
        N=N,
        method=measure,
        max_length=L,
        discretization=desc_val,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
    )


# ----------
# -- legt --
# ----------


@pytest.fixture
def hr_hippo_lti_legt_fe():
    N = 50
    T = 3
    step = 1e-3
    measure = "legt"
    desc_val = 0.0
    return HRHiPPO_LTI(
        N=N,
        method=measure,
        dt=step,
        T=T,
        discretization=desc_val,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        c=0.0,
    )


# ----------
# -- lmu --
# ----------


@pytest.fixture
def hr_hippo_lti_lmu_fe():
    N = 50
    T = 3
    step = 1e-3
    measure = "lmu"
    desc_val = 0.0
    return HRHiPPO_LTI(
        N=N,
        method=measure,
        dt=step,
        T=T,
        discretization=desc_val,
        lambda_n=2.0,
        alpha=0.0,
        beta=1.0,
        c=0.0,
    )


# ----------
# -- lagt --
# ----------


@pytest.fixture
def hr_hippo_lti_lagt_fe():
    N = 50
    T = 3
    step = 1e-3
    measure = "lagt"
    desc_val = 0.0
    return HRHiPPO_LTI(
        N=N,
        method=measure,
        dt=step,
        T=T,
        discretization=desc_val,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        c=0.0,
    )


# ----------
# --- fru --
# ----------


@pytest.fixture
def hr_hippo_lti_fru_fe():
    N = 50
    T = 3
    step = 1e-3
    measure = "fru"
    desc_val = 0.0
    return HRHiPPO_LTI(
        N=N,
        method=measure,
        dt=step,
        T=T,
        discretization=desc_val,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        c=0.0,
    )


# ------------
# --- fout ---
# ------------


@pytest.fixture
def hr_hippo_lti_fout_fe():
    N = 50
    T = 3
    step = 1e-3
    measure = "fout"
    desc_val = 0.0
    return HRHiPPO_LTI(
        N=N,
        method=measure,
        dt=step,
        T=T,
        discretization=desc_val,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        c=0.0,
    )


# ------------
# --- foud ---
# ------------


@pytest.fixture
def hr_hippo_lti_foud_fe():
    N = 50
    T = 3
    step = 1e-3
    measure = "foud"
    desc_val = 0.0
    return HRHiPPO_LTI(
        N=N,
        method=measure,
        dt=step,
        T=T,
        discretization=desc_val,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        c=0.0,
    )


# --------------------
# -- Backward Euler --
# --------------------

# ----------
# -- legs --
# ----------


@pytest.fixture
def hr_hippo_lti_legs_be():
    N = 50
    T = 3
    step = 1e-3
    measure = "legs"
    desc_val = 1.0
    return HRHiPPO_LTI(
        N=N,
        method=measure,
        dt=step,
        T=T,
        discretization=desc_val,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        c=0.0,
    )


@pytest.fixture
def hr_hippo_lsi_legs_be():
    N = 50
    T = 3
    step = 1e-3
    L = int(T / step)
    measure = "legs"
    desc_val = 1.0
    return HRHiPPO_LSI(
        N=N,
        method=measure,
        max_length=L,
        discretization=desc_val,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
    )


# ----------
# -- legt --
# ----------


@pytest.fixture
def hr_hippo_lti_legt_be():
    N = 50
    T = 3
    step = 1e-3
    measure = "legt"
    desc_val = 1.0
    return HRHiPPO_LTI(
        N=N,
        method=measure,
        dt=step,
        T=T,
        discretization=desc_val,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        c=0.0,
    )


# ----------
# -- lmu --
# ----------


@pytest.fixture
def hr_hippo_lti_lmu_be():
    N = 50
    T = 3
    step = 1e-3
    measure = "lmu"
    desc_val = 1.0
    return HRHiPPO_LTI(
        N=N,
        method=measure,
        dt=step,
        T=T,
        discretization=desc_val,
        lambda_n=2.0,
        alpha=0.0,
        beta=1.0,
        c=0.0,
    )


# ----------
# -- lagt --
# ----------


@pytest.fixture
def hr_hippo_lti_lagt_be():
    N = 50
    T = 3
    step = 1e-3
    measure = "lagt"
    desc_val = 1.0
    return HRHiPPO_LTI(
        N=N,
        method=measure,
        dt=step,
        T=T,
        discretization=desc_val,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        c=0.0,
    )


# ----------
# --- fru --
# ----------


@pytest.fixture
def hr_hippo_lti_fru_be():
    N = 50
    T = 3
    step = 1e-3
    measure = "fru"
    desc_val = 1.0
    return HRHiPPO_LTI(
        N=N,
        method=measure,
        dt=step,
        T=T,
        discretization=desc_val,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        c=0.0,
    )


# ------------
# --- fout ---
# ------------


@pytest.fixture
def hr_hippo_lti_fout_be():
    N = 50
    T = 3
    step = 1e-3
    measure = "fout"
    desc_val = 1.0
    return HRHiPPO_LTI(
        N=N,
        method=measure,
        dt=step,
        T=T,
        discretization=desc_val,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        c=0.0,
    )


# ------------
# --- foud ---
# ------------


@pytest.fixture
def hr_hippo_lti_foud_be():
    N = 50
    T = 3
    step = 1e-3
    measure = "foud"
    desc_val = 1.0
    return HRHiPPO_LTI(
        N=N,
        method=measure,
        dt=step,
        T=T,
        discretization=desc_val,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        c=0.0,
    )


# --------------------
# ----- Bilinear -----
# --------------------

# ----------
# -- legs --
# ----------


@pytest.fixture
def hr_hippo_lti_legs_bi():
    N = 50
    T = 3
    step = 1e-3
    measure = "legs"
    desc_val = 0.5
    return HRHiPPO_LTI(
        N=N,
        method=measure,
        dt=step,
        T=T,
        discretization=desc_val,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        c=0.0,
    )


@pytest.fixture
def hr_hippo_lsi_legs_bi():
    N = 50
    T = 3
    step = 1e-3
    L = int(T / step)
    measure = "legs"
    desc_val = 0.5
    return HRHiPPO_LSI(
        N=N,
        method=measure,
        max_length=L,
        discretization=desc_val,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
    )


# ----------
# -- legt --
# ----------


@pytest.fixture
def hr_hippo_lti_legt_bi():
    N = 50
    T = 3
    step = 1e-3
    measure = "legt"
    desc_val = 0.5
    return HRHiPPO_LTI(
        N=N,
        method=measure,
        dt=step,
        T=T,
        discretization=desc_val,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        c=0.0,
    )


# ----------
# -- lmu --
# ----------


@pytest.fixture
def hr_hippo_lti_lmu_bi():
    N = 50
    T = 3
    step = 1e-3
    measure = "lmu"
    desc_val = 0.5
    return HRHiPPO_LTI(
        N=N,
        method=measure,
        dt=step,
        T=T,
        discretization=desc_val,
        lambda_n=2.0,
        alpha=0.0,
        beta=1.0,
        c=0.0,
    )


# ----------
# -- lagt --
# ----------


@pytest.fixture
def hr_hippo_lti_lagt_bi():
    N = 50
    T = 3
    step = 1e-3
    measure = "lagt"
    desc_val = 0.5
    return HRHiPPO_LTI(
        N=N,
        method=measure,
        dt=step,
        T=T,
        discretization=desc_val,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        c=0.0,
    )


# ----------
# --- fru --
# ----------


@pytest.fixture
def hr_hippo_lti_fru_bi():
    N = 50
    T = 3
    step = 1e-3
    measure = "fru"
    desc_val = 0.5
    return HRHiPPO_LTI(
        N=N,
        method=measure,
        dt=step,
        T=T,
        discretization=desc_val,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        c=0.0,
    )


# ------------
# --- fout ---
# ------------


@pytest.fixture
def hr_hippo_lti_fout_bi():
    N = 50
    T = 3
    step = 1e-3
    measure = "fout"
    desc_val = 0.5
    return HRHiPPO_LTI(
        N=N,
        method=measure,
        dt=step,
        T=T,
        discretization=desc_val,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        c=0.0,
    )


# ------------
# --- foud ---
# ------------


@pytest.fixture
def hr_hippo_lti_foud_bi():
    N = 50
    T = 3
    step = 1e-3
    measure = "foud"
    desc_val = 0.5
    return HRHiPPO_LTI(
        N=N,
        method=measure,
        dt=step,
        T=T,
        discretization=desc_val,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        c=0.0,
    )


# --------------------
# - Zero-order Hold --
# --------------------

# ----------
# -- legs --
# ----------


@pytest.fixture
def hr_hippo_lti_legs_zoh():
    N = 50
    T = 3
    step = 1e-3
    measure = "legs"
    desc_val = "zoh"
    return HRHiPPO_LTI(
        N=N,
        method=measure,
        dt=step,
        T=T,
        discretization=desc_val,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        c=0.0,
    )


@pytest.fixture
def hr_hippo_lsi_legs_zoh():
    N = 50
    T = 3
    step = 1e-3
    L = int(T / step)
    measure = "legs"
    desc_val = "zoh"
    return HRHiPPO_LSI(
        N=N,
        method=measure,
        max_length=L,
        discretization=desc_val,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
    )


# ----------
# -- legt --
# ----------


@pytest.fixture
def hr_hippo_lti_legt_zoh():
    N = 50
    T = 3
    step = 1e-3
    measure = "legt"
    desc_val = "zoh"
    return HRHiPPO_LTI(
        N=N,
        method=measure,
        dt=step,
        T=T,
        discretization=desc_val,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        c=0.0,
    )


# ----------
# -- lmu --
# ----------


@pytest.fixture
def hr_hippo_lti_lmu_zoh():
    N = 50
    T = 3
    step = 1e-3
    measure = "lmu"
    desc_val = "zoh"
    return HRHiPPO_LTI(
        N=N,
        method=measure,
        dt=step,
        T=T,
        discretization=desc_val,
        lambda_n=2.0,
        alpha=0.0,
        beta=1.0,
        c=0.0,
    )


# ----------
# -- lagt --
# ----------


@pytest.fixture
def hr_hippo_lti_lagt_zoh():
    N = 50
    T = 3
    step = 1e-3
    measure = "lagt"
    desc_val = "zoh"
    return HRHiPPO_LTI(
        N=N,
        method=measure,
        dt=step,
        T=T,
        discretization=desc_val,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        c=0.0,
    )


# ----------
# --- fru --
# ----------


@pytest.fixture
def hr_hippo_lti_fru_zoh():
    N = 50
    T = 3
    step = 1e-3
    measure = "fru"
    desc_val = "zoh"
    return HRHiPPO_LTI(
        N=N,
        method=measure,
        dt=step,
        T=T,
        discretization=desc_val,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        c=0.0,
    )


# ------------
# --- fout ---
# ------------


@pytest.fixture
def hr_hippo_lti_fout_zoh():
    N = 50
    T = 3
    step = 1e-3
    measure = "fout"
    desc_val = "zoh"
    return HRHiPPO_LTI(
        N=N,
        method=measure,
        dt=step,
        T=T,
        discretization=desc_val,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        c=0.0,
    )


# ------------
# --- foud ---
# ------------


@pytest.fixture
def hr_hippo_lti_foud_zoh():
    N = 50
    T = 3
    step = 1e-3
    measure = "fru"
    desc_val = "zoh"
    return HRHiPPO_LTI(
        N=N,
        method=measure,
        dt=step,
        T=T,
        discretization=desc_val,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        c=0.0,
    )
