import jax.numpy as jnp
import pytest

from src.models.hippo.gu_hippo import gu_HiPPO_LSI, gu_HiPPO_LTI
from src.models.hippo.hippo import HiPPO, HiPPOLTI, HiPPOLSI
from src.tests.hippo_tests.hippo_utils import (
    N2,
    N16,
    N,
    big_N,
    random_1_input,
    random_16_input,
    random_32_input,
    random_64_input,
)
from src.tests.hippo_tests.trans_matrices import (
    foud_matrices,
    fout_matrices,
    fru_matrices,
    gu_foud_matrices,
    gu_fout_matrices,
    gu_fru_matrices,
    gu_lagt_matrices,
    gu_legs_matrices,
    gu_legt_lmu_matrices,
    gu_legt_matrices,
    lagt_matrices,
    legs_matrices,
    legt_lmu_matrices,
    legt_matrices,
)

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
def hippo_lti_legs_fe(random_16_input):
    N = 50
    measure = "legs"
    L = random_16_input.shape[1]
    desc_val = 0.0
    s_t = "lti"
    return HiPPO(
        N=N,
        max_length=L,
        step_size=1.0,
        basis_length=1.0,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        s_t=s_t,
        truncate_measure=True,
        dtype=jnp.float32,
        verbose=True,
    )


@pytest.fixture
def hippo_lsi_legs_fe(random_16_input):
    N = 50
    L = random_16_input.shape[1]
    desc_val = 0.0
    measure = "legs"
    s_t = "lsi"
    return HiPPO(
        N=N,
        max_length=L,
        step_size=1.0,
        basis_length=1.0,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        s_t=s_t,
        truncate_measure=True,
        dtype=jnp.float32,
        verbose=True,
    )


# ----------
# -- legt --
# ----------


@pytest.fixture
def hippo_lti_legt_fe(random_16_input):
    N = 50
    measure = "legt"
    L = random_16_input.shape[1]
    desc_val = 0.0
    s_t = "lti"
    return HiPPO(
        N=N,
        max_length=L,
        step_size=1.0,
        basis_length=1.0,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        s_t=s_t,
        truncate_measure=True,
        dtype=jnp.float32,
        verbose=True,
    )


# ----------
# -- lmu --
# ----------


@pytest.fixture
def hippo_lti_lmu_fe(random_16_input):
    N = 50
    measure = "lmu"
    L = random_16_input.shape[1]
    desc_val = 0.0
    s_t = "lti"
    return HiPPO(
        N=N,
        max_length=L,
        step_size=1.0,
        basis_length=1.0,
        lambda_n=2.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        s_t=s_t,
        truncate_measure=True,
        dtype=jnp.float32,
        verbose=True,
    )


# ----------
# -- lagt --
# ----------


@pytest.fixture
def hippo_lti_lagt_fe(random_16_input):
    N = 50
    measure = "lagt"
    L = random_16_input.shape[1]
    desc_val = 0.0
    s_t = "lti"
    return HiPPO(
        N=N,
        max_length=L,
        step_size=1.0,
        basis_length=1.0,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        s_t=s_t,
        truncate_measure=True,
        dtype=jnp.float32,
        verbose=True,
    )


# ----------
# --- fru --
# ----------


@pytest.fixture
def hippo_lti_fru_fe(random_16_input):
    N = 50
    measure = "fru"
    L = random_16_input.shape[1]
    desc_val = 0.0
    s_t = "lti"
    return HiPPO(
        N=N,
        max_length=L,
        step_size=1.0,
        basis_length=1.0,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        s_t=s_t,
        truncate_measure=True,
        dtype=jnp.float32,
        verbose=True,
    )


# ------------
# --- fout ---
# ------------


@pytest.fixture
def hippo_lti_fout_fe(random_16_input):
    N = 50
    measure = "fout"
    L = random_16_input.shape[1]
    desc_val = 0.0
    s_t = "lti"
    return HiPPO(
        N=N,
        max_length=L,
        step_size=1.0,
        basis_length=1.0,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        s_t=s_t,
        truncate_measure=True,
        dtype=jnp.float32,
        verbose=True,
    )


# ------------
# --- foud ---
# ------------


@pytest.fixture
def hippo_lti_foud_fe(random_16_input):
    N = 50
    measure = "foud"
    L = random_16_input.shape[1]
    desc_val = 0.0
    s_t = "lti"
    return HiPPO(
        N=N,
        max_length=L,
        step_size=1.0,
        basis_length=1.0,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        s_t=s_t,
        truncate_measure=True,
        dtype=jnp.float32,
        verbose=True,
    )


# --------------------
# -- Backward Euler --
# --------------------

# ----------
# -- legs --
# ----------


@pytest.fixture
def hippo_lti_legs_be(random_16_input):
    N = 50
    measure = "legs"
    L = random_16_input.shape[1]
    desc_val = 1.0
    s_t = "lti"
    return HiPPO(
        N=N,
        max_length=L,
        step_size=1.0,
        basis_length=1.0,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        s_t=s_t,
        truncate_measure=True,
        dtype=jnp.float32,
        verbose=True,
    )


@pytest.fixture
def hippo_lsi_legs_be(random_16_input):
    N = 50
    measure = "legs"
    L = random_16_input.shape[1]
    desc_val = 1.0
    s_t = "lsi"
    return HiPPO(
        N=N,
        max_length=L,
        step_size=1.0,
        basis_length=1.0,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        s_t=s_t,
        truncate_measure=True,
        dtype=jnp.float32,
        verbose=True,
    )


# ----------
# -- legt --
# ----------


@pytest.fixture
def hippo_lti_legt_be(random_16_input):
    N = 50
    measure = "legt"
    L = random_16_input.shape[1]
    desc_val = 1.0
    s_t = "lti"
    return HiPPO(
        N=N,
        max_length=L,
        step_size=1.0,
        basis_length=1.0,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        s_t=s_t,
        truncate_measure=True,
        dtype=jnp.float32,
        verbose=True,
    )


# ----------
# -- lmu --
# ----------


@pytest.fixture
def hippo_lti_lmu_be(random_16_input):
    N = 50
    measure = "lmu"
    L = random_16_input.shape[1]
    desc_val = 1.0
    s_t = "lti"
    return HiPPO(
        N=N,
        max_length=L,
        step_size=1.0,
        basis_length=1.0,
        lambda_n=2.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        s_t=s_t,
        truncate_measure=True,
        dtype=jnp.float32,
        verbose=True,
    )


# ----------
# -- lagt --
# ----------


@pytest.fixture
def hippo_lti_lagt_be(random_16_input):
    N = 50
    measure = "lagt"
    L = random_16_input.shape[1]
    desc_val = 1.0
    s_t = "lti"
    return HiPPO(
        N=N,
        max_length=L,
        step_size=1.0,
        basis_length=1.0,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        s_t=s_t,
        truncate_measure=True,
        dtype=jnp.float32,
        verbose=True,
    )


# ----------
# --- fru --
# ----------


@pytest.fixture
def hippo_lti_fru_be(random_16_input):
    N = 50
    measure = "fru"
    L = random_16_input.shape[1]
    desc_val = 1.0
    s_t = "lti"
    return HiPPO(
        N=N,
        max_length=L,
        step_size=1.0,
        basis_length=1.0,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        s_t=s_t,
        truncate_measure=True,
        dtype=jnp.float32,
        verbose=True,
    )


# ------------
# --- fout ---
# ------------


@pytest.fixture
def hippo_lti_fout_be(random_16_input):
    N = 50
    measure = "fout"
    L = random_16_input.shape[1]
    desc_val = 1.0
    s_t = "lti"
    return HiPPO(
        N=N,
        max_length=L,
        step_size=1.0,
        basis_length=1.0,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        s_t=s_t,
        truncate_measure=True,
        dtype=jnp.float32,
        verbose=True,
    )


# ------------
# --- foud ---
# ------------


@pytest.fixture
def hippo_lti_foud_be(random_16_input):
    N = 50
    measure = "foud"
    L = random_16_input.shape[1]
    desc_val = 1.0
    s_t = "lti"
    return HiPPO(
        N=N,
        max_length=L,
        step_size=1.0,
        basis_length=1.0,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        s_t=s_t,
        truncate_measure=True,
        dtype=jnp.float32,
        verbose=True,
    )


# --------------------
# ----- Bilinear -----
# --------------------

# ----------
# -- legs --
# ----------


@pytest.fixture
def hippo_lti_legs_bi(random_16_input):
    N = 50
    measure = "legs"
    L = random_16_input.shape[1]
    desc_val = 0.5
    s_t = "lti"
    return HiPPO(
        N=N,
        max_length=L,
        step_size=1.0,
        basis_length=1.0,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        s_t=s_t,
        truncate_measure=True,
        dtype=jnp.float32,
        verbose=True,
    )


@pytest.fixture
def hippo_lsi_legs_bi(random_16_input):
    N = 50
    measure = "legs"
    L = random_16_input.shape[1]
    desc_val = 0.5
    s_t = "lsi"
    return HiPPO(
        N=N,
        max_length=L,
        step_size=1.0,
        basis_length=1.0,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        s_t=s_t,
        truncate_measure=True,
        dtype=jnp.float32,
        verbose=True,
    )


# ----------
# -- legt --
# ----------


@pytest.fixture
def hippo_lti_legt_bi(random_16_input):
    N = 50
    measure = "legt"
    L = random_16_input.shape[1]
    desc_val = 0.5
    s_t = "lti"
    return HiPPO(
        N=N,
        max_length=L,
        step_size=1.0,
        basis_length=1.0,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        s_t=s_t,
        truncate_measure=True,
        dtype=jnp.float32,
        verbose=True,
    )


# ----------
# -- lmu --
# ----------


@pytest.fixture
def hippo_lti_lmu_bi(random_16_input):
    N = 50
    measure = "lmu"
    L = random_16_input.shape[1]
    desc_val = 0.5
    s_t = "lti"
    return HiPPO(
        N=N,
        max_length=L,
        step_size=1.0,
        basis_length=1.0,
        lambda_n=2.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        s_t=s_t,
        truncate_measure=True,
        dtype=jnp.float32,
        verbose=True,
    )


# ----------
# -- lagt --
# ----------


@pytest.fixture
def hippo_lti_lagt_bi(random_16_input):
    N = 50
    measure = "lagt"
    L = random_16_input.shape[1]
    desc_val = 0.5
    s_t = "lti"
    return HiPPO(
        N=N,
        max_length=L,
        step_size=1.0,
        basis_length=1.0,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        s_t=s_t,
        truncate_measure=True,
        dtype=jnp.float32,
        verbose=True,
    )


# ----------
# --- fru --
# ----------


@pytest.fixture
def hippo_lti_fru_bi(random_16_input):
    N = 50
    measure = "fru"
    L = random_16_input.shape[1]
    desc_val = 0.5
    s_t = "lti"
    return HiPPO(
        N=N,
        max_length=L,
        step_size=1.0,
        basis_length=1.0,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        s_t=s_t,
        truncate_measure=True,
        dtype=jnp.float32,
        verbose=True,
    )


# ------------
# --- fout ---
# ------------


@pytest.fixture
def hippo_lti_fout_bi(random_16_input):
    N = 50
    measure = "fout"
    L = random_16_input.shape[1]
    desc_val = 0.5
    s_t = "lti"
    return HiPPO(
        N=N,
        max_length=L,
        step_size=1.0,
        basis_length=1.0,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        s_t=s_t,
        truncate_measure=True,
        dtype=jnp.float32,
        verbose=True,
    )


# ------------
# --- foud ---
# ------------


@pytest.fixture
def hippo_lti_foud_bi(random_16_input):
    N = 50
    measure = "foud"
    L = random_16_input.shape[1]
    desc_val = 0.5
    s_t = "lti"
    return HiPPO(
        N=N,
        max_length=L,
        step_size=1.0,
        basis_length=1.0,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        s_t=s_t,
        truncate_measure=True,
        dtype=jnp.float32,
        verbose=True,
    )


# --------------------
# - Zero-order Hold --
# --------------------

# ----------
# -- legs --
# ----------


@pytest.fixture
def hippo_lti_legs_zoh(random_16_input):
    N = 50
    measure = "legs"
    L = random_16_input.shape[1]
    desc_val = 2.0
    s_t = "lti"
    return HiPPO(
        N=N,
        max_length=L,
        step_size=1.0,
        basis_length=1.0,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        s_t=s_t,
        truncate_measure=True,
        dtype=jnp.float32,
        verbose=True,
    )


@pytest.fixture
def hippo_lsi_legs_zoh(random_16_input):
    N = 50
    measure = "legs"
    L = random_16_input.shape[1]
    desc_val = 2.0
    s_t = "lsi"
    return HiPPO(
        N=N,
        max_length=L,
        step_size=1.0,
        basis_length=1.0,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        s_t=s_t,
        truncate_measure=True,
        dtype=jnp.float32,
        verbose=True,
    )


# ----------
# -- legt --
# ----------


@pytest.fixture
def hippo_lti_legt_zoh(random_16_input):
    N = 50
    measure = "legt"
    L = random_16_input.shape[1]
    desc_val = 2.0
    s_t = "lti"
    return HiPPO(
        N=N,
        max_length=L,
        step_size=1.0,
        basis_length=1.0,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        s_t=s_t,
        truncate_measure=True,
        dtype=jnp.float32,
        verbose=True,
    )


# ----------
# -- lmu --
# ----------


@pytest.fixture
def hippo_lti_lmu_zoh(random_16_input):
    N = 50
    measure = "lmu"
    L = random_16_input.shape[1]
    desc_val = 2.0
    s_t = "lti"
    return HiPPO(
        N=N,
        max_length=L,
        step_size=1.0,
        basis_length=1.0,
        lambda_n=2.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        s_t=s_t,
        truncate_measure=True,
        dtype=jnp.float32,
        verbose=True,
    )


# ----------
# -- lagt --
# ----------


@pytest.fixture
def hippo_lti_lagt_zoh(random_16_input):
    N = 50
    measure = "lagt"
    L = random_16_input.shape[1]
    desc_val = 2.0
    s_t = "lti"
    return HiPPO(
        N=N,
        max_length=L,
        step_size=1.0,
        basis_length=1.0,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        s_t=s_t,
        truncate_measure=True,
        dtype=jnp.float32,
        verbose=True,
    )


# ----------
# --- fru --
# ----------


@pytest.fixture
def hippo_lti_fru_zoh(random_16_input):
    N = 50
    measure = "fru"
    L = random_16_input.shape[1]
    desc_val = 2.0
    s_t = "lti"
    return HiPPO(
        N=N,
        max_length=L,
        step_size=1.0,
        basis_length=1.0,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        s_t=s_t,
        truncate_measure=True,
        dtype=jnp.float32,
        verbose=True,
    )


# ------------
# --- fout ---
# ------------


@pytest.fixture
def hippo_lti_fout_zoh(random_16_input):
    N = 50
    measure = "fout"
    L = random_16_input.shape[1]
    desc_val = 2.0
    s_t = "lti"
    return HiPPO(
        N=N,
        max_length=L,
        step_size=1.0,
        basis_length=1.0,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        s_t=s_t,
        truncate_measure=True,
        dtype=jnp.float32,
        verbose=True,
    )


# ------------
# --- foud ---
# ------------


@pytest.fixture
def hippo_lti_foud_zoh(random_16_input):
    N = 50
    measure = "foud"
    L = random_16_input.shape[1]
    desc_val = 2.0
    s_t = "lti"
    return HiPPO(
        N=N,
        max_length=L,
        step_size=1.0,
        basis_length=1.0,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=desc_val,
        measure=measure,
        s_t=s_t,
        truncate_measure=True,
        dtype=jnp.float32,
        verbose=True,
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
def gu_hippo_lti_legs_fe(random_16_input):
    N = 50
    measure = "legs"
    L = random_16_input.shape[1]
    desc_val = 0.0
    s_t = "lti"
    return gu_HiPPO_LTI(
        N=50,
        method=measure,
        dt=1.0,
        T=1.0,
        discretization=desc_val,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        c=0.0,
    )


@pytest.fixture
def gu_hippo_lsi_legs_fe(random_16_input):
    N = 50
    measure = "legs"
    L = random_16_input.shape[1]
    desc_val = 0.0
    s_t = "lti"
    return gu_HiPPO_LSI(
        N=50,
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
def gu_hippo_lti_legt_fe(random_16_input):
    N = 50
    measure = "legt"
    L = random_16_input.shape[1]
    desc_val = 0.0
    s_t = "lti"
    return gu_HiPPO_LTI(
        N=50,
        method=measure,
        dt=1.0,
        T=1.0,
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
def gu_hippo_lti_lmu_fe(random_16_input):
    N = 50
    measure = "lmu"
    L = random_16_input.shape[1]
    desc_val = 0.0
    s_t = "lti"
    return gu_HiPPO_LTI(
        N=50,
        method=measure,
        dt=1.0,
        T=1.0,
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
def gu_hippo_lti_lagt_fe(random_16_input):
    N = 50
    measure = "lagt"
    L = random_16_input.shape[1]
    desc_val = 0.0
    s_t = "lti"
    return gu_HiPPO_LTI(
        N=50,
        method=measure,
        dt=1.0,
        T=1.0,
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
def gu_hippo_lti_fru_fe(random_16_input):
    N = 50
    measure = "fru"
    L = random_16_input.shape[1]
    desc_val = 0.0
    s_t = "lti"
    return gu_HiPPO_LTI(
        N=50,
        method=measure,
        dt=1.0,
        T=1.0,
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
def gu_hippo_lti_fout_fe(random_16_input):
    N = 50
    measure = "fout"
    L = random_16_input.shape[1]
    desc_val = 0.0
    s_t = "lti"
    return gu_HiPPO_LTI(
        N=50,
        method=measure,
        dt=1.0,
        T=1.0,
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
def gu_hippo_lti_foud_fe(random_16_input):
    N = 50
    measure = "foud"
    L = random_16_input.shape[1]
    desc_val = 0.0
    s_t = "lti"
    return gu_HiPPO_LTI(
        N=50,
        method=measure,
        dt=1.0,
        T=1.0,
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
def gu_hippo_lti_legs_be(random_16_input):
    N = 50
    measure = "legs"
    L = random_16_input.shape[1]
    desc_val = 1.0
    s_t = "lti"
    return gu_HiPPO_LTI(
        N=50,
        method=measure,
        dt=1.0,
        T=1.0,
        discretization=desc_val,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        c=0.0,
    )


@pytest.fixture
def gu_hippo_lsi_legs_be(random_16_input):
    N = 50
    measure = "legs"
    L = random_16_input.shape[1]
    desc_val = 1.0
    s_t = "lti"
    return gu_HiPPO_LSI(
        N=50,
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
def gu_hippo_lti_legt_be(random_16_input):
    N = 50
    measure = "legt"
    L = random_16_input.shape[1]
    desc_val = 1.0
    s_t = "lti"
    return gu_HiPPO_LTI(
        N=50,
        method=measure,
        dt=1.0,
        T=1.0,
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
def gu_hippo_lti_lmu_be(random_16_input):
    N = 50
    measure = "lmu"
    L = random_16_input.shape[1]
    desc_val = 1.0
    s_t = "lti"
    return gu_HiPPO_LTI(
        N=50,
        method=measure,
        dt=1.0,
        T=1.0,
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
def gu_hippo_lti_lagt_be(random_16_input):
    N = 50
    measure = "lagt"
    L = random_16_input.shape[1]
    desc_val = 1.0
    s_t = "lti"
    return gu_HiPPO_LTI(
        N=50,
        method=measure,
        dt=1.0,
        T=1.0,
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
def gu_hippo_lti_fru_be(random_16_input):
    N = 50
    measure = "fru"
    L = random_16_input.shape[1]
    desc_val = 1.0
    s_t = "lti"
    return gu_HiPPO_LTI(
        N=50,
        method=measure,
        dt=1.0,
        T=1.0,
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
def gu_hippo_lti_fout_be(random_16_input):
    N = 50
    measure = "fout"
    L = random_16_input.shape[1]
    desc_val = 1.0
    s_t = "lti"
    return gu_HiPPO_LTI(
        N=50,
        method=measure,
        dt=1.0,
        T=1.0,
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
def gu_hippo_lti_foud_be(random_16_input):
    N = 50
    measure = "foud"
    L = random_16_input.shape[1]
    desc_val = 1.0
    s_t = "lti"
    return gu_HiPPO_LTI(
        N=50,
        method=measure,
        dt=1.0,
        T=1.0,
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
def gu_hippo_lti_legs_bi(random_16_input):
    N = 50
    measure = "legs"
    L = random_16_input.shape[1]
    desc_val = 0.5
    s_t = "lti"
    return gu_HiPPO_LTI(
        N=50,
        method=measure,
        dt=1.0,
        T=1.0,
        discretization=desc_val,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        c=0.0,
    )


@pytest.fixture
def gu_hippo_lsi_legs_bi(random_16_input):
    N = 50
    measure = "legs"
    L = random_16_input.shape[1]
    desc_val = 0.5
    s_t = "lti"
    return gu_HiPPO_LSI(
        N=50,
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
def gu_hippo_lti_legt_bi(random_16_input):
    N = 50
    measure = "legt"
    L = random_16_input.shape[1]
    desc_val = 0.5
    s_t = "lti"
    return gu_HiPPO_LTI(
        N=50,
        method=measure,
        dt=1.0,
        T=1.0,
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
def gu_hippo_lti_lmu_bi(random_16_input):
    N = 50
    measure = "lmu"
    L = random_16_input.shape[1]
    desc_val = 0.5
    s_t = "lti"
    return gu_HiPPO_LTI(
        N=50,
        method=measure,
        dt=1.0,
        T=1.0,
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
def gu_hippo_lti_lagt_bi(random_16_input):
    N = 50
    measure = "lagt"
    L = random_16_input.shape[1]
    desc_val = 0.5
    s_t = "lti"
    return gu_HiPPO_LTI(
        N=50,
        method=measure,
        dt=1.0,
        T=1.0,
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
def gu_hippo_lti_fru_bi(random_16_input):
    N = 50
    measure = "fru"
    L = random_16_input.shape[1]
    desc_val = 0.5
    s_t = "lti"
    return gu_HiPPO_LTI(
        N=50,
        method=measure,
        dt=1.0,
        T=1.0,
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
def gu_hippo_lti_fout_bi(random_16_input):
    N = 50
    measure = "fout"
    L = random_16_input.shape[1]
    desc_val = 0.5
    s_t = "lti"
    return gu_HiPPO_LTI(
        N=50,
        method=measure,
        dt=1.0,
        T=1.0,
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
def gu_hippo_lti_foud_bi(random_16_input):
    N = 50
    measure = "foud"
    L = random_16_input.shape[1]
    desc_val = 0.5
    s_t = "lti"
    return gu_HiPPO_LTI(
        N=50,
        method=measure,
        dt=1.0,
        T=1.0,
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
def gu_hippo_lti_legs_zoh(random_16_input):
    N = 50
    measure = "legs"
    L = random_16_input.shape[1]
    desc_val = "zoh"
    s_t = "lti"
    return gu_HiPPO_LTI(
        N=50,
        method=measure,
        dt=1.0,
        T=1.0,
        discretization=desc_val,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        c=0.0,
    )


@pytest.fixture
def gu_hippo_lsi_legs_zoh(random_16_input):
    N = 50
    measure = "legs"
    L = random_16_input.shape[1]
    desc_val = "zoh"
    s_t = "lti"
    return gu_HiPPO_LSI(
        N=50,
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
def gu_hippo_lti_legt_zoh(random_16_input):
    N = 50
    measure = "legt"
    L = random_16_input.shape[1]
    desc_val = "zoh"
    s_t = "lti"
    return gu_HiPPO_LTI(
        N=50,
        method=measure,
        dt=1.0,
        T=1.0,
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
def gu_hippo_lti_lmu_zoh(random_16_input):
    N = 50
    measure = "lmu"
    L = random_16_input.shape[1]
    desc_val = "zoh"
    s_t = "lti"
    return gu_HiPPO_LTI(
        N=50,
        method=measure,
        dt=1.0,
        T=1.0,
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
def gu_hippo_lti_lagt_zoh(random_16_input):
    N = 50
    measure = "lagt"
    L = random_16_input.shape[1]
    desc_val = "zoh"
    s_t = "lti"
    return gu_HiPPO_LTI(
        N=50,
        method=measure,
        dt=1.0,
        T=1.0,
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
def gu_hippo_lti_fru_zoh(random_16_input):
    N = 50
    measure = "fru"
    L = random_16_input.shape[1]
    desc_val = "zoh"
    s_t = "lti"
    return gu_HiPPO_LTI(
        N=50,
        method=measure,
        dt=1.0,
        T=1.0,
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
def gu_hippo_lti_fout_zoh(random_16_input):
    N = 50
    measure = "fout"
    L = random_16_input.shape[1]
    desc_val = "zoh"
    s_t = "lti"
    return gu_HiPPO_LTI(
        N=50,
        method=measure,
        dt=1.0,
        T=1.0,
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
def gu_hippo_lti_foud_zoh(random_16_input):
    N = 50
    measure = "foud"
    L = random_16_input.shape[1]
    desc_val = "zoh"
    s_t = "lti"
    return gu_HiPPO_LTI(
        N=50,
        method=measure,
        dt=1.0,
        T=1.0,
        discretization=desc_val,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        c=0.0,
    )
