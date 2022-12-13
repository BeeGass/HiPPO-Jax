import pytest
from src.models.hippo.hippo import HiPPO
from src.tests.hippo_tests.trans_matrices import (
    legs_matrices,
    legt_matrices,
    legt_lmu_matrices,
    lagt_matrices,
    fru_matrices,
    fout_matrices,
    foud_matrices,
)
from src.tests.hippo_tests.trans_matrices import (
    gu_legs_matrices,
    gu_legt_matrices,
    gu_legt_lmu_matrices,
    gu_lagt_matrices,
    gu_fru_matrices,
    gu_fout_matrices,
    gu_foud_matrices,
)
from src.tests.hippo_tests.hippo_utils import N, N2, N16, big_N
from src.tests.hippo_tests.hippo_utils import (
    random_1_input,
    random_16_input,
    random_32_input,
    random_64_input,
)
from src.models.hippo.gu_hippo import HiPPO_LSI, HiPPO_LTI
import jax.numpy as jnp


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
    measure = "legs"
    L = random_16_input.shape[1]
    return HiPPO(
        max_length=L,
        step_size=1.0,
        N=100,
        GBT_alpha=0.0,
        measure=measure,
        dtype=jnp.float32,
        verbose=True,
    )


@pytest.fixture
def hippo_lsi_legs_fe(random_16_input):
    measure = "legs"
    L = random_16_input.shape[1]
    return HiPPO(
        max_length=L,
        step_size=1.0 / L,
        N=100,
        GBT_alpha=0.0,
        measure=measure,
        dtype=jnp.float32,
        verbose=True,
    )


# ----------
# -- legt --
# ----------


@pytest.fixture
def hippo_legt_fe(random_16_input):
    measure = "legt"
    L = random_16_input.shape[1]
    return HiPPO(
        max_length=L,
        step_size=1.0 / L,
        N=100,
        lambda_n=1.0,
        GBT_alpha=0.0,
        measure=measure,
        dtype=jnp.float32,
        verbose=True,
    )


# ----------
# -- lmu --
# ----------


@pytest.fixture
def hippo_lmu_fe(random_16_input):
    measure = "legt"
    L = random_16_input.shape[1]
    return HiPPO(
        max_length=L,
        step_size=1.0 / L,
        N=100,
        lambda_n=1.0,
        GBT_alpha=0.0,
        measure=measure,
        dtype=jnp.float32,
        verbose=True,
    )


# ----------
# -- lagt --
# ----------


@pytest.fixture
def hippo_lagt_fe(random_16_input):
    measure = "lagt"
    L = random_16_input.shape[1]
    return HiPPO(
        max_length=L,
        step_size=1.0 / L,
        N=100,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=0.0,
        measure=measure,
        dtype=jnp.float32,
        verbose=True,
    )


# ----------
# --- fru --
# ----------


@pytest.fixture
def hippo_fru_fe(random_16_input):
    measure = "fru"
    L = random_16_input.shape[1]
    return HiPPO(
        max_length=L,
        step_size=1.0 / L,
        N=100,
        GBT_alpha=0.0,
        measure=measure,
        dtype=jnp.float32,
        verbose=True,
    )


# ------------
# --- fout ---
# ------------


@pytest.fixture
def hippo_fout_fe(random_16_input):
    measure = "fout"
    L = random_16_input.shape[1]
    return HiPPO(
        max_length=L,
        step_size=1.0 / L,
        N=100,
        GBT_alpha=0.0,
        measure=measure,
        dtype=jnp.float32,
        verbose=True,
    )


# ------------
# --- foud ---
# ------------


@pytest.fixture
def hippo_foud_fe(random_16_input):
    measure = "foud"
    L = random_16_input.shape[1]
    return HiPPO(
        max_length=L,
        step_size=1.0 / L,
        N=100,
        GBT_alpha=0.0,
        measure=measure,
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
    measure = "legs"
    L = random_16_input.shape[1]
    return HiPPO(
        max_length=L,
        step_size=1.0,
        N=100,
        GBT_alpha=1.0,
        measure=measure,
        dtype=jnp.float32,
        verbose=True,
    )


@pytest.fixture
def hippo_lsi_legs_be(random_16_input):
    measure = "legs"
    L = random_16_input.shape[1]
    return HiPPO(
        max_length=L,
        step_size=1.0 / L,
        N=100,
        GBT_alpha=1.0,
        measure=measure,
        dtype=jnp.float32,
        verbose=True,
    )


# ----------
# -- legt --
# ----------


@pytest.fixture
def hippo_legt_be(random_16_input):
    measure = "legt"
    L = random_16_input.shape[1]
    return HiPPO(
        max_length=L,
        step_size=1.0 / L,
        N=100,
        lambda_n=1.0,
        GBT_alpha=1.0,
        measure=measure,
        dtype=jnp.float32,
        verbose=True,
    )


# ----------
# -- lmu --
# ----------


@pytest.fixture
def hippo_lmu_be(random_16_input):
    measure = "legt"
    L = random_16_input.shape[1]
    return HiPPO(
        max_length=L,
        step_size=1.0 / L,
        N=100,
        lambda_n=1.0,
        GBT_alpha=1.0,
        measure=measure,
        dtype=jnp.float32,
        verbose=True,
    )


# ----------
# -- lagt --
# ----------


@pytest.fixture
def hippo_lagt_be(random_16_input):
    measure = "lagt"
    L = random_16_input.shape[1]
    return HiPPO(
        max_length=L,
        step_size=1.0 / L,
        N=100,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=1.0,
        measure=measure,
        dtype=jnp.float32,
        verbose=True,
    )


# ----------
# --- fru --
# ----------


@pytest.fixture
def hippo_fru_be(random_16_input):
    measure = "fru"
    L = random_16_input.shape[1]
    return HiPPO(
        max_length=L,
        step_size=1.0 / L,
        N=100,
        GBT_alpha=1.0,
        measure=measure,
        dtype=jnp.float32,
        verbose=True,
    )


# ------------
# --- fout ---
# ------------


@pytest.fixture
def hippo_fout_be(random_16_input):
    measure = "fout"
    L = random_16_input.shape[1]
    return HiPPO(
        max_length=L,
        step_size=1.0 / L,
        N=100,
        GBT_alpha=1.0,
        measure=measure,
        dtype=jnp.float32,
        verbose=True,
    )


# ------------
# --- foud ---
# ------------


@pytest.fixture
def hippo_foud_be(random_16_input):
    measure = "foud"
    L = random_16_input.shape[1]
    return HiPPO(
        max_length=L,
        step_size=1.0 / L,
        N=100,
        GBT_alpha=1.0,
        measure=measure,
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
    measure = "legs"
    L = random_16_input.shape[1]
    return HiPPO(
        max_length=L,
        step_size=1.0,
        N=100,
        GBT_alpha=0.5,
        measure=measure,
        dtype=jnp.float32,
        verbose=True,
    )


@pytest.fixture
def hippo_lsi_legs_bi(random_16_input):
    measure = "legs"
    L = random_16_input.shape[1]
    return HiPPO(
        max_length=L,
        step_size=1.0 / L,
        N=100,
        GBT_alpha=0.5,
        measure=measure,
        dtype=jnp.float32,
        verbose=True,
    )


# ----------
# -- legt --
# ----------


@pytest.fixture
def hippo_legt_bi(random_16_input):
    measure = "legt"
    L = random_16_input.shape[1]
    return HiPPO(
        max_length=L,
        step_size=1.0 / L,
        N=100,
        lambda_n=1.0,
        GBT_alpha=0.5,
        measure=measure,
        dtype=jnp.float32,
        verbose=True,
    )


# ----------
# -- lmu --
# ----------


@pytest.fixture
def hippo_lmu_bi(random_16_input):
    measure = "legt"
    L = random_16_input.shape[1]
    return HiPPO(
        max_length=L,
        step_size=1.0 / L,
        N=100,
        lambda_n=1.0,
        GBT_alpha=0.5,
        measure=measure,
        dtype=jnp.float32,
        verbose=True,
    )


# ----------
# -- lagt --
# ----------


@pytest.fixture
def hippo_lagt_bi(random_16_input):
    measure = "lagt"
    L = random_16_input.shape[1]
    return HiPPO(
        max_length=L,
        step_size=1.0 / L,
        N=100,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=0.5,
        measure=measure,
        dtype=jnp.float32,
        verbose=True,
    )


# ----------
# --- fru --
# ----------


@pytest.fixture
def hippo_fru_bi(random_16_input):
    measure = "fru"
    L = random_16_input.shape[1]
    return HiPPO(
        max_length=L,
        step_size=1.0 / L,
        N=100,
        GBT_alpha=0.5,
        measure=measure,
        dtype=jnp.float32,
        verbose=True,
    )


# ------------
# --- fout ---
# ------------


@pytest.fixture
def hippo_fout_bi(random_16_input):
    measure = "fout"
    L = random_16_input.shape[1]
    return HiPPO(
        max_length=L,
        step_size=1.0 / L,
        N=100,
        GBT_alpha=0.5,
        measure=measure,
        dtype=jnp.float32,
        verbose=True,
    )


# ------------
# --- foud ---
# ------------


@pytest.fixture
def hippo_foud_bi(random_16_input):
    measure = "foud"
    L = random_16_input.shape[1]
    return HiPPO(
        max_length=L,
        step_size=1.0 / L,
        N=100,
        GBT_alpha=0.5,
        measure=measure,
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
    measure = "legs"
    L = random_16_input.shape[1]
    return HiPPO(
        max_length=L,
        step_size=1.0,
        N=100,
        GBT_alpha=2.0,
        measure=measure,
        dtype=jnp.float32,
        verbose=True,
    )


@pytest.fixture
def hippo_lsi_legs_zoh(random_16_input):
    measure = "legs"
    L = random_16_input.shape[1]
    return HiPPO(
        max_length=L,
        step_size=1.0 / L,
        N=100,
        GBT_alpha=0.5,
        measure=measure,
        dtype=jnp.float32,
        verbose=True,
    )


# ----------
# -- legt --
# ----------


@pytest.fixture
def hippo_legt_zoh(random_16_input):
    measure = "legt"
    L = random_16_input.shape[1]
    return HiPPO(
        max_length=L,
        step_size=1.0 / L,
        N=100,
        lambda_n=1.0,
        GBT_alpha=2.0,
        measure=measure,
        dtype=jnp.float32,
        verbose=True,
    )


# ----------
# -- lmu --
# ----------


@pytest.fixture
def hippo_lmu_zoh(random_16_input):
    measure = "legt"
    L = random_16_input.shape[1]
    return HiPPO(
        max_length=L,
        step_size=1.0 / L,
        N=100,
        lambda_n=1.0,
        GBT_alpha=2.0,
        measure=measure,
        dtype=jnp.float32,
        verbose=True,
    )


# ----------
# -- lagt --
# ----------


@pytest.fixture
def hippo_lagt_zoh(random_16_input):
    measure = "lagt"
    L = random_16_input.shape[1]
    return HiPPO(
        max_length=L,
        step_size=1.0 / L,
        N=100,
        alpha=0.0,
        beta=1.0,
        GBT_alpha=2.0,
        measure=measure,
        dtype=jnp.float32,
        verbose=True,
    )


# ----------
# --- fru --
# ----------


@pytest.fixture
def hippo_fru_zoh(random_16_input):
    measure = "fru"
    L = random_16_input.shape[1]
    return HiPPO(
        max_length=L,
        step_size=1.0 / L,
        N=100,
        GBT_alpha=2.0,
        measure=measure,
        dtype=jnp.float32,
        verbose=True,
    )


# ------------
# --- fout ---
# ------------


@pytest.fixture
def hippo_fout_zoh(random_16_input):
    measure = "fout"
    L = random_16_input.shape[1]
    return HiPPO(
        max_length=L,
        step_size=1.0 / L,
        N=100,
        GBT_alpha=2.0,
        measure=measure,
        dtype=jnp.float32,
        verbose=True,
    )


# ------------
# --- foud ---
# ------------


@pytest.fixture
def hippo_foud_zoh(random_16_input):
    measure = "foud"
    L = random_16_input.shape[1]
    return HiPPO(
        max_length=L,
        step_size=1.0 / L,
        N=100,
        GBT_alpha=2.0,
        measure=measure,
        dtype=jnp.float32,
        verbose=True,
    )


# ----------------------------------------------------------------
# --------------------- Gu's Implementations ---------------------
# ----------------------------------------------------------------


@pytest.fixture
def gu_hippo_legs(random_16_input):
    measure = "legs"
    L = random_16_input.shape[1]
    return HiPPO_LegS(N=8, max_length=L, measure=measure, discretization="bilinear")


@pytest.fixture
def gu_hippo_legt(random_16_input):
    measure = "legt"
    A, B = gu_legt_matrices
    L = random_16_input.shape[1]
    return HiPPO_LegT(N=8, dt=1.0, discretization="bilinear", lambda_n=1.0)


@pytest.fixture
def gu_hippo_lmu(random_16_input):
    measure = "lmu"
    A, B = gu_legt_lmu_matrices
    L = random_16_input.shape[1]
    # raise NotImplementedError("HiPPO_LegT_LMU not implemented yet")
    return HiPPO_LegT(N=8, dt=1.0, discretization="bilinear", lambda_n=2.0)


@pytest.fixture
def gu_hippo_lagt(random_16_input):
    measure = "lagt"
    A, B = gu_lagt_matrices
    L = random_16_input.shape[1]
    raise NotImplementedError("HiPPO_LagT not implemented yet")
    return HiPPO(
        N=8,
        max_length=L,
        step=1.0 / L,
        GBT_alpha=0.5,
        seq_L=L,
        A=A,
        B=B,
        measure=measure,
    )


@pytest.fixture
def gu_hippo_fru(random_16_input):
    measure = "fru"
    A, B = gu_fru_matrices
    L = random_16_input.shape[1]
    raise NotImplementedError("HiPPO_FRU not implemented yet")
    return HiPPO(
        N=8,
        max_length=L,
        step=1.0 / L,
        GBT_alpha=0.5,
        seq_L=L,
        A=A,
        B=B,
        measure=measure,
    )


@pytest.fixture
def gu_hippo_fout(random_16_input):
    measure = "fout"
    A, B = gu_fout_matrices
    L = random_16_input.shape[1]
    raise NotImplementedError("HiPPO_FouT not implemented yet")
    return HiPPO(
        N=8,
        max_length=L,
        step=1.0 / L,
        GBT_alpha=0.5,
        seq_L=L,
        A=A,
        B=B,
        measure=measure,
    )


@pytest.fixture
def gu_hippo_foud(random_16_input):
    measure = "foud"
    A, B = gu_foud_matrices
    L = random_16_input.shape[1]
    raise NotImplementedError("HiPPO_FouD not implemented yet")
    return HiPPO(
        N=8,
        max_length=L,
        step=1.0 / L,
        GBT_alpha=0.5,
        seq_L=L,
        A=A,
        B=B,
        measure=measure,
    )


# --------------------
# -- Forward Euler --
# --------------------

# ----------
# -- legs --
# ----------


@pytest.fixture
def gu_hippo_lti_legs_fe(random_16_input):
    measure = "legs"
    L = random_16_input.shape[1]
    return HiPPO_LTI(
        N=100,
        method=measure,
        dt=1.0,
        T=1.0,
        discretization="forward",
        c=0.0,
    )


@pytest.fixture
def gu_hippo_lsi_legs_fe(random_16_input):
    measure = "legs"
    L = random_16_input.shape[1]
    return HiPPO_LSI(N=100, method=measure, max_length=L, discretization="forward")


# ----------
# -- legt --
# ----------


@pytest.fixture
def gu_hippo_lti_legt_fe(random_16_input):
    measure = "legt"
    L = random_16_input.shape[1]
    return HiPPO_LTI(
        N=100,
        method=measure,
        dt=1.0,
        T=1.0,
        discretization="forward",
        lambda_n=1.0,
        c=0.0,
    )


# ----------
# -- lmu --
# ----------


@pytest.fixture
def gu_hippo_lti_lmu_fe(random_16_input):
    measure = "lmu"
    L = random_16_input.shape[1]
    return HiPPO_LTI(
        N=100,
        method=measure,
        dt=1.0,
        T=1.0,
        discretization="forward",
        lambda_n=2.0,
        c=0.0,
    )


# ----------
# -- lagt --
# ----------


@pytest.fixture
def gu_hippo_lti_lagt_fe(random_16_input):
    measure = "lagt"
    L = random_16_input.shape[1]
    return HiPPO_LTI(
        N=100,
        method=measure,
        dt=1.0,
        T=1.0,
        discretization="forward",
        alpha=0.0,
        beta=1.0,
        c=0.0,
    )


# ----------
# --- fru --
# ----------


@pytest.fixture
def gu_hippo_lti_fru_fe(random_16_input):
    measure = "fru"
    L = random_16_input.shape[1]
    return HiPPO_LTI(
        N=100,
        method=measure,
        dt=1.0,
        T=1.0,
        discretization="forward",
        lambda_n=2.0,
        c=0.0,
    )


# ------------
# --- fout ---
# ------------


@pytest.fixture
def gu_hippo_lti_fout_fe(random_16_input):
    measure = "fout"
    L = random_16_input.shape[1]
    return HiPPO_LTI(
        N=100,
        method=measure,
        dt=1.0,
        T=1.0,
        discretization="forward",
        c=0.0,
    )


# ------------
# --- foud ---
# ------------


@pytest.fixture
def gu_hippo_lti_foud_fe(random_16_input):
    measure = "foud"
    L = random_16_input.shape[1]
    return HiPPO_LTI(
        N=100,
        method=measure,
        dt=1.0,
        T=1.0,
        discretization="forward",
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
    measure = "legs"
    L = random_16_input.shape[1]
    return HiPPO_LTI(
        N=100,
        method=measure,
        dt=1.0,
        T=1.0,
        discretization="backward",
        c=0.0,
    )


@pytest.fixture
def gu_hippo_lsi_legs_be(random_16_input):
    measure = "legs"
    L = random_16_input.shape[1]
    return HiPPO_LSI(N=100, method=measure, max_length=L, discretization="backward")


# ----------
# -- legt --
# ----------


@pytest.fixture
def gu_hippo_lti_legt_be(random_16_input):
    measure = "legt"
    L = random_16_input.shape[1]
    return HiPPO_LTI(
        N=100,
        method=measure,
        dt=1.0,
        T=1.0,
        discretization="backward",
        lambda_n=1.0,
        c=0.0,
    )


# ----------
# -- lmu --
# ----------


@pytest.fixture
def gu_hippo_lti_lmu_be(random_16_input):
    measure = "legt"
    L = random_16_input.shape[1]
    return HiPPO_LTI(
        N=100,
        method=measure,
        dt=1.0,
        T=1.0,
        discretization="backward",
        lambda_n=2.0,
        c=0.0,
    )


# ----------
# -- lagt --
# ----------


@pytest.fixture
def gu_hippo_lti_lagt_be(random_16_input):
    measure = "lagt"
    L = random_16_input.shape[1]
    return HiPPO_LTI(
        N=100,
        method=measure,
        dt=1.0,
        T=1.0,
        discretization="backward",
        alpha=0.0,
        beta=1.0,
        c=0.0,
    )


# ----------
# --- fru --
# ----------


@pytest.fixture
def gu_hippo_lti_fru_be(random_16_input):
    measure = "fru"
    L = random_16_input.shape[1]
    return HiPPO_LTI(
        N=100,
        method=measure,
        dt=1.0,
        T=1.0,
        discretization="backward",
        c=0.0,
    )


# ------------
# --- fout ---
# ------------


@pytest.fixture
def gu_hippo_lti_fout_be(random_16_input):
    measure = "fout"
    L = random_16_input.shape[1]
    return HiPPO_LTI(
        N=100,
        method=measure,
        dt=1.0,
        T=1.0,
        discretization="backward",
        c=0.0,
    )


# ------------
# --- foud ---
# ------------


@pytest.fixture
def gu_hippo_lti_foud_be(random_16_input):
    measure = "foud"
    L = random_16_input.shape[1]
    return HiPPO_LTI(
        N=100,
        method=measure,
        dt=1.0,
        T=1.0,
        discretization="backward",
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
    measure = "legs"
    L = random_16_input.shape[1]
    return HiPPO_LTI(
        N=100,
        method="legs",
        dt=1.0,
        T=1.0,
        discretization="bilinear",
        c=0.0,
    )


@pytest.fixture
def gu_hippo_lsi_legs_bi(random_16_input):
    measure = "legs"
    L = random_16_input.shape[1]
    return HiPPO_LSI(N=100, method="legs", max_length=L, discretization="bilinear")


# ----------
# -- legt --
# ----------


@pytest.fixture
def gu_hippo_lti_legt_bi(random_16_input):
    measure = "legt"
    L = random_16_input.shape[1]
    return HiPPO_LTI(
        N=100,
        method=measure,
        dt=1.0,
        T=1.0,
        discretization="bilinear",
        lambda_n=1.0,
        c=0.0,
    )


# ----------
# -- lmu --
# ----------


@pytest.fixture
def gu_hippo_lti_lmu_bi(random_16_input):
    measure = "legt"
    L = random_16_input.shape[1]
    return HiPPO_LTI(
        N=100,
        method=measure,
        dt=1.0,
        T=1.0,
        discretization="bilinear",
        lambda_n=2.0,
        c=0.0,
    )


# ----------
# -- lagt --
# ----------


@pytest.fixture
def gu_hippo_lti_lagt_bi(random_16_input):
    measure = "lagt"
    L = random_16_input.shape[1]
    return HiPPO_LTI(
        N=100,
        method=measure,
        dt=1.0,
        T=1.0,
        discretization="bilinear",
        alpha=0.0,
        beta=1.0,
        c=0.0,
    )


# ----------
# --- fru --
# ----------


@pytest.fixture
def gu_hippo_lti_fru_bi(random_16_input):
    measure = "fru"
    L = random_16_input.shape[1]
    return HiPPO_LTI(
        N=100,
        method=measure,
        dt=1.0,
        T=1.0,
        discretization="bilinear",
        c=0.0,
    )


# ------------
# --- fout ---
# ------------


@pytest.fixture
def gu_hippo_lti_fout_bi(random_16_input):
    measure = "fout"
    L = random_16_input.shape[1]
    return HiPPO_LTI(
        N=100,
        method=measure,
        dt=1.0,
        T=1.0,
        discretization="bilinear",
        c=0.0,
    )


# ------------
# --- foud ---
# ------------


@pytest.fixture
def gu_hippo_lti_foud_bi(random_16_input):
    measure = "foud"
    L = random_16_input.shape[1]
    return HiPPO_LTI(
        N=100,
        method=measure,
        dt=1.0,
        T=1.0,
        discretization="bilinear",
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
    measure = "legs"
    L = random_16_input.shape[1]
    return HiPPO_LTI(
        N=100,
        method="legs",
        dt=1.0,
        T=1.0,
        discretization="zoh",
        c=0.0,
    )


@pytest.fixture
def gu_hippo_lsi_legs_zoh(random_16_input):
    measure = "legs"
    L = random_16_input.shape[1]
    return HiPPO_LSI(N=100, method="legs", max_length=L, discretization="zoh")


# ----------
# -- legt --
# ----------


@pytest.fixture
def gu_hippo_lti_legt_zoh(random_16_input):
    measure = "legt"
    L = random_16_input.shape[1]
    return HiPPO_LTI(
        N=100,
        method=measure,
        dt=1.0,
        T=1.0,
        discretization="zoh",
        lambda_n=1.0,
        c=0.0,
    )


# ----------
# -- lmu --
# ----------


@pytest.fixture
def gu_hippo_lti_lmu_zoh(random_16_input):
    measure = "legt"
    L = random_16_input.shape[1]
    return HiPPO_LTI(
        N=100,
        method=measure,
        dt=1.0,
        T=1.0,
        discretization="zoh",
        lambda_n=2.0,
        c=0.0,
    )


# ----------
# -- lagt --
# ----------


@pytest.fixture
def gu_hippo_lti_lagt_zoh(random_16_input):
    measure = "lagt"
    L = random_16_input.shape[1]
    return HiPPO_LTI(
        N=100,
        method=measure,
        dt=1.0,
        T=1.0,
        discretization="zoh",
        alpha=0.0,
        beta=1.0,
        c=0.0,
    )


# ----------
# --- fru --
# ----------


@pytest.fixture
def gu_hippo_lti_fru_zoh(random_16_input):
    measure = "fru"
    L = random_16_input.shape[1]
    return HiPPO_LTI(
        N=100,
        method=measure,
        dt=1.0,
        T=1.0,
        discretization="zoh",
        c=0.0,
    )


# ------------
# --- fout ---
# ------------


@pytest.fixture
def gu_hippo_lti_fout_zoh(random_16_input):
    measure = "fout"
    L = random_16_input.shape[1]
    return HiPPO_LTI(
        N=100,
        method=measure,
        dt=1.0,
        T=1.0,
        discretization="zoh",
        c=0.0,
    )


# ------------
# --- foud ---
# ------------


@pytest.fixture
def gu_hippo_lti_foud_zoh(random_16_input):
    measure = "foud"
    L = random_16_input.shape[1]
    return HiPPO_LTI(
        N=100,
        method=measure,
        dt=1.0,
        T=1.0,
        discretization="zoh",
        c=0.0,
    )
