import jax.numpy as jnp
import pytest

from src.models.hippo.hr_transition import GuLowRankMatrix, HRTransMatrix
from src.models.hippo.transition import LowRankMatrix, TransMatrix
from src.tests.hippo_tests.hippo_utils import (
    random_1_input,
    random_16_input,
    random_32_input,
    random_64_input,
)

# ----------------------------------------------------------------
# --- Home Grown, Grass Fed, All Organic HiPPO Implementations ---
# ----------------------------------------------------------------

# ------------
# --- legs ---
# ------------
@pytest.fixture
def legs_matrices():
    the_measure = "legs"
    legs_matrices = TransMatrix(N=50, measure=the_measure)
    return legs_matrices.A, legs_matrices.B


@pytest.fixture
def nplr_legs():
    the_measure = "legs"
    rank = 1
    nplr_legs = LowRankMatrix(N=50, rank=rank, measure=the_measure, DPLR=False)
    return nplr_legs


@pytest.fixture
def dplr_legs():
    the_measure = "legs"
    rank = 1
    dplr_legs = LowRankMatrix(N=50, rank=rank, measure=the_measure, DPLR=True)
    return dplr_legs


# ------------
# --- legt ---
# ------------
@pytest.fixture
def legt_matrices():
    the_measure = "legt"
    legt_matrices = TransMatrix(N=50, measure=the_measure, lambda_n=1.0)
    return legt_matrices.A, legt_matrices.B


@pytest.fixture
def nplr_legt():
    the_measure = "legt"
    rank = 2
    nplr_legt = LowRankMatrix(
        N=50, rank=rank, measure=the_measure, lambda_n=1.0, DPLR=False
    )
    return nplr_legt


@pytest.fixture
def dplr_legt():
    the_measure = "legt"
    rank = 2
    dplr_legt = LowRankMatrix(
        N=50, rank=rank, measure=the_measure, lambda_n=1.0, DPLR=True
    )
    return dplr_legt


# ------------
# --- lmu ---
# ------------
@pytest.fixture
def legt_lmu_matrices():
    the_measure = "lmu"
    lmu_matrices = TransMatrix(
        N=50,
        measure=the_measure,
        lambda_n=2.0,
        alpha=0.0,
        beta=1.0,
        dtype=jnp.float32,
    )
    return lmu_matrices.A, lmu_matrices.B


@pytest.fixture
def nplr_lmu():
    the_measure = "lmu"
    rank = 2
    nplr_lmu = LowRankMatrix(
        N=50, rank=rank, measure=the_measure, lambda_n=2.0, DPLR=False
    )  # change lambda so resulting matrix is in the form of LMU
    return nplr_lmu


@pytest.fixture
def dplr_lmu():
    the_measure = "lmu"
    rank = 2
    dplr_lmu = LowRankMatrix(
        N=50, rank=rank, measure=the_measure, lambda_n=2.0, DPLR=True
    )  # change lambda so resulting matrix is in the form of LMU
    return dplr_lmu


# ------------
# --- lagt ---
# ------------
@pytest.fixture
def lagt_matrices():
    the_measure = "lagt"
    lagt_matrices = TransMatrix(
        N=50,
        measure=the_measure,
        alpha=0.0,  # change resulting tilt through alpha and beta
        beta=1.0,
    )  # change resulting tilt through alpha and beta
    return lagt_matrices.A, lagt_matrices.B


@pytest.fixture
def nplr_lagt():
    the_measure = "lagt"
    rank = 1
    nplr_lagt = LowRankMatrix(
        N=50,
        rank=rank,
        measure=the_measure,
        alpha=0.0,  # change resulting tilt through alpha and beta
        beta=1.0,
        DPLR=False,
    )  # change resulting tilt through alpha and beta
    return nplr_lagt


@pytest.fixture
def dplr_lagt():
    the_measure = "lagt"
    rank = 1
    dplr_lagt = LowRankMatrix(
        N=50,
        rank=rank,
        measure=the_measure,
        alpha=0.0,  # change resulting tilt through alpha and beta
        beta=1.0,
        DPLR=True,
    )  # change resulting tilt through alpha and beta
    return dplr_lagt


# ------------
# --- fru ---
# ------------
@pytest.fixture
def fru_matrices():
    the_measure = "fru"
    fru_matrices = TransMatrix(N=50, measure=the_measure)
    return fru_matrices.A, fru_matrices.B


@pytest.fixture
def nplr_fru():
    the_measure = "fru"
    rank = 1
    nplr_fru = LowRankMatrix(N=50, rank=rank, measure=the_measure, DPLR=False)
    return nplr_fru


@pytest.fixture
def dplr_fru():
    the_measure = "fru"
    rank = 1
    dplr_fru = LowRankMatrix(N=50, rank=rank, measure=the_measure, DPLR=True)
    return dplr_fru


# ------------
# --- fout ---
# ------------
@pytest.fixture
def fout_matrices():
    the_measure = "fout"
    fout_matrices = TransMatrix(N=50, measure=the_measure)
    return fout_matrices.A, fout_matrices.B


@pytest.fixture
def nplr_fout():
    the_measure = "fout"
    rank = 1
    nplr_fout = LowRankMatrix(N=50, rank=rank, measure=the_measure, DPLR=False)
    return nplr_fout


@pytest.fixture
def dplr_fout():
    the_measure = "fout"
    rank = 1
    dplr_fout = LowRankMatrix(N=50, rank=rank, measure=the_measure, DPLR=True)
    return dplr_fout


# ------------
# --- foud ---
# ------------
@pytest.fixture
def foud_matrices():
    the_measure = "foud"
    foud_matrices = TransMatrix(N=50, measure=the_measure)
    return foud_matrices.A, foud_matrices.B


@pytest.fixture
def nplr_foud():
    the_measure = "foud"
    rank = 1
    nplr_foud = LowRankMatrix(N=50, rank=rank, measure=the_measure, DPLR=False)
    return nplr_foud


@pytest.fixture
def dplr_foud():
    the_measure = "foud"
    rank = 1
    dplr_foud = LowRankMatrix(N=50, rank=rank, measure=the_measure, DPLR=True)
    return dplr_foud


# ----------------------------------------------------
# --------------- Gu's Implementations ---------------
# ----------------------------------------------------

# ------------
# --- legs ---
# ------------
@pytest.fixture
def hr_legs_matrices():
    the_measure = "legs"
    legs_matrices = HRTransMatrix(N=50, measure=the_measure)
    return legs_matrices.A, legs_matrices.B


@pytest.fixture
def hr_nplr_legs():
    the_measure = "legs"
    rank = 1
    hr_nplr_legs = GuLowRankMatrix(N=50, rank=rank, measure=the_measure, DPLR=False)
    return hr_nplr_legs


@pytest.fixture
def hr_dplr_legs():
    the_measure = "legs"
    rank = 1
    hr_dplr_legs = GuLowRankMatrix(N=50, rank=rank, measure=the_measure, DPLR=True)
    return hr_dplr_legs


# ------------
# --- legt ---
# ------------
@pytest.fixture
def hr_legt_matrices():
    the_measure = "legt"
    legt_matrices = HRTransMatrix(N=50, measure=the_measure, lambda_n=1.0)
    return legt_matrices.A, legt_matrices.B


@pytest.fixture
def hr_nplr_legt():
    the_measure = "legt"
    rank = 2
    hr_nplr_legt = GuLowRankMatrix(
        N=50, rank=rank, measure=the_measure, lambda_n=1.0, DPLR=False
    )
    return hr_nplr_legt


@pytest.fixture
def hr_dplr_legt():
    the_measure = "legt"
    rank = 2
    hr_dplr_legt = GuLowRankMatrix(
        N=50, rank=rank, measure=the_measure, lambda_n=1.0, DPLR=True
    )
    return hr_dplr_legt


# ------------
# ---- lmu ---
# ------------
@pytest.fixture
def hr_legt_lmu_matrices():
    the_measure = "lmu"
    lmu_matrices = HRTransMatrix(
        N=50, measure=the_measure, lambda_n=2.0, alpha=0.0, beta=1.0
    )
    return lmu_matrices.A, lmu_matrices.B


@pytest.fixture
def hr_nplr_lmu():
    the_measure = "lmu"
    rank = 2
    hr_nplr_lmu = GuLowRankMatrix(
        N=50, rank=rank, measure=the_measure, lambda_n=2.0, DPLR=False
    )  # change lambda so resulting matrix is in the form of LMU
    return hr_nplr_lmu


@pytest.fixture
def hr_dplr_lmu():
    the_measure = "lmu"
    rank = 2
    hr_dplr_lmu = GuLowRankMatrix(
        N=50, rank=rank, measure=the_measure, lambda_n=2.0, DPLR=True
    )  # change lambda so resulting matrix is in the form of LMU
    return hr_dplr_lmu


# ------------
# --- lagt ---
# ------------
@pytest.fixture
def hr_lagt_matrices():
    the_measure = "lagt"
    lagt_matrices = HRTransMatrix(
        N=50,
        measure=the_measure,
        alpha=0.0,  # change resulting tilt through alpha and beta
        beta=1.0,
    )  # change resulting tilt through alpha and beta
    return lagt_matrices.A, lagt_matrices.B


@pytest.fixture
def hr_nplr_lagt():
    the_measure = "lagt"
    rank = 1
    hr_nplr_lagt = GuLowRankMatrix(
        N=50,
        rank=rank,
        measure=the_measure,
        alpha=0.0,  # change resulting tilt through alpha and beta
        beta=1.0,
        DPLR=False,
    )  # change resulting tilt through alpha and beta
    return hr_nplr_lagt


@pytest.fixture
def hr_dplr_lagt():
    the_measure = "lagt"
    rank = 1
    hr_dplr_lagt = GuLowRankMatrix(
        N=50,
        rank=rank,
        measure=the_measure,
        alpha=0.0,  # change resulting tilt through alpha and beta
        beta=1.0,
        DPLR=True,
    )  # change resulting tilt through alpha and beta
    return hr_dplr_lagt


# ------------
# ---- fru ---
# ------------
@pytest.fixture
def hr_fru_matrices():
    the_measure = "fru"
    fru_matrices = HRTransMatrix(N=50, measure=the_measure)
    return fru_matrices.A, fru_matrices.B


@pytest.fixture
def hr_nplr_fru():
    the_measure = "fru"
    rank = 1
    hr_nplr_fru = GuLowRankMatrix(N=50, rank=rank, measure=the_measure, DPLR=False)
    return hr_nplr_fru


@pytest.fixture
def hr_dplr_fru():
    the_measure = "fru"
    rank = 1
    hr_dplr_fru = GuLowRankMatrix(N=50, rank=rank, measure=the_measure, DPLR=True)
    return hr_dplr_fru


# ------------
# --- fout ---
# ------------
@pytest.fixture
def hr_fout_matrices():
    the_measure = "fout"
    fout_matrices = HRTransMatrix(N=50, measure=the_measure)
    return fout_matrices.A, fout_matrices.B


@pytest.fixture
def hr_nplr_fout():
    the_measure = "fout"
    rank = 1
    hr_nplr_fout = GuLowRankMatrix(N=50, rank=rank, measure=the_measure, DPLR=False)
    return hr_nplr_fout


@pytest.fixture
def hr_dplr_fout():
    the_measure = "fout"
    rank = 1
    hr_dplr_fout = GuLowRankMatrix(N=50, rank=rank, measure=the_measure, DPLR=True)
    return hr_dplr_fout


# ------------
# --- foud ---
# ------------
@pytest.fixture
def hr_foud_matrices():
    the_measure = "foud"
    foud_matrices = HRTransMatrix(N=50, measure=the_measure)
    return foud_matrices.A, foud_matrices.B


@pytest.fixture
def hr_nplr_foud():
    the_measure = "foud"
    rank = 1
    hr_nplr_foud = GuLowRankMatrix(N=50, rank=rank, measure=the_measure, DPLR=False)
    return hr_nplr_foud


@pytest.fixture
def hr_dplr_foud():
    the_measure = "foud"
    rank = 1
    hr_dplr_foud = GuLowRankMatrix(N=50, rank=rank, measure=the_measure, DPLR=True)
    return hr_dplr_foud
