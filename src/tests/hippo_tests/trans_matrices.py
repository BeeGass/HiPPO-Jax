import pytest
from src.models.hippo.gu_transition import GuTransMatrix, GuLowRankMatrix
from src.models.hippo.transition import TransMatrix, LowRankMatrix
from src.tests.hippo_tests.hippo_utils import (
    random_input,
    ones_input,
    zeros_input,
    desc_input,
)

# ----------------------------------------------------------
# --- Home Grown, Grass Fed, All Organic Implementations ---
# ----------------------------------------------------------

# ------------
# --- legs ---
# ------------
@pytest.fixture
def legs_matrices():
    the_measure = "legs"
    legs_matrices = TransMatrix(N=8, measure=the_measure)
    return legs_matrices.A_matrix, legs_matrices.B_matrix


@pytest.fixture
def nplr_legs():
    the_measure = "legs"
    rank = 1
    nplr_legs = LowRankMatrix(N=8, rank=rank, measure=the_measure, DPLR=False)
    return nplr_legs.Lambda, nplr_legs.P, nplr_legs.B, nplr_legs.V


@pytest.fixture
def dplr_legs():
    the_measure = "legs"
    rank = 1
    dplr_legs = LowRankMatrix(N=8, rank=rank, measure=the_measure, DPLR=True)
    return dplr_legs.Lambda, dplr_legs.P, dplr_legs.B, dplr_legs.V


# ------------
# --- legt ---
# ------------
@pytest.fixture
def legt_matrices():
    the_measure = "legt"
    legt_matrices = TransMatrix(N=8, measure=the_measure, lambda_n=1.0)
    return legt_matrices.A_matrix, legt_matrices.B_matrix


@pytest.fixture
def nplr_legt():
    the_measure = "legt"
    rank = 2
    nplr_legt = LowRankMatrix(
        N=8, rank=rank, measure=the_measure, lambda_n=1.0, DPLR=False
    )
    return nplr_legt.Lambda, nplr_legt.P, nplr_legt.B, nplr_legt.V


@pytest.fixture
def dplr_legt():
    the_measure = "legt"
    rank = 2
    dplr_legt = LowRankMatrix(
        N=8, rank=rank, measure=the_measure, lambda_n=1.0, DPLR=True
    )
    return dplr_legt.Lambda, dplr_legt.P, dplr_legt.B, dplr_legt.V


# ------------
# --- lmu ---
# ------------
@pytest.fixture
def legt_lmu_matrices():
    the_measure = "legt"
    lmu_matrices = TransMatrix(
        N=8, measure=the_measure, lambda_n=2.0
    )  # change lambda so resulting matrix is in the form of LMU
    return lmu_matrices.A_matrix, lmu_matrices.B_matrix


@pytest.fixture
def nplr_lmu():
    the_measure = "legt"
    rank = 2
    nplr_lmu = LowRankMatrix(
        N=8, rank=rank, measure=the_measure, lambda_n=2.0, DPLR=False
    )  # change lambda so resulting matrix is in the form of LMU
    return nplr_lmu.Lambda, nplr_lmu.P, nplr_lmu.B, nplr_lmu.V


@pytest.fixture
def dplr_lmu():
    the_measure = "legt"
    rank = 2
    dplr_lmu = LowRankMatrix(
        N=8, rank=rank, measure=the_measure, lambda_n=2.0, DPLR=True
    )  # change lambda so resulting matrix is in the form of LMU
    return dplr_lmu.Lambda, dplr_lmu.P, dplr_lmu.B, dplr_lmu.V


# ------------
# --- lagt ---
# ------------
@pytest.fixture
def lagt_matrices():
    the_measure = "lagt"
    lagt_matrices = TransMatrix(
        N=8,
        measure=the_measure,
        alpha=0.0,  # change resulting tilt through alpha and beta
        beta=1.0,
    )  # change resulting tilt through alpha and beta
    return lagt_matrices.A_matrix, lagt_matrices.B_matrix


@pytest.fixture
def nplr_lagt():
    the_measure = "lagt"
    rank = 1
    nplr_lagt = LowRankMatrix(
        N=8,
        rank=rank,
        measure=the_measure,
        alpha=0.0,  # change resulting tilt through alpha and beta
        beta=1.0,
        DPLR=False,
    )  # change resulting tilt through alpha and beta
    return nplr_lagt.Lambda, nplr_lagt.P, nplr_lagt.B, nplr_lagt.V


@pytest.fixture
def dplr_lagt():
    the_measure = "lagt"
    rank = 1
    dplr_lagt = LowRankMatrix(
        N=8,
        rank=rank,
        measure=the_measure,
        alpha=0.0,  # change resulting tilt through alpha and beta
        beta=1.0,
        DPLR=True,
    )  # change resulting tilt through alpha and beta
    return dplr_lagt.Lambda, dplr_lagt.P, dplr_lagt.B, dplr_lagt.V


# ------------
# --- fru ---
# ------------
@pytest.fixture
def fru_matrices():
    the_measure = "fourier"
    fourier_type = "fru"
    fru_matrices = TransMatrix(N=8, measure=the_measure, fourier_type=fourier_type)
    return fru_matrices.A_matrix, fru_matrices.B_matrix


@pytest.fixture
def nplr_fru():
    the_measure = "fourier"
    fourier_type = "fru"
    rank = 1
    nplr_fru = LowRankMatrix(
        N=8, rank=rank, measure=the_measure, fourier_type=fourier_type, DPLR=False
    )
    return nplr_fru.Lambda, nplr_fru.P, nplr_fru.B, nplr_fru.V


@pytest.fixture
def dplr_fru():
    the_measure = "fourier"
    fourier_type = "fru"
    rank = 1
    dplr_fru = LowRankMatrix(
        N=8, rank=rank, measure=the_measure, fourier_type=fourier_type, DPLR=True
    )
    return dplr_fru.Lambda, dplr_fru.P, dplr_fru.B, dplr_fru.V


# ------------
# --- fout ---
# ------------
@pytest.fixture
def fout_matrices():
    the_measure = "fourier"
    fourier_type = "fout"
    fout_matrices = TransMatrix(N=8, measure=the_measure, fourier_type=fourier_type)
    return fout_matrices.A_matrix, fout_matrices.B_matrix


@pytest.fixture
def nplr_fout():
    the_measure = "fourier"
    fourier_type = "fout"
    rank = 1
    nplr_fout = LowRankMatrix(
        N=8, rank=rank, measure=the_measure, fourier_type=fourier_type, DPLR=False
    )
    return nplr_fout.Lambda, nplr_fout.P, nplr_fout.B, nplr_fout.V


@pytest.fixture
def dplr_fout():
    the_measure = "fourier"
    fourier_type = "fout"
    rank = 1
    dplr_fout = LowRankMatrix(
        N=8, rank=rank, measure=the_measure, fourier_type=fourier_type, DPLR=True
    )
    return dplr_fout.Lambda, dplr_fout.P, dplr_fout.B, dplr_fout.V


# ------------
# --- foud ---
# ------------
@pytest.fixture
def foud_matrices():
    the_measure = "fourier"
    fourier_type = "foud"
    foud_matrices = TransMatrix(N=8, measure=the_measure, fourier_type=fourier_type)
    return foud_matrices.A_matrix, foud_matrices.B_matrix


@pytest.fixture
def nplr_foud():
    the_measure = "fourier"
    fourier_type = "foud"
    rank = 1
    nplr_foud = LowRankMatrix(
        N=8, rank=rank, measure=the_measure, fourier_type=fourier_type, DPLR=False
    )
    return nplr_foud.Lambda, nplr_foud.P, nplr_foud.B, nplr_foud.V


@pytest.fixture
def dplr_foud():
    the_measure = "fourier"
    fourier_type = "foud"
    rank = 1
    dplr_foud = LowRankMatrix(
        N=8, rank=rank, measure=the_measure, fourier_type=fourier_type, DPLR=True
    )
    return dplr_foud.Lambda, dplr_foud.P, dplr_foud.B, dplr_foud.V


# ----------------------------------------------------
# --------------- Gu's Implementations ---------------
# ----------------------------------------------------

# ------------
# --- legs ---
# ------------
@pytest.fixture
def gu_legs_matrices():
    the_measure = "legs"
    legs_matrices = GuTransMatrix(N=8, measure=the_measure)
    return legs_matrices.A_matrix, legs_matrices.B_matrix


@pytest.fixture
def gu_nplr_legs():
    the_measure = "legs"
    rank = 1
    gu_nplr_legs = GuLowRankMatrix(N=8, rank=rank, measure=the_measure, DPLR=False)
    return (
        gu_nplr_legs.Lambda,
        gu_nplr_legs.P,
        gu_nplr_legs.B,
        gu_nplr_legs.V,
    )


@pytest.fixture
def gu_dplr_legs():
    the_measure = "legs"
    rank = 1
    gu_dplr_legs = GuLowRankMatrix(N=8, rank=rank, measure=the_measure, DPLR=True)
    return (
        gu_dplr_legs.Lambda,
        gu_dplr_legs.P,
        gu_dplr_legs.B,
        gu_dplr_legs.V,
    )


# ------------
# --- legt ---
# ------------
@pytest.fixture
def gu_legt_matrices():
    the_measure = "legt"
    legt_matrices = GuTransMatrix(N=8, measure=the_measure, lambda_n=1.0)
    return legt_matrices.A_matrix, legt_matrices.B_matrix


@pytest.fixture
def gu_nplr_legt():
    the_measure = "legt"
    rank = 2
    gu_nplr_legt = GuLowRankMatrix(
        N=8, rank=rank, measure=the_measure, lambda_n=1.0, DPLR=False
    )
    return (
        gu_nplr_legt.Lambda,
        gu_nplr_legt.P,
        gu_nplr_legt.B,
        gu_nplr_legt.V,
    )


@pytest.fixture
def gu_dplr_legt():
    the_measure = "legt"
    rank = 2
    gu_dplr_legt = GuLowRankMatrix(
        N=8, rank=rank, measure=the_measure, lambda_n=1.0, DPLR=True
    )
    return (
        gu_dplr_legt.Lambda,
        gu_dplr_legt.P,
        gu_dplr_legt.B,
        gu_dplr_legt.V,
    )


# ------------
# ---- lmu ---
# ------------
@pytest.fixture
def gu_legt_lmu_matrices():
    the_measure = "legt"
    lmu_matrices = GuTransMatrix(
        N=8, measure=the_measure, lambda_n=2.0
    )  # change lambda so resulting matrix is in the form of LMU
    return lmu_matrices.A_matrix, lmu_matrices.B_matrix


@pytest.fixture
def gu_nplr_lmu():
    the_measure = "legt"
    rank = 2
    gu_nplr_lmu = GuLowRankMatrix(
        N=8, rank=rank, measure=the_measure, lambda_n=2.0, DPLR=False
    )  # change lambda so resulting matrix is in the form of LMU
    return (
        gu_nplr_lmu.Lambda,
        gu_nplr_lmu.P,
        gu_nplr_lmu.B,
        gu_nplr_lmu.V,
    )


@pytest.fixture
def gu_dplr_lmu():
    the_measure = "legt"
    rank = 2
    gu_dplr_lmu = GuLowRankMatrix(
        N=8, rank=rank, measure=the_measure, lambda_n=2.0, DPLR=True
    )  # change lambda so resulting matrix is in the form of LMU
    return (
        gu_dplr_lmu.Lambda,
        gu_dplr_lmu.P,
        gu_dplr_lmu.B,
        gu_dplr_lmu.V,
    )


# ------------
# --- lagt ---
# ------------
@pytest.fixture
def gu_lagt_matrices():
    the_measure = "lagt"
    lagt_matrices = GuTransMatrix(
        N=8,
        measure=the_measure,
        alpha=0.0,  # change resulting tilt through alpha and beta
        beta=1.0,
    )  # change resulting tilt through alpha and beta
    return lagt_matrices.A_matrix, lagt_matrices.B_matrix


@pytest.fixture
def gu_nplr_lagt():
    the_measure = "lagt"
    rank = 1
    gu_nplr_lagt = GuLowRankMatrix(
        N=8,
        rank=rank,
        measure=the_measure,
        alpha=0.0,  # change resulting tilt through alpha and beta
        beta=1.0,
        DPLR=False,
    )  # change resulting tilt through alpha and beta
    return (
        gu_nplr_lagt.Lambda,
        gu_nplr_lagt.P,
        gu_nplr_lagt.B,
        gu_nplr_lagt.V,
    )


@pytest.fixture
def gu_dplr_lagt():
    the_measure = "lagt"
    rank = 1
    gu_dplr_lagt = GuLowRankMatrix(
        N=8,
        rank=rank,
        measure=the_measure,
        alpha=0.0,  # change resulting tilt through alpha and beta
        beta=1.0,
        DPLR=True,
    )  # change resulting tilt through alpha and beta
    return (
        gu_dplr_lagt.Lambda,
        gu_dplr_lagt.P,
        gu_dplr_lagt.B,
        gu_dplr_lagt.V,
    )


# ------------
# ---- fru ---
# ------------
@pytest.fixture
def gu_fru_matrices():
    the_measure = "fourier"
    fourier_type = "fru"
    fru_matrices = GuTransMatrix(N=8, measure=the_measure, fourier_type=fourier_type)
    return fru_matrices.A_matrix, fru_matrices.B_matrix


@pytest.fixture
def gu_nplr_fru():
    the_measure = "fourier"
    fourier_type = "fru"
    rank = 1
    gu_nplr_fru = GuLowRankMatrix(
        N=8, rank=rank, measure=the_measure, fourier_type=fourier_type, DPLR=False
    )
    return (
        gu_nplr_fru.Lambda,
        gu_nplr_fru.P,
        gu_nplr_fru.B,
        gu_nplr_fru.V,
    )


@pytest.fixture
def gu_dplr_fru():
    the_measure = "fourier"
    fourier_type = "fru"
    rank = 1
    gu_dplr_fru = GuLowRankMatrix(
        N=8, rank=rank, measure=the_measure, fourier_type=fourier_type, DPLR=True
    )
    return (
        gu_dplr_fru.Lambda,
        gu_dplr_fru.P,
        gu_dplr_fru.B,
        gu_dplr_fru.V,
    )


# ------------
# --- fout ---
# ------------
@pytest.fixture
def gu_fout_matrices():
    the_measure = "fourier"
    fourier_type = "fout"
    fout_matrices = GuTransMatrix(N=8, measure=the_measure, fourier_type=fourier_type)
    return fout_matrices.A_matrix, fout_matrices.B_matrix


@pytest.fixture
def gu_nplr_fout():
    the_measure = "fourier"
    fourier_type = "fout"
    rank = 1
    gu_nplr_fout = GuLowRankMatrix(
        N=8, rank=rank, measure=the_measure, fourier_type=fourier_type, DPLR=False
    )
    return (
        gu_nplr_fout.Lambda,
        gu_nplr_fout.P,
        gu_nplr_fout.B,
        gu_nplr_fout.V,
    )


@pytest.fixture
def gu_dplr_fout():
    the_measure = "fourier"
    fourier_type = "fout"
    rank = 1
    gu_dplr_fout = GuLowRankMatrix(
        N=8, rank=rank, measure=the_measure, fourier_type=fourier_type, DPLR=True
    )
    return (
        gu_dplr_fout.Lambda,
        gu_dplr_fout.P,
        gu_dplr_fout.B,
        gu_dplr_fout.V,
    )


# ------------
# --- foud ---
# ------------
@pytest.fixture
def gu_foud_matrices():
    the_measure = "fourier"
    fourier_type = "fourd"
    foud_matrices = GuTransMatrix(N=8, measure=the_measure, fourier_type=fourier_type)
    return foud_matrices.A_matrix, foud_matrices.B_matrix


@pytest.fixture
def gu_nplr_foud():
    the_measure = "fourier"
    fourier_type = "foud"
    rank = 1
    gu_nplr_foud = GuLowRankMatrix(
        N=8, rank=rank, measure=the_measure, fourier_type=fourier_type, DPLR=False
    )
    return (
        gu_nplr_foud.Lambda,
        gu_nplr_foud.P,
        gu_nplr_foud.B,
        gu_nplr_foud.V,
    )


@pytest.fixture
def gu_dplr_foud():
    the_measure = "fourier"
    fourier_type = "foud"
    rank = 1
    gu_dplr_foud = GuLowRankMatrix(
        N=8, rank=rank, measure=the_measure, fourier_type=fourier_type, DPLR=True
    )
    return (
        gu_dplr_foud.Lambda,
        gu_dplr_foud.P,
        gu_dplr_foud.B,
        gu_dplr_foud.V,
    )
