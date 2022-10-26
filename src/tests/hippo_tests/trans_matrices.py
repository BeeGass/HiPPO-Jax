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
    return nplr_legs


@pytest.fixture
def dplr_legs():
    the_measure = "legs"
    rank = 1
    dplr_legs = LowRankMatrix(N=8, rank=rank, measure=the_measure, DPLR=True)
    return dplr_legs


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
    return nplr_legt


@pytest.fixture
def dplr_legt():
    the_measure = "legt"
    rank = 2
    dplr_legt = LowRankMatrix(
        N=8, rank=rank, measure=the_measure, lambda_n=1.0, DPLR=True
    )
    return dplr_legt


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
    return nplr_lmu


@pytest.fixture
def dplr_lmu():
    the_measure = "legt"
    rank = 2
    dplr_lmu = LowRankMatrix(
        N=8, rank=rank, measure=the_measure, lambda_n=2.0, DPLR=True
    )  # change lambda so resulting matrix is in the form of LMU
    return dplr_lmu


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
    return nplr_lagt


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
    return dplr_lagt


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
    return nplr_fru


@pytest.fixture
def dplr_fru():
    the_measure = "fourier"
    fourier_type = "fru"
    rank = 1
    dplr_fru = LowRankMatrix(
        N=8, rank=rank, measure=the_measure, fourier_type=fourier_type, DPLR=True
    )
    return dplr_fru


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
    return nplr_fout


@pytest.fixture
def dplr_fout():
    the_measure = "fourier"
    fourier_type = "fout"
    rank = 1
    dplr_fout = LowRankMatrix(
        N=8, rank=rank, measure=the_measure, fourier_type=fourier_type, DPLR=True
    )
    return dplr_fout


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
    return nplr_foud


@pytest.fixture
def dplr_foud():
    the_measure = "fourier"
    fourier_type = "foud"
    rank = 1
    dplr_foud = LowRankMatrix(
        N=8, rank=rank, measure=the_measure, fourier_type=fourier_type, DPLR=True
    )
    return dplr_foud


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
    return gu_nplr_legs


@pytest.fixture
def gu_dplr_legs():
    the_measure = "legs"
    rank = 1
    gu_dplr_legs = GuLowRankMatrix(N=8, rank=rank, measure=the_measure, DPLR=True)
    return gu_dplr_legs


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
    return gu_nplr_legt


@pytest.fixture
def gu_dplr_legt():
    the_measure = "legt"
    rank = 2
    gu_dplr_legt = GuLowRankMatrix(
        N=8, rank=rank, measure=the_measure, lambda_n=1.0, DPLR=True
    )
    return gu_dplr_legt


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
    return gu_nplr_lmu


@pytest.fixture
def gu_dplr_lmu():
    the_measure = "legt"
    rank = 2
    gu_dplr_lmu = GuLowRankMatrix(
        N=8, rank=rank, measure=the_measure, lambda_n=2.0, DPLR=True
    )  # change lambda so resulting matrix is in the form of LMU
    return gu_dplr_lmu


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
    return gu_nplr_lagt


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
    return gu_dplr_lagt


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
    return gu_nplr_fru


@pytest.fixture
def gu_dplr_fru():
    the_measure = "fourier"
    fourier_type = "fru"
    rank = 1
    gu_dplr_fru = GuLowRankMatrix(
        N=8, rank=rank, measure=the_measure, fourier_type=fourier_type, DPLR=True
    )
    return gu_dplr_fru


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
    return gu_nplr_fout


@pytest.fixture
def gu_dplr_fout():
    the_measure = "fourier"
    fourier_type = "fout"
    rank = 1
    gu_dplr_fout = GuLowRankMatrix(
        N=8, rank=rank, measure=the_measure, fourier_type=fourier_type, DPLR=True
    )
    return gu_dplr_fout


# ------------
# --- foud ---
# ------------
@pytest.fixture
def gu_foud_matrices():
    the_measure = "fourier"
    fourier_type = "foud"
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
    return gu_nplr_foud


@pytest.fixture
def gu_dplr_foud():
    the_measure = "fourier"
    fourier_type = "foud"
    rank = 1
    gu_dplr_foud = GuLowRankMatrix(
        N=8, rank=rank, measure=the_measure, fourier_type=fourier_type, DPLR=True
    )
    return gu_dplr_foud
