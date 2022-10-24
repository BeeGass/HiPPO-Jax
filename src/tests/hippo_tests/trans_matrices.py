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

# --- legs ---
@pytest.fixture
def legs_matrices():
    the_measure = "legs"
    legs_matrices = TransMatrix(N=8, measure=the_measure)
    return legs_matrices.A_matrix, legs_matrices.B_matrix


@pytest.fixture
def nplr_legs():
    the_measure = "legs"
    nplr_legs = LowRankMatrix(N=8, measure=the_measure, DPLR=False)
    return nplr_legs.A_matrix, nplr_legs.B_matrix


@pytest.fixture
def dplr_legs():
    the_measure = "legs"
    dplr_legs = LowRankMatrix(N=8, measure=the_measure, DPLR=True)
    return dplr_legs.A_matrix, dplr_legs.B_matrix


# --- legt ---
@pytest.fixture
def legt_matrices():
    the_measure = "legt"
    legt_matrices = TransMatrix(N=8, measure=the_measure, lambda_n=1.0)
    return legt_matrices.A_matrix, legt_matrices.B_matrix


@pytest.fixture
def nplr_legt():
    the_measure = "legt"
    nplr_legt = LowRankMatrix(N=8, measure=the_measure, lambda_n=1.0, DPLR=False)
    return nplr_legt.A_matrix, nplr_legt.B_matrix


@pytest.fixture
def dplr_legt():
    the_measure = "legt"
    dplr_legt = LowRankMatrix(N=8, measure=the_measure, lambda_n=1.0, DPLR=True)
    return dplr_legt.A_matrix, dplr_legt.B_matrix


# --- lmu ---
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
    nplr_lmu = LowRankMatrix(
        N=8, measure=the_measure, lambda_n=2.0, DPLR=False
    )  # change lambda so resulting matrix is in the form of LMU
    return nplr_lmu.A_matrix, nplr_lmu.B_matrix


@pytest.fixture
def dplr_lmu():
    the_measure = "legt"
    dplr_lmu = LowRankMatrix(
        N=8, measure=the_measure, lambda_n=2.0, DPLR=True
    )  # change lambda so resulting matrix is in the form of LMU
    return dplr_lmu.A_matrix, dplr_lmu.B_matrix


# --- lagt ---
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
    nplr_lagt = LowRankMatrix(
        N=8,
        measure=the_measure,
        alpha=0.0,  # change resulting tilt through alpha and beta
        beta=1.0,
        DPLR=False,
    )  # change resulting tilt through alpha and beta
    return nplr_lagt.A_matrix, nplr_lagt.B_matrix


@pytest.fixture
def dplr_lagt():
    the_measure = "lagt"
    dplr_lagt = LowRankMatrix(
        N=8,
        measure=the_measure,
        alpha=0.0,  # change resulting tilt through alpha and beta
        beta=1.0,
        DPLR=True,
    )  # change resulting tilt through alpha and beta
    return dplr_lagt.A_matrix, dplr_lagt.B_matrix


# --- fru ---
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
    nplr_fru = LowRankMatrix(
        N=8, measure=the_measure, fourier_type=fourier_type, DPLR=False
    )
    return nplr_fru.A_matrix, nplr_fru.B_matrix


@pytest.fixture
def dplr_fru():
    the_measure = "fourier"
    fourier_type = "fru"
    dplr_fru = LowRankMatrix(
        N=8, measure=the_measure, fourier_type=fourier_type, DPLR=True
    )
    return dplr_fru.A_matrix, dplr_fru.B_matrix


# --- fout ---
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
    nplr_fout = LowRankMatrix(
        N=8, measure=the_measure, fourier_type=fourier_type, DPLR=False
    )
    return nplr_fout.A_matrix, nplr_fout.B_matrix


@pytest.fixture
def dplr_fout():
    the_measure = "fourier"
    fourier_type = "fout"
    dplr_fout = LowRankMatrix(
        N=8, measure=the_measure, fourier_type=fourier_type, DPLR=True
    )
    return dplr_fout.A_matrix, dplr_fout.B_matrix


# --- foud ---
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
    nplr_foud = LowRankMatrix(
        N=8, measure=the_measure, fourier_type=fourier_type, DPLR=False
    )
    return nplr_foud.A_matrix, nplr_foud.B_matrix


@pytest.fixture
def dplr_foud():
    the_measure = "fourier"
    fourier_type = "foud"
    dplr_foud = LowRankMatrix(
        N=8, measure=the_measure, fourier_type=fourier_type, DPLR=True
    )
    return dplr_foud.A_matrix, dplr_foud.B_matrix


# ----------------------------
# --- Gu's Implementations ---
# ----------------------------

# --- legs ---
@pytest.fixture
def gu_legs_matrices():
    the_measure = "legs"
    legs_matrices = GuTransMatrix(N=8, measure=the_measure)
    return legs_matrices.A_matrix, legs_matrices.B_matrix


@pytest.fixture
def gu_nplr_legs():
    the_measure = "legs"
    gu_nplr_legs = GuLowRankMatrix(N=8, measure=the_measure)
    return gu_nplr_legs.A_matrix, gu_nplr_legs.B_matrix


@pytest.fixture
def gu_dplr_legs():
    the_measure = "legs"
    gu_dplr_legs = GuLowRankMatrix(N=8, measure=the_measure)
    return gu_dplr_legs.A_matrix, gu_dplr_legs.B_matrix


# --- legt ---
@pytest.fixture
def gu_legt_matrices():
    the_measure = "legt"
    legt_matrices = GuTransMatrix(N=8, measure=the_measure, lambda_n=1.0)
    return legt_matrices.A_matrix, legt_matrices.B_matrix


@pytest.fixture
def gu_nplr_legt():
    the_measure = "legt"
    gu_nplr_legt = GuLowRankMatrix(N=8, measure=the_measure, lambda_n=1.0)
    return gu_nplr_legt.A_matrix, gu_nplr_legt.B_matrix


@pytest.fixture
def gu_dplr_legt():
    the_measure = "legt"
    gu_dplr_legt = GuLowRankMatrix(N=8, measure=the_measure, lambda_n=1.0)
    return gu_dplr_legt.A_matrix, gu_dplr_legt.B_matrix


# --- lmu ---
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
    gu_nplr_lmu = GuLowRankMatrix(
        N=8, measure=the_measure, lambda_n=2.0
    )  # change lambda so resulting matrix is in the form of LMU
    return gu_nplr_lmu.A_matrix, gu_nplr_lmu.B_matrix


@pytest.fixture
def gu_dplr_lmu():
    the_measure = "legt"
    gu_dplr_lmu = GuLowRankMatrix(
        N=8, measure=the_measure, lambda_n=2.0
    )  # change lambda so resulting matrix is in the form of LMU
    return gu_dplr_lmu.A_matrix, gu_dplr_lmu.B_matrix


# --- lagt ---
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
    gu_nplr_lagt = GuLowRankMatrix(
        N=8,
        measure=the_measure,
        alpha=0.0,  # change resulting tilt through alpha and beta
        beta=1.0,
    )  # change resulting tilt through alpha and beta
    return gu_nplr_lagt.A_matrix, gu_nplr_lagt.B_matrix


@pytest.fixture
def gu_dplr_lagt():
    the_measure = "lagt"
    gu_dplr_lagt = GuLowRankMatrix(
        N=8,
        measure=the_measure,
        alpha=0.0,  # change resulting tilt through alpha and beta
        beta=1.0,
    )  # change resulting tilt through alpha and beta
    return gu_dplr_lagt.A_matrix, gu_dplr_lagt.B_matrix


# --- fru ---
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
    gu_nplr_fru = GuLowRankMatrix(N=8, measure=the_measure, fourier_type=fourier_type)
    return gu_nplr_fru.A_matrix, gu_nplr_fru.B_matrix


@pytest.fixture
def gu_dplr_fru():
    the_measure = "fourier"
    fourier_type = "fru"
    gu_dplr_fru = GuLowRankMatrix(N=8, measure=the_measure, fourier_type=fourier_type)
    return gu_dplr_fru.A_matrix, gu_dplr_fru.B_matrix


# --- fout ---
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
    gu_nplr_fout = GuLowRankMatrix(N=8, measure=the_measure, fourier_type=fourier_type)
    return gu_nplr_fout.A_matrix, gu_nplr_fout.B_matrix


@pytest.fixture
def gu_dplr_fout():
    the_measure = "fourier"
    fourier_type = "fout"
    gu_dplr_fout = GuLowRankMatrix(N=8, measure=the_measure, fourier_type=fourier_type)
    return gu_dplr_fout.A_matrix, gu_dplr_fout.B_matrix


# --- foud ---
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
    gu_nplr_foud = GuLowRankMatrix(N=8, measure=the_measure, fourier_type=fourier_type)
    return gu_nplr_foud.A_matrix, gu_nplr_foud.B_matrix


@pytest.fixture
def gu_dplr_foud():
    the_measure = "fourier"
    fourier_type = "foud"
    gu_dplr_foud = GuLowRankMatrix(N=8, measure=the_measure, fourier_type=fourier_type)
    return gu_dplr_foud.A_matrix, gu_dplr_foud.B_matrix
