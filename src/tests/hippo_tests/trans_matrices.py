import pytest
from src.models.hippo.gu_transition import GuTransMatrix
from src.models.hippo.transition import TransMatrix
from src.tests.hippo_tests.hippo_utils import (
    random_input,
    ones_input,
    zeros_input,
    desc_input,
)

# ----------------------------------------------------------
# --- Home Grown, Grass Fed, All Organic Implementations ---
# ----------------------------------------------------------


@pytest.fixture
def legs_matrices():
    the_measure = "legs"
    legs_matrices = TransMatrix(N=8, measure=the_measure)
    return legs_matrices.A_matrix, legs_matrices.B_matrix


@pytest.fixture
def legt_matrices():
    the_measure = "legt"
    legt_matrices = TransMatrix(N=8, measure=the_measure, lambda_n=1.0)
    return legt_matrices.A_matrix, legt_matrices.B_matrix


@pytest.fixture
def legt_lmu_matrices():
    the_measure = "legt"
    lmu_matrices = TransMatrix(
        N=8, measure=the_measure, lambda_n=2.0
    )  # change lambda so resulting matrix is in the form of LMU
    return lmu_matrices.A_matrix, lmu_matrices.B_matrix


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
def fru_matrices():
    the_measure = "fourier"
    fourier_type = "fru"
    fru_matrices = TransMatrix(N=8, measure=the_measure, fourier_type=fourier_type)
    return fru_matrices.A_matrix, fru_matrices.B_matrix


@pytest.fixture
def fout_matrices():
    the_measure = "fourier"
    fourier_type = "fout"
    fout_matrices = TransMatrix(N=8, measure=the_measure, fourier_type=fourier_type)
    return fout_matrices.A_matrix, fout_matrices.B_matrix


@pytest.fixture
def fourd_matrices():
    the_measure = "fourier"
    fourier_type = "fourd"
    fourd_matrices = TransMatrix(N=8, measure=the_measure, fourier_type=fourier_type)
    return fourd_matrices.A_matrix, fourd_matrices.B_matrix


# ----------------------------
# --- Gu's Implementations ---
# ----------------------------


@pytest.fixture
def gu_legs_matrices():
    the_measure = "legs"
    legs_matrices = GuTransMatrix(N=8, measure=the_measure)
    return legs_matrices.A_matrix, legs_matrices.B_matrix


@pytest.fixture
def gu_legt_matrices():
    the_measure = "legt"
    legt_matrices = GuTransMatrix(N=8, measure=the_measure, lambda_n=1.0)
    return legt_matrices.A_matrix, legt_matrices.B_matrix


@pytest.fixture
def gu_legt_lmu_matrices():
    the_measure = "legt"
    lmu_matrices = GuTransMatrix(
        N=16, measure=the_measure, lambda_n=2.0
    )  # change lambda so resulting matrix is in the form of LMU
    return lmu_matrices.A_matrix, lmu_matrices.B_matrix


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
def gu_fru_matrices():
    the_measure = "fourier"
    fourier_type = "fru"
    fru_matrices = GuTransMatrix(N=8, measure=the_measure, fourier_type=fourier_type)
    return fru_matrices.A_matrix, fru_matrices.B_matrix


@pytest.fixture
def gu_fout_matrices():
    the_measure = "fourier"
    fourier_type = "fout"
    fout_matrices = GuTransMatrix(N=8, measure=the_measure, fourier_type=fourier_type)
    return fout_matrices.A_matrix, fout_matrices.B_matrix


@pytest.fixture
def gu_fourd_matrices():
    the_measure = "fourier"
    fourier_type = "fourd"
    fourd_matrices = GuTransMatrix(N=8, measure=the_measure, fourier_type=fourier_type)
    return fourd_matrices.A_matrix, fourd_matrices.B_matrix
