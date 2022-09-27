import pytest
from ...src.models.hippo.gu_transition import GuTransMatrix
from ...src.models.hippo.transition import TransMatrix
from hippo_utils import N, N2, N16, big_N
from hippo_utils import random_input, ones_input, zeros_input, desc_input

@pytest.fixture
def legs_matrices(N16):
    the_measure = "legs"
    legs_matrices = TransMatrix(N=N16, measure=the_measure)
    return legs_matrices.A, legs_matrices.B

@pytest.fixture
def legt_matrices(N16):
    the_measure = "legt"
    legt_matrices = TransMatrix(N=N16, measure=the_measure, lambda_n=1.0)
    return legt_matrices.A, legt_matrices.B

@pytest.fixture
def legt_lmu_matrices(N16):
    the_measure = "legt"
    lmu_matrices = TransMatrix(N=N16, measure=the_measure, lambda_n=2.0) # change lambda so resulting matrix is in the form of LMU
    return lmu_matrices.A, lmu_matrices.B

@pytest.fixture
def lagt_matrices(N16):
    the_measure = "lagt"
    lagt_matrices = TransMatrix(N=N16, 
                         measure=the_measure
                         alpha=0.0, # change resulting tilt through alpha and beta
                         beta=1.0) # change resulting tilt through alpha and beta
    return lagt_matrices.A, lagt_matrices.B

@pytest.fixture
def fru_matrices(N16):
    the_measure = "fourier"
    fourier_type = "fru"
    fru_matrices = TransMatrix(N=N16, measure=the_measure, fourier_type=fourier_type)
    return fru_matrices.A, fru_matrices.B

@pytest.fixture
def fout_matrices(N16):
    the_measure = "fourier"
    fourier_type = "fout"
    fout_matrices = TransMatrix(N=N16, measure=the_measure, fourier_type=fourier_type)
    return fout_matrices.A, fout_matrices.B

@pytest.fixture
def fourd_matrices(N16):
    the_measure = "fourier"
    fourier_type = "fourd"
    fourd_matrices = TransMatrix(N=N16, measure=the_measure, fourier_type=fourier_type)
    return fourd_matrices.A, fourd_matrices.B

# --- Gu's Implementations ---
@pytest.fixture
def gu_legs_matrices(N16):
    the_measure = "legs"
    legs_matrices = GuTransMatrix(N=N16, measure=the_measure)
    return legs_matrices.A, legs_matrices.B

@pytest.fixture
def gu_legt_matrices(N16):
    the_measure = "legt"
    legt_matrices = GuTransMatrix(N=N16, measure=the_measure, lambda_n=1.0)
    return legt_matrices.A, legt_matrices.B

@pytest.fixture
def gu_legt_lmu_matrices(N16):
    the_measure = "legt"
    lmu_matrices = GuTransMatrix(N=N16, measure=the_measure, lambda_n=2.0) # change lambda so resulting matrix is in the form of LMU
    return lmu_matrices.A, lmu_matrices.B

@pytest.fixture
def gu_lagt_matrices(N16):
    the_measure = "lagt"
    lagt_matrices = GuTransMatrix(N=N16, 
                         measure=the_measure
                         alpha=0.0, # change resulting tilt through alpha and beta
                         beta=1.0) # change resulting tilt through alpha and beta
    return lagt_matrices.A, lagt_matrices.B

@pytest.fixture
def gu_fru_matrices(N16):
    the_measure = "fourier"
    fourier_type = "fru"
    fru_matrices = GuTransMatrix(N=N16, measure=the_measure, fourier_type=fourier_type)
    return fru_matrices.A, fru_matrices.B

@pytest.fixture
def gu_fout_matrices(N16):
    the_measure = "fourier"
    fourier_type = "fout"
    fout_matrices = GuTransMatrix(N=N16, measure=the_measure, fourier_type=fourier_type)
    return fout_matrices.A, fout_matrices.B

@pytest.fixture
def gu_fourd_matrices(N16):
    the_measure = "fourier"
    fourier_type = "fourd"
    fourd_matrices = GuTransMatrix(N=N16, measure=the_measure, fourier_type=fourier_type)
    return fourd_matrices.A, fourd_matrices.B
