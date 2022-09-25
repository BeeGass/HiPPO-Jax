import pytest
from hippo import HiPPO
from transition

@pytest.fixture
def legs_matrices(N16):
    the_measure = "legs"
    legs_matrices = TransMatrix(N=N16, measure=the_measure)
    return legs_matrices.A, legs_matrices.B

@pytest.fixture
def legt_matrices(N16):
    the_measure = "legt"
    legt_matrices = TransMatrix(N=N16, 
                         measure=the_measure, 
                         lambda_n=1.0)
    return legt_matrices.A, legt_matrices.B

@pytest.fixture
def legt_lmu_matrices(N16):
    the_measure = "legt"
    lmu_matrices = TransMatrix(N=N16, measure=the_measure, lambda_n=2.0) # change lambda so resulting matrix is in the form of LMU
    return lmu_matrices.A, lmu_matrices.B

@pytest.fixture
def lagt_matrices(N16):
    the_measure = "lagt"
    lmu_matrices = TransMatrix(N=N16, 
                         measure=the_measure
                         alpha=0.0, # change resulting tilt through alpha and beta
                         beta=1.0) # change resulting tilt through alpha and beta
    return lmu_matrices.A, lmu_matrices.B

@pytest.fixture
def fru_matrices(N16):
    the_measure = "fourier"
    fourier_type = "fru"
    lmu_matrices = TransMatrix(N=N16, measure=the_measure, fourier_type=fourier_type)
    return lmu_matrices.A, lmu_matrices.B

@pytest.fixture
def fout_matrices(N16):
    the_measure = "fourier"
    fourier_type = "fout"
    lmu_matrices = TransMatrix(N=N16, measure=the_measure, fourier_type=fourier_type)
    return lmu_matrices.A, lmu_matrices.B

@pytest.fixture
def fourd_matrices(N16):
    the_measure = "fourier"
    fourier_type = "fourd"
    lmu_matrices = TransMatrix(N=N16, measure=the_measure, fourier_type=fourier_type)
    return lmu_matrices.A, lmu_matrices.B