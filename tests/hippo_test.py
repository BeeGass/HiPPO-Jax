import pytest
from hippo import HiPPO
from transition import 



@pytest.fixture
def hippo_legs():
    N = 10
    L = 10
    legs_matrices = TransMatrix(N=N, 
                         measure=the_measure, 
                         lambda_n=1.0, 
                         fourier_type="fru", 
                         alpha=0, 
                         beta=1)
    A = legs_matrices.A
    B = legs_matrices.B
    the_measure = "legs"
    return HiPPO(
        N=N,
        max_length=L,
        measure=the_measure,
        step=1.0 / L,
        GBT_alpha=0.5,
        seq_L=L,
        v="nv",
        lambda_n=1.0,
        fourier_type="fru",
        alpha=0.0,
        beta=1.0,
    )


@pytest.fixture
def hippo_legt():
    N = 10
    L = 10
    the_measure = "legt"
    return HiPPO(
        N=N,
        max_length=L,
        measure=the_measure,
        step=1.0 / L,
        GBT_alpha=0.5,
        seq_L=L,
        v="nv",
        lambda_n=1.0,
        fourier_type="fru",
        alpha=0.0,
        beta=1.0,
    )


@pytest.fixture
def hippo_lmu():
    N = 10
    L = 10
    the_measure = "legt"
    lambda_n = 1.0  # TODO: change value to match LMU
    return HiPPO(
        N=N,
        max_length=L,
        measure=the_measure,
        step=1.0 / L,
        GBT_alpha=0.5,
        seq_L=L,
        v="nv",
        lambda_n=1.0,
        fourier_type="fru",
        alpha=0.0,
        beta=1.0,
    )


@pytest.fixture
def hippo_lagt():
    N = 10
    L = 10
    alpha = 0.0
    beta = 1.0
    the_measure = "lagt"
    return HiPPO(
        N=N,
        max_length=L,
        measure=the_measure,
        step=1.0 / L,
        GBT_alpha=0.5,
        seq_L=L,
        v="nv",
        lambda_n=1.0,
        fourier_type="fru",
        alpha=alpha,
        beta=beta,
    )


@pytest.fixture
def hippo_fru():
    N = 10
    L = 10
    the_measure = "fourier"
    fourier_type = "fru"
    return HiPPO(
        N=N,
        max_length=L,
        measure=the_measure,
        step=1.0 / L,
        GBT_alpha=0.5,
        seq_L=L,
        v="nv",
        lambda_n=1.0,
        fourier_type=fourier_type,
        alpha=0.0,
        beta=1.0,
    )


@pytest.fixture
def hippo_fout():
    N = 10
    L = 10
    the_measure = "fourier"
    fourier_type = "fout"
    return HiPPO(
        N=N,
        max_length=L,
        measure=the_measure,
        step=1.0 / L,
        GBT_alpha=0.5,
        seq_L=L,
        v="nv",
        lambda_n=1.0,
        fourier_type=fourier_type,
        alpha=0.0,
        beta=1.0,
    )


@pytest.fixture
def hippo_fourd():
    N = 10
    L = 10
    the_measure = "fourier"
    fourier_type = "fourd"
    return HiPPO(
        N=N,
        max_length=L,
        measure=the_measure,
        step=1.0 / L,
        GBT_alpha=0.5,
        seq_L=L,
        v="nv",
        lambda_n=1.0,
        fourier_type=fourier_type,
        alpha=0.0,
        beta=1.0,
    )
