import pytest
from hippo import HiPPO
from transition

@pytest.fixture
def hippo_legs(legs_matrices, N16): #TODO: add L
    A, B = legs_matrices
    return HiPPO(
        N=N16,
        max_length=L,
        step=1.0 / L,
        GBT_alpha=0.5,
        seq_L=L,
        A=A,
        B=B
    )
    
@pytest.fixture
def hippo_legt(legt_matrices, N16): #TODO: add L
    A, B = legt_matrices
    return HiPPO(
        N=N16,
        max_length=L,
        step=1.0 / L,
        GBT_alpha=0.5,
        seq_L=L,
        A=A,
        B=B
    )
    
@pytest.fixture
def hippo_legt(legt_matrices, N16): #TODO: add L
    A, B = legt_matrices
    return HiPPO(
        N=N16,
        max_length=L,
        step=1.0 / L,
        GBT_alpha=0.5,
        seq_L=L,
        A=A,
        B=B
    )
    
@pytest.fixture
def hippo_lmu(legt_lmu_matrices, N16): #TODO: add L
    A, B = legt_lmu_matrices
    return HiPPO(
        N=N16,
        max_length=L,
        step=1.0 / L,
        GBT_alpha=0.5,
        seq_L=L,
        A=A,
        B=B
    )
    
@pytest.fixture
def hippo_lagt(lagt_matrices, N16): #TODO: add L
    A, B = lagt_matrices
    return HiPPO(
        N=N16,
        max_length=L,
        step=1.0 / L,
        GBT_alpha=0.5,
        seq_L=L,
        A=A,
        B=B
    )
    
@pytest.fixture
def hippo_fru(fru_matrices, N16): #TODO: add L
    A, B = fru_matrices
    return HiPPO(
        N=N16,
        max_length=L,
        step=1.0 / L,
        GBT_alpha=0.5,
        seq_L=L,
        A=A,
        B=B
    )
    
@pytest.fixture
def hippo_fout(fout_matrices, N16): #TODO: add L
    A, B = fout_matrices
    return HiPPO(
        N=N16,
        max_length=L,
        step=1.0 / L,
        GBT_alpha=0.5,
        seq_L=L,
        A=A,
        B=B
    )

@pytest.fixture
def hippo_fourd(fourd_matrices): #TODO: add L
    A, B = fourd_matrices
    return HiPPO(
        N=N16,
        max_length=L,
        step=1.0 / L,
        GBT_alpha=0.5,
        seq_L=L,
        A=A,
        B=B
    )