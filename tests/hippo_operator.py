import pytest
from hippo import HiPPO
from transition
from trans_matrices import (
    legs_matrices, 
    legt_matrices, 
    legt_lmu_matrices, 
    lagt_matrices, 
    fru_matrices, 
    fout_matrices, 
    fourd_matrices
)
from trans_matrices import (
    gu_legs_matrices, 
    gu_legt_matrices, 
    gu_legt_lmu_matrices, 
    gu_lagt_matrices, 
    gu_fru_matrices, 
    gu_fout_matrices, 
    gu_fourd_matrices
)
from hippo_test_params import N16

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
def hippo_fourd(fourd_matrices, N16): #TODO: add L
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
    
# --- Gu's Implementations ---

@pytest.fixture
def gu_hippo_legs(legs_matrices, N16): #TODO: add L
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
def gu_hippo_legt(legt_matrices, N16): #TODO: add L
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
def gu_hippo_legt(legt_matrices, N16): #TODO: add L
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
def gu_hippo_lmu(legt_lmu_matrices, N16): #TODO: add L
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
def gu_hippo_lagt(lagt_matrices, N16): #TODO: add L
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
def gu_hippo_fru(fru_matrices, N16): #TODO: add L
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
def gu_hippo_fout(fout_matrices, N16): #TODO: add L
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
def gu_hippo_fourd(fourd_matrices, N16): #TODO: add L
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