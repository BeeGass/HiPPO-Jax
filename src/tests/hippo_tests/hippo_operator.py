import pytest
from src.models.hippo.hippo import HiPPO
from src.tests.hippo_tests.trans_matrices import (
    legs_matrices,
    legt_matrices,
    legt_lmu_matrices,
    lagt_matrices,
    fru_matrices,
    fout_matrices,
    fourd_matrices,
)
from src.tests.hippo_tests.trans_matrices import (
    gu_legs_matrices,
    gu_legt_matrices,
    gu_legt_lmu_matrices,
    gu_lagt_matrices,
    gu_fru_matrices,
    gu_fout_matrices,
    gu_fourd_matrices,
)
from src.tests.hippo_tests.hippo_utils import (
    random_input,
    ones_input,
    zeros_input,
    desc_input,
)
from src.tests.hippo_tests.hippo_utils import N, N2, N16, big_N
from src.tests.hippo_tests.hippo_utils import (
    random_input,
    ones_input,
    zeros_input,
    desc_input,
)

# --- HiPPO operators ---


@pytest.fixture
def hippo_legs(legs_matrices, random_input):
    A, B = legs_matrices
    L = random_input.shape[0]
    return HiPPO(N=16, max_length=L, step=1.0 / L, GBT_alpha=0.5, seq_L=L, A=A, B=B)


@pytest.fixture
def hippo_legt(legt_matrices, random_input):
    A, B = legt_matrices
    L = random_input.shape[0]
    return HiPPO(N=16, max_length=L, step=1.0 / L, GBT_alpha=0.5, seq_L=L, A=A, B=B)


@pytest.fixture
def hippo_legt(legt_matrices, random_input):
    A, B = legt_matrices
    L = random_input.shape[0]
    return HiPPO(N=16, max_length=L, step=1.0 / L, GBT_alpha=0.5, seq_L=L, A=A, B=B)


@pytest.fixture
def hippo_lmu(legt_lmu_matrices, random_input):
    A, B = legt_lmu_matrices
    L = random_input.shape[0]
    return HiPPO(N=16, max_length=L, step=1.0 / L, GBT_alpha=0.5, seq_L=L, A=A, B=B)


@pytest.fixture
def hippo_lagt(lagt_matrices, random_input):
    A, B = lagt_matrices
    L = random_input.shape[0]
    return HiPPO(N=16, max_length=L, step=1.0 / L, GBT_alpha=0.5, seq_L=L, A=A, B=B)


@pytest.fixture
def hippo_fru(fru_matrices, random_input):
    A, B = fru_matrices
    L = random_input.shape[0]
    return HiPPO(N=16, max_length=L, step=1.0 / L, GBT_alpha=0.5, seq_L=L, A=A, B=B)


@pytest.fixture
def hippo_fout(fout_matrices, random_input):
    A, B = fout_matrices
    L = random_input.shape[0]
    return HiPPO(N=16, max_length=L, step=1.0 / L, GBT_alpha=0.5, seq_L=L, A=A, B=B)


@pytest.fixture
def hippo_fourd(fourd_matrices, random_input):
    A, B = fourd_matrices
    L = random_input.shape[0]
    return HiPPO(N=16, max_length=L, step=1.0 / L, GBT_alpha=0.5, seq_L=L, A=A, B=B)


# --- Gu's Implementations ---


@pytest.fixture
def gu_hippo_legs(gu_legs_matrices, random_input):
    A, B = gu_legs_matrices
    L = random_input.shape[0]
    return HiPPO(N=16, max_length=L, step=1.0 / L, GBT_alpha=0.5, seq_L=L, A=A, B=B)


@pytest.fixture
def gu_hippo_legt(gu_legt_matrices, random_input):
    A, B = gu_legt_matrices
    L = random_input.shape[0]
    return HiPPO(N=16, max_length=L, step=1.0 / L, GBT_alpha=0.5, seq_L=L, A=A, B=B)


@pytest.fixture
def gu_hippo_legt(gu_legt_matrices, random_input):
    A, B = gu_legt_matrices
    L = random_input.shape[0]
    return HiPPO(N=16, max_length=L, step=1.0 / L, GBT_alpha=0.5, seq_L=L, A=A, B=B)


@pytest.fixture
def gu_hippo_lmu(gu_legt_lmu_matrices, random_input):
    A, B = gu_legt_lmu_matrices
    L = random_input.shape[0]
    return HiPPO(N=16, max_length=L, step=1.0 / L, GBT_alpha=0.5, seq_L=L, A=A, B=B)


@pytest.fixture
def gu_hippo_lagt(gu_lagt_matrices, random_input):
    A, B = gu_lagt_matrices
    L = random_input.shape[0]
    return HiPPO(N=16, max_length=L, step=1.0 / L, GBT_alpha=0.5, seq_L=L, A=A, B=B)


@pytest.fixture
def gu_hippo_fru(gu_fru_matrices, random_input):
    A, B = gu_fru_matrices
    L = random_input.shape[0]
    return HiPPO(N=16, max_length=L, step=1.0 / L, GBT_alpha=0.5, seq_L=L, A=A, B=B)


@pytest.fixture
def gu_hippo_fout(gu_fout_matrices, random_input):
    A, B = gu_fout_matrices
    L = random_input.shape[0]
    return HiPPO(N=16, max_length=L, step=1.0 / L, GBT_alpha=0.5, seq_L=L, A=A, B=B)


@pytest.fixture
def gu_hippo_fourd(gu_fourd_matrices, random_input):
    A, B = gu_fourd_matrices
    L = random_input.shape[0]
    return HiPPO(N=16, max_length=L, step=1.0 / L, GBT_alpha=0.5, seq_L=L, A=A, B=B)
