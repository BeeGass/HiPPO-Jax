import pytest
from src.tests.hippo_tests.hippo_operator import (
    gu_hippo_legs,
    gu_hippo_legt,
    gu_hippo_lmu,
    gu_hippo_lagt,
    gu_hippo_fru,
    gu_hippo_fout,
    gu_hippo_fourd,
)
from src.tests.hippo_tests.hippo_operator import (
    hippo_legs,
    hippo_legt,
    hippo_lmu,
    hippo_lagt,
    hippo_fru,
    hippo_fout,
    hippo_fourd,
)
from src.tests.hippo_tests.trans_matrices import (  # transition matrices A and B from respective polynomials
    legs_matrices,
    legt_matrices,
    legt_lmu_matrices,
    lagt_matrices,
    fru_matrices,
    fout_matrices,
    fourd_matrices,
)
from src.tests.hippo_tests.trans_matrices import (  # transition matrices A and B from respective polynomials made by Albert Gu
    gu_legs_matrices,
    gu_legt_matrices,
    gu_legt_lmu_matrices,
    gu_lagt_matrices,
    gu_fru_matrices,
    gu_fout_matrices,
    gu_fourd_matrices,
)
from src.tests.hippo_tests.hippo_utils import N, N2, N16, big_N
from src.tests.hippo_tests.hippo_utils import (
    random_input,
    ones_input,
    zeros_input,
    desc_input,
)
from src.tests.hippo_tests.hippo_utils import key_generator
