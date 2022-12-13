from src.tests.hippo_tests.hippo_operator import (
    gu_hippo_legs,
    gu_hippo_legt,
    gu_hippo_lmu,
    gu_hippo_lagt,
    gu_hippo_fru,
    gu_hippo_fout,
    gu_hippo_foud,
)
from src.tests.hippo_tests.hippo_operator import (
    hippo_lti_legs_fe,
    hippo_lsi_legs_fe,
    hippo_legt_fe,
    hippo_lmu_fe,
    hippo_lagt_fe,
    hippo_fru_fe,
    hippo_fout_fe,
    hippo_foud_fe,
)
from src.tests.hippo_tests.hippo_operator import (
    hippo_lti_legs_be,
    hippo_lsi_legs_be,
    hippo_legt_be,
    hippo_lmu_be,
    hippo_lagt_be,
    hippo_fru_be,
    hippo_fout_be,
    hippo_foud_be,
)
from src.tests.hippo_tests.hippo_operator import (
    hippo_lti_legs_bi,
    hippo_lsi_legs_bi,
    hippo_legt_bi,
    hippo_lmu_bi,
    hippo_lagt_bi,
    hippo_fru_bi,
    hippo_fout_bi,
    hippo_foud_bi,
)
from src.tests.hippo_tests.hippo_operator import (
    hippo_lti_legs_zoh,
    hippo_lsi_legs_zoh,
    hippo_legt_zoh,
    hippo_lmu_zoh,
    hippo_lagt_zoh,
    hippo_fru_zoh,
    hippo_fout_zoh,
    hippo_foud_zoh,
)
from src.tests.hippo_tests.hippo_operator import (
    gu_hippo_lti_legs_fe,
    gu_hippo_lsi_legs_fe,
    gu_hippo_lti_legt_fe,
    gu_hippo_lti_lmu_fe,
    gu_hippo_lti_lagt_fe,
    gu_hippo_lti_fru_fe,
    gu_hippo_lti_fout_fe,
    gu_hippo_lti_foud_fe,
)
from src.tests.hippo_tests.hippo_operator import (
    gu_hippo_lti_legs_be,
    gu_hippo_lsi_legs_be,
    gu_hippo_lti_legt_be,
    gu_hippo_lti_lmu_be,
    gu_hippo_lti_lagt_be,
    gu_hippo_lti_fru_be,
    gu_hippo_lti_fout_be,
    gu_hippo_lti_foud_be,
)
from src.tests.hippo_tests.hippo_operator import (
    gu_hippo_lti_legs_bi,
    gu_hippo_lsi_legs_bi,
    gu_hippo_lti_legt_bi,
    gu_hippo_lti_lmu_bi,
    gu_hippo_lti_lagt_bi,
    gu_hippo_lti_fru_bi,
    gu_hippo_lti_fout_bi,
    gu_hippo_lti_foud_bi,
)
from src.tests.hippo_tests.hippo_operator import (
    gu_hippo_lti_legs_zoh,
    gu_hippo_lsi_legs_zoh,
    gu_hippo_lti_legt_zoh,
    gu_hippo_lti_lmu_zoh,
    gu_hippo_lti_lagt_zoh,
    gu_hippo_lti_fru_zoh,
    gu_hippo_lti_fout_zoh,
    gu_hippo_lti_foud_zoh,
)
from src.tests.hippo_tests.trans_matrices import (  # transition matrices A and B from respective polynomials
    legs_matrices,
    legt_matrices,
    legt_lmu_matrices,
    lagt_matrices,
    fru_matrices,
    fout_matrices,
    foud_matrices,
)
from src.tests.hippo_tests.trans_matrices import (  # transition matrices A and B from respective polynomials made by Albert Gu
    gu_legs_matrices,
    gu_legt_matrices,
    gu_legt_lmu_matrices,
    gu_lagt_matrices,
    gu_fru_matrices,
    gu_fout_matrices,
    gu_foud_matrices,
)
from src.tests.hippo_tests.trans_matrices import (  # transition nplr matrices from respective polynomials
    nplr_legs,
    nplr_legt,
    nplr_lmu,
    nplr_lagt,
    nplr_fru,
    nplr_fout,
    nplr_foud,
)
from src.tests.hippo_tests.trans_matrices import (  # transition nplr matrices from respective polynomials made by Albert Gu
    gu_nplr_legs,
    gu_nplr_legt,
    gu_nplr_lmu,
    gu_nplr_lagt,
    gu_nplr_fru,
    gu_nplr_fout,
    gu_nplr_foud,
)
from src.tests.hippo_tests.trans_matrices import (  # transition dplr matrices from respective polynomials
    dplr_legs,
    dplr_legt,
    dplr_lmu,
    dplr_lagt,
    dplr_fru,
    dplr_fout,
    dplr_foud,
)
from src.tests.hippo_tests.trans_matrices import (  # transition dplr matrices from respective polynomials made by Albert Gu
    gu_dplr_legs,
    gu_dplr_legt,
    gu_dplr_lmu,
    gu_dplr_lagt,
    gu_dplr_fru,
    gu_dplr_fout,
    gu_dplr_foud,
)
from src.tests.hippo_tests.hippo_utils import N, N2, N16, big_N
from src.tests.hippo_tests.hippo_utils import (
    random_1_input,
    random_16_input,
    random_32_input,
    random_64_input,
)
from src.tests.hippo_tests.hippo_utils import (
    key_generator,
    legt_key,
    lmu_key,
    lagt_key,
    legs_key,
    fru_key,
    fout_key,
    foud_key,
)
