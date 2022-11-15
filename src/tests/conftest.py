import pytest
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
    hippo_legs,
    hippo_legt,
    hippo_lmu,
    hippo_lagt,
    hippo_fru,
    hippo_fout,
    hippo_foud,
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
    random_input,
    ones_input,
    zeros_input,
    desc_input,
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

# RNNs
from src.tests.rnn_tests.deep_rnn_fixtures import (
    deep_rnn,
    deep_lstm,
    deep_gru,
)
from src.tests.rnn_tests.deep_rnn_fixtures import (
    deep_hippo_legs_lstm,
    deep_hippo_legt_lstm,
    deep_hippo_lmu_lstm,
    deep_hippo_lagt_lstm,
    deep_hippo_fru_lstm,
    deep_hippo_fout_lstm,
    deep_hippo_foud_lstm,
)
from src.tests.rnn_tests.deep_rnn_fixtures import (
    deep_hippo_legs_gru,
    deep_hippo_legt_gru,
    deep_hippo_lmu_gru,
    deep_hippo_lagt_gru,
    deep_hippo_fru_gru,
    deep_hippo_fout_gru,
    deep_hippo_foud_gru,
)
from src.tests.rnn_tests.cell_fixtures import gru_cell, lstm_cell, rnn_cell
from src.tests.rnn_tests.cell_fixtures import (
    hippo_legs_gru_cell,
    hippo_legs_lstm_cell,
)
from src.tests.rnn_tests.cell_fixtures import hippo_legt_gru_cell, hippo_legt_lstm_cell
from src.tests.rnn_tests.cell_fixtures import hippo_lmu_gru_cell, hippo_lmu_lstm_cell
from src.tests.rnn_tests.cell_fixtures import hippo_lagt_gru_cell, hippo_lagt_lstm_cell
from src.tests.rnn_tests.cell_fixtures import hippo_fru_gru_cell, hippo_fru_lstm_cell
from src.tests.rnn_tests.cell_fixtures import hippo_fout_gru_cell, hippo_fout_lstm_cell
from src.tests.rnn_tests.cell_fixtures import hippo_foud_gru_cell, hippo_foud_lstm_cell
from src.tests.rnn_tests.rnn_fixtures import (
    rnn_cell_list,
    lstm_cell_list,
    gru_cell_list,
)
from src.tests.rnn_tests.rnn_fixtures import (
    hippo_legs_lstm_cell_list,
    hippo_legt_lstm_cell_list,
    hippo_lmu_lstm_cell_list,
    hippo_lagt_lstm_cell_list,
    hippo_fru_lstm_cell_list,
    hippo_fout_lstm_cell_list,
    hippo_foud_lstm_cell_list,
)
from src.tests.rnn_tests.rnn_fixtures import (
    hippo_legs_gru_cell_list,
    hippo_legt_gru_cell_list,
    hippo_lmu_gru_cell_list,
    hippo_lagt_gru_cell_list,
    hippo_fru_gru_cell_list,
    hippo_fout_gru_cell_list,
    hippo_foud_gru_cell_list,
)
from src.tests.rnn_tests.rnn_utils import rnn_key, gru_key, lstm_key
from src.tests.rnn_tests.rnn_utils import (
    lstm_legt_key,
    lstm_lmu_key,
    lstm_lagt_key,
    lstm_legs_key,
    lstm_fru_key,
    lstm_fout_key,
    lstm_foud_key,
)
from src.tests.rnn_tests.rnn_utils import (
    gru_legt_key,
    gru_lmu_key,
    gru_lagt_key,
    gru_legs_key,
    gru_fru_key,
    gru_fout_key,
    gru_foud_key,
)
from src.tests.rnn_tests.rnn_utils import random_32_input, random_64_input
