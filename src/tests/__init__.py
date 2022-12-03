# Gu's implementation of HiPPO Operators
from src.tests.hippo_tests.hippo_operator import (
    gu_hippo_legs,
    gu_hippo_legt,
    gu_hippo_lmu,
    gu_hippo_lagt,
    gu_hippo_fru,
    gu_hippo_fout,
    gu_hippo_foud,
)

# HiPPO Operators
from src.tests.hippo_tests.hippo_operator import (
    hippo_legs,
    hippo_legt,
    hippo_lmu,
    hippo_lagt,
    hippo_fru,
    hippo_fout,
    hippo_foud,
)

# Gu's implementation of HiPPO matrices
from src.tests.hippo_tests.trans_matrices import (  # transition matrices A and B from respective polynomials made by Albert Gu
    gu_legs_matrices,
    gu_legt_matrices,
    gu_legt_lmu_matrices,
    gu_lagt_matrices,
    gu_fru_matrices,
    gu_fout_matrices,
    gu_foud_matrices,
)

# HiPPO matrices
from src.tests.hippo_tests.trans_matrices import (  # transition matrices A and B from respective polynomials
    legs_matrices,
    legt_matrices,
    legt_lmu_matrices,
    lagt_matrices,
    fru_matrices,
    fout_matrices,
    foud_matrices,
)

# Gu's implementation of NPLR matrices
from src.tests.hippo_tests.trans_matrices import (  # transition nplr matrices from respective polynomials made by Albert Gu
    gu_nplr_legs,
    gu_nplr_legt,
    gu_nplr_lmu,
    gu_nplr_lagt,
    gu_nplr_fru,
    gu_nplr_fout,
    gu_nplr_foud,
)

# NPLR matrices
from src.tests.hippo_tests.trans_matrices import (  # transition nplr matrices from respective polynomials
    nplr_legs,
    nplr_legt,
    nplr_lmu,
    nplr_lagt,
    nplr_fru,
    nplr_fout,
    nplr_foud,
)

# Gu's implementation of DPLR matrices
from src.tests.hippo_tests.trans_matrices import (  # transition dplr matrices from respective polynomials made by Albert Gu
    gu_dplr_legs,
    gu_dplr_legt,
    gu_dplr_lmu,
    gu_dplr_lagt,
    gu_dplr_fru,
    gu_dplr_fout,
    gu_dplr_foud,
)

# DPLR matrices
from src.tests.hippo_tests.trans_matrices import (  # transition dplr matrices from respective polynomials
    dplr_legs,
    dplr_legt,
    dplr_lmu,
    dplr_lagt,
    dplr_fru,
    dplr_fout,
    dplr_foud,
)

# Size of the transition, NPLR and DPLR matrices
from src.tests.hippo_tests.hippo_utils import N, N2, N16, big_N

# inputs for the HiPPO operators
from src.tests.hippo_tests.hippo_utils import (
    random_input,
    ones_input,
    zeros_input,
    desc_input,
)

# Psuedo-random number generator
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

# Single Cell RNNs
from src.tests.rnn_tests.architecture_fixtures import (
    one_to_many_single_cell_rnn,
    one_to_many_single_cell_lstm,
    one_to_many_single_cell_gru,
)
from src.tests.rnn_tests.architecture_fixtures import (
    many_to_one_single_cell_rnn,
    many_to_one_single_cell_lstm,
    many_to_one_single_cell_gru,
)
from src.tests.rnn_tests.architecture_fixtures import (
    many_to_many_single_cell_rnn,
    many_to_many_single_cell_lstm,
    many_to_many_single_cell_gru,
)

# Single Cell Bidirectional RNNs
from src.tests.rnn_tests.architecture_fixtures import (
    single_cell_birnn,
    single_cell_bilstm,
    single_cell_bigru,
)

# Deep RNNs
from src.tests.rnn_tests.architecture_fixtures import (
    one_to_many_deep_rnn,
    one_to_many_deep_lstm,
    one_to_many_deep_gru,
)
from src.tests.rnn_tests.architecture_fixtures import (
    many_to_one_deep_rnn,
    many_to_one_deep_lstm,
    many_to_one_deep_gru,
)
from src.tests.rnn_tests.architecture_fixtures import (
    many_to_many_deep_rnn,
    many_to_many_deep_lstm,
    many_to_many_deep_gru,
)

# Deep Bidirectional RNNs
from src.tests.rnn_tests.architecture_fixtures import (
    deep_birnn,
    deep_bilstm,
    deep_bigru,
)

# Single Cell HiPPO RNNs
from src.tests.rnn_tests.architecture_fixtures import (
    one_to_many_single_hippo_legs_lstm,
    one_to_many_single_hippo_legt_lstm,
    one_to_many_single_hippo_lmu_lstm,
    one_to_many_single_hippo_lagt_lstm,
    one_to_many_single_hippo_fru_lstm,
    one_to_many_single_hippo_fout_lstm,
    one_to_many_single_hippo_foud_lstm,
)
from src.tests.rnn_tests.architecture_fixtures import (
    many_to_one_single_hippo_legs_lstm,
    many_to_one_single_hippo_legt_lstm,
    many_to_one_single_hippo_lmu_lstm,
    many_to_one_single_hippo_lagt_lstm,
    many_to_one_single_hippo_fru_lstm,
    many_to_one_single_hippo_fout_lstm,
    many_to_one_single_hippo_foud_lstm,
)
from src.tests.rnn_tests.architecture_fixtures import (
    many_to_many_single_hippo_legs_lstm,
    many_to_many_single_hippo_legt_lstm,
    many_to_many_single_hippo_lmu_lstm,
    many_to_many_single_hippo_lagt_lstm,
    many_to_many_single_hippo_fru_lstm,
    many_to_many_single_hippo_fout_lstm,
    many_to_many_single_hippo_foud_lstm,
)
from src.tests.rnn_tests.architecture_fixtures import (
    one_to_many_single_hippo_legs_gru,
    one_to_many_single_hippo_legt_gru,
    one_to_many_single_hippo_lmu_gru,
    one_to_many_single_hippo_lagt_gru,
    one_to_many_single_hippo_fru_gru,
    one_to_many_single_hippo_fout_gru,
    one_to_many_single_hippo_foud_gru,
)
from src.tests.rnn_tests.architecture_fixtures import (
    many_to_one_single_hippo_legs_gru,
    many_to_one_single_hippo_legt_gru,
    many_to_one_single_hippo_lmu_gru,
    many_to_one_single_hippo_lagt_gru,
    many_to_one_single_hippo_fru_gru,
    many_to_one_single_hippo_fout_gru,
    many_to_one_single_hippo_foud_gru,
)
from src.tests.rnn_tests.architecture_fixtures import (
    many_to_many_single_hippo_legs_gru,
    many_to_many_single_hippo_legt_gru,
    many_to_many_single_hippo_lmu_gru,
    many_to_many_single_hippo_lagt_gru,
    many_to_many_single_hippo_fru_gru,
    many_to_many_single_hippo_fout_gru,
    many_to_many_single_hippo_foud_gru,
)
from src.tests.rnn_tests.architecture_fixtures import (
    single_cell_hippo_legs_bilstm,
    single_cell_hippo_legt_bilstm,
    single_cell_hippo_lmu_bilstm,
    single_cell_hippo_lagt_bilstm,
    single_cell_hippo_fru_bilstm,
    single_cell_hippo_fout_bilstm,
    single_cell_hippo_foud_bilstm,
)
from src.tests.rnn_tests.architecture_fixtures import (
    single_cell_hippo_legs_bigru,
    single_cell_hippo_legt_bigru,
    single_cell_hippo_lmu_bigru,
    single_cell_hippo_lagt_bigru,
    single_cell_hippo_fru_bigru,
    single_cell_hippo_fout_bigru,
    single_cell_hippo_foud_bigru,
)

# Deep HiPPO RNNs
from src.tests.rnn_tests.architecture_fixtures import (
    one_to_many_deep_hippo_legs_lstm,
    one_to_many_deep_hippo_legt_lstm,
    one_to_many_deep_hippo_lmu_lstm,
    one_to_many_deep_hippo_lagt_lstm,
    one_to_many_deep_hippo_fru_lstm,
    one_to_many_deep_hippo_fout_lstm,
    one_to_many_deep_hippo_foud_lstm,
)
from src.tests.rnn_tests.architecture_fixtures import (
    many_to_one_deep_hippo_legs_lstm,
    many_to_one_deep_hippo_legt_lstm,
    many_to_one_deep_hippo_lmu_lstm,
    many_to_one_deep_hippo_lagt_lstm,
    many_to_one_deep_hippo_fru_lstm,
    many_to_one_deep_hippo_fout_lstm,
    many_to_one_deep_hippo_foud_lstm,
)
from src.tests.rnn_tests.architecture_fixtures import (
    many_to_many_deep_hippo_legs_lstm,
    many_to_many_deep_hippo_legt_lstm,
    many_to_many_deep_hippo_lmu_lstm,
    many_to_many_deep_hippo_lagt_lstm,
    many_to_many_deep_hippo_fru_lstm,
    many_to_many_deep_hippo_fout_lstm,
    many_to_many_deep_hippo_foud_lstm,
)
from src.tests.rnn_tests.architecture_fixtures import (
    one_to_many_deep_hippo_legs_gru,
    one_to_many_deep_hippo_legt_gru,
    one_to_many_deep_hippo_lmu_gru,
    one_to_many_deep_hippo_lagt_gru,
    one_to_many_deep_hippo_fru_gru,
    one_to_many_deep_hippo_fout_gru,
    one_to_many_deep_hippo_foud_gru,
)
from src.tests.rnn_tests.architecture_fixtures import (
    many_to_one_deep_hippo_legs_gru,
    many_to_one_deep_hippo_legt_gru,
    many_to_one_deep_hippo_lmu_gru,
    many_to_one_deep_hippo_lagt_gru,
    many_to_one_deep_hippo_fru_gru,
    many_to_one_deep_hippo_fout_gru,
    many_to_one_deep_hippo_foud_gru,
)
from src.tests.rnn_tests.architecture_fixtures import (
    many_to_many_deep_hippo_legs_gru,
    many_to_many_deep_hippo_legt_gru,
    many_to_many_deep_hippo_lmu_gru,
    many_to_many_deep_hippo_lagt_gru,
    many_to_many_deep_hippo_fru_gru,
    many_to_many_deep_hippo_fout_gru,
    many_to_many_deep_hippo_foud_gru,
)

# Deep Bidirectional RNNs
from src.tests.rnn_tests.architecture_fixtures import (
    deep_hippo_legs_bilstm,
    deep_hippo_legt_bilstm,
    deep_hippo_lmu_bilstm,
    deep_hippo_lagt_bilstm,
    deep_hippo_fru_bilstm,
    deep_hippo_fout_bilstm,
    deep_hippo_foud_bilstm,
)
from src.tests.rnn_tests.architecture_fixtures import (
    deep_hippo_legs_bigru,
    deep_hippo_legt_bigru,
    deep_hippo_lmu_bigru,
    deep_hippo_lagt_bigru,
    deep_hippo_fru_bigru,
    deep_hippo_fout_bigru,
    deep_hippo_foud_bigru,
)

# RNN Cells
from src.tests.rnn_tests.cell_fixtures import gru_cell, lstm_cell, rnn_cell

# HiPPO-RNN Cells
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

# List Of RNN Cells
from src.tests.rnn_tests.rnn_fixtures import (
    rnn_cell_list,
    lstm_cell_list,
    gru_cell_list,
)

# List Of HiPPO-LSTM Cells
from src.tests.rnn_tests.rnn_fixtures import (
    hippo_legs_lstm_cell_list,
    hippo_legt_lstm_cell_list,
    hippo_lmu_lstm_cell_list,
    hippo_lagt_lstm_cell_list,
    hippo_fru_lstm_cell_list,
    hippo_fout_lstm_cell_list,
    hippo_foud_lstm_cell_list,
)

# List Of HiPPO-GRU Cells
from src.tests.rnn_tests.rnn_fixtures import (
    hippo_legs_gru_cell_list,
    hippo_legt_gru_cell_list,
    hippo_lmu_gru_cell_list,
    hippo_lagt_gru_cell_list,
    hippo_fru_gru_cell_list,
    hippo_fout_gru_cell_list,
    hippo_foud_gru_cell_list,
)

# List Only Containing One RNN Cell
from src.tests.rnn_tests.rnn_fixtures import (
    rnn_cell_single,
    lstm_cell_single,
    gru_cell_single,
)

# List Only Containing One HiPPO-LSTM Cell
from src.tests.rnn_tests.rnn_fixtures import (
    hippo_legs_lstm_cell_single,
    hippo_legt_lstm_cell_single,
    hippo_lmu_lstm_cell_single,
    hippo_lagt_lstm_cell_single,
    hippo_fru_lstm_cell_single,
    hippo_fout_lstm_cell_single,
    hippo_foud_lstm_cell_single,
)

# List Only Containing One HiPPO-GRU Cell
from src.tests.rnn_tests.rnn_fixtures import (
    hippo_legs_gru_cell_single,
    hippo_legt_gru_cell_single,
    hippo_lmu_gru_cell_single,
    hippo_lagt_gru_cell_single,
    hippo_fru_gru_cell_single,
    hippo_fout_gru_cell_single,
    hippo_foud_gru_cell_single,
)

# Psuedo-Random Number Generator Keys for Single Cell Bidirectional RNNs
from src.tests.rnn_tests.rnn_utils import (
    single_cell_birnn_key,
    single_cell_bilstm_key,
    single_cell_bigru_key,
)

# Psuedo-Random Number Generator Keys for Deep RNNs
from src.tests.rnn_tests.rnn_utils import (
    one_to_many_deep_rnn_key,
    one_to_many_deep_lstm_key,
    one_to_many_deep_gru_key,
)
from src.tests.rnn_tests.rnn_utils import (
    many_to_one_deep_rnn_key,
    many_to_one_deep_lstm_key,
    many_to_one_deep_gru_key,
)
from src.tests.rnn_tests.rnn_utils import (
    many_to_many_deep_rnn_key,
    many_to_many_deep_lstm_key,
    many_to_many_deep_gru_key,
)

# Psuedo-Random Number Generator Keys for Deep Bidirectional RNNs
from src.tests.rnn_tests.rnn_utils import (
    deep_birnn_key,
    deep_bilstm_key,
    deep_bigru_key,
)

# Psuedo-Random Number Generator Keys for Single Cell HiPPO RNNs
from src.tests.rnn_tests.rnn_utils import (
    one_to_many_single_hippo_legs_lstm_key,
    one_to_many_single_hippo_legt_lstm_key,
    one_to_many_single_hippo_lmu_lstm_key,
    one_to_many_single_hippo_lagt_lstm_key,
    one_to_many_single_hippo_fru_lstm_key,
    one_to_many_single_hippo_fout_lstm_key,
    one_to_many_single_hippo_foud_lstm_key,
)
from src.tests.rnn_tests.rnn_utils import (
    many_to_one_single_hippo_legs_lstm_key,
    many_to_one_single_hippo_legt_lstm_key,
    many_to_one_single_hippo_lmu_lstm_key,
    many_to_one_single_hippo_lagt_lstm_key,
    many_to_one_single_hippo_fru_lstm_key,
    many_to_one_single_hippo_fout_lstm_key,
    many_to_one_single_hippo_foud_lstm_key,
)
from src.tests.rnn_tests.rnn_utils import (
    many_to_many_single_hippo_legs_lstm_key,
    many_to_many_single_hippo_legt_lstm_key,
    many_to_many_single_hippo_lmu_lstm_key,
    many_to_many_single_hippo_lagt_lstm_key,
    many_to_many_single_hippo_fru_lstm_key,
    many_to_many_single_hippo_fout_lstm_key,
    many_to_many_single_hippo_foud_lstm_key,
)
from src.tests.rnn_tests.rnn_utils import (
    one_to_many_single_hippo_legs_gru_key,
    one_to_many_single_hippo_legt_gru_key,
    one_to_many_single_hippo_lmu_gru_key,
    one_to_many_single_hippo_lagt_gru_key,
    one_to_many_single_hippo_fru_gru_key,
    one_to_many_single_hippo_fout_gru_key,
    one_to_many_single_hippo_foud_gru_key,
)
from src.tests.rnn_tests.rnn_utils import (
    many_to_one_single_hippo_legs_gru_key,
    many_to_one_single_hippo_legt_gru_key,
    many_to_one_single_hippo_lmu_gru_key,
    many_to_one_single_hippo_lagt_gru_key,
    many_to_one_single_hippo_fru_gru_key,
    many_to_one_single_hippo_fout_gru_key,
    many_to_one_single_hippo_foud_gru_key,
)
from src.tests.rnn_tests.rnn_utils import (
    many_to_many_single_hippo_legs_gru_key,
    many_to_many_single_hippo_legt_gru_key,
    many_to_many_single_hippo_lmu_gru_key,
    many_to_many_single_hippo_lagt_gru_key,
    many_to_many_single_hippo_fru_gru_key,
    many_to_many_single_hippo_fout_gru_key,
    many_to_many_single_hippo_foud_gru_key,
)
from src.tests.rnn_tests.rnn_utils import (
    single_cell_hippo_legs_bilstm_key,
    single_cell_hippo_legt_bilstm_key,
    single_cell_hippo_lmu_bilstm_key,
    single_cell_hippo_lagt_bilstm_key,
    single_cell_hippo_fru_bilstm_key,
    single_cell_hippo_fout_bilstm_key,
    single_cell_hippo_foud_bilstm_key,
)
from src.tests.rnn_tests.rnn_utils import (
    single_cell_hippo_legs_bigru_key,
    single_cell_hippo_legt_bigru_key,
    single_cell_hippo_lmu_bigru_key,
    single_cell_hippo_lagt_bigru_key,
    single_cell_hippo_fru_bigru_key,
    single_cell_hippo_fout_bigru_key,
    single_cell_hippo_foud_bigru_key,
)

# Psuedo-Random Number Generator Keys for Deep HiPPO RNNs
from src.tests.rnn_tests.rnn_utils import (
    one_to_many_deep_hippo_legs_lstm_key,
    one_to_many_deep_hippo_legt_lstm_key,
    one_to_many_deep_hippo_lmu_lstm_key,
    one_to_many_deep_hippo_lagt_lstm_key,
    one_to_many_deep_hippo_fru_lstm_key,
    one_to_many_deep_hippo_fout_lstm_key,
    one_to_many_deep_hippo_foud_lstm_key,
)
from src.tests.rnn_tests.rnn_utils import (
    many_to_one_deep_hippo_legs_lstm_key,
    many_to_one_deep_hippo_legt_lstm_key,
    many_to_one_deep_hippo_lmu_lstm_key,
    many_to_one_deep_hippo_lagt_lstm_key,
    many_to_one_deep_hippo_fru_lstm_key,
    many_to_one_deep_hippo_fout_lstm_key,
    many_to_one_deep_hippo_foud_lstm_key,
)
from src.tests.rnn_tests.rnn_utils import (
    many_to_many_deep_hippo_legs_lstm_key,
    many_to_many_deep_hippo_legt_lstm_key,
    many_to_many_deep_hippo_lmu_lstm_key,
    many_to_many_deep_hippo_lagt_lstm_key,
    many_to_many_deep_hippo_fru_lstm_key,
    many_to_many_deep_hippo_fout_lstm_key,
    many_to_many_deep_hippo_foud_lstm_key,
)
from src.tests.rnn_tests.rnn_utils import (
    one_to_many_deep_hippo_legs_gru_key,
    one_to_many_deep_hippo_legt_gru_key,
    one_to_many_deep_hippo_lmu_gru_key,
    one_to_many_deep_hippo_lagt_gru_key,
    one_to_many_deep_hippo_fru_gru_key,
    one_to_many_deep_hippo_fout_gru_key,
    one_to_many_deep_hippo_foud_gru_key,
)
from src.tests.rnn_tests.rnn_utils import (
    many_to_one_deep_hippo_legs_gru_key,
    many_to_one_deep_hippo_legt_gru_key,
    many_to_one_deep_hippo_lmu_gru_key,
    many_to_one_deep_hippo_lagt_gru_key,
    many_to_one_deep_hippo_fru_gru_key,
    many_to_one_deep_hippo_fout_gru_key,
    many_to_one_deep_hippo_foud_gru_key,
)
from src.tests.rnn_tests.rnn_utils import (
    many_to_many_deep_hippo_legs_gru_key,
    many_to_many_deep_hippo_legt_gru_key,
    many_to_many_deep_hippo_lmu_gru_key,
    many_to_many_deep_hippo_lagt_gru_key,
    many_to_many_deep_hippo_fru_gru_key,
    many_to_many_deep_hippo_fout_gru_key,
    many_to_many_deep_hippo_foud_gru_key,
)

# Psuedo-Random Number Generator Keys for Deep Bidirectional RNNs
from src.tests.rnn_tests.rnn_utils import (
    deep_hippo_legs_bilstm_key,
    deep_hippo_legt_bilstm_key,
    deep_hippo_lmu_bilstm_key,
    deep_hippo_lagt_bilstm_key,
    deep_hippo_fru_bilstm_key,
    deep_hippo_fout_bilstm_key,
    deep_hippo_foud_bilstm_key,
)
from src.tests.rnn_tests.rnn_utils import (
    deep_hippo_legs_bigru_key,
    deep_hippo_legt_bigru_key,
    deep_hippo_lmu_bigru_key,
    deep_hippo_lagt_bigru_key,
    deep_hippo_fru_bigru_key,
    deep_hippo_fout_bigru_key,
    deep_hippo_foud_bigru_key,
)

# Inputs for the models
from src.tests.rnn_tests.rnn_utils import (
    random_16_input,
    random_32_input,
    random_64_input,
)
