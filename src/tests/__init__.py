# Gu's implementation of HiPPO Operators
# implementation of HiPPO Operators
from src.tests.hippo_tests.hippo_operator import (
    hr_hippo_lsi_legs_be,
    hr_hippo_lsi_legs_bi,
    hr_hippo_lsi_legs_fe,
    hr_hippo_lsi_legs_zoh,
    hr_hippo_lti_foud_be,
    hr_hippo_lti_foud_bi,
    hr_hippo_lti_foud_fe,
    hr_hippo_lti_foud_zoh,
    hr_hippo_lti_fout_be,
    hr_hippo_lti_fout_bi,
    hr_hippo_lti_fout_fe,
    hr_hippo_lti_fout_zoh,
    hr_hippo_lti_fru_be,
    hr_hippo_lti_fru_bi,
    hr_hippo_lti_fru_fe,
    hr_hippo_lti_fru_zoh,
    hr_hippo_lti_lagt_be,
    hr_hippo_lti_lagt_bi,
    hr_hippo_lti_lagt_fe,
    hr_hippo_lti_lagt_zoh,
    hr_hippo_lti_legs_be,
    hr_hippo_lti_legs_bi,
    hr_hippo_lti_legs_fe,
    hr_hippo_lti_legs_zoh,
    hr_hippo_lti_legt_be,
    hr_hippo_lti_legt_bi,
    hr_hippo_lti_legt_fe,
    hr_hippo_lti_legt_zoh,
    hr_hippo_lti_lmu_be,
    hr_hippo_lti_lmu_bi,
    hr_hippo_lti_lmu_fe,
    hr_hippo_lti_lmu_zoh,
    hippo_lsi_legs_be,
    hippo_lsi_legs_bi,
    hippo_lsi_legs_fe,
    hippo_lsi_legs_zoh,
    hippo_lti_foud_be,
    hippo_lti_foud_bi,
    hippo_lti_foud_fe,
    hippo_lti_foud_zoh,
    hippo_lti_fout_be,
    hippo_lti_fout_bi,
    hippo_lti_fout_fe,
    hippo_lti_fout_zoh,
    hippo_lti_fru_be,
    hippo_lti_fru_bi,
    hippo_lti_fru_fe,
    hippo_lti_fru_zoh,
    hippo_lti_lagt_be,
    hippo_lti_lagt_bi,
    hippo_lti_lagt_fe,
    hippo_lti_lagt_zoh,
    hippo_lti_legs_be,
    hippo_lti_legs_bi,
    hippo_lti_legs_fe,
    hippo_lti_legs_zoh,
    hippo_lti_legt_be,
    hippo_lti_legt_bi,
    hippo_lti_legt_fe,
    hippo_lti_legt_zoh,
    hippo_lti_lmu_be,
    hippo_lti_lmu_bi,
    hippo_lti_lmu_fe,
    hippo_lti_lmu_zoh,
)

# Psuedo-random number generator
# inputs for the HiPPO operators
# Size of the transition, NPLR and DPLR matrices
from src.tests.hippo_tests.hippo_utils import (
    N2,
    N16,
    N,
    big_N,
    foud_key,
    fout_key,
    fru_key,
    key_generator,
    lagt_key,
    legs_key,
    legt_key,
    lmu_key,
    random_1_input,
    random_16_input,
    random_32_input,
    random_64_input,
)

# DPLR matrices
# Gu's implementation of DPLR matrices
# NPLR matrices
# Gu's implementation of NPLR matrices
# HiPPO matrices
# Gu's implementation of HiPPO matrices
from src.tests.hippo_tests.trans_matrices import (  # transition matrices A and B from respective polynomials made by Albert Gu; transition matrices A and B from respective polynomials; transition nplr matrices from respective polynomials made by Albert Gu; transition nplr matrices from respective polynomials; transition dplr matrices from respective polynomials made by Albert Gu; transition dplr matrices from respective polynomials
    dplr_foud,
    dplr_fout,
    dplr_fru,
    dplr_lagt,
    dplr_legs,
    dplr_legt,
    dplr_lmu,
    foud_matrices,
    fout_matrices,
    fru_matrices,
    hr_dplr_foud,
    hr_dplr_fout,
    hr_dplr_fru,
    hr_dplr_lagt,
    hr_dplr_legs,
    hr_dplr_legt,
    hr_dplr_lmu,
    hr_foud_matrices,
    hr_fout_matrices,
    hr_fru_matrices,
    hr_lagt_matrices,
    hr_legs_matrices,
    hr_legt_lmu_matrices,
    hr_legt_matrices,
    hr_nplr_foud,
    hr_nplr_fout,
    hr_nplr_fru,
    hr_nplr_lagt,
    hr_nplr_legs,
    hr_nplr_legt,
    hr_nplr_lmu,
    lagt_matrices,
    legs_matrices,
    legt_lmu_matrices,
    legt_matrices,
    nplr_foud,
    nplr_fout,
    nplr_fru,
    nplr_lagt,
    nplr_legs,
    nplr_legt,
    nplr_lmu,
)

# Deep Bidirectional RNNs
# Deep HiPPO RNNs
# Single Cell HiPPO RNNs
# Deep Bidirectional RNNs
# Deep RNNs
# Single Cell Bidirectional RNNs
# Single Cell RNNs
from src.tests.rnn_tests.architecture_fixtures import (
    deep_bigru,
    deep_bilstm,
    deep_birnn,
    deep_hippo_foud_bigru,
    deep_hippo_foud_bilstm,
    deep_hippo_fout_bigru,
    deep_hippo_fout_bilstm,
    deep_hippo_fru_bigru,
    deep_hippo_fru_bilstm,
    deep_hippo_lagt_bigru,
    deep_hippo_lagt_bilstm,
    deep_hippo_legs_bigru,
    deep_hippo_legs_bilstm,
    deep_hippo_legt_bigru,
    deep_hippo_legt_bilstm,
    deep_hippo_lmu_bigru,
    deep_hippo_lmu_bilstm,
    many_to_many_deep_gru,
    many_to_many_deep_hippo_foud_gru,
    many_to_many_deep_hippo_foud_lstm,
    many_to_many_deep_hippo_fout_gru,
    many_to_many_deep_hippo_fout_lstm,
    many_to_many_deep_hippo_fru_gru,
    many_to_many_deep_hippo_fru_lstm,
    many_to_many_deep_hippo_lagt_gru,
    many_to_many_deep_hippo_lagt_lstm,
    many_to_many_deep_hippo_legs_gru,
    many_to_many_deep_hippo_legs_lstm,
    many_to_many_deep_hippo_legt_gru,
    many_to_many_deep_hippo_legt_lstm,
    many_to_many_deep_hippo_lmu_gru,
    many_to_many_deep_hippo_lmu_lstm,
    many_to_many_deep_lstm,
    many_to_many_deep_rnn,
    many_to_many_single_cell_gru,
    many_to_many_single_cell_lstm,
    many_to_many_single_cell_rnn,
    many_to_many_single_hippo_foud_gru,
    many_to_many_single_hippo_foud_lstm,
    many_to_many_single_hippo_fout_gru,
    many_to_many_single_hippo_fout_lstm,
    many_to_many_single_hippo_fru_gru,
    many_to_many_single_hippo_fru_lstm,
    many_to_many_single_hippo_lagt_gru,
    many_to_many_single_hippo_lagt_lstm,
    many_to_many_single_hippo_legs_gru,
    many_to_many_single_hippo_legs_lstm,
    many_to_many_single_hippo_legt_gru,
    many_to_many_single_hippo_legt_lstm,
    many_to_many_single_hippo_lmu_gru,
    many_to_many_single_hippo_lmu_lstm,
    many_to_one_deep_gru,
    many_to_one_deep_hippo_foud_gru,
    many_to_one_deep_hippo_foud_lstm,
    many_to_one_deep_hippo_fout_gru,
    many_to_one_deep_hippo_fout_lstm,
    many_to_one_deep_hippo_fru_gru,
    many_to_one_deep_hippo_fru_lstm,
    many_to_one_deep_hippo_lagt_gru,
    many_to_one_deep_hippo_lagt_lstm,
    many_to_one_deep_hippo_legs_gru,
    many_to_one_deep_hippo_legs_lstm,
    many_to_one_deep_hippo_legt_gru,
    many_to_one_deep_hippo_legt_lstm,
    many_to_one_deep_hippo_lmu_gru,
    many_to_one_deep_hippo_lmu_lstm,
    many_to_one_deep_lstm,
    many_to_one_deep_rnn,
    many_to_one_single_cell_gru,
    many_to_one_single_cell_lstm,
    many_to_one_single_cell_rnn,
    many_to_one_single_hippo_foud_gru,
    many_to_one_single_hippo_foud_lstm,
    many_to_one_single_hippo_fout_gru,
    many_to_one_single_hippo_fout_lstm,
    many_to_one_single_hippo_fru_gru,
    many_to_one_single_hippo_fru_lstm,
    many_to_one_single_hippo_lagt_gru,
    many_to_one_single_hippo_lagt_lstm,
    many_to_one_single_hippo_legs_gru,
    many_to_one_single_hippo_legs_lstm,
    many_to_one_single_hippo_legt_gru,
    many_to_one_single_hippo_legt_lstm,
    many_to_one_single_hippo_lmu_gru,
    many_to_one_single_hippo_lmu_lstm,
    one_to_many_deep_gru,
    one_to_many_deep_hippo_foud_gru,
    one_to_many_deep_hippo_foud_lstm,
    one_to_many_deep_hippo_fout_gru,
    one_to_many_deep_hippo_fout_lstm,
    one_to_many_deep_hippo_fru_gru,
    one_to_many_deep_hippo_fru_lstm,
    one_to_many_deep_hippo_lagt_gru,
    one_to_many_deep_hippo_lagt_lstm,
    one_to_many_deep_hippo_legs_gru,
    one_to_many_deep_hippo_legs_lstm,
    one_to_many_deep_hippo_legt_gru,
    one_to_many_deep_hippo_legt_lstm,
    one_to_many_deep_hippo_lmu_gru,
    one_to_many_deep_hippo_lmu_lstm,
    one_to_many_deep_lstm,
    one_to_many_deep_rnn,
    one_to_many_single_cell_gru,
    one_to_many_single_cell_lstm,
    one_to_many_single_cell_rnn,
    one_to_many_single_hippo_foud_gru,
    one_to_many_single_hippo_foud_lstm,
    one_to_many_single_hippo_fout_gru,
    one_to_many_single_hippo_fout_lstm,
    one_to_many_single_hippo_fru_gru,
    one_to_many_single_hippo_fru_lstm,
    one_to_many_single_hippo_lagt_gru,
    one_to_many_single_hippo_lagt_lstm,
    one_to_many_single_hippo_legs_gru,
    one_to_many_single_hippo_legs_lstm,
    one_to_many_single_hippo_legt_gru,
    one_to_many_single_hippo_legt_lstm,
    one_to_many_single_hippo_lmu_gru,
    one_to_many_single_hippo_lmu_lstm,
    single_cell_bigru,
    single_cell_bilstm,
    single_cell_birnn,
    single_cell_hippo_foud_bigru,
    single_cell_hippo_foud_bilstm,
    single_cell_hippo_fout_bigru,
    single_cell_hippo_fout_bilstm,
    single_cell_hippo_fru_bigru,
    single_cell_hippo_fru_bilstm,
    single_cell_hippo_lagt_bigru,
    single_cell_hippo_lagt_bilstm,
    single_cell_hippo_legs_bigru,
    single_cell_hippo_legs_bilstm,
    single_cell_hippo_legt_bigru,
    single_cell_hippo_legt_bilstm,
    single_cell_hippo_lmu_bigru,
    single_cell_hippo_lmu_bilstm,
)

# HiPPO-RNN Cells
# RNN Cells
from src.tests.rnn_tests.cell_fixtures import (
    gru_cell,
    hippo_foud_gru_cell,
    hippo_foud_lstm_cell,
    hippo_fout_gru_cell,
    hippo_fout_lstm_cell,
    hippo_fru_gru_cell,
    hippo_fru_lstm_cell,
    hippo_lagt_gru_cell,
    hippo_lagt_lstm_cell,
    hippo_legs_gru_cell,
    hippo_legs_lstm_cell,
    hippo_legt_gru_cell,
    hippo_legt_lstm_cell,
    hippo_lmu_gru_cell,
    hippo_lmu_lstm_cell,
    lstm_cell,
    rnn_cell,
)

# List Only Containing One HiPPO-GRU Cell
# List Only Containing One HiPPO-LSTM Cell
# List Only Containing One RNN Cell
# List Of HiPPO-GRU Cells
# List Of HiPPO-LSTM Cells
# List Of RNN Cells
from src.tests.rnn_tests.rnn_fixtures import (
    gru_cell_list,
    gru_cell_single,
    hippo_foud_gru_cell_list,
    hippo_foud_gru_cell_single,
    hippo_foud_lstm_cell_list,
    hippo_foud_lstm_cell_single,
    hippo_fout_gru_cell_list,
    hippo_fout_gru_cell_single,
    hippo_fout_lstm_cell_list,
    hippo_fout_lstm_cell_single,
    hippo_fru_gru_cell_list,
    hippo_fru_gru_cell_single,
    hippo_fru_lstm_cell_list,
    hippo_fru_lstm_cell_single,
    hippo_lagt_gru_cell_list,
    hippo_lagt_gru_cell_single,
    hippo_lagt_lstm_cell_list,
    hippo_lagt_lstm_cell_single,
    hippo_legs_gru_cell_list,
    hippo_legs_gru_cell_single,
    hippo_legs_lstm_cell_list,
    hippo_legs_lstm_cell_single,
    hippo_legt_gru_cell_list,
    hippo_legt_gru_cell_single,
    hippo_legt_lstm_cell_list,
    hippo_legt_lstm_cell_single,
    hippo_lmu_gru_cell_list,
    hippo_lmu_gru_cell_single,
    hippo_lmu_lstm_cell_list,
    hippo_lmu_lstm_cell_single,
    lstm_cell_list,
    lstm_cell_single,
    rnn_cell_list,
    rnn_cell_single,
)

# Inputs for the models
# Psuedo-Random Number Generator Keys for Deep Bidirectional RNNs
# Psuedo-Random Number Generator Keys for Deep HiPPO RNNs
# Psuedo-Random Number Generator Keys for Single Cell HiPPO RNNs
# Psuedo-Random Number Generator Keys for Deep Bidirectional RNNs
# Psuedo-Random Number Generator Keys for Deep RNNs
# Psuedo-Random Number Generator Keys for Single Cell Bidirectional RNNs
from src.tests.rnn_tests.rnn_utils import (
    deep_bigru_key,
    deep_bilstm_key,
    deep_birnn_key,
    deep_hippo_foud_bigru_key,
    deep_hippo_foud_bilstm_key,
    deep_hippo_fout_bigru_key,
    deep_hippo_fout_bilstm_key,
    deep_hippo_fru_bigru_key,
    deep_hippo_fru_bilstm_key,
    deep_hippo_lagt_bigru_key,
    deep_hippo_lagt_bilstm_key,
    deep_hippo_legs_bigru_key,
    deep_hippo_legs_bilstm_key,
    deep_hippo_legt_bigru_key,
    deep_hippo_legt_bilstm_key,
    deep_hippo_lmu_bigru_key,
    deep_hippo_lmu_bilstm_key,
    many_to_many_deep_gru_key,
    many_to_many_deep_hippo_foud_gru_key,
    many_to_many_deep_hippo_foud_lstm_key,
    many_to_many_deep_hippo_fout_gru_key,
    many_to_many_deep_hippo_fout_lstm_key,
    many_to_many_deep_hippo_fru_gru_key,
    many_to_many_deep_hippo_fru_lstm_key,
    many_to_many_deep_hippo_lagt_gru_key,
    many_to_many_deep_hippo_lagt_lstm_key,
    many_to_many_deep_hippo_legs_gru_key,
    many_to_many_deep_hippo_legs_lstm_key,
    many_to_many_deep_hippo_legt_gru_key,
    many_to_many_deep_hippo_legt_lstm_key,
    many_to_many_deep_hippo_lmu_gru_key,
    many_to_many_deep_hippo_lmu_lstm_key,
    many_to_many_deep_lstm_key,
    many_to_many_deep_rnn_key,
    many_to_many_single_hippo_foud_gru_key,
    many_to_many_single_hippo_foud_lstm_key,
    many_to_many_single_hippo_fout_gru_key,
    many_to_many_single_hippo_fout_lstm_key,
    many_to_many_single_hippo_fru_gru_key,
    many_to_many_single_hippo_fru_lstm_key,
    many_to_many_single_hippo_lagt_gru_key,
    many_to_many_single_hippo_lagt_lstm_key,
    many_to_many_single_hippo_legs_gru_key,
    many_to_many_single_hippo_legs_lstm_key,
    many_to_many_single_hippo_legt_gru_key,
    many_to_many_single_hippo_legt_lstm_key,
    many_to_many_single_hippo_lmu_gru_key,
    many_to_many_single_hippo_lmu_lstm_key,
    many_to_one_deep_gru_key,
    many_to_one_deep_hippo_foud_gru_key,
    many_to_one_deep_hippo_foud_lstm_key,
    many_to_one_deep_hippo_fout_gru_key,
    many_to_one_deep_hippo_fout_lstm_key,
    many_to_one_deep_hippo_fru_gru_key,
    many_to_one_deep_hippo_fru_lstm_key,
    many_to_one_deep_hippo_lagt_gru_key,
    many_to_one_deep_hippo_lagt_lstm_key,
    many_to_one_deep_hippo_legs_gru_key,
    many_to_one_deep_hippo_legs_lstm_key,
    many_to_one_deep_hippo_legt_gru_key,
    many_to_one_deep_hippo_legt_lstm_key,
    many_to_one_deep_hippo_lmu_gru_key,
    many_to_one_deep_hippo_lmu_lstm_key,
    many_to_one_deep_lstm_key,
    many_to_one_deep_rnn_key,
    many_to_one_single_hippo_foud_gru_key,
    many_to_one_single_hippo_foud_lstm_key,
    many_to_one_single_hippo_fout_gru_key,
    many_to_one_single_hippo_fout_lstm_key,
    many_to_one_single_hippo_fru_gru_key,
    many_to_one_single_hippo_fru_lstm_key,
    many_to_one_single_hippo_lagt_gru_key,
    many_to_one_single_hippo_lagt_lstm_key,
    many_to_one_single_hippo_legs_gru_key,
    many_to_one_single_hippo_legs_lstm_key,
    many_to_one_single_hippo_legt_gru_key,
    many_to_one_single_hippo_legt_lstm_key,
    many_to_one_single_hippo_lmu_gru_key,
    many_to_one_single_hippo_lmu_lstm_key,
    one_to_many_deep_gru_key,
    one_to_many_deep_hippo_foud_gru_key,
    one_to_many_deep_hippo_foud_lstm_key,
    one_to_many_deep_hippo_fout_gru_key,
    one_to_many_deep_hippo_fout_lstm_key,
    one_to_many_deep_hippo_fru_gru_key,
    one_to_many_deep_hippo_fru_lstm_key,
    one_to_many_deep_hippo_lagt_gru_key,
    one_to_many_deep_hippo_lagt_lstm_key,
    one_to_many_deep_hippo_legs_gru_key,
    one_to_many_deep_hippo_legs_lstm_key,
    one_to_many_deep_hippo_legt_gru_key,
    one_to_many_deep_hippo_legt_lstm_key,
    one_to_many_deep_hippo_lmu_gru_key,
    one_to_many_deep_hippo_lmu_lstm_key,
    one_to_many_deep_lstm_key,
    one_to_many_deep_rnn_key,
    one_to_many_single_hippo_foud_gru_key,
    one_to_many_single_hippo_foud_lstm_key,
    one_to_many_single_hippo_fout_gru_key,
    one_to_many_single_hippo_fout_lstm_key,
    one_to_many_single_hippo_fru_gru_key,
    one_to_many_single_hippo_fru_lstm_key,
    one_to_many_single_hippo_lagt_gru_key,
    one_to_many_single_hippo_lagt_lstm_key,
    one_to_many_single_hippo_legs_gru_key,
    one_to_many_single_hippo_legs_lstm_key,
    one_to_many_single_hippo_legt_gru_key,
    one_to_many_single_hippo_legt_lstm_key,
    one_to_many_single_hippo_lmu_gru_key,
    one_to_many_single_hippo_lmu_lstm_key,
    random_16_input,
    random_32_input,
    random_64_input,
    single_cell_bigru_key,
    single_cell_bilstm_key,
    single_cell_birnn_key,
    single_cell_hippo_foud_bigru_key,
    single_cell_hippo_foud_bilstm_key,
    single_cell_hippo_fout_bigru_key,
    single_cell_hippo_fout_bilstm_key,
    single_cell_hippo_fru_bigru_key,
    single_cell_hippo_fru_bilstm_key,
    single_cell_hippo_lagt_bigru_key,
    single_cell_hippo_lagt_bilstm_key,
    single_cell_hippo_legs_bigru_key,
    single_cell_hippo_legs_bilstm_key,
    single_cell_hippo_legt_bigru_key,
    single_cell_hippo_legt_bilstm_key,
    single_cell_hippo_lmu_bigru_key,
    single_cell_hippo_lmu_bilstm_key,
)