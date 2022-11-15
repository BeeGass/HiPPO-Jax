import pytest
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
from src.models.rnn.rnn import DeepRNN

# -----------------------
# ------ Deep RNNs ------
# -----------------------


@pytest.fixture
def deep_rnn(rnn_cell_list):
    output_size = 10
    return DeepRNN(
        output_size=output_size,
        layers=rnn_cell_list,
        skip_connections=False,
    )


@pytest.fixture
def deep_lstm(lstm_cell_list):
    output_size = 10
    return DeepRNN(
        output_size=output_size,
        layers=lstm_cell_list,
        skip_connections=False,
    )


@pytest.fixture
def deep_gru(gru_cell_list):
    output_size = 10
    return DeepRNN(
        output_size=output_size,
        layers=gru_cell_list,
        skip_connections=False,
    )


# ---------------------------
# ----- Deep HiPPO LSTM -----
# ---------------------------

# ----------
# -- legs --
# ----------
@pytest.fixture
def deep_hippo_legs_lstm(hippo_legs_lstm_cell_list):
    output_size = 10
    return DeepRNN(
        output_size=output_size,
        layers=hippo_legs_lstm_cell_list,
        skip_connections=False,
    )


# ----------
# -- legt --
# ----------
@pytest.fixture
def deep_hippo_legt_lstm(hippo_legt_lstm_cell_list):
    output_size = 10
    return DeepRNN(
        output_size=output_size,
        layers=hippo_legt_lstm_cell_list,
        skip_connections=False,
    )


# ----------
# -- lmu --
# ----------
@pytest.fixture
def deep_hippo_lmu_lstm(hippo_lmu_lstm_cell_list):
    output_size = 10
    return DeepRNN(
        output_size=output_size,
        layers=hippo_lmu_lstm_cell_list,
        skip_connections=False,
    )


# ----------
# -- lagt --
# ----------
@pytest.fixture
def deep_hippo_lagt_lstm(hippo_lagt_lstm_cell_list):
    output_size = 10
    return DeepRNN(
        output_size=output_size,
        layers=hippo_lagt_lstm_cell_list,
        skip_connections=False,
    )


# ----------
# --- fru --
# ----------
@pytest.fixture
def deep_hippo_fru_lstm(hippo_fru_lstm_cell_list):
    output_size = 10
    return DeepRNN(
        output_size=output_size,
        layers=hippo_fru_lstm_cell_list,
        skip_connections=False,
    )


# ------------
# --- fout ---
# ------------
@pytest.fixture
def deep_hippo_fout_lstm(hippo_fout_lstm_cell_list):
    output_size = 10
    return DeepRNN(
        output_size=output_size,
        layers=hippo_fout_lstm_cell_list,
        skip_connections=False,
    )


# ------------
# --- foud ---
# ------------
@pytest.fixture
def deep_hippo_foud_lstm(hippo_foud_lstm_cell_list):
    output_size = 10
    return DeepRNN(
        output_size=output_size,
        layers=hippo_foud_lstm_cell_list,
        skip_connections=False,
    )


# ---------------------------
# ----- Deep HiPPO GRU  -----
# ---------------------------

# ----------
# -- legs --
# ----------
@pytest.fixture
def deep_hippo_legs_gru(hippo_legs_gru_cell_list):
    output_size = 10
    return DeepRNN(
        output_size=output_size,
        layers=hippo_legs_gru_cell_list,
        skip_connections=False,
    )


# ----------
# -- legt --
# ----------
@pytest.fixture
def deep_hippo_legt_gru(hippo_legt_gru_cell_list):
    output_size = 10
    return DeepRNN(
        output_size=output_size,
        layers=hippo_legt_gru_cell_list,
        skip_connections=False,
    )


# ----------
# -- lmu --
# ----------
@pytest.fixture
def deep_hippo_lmu_gru(hippo_lmu_gru_cell_list):
    output_size = 10
    return DeepRNN(
        output_size=output_size,
        layers=hippo_lmu_gru_cell_list,
        skip_connections=False,
    )


# ----------
# -- lagt --
# ----------
@pytest.fixture
def deep_hippo_lagt_gru(hippo_lagt_gru_cell_list):
    output_size = 10
    return DeepRNN(
        output_size=output_size,
        layers=hippo_lagt_gru_cell_list,
        skip_connections=False,
    )


# ----------
# --- fru --
# ----------
@pytest.fixture
def deep_hippo_fru_gru(hippo_fru_gru_cell_list):
    output_size = 28 * 28
    return DeepRNN(
        output_size=output_size,
        layers=hippo_fru_gru_cell_list,
        skip_connections=False,
    )


# ------------
# --- fout ---
# ------------
@pytest.fixture
def deep_hippo_fout_gru(hippo_fout_gru_cell_list):
    output_size = 28 * 28
    return DeepRNN(
        output_size=output_size,
        layers=hippo_fout_gru_cell_list,
        skip_connections=False,
    )


# ------------
# --- foud ---
# ------------
@pytest.fixture
def deep_hippo_foud_gru(hippo_foud_gru_cell_list):
    output_size = 28 * 28
    return DeepRNN(
        output_size=output_size,
        layers=hippo_foud_gru_cell_list,
        skip_connections=False,
    )
