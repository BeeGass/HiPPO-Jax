# List Of RNN Cells
import pytest

from src.models.rnn.rnn import (
    BiRNN,
    DeepBiRNN,
    ManyToManyDeepRNN,
    ManyToManyRNN,
    ManyToOneDeepRNN,
    ManyToOneRNN,
    OneToManyDeepRNN,
    OneToManyRNN,
)

# List Only Containing One HiPPO-GRU Cell
# List Only Containing One HiPPO-LSTM Cell
# List Only Containing One RNN Cell
# List Of HiPPO-GRU Cells
# List Of HiPPO-LSTM Cells
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

# ----------------------------------------------------------------
# --------------------- Single Cell Architectures ----------------
# ----------------------------------------------------------------

# -------------------------
# ------ one to many ------
# -------------------------


@pytest.fixture
def one_to_many_single_cell_rnn(rnn_cell_single):
    output_size = 10
    return OneToManyRNN(
        output_size=output_size,
        layer=rnn_cell_single,
        skip_connections=False,
    )


@pytest.fixture
def one_to_many_single_cell_lstm(lstm_cell_single):
    output_size = 10
    return OneToManyRNN(
        output_size=output_size,
        layer=lstm_cell_single,
        skip_connections=False,
    )


@pytest.fixture
def one_to_many_single_cell_gru(gru_cell_single):
    output_size = 10
    return OneToManyRNN(
        output_size=output_size,
        layer=gru_cell_single,
        skip_connections=False,
    )


# -------------------------
# ------ many to one ------
# -------------------------


@pytest.fixture
def many_to_one_single_cell_rnn(rnn_cell_single):
    output_size = 10
    return ManyToOneRNN(
        output_size=output_size,
        layer=rnn_cell_single,
        skip_connections=False,
    )


@pytest.fixture
def many_to_one_single_cell_lstm(lstm_cell_single):
    output_size = 10
    return ManyToOneRNN(
        output_size=output_size,
        layer=lstm_cell_single,
        skip_connections=False,
    )


@pytest.fixture
def many_to_one_single_cell_gru(gru_cell_single):
    output_size = 10
    return ManyToOneRNN(
        output_size=output_size,
        layer=gru_cell_single,
        skip_connections=False,
    )


# -------------------------
# ------ many to many -----
# -------------------------


@pytest.fixture
def many_to_many_single_cell_rnn(rnn_cell_single):
    output_size = 10
    return ManyToManyRNN(
        output_size=output_size,
        layer=rnn_cell_single,
        skip_connections=False,
    )


@pytest.fixture
def many_to_many_single_cell_lstm(lstm_cell_single):
    output_size = 10
    return ManyToManyRNN(
        output_size=output_size,
        layer=lstm_cell_single,
        skip_connections=False,
    )


@pytest.fixture
def many_to_many_single_cell_gru(gru_cell_single):
    output_size = 10
    return ManyToManyRNN(
        output_size=output_size,
        layer=gru_cell_single,
        skip_connections=False,
    )


# ----------------------------------------------------------------
# --------------------- Single Cell Bidirectional ----------------
# ----------------------------------------------------------------


@pytest.fixture
def single_cell_birnn(rnn_cell_single):
    output_size = 10
    return BiRNN(
        output_size=output_size,
        forward_layer=rnn_cell_single,
        backward_layer=rnn_cell_single,
        skip_connections=False,
    )


@pytest.fixture
def single_cell_bilstm(lstm_cell_single):
    output_size = 10
    return BiRNN(
        output_size=output_size,
        forward_layer=lstm_cell_single,
        backward_layer=lstm_cell_single,
        skip_connections=False,
    )


@pytest.fixture
def single_cell_bigru(gru_cell_single):
    output_size = 10
    return BiRNN(
        output_size=output_size,
        forward_layer=gru_cell_single,
        backward_layer=gru_cell_single,
        skip_connections=False,
    )


# ----------------------------------------------------------------
# --------------------------- Deep RNNs --------------------------
# ----------------------------------------------------------------


# -------------------
# --- one to many ---
# -------------------


@pytest.fixture
def one_to_many_deep_rnn(rnn_cell_list):
    output_size = 10
    return OneToManyDeepRNN(
        output_size=output_size,
        layers=rnn_cell_list,
        skip_connections=False,
    )


@pytest.fixture
def one_to_many_deep_lstm(lstm_cell_list):
    output_size = 10
    return OneToManyDeepRNN(
        output_size=output_size,
        layers=lstm_cell_list,
        skip_connections=False,
    )


@pytest.fixture
def one_to_many_deep_gru(gru_cell_list):
    output_size = 10
    return OneToManyDeepRNN(
        output_size=output_size,
        layers=gru_cell_list,
        skip_connections=False,
    )


# -------------------
# --- many to one ---
# -------------------


@pytest.fixture
def many_to_one_deep_rnn(rnn_cell_list):
    output_size = 10
    return ManyToOneDeepRNN(
        output_size=output_size,
        layers=rnn_cell_list,
        skip_connections=False,
    )


@pytest.fixture
def many_to_one_deep_lstm(lstm_cell_list):
    output_size = 10
    return ManyToOneDeepRNN(
        output_size=output_size,
        layers=lstm_cell_list,
        skip_connections=False,
    )


@pytest.fixture
def many_to_one_deep_gru(gru_cell_list):
    output_size = 10
    return ManyToOneDeepRNN(
        output_size=output_size,
        layers=gru_cell_list,
        skip_connections=False,
    )


# --------------------
# --- many to many ---
# --------------------


@pytest.fixture
def many_to_many_deep_rnn(rnn_cell_list):
    output_size = 10
    return ManyToManyDeepRNN(
        output_size=output_size,
        layers=rnn_cell_list,
        skip_connections=False,
    )


@pytest.fixture
def many_to_many_deep_lstm(lstm_cell_list):
    output_size = 10
    return ManyToManyDeepRNN(
        output_size=output_size,
        layers=lstm_cell_list,
        skip_connections=False,
    )


@pytest.fixture
def many_to_many_deep_gru(gru_cell_list):
    output_size = 10
    return ManyToManyDeepRNN(
        output_size=output_size,
        layers=gru_cell_list,
        skip_connections=False,
    )


# ----------------------------------------------------------------
# ------------------------- Deep Bidirectional -------------------
# ----------------------------------------------------------------


@pytest.fixture
def deep_birnn(rnn_cell_list):
    output_size = 10
    return DeepBiRNN(
        output_size=output_size,
        forward_layers=rnn_cell_list,
        backward_layers=rnn_cell_list,
        skip_connections=False,
    )


@pytest.fixture
def deep_bilstm(lstm_cell_list):
    output_size = 10
    return DeepBiRNN(
        output_size=output_size,
        forward_layers=lstm_cell_list,
        backward_layers=lstm_cell_list,
        skip_connections=False,
    )


@pytest.fixture
def deep_bigru(gru_cell_list):
    output_size = 10
    return DeepBiRNN(
        output_size=output_size,
        forward_layers=gru_cell_list,
        backward_layers=gru_cell_list,
        skip_connections=False,
    )


# ----------------------------------------------------------------
# -------------------- Single Cell HiPPO LSTM --------------------
# ----------------------------------------------------------------


# --------------------------
# ------- one to many ------
# --------------------------


# ----------
# -- legs --
# ----------
@pytest.fixture
def one_to_many_single_hippo_legs_lstm(hippo_legs_lstm_cell_single):
    output_size = 10
    return OneToManyRNN(
        output_size=output_size,
        layer=hippo_legs_lstm_cell_single,
        skip_connections=False,
    )


# ----------
# -- legt --
# ----------
@pytest.fixture
def one_to_many_single_hippo_legt_lstm(hippo_legt_lstm_cell_single):
    output_size = 10
    return OneToManyRNN(
        output_size=output_size,
        layer=hippo_legt_lstm_cell_single,
        skip_connections=False,
    )


# ----------
# -- lmu --
# ----------
@pytest.fixture
def one_to_many_single_hippo_lmu_lstm(hippo_lmu_lstm_cell_single):
    output_size = 10
    return OneToManyRNN(
        output_size=output_size,
        layer=hippo_lmu_lstm_cell_single,
        skip_connections=False,
    )


# ----------
# -- lagt --
# ----------
@pytest.fixture
def one_to_many_single_hippo_lagt_lstm(hippo_lagt_lstm_cell_single):
    output_size = 10
    return OneToManyRNN(
        output_size=output_size,
        layer=hippo_lagt_lstm_cell_single,
        skip_connections=False,
    )


# ----------
# --- fru --
# ----------
@pytest.fixture
def one_to_many_single_hippo_fru_lstm(hippo_fru_lstm_cell_single):
    output_size = 10
    return OneToManyRNN(
        output_size=output_size,
        layer=hippo_fru_lstm_cell_single,
        skip_connections=False,
    )


# ------------
# --- fout ---
# ------------
@pytest.fixture
def one_to_many_single_hippo_fout_lstm(hippo_fout_lstm_cell_single):
    output_size = 10
    return OneToManyRNN(
        output_size=output_size,
        layer=hippo_fout_lstm_cell_single,
        skip_connections=False,
    )


# ------------
# --- foud ---
# ------------
@pytest.fixture
def one_to_many_single_hippo_foud_lstm(hippo_foud_lstm_cell_single):
    output_size = 10
    return OneToManyRNN(
        output_size=output_size,
        layer=hippo_foud_lstm_cell_single,
        skip_connections=False,
    )


# --------------------------
# ------- many to one ------
# --------------------------

# ----------
# -- legs --
# ----------
@pytest.fixture
def many_to_one_single_hippo_legs_lstm(hippo_legs_lstm_cell_single):
    output_size = 10
    return ManyToOneRNN(
        output_size=output_size,
        layer=hippo_legs_lstm_cell_single,
        skip_connections=False,
    )


# ----------
# -- legt --
# ----------
@pytest.fixture
def many_to_one_single_hippo_legt_lstm(hippo_legt_lstm_cell_single):
    output_size = 10
    return ManyToOneDeepRNN(
        output_size=output_size,
        layer=hippo_legt_lstm_cell_single,
        skip_connections=False,
    )


# ----------
# -- lmu --
# ----------
@pytest.fixture
def many_to_one_single_hippo_lmu_lstm(hippo_lmu_lstm_cell_single):
    output_size = 10
    return ManyToOneRNN(
        output_size=output_size,
        layer=hippo_lmu_lstm_cell_single,
        skip_connections=False,
    )


# ----------
# -- lagt --
# ----------
@pytest.fixture
def many_to_one_single_hippo_lagt_lstm(hippo_lagt_lstm_cell_single):
    output_size = 10
    return ManyToOneRNN(
        output_size=output_size,
        layer=hippo_lagt_lstm_cell_single,
        skip_connections=False,
    )


# ----------
# --- fru --
# ----------
@pytest.fixture
def many_to_one_single_hippo_fru_lstm(hippo_fru_lstm_cell_single):
    output_size = 10
    return ManyToOneRNN(
        output_size=output_size,
        layer=hippo_fru_lstm_cell_single,
        skip_connections=False,
    )


# ------------
# --- fout ---
# ------------
@pytest.fixture
def many_to_one_single_hippo_fout_lstm(hippo_fout_lstm_cell_single):
    output_size = 10
    return ManyToOneRNN(
        output_size=output_size,
        layer=hippo_fout_lstm_cell_single,
        skip_connections=False,
    )


# ------------
# --- foud ---
# ------------
@pytest.fixture
def many_to_one_single_hippo_foud_lstm(hippo_foud_lstm_cell_single):
    output_size = 10
    return ManyToOneRNN(
        output_size=output_size,
        layer=hippo_foud_lstm_cell_single,
        skip_connections=False,
    )


# ---------------------------
# ------- many to many ------
# ---------------------------

# ----------
# -- legs --
# ----------
@pytest.fixture
def many_to_many_single_hippo_legs_lstm(hippo_legs_lstm_cell_single):
    output_size = 10
    return ManyToManyRNN(
        output_size=output_size,
        layer=hippo_legs_lstm_cell_single,
        skip_connections=False,
    )


# ----------
# -- legt --
# ----------
@pytest.fixture
def many_to_many_single_hippo_legt_lstm(hippo_legt_lstm_cell_single):
    output_size = 10
    return ManyToManyRNN(
        output_size=output_size,
        layer=hippo_legt_lstm_cell_single,
        skip_connections=False,
    )


# ----------
# -- lmu --
# ----------
@pytest.fixture
def many_to_many_single_hippo_lmu_lstm(hippo_lmu_lstm_cell_single):
    output_size = 10
    return ManyToManyRNN(
        output_size=output_size,
        layer=hippo_lmu_lstm_cell_single,
        skip_connections=False,
    )


# ----------
# -- lagt --
# ----------
@pytest.fixture
def many_to_many_single_hippo_lagt_lstm(hippo_lagt_lstm_cell_single):
    output_size = 10
    return ManyToManyRNN(
        output_size=output_size,
        layer=hippo_lagt_lstm_cell_single,
        skip_connections=False,
    )


# ----------
# --- fru --
# ----------
@pytest.fixture
def many_to_many_single_hippo_fru_lstm(hippo_fru_lstm_cell_single):
    output_size = 10
    return ManyToManyRNN(
        output_size=output_size,
        layer=hippo_fru_lstm_cell_single,
        skip_connections=False,
    )


# ------------
# --- fout ---
# ------------
@pytest.fixture
def many_to_many_single_hippo_fout_lstm(hippo_fout_lstm_cell_single):
    output_size = 10
    return ManyToManyRNN(
        output_size=output_size,
        layer=hippo_fout_lstm_cell_single,
        skip_connections=False,
    )


# ------------
# --- foud ---
# ------------
@pytest.fixture
def many_to_many_single_hippo_foud_lstm(hippo_foud_lstm_cell_single):
    output_size = 10
    return ManyToManyRNN(
        output_size=output_size,
        layer=hippo_foud_lstm_cell_single,
        skip_connections=False,
    )


# ----------------------------------------------------------------
# -------------------- Single Cell HiPPO GRU ---------------------
# ----------------------------------------------------------------

# --------------------------
# ------- one to many ------
# --------------------------

# ----------
# -- legs --
# ----------
@pytest.fixture
def one_to_many_single_hippo_legs_gru(hippo_legs_gru_cell_single):
    output_size = 10
    return OneToManyRNN(
        output_size=output_size,
        layer=hippo_legs_gru_cell_single,
        skip_connections=False,
    )


# ----------
# -- legt --
# ----------
@pytest.fixture
def one_to_many_single_hippo_legt_gru(hippo_legt_gru_cell_single):
    output_size = 10
    return OneToManyRNN(
        output_size=output_size,
        layer=hippo_legt_gru_cell_single,
        skip_connections=False,
    )


# ----------
# -- lmu --
# ----------
@pytest.fixture
def one_to_many_single_hippo_lmu_gru(hippo_lmu_gru_cell_single):
    output_size = 10
    return OneToManyRNN(
        output_size=output_size,
        layer=hippo_lmu_gru_cell_single,
        skip_connections=False,
    )


# ----------
# -- lagt --
# ----------
@pytest.fixture
def one_to_many_single_hippo_lagt_gru(hippo_lagt_gru_cell_single):
    output_size = 10
    return OneToManyRNN(
        output_size=output_size,
        layer=hippo_lagt_gru_cell_single,
        skip_connections=False,
    )


# ----------
# --- fru --
# ----------
@pytest.fixture
def one_to_many_single_hippo_fru_gru(hippo_fru_gru_cell_single):
    output_size = 28 * 28
    return OneToManyRNN(
        output_size=output_size,
        layer=hippo_fru_gru_cell_single,
        skip_connections=False,
    )


# ------------
# --- fout ---
# ------------
@pytest.fixture
def one_to_many_single_hippo_fout_gru(hippo_fout_gru_cell_single):
    output_size = 28 * 28
    return OneToManyRNN(
        output_size=output_size,
        layer=hippo_fout_gru_cell_single,
        skip_connections=False,
    )


# ------------
# --- foud ---
# ------------
@pytest.fixture
def one_to_many_single_hippo_foud_gru(hippo_foud_gru_cell_single):
    output_size = 28 * 28
    return OneToManyRNN(
        output_size=output_size,
        layer=hippo_foud_gru_cell_single,
        skip_connections=False,
    )


# --------------------------
# ------- many to one ------
# --------------------------

# ----------
# -- legs --
# ----------
@pytest.fixture
def many_to_one_single_hippo_legs_gru(hippo_legs_gru_cell_single):
    output_size = 10
    return ManyToOneRNN(
        output_size=output_size,
        layer=hippo_legs_gru_cell_single,
        skip_connections=False,
    )


# ----------
# -- legt --
# ----------
@pytest.fixture
def many_to_one_single_hippo_legt_gru(hippo_legt_gru_cell_single):
    output_size = 10
    return ManyToOneRNN(
        output_size=output_size,
        layer=hippo_legt_gru_cell_single,
        skip_connections=False,
    )


# ----------
# -- lmu --
# ----------
@pytest.fixture
def many_to_one_single_hippo_lmu_gru(hippo_lmu_gru_cell_single):
    output_size = 10
    return ManyToOneRNN(
        output_size=output_size,
        layer=hippo_lmu_gru_cell_single,
        skip_connections=False,
    )


# ----------
# -- lagt --
# ----------
@pytest.fixture
def many_to_one_single_hippo_lagt_gru(hippo_lagt_gru_cell_single):
    output_size = 10
    return ManyToOneRNN(
        output_size=output_size,
        layer=hippo_lagt_gru_cell_single,
        skip_connections=False,
    )


# ----------
# --- fru --
# ----------
@pytest.fixture
def many_to_one_single_hippo_fru_gru(hippo_fru_gru_cell_single):
    output_size = 28 * 28
    return ManyToOneRNN(
        output_size=output_size,
        layer=hippo_fru_gru_cell_single,
        skip_connections=False,
    )


# ------------
# --- fout ---
# ------------
@pytest.fixture
def many_to_one_single_hippo_fout_gru(hippo_fout_gru_cell_single):
    output_size = 28 * 28
    return ManyToOneRNN(
        output_size=output_size,
        layer=hippo_fout_gru_cell_single,
        skip_connections=False,
    )


# ------------
# --- foud ---
# ------------
@pytest.fixture
def many_to_one_single_hippo_foud_gru(hippo_foud_gru_cell_single):
    output_size = 28 * 28
    return ManyToOneRNN(
        output_size=output_size,
        layer=hippo_foud_gru_cell_single,
        skip_connections=False,
    )


# ---------------------------
# ------- many to many ------
# ---------------------------

# ----------
# -- legs --
# ----------
@pytest.fixture
def many_to_many_single_hippo_legs_gru(hippo_legs_gru_cell_single):
    output_size = 10
    return ManyToManyRNN(
        output_size=output_size,
        layer=hippo_legs_gru_cell_single,
        skip_connections=False,
    )


# ----------
# -- legt --
# ----------
@pytest.fixture
def many_to_many_single_hippo_legt_gru(hippo_legt_gru_cell_single):
    output_size = 10
    return ManyToManyRNN(
        output_size=output_size,
        layer=hippo_legt_gru_cell_single,
        skip_connections=False,
    )


# ----------
# -- lmu --
# ----------
@pytest.fixture
def many_to_many_single_hippo_lmu_gru(hippo_lmu_gru_cell_single):
    output_size = 10
    return ManyToManyRNN(
        output_size=output_size,
        layer=hippo_lmu_gru_cell_single,
        skip_connections=False,
    )


# ----------
# -- lagt --
# ----------
@pytest.fixture
def many_to_many_single_hippo_lagt_gru(hippo_lagt_gru_cell_single):
    output_size = 10
    return ManyToManyRNN(
        output_size=output_size,
        layer=hippo_lagt_gru_cell_single,
        skip_connections=False,
    )


# ----------
# --- fru --
# ----------
@pytest.fixture
def many_to_many_single_hippo_fru_gru(hippo_fru_gru_cell_single):
    output_size = 28 * 28
    return ManyToManyRNN(
        output_size=output_size,
        layer=hippo_fru_gru_cell_single,
        skip_connections=False,
    )


# ------------
# --- fout ---
# ------------
@pytest.fixture
def many_to_many_single_hippo_fout_gru(hippo_fout_gru_cell_single):
    output_size = 28 * 28
    return ManyToManyRNN(
        output_size=output_size,
        layer=hippo_fout_gru_cell_single,
        skip_connections=False,
    )


# ------------
# --- foud ---
# ------------
@pytest.fixture
def many_to_many_single_hippo_foud_gru(hippo_foud_gru_cell_single):
    output_size = 28 * 28
    return ManyToManyRNN(
        output_size=output_size,
        layer=hippo_foud_gru_cell_single,
        skip_connections=False,
    )


# ----------------------------------------------------------------
# ------------ Single Cell Bidirectional HiPPO LSTM --------------
# ----------------------------------------------------------------

# ----------
# -- legs --
# ----------
@pytest.fixture
def single_cell_hippo_legs_bilstm(hippo_legs_lstm_cell_single):
    output_size = 10
    return BiRNN(
        output_size=output_size,
        forward_layer=hippo_legs_lstm_cell_single,
        backward_layer=hippo_legs_lstm_cell_single,
        skip_connections=False,
    )


# ----------
# -- legt --
# ----------
@pytest.fixture
def single_cell_hippo_legt_bilstm(hippo_legt_lstm_cell_single):
    output_size = 10
    return BiRNN(
        output_size=output_size,
        forward_layer=hippo_legt_lstm_cell_single,
        backward_layer=hippo_legt_lstm_cell_single,
        skip_connections=False,
    )


# ----------
# -- lmu --
# ----------
@pytest.fixture
def single_cell_hippo_lmu_bilstm(hippo_lmu_lstm_cell_single):
    output_size = 10
    return BiRNN(
        output_size=output_size,
        forward_layer=hippo_lmu_lstm_cell_single,
        backward_layer=hippo_lmu_lstm_cell_single,
        skip_connections=False,
    )


# ----------
# -- lagt --
# ----------
@pytest.fixture
def single_cell_hippo_lagt_bilstm(hippo_lagt_lstm_cell_single):
    output_size = 10
    return BiRNN(
        output_size=output_size,
        forward_layer=hippo_lagt_lstm_cell_single,
        backward_layer=hippo_lagt_lstm_cell_single,
        skip_connections=False,
    )


# ----------
# --- fru --
# ----------
@pytest.fixture
def single_cell_hippo_fru_bilstm(hippo_fru_lstm_cell_single):
    output_size = 10
    return BiRNN(
        output_size=output_size,
        forward_layer=hippo_fru_lstm_cell_single,
        backward_layer=hippo_fru_lstm_cell_single,
        skip_connections=False,
    )


# ------------
# --- fout ---
# ------------
@pytest.fixture
def single_cell_hippo_fout_bilstm(hippo_fout_lstm_cell_single):
    output_size = 10
    return BiRNN(
        output_size=output_size,
        forward_layer=hippo_fout_lstm_cell_single,
        backward_layer=hippo_fout_lstm_cell_single,
        skip_connections=False,
    )


# ------------
# --- foud ---
# ------------
@pytest.fixture
def single_cell_hippo_foud_bilstm(hippo_foud_lstm_cell_single):
    output_size = 10
    return BiRNN(
        output_size=output_size,
        forward_layer=hippo_foud_lstm_cell_single,
        backward_layer=hippo_foud_lstm_cell_single,
        skip_connections=False,
    )


# ----------------------------------------------------------------
# ------------ Single Cell Bidirectional HiPPO GRU ---------------
# ----------------------------------------------------------------


# ----------
# -- legs --
# ----------
@pytest.fixture
def single_cell_hippo_legs_bigru(hippo_legs_gru_cell_single):
    output_size = 10
    return BiRNN(
        output_size=output_size,
        forward_layer=hippo_legs_gru_cell_single,
        backward_layer=hippo_legs_gru_cell_single,
        skip_connections=False,
    )


# ----------
# -- legt --
# ----------
@pytest.fixture
def single_cell_hippo_legt_bigru(hippo_legt_gru_cell_single):
    output_size = 10
    return BiRNN(
        output_size=output_size,
        forward_layer=hippo_legt_gru_cell_single,
        backward_layer=hippo_legt_gru_cell_single,
        skip_connections=False,
    )


# ----------
# -- lmu --
# ----------
@pytest.fixture
def single_cell_hippo_lmu_bigru(hippo_lmu_gru_cell_single):
    output_size = 10
    return BiRNN(
        output_size=output_size,
        forward_layer=hippo_lmu_gru_cell_single,
        backward_layer=hippo_lmu_gru_cell_single,
        skip_connections=False,
    )


# ----------
# -- lagt --
# ----------
@pytest.fixture
def single_cell_hippo_lagt_bigru(hippo_lagt_gru_cell_single):
    output_size = 10
    return BiRNN(
        output_size=output_size,
        forward_layer=hippo_lagt_gru_cell_single,
        backward_layer=hippo_lagt_gru_cell_single,
        skip_connections=False,
    )


# ----------
# --- fru --
# ----------
@pytest.fixture
def single_cell_hippo_fru_bigru(hippo_fru_gru_cell_single):
    output_size = 28 * 28
    return BiRNN(
        output_size=output_size,
        forward_layer=hippo_fru_gru_cell_single,
        backward_layer=hippo_fru_gru_cell_single,
        skip_connections=False,
    )


# ------------
# --- fout ---
# ------------
@pytest.fixture
def single_cell_hippo_fout_bigru(hippo_fout_gru_cell_single):
    output_size = 28 * 28
    return BiRNN(
        output_size=output_size,
        forward_layer=hippo_fout_gru_cell_single,
        backward_layer=hippo_fout_gru_cell_single,
        skip_connections=False,
    )


# ------------
# --- foud ---
# ------------
@pytest.fixture
def single_cell_hippo_foud_bigru(hippo_foud_gru_cell_single):
    output_size = 28 * 28
    return BiRNN(
        output_size=output_size,
        forward_layer=hippo_foud_gru_cell_single,
        backward_layer=hippo_foud_gru_cell_single,
        skip_connections=False,
    )


# ----------------------------------------------------------------
# ------------------------ Deep HiPPO LSTM -----------------------
# ----------------------------------------------------------------

# --------------------------
# ------- one to many ------
# --------------------------


# ----------
# -- legs --
# ----------
@pytest.fixture
def one_to_many_deep_hippo_legs_lstm(hippo_legs_lstm_cell_list):
    output_size = 10
    return OneToManyDeepRNN(
        output_size=output_size,
        layers=hippo_legs_lstm_cell_list,
        skip_connections=False,
    )


# ----------
# -- legt --
# ----------
@pytest.fixture
def one_to_many_deep_hippo_legt_lstm(hippo_legt_lstm_cell_list):
    output_size = 10
    return OneToManyDeepRNN(
        output_size=output_size,
        layers=hippo_legt_lstm_cell_list,
        skip_connections=False,
    )


# ----------
# -- lmu --
# ----------
@pytest.fixture
def one_to_many_deep_hippo_lmu_lstm(hippo_lmu_lstm_cell_list):
    output_size = 10
    return OneToManyDeepRNN(
        output_size=output_size,
        layers=hippo_lmu_lstm_cell_list,
        skip_connections=False,
    )


# ----------
# -- lagt --
# ----------
@pytest.fixture
def one_to_many_deep_hippo_lagt_lstm(hippo_lagt_lstm_cell_list):
    output_size = 10
    return OneToManyDeepRNN(
        output_size=output_size,
        layers=hippo_lagt_lstm_cell_list,
        skip_connections=False,
    )


# ----------
# --- fru --
# ----------
@pytest.fixture
def one_to_many_deep_hippo_fru_lstm(hippo_fru_lstm_cell_list):
    output_size = 10
    return OneToManyDeepRNN(
        output_size=output_size,
        layers=hippo_fru_lstm_cell_list,
        skip_connections=False,
    )


# ------------
# --- fout ---
# ------------
@pytest.fixture
def one_to_many_deep_hippo_fout_lstm(hippo_fout_lstm_cell_list):
    output_size = 10
    return OneToManyDeepRNN(
        output_size=output_size,
        layers=hippo_fout_lstm_cell_list,
        skip_connections=False,
    )


# ------------
# --- foud ---
# ------------
@pytest.fixture
def one_to_many_deep_hippo_foud_lstm(hippo_foud_lstm_cell_list):
    output_size = 10
    return OneToManyDeepRNN(
        output_size=output_size,
        layers=hippo_foud_lstm_cell_list,
        skip_connections=False,
    )


# --------------------------
# ------- many to one ------
# --------------------------

# ----------
# -- legs --
# ----------
@pytest.fixture
def many_to_one_deep_hippo_legs_lstm(hippo_legs_lstm_cell_list):
    output_size = 10
    return ManyToOneDeepRNN(
        output_size=output_size,
        layers=hippo_legs_lstm_cell_list,
        skip_connections=False,
    )


# ----------
# -- legt --
# ----------
@pytest.fixture
def many_to_one_deep_hippo_legt_lstm(hippo_legt_lstm_cell_list):
    output_size = 10
    return ManyToOneDeepRNN(
        output_size=output_size,
        layers=hippo_legt_lstm_cell_list,
        skip_connections=False,
    )


# ----------
# -- lmu --
# ----------
@pytest.fixture
def many_to_one_deep_hippo_lmu_lstm(hippo_lmu_lstm_cell_list):
    output_size = 10
    return ManyToOneDeepRNN(
        output_size=output_size,
        layers=hippo_lmu_lstm_cell_list,
        skip_connections=False,
    )


# ----------
# -- lagt --
# ----------
@pytest.fixture
def many_to_one_deep_hippo_lagt_lstm(hippo_lagt_lstm_cell_list):
    output_size = 10
    return ManyToOneDeepRNN(
        output_size=output_size,
        layers=hippo_lagt_lstm_cell_list,
        skip_connections=False,
    )


# ----------
# --- fru --
# ----------
@pytest.fixture
def many_to_one_deep_hippo_fru_lstm(hippo_fru_lstm_cell_list):
    output_size = 10
    return ManyToOneDeepRNN(
        output_size=output_size,
        layers=hippo_fru_lstm_cell_list,
        skip_connections=False,
    )


# ------------
# --- fout ---
# ------------
@pytest.fixture
def many_to_one_deep_hippo_fout_lstm(hippo_fout_lstm_cell_list):
    output_size = 10
    return ManyToOneDeepRNN(
        output_size=output_size,
        layers=hippo_fout_lstm_cell_list,
        skip_connections=False,
    )


# ------------
# --- foud ---
# ------------
@pytest.fixture
def many_to_one_deep_hippo_foud_lstm(hippo_foud_lstm_cell_list):
    output_size = 10
    return ManyToOneDeepRNN(
        output_size=output_size,
        layers=hippo_foud_lstm_cell_list,
        skip_connections=False,
    )


# ---------------------------
# ------- many to many ------
# ---------------------------

# ----------
# -- legs --
# ----------
@pytest.fixture
def many_to_many_deep_hippo_legs_lstm(hippo_legs_lstm_cell_list):
    output_size = 10
    return ManyToManyDeepRNN(
        output_size=output_size,
        layers=hippo_legs_lstm_cell_list,
        skip_connections=False,
    )


# ----------
# -- legt --
# ----------
@pytest.fixture
def many_to_many_deep_hippo_legt_lstm(hippo_legt_lstm_cell_list):
    output_size = 10
    return ManyToManyDeepRNN(
        output_size=output_size,
        layers=hippo_legt_lstm_cell_list,
        skip_connections=False,
    )


# ----------
# -- lmu --
# ----------
@pytest.fixture
def many_to_many_deep_hippo_lmu_lstm(hippo_lmu_lstm_cell_list):
    output_size = 10
    return ManyToManyDeepRNN(
        output_size=output_size,
        layers=hippo_lmu_lstm_cell_list,
        skip_connections=False,
    )


# ----------
# -- lagt --
# ----------
@pytest.fixture
def many_to_many_deep_hippo_lagt_lstm(hippo_lagt_lstm_cell_list):
    output_size = 10
    return ManyToManyDeepRNN(
        output_size=output_size,
        layers=hippo_lagt_lstm_cell_list,
        skip_connections=False,
    )


# ----------
# --- fru --
# ----------
@pytest.fixture
def many_to_many_deep_hippo_fru_lstm(hippo_fru_lstm_cell_list):
    output_size = 10
    return ManyToManyDeepRNN(
        output_size=output_size,
        layers=hippo_fru_lstm_cell_list,
        skip_connections=False,
    )


# ------------
# --- fout ---
# ------------
@pytest.fixture
def many_to_many_deep_hippo_fout_lstm(hippo_fout_lstm_cell_list):
    output_size = 10
    return ManyToManyDeepRNN(
        output_size=output_size,
        layers=hippo_fout_lstm_cell_list,
        skip_connections=False,
    )


# ------------
# --- foud ---
# ------------
@pytest.fixture
def many_to_many_deep_hippo_foud_lstm(hippo_foud_lstm_cell_list):
    output_size = 10
    return ManyToManyDeepRNN(
        output_size=output_size,
        layers=hippo_foud_lstm_cell_list,
        skip_connections=False,
    )


# ----------------------------------------------------------------
# ------------------------ Deep HiPPO GRU ------------------------
# ----------------------------------------------------------------

# --------------------------
# ------- one to many ------
# --------------------------

# ----------
# -- legs --
# ----------
@pytest.fixture
def one_to_many_deep_hippo_legs_gru(hippo_legs_gru_cell_list):
    output_size = 10
    return OneToManyDeepRNN(
        output_size=output_size,
        layers=hippo_legs_gru_cell_list,
        skip_connections=False,
    )


# ----------
# -- legt --
# ----------
@pytest.fixture
def one_to_many_deep_hippo_legt_gru(hippo_legt_gru_cell_list):
    output_size = 10
    return OneToManyDeepRNN(
        output_size=output_size,
        layers=hippo_legt_gru_cell_list,
        skip_connections=False,
    )


# ----------
# -- lmu --
# ----------
@pytest.fixture
def one_to_many_deep_hippo_lmu_gru(hippo_lmu_gru_cell_list):
    output_size = 10
    return OneToManyDeepRNN(
        output_size=output_size,
        layers=hippo_lmu_gru_cell_list,
        skip_connections=False,
    )


# ----------
# -- lagt --
# ----------
@pytest.fixture
def one_to_many_deep_hippo_lagt_gru(hippo_lagt_gru_cell_list):
    output_size = 10
    return OneToManyDeepRNN(
        output_size=output_size,
        layers=hippo_lagt_gru_cell_list,
        skip_connections=False,
    )


# ----------
# --- fru --
# ----------
@pytest.fixture
def one_to_many_deep_hippo_fru_gru(hippo_fru_gru_cell_list):
    output_size = 28 * 28
    return OneToManyDeepRNN(
        output_size=output_size,
        layers=hippo_fru_gru_cell_list,
        skip_connections=False,
    )


# ------------
# --- fout ---
# ------------
@pytest.fixture
def one_to_many_deep_hippo_fout_gru(hippo_fout_gru_cell_list):
    output_size = 28 * 28
    return OneToManyDeepRNN(
        output_size=output_size,
        layers=hippo_fout_gru_cell_list,
        skip_connections=False,
    )


# ------------
# --- foud ---
# ------------
@pytest.fixture
def one_to_many_deep_hippo_foud_gru(hippo_foud_gru_cell_list):
    output_size = 28 * 28
    return OneToManyDeepRNN(
        output_size=output_size,
        layers=hippo_foud_gru_cell_list,
        skip_connections=False,
    )


# --------------------------
# ------- many to one ------
# --------------------------

# ----------
# -- legs --
# ----------
@pytest.fixture
def many_to_one_deep_hippo_legs_gru(hippo_legs_gru_cell_list):
    output_size = 10
    return ManyToOneDeepRNN(
        output_size=output_size,
        layers=hippo_legs_gru_cell_list,
        skip_connections=False,
    )


# ----------
# -- legt --
# ----------
@pytest.fixture
def many_to_one_deep_hippo_legt_gru(hippo_legt_gru_cell_list):
    output_size = 10
    return ManyToOneDeepRNN(
        output_size=output_size,
        layers=hippo_legt_gru_cell_list,
        skip_connections=False,
    )


# ----------
# -- lmu --
# ----------
@pytest.fixture
def many_to_one_deep_hippo_lmu_gru(hippo_lmu_gru_cell_list):
    output_size = 10
    return ManyToOneDeepRNN(
        output_size=output_size,
        layers=hippo_lmu_gru_cell_list,
        skip_connections=False,
    )


# ----------
# -- lagt --
# ----------
@pytest.fixture
def many_to_one_deep_hippo_lagt_gru(hippo_lagt_gru_cell_list):
    output_size = 10
    return ManyToOneDeepRNN(
        output_size=output_size,
        layers=hippo_lagt_gru_cell_list,
        skip_connections=False,
    )


# ----------
# --- fru --
# ----------
@pytest.fixture
def many_to_one_deep_hippo_fru_gru(hippo_fru_gru_cell_list):
    output_size = 28 * 28
    return ManyToOneDeepRNN(
        output_size=output_size,
        layers=hippo_fru_gru_cell_list,
        skip_connections=False,
    )


# ------------
# --- fout ---
# ------------
@pytest.fixture
def many_to_one_deep_hippo_fout_gru(hippo_fout_gru_cell_list):
    output_size = 28 * 28
    return ManyToOneDeepRNN(
        output_size=output_size,
        layers=hippo_fout_gru_cell_list,
        skip_connections=False,
    )


# ------------
# --- foud ---
# ------------
@pytest.fixture
def many_to_one_deep_hippo_foud_gru(hippo_foud_gru_cell_list):
    output_size = 28 * 28
    return ManyToOneDeepRNN(
        output_size=output_size,
        layers=hippo_foud_gru_cell_list,
        skip_connections=False,
    )


# ---------------------------
# ------- many to many ------
# ---------------------------

# ----------
# -- legs --
# ----------
@pytest.fixture
def many_to_many_deep_hippo_legs_gru(hippo_legs_gru_cell_list):
    output_size = 10
    return ManyToManyDeepRNN(
        output_size=output_size,
        layers=hippo_legs_gru_cell_list,
        skip_connections=False,
    )


# ----------
# -- legt --
# ----------
@pytest.fixture
def many_to_many_deep_hippo_legt_gru(hippo_legt_gru_cell_list):
    output_size = 10
    return ManyToManyDeepRNN(
        output_size=output_size,
        layers=hippo_legt_gru_cell_list,
        skip_connections=False,
    )


# ----------
# -- lmu --
# ----------
@pytest.fixture
def many_to_many_deep_hippo_lmu_gru(hippo_lmu_gru_cell_list):
    output_size = 10
    return ManyToManyDeepRNN(
        output_size=output_size,
        layers=hippo_lmu_gru_cell_list,
        skip_connections=False,
    )


# ----------
# -- lagt --
# ----------
@pytest.fixture
def many_to_many_deep_hippo_lagt_gru(hippo_lagt_gru_cell_list):
    output_size = 10
    return ManyToManyDeepRNN(
        output_size=output_size,
        layers=hippo_lagt_gru_cell_list,
        skip_connections=False,
    )


# ----------
# --- fru --
# ----------
@pytest.fixture
def many_to_many_deep_hippo_fru_gru(hippo_fru_gru_cell_list):
    output_size = 28 * 28
    return ManyToManyDeepRNN(
        output_size=output_size,
        layers=hippo_fru_gru_cell_list,
        skip_connections=False,
    )


# ------------
# --- fout ---
# ------------
@pytest.fixture
def many_to_many_deep_hippo_fout_gru(hippo_fout_gru_cell_list):
    output_size = 28 * 28
    return ManyToManyDeepRNN(
        output_size=output_size,
        layers=hippo_fout_gru_cell_list,
        skip_connections=False,
    )


# ------------
# --- foud ---
# ------------
@pytest.fixture
def many_to_many_deep_hippo_foud_gru(hippo_foud_gru_cell_list):
    output_size = 28 * 28
    return ManyToManyDeepRNN(
        output_size=output_size,
        layers=hippo_foud_gru_cell_list,
        skip_connections=False,
    )


# ----------------------------------------------------------------
# ---------------- Deep Bidirectional HiPPO LSTM -----------------
# ----------------------------------------------------------------

# ----------
# -- legs --
# ----------
@pytest.fixture
def deep_hippo_legs_bilstm(hippo_legs_lstm_cell_list):
    output_size = 10
    return DeepBiRNN(
        output_size=output_size,
        forward_layers=hippo_legs_lstm_cell_list,
        backward_layers=hippo_legs_lstm_cell_list,
        skip_connections=False,
    )


# ----------
# -- legt --
# ----------
@pytest.fixture
def deep_hippo_legt_bilstm(hippo_legt_lstm_cell_list):
    output_size = 10
    return DeepBiRNN(
        output_size=output_size,
        forward_layers=hippo_legt_lstm_cell_list,
        backward_layers=hippo_legt_lstm_cell_list,
        skip_connections=False,
    )


# ----------
# -- lmu --
# ----------
@pytest.fixture
def deep_hippo_lmu_bilstm(hippo_lmu_lstm_cell_list):
    output_size = 10
    return DeepBiRNN(
        output_size=output_size,
        forward_layers=hippo_lmu_lstm_cell_list,
        backward_layers=hippo_lmu_lstm_cell_list,
        skip_connections=False,
    )


# ----------
# -- lagt --
# ----------
@pytest.fixture
def deep_hippo_lagt_bilstm(hippo_lagt_lstm_cell_list):
    output_size = 10
    return DeepBiRNN(
        output_size=output_size,
        forward_layers=hippo_lagt_lstm_cell_list,
        backward_layers=hippo_lagt_lstm_cell_list,
        skip_connections=False,
    )


# ----------
# --- fru --
# ----------
@pytest.fixture
def deep_hippo_fru_bilstm(hippo_fru_lstm_cell_list):
    output_size = 10
    return DeepBiRNN(
        output_size=output_size,
        forward_layers=hippo_fru_lstm_cell_list,
        backward_layers=hippo_fru_lstm_cell_list,
        skip_connections=False,
    )


# ------------
# --- fout ---
# ------------
@pytest.fixture
def deep_hippo_fout_bilstm(hippo_fout_lstm_cell_list):
    output_size = 10
    return DeepBiRNN(
        output_size=output_size,
        forward_layers=hippo_fout_lstm_cell_list,
        backward_layers=hippo_fout_lstm_cell_list,
        skip_connections=False,
    )


# ------------
# --- foud ---
# ------------
@pytest.fixture
def deep_hippo_foud_bilstm(hippo_foud_lstm_cell_list):
    output_size = 10
    return DeepBiRNN(
        output_size=output_size,
        forward_layers=hippo_foud_lstm_cell_list,
        backward_layers=hippo_foud_lstm_cell_list,
        skip_connections=False,
    )


# ----------------------------------------------------------------
# ----------------- Deep Bidirectional HiPPO GRU -----------------
# ----------------------------------------------------------------

# ----------
# -- legs --
# ----------
@pytest.fixture
def deep_hippo_legs_bigru(hippo_legs_gru_cell_list):
    output_size = 10
    return DeepBiRNN(
        output_size=output_size,
        forward_layers=hippo_legs_gru_cell_list,
        backward_layers=hippo_legs_gru_cell_list,
        skip_connections=False,
    )


# ----------
# -- legt --
# ----------
@pytest.fixture
def deep_hippo_legt_bigru(hippo_legt_gru_cell_list):
    output_size = 10
    return DeepBiRNN(
        output_size=output_size,
        forward_layers=hippo_legt_gru_cell_list,
        backward_layers=hippo_legt_gru_cell_list,
        skip_connections=False,
    )


# ----------
# -- lmu --
# ----------
@pytest.fixture
def deep_hippo_lmu_bigru(hippo_lmu_gru_cell_list):
    output_size = 10
    return DeepBiRNN(
        output_size=output_size,
        forward_layers=hippo_lmu_gru_cell_list,
        backward_layers=hippo_lmu_gru_cell_list,
        skip_connections=False,
    )


# ----------
# -- lagt --
# ----------
@pytest.fixture
def deep_hippo_lagt_bigru(hippo_lagt_gru_cell_list):
    output_size = 10
    return DeepBiRNN(
        output_size=output_size,
        forward_layers=hippo_lagt_gru_cell_list,
        backward_layers=hippo_lagt_gru_cell_list,
        skip_connections=False,
    )


# ----------
# --- fru --
# ----------
@pytest.fixture
def deep_hippo_fru_bigru(hippo_fru_gru_cell_list):
    output_size = 28 * 28
    return DeepBiRNN(
        output_size=output_size,
        forward_layers=hippo_fru_gru_cell_list,
        backward_layers=hippo_fru_gru_cell_list,
        skip_connections=False,
    )


# ------------
# --- fout ---
# ------------
@pytest.fixture
def deep_hippo_fout_bigru(hippo_fout_gru_cell_list):
    output_size = 28 * 28
    return DeepBiRNN(
        output_size=output_size,
        forward_layers=hippo_fout_gru_cell_list,
        backward_layers=hippo_fout_gru_cell_list,
        skip_connections=False,
    )


# ------------
# --- foud ---
# ------------
@pytest.fixture
def deep_hippo_foud_bigru(hippo_foud_gru_cell_list):
    output_size = 28 * 28
    return DeepBiRNN(
        output_size=output_size,
        forward_layers=hippo_foud_gru_cell_list,
        backward_layers=hippo_foud_gru_cell_list,
        skip_connections=False,
    )
