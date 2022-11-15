import pytest
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
from src.models.rnn.cells import GRUCell, HiPPOCell, LSTMCell, RNNCell
import jax.numpy as jnp

# -----------------------
# -------- cells --------
# -----------------------


@pytest.fixture
def rnn_cell_list(rnn_cell):
    stack_number = 3
    return [rnn_cell for _ in range(stack_number)]


@pytest.fixture
def lstm_cell_list(lstm_cell):
    stack_number = 3
    return [lstm_cell for _ in range(stack_number)]


@pytest.fixture
def gru_cell_list(gru_cell):
    stack_number = 3
    return [gru_cell for _ in range(stack_number)]


# ----------------------------
# ----- HiPPO LSTM cells -----
# ----------------------------

# ----------
# -- legs --
# ----------
@pytest.fixture
def hippo_legs_lstm_cell_list(hippo_legs_lstm_cell):
    stack_number = 3
    return [hippo_legs_lstm_cell for _ in range(stack_number)]


# ----------
# -- legt --
# ----------
@pytest.fixture
def hippo_legt_lstm_cell_list(hippo_legt_lstm_cell):
    stack_number = 3
    return [hippo_legt_lstm_cell for _ in range(stack_number)]


# ----------
# -- lmu --
# ----------
@pytest.fixture
def hippo_lmu_lstm_cell_list(hippo_lmu_lstm_cell):
    stack_number = 3
    return [hippo_lmu_lstm_cell for _ in range(stack_number)]


# ----------
# -- lagt --
# ----------
@pytest.fixture
def hippo_lagt_lstm_cell_list(hippo_lagt_lstm_cell):
    stack_number = 3
    return [hippo_lagt_lstm_cell for _ in range(stack_number)]


# ----------
# --- fru --
# ----------
@pytest.fixture
def hippo_fru_lstm_cell_list(hippo_fru_lstm_cell):
    stack_number = 3
    return [hippo_fru_lstm_cell for _ in range(stack_number)]


# ------------
# --- fout ---
# ------------
@pytest.fixture
def hippo_fout_lstm_cell_list(hippo_fout_lstm_cell):
    stack_number = 3
    return [hippo_fout_lstm_cell for _ in range(stack_number)]


# ------------
# --- foud ---
# ------------
@pytest.fixture
def hippo_foud_lstm_cell_list(hippo_foud_lstm_cell):
    stack_number = 3
    return [hippo_foud_lstm_cell for _ in range(stack_number)]


# ---------------------------
# ----- HiPPO GRU cells -----
# ---------------------------

# ----------
# -- legs --
# ----------
@pytest.fixture
def hippo_legs_gru_cell_list(hippo_legs_gru_cell):
    stack_number = 3
    return [hippo_legs_gru_cell for _ in range(stack_number)]


# ----------
# -- legt --
# ----------
@pytest.fixture
def hippo_legt_gru_cell_list(hippo_legt_gru_cell):
    stack_number = 3
    return jnp.array([hippo_legt_gru_cell for _ in range(stack_number)])


# ----------
# -- lmu --
# ----------
@pytest.fixture
def hippo_lmu_gru_cell_list(hippo_lmu_gru_cell):
    stack_number = 3
    return [hippo_lmu_gru_cell for _ in range(stack_number)]


# ----------
# -- lagt --
# ----------
@pytest.fixture
def hippo_lagt_gru_cell_list(hippo_lagt_gru_cell):
    stack_number = 3
    return [hippo_lagt_gru_cell for _ in range(stack_number)]


# ----------
# --- fru --
# ----------
@pytest.fixture
def hippo_fru_gru_cell_list(hippo_fru_gru_cell):
    stack_number = 3
    return [hippo_fru_gru_cell for _ in range(stack_number)]


# ------------
# --- fout ---
# ------------
@pytest.fixture
def hippo_fout_gru_cell_list(hippo_fout_gru_cell):
    stack_number = 3
    return [hippo_fout_gru_cell for _ in range(stack_number)]


# ------------
# --- foud ---
# ------------
@pytest.fixture
def hippo_foud_gru_cell_list(hippo_foud_gru_cell):
    stack_number = 3
    return [hippo_foud_gru_cell for _ in range(stack_number)]
