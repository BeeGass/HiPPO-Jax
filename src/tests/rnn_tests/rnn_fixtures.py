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
from flax.linen.activation import tanh
from flax.linen.activation import sigmoid

# -----------------------
# -------- cells --------
# -----------------------


@pytest.fixture
def rnn_cell_list():
    stack_number = 3
    input_size = 28 * 28
    hidden_size = 256
    return [
        RNNCell(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=True,
            param_dtype=jnp.float32,
            activation_fn=tanh,
        )
        for _ in range(stack_number)
    ]


@pytest.fixture
def lstm_cell_list():
    stack_number = 3
    input_size = 28 * 28
    hidden_size = 256
    return [
        LSTMCell(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=True,
            param_dtype=jnp.float32,
            gate_fn=sigmoid,
            activation_fn=tanh,
        )
        for _ in range(stack_number)
    ]


@pytest.fixture
def gru_cell_list():
    stack_number = 3
    input_size = 28 * 28
    hidden_size = 256
    return [
        GRUCell(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=True,
            param_dtype=jnp.float32,
            gate_fn=sigmoid,
            activation_fn=tanh,
        )
        for _ in range(stack_number)
    ]


# ----------------------------
# ----- HiPPO LSTM cells -----
# ----------------------------

# ----------
# -- legs --
# ----------
@pytest.fixture
def hippo_legs_lstm_cell_list():
    stack_number = 3
    input_size = 28 * 28
    hidden_size = 256
    return [
        HiPPOCell(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=True,
            param_dtype=jnp.float32,
            activation_fn=tanh,
            measure="legs",
            rnn_cell=LSTMCell,
        )
        for _ in range(stack_number)
    ]


# ----------
# -- legt --
# ----------
@pytest.fixture
def hippo_legt_lstm_cell_list():
    stack_number = 3
    input_size = 28 * 28
    hidden_size = 256
    return [
        HiPPOCell(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=True,
            param_dtype=jnp.float32,
            activation_fn=tanh,
            measure="legt",
            lambda_n=1.0,
            rnn_cell=LSTMCell,
        )
        for _ in range(stack_number)
    ]


# ----------
# -- lmu --
# ----------
@pytest.fixture
def hippo_lmu_lstm_cell_list():
    stack_number = 3
    input_size = 28 * 28
    hidden_size = 256
    return [
        HiPPOCell(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=True,
            param_dtype=jnp.float32,
            activation_fn=tanh,
            measure="legt",
            lambda_n=2.0,
            rnn_cell=LSTMCell,
        )
        for _ in range(stack_number)
    ]


# ----------
# -- lagt --
# ----------
@pytest.fixture
def hippo_lagt_lstm_cell_list():
    stack_number = 3
    input_size = 28 * 28
    hidden_size = 256
    return [
        HiPPOCell(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=True,
            param_dtype=jnp.float32,
            activation_fn=tanh,
            measure="lagt",
            alpha=0.0,
            beta=1.0,
            rnn_cell=LSTMCell,
        )
        for _ in range(stack_number)
    ]


# ----------
# --- fru --
# ----------
@pytest.fixture
def hippo_fru_lstm_cell_list():
    stack_number = 3
    input_size = 28 * 28
    hidden_size = 256
    return [
        HiPPOCell(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=True,
            param_dtype=jnp.float32,
            activation_fn=tanh,
            measure="fourier",
            fourier_type="fru",
            rnn_cell=LSTMCell,
        )
        for _ in range(stack_number)
    ]


# ------------
# --- fout ---
# ------------
@pytest.fixture
def hippo_fout_lstm_cell_list():
    stack_number = 3
    input_size = 28 * 28
    hidden_size = 256
    return [
        HiPPOCell(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=True,
            param_dtype=jnp.float32,
            activation_fn=tanh,
            measure="fourier",
            fourier_type="fout",
            rnn_cell=LSTMCell,
        )
        for _ in range(stack_number)
    ]


# ------------
# --- foud ---
# ------------
@pytest.fixture
def hippo_foud_lstm_cell_list():
    stack_number = 3
    input_size = 28 * 28
    hidden_size = 256
    return [
        HiPPOCell(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=True,
            param_dtype=jnp.float32,
            activation_fn=tanh,
            measure="fourier",
            fourier_type="foud",
            rnn_cell=LSTMCell,
        )
        for _ in range(stack_number)
    ]


# ---------------------------
# ----- HiPPO GRU cells -----
# ---------------------------

# ----------
# -- legs --
# ----------
@pytest.fixture
def hippo_legs_gru_cell_list():
    stack_number = 3
    input_size = 28 * 28
    hidden_size = 256
    return [
        HiPPOCell(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=True,
            param_dtype=jnp.float32,
            activation_fn=tanh,
            measure="legs",
            rnn_cell=GRUCell,
        )
        for _ in range(stack_number)
    ]


# ----------
# -- legt --
# ----------
@pytest.fixture
def hippo_legt_gru_cell_list():
    stack_number = 3
    input_size = 28 * 28
    hidden_size = 256
    return jnp.array(
        [
            HiPPOCell(
                input_size=input_size,
                hidden_size=hidden_size,
                bias=True,
                param_dtype=jnp.float32,
                activation_fn=tanh,
                measure="legt",
                lambda_n=1.0,
                rnn_cell=GRUCell,
            )
            for _ in range(stack_number)
        ]
    )


# ----------
# -- lmu --
# ----------
@pytest.fixture
def hippo_lmu_gru_cell_list():
    stack_number = 3
    input_size = 28 * 28
    hidden_size = 256
    return [
        HiPPOCell(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=True,
            param_dtype=jnp.float32,
            activation_fn=tanh,
            measure="legt",
            lambda_n=2.0,
            rnn_cell=GRUCell,
        )
        for _ in range(stack_number)
    ]


# ----------
# -- lagt --
# ----------
@pytest.fixture
def hippo_lagt_gru_cell_list():
    stack_number = 3
    input_size = 28 * 28
    hidden_size = 256
    return [
        HiPPOCell(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=True,
            param_dtype=jnp.float32,
            activation_fn=tanh,
            measure="lagt",
            alpha=0.0,
            beta=1.0,
            rnn_cell=GRUCell,
        )
        for _ in range(stack_number)
    ]


# ----------
# --- fru --
# ----------
@pytest.fixture
def hippo_fru_gru_cell_list():
    stack_number = 3
    input_size = 28 * 28
    hidden_size = 256
    return [
        HiPPOCell(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=True,
            param_dtype=jnp.float32,
            activation_fn=tanh,
            measure="fourier",
            fourier_type="fru",
            rnn_cell=GRUCell,
        )
        for _ in range(stack_number)
    ]


# ------------
# --- fout ---
# ------------
@pytest.fixture
def hippo_fout_gru_cell_list():
    stack_number = 3
    input_size = 28 * 28
    hidden_size = 256
    return [
        HiPPOCell(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=True,
            param_dtype=jnp.float32,
            activation_fn=tanh,
            measure="fourier",
            fourier_type="fout",
            rnn_cell=GRUCell,
        )
        for _ in range(stack_number)
    ]


# ------------
# --- foud ---
# ------------
@pytest.fixture
def hippo_foud_gru_cell_list():
    stack_number = 3
    input_size = 28 * 28
    hidden_size = 256
    return [
        HiPPOCell(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=True,
            param_dtype=jnp.float32,
            activation_fn=tanh,
            measure="fourier",
            fourier_type="foud",
            rnn_cell=GRUCell,
        )
        for _ in range(stack_number)
    ]
