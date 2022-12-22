import pytest
from src.models.rnn.cells import GRUCell, HiPPOCell, LSTMCell, RNNCell
import jax.numpy as jnp
from flax.linen.activation import tanh
from flax.linen.activation import sigmoid

# -----------------------
# -------- cells --------
# -----------------------


@pytest.fixture
def rnn_cell():
    input_size = 28 * 28
    hidden_size = 256
    return RNNCell(
        input_size=input_size,
        hidden_size=hidden_size,
        bias=True,
        param_dtype=jnp.float32,
        activation_fn=tanh,
    )


@pytest.fixture
def lstm_cell():
    input_size = 28 * 28
    hidden_size = 256
    return LSTMCell(
        input_size=input_size,
        hidden_size=hidden_size,
        bias=True,
        param_dtype=jnp.float32,
        gate_fn=sigmoid,
        activation_fn=tanh,
    )


@pytest.fixture
def gru_cell():
    input_size = 28 * 28
    hidden_size = 256
    return GRUCell(
        input_size=input_size,
        hidden_size=hidden_size,
        bias=True,
        param_dtype=jnp.float32,
        gate_fn=sigmoid,
        activation_fn=tanh,
    )


# ----------------------------
# ----- HiPPO LSTM cells -----
# ----------------------------

# ----------
# -- legs --
# ----------
@pytest.fixture
def hippo_legs_lstm_cell():
    input_size = 28 * 28
    hidden_size = 256
    return HiPPOCell(
        input_size=input_size,
        hidden_size=hidden_size,
        bias=True,
        param_dtype=jnp.float32,
        activation_fn=tanh,
        measure="legs",
        rnn_cell=LSTMCell,
    )


# ----------
# -- legt --
# ----------
@pytest.fixture
def hippo_legt_lstm_cell():
    input_size = 28 * 28
    hidden_size = 256
    return HiPPOCell(
        input_size=input_size,
        hidden_size=hidden_size,
        bias=True,
        param_dtype=jnp.float32,
        activation_fn=tanh,
        measure="legt",
        lambda_n=1.0,
        rnn_cell=LSTMCell,
    )


# ----------
# -- lmu --
# ----------
@pytest.fixture
def hippo_lmu_lstm_cell():
    input_size = 28 * 28
    hidden_size = 256
    return HiPPOCell(
        input_size=input_size,
        hidden_size=hidden_size,
        bias=True,
        param_dtype=jnp.float32,
        activation_fn=tanh,
        measure="lmu",
        lambda_n=2.0,
        rnn_cell=LSTMCell,
    )


# ----------
# -- lagt --
# ----------
@pytest.fixture
def hippo_lagt_lstm_cell():
    input_size = 28 * 28
    hidden_size = 256
    return HiPPOCell(
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


# ----------
# --- fru --
# ----------
@pytest.fixture
def hippo_fru_lstm_cell():
    input_size = 28 * 28
    hidden_size = 256
    return HiPPOCell(
        input_size=input_size,
        hidden_size=hidden_size,
        bias=True,
        param_dtype=jnp.float32,
        activation_fn=tanh,
        measure="fourier",
        fourier_type="fru",
        rnn_cell=LSTMCell,
    )


# ------------
# --- fout ---
# ------------
@pytest.fixture
def hippo_fout_lstm_cell():
    input_size = 28 * 28
    hidden_size = 256
    return HiPPOCell(
        input_size=input_size,
        hidden_size=hidden_size,
        bias=True,
        param_dtype=jnp.float32,
        activation_fn=tanh,
        measure="fourier",
        fourier_type="fout",
        rnn_cell=LSTMCell,
    )


# ------------
# --- foud ---
# ------------
@pytest.fixture
def hippo_foud_lstm_cell():
    input_size = 28 * 28
    hidden_size = 256
    return HiPPOCell(
        input_size=input_size,
        hidden_size=hidden_size,
        bias=True,
        param_dtype=jnp.float32,
        activation_fn=tanh,
        measure="fourier",
        fourier_type="foud",
        rnn_cell=LSTMCell,
    )


# ---------------------------
# ----- HiPPO GRU cells -----
# ---------------------------

# ----------
# -- legs --
# ----------
@pytest.fixture
def hippo_legs_gru_cell():
    input_size = 28 * 28
    hidden_size = 256
    return HiPPOCell(
        input_size=input_size,
        hidden_size=hidden_size,
        bias=True,
        param_dtype=jnp.float32,
        activation_fn=tanh,
        measure="legs",
        rnn_cell=GRUCell,
    )


# ----------
# -- legt --
# ----------
@pytest.fixture
def hippo_legt_gru_cell():
    input_size = 28 * 28
    hidden_size = 256
    return HiPPOCell(
        input_size=input_size,
        hidden_size=hidden_size,
        bias=True,
        param_dtype=jnp.float32,
        activation_fn=tanh,
        measure="legt",
        lambda_n=1.0,
        rnn_cell=GRUCell,
    )


# ----------
# -- lmu --
# ----------
@pytest.fixture
def hippo_lmu_gru_cell():
    input_size = 28 * 28
    hidden_size = 256
    return HiPPOCell(
        input_size=input_size,
        hidden_size=hidden_size,
        bias=True,
        param_dtype=jnp.float32,
        activation_fn=tanh,
        measure="lmu",
        lambda_n=2.0,
        rnn_cell=GRUCell,
    )


# ----------
# -- lagt --
# ----------
@pytest.fixture
def hippo_lagt_gru_cell():
    input_size = 28 * 28
    hidden_size = 256
    return HiPPOCell(
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


# ----------
# --- fru --
# ----------
@pytest.fixture
def hippo_fru_gru_cell():
    input_size = 28 * 28
    hidden_size = 256
    return HiPPOCell(
        input_size=input_size,
        hidden_size=hidden_size,
        bias=True,
        param_dtype=jnp.float32,
        activation_fn=tanh,
        measure="fourier",
        fourier_type="fru",
        rnn_cell=GRUCell,
    )


# ------------
# --- fout ---
# ------------
@pytest.fixture
def hippo_fout_gru_cell():
    input_size = 28 * 28
    hidden_size = 256
    return HiPPOCell(
        input_size=input_size,
        hidden_size=hidden_size,
        bias=True,
        param_dtype=jnp.float32,
        activation_fn=tanh,
        measure="fourier",
        fourier_type="fout",
        rnn_cell=GRUCell,
    )


# ------------
# --- foud ---
# ------------
@pytest.fixture
def hippo_foud_gru_cell():
    input_size = 28 * 28
    hidden_size = 256
    return HiPPOCell(
        input_size=input_size,
        hidden_size=hidden_size,
        bias=True,
        param_dtype=jnp.float32,
        activation_fn=tanh,
        measure="fourier",
        fourier_type="foud",
        rnn_cell=GRUCell,
    )
