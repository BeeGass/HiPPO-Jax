import pytest
import jax
from jax import vmap
from src.data.process import moving_window, rolling_window

# --- Random Keys


@pytest.fixture
def key_generator():
    seed = 1701
    key = jax.random.PRNGKey(seed)
    num_copies = 20
    return jax.random.split(key, num=num_copies)


@pytest.fixture
def rnn_key(key_generator):
    return key_generator[1]


@pytest.fixture
def lstm_key(key_generator):
    return key_generator[2]


@pytest.fixture
def gru_key(key_generator):
    return key_generator[3]


# -----------------------
# ----- HiPPO LSTM ------
# -----------------------


@pytest.fixture
def lstm_legt_key(key_generator):
    return key_generator[4]


@pytest.fixture
def lstm_lmu_key(key_generator):
    return key_generator[5]


@pytest.fixture
def lstm_lagt_key(key_generator):
    return key_generator[6]


@pytest.fixture
def lstm_legs_key(key_generator):
    return key_generator[7]


@pytest.fixture
def lstm_fru_key(key_generator):
    return key_generator[8]


@pytest.fixture
def lstm_fout_key(key_generator):
    return key_generator[9]


@pytest.fixture
def lstm_foud_key(key_generator):
    return key_generator[10]


# -----------------------
# ------ HiPPO GRU ------
# -----------------------


@pytest.fixture
def gru_legt_key(key_generator):
    return key_generator[11]


@pytest.fixture
def gru_lmu_key(key_generator):
    return key_generator[12]


@pytest.fixture
def gru_lagt_key(key_generator):
    return key_generator[13]


@pytest.fixture
def gru_legs_key(key_generator):
    return key_generator[14]


@pytest.fixture
def gru_fru_key(key_generator):
    return key_generator[15]


@pytest.fixture
def gru_fout_key(key_generator):
    return key_generator[16]


@pytest.fixture
def gru_foud_key(key_generator):
    return key_generator[17]


# -----------------------
# --- Sequence Inputs ---
# -----------------------


@pytest.fixture
def random_32_input(key_generator):
    batch_size = 32
    data_size = 28 * 28
    input_size = 5
    x = jax.random.randint(key_generator[18], (batch_size, data_size), 0, 244)
    return vmap(moving_window, in_axes=(0, None))(x, input_size)


@pytest.fixture
def random_64_input(key_generator):
    batch_size = 64
    data_size = 28 * 28
    input_size = 5
    x = jax.random.randint(key_generator[19], (batch_size, data_size), 0, 244)
    return vmap(moving_window, in_axes=(0, None))(x, input_size)
