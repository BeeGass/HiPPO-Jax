import jax
import pytest

from src.data.process import whitesignal

# --- Random Keys


@pytest.fixture
def key_generator():
    seed = 1701
    key = jax.random.PRNGKey(seed)
    num_copies = 13
    return jax.random.split(key, num=num_copies)


@pytest.fixture
def legt_key(key_generator):
    return key_generator[1]


@pytest.fixture
def lmu_key(key_generator):
    return key_generator[2]


@pytest.fixture
def lagt_key(key_generator):
    return key_generator[3]


@pytest.fixture
def legs_key(key_generator):
    return key_generator[4]


@pytest.fixture
def fru_key(key_generator):
    return key_generator[5]


@pytest.fixture
def fout_key(key_generator):
    return key_generator[6]


@pytest.fixture
def foud_key(key_generator):
    return key_generator[7]


# --- Coefficients hyperparameter value ---


@pytest.fixture
def N():
    N = 1
    return N


@pytest.fixture
def N2():
    N = 2
    return N


@pytest.fixture
def N16():
    N = 16
    return N


@pytest.fixture
def big_N():
    N = 256
    return N


# --- Sequence Inputs ---
@pytest.fixture
def random_1_input(key_generator):
    batch_size = 1
    period = 3
    dt = 1e-3
    freq = 3

    return whitesignal(
        key=key_generator[8], period=period, dt=dt, freq=freq, batch_shape=(batch_size,)
    )


@pytest.fixture
def random_16_input(key_generator):
    batch_size = 16
    period = 3
    dt = 1e-3
    freq = 3

    return whitesignal(
        key=key_generator[9], period=period, dt=dt, freq=freq, batch_shape=(batch_size,)
    )


@pytest.fixture
def random_32_input(key_generator):
    batch_size = 32
    period = 3
    dt = 1e-3
    freq = 3

    return whitesignal(
        key=key_generator[10],
        period=period,
        dt=dt,
        freq=freq,
        batch_shape=(batch_size,),
    )


@pytest.fixture
def random_64_input(key_generator):
    batch_size = 64
    period = 3
    dt = 1e-3
    freq = 3

    return whitesignal(
        key=key_generator[11],
        period=period,
        dt=dt,
        freq=freq,
        batch_shape=(batch_size,),
    )
