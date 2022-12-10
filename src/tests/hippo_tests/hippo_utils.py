import pytest
import jax
from jax import vmap
import jax.numpy as jnp
import numpy as np
from src.data.process import moving_window, rolling_window


# --- Random Keys


@pytest.fixture
def key_generator():
    seed = 1701
    key = jax.random.PRNGKey(seed)
    num_copies = 12
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
    data_size = 512
    input_size = 1
    x = jax.random.randint(key_generator[8], (batch_size, data_size), 0, 255)
    return vmap(moving_window, in_axes=(0, None))(x, input_size)


@pytest.fixture
def random_16_input(key_generator):
    batch_size = 16
    data_size = 512
    input_size = 1
    x = jax.random.randint(key_generator[8], (batch_size, data_size), 0, 255)
    return vmap(moving_window, in_axes=(0, None))(x, input_size)


@pytest.fixture
def random_32_input(key_generator):
    batch_size = 32
    data_size = 512
    input_size = 1
    x = jax.random.randint(key_generator[9], (batch_size, data_size), 0, 255)
    return vmap(moving_window, in_axes=(0, None))(x, input_size)


@pytest.fixture
def random_64_input(key_generator):
    batch_size = 64
    data_size = 512
    input_size = 1
    x = jax.random.randint(key_generator[10], (batch_size, data_size), 0, 255)
    return vmap(moving_window, in_axes=(0, None))(x, input_size)
