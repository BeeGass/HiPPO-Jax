import pytest
import jax
import jax.numpy as jnp
import numpy as np

# --- Random Keys


@pytest.fixture
def key_generator():
    seed = 1701
    key = jax.random.PRNGKey(seed)
    num_copies = 8
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
def random_input():
    np.random.seed(1701)
    return np.array(
        [
            [0.3527],
            [0.6617],
            [0.2434],
            [0.6674],
            [1.2293],
            [0.0964],
            [-2.2756],
            [0.5618],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def ones_input():
    np.random.seed(1701)
    return np.array(
        [
            [1.0000],
            [1.0000],
            [1.0000],
            [1.0000],
            [1.0000],
            [1.0000],
            [1.0000],
            [1.0000],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def zeros_input():
    np.random.seed(1701)
    return np.array(
        [
            [0.0000],
            [0.0000],
            [0.0000],
            [0.0000],
            [0.0000],
            [0.0000],
            [0.0000],
            [0.0000],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def desc_input():
    np.random.seed(1701)
    return np.array(
        [
            [7.0000],
            [6.0000],
            [5.0000],
            [4.0000],
            [3.0000],
            [2.0000],
            [1.0000],
            [0.0000],
        ],
        dtype=np.float32,
    )
