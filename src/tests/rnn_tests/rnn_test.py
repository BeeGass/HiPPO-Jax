import pytest
from src.tests.rnn_tests.deep_rnn_fixtures import (
    deep_rnn,
    deep_lstm,
    deep_gru,
)
from src.tests.rnn_tests.deep_rnn_fixtures import (
    deep_hippo_legs_lstm,
    deep_hippo_legt_lstm,
    deep_hippo_lmu_lstm,
    deep_hippo_lagt_lstm,
    deep_hippo_fru_lstm,
    deep_hippo_fout_lstm,
    deep_hippo_foud_lstm,
)
from src.tests.rnn_tests.deep_rnn_fixtures import (
    deep_hippo_legs_gru,
    deep_hippo_legt_gru,
    deep_hippo_lmu_gru,
    deep_hippo_lagt_gru,
    deep_hippo_fru_gru,
    deep_hippo_fout_gru,
    deep_hippo_foud_gru,
)
from src.tests.rnn_tests.rnn_utils import rnn_key, gru_key, lstm_key
from src.tests.rnn_tests.rnn_utils import (
    lstm_legt_key,
    lstm_lmu_key,
    lstm_lagt_key,
    lstm_legs_key,
    lstm_fru_key,
    lstm_fout_key,
    lstm_foud_key,
)
from src.tests.rnn_tests.rnn_utils import (
    gru_legt_key,
    gru_lmu_key,
    gru_lagt_key,
    gru_legs_key,
    gru_fru_key,
    gru_fout_key,
    gru_foud_key,
)
from src.tests.rnn_tests.rnn_utils import random_32_input, random_64_input
import jax
from flax import linen as nn

# ------------------------------------------------ #
# -------------------- Test RNNs ----------------- #
# ------------------------------------------------ #


def test_deep_rnn(deep_rnn, random_32_input, rnn_key):
    print("Testing Deep RNN")
    num_copies = 3
    rng, key, subkey = jax.random.split(rnn_key, num=num_copies)
    print(f"input shape: {random_32_input.shape}")
    batch_size = 32
    hidden_size = 256
    params = deep_rnn.init(
        rng,
        deep_rnn.initialize_carry(
            rng=key,
            batch_size=(batch_size,),
            hidden_size=hidden_size,
            init_fn=nn.initializers.zeros,
        ),
        random_32_input,
    )

    carry, y = deep_rnn.apply(
        params,
        deep_rnn.initialize_carry(
            rng=subkey,
            batch_size=(batch_size,),
            hidden_size=hidden_size,
            init_fn=nn.initializers.zeros,
        ),
        random_32_input,
    )

    h_t, c_t = carry
    print(f"h_t shape: {h_t.shape}")
    print(f"c_t shape: {c_t.shape}")
    print(f"y shape: {y.shape}")
    assert (h_t.shape) == (32, 256)
    assert (c_t.shape) == (32, 256)
    assert (y.shape) == (32, 10)


def test_deep_lstm(deep_lstm, random_32_input, lstm_key):
    print("Testing Deep LSTM")
    num_copies = 3
    rng, key, subkey = jax.random.split(lstm_key, num=num_copies)
    print(f"input shape: {random_32_input.shape}")
    batch_size = 32
    hidden_size = 256
    params = deep_lstm.init(
        rng,
        deep_lstm.initialize_carry(
            rng=key,
            batch_size=(batch_size,),
            hidden_size=hidden_size,
            init_fn=nn.initializers.zeros,
        ),
        random_32_input,
    )

    carry, y = deep_lstm.apply(
        params,
        deep_lstm.initialize_carry(
            rng=subkey,
            batch_size=(batch_size,),
            hidden_size=hidden_size,
            init_fn=nn.initializers.zeros,
        ),
        random_32_input,
    )

    h_t, c_t = carry
    print(f"h_t shape: {h_t.shape}")
    print(f"c_t shape: {c_t.shape}")
    print(f"y shape: {y.shape}")
    assert (h_t.shape) == (32, 256)
    assert (c_t.shape) == (32, 256)
    assert (y.shape) == (32, 10)


def test_deep_gru(deep_gru, random_32_input, gru_key):
    print("Testing Deep GRU")
    num_copies = 3
    rng, key, subkey = jax.random.split(gru_key, num=num_copies)
    print(f"input shape: {random_32_input.shape}")
    batch_size = 32
    hidden_size = 256
    params = deep_gru.init(
        rng,
        deep_gru.initialize_carry(
            rng=key,
            batch_size=(batch_size,),
            hidden_size=hidden_size,
            init_fn=nn.initializers.zeros,
        ),
        random_32_input,
    )

    carry, y = deep_gru.apply(
        params,
        deep_gru.initialize_carry(
            rng=subkey,
            batch_size=(batch_size,),
            hidden_size=hidden_size,
            init_fn=nn.initializers.zeros,
        ),
        random_32_input,
    )

    h_t, c_t = carry
    print(f"h_t shape: {h_t.shape}")
    print(f"c_t shape: {c_t.shape}")
    print(f"y shape: {y.shape}")
    assert (h_t.shape) == (32, 256)
    assert (c_t.shape) == (32, 256)
    assert (y.shape) == (32, 10)
