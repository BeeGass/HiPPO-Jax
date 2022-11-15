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


def test_deep_rnn_shaping(deep_rnn, random_32_input, rnn_key):
    print("Testing Deep RNN")
    key1, key2, = (
        rnn_key[0],
        rnn_key[1],
    )
    batch_size = 32
    hidden_size = 256
    init_carry = deep_rnn.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = deep_rnn.init(
        key2,
        init_carry,
        random_32_input,
    )

    carry, y = deep_rnn.apply(
        params,
        init_carry,
        random_32_input,
    )

    h_t, c_t = carry
    print(f"h_t shape: {h_t.shape}")
    print(f"c_t shape: {c_t.shape}")
    print(f"y shape: {y.shape}")
    assert (h_t.shape) == (32, 256)
    assert (c_t.shape) == (32, 256)
    assert (y.shape) == (32, 10)


def test_deep_lstm_shaping(deep_lstm, random_32_input, lstm_key):
    print("Testing Deep LSTM")
    key1, key2, = (
        lstm_key[0],
        lstm_key[1],
    )
    batch_size = 32
    hidden_size = 256
    init_carry = deep_lstm.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = deep_lstm.init(
        key2,
        init_carry,
        random_32_input,
    )

    carry, y = deep_lstm.apply(
        params,
        init_carry,
        random_32_input,
    )

    h_t, c_t = carry
    print(f"h_t shape: {h_t.shape}")
    print(f"c_t shape: {c_t.shape}")
    print(f"y shape: {y.shape}")
    assert (h_t.shape) == (32, 256)
    assert (c_t.shape) == (32, 256)
    assert (y.shape) == (32, 10)


def test_deep_gru_shaping(deep_gru, random_32_input, gru_key):
    print("Testing Deep GRU")
    key1, key2, = (
        gru_key[0],
        gru_key[1],
    )
    batch_size = 32
    hidden_size = 256
    init_carry = deep_gru.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = deep_gru.init(
        key2,
        init_carry,
        random_32_input,
    )

    carry, y = deep_gru.apply(
        params,
        init_carry,
        random_32_input,
    )

    h_t, c_t = carry
    print(f"h_t shape: {h_t.shape}")
    print(f"c_t shape: {c_t.shape}")
    print(f"y shape: {y.shape}")
    assert (h_t.shape) == (32, 256)
    assert (c_t.shape) == (32, 256)
    assert (y.shape) == (32, 10)
