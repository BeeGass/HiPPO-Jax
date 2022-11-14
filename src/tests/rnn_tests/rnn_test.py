import pytest
from src.tests.rnn_tests.deep_rnn_fixtures import (
    deep_rnn,
    deep_lstm,
    deep_gru,
)
from src.tests.rnn_tests.rnn_fixtures import (
    deep_hippo_legs_lstm,
    deep_hippo_legt_lstm,
    deep_hippo_lmu_lstm,
    deep_hippo_lagt_lstm,
    deep_hippo_fru_lstm,
    deep_hippo_fout_lstm,
    deep_hippo_foud_lstm,
)
from src.tests.rnn_tests.rnn_fixtures import (
    deep_hippo_legs_gru,
    deep_hippo_legt_gru,
    deep_hippo_lmu_gru,
    deep_hippo_lagt_gru,
    deep_hippo_fru_gru,
    deep_hippo_fout_gru,
    deep_hippo_foud_gru,
)
import jax
from flax import linen as nn
import jax.numpy as jnp

# ------------------------------------------------ #
# -------------------- Test RNNs ----------------- #
# ------------------------------------------------ #


def test_deep_rnn(deep_rnn, flax_deep_rnn, random_input, rnn_key):
    print("Testing Deep RNN")
    key, subkey = jax.random.split(rnn_key)
    batch_size = 1
    hidden_size = 256
    # TODO: pass in sequence data
    params = deep_rnn.init(
        deep_rnn.initialize_carry(
            rng=key,
            batch_size=batch_size,
            hidden_size=hidden_size,
            init_fn=nn.initializers.zeros,
        ),
        input=x,
    )
    carry, y = deep_rnn.apply(
        params,
        deep_rnn.initialize_carry(
            rng=key,
            batch_size=batch_size,
            hidden_size=hidden_size,
            init_fn=nn.initializers.zeros,
        ),
        input=x,
    )
    # TODO: init and apply flax_deep_rnn
    # TODO: compare carry and y shapes from both deep_rnn and flax_deep_rnn


def test_deep_lstm(deep_lstm, flax_deep_lstm, random_input, lstm_key):
    print("Testing Deep LSTM")
    key, subkey = jax.random.split(lstm_key)
    batch_size = 1
    hidden_size = 256
    # TODO: pass in sequence data
    params = deep_lstm.init(
        deep_lstm.initialize_carry(
            rng=key,
            batch_size=batch_size,
            hidden_size=hidden_size,
            init_fn=nn.initializers.zeros,
        ),
        input=x,
    )
    carry, y = deep_lstm.apply(
        params,
        deep_lstm.initialize_carry(
            rng=key,
            batch_size=batch_size,
            hidden_size=hidden_size,
            init_fn=nn.initializers.zeros,
        ),
        input=x,
    )
    # TODO: init and apply flax_deep_lstm
    # TODO: compare carry and y shapes from both deep_lstm and flax_deep_lstm


def test_deep_gru(deep_gru, flax_deep_gru, random_input, gru_key):
    print("Testing Deep GRU")
    key, subkey = jax.random.split(gru_key)
    batch_size = 1
    hidden_size = 256
    # TODO: pass in sequence data
    params = deep_gru.init(
        deep_gru.initialize_carry(
            rng=key,
            batch_size=batch_size,
            hidden_size=hidden_size,
            init_fn=nn.initializers.zeros,
        ),
        input=x,
    )
    carry, y = deep_gru.apply(
        params,
        deep_gru.initialize_carry(
            rng=key,
            batch_size=batch_size,
            hidden_size=hidden_size,
            init_fn=nn.initializers.zeros,
        ),
        input=x,
    )
    # TODO: init and apply flax_deep_gru
    # TODO: compare carry and y shapes from both deep_gru and flax_deep_gru
