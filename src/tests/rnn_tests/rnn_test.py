# List Of RNN Cells
import jax
import pytest
from flax import linen as nn

# List Only Containing One HiPPO-GRU Cell
# List Only Containing One HiPPO-LSTM Cell
# List Only Containing One RNN Cell
# List Of HiPPO-GRU Cells
# List Of HiPPO-LSTM Cells
from src.tests.rnn_tests.rnn_fixtures import (
    gru_cell_list,
    gru_cell_single,
    hippo_foud_gru_cell_list,
    hippo_foud_gru_cell_single,
    hippo_foud_lstm_cell_list,
    hippo_foud_lstm_cell_single,
    hippo_fout_gru_cell_list,
    hippo_fout_gru_cell_single,
    hippo_fout_lstm_cell_list,
    hippo_fout_lstm_cell_single,
    hippo_fru_gru_cell_list,
    hippo_fru_gru_cell_single,
    hippo_fru_lstm_cell_list,
    hippo_fru_lstm_cell_single,
    hippo_lagt_gru_cell_list,
    hippo_lagt_gru_cell_single,
    hippo_lagt_lstm_cell_list,
    hippo_lagt_lstm_cell_single,
    hippo_legs_gru_cell_list,
    hippo_legs_gru_cell_single,
    hippo_legs_lstm_cell_list,
    hippo_legs_lstm_cell_single,
    hippo_legt_gru_cell_list,
    hippo_legt_gru_cell_single,
    hippo_legt_lstm_cell_list,
    hippo_legt_lstm_cell_single,
    hippo_lmu_gru_cell_list,
    hippo_lmu_gru_cell_single,
    hippo_lmu_lstm_cell_list,
    hippo_lmu_lstm_cell_single,
    lstm_cell_list,
    lstm_cell_single,
    rnn_cell_list,
    rnn_cell_single,
)

# Inputs for the models
# Psuedo-Random Number Generator Keys for Deep Bidirectional RNNs
# Psuedo-Random Number Generator Keys for Deep HiPPO RNNs
# Psuedo-Random Number Generator Keys for Single Cell HiPPO RNNs
# Psuedo-Random Number Generator Keys for Deep Bidirectional RNNs
# Psuedo-Random Number Generator Keys for Deep RNNs
# Psuedo-Random Number Generator Keys for Single Cell Bidirectional RNNs
# Psuedo-Random Number Generator Keys for Single Cell RNNs
from src.tests.rnn_tests.rnn_utils import (
    deep_bigru_key,
    deep_bilstm_key,
    deep_birnn_key,
    deep_hippo_foud_bigru_key,
    deep_hippo_foud_bilstm_key,
    deep_hippo_fout_bigru_key,
    deep_hippo_fout_bilstm_key,
    deep_hippo_fru_bigru_key,
    deep_hippo_fru_bilstm_key,
    deep_hippo_lagt_bigru_key,
    deep_hippo_lagt_bilstm_key,
    deep_hippo_legs_bigru_key,
    deep_hippo_legs_bilstm_key,
    deep_hippo_legt_bigru_key,
    deep_hippo_legt_bilstm_key,
    deep_hippo_lmu_bigru_key,
    deep_hippo_lmu_bilstm_key,
    many_to_many_deep_gru_key,
    many_to_many_deep_hippo_foud_gru_key,
    many_to_many_deep_hippo_foud_lstm_key,
    many_to_many_deep_hippo_fout_gru_key,
    many_to_many_deep_hippo_fout_lstm_key,
    many_to_many_deep_hippo_fru_gru_key,
    many_to_many_deep_hippo_fru_lstm_key,
    many_to_many_deep_hippo_lagt_gru_key,
    many_to_many_deep_hippo_lagt_lstm_key,
    many_to_many_deep_hippo_legs_gru_key,
    many_to_many_deep_hippo_legs_lstm_key,
    many_to_many_deep_hippo_legt_gru_key,
    many_to_many_deep_hippo_legt_lstm_key,
    many_to_many_deep_hippo_lmu_gru_key,
    many_to_many_deep_hippo_lmu_lstm_key,
    many_to_many_deep_lstm_key,
    many_to_many_deep_rnn_key,
    many_to_many_single_cell_gru_key,
    many_to_many_single_cell_lstm_key,
    many_to_many_single_cell_rnn_key,
    many_to_many_single_hippo_foud_gru_key,
    many_to_many_single_hippo_foud_lstm_key,
    many_to_many_single_hippo_fout_gru_key,
    many_to_many_single_hippo_fout_lstm_key,
    many_to_many_single_hippo_fru_gru_key,
    many_to_many_single_hippo_fru_lstm_key,
    many_to_many_single_hippo_lagt_gru_key,
    many_to_many_single_hippo_lagt_lstm_key,
    many_to_many_single_hippo_legs_gru_key,
    many_to_many_single_hippo_legs_lstm_key,
    many_to_many_single_hippo_legt_gru_key,
    many_to_many_single_hippo_legt_lstm_key,
    many_to_many_single_hippo_lmu_gru_key,
    many_to_many_single_hippo_lmu_lstm_key,
    many_to_one_deep_gru_key,
    many_to_one_deep_hippo_foud_gru_key,
    many_to_one_deep_hippo_foud_lstm_key,
    many_to_one_deep_hippo_fout_gru_key,
    many_to_one_deep_hippo_fout_lstm_key,
    many_to_one_deep_hippo_fru_gru_key,
    many_to_one_deep_hippo_fru_lstm_key,
    many_to_one_deep_hippo_lagt_gru_key,
    many_to_one_deep_hippo_lagt_lstm_key,
    many_to_one_deep_hippo_legs_gru_key,
    many_to_one_deep_hippo_legs_lstm_key,
    many_to_one_deep_hippo_legt_gru_key,
    many_to_one_deep_hippo_legt_lstm_key,
    many_to_one_deep_hippo_lmu_gru_key,
    many_to_one_deep_hippo_lmu_lstm_key,
    many_to_one_deep_lstm_key,
    many_to_one_deep_rnn_key,
    many_to_one_single_cell_gru_key,
    many_to_one_single_cell_lstm_key,
    many_to_one_single_cell_rnn_key,
    many_to_one_single_hippo_foud_gru_key,
    many_to_one_single_hippo_foud_lstm_key,
    many_to_one_single_hippo_fout_gru_key,
    many_to_one_single_hippo_fout_lstm_key,
    many_to_one_single_hippo_fru_gru_key,
    many_to_one_single_hippo_fru_lstm_key,
    many_to_one_single_hippo_lagt_gru_key,
    many_to_one_single_hippo_lagt_lstm_key,
    many_to_one_single_hippo_legs_gru_key,
    many_to_one_single_hippo_legs_lstm_key,
    many_to_one_single_hippo_legt_gru_key,
    many_to_one_single_hippo_legt_lstm_key,
    many_to_one_single_hippo_lmu_gru_key,
    many_to_one_single_hippo_lmu_lstm_key,
    one_to_many_deep_gru_key,
    one_to_many_deep_hippo_foud_gru_key,
    one_to_many_deep_hippo_foud_lstm_key,
    one_to_many_deep_hippo_fout_gru_key,
    one_to_many_deep_hippo_fout_lstm_key,
    one_to_many_deep_hippo_fru_gru_key,
    one_to_many_deep_hippo_fru_lstm_key,
    one_to_many_deep_hippo_lagt_gru_key,
    one_to_many_deep_hippo_lagt_lstm_key,
    one_to_many_deep_hippo_legs_gru_key,
    one_to_many_deep_hippo_legs_lstm_key,
    one_to_many_deep_hippo_legt_gru_key,
    one_to_many_deep_hippo_legt_lstm_key,
    one_to_many_deep_hippo_lmu_gru_key,
    one_to_many_deep_hippo_lmu_lstm_key,
    one_to_many_deep_lstm_key,
    one_to_many_deep_rnn_key,
    one_to_many_single_cell_gru_key,
    one_to_many_single_cell_lstm_key,
    one_to_many_single_cell_rnn_key,
    one_to_many_single_hippo_foud_gru_key,
    one_to_many_single_hippo_foud_lstm_key,
    one_to_many_single_hippo_fout_gru_key,
    one_to_many_single_hippo_fout_lstm_key,
    one_to_many_single_hippo_fru_gru_key,
    one_to_many_single_hippo_fru_lstm_key,
    one_to_many_single_hippo_lagt_gru_key,
    one_to_many_single_hippo_lagt_lstm_key,
    one_to_many_single_hippo_legs_gru_key,
    one_to_many_single_hippo_legs_lstm_key,
    one_to_many_single_hippo_legt_gru_key,
    one_to_many_single_hippo_legt_lstm_key,
    one_to_many_single_hippo_lmu_gru_key,
    one_to_many_single_hippo_lmu_lstm_key,
    random_16_input,
    random_32_input,
    random_64_input,
    single_cell_bigru_key,
    single_cell_bilstm_key,
    single_cell_birnn_key,
    single_cell_hippo_foud_bigru_key,
    single_cell_hippo_foud_bilstm_key,
    single_cell_hippo_fout_bigru_key,
    single_cell_hippo_fout_bilstm_key,
    single_cell_hippo_fru_bigru_key,
    single_cell_hippo_fru_bilstm_key,
    single_cell_hippo_lagt_bigru_key,
    single_cell_hippo_lagt_bilstm_key,
    single_cell_hippo_legs_bigru_key,
    single_cell_hippo_legs_bilstm_key,
    single_cell_hippo_legt_bigru_key,
    single_cell_hippo_legt_bilstm_key,
    single_cell_hippo_lmu_bigru_key,
    single_cell_hippo_lmu_bilstm_key,
)

# ----------------------------------------------------------------
# ------------------ Single Cell Architectures Tests -------------
# ----------------------------------------------------------------

# -------------------------
# ------ one to many ------
# -------------------------


def test_one_to_many_single_cell_rnn_shaping(
    one_to_many_single_cell_rnn, random_16_input, one_to_many_single_cell_rnn_key
):
    print("Testing One To Many RNN")
    key1, key2, = (
        one_to_many_single_cell_rnn_key[0],
        one_to_many_single_cell_rnn_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = one_to_many_single_cell_rnn
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


def test_one_to_many_single_cell_lstm_shaping(
    one_to_many_single_cell_lstm, random_16_input, one_to_many_single_cell_lstm_key
):
    print("Testing One To Many LSTM")
    key1, key2, = (
        one_to_many_single_cell_lstm_key[0],
        one_to_many_single_cell_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = one_to_many_single_cell_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


def test_one_to_many_single_cell_gru_shaping(
    one_to_many_single_cell_gru, random_16_input, one_to_many_single_cell_gru_key
):
    print("Testing One To Many GRU")
    key1, key2, = (
        one_to_many_single_cell_gru_key[0],
        one_to_many_single_cell_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = one_to_many_single_cell_gru
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# -------------------------
# ------ many to one ------
# -------------------------


def test_many_to_one_single_cell_rnn_shaping(
    many_to_one_single_cell_rnn, random_16_input, many_to_one_single_cell_rnn_key
):
    print("Testing Many To One RNN")
    key1, key2, = (
        many_to_one_single_cell_rnn_key[0],
        many_to_one_single_cell_rnn_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_one_single_cell_rnn
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


def test_many_to_one_single_cell_lstm_shaping(
    many_to_one_single_cell_lstm, random_16_input, many_to_one_single_cell_lstm_key
):
    print("Testing Many To One LSTM")
    key1, key2, = (
        many_to_one_single_cell_lstm_key[0],
        many_to_one_single_cell_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_one_single_cell_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


def test_many_to_one_single_cell_gru_shaping(
    many_to_one_single_cell_gru, random_16_input, many_to_one_single_cell_gru_key
):
    print("Testing Many To One GRU")
    key1, key2, = (
        many_to_one_single_cell_gru_key[0],
        many_to_one_single_cell_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_one_single_cell_gru
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# -------------------------
# ------ many to many -----
# -------------------------


def test_many_to_many_single_cell_rnn_shaping(
    many_to_many_single_cell_rnn, random_16_input, many_to_many_single_cell_rnn_key
):
    print("Testing Many To Many RNN")
    key1, key2, = (
        many_to_many_single_cell_rnn_key[0],
        many_to_many_single_cell_rnn_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_many_single_cell_rnn
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


def test_many_to_many_single_cell_lstm_shaping(
    many_to_many_single_cell_lstm, random_16_input, many_to_many_single_cell_lstm_key
):
    print("Testing Many To Many LSTM")
    key1, key2, = (
        many_to_many_single_cell_lstm_key[0],
        many_to_many_single_cell_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_many_single_cell_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


def test_many_to_many_single_cell_gru_shaping(
    many_to_many_single_cell_gru, random_16_input, many_to_many_single_cell_gru_key
):
    print("Testing Many To Many GRU")
    key1, key2, = (
        many_to_many_single_cell_gru_key[0],
        many_to_many_single_cell_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_many_single_cell_gru
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------------------------------------------------------------
# ------------------ Single Cell Bidirectional Tests -------------
# ----------------------------------------------------------------


def test_single_cell_birnn_shaping(
    single_cell_birnn, random_16_input, single_cell_birnn_key
):
    print("Testing Bidirectional RNN")
    key1, key2, = (
        single_cell_birnn_key[0],
        single_cell_birnn_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = single_cell_birnn
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


def test_single_cell_bilstm_shaping(
    single_cell_bilstm, random_16_input, single_cell_bilstm_key
):
    print("Testing Bidirectional LSTM")
    key1, key2, = (
        single_cell_bilstm_key[0],
        single_cell_bilstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = single_cell_bilstm
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


def test_single_cell_bigru_shaping(
    single_cell_bigru, random_16_input, single_cell_bigru_key
):
    print("Testing Bidirectional GRU")
    key1, key2, = (
        single_cell_bigru_key[0],
        single_cell_bigru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = single_cell_bigru
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------------------------------------------------------------
# --------------------------- Deep RNN Tests ---------------------
# ----------------------------------------------------------------

# -------------------------
# ------ one to many ------
# -------------------------


def test_one_to_many_deep_rnn_shaping(
    one_to_many_deep_rnn, random_16_input, one_to_many_deep_rnn_key
):
    print("Testing One To Many Deep RNN")
    key1, key2, = (
        one_to_many_deep_rnn_key[0],
        one_to_many_deep_rnn_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = one_to_many_deep_rnn
    init_carry = model.initialize_carry(
        rng=key1,
        layers=one_to_many_deep_rnn.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


def test_one_to_many_deep_lstm_shaping(
    one_to_many_deep_lstm, random_16_input, one_to_many_deep_lstm_key
):
    print("Testing One To Many Deep LSTM")
    key1, key2, = (
        one_to_many_deep_lstm_key[0],
        one_to_many_deep_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = one_to_many_deep_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        layers=one_to_many_deep_lstm.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


def test_one_to_many_deep_gru_shaping(
    one_to_many_deep_gru, random_16_input, one_to_many_deep_gru_key
):
    print("Testing One To Many Deep GRU")
    key1, key2, = (
        one_to_many_deep_gru_key[0],
        one_to_many_deep_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = one_to_many_deep_gru
    init_carry = model.initialize_carry(
        rng=key1,
        layers=one_to_many_deep_gru.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# -------------------------
# ------ many to one ------
# -------------------------


def test_many_to_one_deep_rnn_shaping(
    many_to_one_deep_rnn, random_16_input, many_to_one_deep_rnn_key
):
    print("Testing Many To One Deep RNN")
    key1, key2, = (
        many_to_one_deep_rnn_key[0],
        many_to_one_deep_rnn_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_one_deep_rnn
    init_carry = model.initialize_carry(
        rng=key1,
        layers=many_to_one_deep_rnn.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


def test_many_to_one_deep_lstm_shaping(
    many_to_one_deep_lstm, random_16_input, many_to_one_deep_lstm_key
):
    print("Testing Many To One Deep LSTM")
    key1, key2, = (
        many_to_one_deep_lstm_key[0],
        many_to_one_deep_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_one_deep_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        layers=many_to_one_deep_lstm.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


def test_many_to_one_deep_gru_shaping(
    many_to_one_deep_gru, random_16_input, many_to_one_deep_gru_key
):
    print("Testing Many To One Deep GRU")
    key1, key2, = (
        many_to_one_deep_gru_key[0],
        many_to_one_deep_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_one_deep_gru
    init_carry = model.initialize_carry(
        rng=key1,
        layers=many_to_one_deep_gru.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# -------------------------
# ------ many to many -----
# -------------------------


def test_many_to_many_deep_rnn_shaping(
    many_to_many_deep_rnn, random_16_input, many_to_many_deep_rnn_key
):
    print("Testing Many To Many Deep RNN")
    key1, key2, = (
        many_to_many_deep_rnn_key[0],
        many_to_many_deep_rnn_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_many_deep_rnn
    init_carry = model.initialize_carry(
        rng=key1,
        layers=many_to_many_deep_rnn.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


def test_many_to_many_deep_lstm_shaping(
    many_to_many_deep_lstm, random_16_input, many_to_many_deep_lstm_key
):
    print("Testing Many To Many Deep LSTM")
    key1, key2, = (
        many_to_many_deep_lstm_key[0],
        many_to_many_deep_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_many_deep_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        layers=many_to_many_deep_lstm.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


def test_many_to_many_deep_gru_shaping(
    many_to_many_deep_gru, random_16_input, many_to_many_deep_gru_key
):
    print("Testing Many To Many Deep GRU")
    key1, key2, = (
        many_to_many_deep_gru_key[0],
        many_to_many_deep_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_many_deep_gru
    init_carry = model.initialize_carry(
        rng=key1,
        layers=many_to_many_deep_gru.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------------------------------------------------------------
# ------------------------- Deep Bidirectional -------------------
# ----------------------------------------------------------------


def test_deep_birnn_shaping(deep_birnn, random_16_input, deep_birnn_key):

    print("Testing Deep Bidirectional RNN")
    key1, key2, = (
        deep_birnn_key[0],
        deep_birnn_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = deep_birnn
    init_carry = model.initialize_carry(
        rng=key1,
        layers=deep_birnn.forward_layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


def test_deep_bilstm_shaping(deep_bilstm, random_16_input, deep_bilstm_key):

    print("Testing Deep Bidirectional LSTM")
    key1, key2, = (
        deep_bilstm_key[0],
        deep_bilstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = deep_bilstm
    init_carry = model.initialize_carry(
        rng=key1,
        layers=deep_bilstm.forward_layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


def test_deep_bigru_shaping(deep_bigru, random_16_input, deep_bigru_key):

    print("Testing Deep Bidirectional GRU")
    key1, key2, = (
        deep_bigru_key[0],
        deep_bigru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = deep_bigru
    init_carry = model.initialize_carry(
        rng=key1,
        layers=deep_bigru.forward_layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------------------------------------------------------------
# -------------------- Single Cell HiPPO LSTM --------------------
# ----------------------------------------------------------------

# -------------------------
# ------ one to many ------
# -------------------------

# ----------
# -- legs --
# ----------


def test_one_to_many_single_hippo_legs_lstm_shaping(
    one_to_many_single_hippo_legs_lstm,
    random_16_input,
    one_to_many_single_hippo_legs_lstm_key,
):
    print("Testing One To Many HiPPO-LSTM (legs)")
    key1, key2, = (
        one_to_many_single_hippo_legs_lstm_key[0],
        one_to_many_single_hippo_legs_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = one_to_many_single_hippo_legs_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# -- legt --
# ----------


def test_one_to_many_single_hippo_legt_lstm_shaping(
    one_to_many_single_hippo_legt_lstm,
    random_16_input,
    one_to_many_single_hippo_legt_lstm_key,
):
    print("Testing One To Many HiPPO-LSTM (legt)")
    key1, key2, = (
        one_to_many_single_hippo_legt_lstm_key[0],
        one_to_many_single_hippo_legt_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = one_to_many_single_hippo_legt_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# -- lmu --
# ----------


def test_one_to_many_single_hippo_lmu_lstm_shaping(
    one_to_many_single_hippo_lmu_lstm,
    random_16_input,
    one_to_many_single_hippo_lmu_lstm_key,
):
    print("Testing One To Many HiPPO-LSTM (lmu)")
    key1, key2, = (
        one_to_many_single_hippo_lmu_lstm_key[0],
        one_to_many_single_hippo_lmu_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = one_to_many_single_hippo_lmu_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# -- lagt --
# ----------


def test_one_to_many_single_hippo_lagt_lstm_shaping(
    one_to_many_single_hippo_lagt_lstm,
    random_16_input,
    one_to_many_single_hippo_lagt_lstm_key,
):
    print("Testing One To Many HiPPO-LSTM (lagt)")
    key1, key2, = (
        one_to_many_single_hippo_lagt_lstm_key[0],
        one_to_many_single_hippo_lagt_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = one_to_many_single_hippo_lagt_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# --- fru --
# ----------


def test_one_to_many_single_hippo_fru_lstm_shaping(
    one_to_many_single_hippo_fru_lstm,
    random_16_input,
    one_to_many_single_hippo_fru_lstm_key,
):
    print("Testing One To Many HiPPO-LSTM (FRU)")
    key1, key2, = (
        one_to_many_single_hippo_fru_lstm_key[0],
        one_to_many_single_hippo_fru_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = one_to_many_single_hippo_fru_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ------------
# --- fout ---
# ------------


def test_one_to_many_single_hippo_fout_lstm_shaping(
    one_to_many_single_hippo_fout_lstm,
    random_16_input,
    one_to_many_single_hippo_fout_lstm_key,
):
    print("Testing One To Many HiPPO-LSTM (fout)")
    key1, key2, = (
        one_to_many_single_hippo_fout_lstm_key[0],
        one_to_many_single_hippo_fout_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = one_to_many_single_hippo_fout_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ------------
# --- foud ---
# ------------


def test_one_to_many_single_hippo_foud_lstm_shaping(
    one_to_many_single_hippo_foud_lstm,
    random_16_input,
    one_to_many_single_hippo_foud_lstm_key,
):
    print("Testing One To Many HiPPO-LSTM (foud)")
    key1, key2, = (
        one_to_many_single_hippo_foud_lstm_key[0],
        one_to_many_single_hippo_foud_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = one_to_many_single_hippo_foud_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# -------------------------
# ------ many to one ------
# -------------------------

# ----------
# -- legs --
# ----------


def test_many_to_one_single_hippo_legs_lstm_shaping(
    many_to_one_single_hippo_legs_lstm,
    random_16_input,
    many_to_one_single_hippo_legs_lstm_key,
):
    print("Testing Many To One HiPPO-LSTM (legs)")
    key1, key2, = (
        many_to_one_single_hippo_legs_lstm_key[0],
        many_to_one_single_hippo_legs_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_one_single_hippo_legs_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# ----------
# -- legt --
# ----------


def test_many_to_one_single_hippo_legt_lstm_shaping(
    many_to_one_single_hippo_legt_lstm,
    random_16_input,
    many_to_one_single_hippo_legt_lstm_key,
):
    print("Testing Many To One HiPPO-LSTM (legt)")
    key1, key2, = (
        many_to_one_single_hippo_legt_lstm_key[0],
        many_to_one_single_hippo_legt_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_one_single_hippo_legt_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# ----------
# -- lmu --
# ----------


def test_many_to_one_single_hippo_lmu_lstm_shaping(
    many_to_one_single_hippo_lmu_lstm,
    random_16_input,
    many_to_one_single_hippo_lmu_lstm_key,
):
    print("Testing Many To One HiPPO-LSTM (lmu)")
    key1, key2, = (
        many_to_one_single_hippo_lmu_lstm_key[0],
        many_to_one_single_hippo_lmu_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_one_single_hippo_lmu_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# ----------
# -- lagt --
# ----------


def test_many_to_one_single_hippo_lagt_lstm_shaping(
    many_to_one_single_hippo_lagt_lstm,
    random_16_input,
    many_to_one_single_hippo_lagt_lstm_key,
):
    print("Testing Many To One HiPPO-LSTM (lagt)")
    key1, key2, = (
        many_to_one_single_hippo_lagt_lstm_key[0],
        many_to_one_single_hippo_lagt_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_one_single_hippo_lagt_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# ----------
# --- fru --
# ----------


def test_many_to_one_single_hippo_fru_lstm_shaping(
    many_to_one_single_hippo_fru_lstm,
    random_16_input,
    many_to_one_single_hippo_fru_lstm_key,
):
    print("Testing Many To One HiPPO-LSTM (fru)")
    key1, key2, = (
        many_to_one_single_hippo_fru_lstm_key[0],
        many_to_one_single_hippo_fru_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_one_single_hippo_fru_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# ------------
# --- fout ---
# ------------


def test_many_to_one_single_hippo_fout_lstm_shaping(
    many_to_one_single_hippo_fout_lstm,
    random_16_input,
    many_to_one_single_hippo_fout_lstm_key,
):
    print("Testing Many To One HiPPO-LSTM (fout)")
    key1, key2, = (
        many_to_one_single_hippo_fout_lstm_key[0],
        many_to_one_single_hippo_fout_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_one_single_hippo_fout_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# ------------
# --- foud ---
# ------------


def test_many_to_one_single_hippo_foud_lstm_shaping(
    many_to_one_single_hippo_foud_lstm,
    random_16_input,
    many_to_one_single_hippo_foud_lstm_key,
):
    print("Testing Many To One HiPPO-LSTM (foud)")
    key1, key2, = (
        many_to_one_single_hippo_foud_lstm_key[0],
        many_to_one_single_hippo_foud_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_one_single_hippo_foud_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert y.shape


# -------------------------
# ------ many to many -----
# -------------------------

# ----------
# -- legs --
# ----------


def test_many_to_many_single_hippo_legs_lstm_shaping(
    many_to_many_single_hippo_legs_lstm,
    random_16_input,
    many_to_many_single_hippo_legs_lstm_key,
):
    print("Testing Many To Many HiPPO-LSTM (legs)")
    key1, key2, = (
        many_to_many_single_hippo_legs_lstm_key[0],
        many_to_many_single_hippo_legs_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_many_single_hippo_legs_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# ----------
# -- legt --
# ----------


def test_many_to_many_single_hippo_legt_lstm_shaping(
    many_to_many_single_hippo_legt_lstm,
    random_16_input,
    many_to_many_single_hippo_legt_lstm_key,
):
    print("Testing Many To Many HiPPO-LSTM (legt)")
    key1, key2, = (
        many_to_many_single_hippo_legt_lstm_key[0],
        many_to_many_single_hippo_legt_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_many_single_hippo_legt_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# ----------
# -- lmu --
# ----------


def test_many_to_many_single_hippo_lmu_lstm_shaping(
    many_to_many_single_hippo_lmu_lstm,
    random_16_input,
    many_to_many_single_hippo_lmu_lstm_key,
):
    print("Testing Many To Many HiPPO-LSTM (lmu)")
    key1, key2, = (
        many_to_many_single_hippo_lmu_lstm_key[0],
        many_to_many_single_hippo_lmu_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_many_single_hippo_lmu_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# ----------
# -- lagt --
# ----------


def test_many_to_many_single_hippo_lagt_lstm_shaping(
    many_to_many_single_hippo_lagt_lstm,
    random_16_input,
    many_to_many_single_hippo_lagt_lstm_key,
):
    print("Testing Many To Many HiPPO-LSTM (lagt)")
    key1, key2, = (
        many_to_many_single_hippo_lagt_lstm_key[0],
        many_to_many_single_hippo_lagt_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_many_single_hippo_lagt_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# ----------
# --- fru --
# ----------


def test_many_to_many_single_hippo_fru_lstm_shaping(
    many_to_many_single_hippo_fru_lstm,
    random_16_input,
    many_to_many_single_hippo_fru_lstm_key,
):
    print("Testing Many To Many HiPPO-LSTM (fru)")
    key1, key2, = (
        many_to_many_single_hippo_fru_lstm_key[0],
        many_to_many_single_hippo_fru_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_many_single_hippo_fru_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# ------------
# --- fout ---
# ------------


def test_many_to_many_single_hippo_fout_lstm_shaping(
    many_to_many_single_hippo_fout_lstm,
    random_16_input,
    many_to_many_single_hippo_fout_lstm_key,
):
    print("Testing Many To Many HiPPO-LSTM (fout)")
    key1, key2, = (
        many_to_many_single_hippo_fout_lstm_key[0],
        many_to_many_single_hippo_fout_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_many_single_hippo_fout_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# ------------
# --- foud ---
# ------------


def test_many_to_many_single_hippo_foud_lstm_shaping(
    many_to_many_single_hippo_foud_lstm,
    random_16_input,
    many_to_many_single_hippo_foud_lstm_key,
):
    print("Testing Many To Many HiPPO-LSTM (foud)")
    key1, key2, = (
        many_to_many_single_hippo_foud_lstm_key[0],
        many_to_many_single_hippo_foud_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_many_single_hippo_foud_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert y.shape == (16, 10)


# ----------------------------------------------------------------
# -------------------- Single Cell HiPPO GRU ---------------------
# ----------------------------------------------------------------

# -------------------------
# ------ one to many ------
# -------------------------

# ----------
# -- legs --
# ----------


def test_one_to_many_single_hippo_legs_gru_shaping(
    one_to_many_single_hippo_legs_gru,
    random_16_input,
    one_to_many_single_hippo_legs_gru_key,
):
    print("Testing One To Many HiPPO-GRU (legs)")
    key1, key2, = (
        one_to_many_single_hippo_legs_gru_key[0],
        one_to_many_single_hippo_legs_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = one_to_many_single_hippo_legs_gru
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# -- legt --
# ----------


def test_one_to_many_single_hippo_legt_gru_shaping(
    one_to_many_single_hippo_legt_gru,
    random_16_input,
    one_to_many_single_hippo_legt_gru_key,
):
    print("Testing One To Many HiPPO-GRU (legt)")
    key1, key2, = (
        one_to_many_single_hippo_legt_gru_key[0],
        one_to_many_single_hippo_legt_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = one_to_many_single_hippo_legt_gru
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# -- lmu --
# ----------


def test_one_to_many_single_hippo_lmu_gru_shaping(
    one_to_many_single_hippo_lmu_gru,
    random_16_input,
    one_to_many_single_hippo_lmu_gru_key,
):
    print("Testing One To Many HiPPO-GRU (lmu)")
    key1, key2, = (
        one_to_many_single_hippo_lmu_gru_key[0],
        one_to_many_single_hippo_lmu_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = one_to_many_single_hippo_lmu_gru
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# -- lagt --
# ----------


def test_one_to_many_single_hippo_lagt_gru_shaping(
    one_to_many_single_hippo_lagt_gru,
    random_16_input,
    one_to_many_single_hippo_lagt_gru_key,
):
    print("Testing One To Many HiPPO-GRU (lagt)")
    key1, key2, = (
        one_to_many_single_hippo_lagt_gru_key[0],
        one_to_many_single_hippo_lagt_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = one_to_many_single_hippo_lagt_gru
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# --- fru --
# ----------


def test_one_to_many_single_hippo_fru_gru_shaping(
    one_to_many_single_hippo_fru_gru,
    random_16_input,
    one_to_many_single_hippo_fru_gru_key,
):
    print("Testing One To Many HiPPO-GRU (fru)")
    key1, key2, = (
        one_to_many_single_hippo_fru_gru_key[0],
        one_to_many_single_hippo_fru_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = one_to_many_single_hippo_fru_gru
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ------------
# --- fout ---
# ------------


def test_one_to_many_single_hippo_fout_gru_shaping(
    one_to_many_single_hippo_fout_gru,
    random_16_input,
    one_to_many_single_hippo_fout_gru_key,
):
    print("Testing One To Many HiPPO-GRU (fout)")
    key1, key2, = (
        one_to_many_single_hippo_fout_gru_key[0],
        one_to_many_single_hippo_fout_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = one_to_many_single_hippo_fout_gru
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ------------
# --- foud ---
# ------------


def test_one_to_many_single_hippo_foud_gru_shaping(
    one_to_many_single_hippo_foud_gru,
    random_16_input,
    one_to_many_single_hippo_foud_gru_key,
):
    print("Testing One To Many HiPPO-GRU (foud)")
    key1, key2, = (
        one_to_many_single_hippo_foud_gru_key[0],
        one_to_many_single_hippo_foud_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = one_to_many_single_hippo_foud_gru
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# -------------------------
# ------ many to one ------
# -------------------------

# ----------
# -- legs --
# ----------


def test_many_to_one_single_hippo_legs_gru_shaping(
    many_to_one_single_hippo_legs_gru,
    random_16_input,
    many_to_one_single_hippo_legs_gru_key,
):
    print("Testing Many To One HiPPO-GRU (legs)")
    key1, key2, = (
        many_to_one_single_hippo_legs_gru_key[0],
        many_to_one_single_hippo_legs_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_one_single_hippo_legs_gru
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# ----------
# -- legt --
# ----------


def test_many_to_one_single_hippo_legt_gru_shaping(
    many_to_one_single_hippo_legt_gru,
    random_16_input,
    many_to_one_single_hippo_legt_gru_key,
):
    print("Testing Many To One HiPPO-GRU (legt)")
    key1, key2, = (
        many_to_one_single_hippo_legt_gru_key[0],
        many_to_one_single_hippo_legt_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_one_single_hippo_legt_gru
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# ----------
# -- lmu --
# ----------


def test_many_to_one_single_hippo_lmu_gru_shaping(
    many_to_one_single_hippo_lmu_gru,
    random_16_input,
    many_to_one_single_hippo_lmu_gru_key,
):
    print("Testing Many To One HiPPO-GRU (lmu)")
    key1, key2, = (
        many_to_one_single_hippo_lmu_gru_key[0],
        many_to_one_single_hippo_lmu_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_one_single_hippo_lmu_gru
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# ----------
# -- lagt --
# ----------


def test_many_to_one_single_hippo_lagt_gru_shaping(
    many_to_one_single_hippo_lagt_gru,
    random_16_input,
    many_to_one_single_hippo_lagt_gru_key,
):
    print("Testing Many To One HiPPO-GRU (lagt)")
    key1, key2, = (
        many_to_one_single_hippo_lagt_gru_key[0],
        many_to_one_single_hippo_lagt_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_one_single_hippo_lagt_gru
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# ----------
# --- fru --
# ----------


def test_many_to_one_single_hippo_fru_gru_shaping(
    many_to_one_single_hippo_fru_gru,
    random_16_input,
    many_to_one_single_hippo_fru_gru_key,
):
    print("Testing Many To One HiPPO-GRU (fru)")
    key1, key2, = (
        many_to_one_single_hippo_fru_gru_key[0],
        many_to_one_single_hippo_fru_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_one_single_hippo_fru_gru
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# ------------
# --- fout ---
# ------------


def test_many_to_one_single_hippo_fout_gru_shaping(
    many_to_one_single_hippo_fout_gru,
    random_16_input,
    many_to_one_single_hippo_fout_gru_key,
):
    print("Testing Many To One HiPPO-GRU (fout)")
    key1, key2, = (
        many_to_one_single_hippo_fout_gru_key[0],
        many_to_one_single_hippo_fout_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_one_single_hippo_fout_gru
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# ------------
# --- foud ---
# ------------


def test_many_to_one_single_hippo_foud_gru_shaping(
    many_to_one_single_hippo_foud_gru,
    random_16_input,
    many_to_one_single_hippo_foud_gru_key,
):
    print("Testing Many To One HiPPO-GRU (foud)")
    key1, key2, = (
        many_to_one_single_hippo_foud_gru_key[0],
        many_to_one_single_hippo_foud_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_one_single_hippo_foud_gru
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert y.shape


# -------------------------
# ------ many to many -----
# -------------------------

# ----------
# -- legs --
# ----------


def test_many_to_many_single_hippo_legs_gru_shaping(
    many_to_many_single_hippo_legs_gru,
    random_16_input,
    many_to_many_single_hippo_legs_gru_key,
):
    print("Testing Many To Many HiPPO-GRU (legs)")
    key1, key2, = (
        many_to_many_single_hippo_legs_gru_key[0],
        many_to_many_single_hippo_legs_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_many_single_hippo_legs_gru
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# ----------
# -- legt --
# ----------


def test_many_to_many_single_hippo_legt_gru_shaping(
    many_to_many_single_hippo_legt_gru,
    random_16_input,
    many_to_many_single_hippo_legt_gru_key,
):
    print("Testing Many To Many HiPPO-GRU (legt)")
    key1, key2, = (
        many_to_many_single_hippo_legt_gru_key[0],
        many_to_many_single_hippo_legt_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_many_single_hippo_legt_gru
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# ----------
# -- lmu --
# ----------


def test_many_to_many_single_hippo_lmu_gru_shaping(
    many_to_many_single_hippo_lmu_gru,
    random_16_input,
    many_to_many_single_hippo_lmu_gru_key,
):
    print("Testing Many To Many HiPPO-GRU (lmu)")
    key1, key2, = (
        many_to_many_single_hippo_lmu_gru_key[0],
        many_to_many_single_hippo_lmu_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_many_single_hippo_lmu_gru
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# ----------
# -- lagt --
# ----------


def test_many_to_many_single_hippo_lagt_gru_shaping(
    many_to_many_single_hippo_lagt_gru,
    random_16_input,
    many_to_many_single_hippo_lagt_gru_key,
):
    print("Testing Many To Many HiPPO-GRU (lagt)")
    key1, key2, = (
        many_to_many_single_hippo_lagt_gru_key[0],
        many_to_many_single_hippo_lagt_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_many_single_hippo_lagt_gru
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# ----------
# --- fru --
# ----------


def test_many_to_many_single_hippo_fru_gru_shaping(
    many_to_many_single_hippo_fru_gru,
    random_16_input,
    many_to_many_single_hippo_fru_gru_key,
):
    print("Testing Many To Many HiPPO-GRU (fru)")
    key1, key2, = (
        many_to_many_single_hippo_fru_gru_key[0],
        many_to_many_single_hippo_fru_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_many_single_hippo_fru_gru
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# ------------
# --- fout ---
# ------------


def test_many_to_many_single_hippo_fout_gru_shaping(
    many_to_many_single_hippo_fout_gru,
    random_16_input,
    many_to_many_single_hippo_fout_gru_key,
):
    print("Testing Many To Many HiPPO-GRU (fout)")
    key1, key2, = (
        many_to_many_single_hippo_fout_gru_key[0],
        many_to_many_single_hippo_fout_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_many_single_hippo_fout_gru
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# ------------
# --- foud ---
# ------------


def test_many_to_many_single_hippo_foud_gru_shaping(
    many_to_many_single_hippo_foud_gru,
    random_16_input,
    many_to_many_single_hippo_foud_gru_key,
):
    print("Testing Many To Many HiPPO-GRU (foud)")
    key1, key2, = (
        many_to_many_single_hippo_foud_gru_key[0],
        many_to_many_single_hippo_foud_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_many_single_hippo_foud_gru
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert y.shape == (16, 10)


# ----------------------------------------------------------------
# ------------ Single Cell Bidirectional HiPPO LSTM --------------
# ----------------------------------------------------------------

# ----------
# -- legs --
# ----------


def test_single_cell_hippo_legs_bilstm_shaping(
    single_cell_hippo_legs_bilstm,
    random_16_input,
    single_cell_hippo_legs_bilstm_key,
):
    print("Testing Single Cell Bidirectional HiPPO-LSTM (legs)")
    key1, key2, = (
        single_cell_hippo_legs_bilstm_key[0],
        single_cell_hippo_legs_bilstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = single_cell_hippo_legs_bilstm
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# -- legt --
# ----------


def test_single_cell_hippo_legt_bilstm_shaping(
    single_cell_hippo_legt_bilstm,
    random_16_input,
    single_cell_hippo_legt_bilstm_key,
):
    print("Testing Single Cell Bidirectional HiPPO-LSTM (legt)")
    key1, key2, = (
        single_cell_hippo_legt_bilstm_key[0],
        single_cell_hippo_legt_bilstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = single_cell_hippo_legt_bilstm
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# -- lmu --
# ----------


def test_single_cell_hippo_lmu_bilstm_shaping(
    single_cell_hippo_lmu_bilstm,
    random_16_input,
    single_cell_hippo_lmu_bilstm_key,
):
    print("Testing Single Cell Bidirectional HiPPO-LSTM (lmu)")
    key1, key2, = (
        single_cell_hippo_lmu_bilstm_key[0],
        single_cell_hippo_lmu_bilstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = single_cell_hippo_lmu_bilstm
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# -- lagt --
# ----------


def test_single_cell_hippo_lagt_bilstm_shaping(
    single_cell_hippo_lagt_bilstm,
    random_16_input,
    single_cell_hippo_lagt_bilstm_key,
):
    print("Testing Single Cell Bidirectional HiPPO-LSTM (lagt)")
    key1, key2, = (
        single_cell_hippo_lagt_bilstm_key[0],
        single_cell_hippo_lagt_bilstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = single_cell_hippo_lagt_bilstm
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# --- fru --
# ----------


def test_single_cell_hippo_fru_bilstm_shaping(
    single_cell_hippo_fru_bilstm,
    random_16_input,
    single_cell_hippo_fru_bilstm_key,
):
    print("Testing Single Cell Bidirectional HiPPO-LSTM (fru)")
    key1, key2, = (
        single_cell_hippo_fru_bilstm_key[0],
        single_cell_hippo_fru_bilstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = single_cell_hippo_fru_bilstm
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ------------
# --- fout ---
# ------------


def test_single_cell_hippo_fout_bilstm_shaping(
    single_cell_hippo_fout_bilstm,
    random_16_input,
    single_cell_hippo_fout_bilstm_key,
):
    print("Testing Single Cell Bidirectional HiPPO-LSTM (fout)")
    key1, key2, = (
        single_cell_hippo_fout_bilstm_key[0],
        single_cell_hippo_fout_bilstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = single_cell_hippo_fout_bilstm
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ------------
# --- foud ---
# ------------


def test_single_cell_hippo_foud_bilstm_shaping(
    single_cell_hippo_foud_bilstm,
    random_16_input,
    single_cell_hippo_foud_bilstm_key,
):
    print("Testing Single Cell Bidirectional HiPPO-LSTM (foud)")
    key1, key2, = (
        single_cell_hippo_foud_bilstm_key[0],
        single_cell_hippo_foud_bilstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = single_cell_hippo_foud_bilstm
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------------------------------------------------------------
# ------------ Single Cell Bidirectional HiPPO GRU ---------------
# ----------------------------------------------------------------

# ----------
# -- legs --
# ----------


def test_single_cell_hippo_legs_bigru_shaping(
    single_cell_hippo_legs_bigru,
    random_16_input,
    single_cell_hippo_legs_bigru_key,
):
    print("Testing Single Cell Bidirectional HiPPO-GRU (legs)")
    key1, key2, = (
        single_cell_hippo_legs_bigru_key[0],
        single_cell_hippo_legs_bigru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = single_cell_hippo_legs_bigru
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# -- legt --
# ----------


def test_single_cell_hippo_legt_bigru_shaping(
    single_cell_hippo_legt_bigru,
    random_16_input,
    single_cell_hippo_legt_bigru_key,
):
    print("Testing Single Cell Bidirectional HiPPO-GRU (legt)")
    key1, key2, = (
        single_cell_hippo_legt_bigru_key[0],
        single_cell_hippo_legt_bigru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = single_cell_hippo_legt_bigru
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# -- lmu --
# ----------


def test_single_cell_hippo_lmu_bigru_shaping(
    single_cell_hippo_lmu_bigru,
    random_16_input,
    single_cell_hippo_lmu_bigru_key,
):
    print("Testing Single Cell Bidirectional HiPPO-GRU (lmu)")
    key1, key2, = (
        single_cell_hippo_lmu_bigru_key[0],
        single_cell_hippo_lmu_bigru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = single_cell_hippo_lmu_bigru
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# -- lagt --
# ----------


def test_single_cell_hippo_lagt_bigru_key_shaping(
    single_cell_hippo_lagt_bigru,
    random_16_input,
    single_cell_hippo_lagt_bigru_key,
):
    print("Testing Single Cell Bidirectional HiPPO-GRU (lagt)")
    key1, key2, = (
        single_cell_hippo_lagt_bigru_key[0],
        single_cell_hippo_lagt_bigru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = single_cell_hippo_lagt_bigru
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# --- fru --
# ----------


def test_single_cell_hippo_fru_bigru_shaping(
    single_cell_hippo_fru_bigru,
    random_16_input,
    single_cell_hippo_fru_bigru_key,
):
    print("Testing Single Cell Bidirectional HiPPO-GRU (fru)")
    key1, key2, = (
        single_cell_hippo_fru_bigru_key[0],
        single_cell_hippo_fru_bigru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = single_cell_hippo_fru_bigru
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ------------
# --- fout ---
# ------------


def test_single_cell_hippo_fout_bigru_shaping(
    single_cell_hippo_fout_bigru,
    random_16_input,
    single_cell_hippo_fout_bigru_key,
):
    print("Testing Single Cell Bidirectional HiPPO-GRU (fout)")
    key1, key2, = (
        single_cell_hippo_fout_bigru_key[0],
        single_cell_hippo_fout_bigru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = single_cell_hippo_fout_bigru
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ------------
# --- foud ---
# ------------


def test_single_cell_hippo_foud_bigru_shaping(
    single_cell_hippo_foud_bigru,
    random_16_input,
    single_cell_hippo_foud_bigru_key,
):
    print("Testing Single Cell Bidirectional HiPPO-GRU (foud)")
    key1, key2, = (
        single_cell_hippo_foud_bigru_key[0],
        single_cell_hippo_foud_bigru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = single_cell_hippo_foud_bigru
    init_carry = model.initialize_carry(
        rng=key1,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------------------------------------------------------------
# ------------------------ Deep HiPPO LSTM -----------------------
# ----------------------------------------------------------------

# -------------------------
# ------ one to many ------
# -------------------------

# ----------
# -- legs --
# ----------


def test_one_to_many_deep_hippo_legs_lstm_shaping(
    one_to_many_deep_hippo_legs_lstm,
    random_16_input,
    one_to_many_deep_hippo_legs_lstm_key,
):
    print("Testing One To Many Deep HiPPO-LSTM (legs)")
    key1, key2, = (
        one_to_many_deep_hippo_legs_lstm_key[0],
        one_to_many_deep_hippo_legs_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = one_to_many_deep_hippo_legs_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        layers=one_to_many_deep_hippo_legs_lstm.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# -- legt --
# ----------


def test_one_to_many_deep_hippo_legt_lstm_shaping(
    one_to_many_deep_hippo_legt_lstm,
    random_16_input,
    one_to_many_deep_hippo_legt_lstm_key,
):
    print("Testing One To Many Deep HiPPO-LSTM (legt)")
    key1, key2, = (
        one_to_many_deep_hippo_legt_lstm_key[0],
        one_to_many_deep_hippo_legt_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = one_to_many_deep_hippo_legt_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        layers=one_to_many_deep_hippo_legt_lstm.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# -- lmu --
# ----------


def test_one_to_many_deep_hippo_lmu_lstm_shaping(
    one_to_many_deep_hippo_lmu_lstm,
    random_16_input,
    one_to_many_deep_hippo_lmu_lstm_key,
):
    print("Testing One To Many Deep HiPPO-LSTM (lmu)")
    key1, key2, = (
        one_to_many_deep_hippo_lmu_lstm_key[0],
        one_to_many_deep_hippo_lmu_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = one_to_many_deep_hippo_lmu_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        layers=one_to_many_deep_hippo_lmu_lstm.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# -- lagt --
# ----------


def test_one_to_many_deep_hippo_lagt_lstm_shaping(
    one_to_many_deep_hippo_lagt_lstm,
    random_16_input,
    one_to_many_deep_hippo_lagt_lstm_key,
):
    print("Testing One To Many Deep HiPPO-LSTM (lagt)")
    key1, key2, = (
        one_to_many_deep_hippo_lagt_lstm_key[0],
        one_to_many_deep_hippo_lagt_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = one_to_many_deep_hippo_lagt_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        layers=one_to_many_deep_hippo_lagt_lstm.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# --- fru --
# ----------


def test_one_to_many_deep_hippo_fru_lstm_shaping(
    one_to_many_deep_hippo_fru_lstm,
    random_16_input,
    one_to_many_deep_hippo_fru_lstm_key,
):
    print("Testing One To Many Deep HiPPO-LSTM (fru)")
    key1, key2, = (
        one_to_many_deep_hippo_fru_lstm_key[0],
        one_to_many_deep_hippo_fru_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = one_to_many_deep_hippo_fru_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        layers=one_to_many_deep_hippo_fru_lstm.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ------------
# --- fout ---
# ------------


def test_one_to_many_deep_hippo_fout_lstm_shaping(
    one_to_many_deep_hippo_fout_lstm,
    random_16_input,
    one_to_many_deep_hippo_fout_lstm_key,
):
    print("Testing One To Many Deep HiPPO-LSTM (fout)")
    key1, key2, = (
        one_to_many_deep_hippo_fout_lstm_key[0],
        one_to_many_deep_hippo_fout_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = one_to_many_deep_hippo_fout_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        layers=one_to_many_deep_hippo_fout_lstm.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ------------
# --- foud ---
# ------------


def test_one_to_many_deep_hippo_foud_lstm_shaping(
    one_to_many_deep_hippo_foud_lstm,
    random_16_input,
    one_to_many_deep_hippo_foud_lstm_key,
):
    print("Testing One To Many Deep HiPPO-LSTM (foud)")
    key1, key2, = (
        one_to_many_deep_hippo_foud_lstm_key[0],
        one_to_many_deep_hippo_foud_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = one_to_many_deep_hippo_foud_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        layers=one_to_many_deep_hippo_foud_lstm.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# -------------------------
# ------ many to one ------
# -------------------------

# ----------
# -- legs --
# ----------


def test_many_to_one_deep_hippo_legs_lstm_shaping(
    many_to_one_deep_hippo_legs_lstm,
    random_16_input,
    many_to_one_deep_hippo_legs_lstm_key,
):
    print("Testing Many To One Deep HiPPO-LSTM (legs)")
    key1, key2, = (
        many_to_one_deep_hippo_legs_lstm_key[0],
        many_to_one_deep_hippo_legs_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_one_deep_hippo_legs_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        layers=many_to_one_deep_hippo_legs_lstm.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# ----------
# -- legt --
# ----------


def test_many_to_one_deep_hippo_legt_lstm_shaping(
    many_to_one_deep_hippo_legt_lstm,
    random_16_input,
    many_to_one_deep_hippo_legt_lstm_key,
):
    print("Testing Many To One Deep HiPPO-LSTM (legt)")
    key1, key2, = (
        many_to_one_deep_hippo_legt_lstm_key[0],
        many_to_one_deep_hippo_legt_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_one_deep_hippo_legt_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        layers=many_to_one_deep_hippo_legt_lstm.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# ----------
# -- lmu --
# ----------


def test_many_to_one_deep_hippo_lmu_lstm_shaping(
    many_to_one_deep_hippo_lmu_lstm,
    random_16_input,
    many_to_one_deep_hippo_lmu_lstm_key,
):
    print("Testing Many To One Deep HiPPO-LSTM (lmu)")
    key1, key2, = (
        many_to_one_deep_hippo_lmu_lstm_key[0],
        many_to_one_deep_hippo_lmu_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_one_deep_hippo_lmu_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        layers=many_to_one_deep_hippo_lmu_lstm.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# ----------
# -- lagt --
# ----------


def test_many_to_one_deep_hippo_lagt_lstm_shaping(
    many_to_one_deep_hippo_lagt_lstm,
    random_16_input,
    many_to_one_deep_hippo_lagt_lstm_key,
):
    print("Testing Many To One Deep HiPPO-LSTM (lagt)")
    key1, key2, = (
        many_to_one_deep_hippo_lagt_lstm_key[0],
        many_to_one_deep_hippo_lagt_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_one_deep_hippo_lagt_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        layers=many_to_one_deep_hippo_lagt_lstm.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# ----------
# --- fru --
# ----------


def test_many_to_one_deep_hippo_fru_lstm_shaping(
    many_to_one_deep_hippo_fru_lstm,
    random_16_input,
    many_to_one_deep_hippo_fru_lstm_key,
):
    print("Testing Many To One Deep HiPPO-LSTM (fru)")
    key1, key2, = (
        many_to_one_deep_hippo_fru_lstm_key[0],
        many_to_one_deep_hippo_fru_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_one_deep_hippo_fru_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        layers=many_to_one_deep_hippo_fru_lstm.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# ------------
# --- fout ---
# ------------


def test_many_to_one_deep_hippo_fout_lstm_shaping(
    many_to_one_deep_hippo_fout_lstm,
    random_16_input,
    many_to_one_deep_hippo_fout_lstm_key,
):
    print("Testing Many To One Deep HiPPO-LSTM (fout)")
    key1, key2, = (
        many_to_one_deep_hippo_fout_lstm_key[0],
        many_to_one_deep_hippo_fout_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_one_deep_hippo_fout_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        layers=many_to_one_deep_hippo_fout_lstm.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# ------------
# --- foud ---
# ------------


def test_many_to_one_deep_hippo_foud_lstm_shaping(
    many_to_one_deep_hippo_foud_lstm,
    random_16_input,
    many_to_one_deep_hippo_foud_lstm_key,
):
    print("Testing Many To One Deep HiPPO-LSTM (foud)")
    key1, key2, = (
        many_to_one_deep_hippo_foud_lstm_key[0],
        many_to_one_deep_hippo_foud_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_one_deep_hippo_foud_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        layers=many_to_one_deep_hippo_foud_lstm.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# -------------------------
# ------ many to many -----
# -------------------------

# ----------
# -- legs --
# ----------


def test_many_to_many_deep_hippo_legs_lstm_shaping(
    many_to_many_deep_hippo_legs_lstm,
    random_16_input,
    many_to_many_deep_hippo_legs_lstm_key,
):
    print("Testing Many To Many Deep HiPPO-LSTM (legs)")
    key1, key2, = (
        many_to_many_deep_hippo_legs_lstm_key[0],
        many_to_many_deep_hippo_legs_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_many_deep_hippo_legs_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        layers=many_to_many_deep_hippo_legs_lstm.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# -- legt --
# ----------


def test_many_to_many_deep_hippo_legt_lstm_shaping(
    many_to_many_deep_hippo_legt_lstm,
    random_16_input,
    many_to_many_deep_hippo_legt_lstm_key,
):
    print("Testing Many To Many Deep HiPPO-LSTM (legt)")
    key1, key2, = (
        many_to_many_deep_hippo_legt_lstm_key[0],
        many_to_many_deep_hippo_legt_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_many_deep_hippo_legt_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        layers=many_to_many_deep_hippo_legt_lstm.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# -- lmu --
# ----------


def test_many_to_many_deep_hippo_lmu_lstm_shaping(
    many_to_many_deep_hippo_lmu_lstm,
    random_16_input,
    many_to_many_deep_hippo_lmu_lstm_key,
):
    print("Testing Many To Many Deep HiPPO-LSTM (lmu)")
    key1, key2, = (
        many_to_many_deep_hippo_lmu_lstm_key[0],
        many_to_many_deep_hippo_lmu_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_many_deep_hippo_lmu_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        layers=many_to_many_deep_hippo_lmu_lstm.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# -- lagt --
# ----------


def test_many_to_many_deep_hippo_lagt_lstm_shaping(
    many_to_many_deep_hippo_lagt_lstm,
    random_16_input,
    many_to_many_deep_hippo_lagt_lstm_key,
):
    print("Testing Many To Many Deep HiPPO-LSTM (lagt)")
    key1, key2, = (
        many_to_many_deep_hippo_lagt_lstm_key[0],
        many_to_many_deep_hippo_lagt_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_many_deep_hippo_lagt_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        layers=many_to_many_deep_hippo_lagt_lstm.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# --- fru --
# ----------


def test_many_to_many_deep_hippo_fru_lstm_shaping(
    many_to_many_deep_hippo_fru_lstm,
    random_16_input,
    many_to_many_deep_hippo_fru_lstm_key,
):
    print("Testing Many To Many Deep HiPPO-LSTM (fru)")
    key1, key2, = (
        many_to_many_deep_hippo_fru_lstm_key[0],
        many_to_many_deep_hippo_fru_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_many_deep_hippo_fru_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        layers=many_to_many_deep_hippo_fru_lstm.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ------------
# --- fout ---
# ------------


def test_many_to_many_deep_hippo_fout_lstm_shaping(
    many_to_many_deep_hippo_fout_lstm,
    random_16_input,
    many_to_many_deep_hippo_fout_lstm_key,
):
    print("Testing Many To Many Deep HiPPO-LSTM (fout)")
    key1, key2, = (
        many_to_many_deep_hippo_fout_lstm_key[0],
        many_to_many_deep_hippo_fout_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_many_deep_hippo_fout_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        layers=many_to_many_deep_hippo_fout_lstm.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ------------
# --- foud ---
# ------------


def test_many_to_many_deep_hippo_foud_lstm_shaping(
    many_to_many_deep_hippo_foud_lstm,
    random_16_input,
    many_to_many_deep_hippo_foud_lstm_key,
):
    print("Testing Many To Many Deep HiPPO-LSTM (foud)")
    key1, key2, = (
        many_to_many_deep_hippo_foud_lstm_key[0],
        many_to_many_deep_hippo_foud_lstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_many_deep_hippo_foud_lstm
    init_carry = model.initialize_carry(
        rng=key1,
        layers=many_to_many_deep_hippo_foud_lstm.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------------------------------------------------------------
# ------------------------ Deep HiPPO GRU ------------------------
# ----------------------------------------------------------------

# -------------------------
# ------ one to many ------
# -------------------------

# ----------
# -- legs --
# ----------


def test_one_to_many_deep_hippo_legs_gru_shaping(
    one_to_many_deep_hippo_legs_gru,
    random_16_input,
    one_to_many_deep_hippo_legs_gru_key,
):
    print("Testing One To Many Deep HiPPO-GRU (legs)")
    key1, key2, = (
        one_to_many_deep_hippo_legs_gru_key[0],
        one_to_many_deep_hippo_legs_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = one_to_many_deep_hippo_legs_gru
    init_carry = model.initialize_carry(
        rng=key1,
        layers=one_to_many_deep_hippo_legs_gru.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# -- legt --
# ----------


def test_one_to_many_deep_hippo_legt_gru_shaping(
    one_to_many_deep_hippo_legt_gru,
    random_16_input,
    one_to_many_deep_hippo_legt_gru_key,
):
    print("Testing One To Many Deep HiPPO-GRU (legt)")
    key1, key2, = (
        one_to_many_deep_hippo_legt_gru_key[0],
        one_to_many_deep_hippo_legt_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = one_to_many_deep_hippo_legt_gru
    init_carry = model.initialize_carry(
        rng=key1,
        layers=one_to_many_deep_hippo_legt_gru.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# -- lmu --
# ----------


def test_one_to_many_deep_hippo_lmu_gru_shaping(
    one_to_many_deep_hippo_lmu_gru,
    random_16_input,
    one_to_many_deep_hippo_lmu_gru_key,
):
    print("Testing One To Many Deep HiPPO-GRU (lmu)")
    key1, key2, = (
        one_to_many_deep_hippo_lmu_gru_key[0],
        one_to_many_deep_hippo_lmu_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = one_to_many_deep_hippo_lmu_gru
    init_carry = model.initialize_carry(
        rng=key1,
        layers=one_to_many_deep_hippo_lmu_gru.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# -- lagt --
# ----------


def test_one_to_many_deep_hippo_lagt_gru_shaping(
    one_to_many_deep_hippo_lagt_gru,
    random_16_input,
    one_to_many_deep_hippo_lagt_gru_key,
):
    print("Testing One To Many Deep HiPPO-GRU (lagt)")
    key1, key2, = (
        one_to_many_deep_hippo_lagt_gru_key[0],
        one_to_many_deep_hippo_lagt_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = one_to_many_deep_hippo_lagt_gru
    init_carry = model.initialize_carry(
        rng=key1,
        layers=one_to_many_deep_hippo_lagt_gru.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# --- fru --
# ----------


def test_one_to_many_deep_hippo_fru_gru_shaping(
    one_to_many_deep_hippo_fru_gru,
    random_16_input,
    one_to_many_deep_hippo_fru_gru_key,
):
    print("Testing One To Many Deep HiPPO-GRU (fru)")
    key1, key2, = (
        one_to_many_deep_hippo_fru_gru_key[0],
        one_to_many_deep_hippo_fru_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = one_to_many_deep_hippo_fru_gru
    init_carry = model.initialize_carry(
        rng=key1,
        layers=one_to_many_deep_hippo_fru_gru.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ------------
# --- fout ---
# ------------


def test_one_to_many_deep_hippo_fout_gru_shaping(
    one_to_many_deep_hippo_fout_gru,
    random_16_input,
    one_to_many_deep_hippo_fout_gru_key,
):
    print("Testing One To Many Deep HiPPO-GRU (fout)")
    key1, key2, = (
        one_to_many_deep_hippo_fout_gru_key[0],
        one_to_many_deep_hippo_fout_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = one_to_many_deep_hippo_fout_gru
    init_carry = model.initialize_carry(
        rng=key1,
        layers=one_to_many_deep_hippo_fout_gru.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ------------
# --- foud ---
# ------------


def test_one_to_many_deep_hippo_foud_gru_shaping(
    one_to_many_deep_hippo_foud_gru,
    random_16_input,
    one_to_many_deep_hippo_foud_gru_key,
):
    print("Testing One To Many Deep HiPPO-GRU (foud)")
    key1, key2, = (
        one_to_many_deep_hippo_foud_gru_key[0],
        one_to_many_deep_hippo_foud_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = one_to_many_deep_hippo_foud_gru
    init_carry = model.initialize_carry(
        rng=key1,
        layers=one_to_many_deep_hippo_foud_gru.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# -------------------------
# ------ many to one ------
# -------------------------

# ----------
# -- legs --
# ----------


def test_many_to_one_deep_hippo_legs_gru_shaping(
    many_to_one_deep_hippo_legs_gru,
    random_16_input,
    many_to_one_deep_hippo_legs_gru_key,
):
    print("Testing Many To One Deep HiPPO-GRU (legs)")
    key1, key2, = (
        many_to_one_deep_hippo_legs_gru_key[0],
        many_to_one_deep_hippo_legs_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_one_deep_hippo_legs_gru
    init_carry = model.initialize_carry(
        rng=key1,
        layers=many_to_one_deep_hippo_legs_gru.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# ----------
# -- legt --
# ----------


def test_many_to_one_deep_hippo_legt_gru_shaping(
    many_to_one_deep_hippo_legt_gru,
    random_16_input,
    many_to_one_deep_hippo_legt_gru_key,
):
    print("Testing Many To One Deep HiPPO-GRU (legt)")
    key1, key2, = (
        many_to_one_deep_hippo_legt_gru_key[0],
        many_to_one_deep_hippo_legt_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_one_deep_hippo_legt_gru
    init_carry = model.initialize_carry(
        rng=key1,
        layers=many_to_one_deep_hippo_legt_gru.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# ----------
# -- lmu --
# ----------


def test_many_to_one_deep_hippo_lmu_gru_shaping(
    many_to_one_deep_hippo_lmu_gru,
    random_16_input,
    many_to_one_deep_hippo_lmu_gru_key,
):
    print("Testing Many To One Deep HiPPO-GRU (lmu)")
    key1, key2, = (
        many_to_one_deep_hippo_lmu_gru_key[0],
        many_to_one_deep_hippo_lmu_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_one_deep_hippo_lmu_gru
    init_carry = model.initialize_carry(
        rng=key1,
        layers=many_to_one_deep_hippo_lmu_gru.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# ----------
# -- lagt --
# ----------


def test_many_to_one_deep_hippo_lagt_gru_shaping(
    many_to_one_deep_hippo_lagt_gru,
    random_16_input,
    many_to_one_deep_hippo_lagt_gru_key,
):
    print("Testing Many To One Deep HiPPO-GRU (lagt)")
    key1, key2, = (
        many_to_one_deep_hippo_lagt_gru_key[0],
        many_to_one_deep_hippo_lagt_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_one_deep_hippo_lagt_gru
    init_carry = model.initialize_carry(
        rng=key1,
        layers=many_to_one_deep_hippo_lagt_gru.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# ----------
# --- fru --
# ----------


def test_many_to_one_deep_hippo_fru_gru_shaping(
    many_to_one_deep_hippo_fru_gru,
    random_16_input,
    many_to_one_deep_hippo_fru_gru_key,
):
    print("Testing Many To One Deep HiPPO-GRU (fru)")
    key1, key2, = (
        many_to_one_deep_hippo_fru_gru_key[0],
        many_to_one_deep_hippo_fru_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_one_deep_hippo_fru_gru
    init_carry = model.initialize_carry(
        rng=key1,
        layers=many_to_one_deep_hippo_fru_gru.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# ------------
# --- fout ---
# ------------


def test_many_to_one_deep_hippo_fout_gru_shaping(
    many_to_one_deep_hippo_fout_gru,
    random_16_input,
    many_to_one_deep_hippo_fout_gru_key,
):
    print("Testing Many To One Deep HiPPO-GRU (fout)")
    key1, key2, = (
        many_to_one_deep_hippo_fout_gru_key[0],
        many_to_one_deep_hippo_fout_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_one_deep_hippo_fout_gru
    init_carry = model.initialize_carry(
        rng=key1,
        layers=many_to_one_deep_hippo_fout_gru.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# ------------
# --- foud ---
# ------------


def test_many_to_one_deep_hippo_foud_gru_shaping(
    many_to_one_deep_hippo_foud_gru,
    random_16_input,
    many_to_one_deep_hippo_foud_gru_key,
):
    print("Testing Many To One Deep HiPPO-GRU (foud)")
    key1, key2, = (
        many_to_one_deep_hippo_foud_gru_key[0],
        many_to_one_deep_hippo_foud_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_one_deep_hippo_foud_gru
    init_carry = model.initialize_carry(
        rng=key1,
        layers=many_to_one_deep_hippo_foud_gru.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y shape: {y.shape}")
    assert (y.shape) == (batch_size, 10)


# -------------------------
# ------ many to many -----
# -------------------------

# ----------
# -- legs --
# ----------


def test_many_to_many_deep_hippo_legs_gru_shaping(
    many_to_many_deep_hippo_legs_gru,
    random_16_input,
    many_to_many_deep_hippo_legs_gru_key,
):
    print("Testing Many To Many Deep HiPPO-GRU (legs)")
    key1, key2, = (
        many_to_many_deep_hippo_legs_gru_key[0],
        many_to_many_deep_hippo_legs_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_many_deep_hippo_legs_gru
    init_carry = model.initialize_carry(
        rng=key1,
        layers=many_to_many_deep_hippo_legs_gru.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# -- legt --
# ----------


def test_many_to_many_deep_hippo_legt_gru_shaping(
    many_to_many_deep_hippo_legt_gru,
    random_16_input,
    many_to_many_deep_hippo_legt_gru_key,
):
    print("Testing Many To Many Deep HiPPO-GRU (legt)")
    key1, key2, = (
        many_to_many_deep_hippo_legt_gru_key[0],
        many_to_many_deep_hippo_legt_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_many_deep_hippo_legt_gru
    init_carry = model.initialize_carry(
        rng=key1,
        layers=many_to_many_deep_hippo_legt_gru.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# -- lmu --
# ----------


def test_many_to_many_deep_hippo_lmu_gru_shaping(
    many_to_many_deep_hippo_lmu_gru,
    random_16_input,
    many_to_many_deep_hippo_lmu_gru_key,
):
    print("Testing Many To Many Deep HiPPO-GRU (lmu)")
    key1, key2, = (
        many_to_many_deep_hippo_lmu_gru_key[0],
        many_to_many_deep_hippo_lmu_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_many_deep_hippo_lmu_gru
    init_carry = model.initialize_carry(
        rng=key1,
        layers=many_to_many_deep_hippo_lmu_gru.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# -- lagt --
# ----------


def test_many_to_many_deep_hippo_lagt_gru_shaping(
    many_to_many_deep_hippo_lagt_gru,
    random_16_input,
    many_to_many_deep_hippo_lagt_gru_key,
):
    print("Testing Many To Many Deep HiPPO-GRU (lagt)")
    key1, key2, = (
        many_to_many_deep_hippo_lagt_gru_key[0],
        many_to_many_deep_hippo_lagt_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_many_deep_hippo_lagt_gru
    init_carry = model.initialize_carry(
        rng=key1,
        layers=many_to_many_deep_hippo_lagt_gru.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# --- fru --
# ----------


def test_many_to_many_deep_hippo_fru_gru_shaping(
    many_to_many_deep_hippo_fru_gru,
    random_16_input,
    many_to_many_deep_hippo_fru_gru_key,
):
    print("Testing Many To Many Deep HiPPO-GRU (fru)")
    key1, key2, = (
        many_to_many_deep_hippo_fru_gru_key[0],
        many_to_many_deep_hippo_fru_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_many_deep_hippo_fru_gru
    init_carry = model.initialize_carry(
        rng=key1,
        layers=many_to_many_deep_hippo_fru_gru.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ------------
# --- fout ---
# ------------


def test_many_to_many_deep_hippo_fout_gru_shaping(
    many_to_many_deep_hippo_fout_gru,
    random_16_input,
    many_to_many_deep_hippo_fout_gru_key,
):
    print("Testing Many To Many Deep HiPPO-GRU (fout)")
    key1, key2, = (
        many_to_many_deep_hippo_fout_gru_key[0],
        many_to_many_deep_hippo_fout_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_many_deep_hippo_fout_gru
    init_carry = model.initialize_carry(
        rng=key1,
        layers=many_to_many_deep_hippo_fout_gru.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ------------
# --- foud ---
# ------------


def test_many_to_many_deep_hippo_foud_gru_shaping(
    many_to_many_deep_hippo_foud_gru,
    random_16_input,
    many_to_many_deep_hippo_foud_gru_key,
):
    print("Testing Many To Many Deep HiPPO-GRU (foud)")
    key1, key2, = (
        many_to_many_deep_hippo_foud_gru_key[0],
        many_to_many_deep_hippo_foud_gru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = many_to_many_deep_hippo_foud_gru
    init_carry = model.initialize_carry(
        rng=key1,
        layers=many_to_many_deep_hippo_foud_gru.layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------------------------------------------------------------
# ---------------- Deep Bidirectional HiPPO LSTM -----------------
# ----------------------------------------------------------------

# ----------
# -- legs --
# ----------


def test_deep_hippo_legs_bilstm_shaping(
    deep_hippo_legs_bilstm,
    random_16_input,
    deep_hippo_legs_bilstm_key,
):
    print("Testing Deep Bidirectional HiPPO-LSTM (legs)")
    key1, key2, = (
        deep_hippo_legs_bilstm_key[0],
        deep_hippo_legs_bilstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = deep_hippo_legs_bilstm
    init_carry = model.initialize_carry(
        rng=key1,
        layers=deep_hippo_legs_bilstm.forward_layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# -- legt --
# ----------


def test_deep_hippo_legt_bilstm_shaping(
    deep_hippo_legt_bilstm,
    random_16_input,
    deep_hippo_legt_bilstm_key,
):
    print("Testing Deep Bidirectional HiPPO-LSTM (legt)")
    key1, key2, = (
        deep_hippo_legt_bilstm_key[0],
        deep_hippo_legt_bilstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = deep_hippo_legt_bilstm
    init_carry = model.initialize_carry(
        rng=key1,
        layers=deep_hippo_legt_bilstm.forward_layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# -- lmu --
# ----------


def test_deep_hippo_lmu_bilstm_shaping(
    deep_hippo_lmu_bilstm,
    random_16_input,
    deep_hippo_lmu_bilstm_key,
):
    print("Testing Deep Bidirectional HiPPO-LSTM (lmu)")
    key1, key2, = (
        deep_hippo_lmu_bilstm_key[0],
        deep_hippo_lmu_bilstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = deep_hippo_lmu_bilstm
    init_carry = model.initialize_carry(
        rng=key1,
        layers=deep_hippo_lmu_bilstm.forward_layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# -- lagt --
# ----------


def test_deep_hippo_lagt_bilstm_shaping(
    deep_hippo_lagt_bilstm,
    random_16_input,
    deep_hippo_lagt_bilstm_key,
):
    print("Testing Deep Bidirectional HiPPO-LSTM (lagt)")
    key1, key2, = (
        deep_hippo_lagt_bilstm_key[0],
        deep_hippo_lagt_bilstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = deep_hippo_lagt_bilstm
    init_carry = model.initialize_carry(
        rng=key1,
        layers=deep_hippo_lagt_bilstm.forward_layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# --- fru --
# ----------


def test_deep_hippo_fru_bilstm_shaping(
    deep_hippo_fru_bilstm,
    random_16_input,
    deep_hippo_fru_bilstm_key,
):
    print("Testing Deep Bidirectional HiPPO-LSTM (fru)")
    key1, key2, = (
        deep_hippo_fru_bilstm_key[0],
        deep_hippo_fru_bilstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = deep_hippo_fru_bilstm
    init_carry = model.initialize_carry(
        rng=key1,
        layers=deep_hippo_fru_bilstm.forward_layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ------------
# --- fout ---
# ------------


def test_deep_hippo_fout_bilstm_shaping(
    deep_hippo_fout_bilstm,
    random_16_input,
    deep_hippo_fout_bilstm_key,
):
    print("Testing Deep Bidirectional HiPPO-LSTM (fout)")
    key1, key2, = (
        deep_hippo_fout_bilstm_key[0],
        deep_hippo_fout_bilstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = deep_hippo_fout_bilstm
    init_carry = model.initialize_carry(
        rng=key1,
        layers=deep_hippo_fout_bilstm.forward_layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ------------
# --- foud ---
# ------------


def test_deep_hippo_foud_bilstm_shaping(
    deep_hippo_foud_bilstm,
    random_16_input,
    deep_hippo_foud_bilstm_key,
):
    print("Testing Deep Bidirectional HiPPO-LSTM (foud)")
    key1, key2, = (
        deep_hippo_foud_bilstm_key[0],
        deep_hippo_foud_bilstm_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = deep_hippo_foud_bilstm
    init_carry = model.initialize_carry(
        rng=key1,
        layers=deep_hippo_foud_bilstm.forward_layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------------------------------------------------------------
# ----------------- Deep Bidirectional HiPPO GRU -----------------
# ----------------------------------------------------------------

# ----------
# -- legs --
# ----------


def test_deep_hippo_legs_bigru_shaping(
    deep_hippo_legs_bigru,
    random_16_input,
    deep_hippo_legs_bigru_key,
):
    print("Testing Deep Bidirectional HiPPO-GRU (legs)")
    key1, key2, = (
        deep_hippo_legs_bigru_key[0],
        deep_hippo_legs_bigru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = deep_hippo_legs_bigru
    init_carry = model.initialize_carry(
        rng=key1,
        layers=deep_hippo_legs_bigru.forward_layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# -- legt --
# ----------


def test_deep_hippo_legt_bigru_shaping(
    deep_hippo_legt_bigru,
    random_16_input,
    deep_hippo_legt_bigru_key,
):
    print("Testing Deep Bidirectional HiPPO-GRU (legt)")
    key1, key2, = (
        deep_hippo_legt_bigru_key[0],
        deep_hippo_legt_bigru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = deep_hippo_legt_bigru
    init_carry = model.initialize_carry(
        rng=key1,
        layers=deep_hippo_legt_bigru.forward_layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# -- lmu --
# ----------


def test_deep_hippo_lmu_bigru_shaping(
    deep_hippo_lmu_bigru,
    random_16_input,
    deep_hippo_lmu_bigru_key,
):
    print("Testing Deep Bidirectional HiPPO-GRU (lmu)")
    key1, key2, = (
        deep_hippo_lmu_bigru_key[0],
        deep_hippo_lmu_bigru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = deep_hippo_lmu_bigru
    init_carry = model.initialize_carry(
        rng=key1,
        layers=deep_hippo_lmu_bigru.forward_layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# -- lagt --
# ----------


def test_deep_hippo_lagt_bigru_shaping(
    deep_hippo_lagt_bigru,
    random_16_input,
    deep_hippo_lagt_bigru_key,
):
    print("Testing Deep Bidirectional HiPPO-GRU (lagt)")
    key1, key2, = (
        deep_hippo_lagt_bigru_key[0],
        deep_hippo_lagt_bigru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = deep_hippo_lagt_bigru
    init_carry = model.initialize_carry(
        rng=key1,
        layers=deep_hippo_lagt_bigru.forward_layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ----------
# --- fru --
# ----------


def test_deep_hippo_fru_bigru_shaping(
    deep_hippo_fru_bigru,
    random_16_input,
    deep_hippo_fru_bigru_key,
):
    print("Testing Deep Bidirectional HiPPO-GRU (fru)")
    key1, key2, = (
        deep_hippo_fru_bigru_key[0],
        deep_hippo_fru_bigru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = deep_hippo_fru_bigru
    init_carry = model.initialize_carry(
        rng=key1,
        layers=deep_hippo_fru_bigru.forward_layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ------------
# --- fout ---
# ------------


def test_deep_hippo_fout_bigru_shaping(
    deep_hippo_fout_bigru,
    random_16_input,
    deep_hippo_fout_bigru_key,
):
    print("Testing Deep Bidirectional HiPPO-GRU (fout)")
    key1, key2, = (
        deep_hippo_fout_bigru_key[0],
        deep_hippo_fout_bigru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = deep_hippo_fout_bigru
    init_carry = model.initialize_carry(
        rng=key1,
        layers=deep_hippo_fout_bigru.forward_layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)


# ------------
# --- foud ---
# ------------


def test_deep_hippo_foud_bigru_shaping(
    deep_hippo_foud_bigru,
    random_16_input,
    deep_hippo_foud_bigru_key,
):
    print("Testing Deep Bidirectional HiPPO-GRU (foud)")
    key1, key2, = (
        deep_hippo_foud_bigru_key[0],
        deep_hippo_foud_bigru_key[1],
    )
    batch_size = 16
    hidden_size = 256
    model = deep_hippo_foud_bigru
    init_carry = model.initialize_carry(
        rng=key1,
        layers=deep_hippo_foud_bigru.forward_layers,
        batch_size=(batch_size,),
        hidden_size=hidden_size,
        init_fn=nn.initializers.zeros,
    )
    params = model.init(
        key2,
        init_carry,
        random_16_input,
    )

    y = model.apply(
        params,
        init_carry,
        random_16_input,
    )

    print(f"input shape: {random_16_input.shape}")
    print(f"y:\n{y}")
    for i in range(len(y)):
        print(f"y shape: {y[i].shape}")
        assert y[i].shape == (batch_size, 10)
