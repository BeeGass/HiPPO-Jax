from dataclasses import field
from typing import Any, Callable, Optional, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import zeros

from src.models.rnn.cells import GRUCell, HiPPOCell, LSTMCell, RNNCell


class DeepRNN(nn.Module):
    output_size: int
    layers: Sequence[Any]
    skip_connections: bool
    layer_name: Optional[str] = None

    def setup(self):
        if self.skip_connections:
            for layer in self.layers:
                if not isinstance(layer, nn.Module):
                    raise ValueError(
                        "skip_connections requires for all layers to be "
                        "`nn.Module. Layers is: {}".format(self.layers)
                    )

        self.dense_out = nn.Dense(features=self.output_size)

    def __call__(self, carry, input):
        out_carry = None
        output = None
        h_t, c_t = carry
        h_t_list = []
        c_t_list = []
        states = []

        for t in range(input.shape[1]):
            print("t: ", t)
            for idx, layer in enumerate(self.layers):
                print(f"layer({idx+1})")
                if isinstance(layer, nn.Module):
                    if idx == 0:
                        out_carry, output = layer(carry, input)
                        h_t, c_t = out_carry

                    else:
                        h_t_1, c_t_1 = out_carry
                        out_carry, output = layer(carry, h_t_1)
                        h_t, c_t = out_carry
                        if self.skip_connections:
                            h_t = jnp.concatenate([h_t, h_t_1], axis=-1)
                            c_t = jnp.concatenate([c_t, c_t_1], axis=-1)
                            out_carry = tuple([h_t, c_t])
                else:
                    out_carry, output = layer(out_carry)

                h_t_list.append(h_t)
                c_t_list.append(c_t)
                states.append(output)

        next_carry = None
        concat = lambda *args: jnp.concatenate(args, axis=-1)
        if self.skip_connections:
            h_t = jax.tree_map(concat, *h_t_list)
            c_t = jax.tree_map(concat, *c_t_list)
            next_carry = tuple([h_t, c_t])
        else:
            next_carry = out_carry

        return next_carry, self.dense_out(output)

    @staticmethod
    def initialize_carry(
        rng, batch_size: tuple, hidden_size: int, init_fn=nn.initializers.zeros
    ):
        key1, key2 = jax.random.split(rng)
        mem_shape = batch_size + (1, hidden_size)
        return init_fn(key1, mem_shape), init_fn(key2, mem_shape)


def test():
    seed = 1701
    key = jax.random.PRNGKey(seed)

    num_copies = 4
    rng, key, subkey, subsubkey = jax.random.split(key, num=num_copies)

    hidden_size = 256

    # batch size, sequence length, input size
    batch_size = 32
    seq_L = 1
    input_size = 28 * 28

    # fake data
    x = jax.random.randint(rng, (batch_size, input_size), 1, 100)
    # print(f"x:\n{x}\n")
    print(f"x shape:\n{x.shape}\n")
    x = jnp.expand_dims(x, axis=-1)
    vals = jnp.ones((batch_size, input_size, batch_size - 1)) * input_size
    # print(f"vals:\n{vals}\n")
    print(f"vals shape:\n{vals.shape}\n")
    x = jnp.concatenate([x, vals], axis=-1)

    layer_list = []
    num_of_rnns = 3
    rnn_type = "rnn"
    if rnn_type == "rnn":
        layer_list = [
            RNNCell(input_size=input_size, hidden_size=hidden_size)
            for _ in range(num_of_rnns)
        ]

    elif rnn_type == "lstm":
        layer_list = [
            LSTMCell(input_size=input_size, hidden_size=hidden_size)
            for _ in range(num_of_rnns)
        ]

    elif rnn_type == "gru":
        layer_list = [
            GRUCell(input_size=input_size, hidden_size=hidden_size)
            for _ in range(num_of_rnns)
        ]

    elif rnn_type == "hippo":
        layer_list = [
            HiPPOCell(input_size=input_size, hidden_size=hidden_size)
            for _ in range(num_of_rnns)
        ]

    else:
        raise ValueError("rnn_type must be one of: rnn, lstm, gru, hippo")

    # model
    model = DeepRNN(
        output_size=10,
        layers=layer_list,
        skip_connections=True,
    )

    # get model params
    params = model.init(
        key,
        model.initialize_carry(
            rng=subkey,
            batch_size=(batch_size,),
            hidden_size=hidden_size,
            init_fn=nn.initializers.zeros,
        ),
        input=x,
    )

    print(f"applying model:\n")
    carry, out = model.apply(
        params,
        model.initialize_carry(
            rng=subsubkey,
            batch_size=(batch_size,),
            hidden_size=hidden_size,
            init_fn=nn.initializers.zeros,
        ),
        x,
    )

    xshape = out.shape
    return x, xshape


def tester():
    for i in range(1, 100):
        testx, xdims = test()
        if i % 10 == 0:
            print(f"output array:\n{testx[i]}\n")
            print(f"output array shape:\n{xdims}\n")
        assert xdims == (32, 10)
    print("Size test: passed.")
