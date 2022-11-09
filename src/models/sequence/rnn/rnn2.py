import jax
import jax.ops
import jax.numpy as jnp
from jax.experimental.host_callback import id_print
from jax.tree_util import Partial

import flax
from flax import linen as nn
from flax.linen.recurrent import RNNCellBase

import optax

import numpy as np  # convention: original numpy

from typing import Any, Callable, Sequence, Optional, Tuple, Union
from collections import defaultdict
from functools import partial
import pprint

from src.models.hippo.hippo import HiPPO


# TODO: refer to https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/recurrent.py#L714-L762
# also refer to https://dm-haiku.readthedocs.io/en/latest/api.html?highlight=DeepRNN#deeprnn
class _DeepRNN(RNNCellBase):
    hidden_size: int
    layers: Sequence[Any]
    skip_connections: bool
    hidden_to_output_layer: bool
    layer_name: Optional[str]

    def setup(self):
        if self.skip_connections:
            for layer in self.layers:
                if not (isinstance(layer, RNNCellBase) or isinstance(layer, HiPPOCell)):
                    raise ValueError(
                        "skip_connections requires for all layers to be "
                        "`hk.RNNCore`s. Layers is: {}".format(self.layers)
                    )
                    # raise ValueError(
                    #     f"{self.layer_name} layer {layer} is not a RNNCellBase or HiPPOCell"
                    # )

    def __call__(self, carry, inputs):
        current_carry = carry
        next_states = []
        h_t_outputs = []
        c_t_outputs = []
        state_idx = 0
        # print(f"inside deep rnn, inputs:\n{inputs.shape}")
        # print(f"inside deep rnn, carry:\n{carry.shape}")
        h_t, c_t = carry  # c_t may actually be h_t in which case dont use it
        (
            h_t_copy,
            c_t_copy,
        ) = current_carry  # c_t may actually be h_t in which case dont use it
        concat = lambda *args: jnp.concatenate(args, axis=-1)
        print(f"before main loop")
        for idx, layer in enumerate(self.layers):
            print(f"inside deep rnn cell, h_t:\n{h_t.shape}\nc_t:\n{c_t.shape}")
            print(f"inside deep rnn cell, THE INPUTS:\n{inputs[idx].shape}")
            if self.skip_connections and idx > 0:
                skip_h_t = jax.tree_map(concat, h_t, h_t_copy)
                skip_c_t = jax.tree_map(concat, c_t, c_t_copy)
                current_carry = tuple([skip_h_t, skip_c_t])

            if isinstance(layer, RNNCellBase) or isinstance(layer, HiPPOCell):
                # print(f"inside deep rnn, inputs:\n{inputs}")
                # print(f"inside deep rnn, state_idx:\n{state_idx}")
                print(f"inside deep rnn, inputs[state_idx]:\n{inputs[state_idx].shape}")
                h_t, c_t, next_state = layer(
                    current_carry, inputs[state_idx]
                )  # problem line
                h_t_outputs.append(h_t)
                c_t_outputs.append(c_t)
                next_states.append(next_state)
                state_idx += 1

            else:
                print(f"current_carry before layer: {current_carry.shape}")
                print(f"layer: {layer}")
                current_carry = layer(current_carry)
                print(f"current_carry:\n {current_carry.shape}")

        print(f"third conditional")
        if self.skip_connections:
            skip_h_t_out = jax.tree_map(concat, *h_t_outputs)
            skip_c_t_out = jax.tree_map(concat, *c_t_outputs)
            next_carry = (skip_h_t_out, skip_c_t_out)
        else:
            next_carry = current_carry

        print(f"next_states before tuple:\n", next_states)
        pp.pprint(layer)
        print(f"carry before return B:\n", next_carry)

        return next_carry, next_states

    @staticmethod
    def initialize_state(
        num_layers,
        rng,
        batch_size: tuple,
        output_size: int,
        init_fn=nn.initializers.zeros,
    ):
        states = []
        for i in range(num_layers):
            print(f"Layer: {i}\n")
            states.append(
                _DeepRNN.init_state(
                    rng=rng,
                    batch_size=batch_size,
                    output_size=output_size,
                    init_fn=init_fn,
                )
            )

        return states

    @staticmethod
    def init_state(
        rng, batch_size: tuple, output_size: int, init_fn=nn.initializers.zeros
    ):
        print(f"batch_size: {(batch_size,)}")
        print(f"output_size: {(output_size,)}")
        mem_shape = (batch_size,) + (output_size,)
        print(f"state mem_shape: {mem_shape}")

        return init_fn(rng, mem_shape)

    @staticmethod
    def initialize_carry(
        rng, batch_size: tuple, hidden_size: int, init_fn=nn.initializers.zeros
    ):
        print(f"batch_size: {batch_size}")
        print(f"hidden_size: {(hidden_size,)}")
        mem_shape = batch_size + (1, hidden_size)
        print(f"carry mem_shape: {mem_shape}")

        return init_fn(rng, mem_shape), init_fn(rng, mem_shape)


class DeepRNN(_DeepRNN):
    r"""Wraps a sequence of cores and callables as a single core.
        >>> deep_rnn = hk.DeepRNN([
        ...     LSTMCell(hidden_size=4),
        ...     jax.nn.relu,
        ...     LSTMCell(hidden_size=2),
        ... ])
    The state of a :class:`DeepRNN` is a tuple with one element per
    :class:`RNNCore`. If no layers are :class:`RNNCore`\ s, the state is an empty
    tuple.
    """

    def __init__(
        self,
        hidden_size: int,
        layers: Sequence[Any],
        skip_connections: Optional[bool] = False,
        hidden_to_output_layer: Optional[bool] = False,
        name: Optional[str] = None,
    ):
        super().__init__(
            hidden_size=hidden_size,
            layers=layers,
            skip_connections=skip_connections,
            hidden_to_output_layer=hidden_to_output_layer,
            layer_name=name,
        )
