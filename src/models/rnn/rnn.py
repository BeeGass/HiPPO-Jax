from dataclasses import field
from typing import Any, Callable, Optional, Sequence
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap

from flax import linen as nn
from flax.linen.initializers import zeros

from src.models.rnn.cells import GRUCell, HiPPOCell, LSTMCell, RNNCell


class OneToManyRNN(nn.Module):
    output_size: int
    layer: Sequence[Any]
    layer_name: Optional[str] = None

    def setup(self):
        if not isinstance(self.layer[0], nn.Module):
            raise ValueError(
                "skip_connections requires for all layers to be "
                "`nn.Module. Layers is: {}".format(self.layers)
            )

        self.dense_out = nn.Dense(features=self.output_size)

    def __call__(self, carry, input):
        out_carry = None
        output = None
        states = []

        for t in range(input.shape[1]):
            out_carry, output = self.layer[0](carry, input)
            input = self.dense_out(output)
            states.append(input)

            carry = out_carry

        return states

    @staticmethod
    def initialize_carry(
        rng,
        batch_size: tuple,
        hidden_size: int,
        init_fn=nn.initializers.zeros,
    ):
        key1, key2 = jax.random.split(rng)
        mem_shape = batch_size + (hidden_size,)
        return init_fn(key1, mem_shape), init_fn(key2, mem_shape)


class ManyToOneRNN(nn.Module):
    output_size: int
    layer: Sequence[Any]
    layer_name: Optional[str] = None

    def setup(self):
        if not isinstance(self.layer[0], nn.Module):
            raise ValueError(
                "skip_connections requires for all layers to be "
                "`nn.Module. Layers is: {}".format(self.layers)
            )

        self.dense_out = nn.Dense(features=self.output_size)

    def __call__(self, carry, input):
        out_carry = None
        output = None

        for t in range(input.shape[1]):
            out_carry, output = self.layer[0](carry, input[:, t, :])
            carry = out_carry

        return self.dense_out(output)

    @staticmethod
    def initialize_carry(
        rng,
        batch_size: tuple,
        hidden_size: int,
        init_fn=nn.initializers.zeros,
    ):
        key1, key2 = jax.random.split(rng)
        mem_shape = batch_size + (hidden_size,)
        return init_fn(key1, mem_shape), init_fn(key2, mem_shape)


class ManyToManyRNN(nn.Module):
    output_size: int
    layer: Sequence[Any]
    layer_name: Optional[str] = None

    def setup(self):
        if not isinstance(self.layer[0], nn.Module):
            raise ValueError(
                "skip_connections requires for all layers to be "
                "`nn.Module. Layers is: {}".format(self.layers)
            )

        self.dense_out = nn.Dense(features=self.output_size)

    def __call__(self, carry, input):
        out_carry = None
        output = None
        states = []

        for t in range(input.shape[1]):
            out_carry, output = self.layer[0](carry, input[:, t, :])
            output = self.dense_out(output)
            states.append(output)

            carry = out_carry

        return states

    @staticmethod
    def initialize_carry(
        rng,
        batch_size: tuple,
        hidden_size: int,
        init_fn=nn.initializers.zeros,
    ):
        key1, key2 = jax.random.split(rng)
        mem_shape = batch_size + (hidden_size,)
        return init_fn(key1, mem_shape), init_fn(key2, mem_shape)


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
            for idx, layer in enumerate(self.layers):
                if isinstance(layer, nn.Module):
                    if idx == 0:
                        out_carry, output = layer(carry, input[:, t, :])
                        h_t, c_t = out_carry

                    else:
                        h_t_1, c_t_1 = out_carry
                        out_carry, output = layer(carry, h_t_1)
                        h_t, c_t = out_carry
                        if self.skip_connections:
                            h_t = jnp.concatenate([h_t, h_t_1], axis=1)
                            c_t = jnp.concatenate([c_t, c_t_1], axis=1)
                            out_carry = tuple([h_t, c_t])
                else:
                    out_carry, output = layer(out_carry)

                h_t_list.append(h_t)
                c_t_list.append(c_t)
                states.append(output)

            carry = out_carry

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
        rng,
        layers: Sequence[Any],
        batch_size: tuple,
        hidden_size: int,
        init_fn=nn.initializers.zeros,
    ):
        key1, key2 = jax.random.split(rng)
        # mem_shape = batch_size + (input_size, hidden_size)
        mem_shape = batch_size + (hidden_size,)
        return init_fn(key1, mem_shape), init_fn(key2, mem_shape)


class BidirectionalRNN(nn.Module):
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
            for idx, layer in enumerate(self.layers):
                if isinstance(layer, nn.Module):
                    if idx == 0:
                        out_carry, output = layer(carry, input[:, t, :])
                        h_t, c_t = out_carry

                    else:
                        h_t_1, c_t_1 = out_carry
                        out_carry, output = layer(carry, h_t_1)
                        h_t, c_t = out_carry
                        if self.skip_connections:
                            h_t = jnp.concatenate([h_t, h_t_1], axis=1)
                            c_t = jnp.concatenate([c_t, c_t_1], axis=1)
                            out_carry = tuple([h_t, c_t])
                else:
                    out_carry, output = layer(out_carry)

                h_t_list.append(h_t)
                c_t_list.append(c_t)
                states.append(output)

            carry = out_carry

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
        rng,
        layers: Sequence[Any],
        batch_size: tuple,
        hidden_size: int,
        init_fn=nn.initializers.zeros,
    ):
        key1, key2 = jax.random.split(rng)
        # mem_shape = batch_size + (input_size, hidden_size)
        mem_shape = batch_size + (hidden_size,)
        return init_fn(key1, mem_shape), init_fn(key2, mem_shape)
