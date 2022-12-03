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

    def __call__(self, carry, input, teacher_forcing=True):
        out_carry = None
        output = None
        outputs = []
        for t in range(input.shape[1]):
            if teacher_forcing:
                out_carry, output = self.layer[0](carry, input[:, t, :])
                tf_output = self.dense_out(output)
                outputs.append(tf_output)

            else:
                out_carry, output = self.layer[0](carry, input[:, t, :])
                input = self.dense_out(output)
                outputs.append(input)

            carry = out_carry

        return outputs

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
        outputs = []

        for t in range(input.shape[1]):
            out_carry, output = self.layer[0](carry, input[:, t, :])
            output = self.dense_out(output)
            outputs.append(output)

            carry = out_carry

        return outputs

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


class DeepRNN_OG(nn.Module):
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


class OneToManyDeepRNN(nn.Module):
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

    def __call__(self, carry, input, teacher_forcing=True):

        out_carry = None
        output = None
        carries = []
        outputs = []

        for t in range(input.shape[1]):
            for idx, layer in enumerate(self.layers):
                if idx == 0:
                    out_carry, output = layer(carry[idx], input[:, t, :])

                else:
                    h_t_1, c_t_1 = out_carry
                    out_carry, output = layer(carry[idx], h_t_1)

                output = self.dense_out(output)

                carries.append(out_carry)
                outputs.append(output)

            carry = carries
            if not teacher_forcing:
                input = output

        return outputs

    @staticmethod
    def initialize_carry(
        rng,
        layers: Sequence[Any],
        batch_size: tuple,
        hidden_size: int,
        init_fn=nn.initializers.zeros,
    ):
        carries = []
        for _, layer in enumerate(layers):
            carries.append(
                layer.initialize_carry(rng, (batch_size,), hidden_size, init_fn)
            )

        return carries


class ManyToOneDeepRNN(nn.Module):
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
        carries = []
        outputs = []

        for t in range(input.shape[1]):
            for idx, layer in enumerate(self.layers):
                if idx == 0:
                    out_carry, output = layer(carry[idx], input[:, t, :])

                else:
                    h_t_1, c_t_1 = out_carry
                    out_carry, output = layer(carry[idx], h_t_1)

                carries.append(out_carry)
                outputs.append(output)

            carry = carries

        return self.dense_out(output)

    @staticmethod
    def initialize_carry(
        rng,
        layers: Sequence[Any],
        batch_size: tuple,
        hidden_size: int,
        init_fn=nn.initializers.zeros,
    ):
        carries = []
        for _, layer in enumerate(layers):
            carries.append(
                layer.initialize_carry(rng, (batch_size,), hidden_size, init_fn)
            )

        return carries


class ManyToManyDeepRNN(nn.Module):
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
        carries = []
        outputs = []

        for t in range(input.shape[1]):
            for idx, layer in enumerate(self.layers):
                if idx == 0:
                    out_carry, output = layer(carry[idx], input[:, t, :])

                else:
                    h_t_1, c_t_1 = out_carry
                    out_carry, output = layer(carry[idx], h_t_1)

                output = self.dense_out(output)

                carries.append(out_carry)
                outputs.append(output)

            carry = carries

        return outputs

    @staticmethod
    def initialize_carry(
        rng,
        layers: Sequence[Any],
        batch_size: tuple,
        hidden_size: int,
        init_fn=nn.initializers.zeros,
    ):
        carries = []
        for _, layer in enumerate(layers):
            carries.append(
                layer.initialize_carry(rng, (batch_size,), hidden_size, init_fn)
            )

        return carries


class BiRNN(nn.Module):
    output_size: int
    layer: Sequence[Any]
    skip_connections: bool
    layer_name: Optional[str] = None

    def setup(self):
        if not isinstance(self.layer[0], nn.Module):
            raise ValueError(
                "skip_connections requires for all layers to be "
                "`nn.Module. Layers is: {}".format(self.layers[0])
            )

        self.output_size *= 2
        self.dense_out = nn.Dense(features=self.output_size)

    def __call__(self, carry, input):

        out_carry = None
        output = None
        forward_outputs = []
        outputs = []

        for t in range(input.shape[1]):
            out_carry, output = self.layer[0](carry, input[:, t, :])
            forward_outputs.append(output)
            carry = out_carry

        for t in range(input.shape[1], 0, -1):
            out_carry, output = self.layer[0](carry, input[:, t, :])
            carry = out_carry
            output = jnp.concatenate(forward_outputs[t], output)
            output = nn.relu(output)
            output = self.dense_out(output)
            outputs.append(output)

        return outputs

    @staticmethod
    def initialize_carry(
        rng,
        batch_size: tuple,
        hidden_size: int,
        init_fn=nn.initializers.zeros,
    ):
        key1, key2 = jax.random.split(rng)
        # mem_shape = batch_size + (input_size, hidden_size)
        mem_shape = batch_size + (hidden_size,)
        return init_fn(key1, mem_shape), init_fn(key2, mem_shape)


class DeepBiRNN(nn.Module):
    output_size: int
    foward_layers: Sequence[Any]
    backward_layers: Sequence[Any]
    skip_connections: bool
    layer_name: Optional[str] = None

    def setup(self):
        self.output_size *= 2
        self.dense_out = nn.Dense(features=self.output_size)

    def __call__(self, carry, input):

        foward_carries = carry
        backward_carries = carry
        out_carry = None
        output = None
        forward_outputs = []
        outputs = []
        carries = []

        for t in range(input.shape[1]):
            for idx, layer in enumerate(self.foward_layers):
                if idx == 0:
                    out_carry, output = layer(foward_carries[idx], input[:, t, :])
                else:
                    h_t_1, c_t_1 = out_carry
                    out_carry, output = layer(foward_carries[idx], h_t_1)

                carries.append(out_carry)

            foward_carries = carries
            forward_outputs.append(output)

        carries = []
        for t in range(input.shape[1], 0, -1):
            for idx in range((len(self.backward_layers) - 1), 0, -1):
                if idx == (len(self.backward_layers) - 1):
                    out_carry, output = self.backward_layers[idx](
                        backward_carries[idx], input[:, t, :]
                    )
                else:
                    h_t_1, c_t_1 = out_carry
                    out_carry, output = self.backward_layers[idx](
                        backward_carries[idx], h_t_1
                    )

                carries.append(out_carry)

            backward_carries = carries
            output = jnp.concatenate(forward_outputs[t], output)
            output = nn.relu(output)
            output = self.dense_out(output)
            outputs.append(output)

        return outputs

    @staticmethod
    def initialize_carry(
        rng,
        layers: Sequence[Any],
        batch_size: tuple,
        hidden_size: int,
        init_fn=nn.initializers.zeros,
    ):
        carries = []
        for _, layer in enumerate(layers):
            carries.append(
                layer.initialize_carry(rng, (batch_size,), hidden_size, init_fn)
            )

        return carries
