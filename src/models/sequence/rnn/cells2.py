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

pp = pprint.PrettyPrinter(indent=4)

seed = 1701
key = jax.random.PRNGKey(seed)

num_copies = 2
rng, subkey = jax.random.split(key, num=num_copies)


def add_batch(nest, batch_size: Optional[int]):
    """Adds a batch dimension at axis 0 to the leaves of a nested structure."""
    broadcast = lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape)

    return jax.tree_map(broadcast, nest)


class RNNCell(RNNCellBase):
    hidden_size: int

    # def setup(self):
    # self.dense_h = Partial(
    #     nn.Dense,
    #     features=self.hidden_size,
    #     use_bias=True,
    #     kernel_init=nn.initializers.orthogonal(),
    #     bias_init=nn.initializers.zeros,
    #     dtype=None,
    #     param_dtype=jnp.float32,
    # )

    # self.dense_o = Partial(
    #     nn.Dense,
    #     features=self.hidden_size,
    #     use_bias=False,
    #     kernel_init=nn.initializers.orthogonal(),
    #     bias_init=nn.initializers.zeros,
    #     dtype=None,
    #     param_dtype=jnp.float32,
    # )

    # @partial(
    #     nn.transforms.scan, variable_broadcast="params", split_rngs={"params": False}
    # )
    @nn.compact
    def __call__(self, carry, input):
        """
        Description:
            W_xh = x_{t} @ W_{xh} - multiply the previous hidden state with
            W_hh = H_{t-1} @ W_{hh} + b_{h} - this a linear layer

            H_{t} = f_{w}(H_{t-1}, x)
            H_{t} = tanh(H_{t-1} @ W_{hh}) + (x_{t} @ W_{xh})

        Args:
            hidden_size (int): hidden state size
            carry (jnp.ndarray): hidden state from previous time step
            input (jnp.ndarray): # input vector

        Returns:
            A tuple with the new carry and the output.
        """
        ht_1, _ = carry

        print(f"inside the rnn, input:\n{input.shape}")

        h_t = self.rnn_update(ht_1, input)

        return (h_t, h_t), h_t

    def rnn_update(self, ht_1, input):
        print(f"inside the rnn update, input:\n{input.shape}")
        print(f"inside the rnn update, ht_1:\n{ht_1.shape}")

        # print(f"self.dense_h(name='dense rnn_wxh layer')(ht_1):\n{self.dense_h(name='dense rnn_wxh layer')(ht_1)}")
        # W_hh = self.dense_h(ht_1)

        W_hh = nn.Dense(self.hidden_size)(ht_1)
        print(f"W_hh:\n{W_hh.shape}")
        # id_print(W_hh, what="BLAH BLAH BLAH", tap_with_device=True)
        # print(f"W_hh:\n{W_hh}")
        # print(f"input.shape:\n{input.shape}")
        W_xh = nn.Dense(self.hidden_size)(input)
        print(f"W_xh:\n{W_xh.shape}")
        # W_xh = self.dense_o(name="dense rnn_wxh layer")(input)
        print(f"W_hh shape:\n{W_hh.shape}")
        print(f"W_xh shape:\n{W_xh.shape}")
        h_t = nn.relu(W_hh + W_xh)  # H_{t} = tanh(H_{t-1} @ W_{hh}) + (x_{t} @ W_{xh})
        print(f"inside the rnn update in an rnn, h_t:\n{h_t.shape}")

        return h_t

    @staticmethod
    def initialize_carry(rng, batch_size, hidden_size, init_fn=nn.initializers.zeros):
        """Initialize the RNN cell carry.
        Args:
        rng: random number generator passed to the init_fn.
        batch_dims: a tuple providing the shape of the batch dimensions.
        hidden_size: the size or number of features of the memory.
        init_fn: initializer function for the carry.
        Returns:
        An initialized carry for the given RNN cell.
        """
        key1, key2 = jax.random.split(rng)
        mem_shape = batch_size + (hidden_size,)

        return init_fn(key1, mem_shape), init_fn(key2, mem_shape)


class LSTMCell(RNNCellBase):
    hidden_size: int

    @partial(
        nn.transforms.scan, variable_broadcast="params", split_rngs={"params": False}
    )
    @nn.compact
    def __call__(self, carry, input):
        """
        Description:
            i_{t} = sigmoid((W_{ii} @ x_{t} + b_{ii}) + (W_{hi} @ h_{t-1} + b_{hi}))
            f_{t} = sigmoid((W_{if} @ x_{t} + b_{if}) + (W_{hf} @ h_{t-1} + b_{hf}))
            g_{t} = tanh((W_{ig} @ x_{t} + b_{ig}) + (W_{hg} @ h_{t-1} + b_{hg}))
            o_{t} = sigmoid((W_{io} @ x_{t} + b_{io}) + (W_{ho} @ h_{t-1} + b_{ho}))
            c_{t} = f_{t} * c_{t-1} + i_{t} * g_{t}
            h_{t} = o_{t} * tanh(c_{t})

        Args:
            hidden_size (int): hidden state size
            carry (jnp.ndarray): hidden state from previous time step
            input (jnp.ndarray): # input vector

        Returns:
            A tuple with the new carry and the output.
        """
        print(f"inside the LSTMCell, input:\n{input.shape}")
        print(f"inside the LSTMCell, input type:\n{type(input)}")

        print(f"inside the LSTMCell, carry:\n{carry.shape}")
        print(f"inside the LSTMCell, carry type:\n{type(carry)}")
        ht_1, ct_1 = carry
        print(f"carry split:\n{ht_1.shape}\n{ct_1.shape}")

        c_t, h_t = self.rnn_update(input, ht_1, ct_1)
        print(f"inside the LSTMCell, c_t:\n{c_t.shape}")
        print(f"inside the LSTMCell, h_t:\n{h_t.shape}")

        return (h_t, c_t), h_t

    # @partial(
    #     nn.transforms.scan, variable_broadcast="params", split_rngs={"params": False}
    # )
    def rnn_update(self, input, ht_1, ct_1):
        print(f"inside the LSTMCell rnn_update, input:\n{input.shape}")
        print(f"inside the LSTMCell rnn_update, ht_1:\n{ht_1.shape}")
        print(f"inside the LSTMCell rnn_update, ct_1:\n{ct_1.shape}")
        # i_ta = partial(
        #     nn.Dense,
        #     features=ht_1.shape()[0],
        #     use_bias=False,
        #     kernel_init=self.recurrent_kernel_init,
        #     bias_init=self.bias_init,
        # )

        i_ta = nn.Dense(features=ht_1.shape()[0])(input)
        i_tb = nn.Dense(features=self.hidden_size)(ht_1)
        i_t = nn.sigmoid(i_ta + i_tb)  # input gate
        print(f"inside the LSTMCell, input gate output:\n{i_t.shape}")

        o_ta = nn.Dense(self.hidden_size)(input)
        o_tb = nn.Dense(self.hidden_size)(ht_1)
        o_t = nn.sigmoid(o_ta + o_tb)  # output gate
        print(f"inside the LSTMCell, output gate output:\n{o_t.shape}")

        f_ia = nn.Dense(self.hidden_size)(
            input
        )  # b^{f}_{i} + \sum\limits_{j} U^{f}_{i, j} x^{t}_{j}
        f_ib = nn.Dense(self.hidden_size)(
            ht_1
        )  # \sum\limits_{j} W^{f}_{i, j} h^{(t-1)}_{j}
        f_i = nn.sigmoid(f_ia + f_ib)  # forget gate
        print(f"inside the LSTMCell, forget gate output:\n{f_i.shape}")

        g_ia = nn.Dense(self.hidden_size)(
            input
        )  # b^{g}_{i} + \sum\limits_{j} U^{g}_{i, j} x^{t}_{j}
        g_ib = nn.Dense(self.hidden_size)(
            ht_1
        )  # \sum\limits_{j} W^{g}_{i, j} h^{(t-1)}_{j}
        g_i = nn.tanh(g_ia + g_ib)  # (external) input gate
        print(f"inside the LSTMCell, (external) input gate output:\n{g_i.shape}")

        c_t = (f_i * ct_1) + (i_t * g_i)  # internal cell state update
        print(f"inside the LSTMCell, cell state output:\n{c_t.shape}")

        h_t = o_t * nn.tanh(c_t)  # hidden state update
        print(f"inside the LSTMCell, hidden state output:\n{h_t.shape}")

        return h_t, c_t

    @staticmethod
    def initialize_carry(rng, batch_size, hidden_size, init_fn=nn.initializers.zeros):
        """Initialize the RNN cell carry.
        Args:
        rng: random number generator passed to the init_fn.
        batch_dims: a tuple providing the shape of the batch dimensions.
        hidden_size: the size or number of features of the memory.
        init_fn: initializer function for the carry.
        Returns:
        An initialized carry for the given RNN cell.
        """
        key1, key2 = jax.random.split(rng)
        mem_shape = batch_size + (hidden_size,)

        return init_fn(key1, mem_shape), init_fn(key2, mem_shape)


class GRUCell(RNNCellBase):
    hidden_size: int

    @partial(
        nn.transforms.scan, variable_broadcast="params", split_rngs={"params": False}
    )
    @nn.compact
    def __call__(self, carry, input):
        """
        Description:
            z_t = sigmoid((W_{iz} @ x_{t} + b_{iz}) + (W_{hz} @ h_{t-1} + b_{hz}))
            r_t = sigmoid((W_{ir} @ x_{t} + b_{ir}) + (W_{hr} @ h_{t-1} + b_{hr}))
            g_t = tanh(((W_{ig} @ x_{t} + b_{ig}) + r_t) * (W_{hg} @ h_{t-1} + b_{hg}))
            h_t = (z_t * h_{t-1}) + ((1 - z_t) * g_i)

        Args:
            hidden_size (int): hidden state size
            carry (jnp.ndarray): hidden state from previous time step
            input (jnp.ndarray): # input vector

        Returns:
            A tuple with the new carry and the output.
        """
        ht_1 = carry

        h_t = self.rnn_update(input, ht_1)

        return (h_t, h_t), h_t

    # @partial(
    #     nn.transforms.scan, variable_broadcast="params", split_rngs={"params": False}
    # )
    def rnn_update(self, input, ht_1):

        z_ta = nn.Dense(self.hidden_size)(input)
        z_tb = nn.Dense(self.hidden_size)(ht_1)
        z_t = nn.sigmoid(z_ta + z_tb)  # reset gate

        r_ta = nn.Dense(self.hidden_size)(input)
        r_tb = nn.Dense(self.hidden_size)(ht_1)
        r_t = nn.sigmoid(r_ta + r_tb)  # update gate

        g_ta = nn.Dense(self.hidden_size)(input)
        g_tb = nn.Dense(self.hidden_size)(ht_1)
        g_t = nn.tanh((g_ta + r_t) * g_tb)  # (external) input gate

        h_t = ((1 - z_t) * ht_1) + (z_t * g_t)  # internal cell state update

        return h_t

    @staticmethod
    def initialize_carry(rng, batch_size, hidden_size, init_fn=nn.initializers.zeros):
        """Initialize the RNN cell carry.
        Args:
        rng: random number generator passed to the init_fn.
        batch_dims: a tuple providing the shape of the batch dimensions.
        hidden_size: the size or number of features of the memory.
        init_fn: initializer function for the carry.
        Returns:
        An initialized carry for the given RNN cell.
        """
        key1, key2 = jax.random.split(rng)
        mem_shape = batch_size + (hidden_size,)

        return init_fn(key1, mem_shape), init_fn(key2, mem_shape)


class HiPPOCell(nn.Module):
    """
    Description:
        RNN update function
        τ(h, x) = (1 - g(h, x)) ◦ h + g(h, x) ◦ tanh(Lτ (h, x))
        g(h, x) = σ(Lg(h,x))

    Args:
        hidden_size (int): hidden state size
        output_size (int): output size
        hippo (HiPPO): hippo model object
        cell (RNNCellBase): choice of RNN cell object
            - RNNCell
            - LSTMCell
            - GRUCell
    """

    hidden_size: int
    output_size: int
    hippo: HiPPO
    model: RNNCellBase

    # def setup(self):
    #     self.dense_y = Partial(
    #         nn.Dense,
    #         features=self.output_size,
    #         use_bias=True,
    #         kernel_init=nn.initializers.orthogonal(),
    #         bias_init=nn.initializers.zeros,
    #         dtype=None,
    #         param_dtype=jnp.float32,
    #     )
    #     self.cell = self.model(self.hidden_size)

    # @partial(
    #     nn.transforms.scan, variable_broadcast="params", split_rngs={"params": False}
    # )
    @nn.compact
    def __call__(self, carry, input):
        """
        Description:
            RNN update function
            τ(h, x) = (1 - g(h, x)) ◦ h + g(h, x) ◦ tanh(Lτ (h, x))
            g(h, x) = σ(Lg(h,x))

        Args:
            carry (jnp.ndarray): hidden state from previous time step
            input (jnp.ndarray): # input vector

        Returns:
            A tuple with the new carry and the output.
        """

        print(f"inside hippo cell, input:\n{input}")
        h_t, c_t = carry
        print(f"inside hippo cell, h_t:\n{h_t}\nc_t:\n{c_t}")
        print(f"inside hippo cell, the cell:\n{self.model}")
        _, h_t = self.model(self.hidden_size)(carry, input)
        print(f"inside hippo cell, h_t:\n{h_t.shape}")

        # y_t = nn.Dense(self.output_size)(h_t)  # f_t in the paper
        # print(f"inside hippo cell, y_t: \n{y_t}")

        # c_t = self.hippo(y_t, init_state=None, kernel=False)
        # print(f"inside hippo cell, c_t: \n{c_t}")

        return self.rnn_update(input, h_t)

    # @partial(
    #     nn.transforms.scan, variable_broadcast="params", split_rngs={"params": False}
    # )
    def rnn_update(self, input, h_t):

        # y_t = self.dense_y(name="dense hippo input layer")(h_t)
        print(f"inside hippo cell in the update, h_t:\n{h_t.shape}")
        print(f"inside hippo cell in the update, input:\n{input.shape}")
        y_t = nn.Dense(1)(h_t)  # f_t in the paper
        print(f"inside hippo cell, before reshape y_t: \n{y_t.shape}")
        y_t = jnp.swapaxes(y_t, 1, 0)
        y_t = jnp.swapaxes(y_t, 2, 1)
        print(f"inside hippo cell, y_t: \n{y_t.shape}")

        c_t = self.hippo(y_t, init_state=h_t, kernel=False)
        print(f"inside hippo cell, c_t: \n{c_t}")
        print(f"inside hippo cell, c_t: \n{c_t.shape}")

        return (h_t, c_t), h_t

    @staticmethod
    def initialize_carry(rng, batch_size, hidden_size, init_fn=nn.initializers.zeros):
        """Initialize the RNN cell carry.
        Args:
        rng: random number generator passed to the init_fn.
        batch_dims: a tuple providing the shape of the batch dimensions.
        hidden_size: the size or number of features of the memory.
        init_fn: initializer function for the carry.
        Returns:
        An initialized carry for the given RNN cell.
        """
        key1, key2 = jax.random.split(rng)
        mem_shape = batch_size + (hidden_size,)

        return init_fn(key1, mem_shape), init_fn(key2, mem_shape)
