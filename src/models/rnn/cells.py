from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.activation import sigmoid, tanh
from flax.linen.recurrent import GRUCell as r_GRUCell

from src.models.hippo.hippo import HiPPO
from src.models.hippo.transition import TransMatrix


class RNNCell(nn.Module):
    """
    Description:
        W_xh = x_{t} @ W_{xh} - multiply the previous hidden state with
        W_hh = H_{t-1} @ W_{hh} + b_{h} - this a linear layer

        H_{t} = f_{w}(H_{t-1}, x)
        H_{t} = \phi(H_{t-1} @ W_{hh}) + (x_{t} @ W_{xh})

    Args:
        nn (_type_): _description_

    Returns:
        _type_: _description_
    """

    input_size: int
    hidden_size: int
    bias: bool = True
    param_dtype: Any = jnp.float32
    activation_fn: Callable[..., Any] = tanh

    def setup(self):
        self.dense_i = nn.Dense(
            self.hidden_size, use_bias=self.bias, param_dtype=self.param_dtype
        )
        self.dense_h = nn.Dense(
            self.hidden_size, use_bias=self.bias, param_dtype=self.param_dtype
        )

    def __call__(self, carry, input):
        ht_1, _ = carry
        # print(f"ht_1 shape: {ht_1.shape}")
        # print(f"input shape: {input.shape}")

        w_hh = self.dense_h(ht_1)
        # print(f"w_hh shape: {w_hh.shape}")
        w_xh = self.dense_i(input)
        # print(f"w_xh shape: {w_xh.shape}")

        h_t = self.activation_fn(
            (w_hh + w_xh)
        )  # H_{t} = tanh(H_{t-1} @ W_{hh}) + (x_{t} @ W_{xh})

        return (h_t, h_t), h_t

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


class LSTMCell(nn.Module):
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

    input_size: int
    hidden_size: int
    bias: bool = True
    param_dtype: Any = jnp.float32
    gate_fn: Callable[..., Any] = sigmoid
    activation_fn: Callable[..., Any] = tanh

    def setup(self):
        self.dense_i_ti = nn.Dense(
            features=self.hidden_size, use_bias=self.bias, param_dtype=self.param_dtype
        )
        self.dense_i_th = nn.Dense(
            features=self.hidden_size, use_bias=self.bias, param_dtype=self.param_dtype
        )

        self.dense_o_ti = nn.Dense(
            features=self.hidden_size, use_bias=self.bias, param_dtype=self.param_dtype
        )
        self.dense_o_th = nn.Dense(
            features=self.hidden_size, use_bias=self.bias, param_dtype=self.param_dtype
        )

        self.dense_f_ti = nn.Dense(
            features=self.hidden_size, use_bias=self.bias, param_dtype=self.param_dtype
        )
        self.dense_f_th = nn.Dense(
            features=self.hidden_size, use_bias=self.bias, param_dtype=self.param_dtype
        )

        self.dense_g_ti = nn.Dense(
            features=self.hidden_size, use_bias=self.bias, param_dtype=self.param_dtype
        )
        self.dense_g_th = nn.Dense(
            features=self.hidden_size, use_bias=self.bias, param_dtype=self.param_dtype
        )

    def __call__(self, carry, input):
        h_t, c_t = carry

        i_ti = self.dense_i_ti(input)
        i_th = self.dense_i_th(h_t)
        i_t = self.gate_fn(i_ti + i_th)

        o_ti = self.dense_o_ti(input)
        o_th = self.dense_o_th(h_t)
        o_t = self.gate_fn(o_ti + o_th)

        f_ti = self.dense_f_ti(input)
        f_th = self.dense_f_th(h_t)
        f_t = self.gate_fn(f_ti + f_th)

        g_ti = self.dense_g_ti(input)
        g_th = self.dense_g_th(h_t)
        g_t = self.activation_fn(g_ti + g_th)

        c_t = (f_t * c_t) + (i_t * g_t)
        h_t = o_t * self.activation_fn(c_t)

        return (h_t, c_t), h_t

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


class GRUCell(nn.Module):
    """
    Description:
        z_t = sigmoid((W_{iz} @ x_{t} + b_{iz}) + (W_{hz} @ h_{t-1} + b_{hz}))
        r_t = sigmoid((W_{ir} @ x_{t} + b_{ir}) + (W_{hr} @ h_{t-1} + b_{hr}))
        n_t = tanh(((W_{in} @ x_{t} + b_{in}) + r_t) * (W_{hn} @ h_{t-1} + b_{hn}))
        h_t = (z_t * h_{t-1}) + ((1 - z_t) * n_i)

    Args:
        hidden_size (int): hidden state size
        carry (jnp.ndarray): hidden state from previous time step
        input (jnp.ndarray): # input vector

    Returns:
        A tuple with the new carry and the output.
    """

    input_size: int
    hidden_size: int
    bias: bool = True
    param_dtype: Any = jnp.float32
    gate_fn: Callable[..., Any] = sigmoid
    activation_fn: Callable[..., Any] = tanh

    def setup(self):
        self.dense_z_ti = nn.Dense(
            features=self.hidden_size, use_bias=self.bias, param_dtype=self.param_dtype
        )
        self.dense_z_th = nn.Dense(
            features=self.hidden_size, use_bias=self.bias, param_dtype=self.param_dtype
        )

        self.dense_r_ti = nn.Dense(
            features=self.hidden_size, use_bias=self.bias, param_dtype=self.param_dtype
        )
        self.dense_r_th = nn.Dense(
            features=self.hidden_size, use_bias=self.bias, param_dtype=self.param_dtype
        )

        self.dense_n_ti = nn.Dense(
            features=self.hidden_size, use_bias=self.bias, param_dtype=self.param_dtype
        )
        self.dense_n_th = nn.Dense(
            features=self.hidden_size, use_bias=self.bias, param_dtype=self.param_dtype
        )

    def __call__(self, carry, input):
        h_t_1, h_t_1 = carry

        z_ti = self.dense_z_ti(input)
        z_th = self.dense_z_th(h_t_1)
        z_t = self.gate_fn(z_ti + z_th)

        r_ti = self.dense_r_ti(input)
        r_th = self.dense_r_th(h_t_1)
        r_t = self.gate_fn(r_ti + r_th)

        n_ti = self.dense_n_ti(input)
        n_th = self.dense_n_th(h_t_1)
        n_t = self.activation_fn(n_ti + (r_t * n_th))

        h_t = ((1 - z_t) * n_t) + (z_t * h_t_1)

        return (h_t, h_t), h_t

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


class HiPPOCell(nn.Module):
    """
    Description:
        z_t = sigmoid((W_{iz} @ x_{t} + b_{iz}) + (W_{hz} @ h_{t-1} + b_{hz}))
        r_t = sigmoid((W_{ir} @ x_{t} + b_{ir}) + (W_{hr} @ h_{t-1} + b_{hr}))
        n_t = tanh(((W_{in} @ x_{t} + b_{in}) + r_t) * (W_{hn} @ h_{t-1} + b_{hn}))
        h_t = (z_t * h_{t-1}) + ((1 - z_t) * g_i)

    Args:
        hidden_size (int): hidden state size
        carry (jnp.ndarray): hidden state from previous time step
        input (jnp.ndarray): # input vector

    Returns:
        A tuple with the new carry and the output.
    """

    input_size: int
    hidden_size: int
    step_size: float
    bias: bool = True
    param_dtype: Any = jnp.float32
    gate_fn: Callable[..., Any] = sigmoid
    activation_fn: Callable[..., Any] = tanh
    measure: str = "legs"
    lambda_n: float = 1.0
    alpha: float = 0.0
    beta: float = 1.0
    s_t: str = "lsi"
    GBT_alpha: float = 0.5
    rnn_cell: Callable[..., Any] = GRUCell
    dtype: Any = jnp.float32

    def setup(self):
        L = self.input_size
        self.hippo  = HiPPO(
            max_length=L,
            step_size=self.step_size,
            N=self.hidden_size,
            lambda_n=self.lambda_n,
            alpha=self.alpha,
            beta=self.beta
            GBT_alpha=self.GBT_alpha,
            measure=self.measure,
            s_t=self.s_t,
            dtype = self.dtype,
        ) 

        self.rnn = self.rnn_cell(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            bias=self.bias,
            param_dtype=self.param_dtype,
            gate_fn=self.gate_fn,
            activation_fn=self.activation_fn,
        )

        self.dense_f_th = nn.Dense(
            features=self.hidden_size, use_bias=self.bias, param_dtype=self.param_dtype
        )

    @nn.compact
    def __call__(self, carry, input):
        _, c_t_1 = carry

        carry, _ = self.rnn(carry, input)
        h_t, _ = carry

        f_t = self.dense_f_th(h_t)
        c_t = self.hippo(f=f_t, init_state=c_t_1)

        return (h_t, c_t), h_t

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


# this version takes in flax's built-in recurrent cells
class HIPPOCell(nn.Module):
    """
    Description:
        z_t = sigmoid((W_{iz} @ x_{t} + b_{iz}) + (W_{hz} @ h_{t-1} + b_{hz}))
        r_t = sigmoid((W_{ir} @ x_{t} + b_{ir}) + (W_{hr} @ h_{t-1} + b_{hr}))
        n_t = tanh(((W_{in} @ x_{t} + b_{in}) + r_t) * (W_{hn} @ h_{t-1} + b_{hn}))
        h_t = (z_t * h_{t-1}) + ((1 - z_t) * g_i)

    Args:
        hidden_size (int): hidden state size
        carry (jnp.ndarray): hidden state from previous time step
        input (jnp.ndarray): # input vector

    Returns:
        A tuple with the new carry and the output.
    """
    
    input_size: int
    hidden_size: int
    step_size: float
    bias: bool = True
    param_dtype: Any = jnp.float32
    gate_fn: Callable[..., Any] = sigmoid
    activation_fn: Callable[..., Any] = tanh
    measure: str = "legs"
    lambda_n: float = 1.0
    alpha: float = 0.0
    beta: float = 1.0
    s_t: str = "lsi"
    GBT_alpha: float = 0.5
    rnn_cell: Callable[..., Any] = r_GRUCell
    dtype: Any = jnp.float32

    def setup(self):
        L = self.input_size
        self.hippo  = HiPPO(
            max_length=L,
            step_size=self.step_size,
            N=self.hidden_size,
            lambda_n=self.lambda_n,
            alpha=self.alpha,
            beta=self.beta
            GBT_alpha=self.GBT_alpha,
            measure=self.measure,
            s_t=self.s_t,
            dtype = self.dtype,
        ) 

        self.rnn = self.rnn_cell()

        self.dense_f_th = nn.Dense(
            features=self.hidden_size, use_bias=self.bias, param_dtype=self.param_dtype
        )

    def __call__(self, carry, input):
        _, c_t_1 = carry

        carry = self.rnn(carry, input)
        h_t, _ = carry

        f_t = self.dense_f_th(h_t)
        c_t = self.hippo(f=f_t, init_state=c_t_1)

        return (h_t, c_t)

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
