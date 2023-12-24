import einops
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import lecun_normal, lecun_uniform
from flax.linen.activation import sigmoid, tanh
from flax.linen.recurrent import GRUCell as r_GRUCell

from jaxtyping import Array, Float
from typing import Any, Callable, List, Optional, Tuple, Union

from src.models.hippo.hippo import HiPPOLTI, HiPPOLSI
from src.models.hippo.transition import (
    LegSInitializer,
    LegTInitializer,
    LagTInitializer,
    FRUInitializer,
    FouTInitializer,
    FouDInitializer,
    ChebTInitializer,
    legs_initializer,
    legt_initializer,
    lagt_initializer,
    fru_initializer,
    fout_initializer,
    foud_initializer,
    chebt_initializer,
)


class RNNCell(nn.Module):
    """
    Description:
        Basic RNN cell implementation.

        The updates happen as:
        W_xh = x_{t} @ W_{xh} - multiplication of the input with its weight matrix
        W_hh = H_{t-1} @ W_{hh} + b_{h} - linear layer with previous hidden state
        H_{t} = f_{w}(H_{t-1}, x)
        H_{t} = \phi(H_{t-1} @ W_{hh}) + (x_{t} @ W_{xh})

    Args:
        input_size (int): Size of the input.
        hidden_size (int): Size of the hidden state.
        bias (bool): Whether to include bias in the dense layers.
        param_dtype (data-type): Data type of the parameters.
        activation_fn (Callable[..., Any]): Activation function to be used.

    Returns:
        Carry and output: Updated state and output
    """

    input_size: int
    hidden_size: int
    bias: bool = True
    gate_fn: Callable[..., Any] = None
    activation_fn: Callable[..., Any] = tanh
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.dense_i = nn.Dense(
            features=self.hidden_size,
            use_bias=self.bias,
            kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
            bias_init=nn.initializers.zeros,  # Bias init with zeros
            param_dtype=self.dtype,
            name=f"i_layer",
        )
        self.dense_h = nn.Dense(
            features=self.hidden_size,
            use_bias=self.bias,
            kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
            bias_init=nn.initializers.zeros,  # Bias init with zeros
            param_dtype=self.dtype,
            name=f"h_layer",
        )

    def __call__(self, carry, input):
        ht_1, _ = carry

        w_hh = self.dense_h(ht_1)
        w_xh = self.dense_i(input)

        h_t = self.activation_fn(
            (w_hh + w_xh)
        )  # H_{t} = tanh(H_{t-1} @ W_{hh}) + (x_{t} @ W_{xh})

        return h_t, h_t

    @staticmethod
    def initialize_carry(
        rng,
        batch_size: int,
        hidden_size: int,
        init_fn=nn.initializers.zeros,
    ):
        key1, key2 = jax.random.split(rng)
        mem_shape = (batch_size, hidden_size)
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
    gate_fn: Callable[..., Any] = sigmoid
    activation_fn: Callable[..., Any] = tanh
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.dense_i = nn.Dense(
            features=4 * self.hidden_size,
            use_bias=self.bias,
            kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
            bias_init=nn.initializers.zeros,  # Bias init with zeros
            param_dtype=self.dtype,
            name=f"i_layer",
        )
        self.dense_h = nn.Dense(
            features=4 * self.hidden_size,
            use_bias=self.bias,
            kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
            bias_init=nn.initializers.zeros,  # Bias init with zeros
            param_dtype=self.dtype,
            name=f"h_layer",
        )

    def __call__(self, carry, input):
        h_t, c_t = carry
        gates_i = self.dense_i(input)
        gates_h = self.dense_h(h_t)

        # get the gate outputs
        i_t, f_t, g_t, o_t = jnp.split(gates_i + gates_h, 4, axis=-1)
        i_t = self.gate_fn(i_t)
        f_t = self.gate_fn(f_t)
        o_t = self.gate_fn(o_t)
        g_t = self.activation_fn(g_t)

        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * self.activation_fn(c_t)

        return h_t, c_t

    @staticmethod
    def initialize_carry(
        rng,
        batch_size: int,
        hidden_size: int,
        init_fn=nn.initializers.zeros,
    ):
        key1, key2 = jax.random.split(rng)
        mem_shape = (batch_size, hidden_size)
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
    gate_fn: Callable[..., Any] = sigmoid
    activation_fn: Callable[..., Any] = tanh
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.dense_i = nn.Dense(
            features=3 * self.hidden_size,
            use_bias=self.bias,
            kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
            bias_init=nn.initializers.zeros,  # Bias init with zeros
            param_dtype=self.dtype,
            name=f"i_layer",
        )
        self.dense_h = nn.Dense(
            features=3 * self.hidden_size,
            use_bias=self.bias,
            kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
            bias_init=nn.initializers.zeros,  # Bias init with zeros
            param_dtype=self.dtype,
            name=f"h_layer",
        )

    def __call__(self, carry, input):
        h_t_1, _ = carry

        gates_i = self.dense_i(input)
        gates_h = self.dense_h(h_t_1)

        # get the gate outputs
        z_t, r_t, n_t = jnp.split(gates_i + gates_h, 3, axis=-1)
        z_t = self.gate_fn(z_t)
        r_t = self.gate_fn(r_t)
        n_t = self.activation_fn(n_t)

        h_t = ((1 - z_t) * n_t) + (z_t * h_t_1)

        return h_t, h_t

    @staticmethod
    def initialize_carry(
        rng,
        batch_size: int,
        hidden_size: int,
        init_fn=nn.initializers.zeros,
    ):
        key1, key2 = jax.random.split(rng)
        mem_shape = (batch_size, hidden_size)
        return init_fn(key1, mem_shape), init_fn(key2, mem_shape)


class LTICell(nn.Module):

    input_size: int
    hidden_size: int
    trainable_scale: float = 0.0
    tau: nn.Module
    tau_args: tuple
    self.bias: bool = True
    step_size: float = 1.0
    lambda_n: float = 1.0
    alpha: float = 0.0
    beta: float = 1.0
    GBT_alpha: float = 0.5
    measure: str = "legs"
    basis_size: float = 1.0
    unroll: bool = False
    recon: bool = False
    dtype: Any = jnp.float32

    def setup(self) -> None:
        self.tau_layer = self.tau(
            *self.tau_args
        )  # any non-linear function, i.e. GRUCell, MLP, LSTM, etc.

        self.dense_f = nn.Dense(
            features=self.input_size,
            use_bias=self.bias,
            kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
            bias_init=nn.initializers.zeros,  # Bias init with zeros
            param_dtype=self.dtype,
            name=f"f_layer",
        )

        matrices = TransMatrix(
            N=self.hidden_size,
            measure=self.measure,
            lambda_n=self.lambda_n,
            alpha=self.alpha,
            beta=self.beta,
            dtype=self.dtype,
        )

        Ad_, Bd_ = self.discretize(
            A=matrices.A,
            B=matrices.B,
            step=self.step_size,
            alpha=self.GBT_alpha,
            dtype=self.dtype,
        )
        Ad = Ad_ - jnp.eye(self.hidden_size)  # puts into form: x += Ax

        self.trainable_scale = jnp.sqrt(self.trainable_scale)
        if self.trainable_scale <= 0.0:
            self.Ad = self.param("Ad", lambda _: Ad)
            self.Bd = self.param("Bd", lambda _: Bd_)
        else:
            self.Ad = self.param("Ad", lambda _: (Ad / self.trainable_scale))
            self.Bd = self.param("Bd", lambda _: (Bd_ / self.trainable_scale))

    def __call__(self, carry, input):
        (c_t_1, h_t_1) = carry
        x = jnp.concatenate([input, c_t_1], axis=-1)
        h_t, h_f_t = self.tau_layer((c_t_1, h_t_1), x)
        f_t = self.dense_f(h_f_t)

        if self.trainable_scale <= 0.0:
            c_t = c_t_1 + (jnp.dot(c_t_1, (self.Ad).T) + ((self.Bd).T * f_t))
        else:
            c_t = c_t_1 + (
                (jnp.dot(c_t_1, (self.Ad).T) * self.trainable_scale)
                + (((self.Bd).T * f_t) * self.trainable_scale)
            )
        return c_t, h_t


class HiPPOLSICell(nn.Module):
    """

    Args:
        nn (_type_): _description_

    Returns:
        _type_: _description_
    """

    input_size: int
    hidden_size: int
    tau: nn.Module
    bias: bool = True
    gate_fn: Callable[..., Any] = sigmoid
    activation_fn: Callable[..., Any] = tanh

    max_length: int = 1024
    step_size: float = 1.0
    lambda_n: float = 1.0
    alpha: float = 0.0
    beta: float = 1.0
    GBT_alpha: float = 0.5
    measure: str = "legs"
    unroll: bool = True
    recon: bool = False
    tau: Optional[Callable[..., Any]] = None
    dtype: Any = jnp.float32

    def setup(self) -> None:
        # TODO: implement LSI cell
        pass

    def __call__(self, carry, input):
        pass


class HiPPOPlusCell(nn.Module):
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
    step: float  # step size for the GBT
    lambda_n: float = 1.0  # lambda_n for the LegT
    alpha: float = 2.0  # alpha for the GBT,
    measure: str = "legs"  # measure for type of the polynomial,
    basis: float = 1.0  # basis for the polynomial
    unroll: bool = False  # unroll the loop for the output
    recon: bool = False
    tau: Optional[Callable[..., Any]] = None  # tau for the LegT
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        L = self.input_size
        self.hippo = HiPPOLTI(
            N=self.hidden_size,
            step_size=self.step,
            lambda_n=self.lambda_n,
            alpha=0.0,
            beta=1.0,
            GBT_alpha=self.alpha,
            measure=self.measure,
            basis_size=self.basis,
            dtype=jnp.float32,
            unroll=False,
            recon=self.recon,
        )

        self.dense_f = nn.Dense(
            features=3 * self.hidden_size,
            use_bias=self.bias,
            kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
            bias_init=nn.initializers.zeros,  # Bias init with zeros
            param_dtype=self.dtype,
            name=f"f_layer",
        )

    def __call__(self, carry, input):
        h_t_1, c_t_1 = carry

        tau_input = jnp.concatenate([c_t_1, input], axis=-1)

        h_tau, c_tau = self.tau(h_t_1, tau_input)

        f = self.dense_f(h_tau)

        c_t, _ = self.hippo(f=f)

        return h_tau, c_t

    @staticmethod
    def initialize_carry(
        rng,
        batch_size: int,
        hidden_size: int,
        init_fn=nn.initializers.zeros,
    ):
        key1, key2 = jax.random.split(rng)
        mem_shape = (batch_size, hidden_size)
        return init_fn(key1, mem_shape), init_fn(key2, mem_shape)


class LRU(nn.Module):
    """
    Description:

    Args:

    Examples:
        https://github.com/Gothos/LRU-pytorch/blob/main/LRU_pytorch/LRU.py
    """

    in_features: int
    out_features: int
    state_features: int
    rmin: float = 0
    rmax: float = 1
    max_phase: float = 6.283

    def setup(self) -> None:
        # All LRU parameters
        (
            nu_log,
            theta_log,
            B_re,
            B_im,
            C_re,
            C_im,
            D,
            gamma_log,
        ) = self.init_lru_parameters(
            N=self.state_features,
            H=self.in_features,
            r_min=self.rmin,
            r_max=self.rmax,
            max_phase=self.max_phase,
        )

        self.nu_log = self.param("nu_log", lambda _: nu_log)
        self.theta_log = self.param("theta_log", lambda _: theta_log)
        self.B_re = self.param("B_re", lambda _: B_re)
        self.B_im = self.param("B_im", lambda _: B_im)
        self.C_re = self.param("C_re", lambda _: C_re)
        self.C_im = self.param("C_im", lambda _: C_im)
        self.D = self.param("D", lambda _: D)
        self.gamma_log = self.param("gamma_log", lambda _: gamma_log)

    def __call__(
        self, carry: Tuple[Float[Array, ""], Float[Array, ""]], x: jnp.ndarray
    ) -> jnp.ndarray:
        # Materializing the diagonal of Lambda and projections
        Lambda = jnp.exp(-jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log))
        B_norm = (self.B_re + 1j * self.B_im) * jnp.expand_dims(
            jnp.exp(self.gamma_log), axis=-1
        )
        C = self.C_re + 1j * self.C_im

        # Running the LRU + output projection
        # For details on parallel scan, check discussion in Smith et al (2022).
        Lambda_elements = jnp.repeat(Lambda[None, ...], x.shape[0], axis=0)
        Bu_elements = jax.vmap(lambda u: B_norm @ u)(x)
        elements = (Lambda_elements, Bu_elements)
        _, inner_states = jax.lax.parallel_scan(
            self.binary_operator_diag, elements
        )  # all x_k
        y = jax.vmap(lambda x, u: (C @ x).real + self.D * u)(inner_states, x)

        return y

    def binary_operator_diag(self, element_i, element_j):
        # Binary operator for parallel scan of linear recurrence.
        a_i, bu_i = element_i
        a_j, bu_j = element_j
        return (a_j * a_i), (a_j * bu_i + bu_j)

    def init_lru_parameters(self, N, H, r_min=0, r_max=1, max_phase=6.28):
        """Initialize parameters of the LRU layer."""

        # N: state dimension, H: model dimension
        # Initialization of Lambda is complex valued distributed uniformly on ring
        # between r_min and r_max, with phase in [0, max_phase].
        u1 = nn.initializers.lecun_normal()(
            N,
        )
        u2 = nn.initializers.lecun_normal()(
            N,
        )
        nu_log = jnp.log(-0.5 * jnp.log(u1 * (r_max**2 - r_min**2) + r_min**2))
        theta_log = jnp.log(max_phase * u2)

        # Glorot initialized Input/Output projection matrices
        B_re = nn.initializers.lecun_normal()(N, H) / jnp.sqrt(2 * H)
        B_im = nn.initializers.lecun_normal()(N, H) / jnp.sqrt(2 * H)
        C_re = nn.initializers.lecun_normal()(H, N) / jnp.sqrt(N)
        C_im = nn.initializers.lecun_normal()(H, N) / jnp.sqrt(N)
        D = nn.initializers.lecun_normal()(
            H,
        )

        # Normalization factor
        diag_lambda = jnp.exp(-jnp.exp(nu_log) + 1j * jnp.exp(theta_log))
        gamma_log = jnp.log(jnp.sqrt(1 - jnp.abs(diag_lambda) ** 2))

        return nu_log, theta_log, B_re, B_im, C_re, C_im, D, gamma_log
