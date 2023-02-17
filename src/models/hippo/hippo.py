## import packages
import math
from typing import Any, Callable, List, Optional, Tuple, Union

import einops
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.numpy.linalg import inv
from jaxtyping import Array, Float, Float16, Float32, Float64
from scipy import special as ss
from scipy import signal

from src.models.hippo.transition import TransMatrix


class HiPPOLSI(nn.Module):
    """
    class that constructs a Linearly Scale Invariant (LSI) HiPPO model using the defined measure.

    Args:

        N (int):
            order of the HiPPO projection, aka the number of coefficients to describe the matrix

        max_length (int):
            maximum sequence length to be input

        step_size (float):
            step size used for descretization

        lambda_n (float):
            value associated with the tilt of legt
            - 1: tilt on legt
            - \sqrt(2n+1)(-1)^{N}: tilt associated with the legendre memory unit (LMU)

        alpha (float):
            The order of the Laguerre basis.

        beta (float):
            The scale of the Laguerre basis.

        GBT_alpha (float):
            represents which descretization transformation to use based off the alpha value

        measure (str):
            the measure used to define which way to instantiate the HiPPO matrix

        dtype (jnp.float):
            represents the float precision of the class

        unroll (bool):
            shows the rolled out coefficients over time/scale
    """

    N: int
    max_length: int = 1024
    step_size: float = 1.0
    lambda_n: float = 1.0
    alpha: float = 0.0
    beta: float = 1.0
    GBT_alpha: float = 0.5
    measure: str = "legs"
    dtype: Any = jnp.float32
    unroll: bool = False

    def setup(self):
        matrices = TransMatrix(
            N=self.N,
            measure=self.measure,
            lambda_n=self.lambda_n,
            alpha=self.alpha,
            beta=self.beta,
            dtype=self.dtype,
        )

        self.GBT_A_list, self.GBT_B_list = self.temporal_GBT(
            matrices.A, matrices.B, dtype=self.dtype
        )

        vals = jnp.linspace(0.0, 1.0, self.max_length)
        self.eval_matrix = (
            (
                (matrices.B)
                * ss.eval_legendre(
                    jnp.expand_dims(jnp.arange(self.N), -1), 2 * vals - 1
                )
            ).T
        ).astype(self.dtype)

    def __call__(
        self,
        f: Float[Array, "#batch seq_len input_size"],
        init_state: Optional[Float[Array, "#batch input_size N"]] = None,
    ) -> Union[
        Float[Array, "#batch seq_len input_size N"], Float[Array, "#batch input_size N"]
    ]:

        if init_state is None:
            init_state = jnp.zeros((f.shape[0], 1, self.N))

        c_k = self.recurrence(
            A=self.GBT_A_list,
            B=self.GBT_B_list,
            c_0=init_state,
            f=f,
            dtype=self.dtype,
        )

        return c_k

    def temporal_GBT(
        self, A: Float[Array, "N N"], B: Float[Array, "N input_size"], dtype=jnp.float32
    ) -> Tuple[List[Float[Array, "N N"]], List[Float[Array, "N input_size"]]]:
        """
        Creates the list of discretized GBT matrices for the given step size

        Args:
            A (jnp.ndarray):
                shape: (N, N)
                matrix to be discretized

            B (jnp.ndarray):
                shape: (N, 1)
                matrix to be discretized

            dtype (jnp.float):
                type of float precision to be used

        Returns:
            GBT_a_list (list):
                list of discretized A matrices across all time steps

            GBT_b_list (list):
                list of discretized B matrices across all time steps
        """
        GBT_a_list = []
        GBT_b_list = []
        for i in range(1, self.max_length + 1):
            GBT_A, GBT_B = self.discretize(
                A, B, step=i, alpha=self.GBT_alpha, dtype=dtype
            )
            GBT_a_list.append(GBT_A)
            GBT_b_list.append(GBT_B)

        return GBT_a_list, GBT_b_list

    def discretize(
        self,
        A: Float[Array, "N N"],
        B: Float[Array, "N input_size"],
        step: float,
        alpha: Union[float, str] = 0.5,
        dtype: Any = jnp.float32,
    ) -> Tuple[Float[Array, "N N"], Float[Array, "N 1"]]:
        """
        Function used for discretizing the HiPPO A and B matrices

        Args:
            A (jnp.ndarray):
                shape: (N, N)
                matrix to be discretized

            B (jnp.ndarray):
                shape: (N, 1)
                matrix to be discretized

            step (float):
                step size used for discretization

            alpha (float, optional):
                used for determining which generalized bilinear transformation to use
                - forward Euler corresponds to α = 0,
                - backward Euler corresponds to α = 1,
                - bilinear corresponds to α = 0.5,
                - Zero-order Hold corresponds to α > 1

            dtype (jnp.float):
                type of float precision to be used

        Returns:
            GBT_A (jnp.ndarray):
                shape: (N, N)
                discretized A matrix based on the given step size and alpha value

            GBT_B (jnp.ndarray):
                shape: (N, 1)
                discretized B matrix based on the given step size and alpha value
        """
        if alpha <= 1:
            assert alpha in [0, 0.5, 1], "alpha must be 0, 0.5, or 1"
        else:
            assert (
                alpha > 1 or type(alpha) == str
            ), "alpha must be greater than 1 for zero-order hold"
            if type(alpha) == str:
                assert (
                    alpha == "zoh"
                ), "if alpha is a string, it must be defined as 'zoh' for zero-order hold"

        I = jnp.eye(A.shape[0])

        if alpha <= 1:  # Generalized Bilinear Transformation
            step_size = 1 / step
            part1 = I - (step_size * alpha * A)
            part2 = I + (step_size * (1 - alpha) * A)

            GBT_A = jnp.linalg.lstsq(part1, part2, rcond=None)[0]
            GBT_B = jnp.linalg.lstsq(part1, (step_size * B), rcond=None)[0]

        else:  # Zero-order Hold
            # refer to this for why this works
            # https://en.wikipedia.org/wiki/Discretization#:~:text=A%20clever%20trick%20to%20compute%20Ad%20and%20Bd%20in%20one%20step%20is%20by%20utilizing%20the%20following%20property

            n = A.shape[0]
            b_n = B.shape[1]
            A_B_square = jnp.block(
                [[A, B], [jnp.zeros((b_n, n)), jnp.zeros((b_n, b_n))]]
            )
            A_B = jax.scipy.linalg.expm(
                A_B_square
                * (math.log((1 / step) + self.step_size) - math.log((1 / step)))
            )

            GBT_A = A_B[0:n, 0:n]
            GBT_B = A_B[0:-b_n, -b_n:]

        return GBT_A.astype(dtype), GBT_B.astype(dtype)

    def recurrence(
        self,
        A: List[Float[Array, "N N"]],
        B: List[Float[Array, "N input_size"]],
        c_0: Float[Array, "#batch input_size N"],
        f: Float[Array, "#batch seq_len input_size"],
        dtype: Any = jnp.float32,
    ) -> Union[
        Float[Array, "#batch seq_len input_size N"], Float[Array, "#batch input_size N"]
    ]:
        """
        Performs the recurrence of the HiPPO model using the discretized HiPPO A and B matrices as well as the HiPPO operator

        Args:
            A (jnp.ndarray):
                shape: (N, N)
                The list of discretized A matrices

            B (jnp.ndarray):
                shape: (N, 1)
                The list of discretized B matrices

            c_0 (jnp.ndarray):
                shape: (batch size, input length, N)
                the initial hidden state (i.e. the initial coefficients)

            f (jnp.ndarray):
                shape: (sequence length, 1)
                the input sequence


        Returns:
            c_s (list[jnp.ndarray]]):
                shape: (batch size, sequence length, input length, N)
                List of the vector of estimated coefficients representing the function, f(t), at each time step

            c_s[-1] (jnp.ndarray):
                shape: (batch size, sequence length, input length, N)
                Vector of the estimated coefficients representing the function, f(t), at the last time step
        """

        c_s = []

        c_k = c_0.copy()
        for i in range(f.shape[1]):
            c_k = jax.vmap(self.hippo_op, in_axes=(None, None, 0, 0))(
                A[i], B[i], c_k, f[:, i, :]
            )
            c_s.append((c_k.copy()).astype(dtype))

        if self.unroll:
            return (
                einops.rearrange(
                    c_s, "seq_len batch input_size N -> batch seq_len input_size N"
                )
            ).astype(
                dtype
            )  # list of hidden states
        else:
            return (c_s[-1]).astype(dtype)

    def hippo_op(
        self,
        Ad: Float[Array, "N N"],
        Bd: Float[Array, "N input_size"],
        c_k_i: Float[Array, "#batch input_size N"],
        f_k: Float[Array, "#batch seq_len input_size"],
    ) -> Float[Array, "#batch input_size N"]:
        """
        The HiPPO operator, that is used to perform the recurrence of the HiPPO model

        Args:
            Ad (jnp.ndarray):
                shape: (N, N)
                discretized A matrix

            Bd (jnp.ndarray):
                shape: (N, 1)
                discretized B matrix

            c_k_i:
                shape: (input length, N)
                previous hidden state

            f_k:
                shape: (input_size, )
                value of input sequence at time step k

        Returns:
            c_k (jnp.ndarray):
                shape: (input length, N)
                Vector of the estimated coefficients, given the history of the function/sequence up to time step k.
        """

        c_k = (jnp.dot(c_k_i, Ad.T)) + (Bd.T * f_k)

        return c_k

    def reconstruct(
        self, c: Float[Array, "#batch input_size N"]
    ) -> Float[Array, "#batch seq_len input_size"]:
        """reconstructs the input sequence from the estimated coefficients and the evaluation matrix

        Args:
            c (jnp.ndarray):
                shape: (batch size, input length, N)
                Vector of the estimated coefficients, given the history of the function/sequence

        Returns:
            y (jnp.ndarray):
                shape: (batch size, input length, input size)
                The reconstructed input sequence
        """
        eval_matrix = self.eval_matrix

        y = None
        if len(c.shape) == 3:
            c = einops.rearrange(c, "batch input_size N -> batch N input_size")
            y = jax.vmap(jnp.dot, in_axes=(None, 0))(eval_matrix, c)
        elif len(c.shape) == 4:
            c = einops.rearrange(
                c, "batch seq_len input_size N -> batch seq_len N input_size"
            )
            time_dot = jax.vmap(jnp.dot, in_axes=(None, 0))
            batch_time_dot = jax.vmap(time_dot, in_axes=(None, 0))
            y = batch_time_dot(eval_matrix, c)
        else:
            raise ValueError(
                "c must be of shape (batch size, input length, N) or (batch seq_len input_size N)"
            )

        return y


class HiPPOLTI(nn.Module):
    """
    class that constructs a Linearly Time Invariant (LTI) HiPPO model using the defined measure.

    Args:

        N (int):
            Order of the HiPPO projection, aka the number of coefficients to describe the matrix

        step_size (float):
            Step size used for descretization

        lambda_n (float):
            Value associated with the tilt of legt
            - 1: tilt on legt
            - \sqrt(2n+1)(-1)^{N}: tilt associated with the legendre memory unit (LMU)

        alpha (float):
            The order of the Laguerre basis.

        beta (float):
            The scale of the Laguerre basis.

        GBT_alpha (float):
            Represents which descretization transformation to use based off the alpha value

        measure (str):
            The measure used to define which way to instantiate the HiPPO matrix

        basis_size (float):
            The intended maximum value of the basis function for the coefficients to be projected onto

        dtype (jnp.float):
            Represents the float precision of the class

        unroll (bool):
            Shows the rolled out coefficients over time/scale
    """

    N: int
    step_size: float = 1.0
    lambda_n: float = 1.0
    alpha: float = 0.0
    beta: float = 1.0
    GBT_alpha: float = 0.5
    measure: str = "legs"
    basis_size: float = 1.0
    dtype: Any = jnp.float32
    unroll: bool = False

    def setup(self) -> None:
        matrices = TransMatrix(
            N=self.N,
            measure=self.measure,
            lambda_n=self.lambda_n,
            alpha=self.alpha,
            beta=self.beta,
            dtype=self.dtype,
        )

        self.Ad, self.Bd = self.discretize(
            A=matrices.A,
            B=matrices.B,
            step=self.step_size,
            alpha=self.GBT_alpha,
            dtype=self.dtype,
        )

        self.B = matrices.B

        self.vals = jnp.arange(0.0, self.basis_size, self.step_size)
        self.eval_matrix = self.basis(
            B=self.B,
            method=self.measure,
            N=self.N,
            vals=self.vals,
            c=0.0,
            dtype=self.dtype,
        )  # (T/dt, N)

    def __call__(
        self,
        f: Float[Array, "#batch seq_len input_size"],
        init_state: Optional[Float[Array, "#batch input_size N"]] = None,
    ) -> Union[
        Float[Array, "#batch seq_len input_size N"], Float[Array, "#batch input_size N"]
    ]:

        if init_state is None:
            init_state = jnp.zeros((f.shape[0], 1, self.N))

        c_k = self.recurrence(
            Ad=self.Ad,
            Bd=self.Bd,
            c_0=init_state,
            f=f,
            dtype=self.dtype,
        )

        return c_k

    def discretize(
        self,
        A: Float[Array, "N N"],
        B: Float[Array, "N input_size"],
        step: float,
        alpha: Union[float, str] = 0.5,
        dtype: Any = jnp.float32,
    ) -> Tuple[Float[Array, "N N"], Float[Array, "N input_size"]]:
        """
        Function used for discretizing the HiPPO A and B matrices

        Args:
            A (jnp.ndarray):
                shape: (N, N)
                matrix to be discretized

            B (jnp.ndarray):
                shape: (N, 1)
                matrix to be discretized

            step (float):
                step size used for discretization

            alpha (float, optional):
                used for determining which generalized bilinear transformation to use
                - forward Euler corresponds to α = 0,
                - backward Euler corresponds to α = 1,
                - bilinear corresponds to α = 0.5,
                - Zero-order Hold corresponds to α > 1

            dtype (jnp.float):
                type of float precision to be used

        Returns:
            GBT_A (jnp.ndarray):
                shape: (N, N)
                discretized A matrix based on the given step size and alpha value

            GBT_B (jnp.ndarray):
                shape: (N, 1)
                discretized B matrix based on the given step size and alpha value
        """
        if alpha <= 1:
            assert alpha in [0, 0.5, 1], "alpha must be 0, 0.5, or 1"
        else:
            assert (
                alpha > 1 or type(alpha) == str
            ), "alpha must be greater than 1 for zero-order hold"
            if type(alpha) == str:
                assert (
                    alpha == "zoh"
                ), "if alpha is a string, it must be defined as 'zoh' for zero-order hold"

        I = jnp.eye(A.shape[0])

        if alpha <= 1:  # Generalized Bilinear Transformation
            # C = jnp.ones((1, A.shape[0]))
            # D = jnp.zeros((1,))
            step_size = step  # 1 / step
            jax.debug.print("step: {x}", x=step)
            jax.debug.print("step_size: {x}", x=step_size)
            part1 = I - (step_size * alpha * A)
            part2 = I + (step_size * (1 - alpha) * A)

            GBT_A = jnp.linalg.lstsq(part1, part2, rcond=None)[0]
            GBT_B = jnp.linalg.lstsq(part1, (step_size * B), rcond=None)[0]
            # GBT_A, GBT_B, _, _, _ = signal.cont2discrete(
            #     (A, B, C, D), dt=step, method="gbt", alpha=alpha
            # )

        else:  # Zero-order Hold
            # refer to this for why this works
            # https://en.wikipedia.org/wiki/Discretization#:~:text=A%20clever%20trick%20to%20compute%20Ad%20and%20Bd%20in%20one%20step%20is%20by%20utilizing%20the%20following%20property

            n = A.shape[0]
            b_n = B.shape[1]
            A_B_square = jnp.block(
                [[A, B], [jnp.zeros((b_n, n)), jnp.zeros((b_n, b_n))]]
            )
            A_B = jax.scipy.linalg.expm(A_B_square * self.step_size)

            GBT_A = A_B[0:n, 0:n]
            GBT_B = A_B[0:-b_n, -b_n:]

        return GBT_A.astype(dtype), GBT_B.astype(dtype)

    def recurrence(
        self,
        Ad: Float[Array, "N N"],
        Bd: Float[Array, "N input_size"],
        c_0: Float[Array, "#batch input_size N"],
        f: Float[Array, "#batch seq_len input_size"],
        dtype: Any = jnp.float32,
    ) -> Union[
        Float[Array, "#batch seq_len input_size N"], Float[Array, "#batch input_size N"]
    ]:
        """
        Performs the recurrence of the HiPPO model using the discretized HiPPO A and B matrices as well as the HiPPO operator

        Args:
            A (jnp.ndarray):
                shape: (N, N)
                The discretized A matrix weighted by the step size

            B (jnp.ndarray):
                shape: (N, 1)
                The discretized A matrix weighted by the step size

            c_0 (jnp.ndarray):
                shape: (batch size, input length, N)
                the initial hidden state (i.e. the initial coefficients)

            f (jnp.ndarray):
                shape: (sequence length, 1)
                the input sequence


        Returns:
            c_s (list[jnp.ndarray]]):
                shape: (batch size, sequence length, input length, N)
                List of the vector of estimated coefficients representing the function, f(t), at each time step

            c_k (jnp.ndarray):
                shape: (batch size, sequence length, input length, N)
                Vector of the estimated coefficients representing the function, f(t), at the last time step
        """

        def hippo_op(
            c_k_i: Float[Array, "#batch input_size N"],
            f_k: Float[Array, "#batch seq_len input_size"],
        ) -> Tuple[
            Float[Array, "#batch seq_len input_size N"],
            Float[Array, "#batch input_size N"],
        ]:
            """
            Get descretized coefficients of the hidden state by applying HiPPO matrix to input sequence, u_k, and previous hidden state, x_k_1.

            Args:
                c_k_i:
                shape: (input length, N)
                previous hidden state

            f_k:
                shape: (input_size, )
                value of input sequence at time step k

            Returns:
                c_k (jnp.ndarray):
                    shape: (input length, N)
                    Vector of the estimated coefficients, given the history of the function/sequence up to time step k.

                c_k (list[jnp.ndarray]):
                    shape: (input length, N)
                    List of the vector of estimated coefficients representing the function, f(t), at each time step
            """

            c_k = (jnp.dot(c_k_i, Ad.T)) + (Bd.T * f_k)

            return c_k, c_k

        c_k, c_s = jax.vmap(jax.lax.scan, in_axes=(None, 0, 0))(hippo_op, c_0, f)

        if self.unroll:
            return c_s.astype(dtype)
        else:
            return c_k.astype(dtype)

    def measure_fn(self, method: str, c: float = 0.0) -> Callable:

        if method == "legs":
            fn = lambda x: jnp.heaviside(x, 1.0) * jnp.exp(-x)
        elif method in ["legt", "lmu"]:
            fn = lambda x: jnp.heaviside(x, 0.0) * jnp.heaviside(1.0 - x, 0.0)
        elif method == "lagt":
            fn = lambda x: jnp.heaviside(x, 1.0) * jnp.exp(-x)
        elif method in ["fourier", "fru", "fout", "foud"]:
            fn = lambda x: jnp.heaviside(x, 1.0) * jnp.heaviside(1.0 - x, 1.0)
        else:
            raise NotImplementedError

        fn_tilted = lambda x: jnp.exp(c * x) * fn(x)

        return fn_tilted

    def basis(
        self,
        B: Float[Array, "N input_size"],
        method: str,
        N: int,
        vals: Float[Array, "1"],
        c: float = 0.0,
        truncate_measure: bool = True,
        dtype: Any = jnp.float32,
    ) -> Float[Array, "seq_len N"]:
        """
        vals: list of times (forward in time)
        returns: shape (T, N) where T is length of vals
        """

        if method == "legs":
            _vals = jnp.exp(-vals)
            base = (2 * jnp.arange(N) + 1) ** 0.5 * (-1) ** jnp.arange(
                N
            )  # unscaled, untranslated legendre polynomial matrix
            base = einops.rearrange(base, "N -> N 1")
            eval_matrix = (
                base
                * ss.eval_legendre(jnp.expand_dims(jnp.arange(N), -1), 1 - 2 * _vals)
            ).T  # (L, N)

        elif method in ["legt", "lmu"]:
            base = (2 * jnp.arange(N) + 1) ** 0.5 * (-1) ** jnp.arange(
                N
            )  # unscaled, untranslated legendre polynomial matrix
            base = einops.rearrange(base, "N -> N 1")
            eval_matrix = (
                base
                * ss.eval_legendre(jnp.expand_dims(jnp.arange(N), -1), 2 * vals - 1)
            ).T
        elif method == "lagt":
            _vals = vals[::-1]
            eval_matrix = ss.eval_genlaguerre(
                jnp.expand_dims(jnp.arange(N), -1), 0, _vals
            )
            eval_matrix = (eval_matrix * jnp.exp(-_vals / 2)).T
        elif method in ["fourier", "fru", "fout", "foud"]:
            cos = 2**0.5 * jnp.cos(
                2 * jnp.pi * jnp.arange(N // 2)[:, None] * (vals)
            )  # (N/2, T/dt)
            sin = 2**0.5 * jnp.sin(
                2 * jnp.pi * jnp.arange(N // 2)[:, None] * (vals)
            )  # (N/2, T/dt)
            cos = cos.at[0].set(cos[0] / 2**0.5)
            eval_matrix = jnp.stack([cos.T, sin.T], axis=-1).reshape(-1, N)  # (T/dt, N)
        else:
            raise NotImplementedError(f"method {method} not implemented")

        if truncate_measure:
            tilting_fn = self.measure_fn(method, c=c)
            val = tilting_fn(vals)
            eval_matrix = eval_matrix.at[val == 0.0].set(0.0)

        p = eval_matrix * jnp.exp(-c * vals)[:, None]  # [::-1, None]

        return p.astype(dtype)

    def reconstruct(
        self, c: Float[Array, "#batch input_size N"], evals=None
    ) -> Float[Array, "#batch seq_len input_size"]:
        """reconstructs the input sequence from the estimated coefficients and the evaluation matrix

        Args:
            c (jnp.ndarray):
                shape: (batch size, input length, N)
                Vector of the estimated coefficients, given the history of the function/sequence

            evals (jnp.ndarray, optional):
                shape: ()
                Vector of the evaluation points. Defaults to None.

        Returns:
            y (jnp.ndarray):
                shape: (batch size, input length, input size)
                The reconstructed input sequence
        """
        if evals is not None:
            eval_matrix = self.basis(
                B=self.B, method=self.measure, N=self.N, vals=evals
            )
        else:
            eval_matrix = self.eval_matrix

        y = None
        if len(c.shape) == 3:
            c = einops.rearrange(c, "batch input_size N -> batch N input_size")
            y = jax.vmap(jnp.dot, in_axes=(None, 0))(eval_matrix, c)
        elif len(c.shape) == 4:
            c = einops.rearrange(
                c, "batch seq_len input_size N -> batch seq_len N input_size"
            )
            time_dot = jax.vmap(jnp.dot, in_axes=(None, 0))
            batch_time_dot = jax.vmap(time_dot, in_axes=(None, 0))
            y = batch_time_dot(eval_matrix, c)
        else:
            raise ValueError(
                "c must be of shape (batch size, input length, N) or (batch seq_len input_size N)"
            )

        return y


class DLPR_HiPPO:
    def __init__(self) -> None:
        pass

    def discrete_DPLR(self, Lambda, P, Q, B, C, step, L):
        """
        A_bar = (I - (step/2) \dot A)^{-1} (I + (step/2) \dot A)
        B_bar = (I - (step/2) \dot A)^{-1} (step/2) \dot B

        we can reconstruct the A_bar terms to more closely resemble euler methods
        $$
        \begin{align}
            (I - (step/2) \dot A) &= I + (step/2)(\Lambda - PQ^{*}) \\
            (I - (step/2) \dot A) &= step/2 [(step/2) \dot I + (\Lambda - PQ^{*})] \\
            (I - (step/2) \dot A) &= \step/2 \dot A_{0}
        \end{align}
        $$


        Same goes for backward Euler but using the woodbury identity, where $D = ((2/step) - \Lambda)^{-1}$
        $$
        \begin{align}
            (I - (step/2) \dot A)^{-1} &= (I - (step/2)(\Lambda - PQ^{*}))^{-1} \\
            (I - (step/2) \dot A)^{-1} &= (2/step)[(2/step) - \Lambda + PQ^{*}]^{-1} \\
            (I - (step/2) \dot A)^{-1} &= (2/step)[D - DP(1 + Q^{*}DP)^{-1} Q^{*}D]^{-1} \\
            (I - (step/2) \dot A)^{-1} &= (2/step)A_{1} \\
        \end{align}
        $$

        making the discrete ssm:
        $$
        \begin{align}
            x_{k} &= \Bar{A}x_{k-1} + \Bar{B}u_{k} \\
                  &= A_{1}A_{0}x_{k-1} + 2A_{1}B_{0}u_{k} \\
            y_{k} &= Cx_{k} + Du_{k}
        \end{align}
        $$

        Args:
            Lambda ([type]): [description]
            P ([type]): [description]
            Q ([type]): [description]
            B ([type]): [description]
            C ([type]): [description]
            step ([type]): [description]
            L ([type]): [description]

        Returns:
            Ab ([type]): [description]
            Bb ([type]): [description]
            Cb ([type]): [description]
        """

        # Convert parameters to matrices
        B = B[:, jnp.newaxis]
        Ct = C[jnp.newaxis, :]

        N = Lambda.shape[0]
        A = jnp.diag(Lambda) - P[:, jnp.newaxis] @ Q[:, jnp.newaxis].conj().T
        I = jnp.eye(N)

        # Forward Euler
        A0 = (2.0 / step) * I + A

        # Backward Euler
        D = jnp.diag(1.0 / ((2.0 / step) - Lambda))
        Qc = Q.conj().T.reshape(1, -1)
        P2 = P.reshape(-1, 1)
        A1 = D - (D @ P2 * (1.0 / (1 + (Qc @ D @ P2))) * Qc @ D)

        # A bar and B bar
        Ab = A1 @ A0
        Bb = 2 * A1 @ B

        # Recover Cbar from Ct
        Cb = Ct @ jnp.linalg.inv(I - jnp.linalg.matrix_power(Ab, L)).conj()
        return Ab, Bb, Cb.conj()

    def initial_C(self, measure, N, dtype=jnp.float32):
        """Return C that captures the other endpoint in the HiPPO approximation"""

        if measure == "legt":
            C = (jnp.arange(N, dtype=dtype) * 2 + 1) ** 0.5 * (-1) ** jnp.arange(N)
        elif measure == "fourier":
            C = jnp.zeros(N)
            C[0::2] = 2**0.5
            C[0] = 1
        else:
            C = jnp.zeros(N, dtype=dtype)  # (N)

        return C
