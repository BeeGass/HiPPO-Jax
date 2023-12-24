## import packages
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import einops
import jax
import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array, Float, Int
from scipy import special as ss

from src.models.hippo.transition import (
    legs,
    legs_initializer,
    legt,
    legt_initializer,
    lmu,
    lmu_initializer,
    lagt,
    lagt_initializer,
    fru,
    fru_initializer,
    fout,
    fout_initializer,
    foud,
    foud_initializer,
    chebt,
    chebt_initializer,
)
from src.models.model import Model


class HiPPOLSI(Model):
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
    unroll: bool = True
    recon: bool = False

    def setup(self) -> None:
        matrices = TransMatrix(
            N=self.N,
            measure=self.measure,
            lambda_n=self.lambda_n,
            alpha=self.alpha,
            beta=self.beta,
            dtype=self.dtype,
        )

        GBT_A_stacked, GBT_B_stacked = self.temporal_GBT(
            matrices.A, matrices.B, dtype=self.dtype
        )
        self.GBT_A_stacked = self.param("Ad", lambda _: GBT_A_stacked)
        self.GBT_B_stacked = self.param("Bd", lambda _: GBT_B_stacked)

        vals = jnp.linspace(0.0, 1.0, self.max_length)
        self.eval_matrix = (
            (
                jax.lax.stop_gradient(
                    ss.eval_legendre(
                        jnp.expand_dims(jnp.arange(self.N), -1), 2 * vals - 1
                    )
                )
                * (matrices.B)
            ).T
        ).astype(self.dtype)

    def __call__(
        self,
        f: Float[Array, "#batch seq_len input_size"],
        init_state: Optional[Float[Array, "#batch input_size N"]] = None,
    ) -> Tuple[
        Union[
            Float[Array, "#batch seq_len input_size N"],
            Float[Array, "#batch input_size N"],
        ],
        Union[
            Float[Array, "#batch input_size N"],
            Float[Array, "#batch seq_len input_size N"],
        ],
    ]:

        if init_state is None:
            init_state = jnp.zeros((f.shape[0], 1, self.N))

        c_k = self.recurrence(
            A=self.GBT_A_stacked,
            B=self.GBT_B_stacked,
            c_0=init_state,
            f=f,
            dtype=self.dtype,
        )

        if self.recon:
            if self.measure in ["legs", "legt", "lmu", "lagt", "fout"]:
                y = self.reconstruct(c_k)
                return c_k, y
            else:
                return c_k, c_k
        else:
            return c_k, c_k

    def temporal_GBT(
        self, A: Float[Array, "N N"], B: Float[Array, "N input_size"], dtype=jnp.float32
    ) -> Tuple[Float[Array, "time N N"], Float[Array, "time N input_size"]]:
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
            GBT_A_stacked (jnp.ndarray):
                stack of discretized A matrices across all time steps

            GBT_B_stacked (jnp.ndarray):
                stack of discretized B matrices across all time steps
        """
        time = jnp.arange(1, (self.max_length + 1), 1, dtype=int)
        time = time[:, None]

        GBT_A_stacked, GBT_B_stacked = jax.vmap(
            self.discretize, in_axes=(None, None, 0, None, None)
        )(A, B, time, self.GBT_alpha, dtype)

        return GBT_A_stacked, GBT_B_stacked

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
                A_B_square * (jnp.log(step + 1) - jnp.log(step))
            )

            GBT_A = A_B[0:n, 0:n]
            GBT_B = A_B[0:-b_n, -b_n:]

        return GBT_A.astype(dtype), GBT_B.astype(dtype)

    def recurrence(
        self,
        A: Float[Array, "seq_len N N"],
        B: Float[Array, "seq_len N input_size"],
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
                shape: (seq_len, N, N)
                The stack of discretized A matrices

            B (jnp.ndarray):
                shape: (seq_len, N, 1)
                The stack of discretized B matrices

            c_0 (jnp.ndarray):
                shape: (batch_size, input_length, N)
                the initial hidden state (i.e. the initial coefficients)

            f (jnp.ndarray):
                shape: (batch_size, seq_len, 1)
                the input sequence


        Returns:
            c_s (jnp.ndarray):
                shape: (batch size, sequence length, input_length, N)
                List of the vector of estimated coefficients representing the function, f(t), at each time step

            c_s[-1] (jnp.ndarray):
                shape: (batch size, input_length, N)
                Vector of the estimated coefficients representing the function, f(t), at the last time step
        """

        def scan_fn(c_k, i):
            c_k = jax.vmap(self.hippo_op, in_axes=(None, None, 0, 0))(
                A[i], B[i], c_k, f[:, i, :]
            )
            return c_k, c_k

        c_k, c_s = jax.lax.scan(f=scan_fn, init=c_0, xs=jnp.arange(f.shape[1]))

        if self.unroll:
            return (
                einops.rearrange(
                    c_s, "seq_len batch input_size N -> batch seq_len input_size N"
                )
            ).astype(
                dtype
            )  # list of hidden states
        else:
            return (c_k).astype(dtype)

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
        self,
        c: Union[
            Float[Array, "#batch input_size N"],
            Float[Array, "#batch seq_len input_size N"],
        ],
    ) -> Union[
        Float[Array, "#batch seq_len input_size"],
        Float[Array, "#batch time seq_len input_size"],
    ]:
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
            batch_time_dot = jax.vmap(jnp.dot, in_axes=(None, 0))
            time_dot = jax.vmap(batch_time_dot, in_axes=(None, 0))
            y = time_dot(eval_matrix, c)
        else:
            raise ValueError(
                "c must be of shape (batch size, input length, N) or (batch seq_len input_size N)"
            )

        return y


class HiPPOLTI(Model):
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
    recon: bool = False

    def setup(self) -> None:
        matrices = TransMatrix(
            N=self.N,
            measure=self.measure,
            lambda_n=self.lambda_n,
            alpha=self.alpha,
            beta=self.beta,
            dtype=self.dtype,
        )

        Ad, Bd = self.discretize(
            A=matrices.A,
            B=matrices.B,
            step=self.step_size,
            alpha=self.GBT_alpha,
            dtype=self.dtype,
        )

        self.Ad = self.param("Ad", lambda _: Ad)
        self.Bd = self.param("Bd", lambda _: Bd)

        if self.measure in ["legs", "legt", "lmu", "lagt", "fout"]:
            self.vals = jnp.arange(0.0, self.basis_size, self.step_size)
            self.eval_matrix = self.basis(
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
    ) -> Tuple[
        Union[
            Float[Array, "#batch seq_len input_size N"],
            Float[Array, "#batch input_size N"],
        ],
        Union[
            Float[Array, "#batch input_size N"],
            Float[Array, "#batch seq_len input_size N"],
        ],
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

        if self.recon:
            if self.measure in ["legs", "legt", "lmu", "lagt", "fout"]:
                y = self.reconstruct(c_k)
                return c_k, y
            else:
                return c_k, c_k
        else:
            return c_k, c_k

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
            step_size = step
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
            Float[Array, "#batch input_size N"],
            Float[Array, "#batch seq_len input_size N"],
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
        """
        Returns a function that is used to measure the distance between the input sequence and the estimated coefficients

        Args:
            method (str):
                The method used to measure the distance between the input sequence and the estimated coefficients

            c (float):
                The tilt of the function used to measure the distance between the input sequence and the estimated coefficients

        Returns:
            fn_tilted (Callable):
                The function used to measure the distance between the input sequence and the estimated coefficients

        """

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
        method: str,
        N: int,
        vals: Float[Array, "1"],
        c: float = 0.0,
        truncate_measure: bool = True,
        dtype: Any = jnp.float32,
    ) -> Float[Array, "seq_len N"]:
        """
        Creates the basis matrix (eval matrix) for the appropriate HiPPO method.

        Args:
            B (jnp.ndarray):
                shape: (N, 1)
                The HiPPO B matrix

            method (str):
                The HiPPO method to use

            N (int):
                The number of basis functions to use

            vals (jnp.ndarray):
                shape: (seq_len, )
                The values to evaluate the basis functions at

            c (float):
                The constant to use for the tilted measure

            truncate_measure (bool):
                Whether or not to truncate the measure to the interval [0, 1]

            dtype (Any):
                The dtype to use for the basis matrix

        Returns:
            eval_matrix (jnp.ndarray):
                shape: (seq_len, N)
                The basis matrix
        """

        if method == "legs":
            _vals = jnp.exp(-vals)
            base = (2 * jnp.arange(N) + 1) ** 0.5 * (-1) ** jnp.arange(
                N
            )  # unscaled, untranslated legendre polynomial matrix
            base = einops.rearrange(base, "N -> N 1")
            eval_matrix = (
                jax.lax.stop_gradient(
                    ss.eval_legendre(jnp.expand_dims(jnp.arange(N), -1), 1 - 2 * _vals)
                )
                * base
            ).T  # (L, N)

        elif method in ["legt", "lmu"]:
            base = (2 * jnp.arange(N) + 1) ** 0.5 * (-1) ** jnp.arange(
                N
            )  # unscaled, untranslated legendre polynomial matrix
            base = einops.rearrange(base, "N -> N 1")
            eval_matrix = (
                jax.lax.stop_gradient(
                    ss.eval_legendre(jnp.expand_dims(jnp.arange(N), -1), 2 * vals - 1)
                )
                * base
            ).T

        elif method == "lagt":
            _vals = vals[::-1]
            eval_matrix = jax.lax.stop_gradient(
                ss.eval_genlaguerre(jnp.expand_dims(jnp.arange(N), -1), 0, _vals)
            )
            eval_matrix = (eval_matrix * jnp.exp(-_vals / 2)).T
        elif method in ["fourier", "fout"]:
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
            eval_matrix = self.basis(method=self.measure, N=self.N, vals=evals)
        else:
            eval_matrix = self.eval_matrix

        y = None
        if len(c.shape) == 3:
            c = einops.rearrange(c, "batch input_size N -> batch N input_size")
            y = jax.vmap(jnp.dot, in_axes=(None, 0))(eval_matrix, c)
            y = einops.rearrange(y, "batch seq_len 1 -> batch seq_len")
            y = jax.vmap(jnp.flip, in_axes=(0, None))(y, 0)
        elif len(c.shape) == 4:
            c = einops.rearrange(
                c, "batch seq_len input_size N -> batch seq_len N input_size"
            )
            time_dot = jax.vmap(jnp.dot, in_axes=(None, 0))
            batch_time_dot = jax.vmap(time_dot, in_axes=(None, 0))
            y = batch_time_dot(eval_matrix, c)
            y = einops.rearrange(
                y, "batch seq_len 1 seq_len2 -> batch seq_len seq_len2"
            )
            y = jax.vmap(jax.vmap(jnp.flip, in_axes=(0, None)), in_axes=(0, None))(y, 0)
        else:
            raise ValueError(
                "c must be of shape (batch size, input length, N) or (batch seq_len input_size N)"
            )

        return y
