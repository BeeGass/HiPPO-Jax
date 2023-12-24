## import packages
from typing import Any, Callable, Tuple, Union, Dict

import numpy as np
import einops
import jax
import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array, Float, Int
from scipy import special as ss

from src.models.hippo.transition import (
    initializer,
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


class HiPPOCell(Model):
    @staticmethod
    def initialize_state(
        rng,
        batch_size: int,
        hidden_size: int,
        init_fn: Callable,
    ):
        raise NotImplementedError


class HiPPOLSICell(HiPPOCell):

    features: int
    max_length: int
    alpha: float = 0.5
    init_t: int = 0
    recon: bool = True
    A_init_fn: Callable = legs
    B_init_fn: Callable = legs
    dtype: Any = jnp.float32

    def setup(self) -> None:
        A, _, _ = self.A_init_fn(N=self.features, dtype=self.dtype)
        _, B, self.method = self.B_init_fn(N=self.features, dtype=self.dtype)

        As_d, Bs_d = jax.lax.stop_gradient(
            self.temporal_GBT(A=A, B=B, alpha=self.alpha, dtype=self.dtype)
        )

        A_init = initializer(As_d)
        B_init = initializer(Bs_d)

        self.As_d = self.param(
            "As_d",
            A_init,
            (self.max_length, self.features, self.features),
        )
        self.Bs_d = self.param("Bs_d", B_init, (self.max_length, self.features, 1))

        vals = np.linspace(0.0, 1.0, self.max_length)
        self.eval_matrix = jax.lax.stop_gradient(
            (
                jnp.asarray(
                    ss.eval_legendre(
                        np.expand_dims(np.arange(As_d.shape[1]), -1), 2 * vals - 1
                    )
                )
                * (B)
            ).T
        ).astype(self.dtype)

    def __call__(
        self,
        c_t_1: Float[Array, "#batch input_size N"],
        f: Float[Array, "#batch seq_len input_size"],
        t_step: int,
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
        t = t_step + self.init_t
        # c_t = (jnp.dot(c_t_1, (self.As_d[t]).T) + ((self.Bs_d[t]).T * f[t, :])).astype(
        #     self.dtype
        # )
        c_t = (jnp.dot(c_t_1, (self.As_d[t]).T) + ((self.Bs_d[t]).T * f)).astype(
            self.dtype
        )

        if self.method in ["legs", "legt", "lmu", "lagt", "fout"] and self.recon:
            y = self.reconstruct(c_t).astype(self.dtype)
            return c_t, (c_t, y)
        else:
            return c_t, (c_t, c_t)

    def temporal_GBT(
        self,
        A: Float[Array, "N N"],
        B: Float[Array, "N input_size"],
        alpha: float,
        dtype=jnp.float32,
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
        t = jnp.arange(1, (self.max_length + 1), 1, dtype=int)
        t = t[:, None]

        GBT_A_stacked, GBT_B_stacked = jax.vmap(
            self.discretize, in_axes=(None, None, 0, None, None)
        )(A, B, t, alpha, dtype)

        return GBT_A_stacked, GBT_B_stacked

    def discretize(
        self,
        A: Float[Array, "N N"],
        B: Float[Array, "N input_size"],
        step: Int[Array, "1"],
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

            GBT_A = jnp.linalg.solve(part1, part2)
            GBT_B = jnp.linalg.solve(part1, (step_size * B))

            # GBT_A = jnp.linalg.lstsq(part1, part2, rcond=None)[0]
            # GBT_B = jnp.linalg.lstsq(part1, (step_size * B), rcond=None)[0]

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
        if len(c.shape) == 2:
            c = einops.rearrange(c, "... input_size N -> ... N input_size")
            y = jnp.dot(eval_matrix, c)
            y = jnp.flip(y, 0)
        elif len(c.shape) == 3:
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

    @staticmethod
    def initialize_state(
        rng,
        batch_size: int,
        hidden_size: int,
        init_fn=nn.initializers.zeros,
    ):
        mem_shape = (batch_size, hidden_size)
        return init_fn(rng, mem_shape)


class HiPPOLTICell(HiPPOCell):

    features: int
    step_size: float
    basis_size: float
    alpha: float = 0.5
    recon: bool = True
    A_init_fn: Callable = legs
    B_init_fn: Callable = legs
    dtype: Any = jnp.float32

    def setup(self) -> None:
        A, _, _ = self.A_init_fn(N=self.features, dtype=self.dtype)
        _, B, self.method = self.B_init_fn(N=self.features, dtype=self.dtype)

        A_d, B_d = jax.lax.stop_gradient(
            self.discretize(
                A=A,
                B=B,
                step=self.step_size,
                alpha=self.alpha,
                dtype=self.dtype,
            )
        )

        A_init = initializer(A_d)
        B_init = initializer(B_d)

        self.A_d = self.param(
            "A_d",
            A_init,
            (self.features, self.features),
        )
        self.B_d = self.param("B_d", B_init, (self.features, 1))

        if self.method in ["legs", "legt", "lmu", "lagt", "fout"] and self.recon:
            vals = np.arange(0.0, self.basis_size, self.step_size)

            self.eval_matrix = jax.lax.stop_gradient(
                self.basis(
                    method=self.method,
                    N=self.features,
                    vals=vals,
                    c=0.0,
                    dtype=self.dtype,
                )  # (T/dt, N)
            )

    def __call__(
        self,
        c_t_1: Tuple[Float[Array, "#batch input_size N"], Int],
        f: Float[Array, "#batch seq_len input_size"],
    ) -> Tuple[
        Float[Array, "#batch input_size N"], Float[Array, "#batch input_size N"]
    ]:
        c_t = (jnp.dot(c_t_1, (self.A_d).T) + ((self.B_d).T * f)).astype(self.dtype)

        if self.method in ["legs", "legt", "lmu", "lagt", "fout"] and self.recon:
            y = self.reconstruct(c_t)
            return c_t, (c_t, y)
        else:
            return c_t, (c_t, c_t)

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

            GBT_A = jnp.linalg.solve(part1, part2)
            GBT_B = jnp.linalg.solve(part1, (step_size * B))

            # GBT_A = jnp.linalg.lstsq(part1, part2, rcond=None)[0]
            # GBT_B = jnp.linalg.lstsq(part1, (step_size * B), rcond=None)[0]

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
            fn = lambda x: np.heaviside(x, 1.0) * np.exp(-x)
        elif method in ["legt", "lmu"]:
            fn = lambda x: np.heaviside(x, 0.0) * np.heaviside(1.0 - x, 0.0)
        elif method == "lagt":
            fn = lambda x: np.heaviside(x, 1.0) * np.exp(-x)
        elif method in ["fourier", "fru", "fout", "foud"]:
            fn = lambda x: np.heaviside(x, 1.0) * np.heaviside(1.0 - x, 1.0)
        else:
            raise NotImplementedError

        fn_tilted = lambda x: np.exp(c * x) * fn(x)

        return fn_tilted

    def basis(
        self,
        method: str,
        N: int,
        vals: Float[Array, "1"],
        c: float = 0.0,
        truncate_measure: bool = True,
        dtype: Any = np.float32,
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
            _vals = np.exp(-vals)
            base = (2 * np.arange(N) + 1) ** 0.5 * (-1) ** np.arange(
                N
            )  # unscaled, untranslated legendre polynomial matrix
            base = einops.rearrange(base, "N -> N 1")
            eval_matrix = (
                ss.eval_legendre(np.expand_dims(np.arange(N), -1), 1 - 2 * _vals) * base
            ).T  # (L, N)

        elif method in ["legt", "lmu"]:
            base = (2 * np.arange(N) + 1) ** 0.5 * (-1) ** np.arange(
                N
            )  # unscaled, untranslated legendre polynomial matrix
            base = einops.rearrange(base, "N -> N 1")
            eval_matrix = (
                ss.eval_legendre(np.expand_dims(np.arange(N), -1), 2 * vals - 1) * base
            ).T

        elif method == "lagt":
            _vals = vals[::-1]
            eval_matrix = ss.eval_genlaguerre(
                np.expand_dims(np.arange(N), -1), 0, _vals
            )
            eval_matrix = (eval_matrix * np.exp(-_vals / 2)).T
        elif method in ["fourier", "fout"]:
            cos = 2**0.5 * np.cos(
                2 * np.pi * np.arange(N // 2)[:, None] * (vals)
            )  # (N/2, T/dt)
            sin = 2**0.5 * np.sin(
                2 * np.pi * np.arange(N // 2)[:, None] * (vals)
            )  # (N/2, T/dt)
            cos[0] = cos[0] / 2**0.5
            eval_matrix = np.stack([cos.T, sin.T], axis=-1).reshape(-1, N)  # (T/dt, N)
        else:
            raise NotImplementedError(f"method {method} not implemented")

        if truncate_measure:
            tilting_fn = self.measure_fn(method, c=c)
            val = tilting_fn(vals)
            eval_matrix[val == 0.0] = 0.0

        p = eval_matrix * np.exp(-c * vals)[:, None]  # [::-1, None]

        return jnp.asarray(a=p, dtype=dtype)

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
            eval_matrix = self.basis(method=self.method, N=self.features, vals=evals)
        else:
            eval_matrix = self.eval_matrix

        y = None
        if len(c.shape) == 2:
            c = einops.rearrange(c, "... input_size N -> ... N input_size")
            y = jnp.dot(eval_matrix, c)
            y = jnp.flip(y, 0)
        elif len(c.shape) == 3:
            c = einops.rearrange(c, "batch input_size N -> batch N input_size")
            y = jax.vmap(jnp.dot, in_axes=(None, 0))(eval_matrix, c)
            # jax.debug.print("y shape: {x}", x=y.shape)
            # y = einops.rearrange(y, "batch seq_len 1 -> batch seq_len")
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
                f"c must be of shape (batch size, input length, N) or (batch seq_len input_size N), got {c.shape}"
            )

        return y.astype(self.dtype)

    @staticmethod
    def initialize_state(
        rng,
        batch_size: int,
        hidden_size: int,
        init_fn=nn.initializers.zeros,
    ):
        mem_shape = (batch_size, hidden_size)
        return init_fn(rng, mem_shape)


class HiPPO(Model):
    """

    Examples:
            The matrix_args in the format of:
                {N: int,
                 measure: str,
                 lambda_n: Optional[float],
                 alpha: Optional[float], # rotation for lagt
                 beta: Optional[float], # rotation for lagt
                 dtype: Optional[jnp.dtype]
                }

            >>> {N:64, measure:"legs", lambda_n:1.0, dtype:jnp.float16}
            >>> {N:64, measure:"legt", lambda_n:2.0, dtype:jnp.float32} # produces LMU
            >>> {N:64, measure:"legt", lambda_n:1.0, dtype:jnp.float32} # produces LegT
            >>> {N:64, measure:"lagt", alpha:0.0, beta:1.0, dtype:jnp.float6} # produces LagT
            >>> {N:64, measure:"lagt", alpha:0.7, beta:1.4, dtype:jnp.float64} # produces a version of a slightly "rotated" LagT


            HiPPOLSICell, in the format of:
                (max_length: int
                 alpha: Optional[float], # alpha value for discretization
                 measure: Optional[str],
                 recon: Optional[bool],
                 dtype: Optional[jnp.dtype]
                )
            >>> {max_length=1024, alpha=0.0, measure="legs", recon=True, dtype=jnp.float16} # produces HiPPOLSICell w/ forward euler discretization, and reconstruction
            >>> {max_length=512, alpha=1.0, measure="legt", recon=False, dtype=jnp.float32} # produces HiPPOLSICell w/ backward euler discretization, and no reconstruction
            >>> {max_length=256, alpha=0.5, measure="fru", recon=True, dtype=jnp.float32} # produces HiPPOLSICell w/ bilinear transform discretization, and reconstruction
            >>> {max_length=512, alpha=2.0, measure="fout", recon=True, dtype=jnp.float32} # produces HiPPOLSICell w/ zero-order hold discretization, and reconstruction

            HiPPOLTICell, in the format of:
                (step_size: float, # 1 / sequence length
                 basis_size: float, # The intended maximum value of the basis function for the coefficients to be projected onto
                 alpha: Optional[float], # alpha value for discretization
                 recon: Optional[bool],
                 measure: Optional[str],
                 dtype: Optional[jnp.dtype]
                )
            >>> {step_size:1e-3, basis_size:1.0, alpha:0.0, recon:True, measure:"legs", dtype:jnp.float16} # produces HiPPOLTICell w/ forward euler discretization, and reconstruction, discretized every 1/1000 assuming a sequence length is 1000
            >>> {step_size:1e-4, basis_size:1.0, alpha:1.0, recon:True, measure:"lagt", dtype:jnp.float32} # produces HiPPOLTICell w/ backward euler discretization, and reconstruction, discretized every 1/10000 assuming a sequence length is 10000
            >>> {step_size:1e-2, basis_size:1.0, alpha:2.0, recon:True, measure:"foud", dtype:jnp.float64} # produces HiPPOLTICell w/ zero-order hold discretization, and reconstruction, discretized every 1/10000 assuming a sequence length is 100
            >>> {step_size:1e-2, basis_size:1.0, alpha:0.5, recon:True, measure:"fru", dtype:jnp.float16} # produces HiPPOLTICell w/ bilinear transform discretization, and reconstruction, discretized every 1/10000 assuming a sequence length is 100



    Args:
        features (int):
            The size of the hidden state of the HiPPO model

        hippo_cell (HiPPOCell):
            The HiPPOCell class to be used for the HiPPO model

        hippo_args (dict):
            The dict associated with the input parameters into the HiPPOCell class.

        matrix_args (dict):
            The dict associated with the input parameters into the TransMatrix class.

        unroll (bool):
            Determines if you wanted the full history (all time steps) of coefficients, and potentially reconstructions. Defaults to False

    Raises:
        ValueError: Enforces that the inputted cell is a HiPPOCell
    """

    features: int
    hippo_cell: HiPPOCell
    hippo_args: Dict
    init_t: int = 0
    unroll: bool = False
    st: bool = False

    def setup(self) -> None:

        if self.st:
            _hippo = nn.vmap(
                target=self.hippo_cell,
                in_axes=(0, 0, None),
                variable_axes={"params": 0},
                split_rngs={"params": True},
            )
            self.hippo = nn.scan(
                target=_hippo,
                in_axes=(1, 0),
                out_axes=1,
                variable_broadcast="params",
                split_rngs={"params": False},
            )(features=self.features, **self.hippo_args)

        else:
            _hippo = nn.vmap(
                target=self.hippo_cell,
                in_axes=(0, 0),
                variable_axes={"params": 0},
                split_rngs={"params": True},
            )
            self.hippo = nn.scan(
                target=_hippo,
                in_axes=1,
                out_axes=1,
                variable_broadcast="params",
                split_rngs={"params": False},
            )(features=self.features, **self.hippo_args)

    def __call__(
        self,
        f: Float[Array, "#batch seq_len input_size"],
        c_t_1: Float[Array, "#batch input_size N"],
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
        c_n = None
        c_s = None
        y_s = None
        if isinstance(self.hippo, HiPPOLTICell):
            c_n, (c_s, y_s) = self.hippo(c_t_1, f)
        elif isinstance(self.hippo, HiPPOLSICell):
            t = jnp.arange(self.init_t, f.shape[1])
            c_n, (c_s, y_s) = self.hippo(c_t_1, f, t)
        else:
            NotImplementedError(
                f"Only HiPPOLSICell and HiPPOLTICell are supported, got {self.hippo.__class__.__name__}"
            )

        if self.unroll:
            return c_s, y_s
        else:
            y_n = c_n.copy()
            return c_n, y_n  # TODO obtain last y_n from y_s

        # if isinstance(self.hippo, HiPPOLSICell):

        #     def lsi_scan_fn(carry, i):
        #         c_tm1, y_t_1 = carry
        #         c_t, y = jax.vmap(self._hippo, in_axes=(0, 0, None))(f, c_tm1, i)
        #         return (c_t, y), (c_t, y)

        #     (c_n, y_n), (c_s, y_s) = nn.scan(
        #         f=lsi_scan_fn,
        #         init=(c_t_1, jnp.ones(f.shape)),
        #         xs=(jnp.arange(f.shape[1] - self.init_t) + 1),
        #     )

        #     if self.unroll:
        #         return c_s, y_s

        #     else:
        #         return c_n, y_n

        # elif isinstance(self.hippo, HiPPOLTICell):

        #     def lti_scan_fn(carry, i):
        #         c_tm1, y_t_1 = carry
        #         c_t, y = jax.vmap(self._hippo, in_axes=(0, 0, None))(f, c_tm1, i)
        #         return (c_t, y), (c_t, y)

        #     (c_n, y_n), (c_s, y_s) = nn.scan(
        #         f=lti_scan_fn,
        #         init=(c_t_1, jnp.ones(f.shape)),
        #         xs=(jnp.arange(f.shape[1] - self.init_t) + 1),
        #     )

        #     if self.unroll:
        #         return c_s, y_s

        #     else:
        #         return c_n, y_n

        # else:
        #     raise ValueError("hippo must be of type HiPPOLSICell or HiPPOLTICell")

    @staticmethod
    def initialize_state(
        rng,
        batch_size: int,
        hidden_size: int,
        init_fn=nn.initializers.zeros,
    ):
        # mem_shape = (1, hidden_size)
        mem_shape = (batch_size, 1, hidden_size)
        return init_fn(rng, mem_shape)
