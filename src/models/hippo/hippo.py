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
)
from src.models.mlp.mlp import BatchedMLP
from src.models.model import Model

class HiPPOCell(Model):
    def discretize(
        self, A: Float[Array, "N N"], B: Float[Array, "N channel"], **kwargs
    ):
        raise NotImplementedError

    def reconstruct(self, c, **kwargs):
        raise NotImplementedError

    @staticmethod
    def initialize_state(
        rng, batch_size: int, channels: int, hidden_size: int, init_fn: Callable
    ):
        raise NotImplementedError


# ------------------------------------------------------------------------------------
# --------------------------------------- HiPPOLSICell -------------------------------
# ------------------------------------------------------------------------------------


class HiPPOLSICell(HiPPOCell):

    max_length: int
    alpha: float = 0.5
    init_t: int = 0
    recon: bool = True
    A_init: Callable = legs
    B_init: Callable = legs
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self,
        c_t_1: Float[Array, "channel N"],
        x: Float[Array, "t channel"],
        t_step: int = 0,
    ) -> Tuple[
        Float[Array, "channel N"],
        Union[
            Tuple[Float[Array, "channel N"], Float[Array, "channel N"]],
            Tuple[Float[Array, "channel N"], Float[Array, "t channel"]],
        ],
    ]:
        hidden_features = c_t_1.shape[-1]
        channels = c_t_1.shape[-2]

        A, _, _ = self.A_init(N=hidden_features, dtype=self.dtype)
        _, B, method = self.B_init(N=hidden_features, dtype=self.dtype)

        # make channel number of B matrices to account for making channel number of HiPPOs. HiPPO only works on a 1-D signal
        _B = jnp.tile(B, (channels,))

        # We discretize sequence length number of A and B matrices. We will be using a separate A and B matrix for each time step
        _As_d, _Bs_d = jax.lax.stop_gradient(
            self.temporal_GBT(A=A, B=_B, alpha=self.alpha, dtype=self.dtype)
        )

        # initialize the A and B matrices as functions so we can make them trainable parameters (if we want)
        A_init = initializer(_As_d)
        B_init = initializer(_Bs_d)

        # make the A and B matrices trainable parameters
        As_d = self.param(
            "As_d",
            A_init,
            (self.max_length, hidden_features, hidden_features),
        )
        Bs_d = self.param("Bs_d", B_init, (self.max_length, hidden_features, channels))

        # get the current timestep
        t = t_step + self.init_t

        # formulate the HiPPO operator
        c_t = jnp.einsum("mn, cn -> cn", As_d[t], c_t_1) + jnp.einsum(
            "nc, c -> cn", Bs_d[t], x
        )

        # reconstruct the input sequence from the estimated coefficients and the evaluation matrix
        if self.recon:
            y = self.reconstruct(self, c_t, B)
            return c_t, (c_t, y)
        else:
            return c_t, (c_t, c_t)

    def temporal_GBT(
        self,
        A: Float[Array, "N N"],
        B: Float[Array, "N channel"],
        alpha: float,
        dtype=jnp.float32,
    ) -> Tuple[Float[Array, "time N N"], Float[Array, "time N channel"]]:
        """
        Creates the list of discretized GBT matrices for the given step size

        Args:
            A (jnp.ndarray):
                shape: (N, N)
                matrix to be discretized

            B (jnp.ndarray):
                shape: (N, channel)
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
        t = einops.rearrange(t, "t -> t 1")

        GBT_A_stacked, GBT_B_stacked = jax.vmap(
            self.discretize, in_axes=(None, None, 0, None, None)
        )(A, B, t, alpha, dtype)

        return GBT_A_stacked, GBT_B_stacked

    def discretize(
        self,
        A: Float[Array, "N N"],
        B: Float[Array, "N channel"],
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
                shape: (N, channel)
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
                shape: (N, channel)
                discretized B matrix based on the given step size and alpha value
        """
        assert alpha in [0, 0.5, 1, 2], "alpha must be 0, 0.5, 1, 2"

        I = jnp.eye(A.shape[0])

        # Generalized Bilinear Transformation
        # referencing equation 13 within the discretization method section of the HiPPO paper
        if alpha <= 1:
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
            if A_B.dtype != dtype:
                A_B = A_B.astype(dtype)

            GBT_A = A_B[0:n, 0:n]
            GBT_B = A_B[0:-b_n, -b_n:]

        return GBT_A, GBT_B

    def reconstruct(self, c_t, B):
        # vals is the evaluation points for the basis functions
        vals = np.linspace(0.0, 1.0, self.max_length)
        eval_matrix = jax.lax.stop_gradient(
            (
                jnp.asarray(
                    ss.eval_legendre(
                        np.expand_dims(np.arange(B.shape[0]), -1), 2 * vals - 1
                    )
                )
                * (B)
            ).T
        ).astype(self.dtype)
        y = self.weight_basis(c_t, eval_matrix).astype(self.dtype)

    def weight_basis(
        self,
        c: Union[
            Float[Array, "channel N"],
            Float[Array, "t channel N"],
        ],
        eval_matrix: Float[Array, "seq_len N channels"],
    ) -> Union[
        Float[Array, "t channel"],
        Float[Array, "time seq_len channel"],
    ]:
        """reconstructs the input sequence from the estimated coefficients and the evaluation matrix

        Args:
            c (jnp.ndarray):
                shape: (batch size, channel, N)
                Vector of the estimated coefficients, given the history of the function/sequence

        Returns:
            y (jnp.ndarray):
                shape: (batch size, channel, input size)
                The reconstructed input sequence
        """

        y = None
        if len(c.shape) == 2:
            c = einops.rearrange(c, "channel N -> ... N channel")
            y = jnp.dot(eval_matrix, c)
        elif len(c.shape) == 3:
            c = einops.rearrange(c, "channel N -> ... N channel")
            y = jax.vmap(jnp.dot, in_axes=(None, 0))(eval_matrix, c)
        else:
            raise ValueError(
                "c must be of shape (batch size, channel, N) or (batch seq_len channel N)"
            )

        return y.astype(self.dtype)

    @staticmethod
    def initialize_state(
        rng,
        batch_size: int,
        channels: int,
        hidden_size: int,
        init_fn=nn.initializers.zeros,
    ):
        mem_shape = (batch_size, channels, hidden_size)
        return init_fn(rng, mem_shape)


# ------------------------------------------------------------------------------------
# ----------------------------------- BatchedHiPPOLSICell ----------------------------
# ------------------------------------------------------------------------------------


class BatchedHiPPOLSICell(HiPPOLSICell):

    max_length: int
    alpha: float = 0.5
    init_t: int = 0
    recon: bool = True
    A_init: Callable = legs
    B_init: Callable = legs
    dtype: Any = jnp.float32

    def setup(self) -> None:
        self.hippo = nn.vmap(
            target=HiPPOLSICell,
            in_axes=(0, 0, None),
            variable_axes={"params": 0},
            split_rngs={"params": True},
        )(
            max_length=self.max_length,
            alpha=self.alpha,
            init_t=self.init_t,
            recon=self.recon,
            A_init=self.A_init,
            B_init=self.B_init,
            dtype=self.dtype,
        )

    def __call__(
        self,
        c_t_1: Float[Array, "channel N"],
        x: Float[Array, "t channel"],
        t_step: int = 0,
    ) -> Tuple[
        Float[Array, "channel N"],
        Union[
            Tuple[Float[Array, "channel N"], Float[Array, "channel N"]],
            Tuple[Float[Array, "channel N"], Float[Array, "t channel"]],
        ],
    ]:
        return self.hippo(c_t_1, x, t_step)


# ------------------------------------------------------------------------------------
# --------------------------------------- HiPPOLTICell -------------------------------
# ------------------------------------------------------------------------------------


class HiPPOLTICell(HiPPOCell):

    step_size: float
    basis_size: float
    alpha: float = 0.5
    recon: bool = True
    A_init: Callable = legs
    B_init: Callable = legs
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self,
        c_t_1: Float[Array, "channel N"],
        x: Float[Array, "channel"],
    ) -> Tuple[
        Float[Array, "channel N"],
        Union[
            Tuple[Float[Array, "channel N"], Float[Array, "channel N"]],
            Tuple[Float[Array, "channel N"], Float[Array, "t channel"]],
        ],
    ]:
        hidden_features = c_t_1.shape[-1]
        channels = c_t_1.shape[-2]

        A, _, _ = self.A_init(N=hidden_features, dtype=self.dtype)
        _, B, method = self.B_init(N=hidden_features, dtype=self.dtype)

        # make channel number of B matrices to account for making channel number of HiPPOs. HiPPO only works on a 1-D signal
        _B = jnp.tile(B, (channels,))

        _A_d, _B_d = jax.lax.stop_gradient(
            self.discretize(
                A, _B, step=self.step_size, alpha=self.alpha, dtype=self.dtype
            )
        )

        # initialize the A and B matrices as functions so we can make them trainable parameters (if we want)
        A_init_fn = initializer(_A_d)
        B_init_fn = initializer(_B_d)

        # make the A and B matrices trainable parameters
        A_d = self.param("A_d", A_init_fn, (hidden_features, hidden_features))
        B_d = self.param("B_d", B_init_fn, (hidden_features, channels))

        # formulate the HiPPO operator
        c_t = jnp.einsum("mn, cn -> cn", A_d, c_t_1) + jnp.einsum("nc, c -> cn", B_d, x)

        # reconstruct the input sequence from the estimated coefficients and the evaluation matrix
        if self.recon:
            y = self.reconstruct(c_t, hidden_features, method)
            return c_t, (c_t, y)
        else:
            return c_t, (c_t, c_t)

    def discretize(
        self,
        A: Float[Array, "N N"],
        B: Float[Array, "N channel"],
        step: float,
        alpha: Union[float, str] = 0.5,
        dtype: Any = jnp.float32,
    ) -> Tuple[Float[Array, "N N"], Float[Array, "N channel"]]:
        """
        Discretizes the HiPPO A and B matrices.

        This function uses the Generalized Bilinear Transformation (GBT) or Zero-Order Hold method to discretize the input matrices A and B. The method used depends on the value of the alpha parameter.

        Args:
            A (jnp.ndarray): The matrix to be discretized. Must have shape (N, N).
            B (jnp.ndarray): The matrix to be discretized. Must have shape (N, channel).
            step (float): The step size used for discretization.
            alpha (float or str, optional): Determines which generalized bilinear transformation to use. Defaults to 0.5.
                - Forward Euler corresponds to α = 0.
                - Backward Euler corresponds to α = 1.
                - Bilinear corresponds to α = 0.5.
                - Zero-order Hold corresponds to α > 1.
            dtype (jnp.float): The type of float precision to be used. Defaults to jnp.float32.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing the discretized A and B matrices. Both matrices have the same shape as the input matrices.

        Raises:
            AssertionError: If alpha is not in [0, 0.5, 1, 2].

        Notes:
            The GBT method is used when alpha <= 1, and the Zero-Order Hold method is used when alpha > 1. The Zero-Order Hold method uses a trick to compute Ad and Bd in one step, as described in the Wikipedia article on Discretization.
        """
        assert alpha in [0, 0.5, 1, 2], "alpha must be 0, 0.5, 1, 2"

        I = jnp.eye(A.shape[0], dtype=dtype)

        # Generalized Bilinear Transformation
        # referencing equation 13 within the discretization method section of the HiPPO paper
        if alpha <= 1:
            step_size = step
            part1 = I - (step_size * alpha * A)
            part2 = I + (step_size * (1 - alpha) * A)

            GBT_A = jnp.linalg.solve(part1, part2)
            GBT_B = jnp.linalg.solve(part1, step_size * B)

            # GBT_A = jnp.linalg.lstsq(part1, part2, rcond=None)[0]
            # GBT_B = jnp.linalg.lstsq(part1, (step_size * B), rcond=None)[0]

        # Zero-Order Hold
        else:
            # refer to this for why this works
            # https://en.wikipedia.org/wiki/Discretization#:~:text=A%20clever%20trick%20to%20compute%20Ad%20and%20Bd%20in%20one%20step%20is%20by%20utilizing%20the%20following%20property

            n = A.shape[0]
            b_n = B.shape[1]
            A_B_square = jnp.block(
                [[A, B], [jnp.zeros((b_n, n)), jnp.zeros((b_n, b_n))]]
            )
            A_B = jax.scipy.linalg.expm(A_B_square * self.step_size)

            if A_B.dtype != dtype:
                A_B = A_B.astype(dtype)

            GBT_A = A_B[0:n, 0:n]
            GBT_B = A_B[0:-b_n, -b_n:]

        return GBT_A, GBT_B

    def reconstruct(self, c_t, hidden_features, method):
        assert method in [
            "legs",
            "legt",
            "lmu",
            "lagt",
            "fout",
        ], "reconstruction is only implemented for legs, legt, lmu, lagt, and fout methods"
        # vals is the evaluation points for the basis functions
        vals = np.arange(0.0, self.basis_size, self.step_size)

        # eval_matrix is the matrix of basis functions evaluated at the evaluation points
        eval_matrix = jax.lax.stop_gradient(
            self.basis(
                method=method,
                N=hidden_features,
                vals=vals,
                c=0.0,
                dtype=self.dtype,
            )  # (T/dt, N)
        )
        # reconstruct by applying the coefficients to the evaluation matrix thus giving us the reconstructed input sequence
        y = jax.lax.stop_gradient(
            self.weight_basis(
                c=c_t,
                method=method,
                hidden_size=hidden_features,
                eval_matrix=eval_matrix,
            )
        )
        return y.astype(self.dtype)

    def weight_basis(
        self,
        c: Float[Array, "channel N"],
        method: str,
        hidden_size: int,
        evals=None,
        eval_matrix=None,
    ) -> Float[Array, "t channel"]:
        """reconstructs the input sequence from the estimated coefficients and the evaluation matrix

        Args:
            c (jnp.ndarray):
                shape: (batch size, channel, N)
                Vector of the estimated coefficients, given the history of the function/sequence

            evals (jnp.ndarray, optional):
                shape: ()
                Vector of the evaluation points. Defaults to None.

        Returns:
            y (jnp.ndarray):
                shape: (batch size, channel, input size)
                The reconstructed input sequence
        """
        if evals is not None and eval_matrix is None:
            eval_matrix = self.basis(method=method, N=hidden_size, vals=evals)

        y = jnp.einsum("cn,ln->lc", c, eval_matrix)
        if not method == "lagt":
            y = einops.rearrange(y, "t 1 -> t")
            y = jnp.flip(y, 0)
            y = einops.rearrange(y, "t -> t 1")

        return y.astype(self.dtype)

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
                shape: (N, channel)
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

    @staticmethod
    def initialize_state(
        rng,
        batch_size: int,
        channels: int,
        hidden_size: int,
        init_fn=nn.initializers.zeros,
    ):
        mem_shape = (batch_size, channels, hidden_size)
        return init_fn(rng, mem_shape)


# ------------------------------------------------------------------------------------
# ----------------------------------- BatchedHiPPOLTICell ----------------------------
# ------------------------------------------------------------------------------------


class BatchedHiPPOLTICell(HiPPOLTICell):

    step_size: float
    basis_size: float
    alpha: float = 0.5
    recon: bool = True
    A_init: Callable = legs
    B_init: Callable = legs
    dtype: Any = jnp.float32

    def setup(self) -> None:
        self.hippo = nn.vmap(
            target=HiPPOLTICell,
            in_axes=(0, 0),
            variable_axes={"params": 0},
            split_rngs={"params": True},
        )(
            step_size=self.step_size,
            basis_size=self.basis_size,
            alpha=self.alpha,
            recon=self.recon,
            A_init=self.A_init,
            B_init=self.B_init,
            dtype=self.dtype,
        )

    def __call__(
        self,
        c_t_1: Float[Array, "channel N"],
        x: Float[Array, "t channel"],
    ) -> Tuple[
        Float[Array, "channel N"],
        Union[
            Tuple[Float[Array, "channel N"], Float[Array, "channel N"]],
            Tuple[Float[Array, "channel N"], Float[Array, "t channel"]],
        ],
    ]:
        return self.hippo(c_t_1, x)
    
# ------------------------------------------------------------------------------------
# --------------------------------------- HiPPO --------------------------------------
# ------------------------------------------------------------------------------------


class HiPPO(Model):
    """

    Examples:
            HiPPOLSICell, in the format of:
                {max_length: int
                 alpha: Optional[float], # alpha value for discretization
                 measure: Optional[str],
                 recon: Optional[bool],
                 dtype: Optional[jnp.dtype]
                }
            >>> {max_length=1024, alpha=0.0, recon=True, dtype=jnp.float16} # produces HiPPOLSICell w/ forward euler discretization, and reconstruction
            >>> {max_length=512, alpha=1.0, recon=False, dtype=jnp.float32} # produces HiPPOLSICell w/ backward euler discretization, and no reconstruction
            >>> {max_length=256, alpha=0.5, recon=True, dtype=jnp.float32} # produces HiPPOLSICell w/ bilinear transform discretization, and reconstruction
            >>> {max_length=512, alpha=2.0, recon=True, dtype=jnp.float32} # produces HiPPOLSICell w/ zero-order hold discretization, and reconstruction

            HiPPOLTICell, in the format of:
                (step_size: float, # 1 / sequence length
                 basis_size: float, # The intended maximum value of the basis function for the coefficients to be projected onto
                 alpha: Optional[float], # alpha value for discretization
                 recon: Optional[bool],
                 measure: Optional[str],
                 dtype: Optional[jnp.dtype]
                )
            >>> {step_size:1e-3, basis_size:1.0, alpha:0.0, recon:True, dtype:jnp.float16} # produces HiPPOLTICell w/ forward euler discretization, and reconstruction, discretized every 1/1000 assuming a sequence length is 1000
            >>> {step_size:1e-4, basis_size:1.0, alpha:1.0, recon:True, dtype:jnp.float32} # produces HiPPOLTICell w/ backward euler discretization, and reconstruction, discretized every 1/10000 assuming a sequence length is 10000
            >>> {step_size:1e-2, basis_size:1.0, alpha:2.0, recon:True, dtype:jnp.float64} # produces HiPPOLTICell w/ zero-order hold discretization, and reconstruction, discretized every 1/10000 assuming a sequence length is 100
            >>> {step_size:1e-2, basis_size:1.0, alpha:0.5, recon:True, dtype:jnp.float16} # produces HiPPOLTICell w/ bilinear transform discretization, and reconstruction, discretized every 1/10000 assuming a sequence length is 100



    Args:
        features (int):
            The size of the hidden state of the HiPPO model

        hippo_cell (HiPPOCell):
            The HiPPOCell class to be used for the HiPPO model

        hippo_args (dict):
            The dict associated with the input parameters into the HiPPOCell class.

        init_t (int):
            The initial time step for the HiPPO model. Defaults to 0

        unroll (bool):
            Determines if you wanted the full history (all time steps) of coefficients, and potentially reconstructions. Defaults to False

        st (bool):
            Determines if you want to use the scale invariant verion of HiPPO or the time invariant version of HiPPO. Defaults to time invariant via False

    Raises:
        ValueError: Enforces that the inputted cell is a HiPPOCell
    """

    hippo_cell: HiPPOCell
    hippo_args: Dict
    mlp_args: Dict = None
    init_t: int = 0
    unroll: bool = False
    st: bool = False
    train: bool = False

    def setup(self) -> None:

        if self.st:
            bhippo_t = nn.vmap(
                target=self.hippo_cell,
                in_axes=(0, 0, None),
                variable_axes={"params": 0},
                split_rngs={"params": True},
            )
            self.hippo = nn.scan(
                target=bhippo_t,
                in_axes=(1, 0),
                out_axes=1,
                variable_broadcast="params",
                split_rngs={"params": False},
            )(**self.hippo_args)

        else:
            self.hippo = nn.scan(
                target=self.hippo_cell,
                in_axes=1,
                out_axes=1,
                variable_broadcast="params",
                split_rngs={"params": False},
            )(**self.hippo_args)

        if self.mlp_args is not None:
            self.mlp = BatchedMLP(**self.mlp_args)

    def __call__(
        self,
        f: Float[Array, "t channel"],
        c_t_1: Float[Array, "channel N"],
    ) -> Tuple[
        Union[
            Float[Array, "t channel N"],
            Float[Array, "channel N"],
        ],
        Union[
            Float[Array, "channel N"],
            Float[Array, "t channel N"],
        ],
    ]:
        c_n = None
        c_s = None
        y_s = None
        if isinstance(self.hippo, (HiPPOLTICell)):
            if self.train:
                c_n, (c_s, y_s) = self.hippo(c_t_1, f)
            else:
                c_n, (c_s, y_s) = jax.lax.stop_gradient(self.hippo(c_t_1, f))
        elif isinstance(self.hippo, HiPPOLSICell):
            t = jnp.arange(self.init_t, f.shape[1])
            if self.train:
                c_n, (c_s, y_s) = self.hippo(c_t_1, f, t)
            else:
                c_n, (c_s, y_s) = jax.lax.stop_gradient(self.hippo(c_t_1, f, t))
        else:
            NotImplementedError(
                f"Only HiPPOLSICell and HiPPOLTICell are supported, got {self.hippo.__class__.__name__}"
            )

        if self.unroll:
            if self.mlp_args is None:
                return c_s, y_s, c_n
            else:
                output_list = []
                for i in range(c_s.shape[2]):
                    output = self.mlp(c_s[:, :, -1, :])
                    output_list.append(output)
                return c_s, y_s, jnp.stack(output_list, axis=1)
        else:
            if self.mlp_args is None:
                return c_s, y_s[:, :, -1, :], c_n
            else:
                return c_s, y_s[:, :, -1, :], self.mlp(c_n)

    @staticmethod
    def initialize_state(
        rng,
        batch_size: int,
        channels: int,
        hidden_size: int,
        init_fn=nn.initializers.zeros,
    ):
        mem_shape = (batch_size, channels, hidden_size)
        return init_fn(rng, mem_shape)
