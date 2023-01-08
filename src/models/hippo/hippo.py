## import packages
import math
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.numpy.linalg import inv
from scipy import special as ss

from src.models.hippo.transition import TransMatrix


class HiPPOLSI(nn.Module):
    """
    class that constructs HiPPO model using the defined measure.

    Args:

        max_length (int):
            maximum sequence length to be input

        step_size (float):
            step size used for descretization

        N (int):
            order of the HiPPO projection, aka the number of coefficients to describe the matrix

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

        s_t (str):
            choice between LSI and LTI systems
            - "lsi"
            - "lti"

        dtype (jnp.float):
            represents the float precision of the class

        verbose (bool):
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
    verbose: bool = False

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

        self.A = matrices.A
        self.B = matrices.B

        vals = jnp.linspace(0.0, 1.0, self.max_length)
        self.eval_matrix = (
            (matrices.B)[:, None]
            * ss.eval_legendre(jnp.arange(self.N)[:, None], 2 * vals - 1)
        ).T

    def __call__(self, f, init_state=None, kernel=False):
        if init_state is None:
            init_state = jnp.zeros((f.shape[0], 1, self.N))

        c_k = self.recurrence(
            A=self.GBT_A_list,
            B=self.GBT_B_list,
            c_0=init_state,
            f=f,
            dtype=self.dtype,
        )
        c_k = jnp.stack(c_k, axis=0)

        return c_k

    def temporal_GBT(self, A, B, dtype=jnp.float32):
        """
        Creates the list of discretized GBT matrices for the given step size
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

    def discretize(self, A, B, step, alpha=0.5, dtype=jnp.float32):
        """
        function used for discretizing the HiPPO matrix

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
        """
        if alpha <= 1:
            assert (
                alpha == 0.0 or alpha == 0.5 or alpha == 1.0
            ), "alpha must be 0, 0.5, or 1"
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
            if self.s_t == "lsi":
                A_B = jax.scipy.linalg.expm(
                    A_B_square * (math.log(step + self.step_size) - math.log(step))
                )
            else:
                A_B = jax.scipy.linalg.expm(A_B_square * self.step_size)

            GBT_A = A_B[0:n, 0:n]
            GBT_B = A_B[0:-b_n, -b_n:]

        return GBT_A.astype(dtype), GBT_B.astype(dtype)

    def recurrence(self, A, B, c_0, f, dtype=jnp.float32):
        """
        This is for returning the discretized hidden state often needed for an RNN.
        Args:
            A (jnp.ndarray):
                shape: (N, N)
                the discretized A matrix

            B (jnp.ndarray):
                shape: (N, 1)
                the discretized B matrix

            c_0 (jnp.ndarray):
                shape: (batch size, input length, N)
                the initial hidden state

            f (jnp.ndarray):
                shape: (sequence length, 1)
                the input sequence


        Returns:
            the next hidden state (aka coefficients representing the function, f(t))
        """

        c_s = []

        c_k = c_0.copy()
        for i in range(f.shape[1]):
            c_k = jax.vmap(self.step, in_axes=(None, None, None, None, 0, 0))(
                A[i], B[i], c_k, f[:, i, :]
            )
            c_s.append((c_k.copy()).astype(dtype))

        if self.verbose:
            return c_s  # list of hidden states
        else:
            return c_s[-1]  # last hidden state

    def step(self, Ad, Bd, c_k_i, f_k):
        """
        Get descretized coefficients of the hidden state by applying HiPPO matrix to input sequence, u_k, and previous hidden state, x_k_1.
        Args:
            c_k_i:
                shape: (input length, N)
                previous hidden state

            f_k:
                shape: (1, )
                output from function f at, descritized, time step, k.

        Returns:
            c_k: current hidden state
            y_k: current output of hidden state applied to Cb (sorry for being vague, I just dont know yet)
        """

        c_k = (jnp.dot(c_k_i, Ad.T)) + (Bd.T * f_k)

        return c_k

    def reconstruct(self, c):
        y = self.eval_matrix @ c

        return y


class HiPPOLTI(nn.Module):
    """
    class that constructs HiPPO model using the defined measure.

    Args:

        max_length (int):
            maximum sequence length to be input

        step_size (float):
            step size used for descretization

        N (int):
            order of the HiPPO projection, aka the number of coefficients to describe the matrix

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

        s_t (str):
            choice between LSI and LTI systems
            - "lsi"
            - "lti"

        dtype (jnp.float):
            represents the float precision of the class

        verbose (bool):
            shows the rolled out coefficients over time/scale

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
    verbose: bool = False

    def setup(self):
        matrices = TransMatrix(
            N=self.N,
            measure=self.measure,
            lambda_n=self.lambda_n,
            alpha=self.alpha,
            beta=self.beta,
            dtype=self.dtype,
        )

        self.A = matrices.A
        self.B = matrices.B

        self.vals = jnp.arange(0.0, self.basis_size, self.step_size)
        self.eval_matrix = self.basis(
            self.method, self.N, self.vals, c=0.0
        )  # (T/dt, N)

    def __call__(self, f, init_state=None):
        if init_state is None:
            init_state = jnp.zeros((f.shape[0], 1, self.N))

        c_k = self.recurrence(
            A=self.A,
            B=self.B,
            c_0=init_state,
            f=f,
            alpha=self.GBT_alpha,
            step_size=self.step_size,
            dtype=self.dtype,
        )

        return c_k

    def discretize(self, A, B, step, alpha=0.5, dtype=jnp.float32):
        """
        function used for discretizing the HiPPO matrix

        Args:
            A (jnp.ndarray):
                shape: (N, N)
                matrix to be discretized

            B (jnp.ndarray):
                shape: (N, 1)
                matrix to be discretized

            C (jnp.ndarray):
                shape: (N, 1)
                matrix to be discretized

            D (jnp.ndarray):
                shape: (1,)
                matrix to be discretized

            step (float):
                step size used for discretization

            alpha (float, optional):
                used for determining which generalized bilinear transformation to use
                - forward Euler corresponds to α = 0,
                - backward Euler corresponds to α = 1,
                - bilinear corresponds to α = 0.5,
                - Zero-order Hold corresponds to α > 1
        """
        if alpha <= 1:
            assert (
                alpha == 0.0 or alpha == 0.5 or alpha == 1.0
            ), "alpha must be 0, 0.5, or 1"
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
            if self.s_t == "lsi":
                A_B = jax.scipy.linalg.expm(
                    A_B_square * (math.log(step + self.step_size) - math.log(step))
                )
            else:
                A_B = jax.scipy.linalg.expm(A_B_square * self.step_size)

            GBT_A = A_B[0:n, 0:n]
            GBT_B = A_B[0:-b_n, -b_n:]

        return GBT_A.astype(dtype), GBT_B.astype(dtype)

    def recurrence(self, A, B, c_0, f, alpha=0.5, step_size=1.0, dtype=jnp.float32):
        """
        This is for returning the discretized hidden state often needed for an RNN.
        Args:
            A (jnp.ndarray):
                shape: (N, N)
                the discretized A matrix

            B (jnp.ndarray):
                shape: (N, 1)
                the discretized B matrix

            f (jnp.ndarray):
                shape: (sequence length, 1)
                the input sequence

            c_0 (jnp.ndarray):
                shape: (batch size, input length, N)
                the initial hidden state

        Returns:
            the next hidden state (aka coefficients representing the function, f(t))
        """
        Ad, Bd = self.discretize(A=A, B=B, step=step_size, alpha=alpha, dtype=dtype)

        def step(c_k_i, f_k):
            """
            Get descretized coefficients of the hidden state by applying HiPPO matrix to input sequence, u_k, and previous hidden state, x_k_1.
            Args:
                c_k_i:
                    shape: (input length, N)
                    previous hidden state

                f_k:
                    shape: (1, )
                    output from function f at, descritized, time step, k.

            Returns:
                c_k: current hidden state
                y_k: current output of hidden state applied to Cb (sorry for being vague, I just dont know yet)
            """

            c_k = (jnp.dot(c_k_i, Ad.T)) + (Bd.T * f_k)

            return c_k, c_k

        c_k, c_s = jax.vmap(jax.lax.scan, in_axes=(None, 0, 0))(step, c_0, f)

        if self.verbose:
            return c_s
        else:
            return c_k

    def measure_fn(self, method, c=0.0):
        if method == "legt":
            fn = lambda x: jnp.heaviside(x, 0.0) * jnp.heaviside(1.0 - x, 0.0)
        elif method == "legs":
            fn = lambda x: jnp.heaviside(x, 1.0) * jnp.exp(-x)
        elif method == "lagt":
            fn = lambda x: jnp.heaviside(x, 1.0) * jnp.exp(-x)
        elif method in ["fourier"]:
            fn = lambda x: jnp.heaviside(x, 1.0) * jnp.heaviside(1.0 - x, 1.0)
        else:
            raise NotImplementedError

        fn_tilted = lambda x: jnp.exp(c * x) * fn(x)

        return fn_tilted

    def basis(self, method, N, vals, c=0.0, truncate_measure=True):
        """
        vals: list of times (forward in time)
        returns: shape (T, N) where T is length of vals
        """
        eval_matrix = None
        if method in ["legt", "lmu"]:
            eval_matrix = ss.eval_legendre(jnp.arange(N)[:, None], 2 * vals - 1).T
            eval_matrix *= (2 * jnp.arange(N) + 1) ** 0.5 * (-1) ** jnp.arange(N)

        elif method == "legs":
            _vals = jnp.exp(-vals)
            eval_matrix = ss.eval_legendre(
                jnp.arange(N)[:, None], 1 - 2 * _vals
            ).T  # (L, N)
            eval_matrix *= (2 * jnp.arange(N) + 1) ** 0.5 * (-1) ** jnp.arange(N)

        elif method == "lagt":
            vals = vals[::-1]
            eval_matrix = ss.eval_genlaguerre(np.arange(N)[:, None], 0, vals)
            eval_matrix = eval_matrix * jnp.exp(-vals / 2)
            eval_matrix = eval_matrix.T

        elif method in ["fourier", "fru", "fout", "foud"]:
            cos = 2**0.5 * jnp.cos(
                2 * jnp.pi * jnp.arange(N // 2)[:, None] * (vals)
            )  # (N/2, T/dt)
            sin = 2**0.5 * jnp.sin(
                2 * jnp.pi * jnp.arange(N // 2)[:, None] * (vals)
            )  # (N/2, T/dt)
            cos[0] /= 2**0.5
            eval_matrix = jnp.stack([cos.T, sin.T], dim=-1).reshape(-1, N)  # (T/dt, N)
        #     print("eval_matrix shape", eval_matrix.shape)

        if truncate_measure:
            eval_matrix[self.measure_fn(method)(vals) == 0.0] = 0.0

        p = eval_matrix * jnp.exp(-c * vals)[:, None]  # [::-1, None]

    def reconstruct(
        self, c, evals=None
    ):  # TODO take in a times array for reconstruction
        """
        c: (..., N,) HiPPO coefficients (same as x(t) in S4 notation)
        output: (..., L,)
        """
        if evals is not None:
            eval_matrix = self.basis(self.measure, self.N, evals)
        else:
            eval_matrix = self.eval_matrix

        y = eval_matrix @ c

        return y


class HiPPO(nn.Module):

    N: int
    max_length: int = 1024
    step_size: float = 1.0
    basis_length: int = 1.0
    lambda_n: float = 1.0
    alpha: float = 0.0
    beta: float = 1.0
    GBT_alpha: float = 0.5
    measure: str = "legs"
    basis_size: float = 1.0
    s_t: str = "lti"
    truncate_measure: bool = True
    dtype: Any = jnp.float32
    verbose: bool = False

    def setup(self) -> None:

        # Define the encoder that performs the polynomial projections with user specified matrix initialization
        if self.s_t == "lsi":
            self.encoder = HiPPOLSI(
                N=self.N,
                max_length=self.max_length,
                step_size=self.step_size,
                lambda_n=self.lambda_n,
                alpha=self.alpha,
                beta=self.beta,
                GBT_alpha=self.GBT_alpha,
                measure=self.measure,
                dtype=self.dtype,
                verbose=self.verbose,
            )
        elif self.s_t == "lti":
            self.encoder = HiPPOLTI(
                N=self.N,
                step_size=self.step_size,
                lambda_n=self.lambda_n,
                alpha=self.alpha,
                beta=self.beta,
                GBT_alpha=self.GBT_alpha,
                measure=self.measure,
                basis_size=self.basis_size,
                dtype=self.dtype,
                verbose=self.verbose,
            )
        else:
            raise ValueError(
                f"s_t must be either 'lsi' or 'lti'. s_t is currently set to: {self.s_t}"
            )

    def __call__(self, x, init_state=None):

        # Apply the polynomial projections to the input
        hidden = self.encoder(x, init_state=init_state)

        # Decode the polynomial projections to the output space through applying the coefficients to the basis
        if self.s_t == "lti":
            output = self.encoder.reconstruct(c=hidden, evals=x)
        else:
            output = self.encoder.reconstruct(c=hidden)

        return hidden, output


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
