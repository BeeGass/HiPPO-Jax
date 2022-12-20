## import packages
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.numpy.linalg import inv
from scipy import special as ss
from src.models.hippo.transition import TransMatrix
from typing import Any
from functools import partial


class HiPPO(nn.Module):
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

    max_length: int
    step_size: float = 1.0  # < 1.0 if you want to use LTI discretization
    N: int = 100
    lambda_n: float = 1.0
    alpha: float = 0.0
    beta: float = 1.0
    GBT_alpha: float = 0.5
    measure: str = "legs"
    s_t: str = "lti"
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

        self.C = jnp.ones((self.N, 1))
        self.D = jnp.zeros((1,))

        if self.step_size == 1.0:
            self.GBT_A_list, self.GBT_B_list = self.make_GBT_list(
                matrices.A, matrices.B, dtype=self.dtype
            )

        self.eval_matrix = self.create_eval_matrix(matrices.A, matrices.B)

    def __call__(self, f, init_state=None, kernel=False):
        if not kernel:
            if init_state is None:
                # init_state = jnp.zeros((f.shape[0], self.N, 1))
                init_state = jnp.zeros((f.shape[0], 1, self.N))

            if self.s_t == "lsi":
                c_k, y_k = self.lsi_recurrence(
                    A=self.GBT_A_list,
                    B=self.GBT_B_list,
                    C=self.C,
                    D=self.D,
                    c_0=init_state,
                    f=f,
                    alpha=self.GBT_alpha,
                    dtype=self.dtype,
                )
                c_k = jnp.stack(c_k, axis=0)
                y_k = jnp.stack(y_k, axis=0)

            elif self.s_t == "lti":
                c_k, y_k = self.lti_recurrence(
                    A=self.A,
                    B=self.B,
                    C=self.C,
                    D=self.D,
                    c_0=init_state,
                    f=f,
                    alpha=self.GBT_alpha,
                    dtype=self.dtype,
                )
            else:
                raise ValueError(
                    f"Incorrect value associated with invariance options, either pick 'lsi' or 'lti'."
                )

        else:
            Ab, Bb, Cb, Db = self.discretize(
                self.A,
                self.B,
                self.C,
                self.D,
                step=self.step_size,
                alpha=self.GBT_alpha,
            )
            c_k, y_k = self.causal_convolution(
                f, self.K_conv(Ab, Bb, Cb, Db, L=self.max_length)
            )

        return c_k, y_k

    def reconstruct(self, c):
        """
        Uses coeffecients to reconstruct the signal

        Args:
            c (jnp.ndarray): coefficients of the HiPPO projection

        Returns:
            reconstructed signal
        """
        return (self.eval_matrix @ jnp.expand_dims(c, -1)).squeeze(-1)

    def make_GBT_list(self, A, B, dtype=jnp.float32):
        """
        Creates the discretized GBT matrices for the given step size
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

    def create_eval_matrix(self, A, B):
        """
        Creates the evaluation matrix used for reconstructing the signal
        """
        eval_matrix = None
        if self.measure == "legt":
            L = self.max_length
            vals = jnp.arange(0.0, 1.0, L)
            # n = jnp.arange(self.N)[:, None]
            zero_N = self.N - 1
            x = 1 - 2 * vals
            eval_matrix = jax.scipy.special.lpmn_values(
                m=zero_N, n=zero_N, z=x, is_normalized=False
            ).T  # ss.eval_legendre(n, x).T

        elif self.measure == "legs":
            L = self.max_length
            vals = jnp.linspace(0.0, 1.0, L)
            # n = jnp.arange(self.N)[:, None]
            zero_N = self.N - 1
            x = 2 * vals - 1
            eval_matrix = (
                B[:, None]
                * jax.scipy.special.lpmn_values(
                    m=zero_N, n=zero_N, z=x, is_normalized=False
                )
            ).T  # ss.eval_legendre(n, x)).T

        elif self.measure == "lagt":
            raise NotImplementedError("Translated Laguerre measure not implemented yet")

        elif self.measure == "fourier":
            raise NotImplementedError("Fourier measures are not implemented yet")

        else:
            raise ValueError("invalid measure")

        return eval_matrix

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
        I = jnp.eye(A.shape[0])
        step_size = 1 / step
        part1 = I - (step_size * alpha * A)
        part2 = I + (step_size * (1 - alpha) * A)

        GBT_A = jnp.linalg.lstsq(part1, part2, rcond=None)[0]

        GBT_B = jnp.linalg.lstsq(part1, (step_size * B), rcond=None)[0]

        if alpha > 1:  # Zero-order Hold
            GBT_A = jax.scipy.linalg.expm(step_size * A)
            GBT_B = jnp.linalg.inv(A) @ (jax.scipy.linalg.expm(step_size * A) - I) @ B

        return GBT_A.astype(dtype), GBT_B.astype(dtype)

    def lsi_recurrence(self, A, B, C, D, c_0, f, dtype=jnp.float32):
        """
        This is for returning the discretized hidden state often needed for an RNN.
        Args:
            A (jnp.ndarray):
                shape: (N, N)
                the discretized A matrix

            B (jnp.ndarray):
                shape: (N, 1)
                the discretized B matrix

            C (jnp.ndarray):
                shape: (N, 1)
                the discretized C matrix

            c_0 (jnp.ndarray):
                shape: (batch size, input length, N)
                the initial hidden state

            f (jnp.ndarray):
                shape: (sequence length, 1)
                the input sequence


        Returns:
            the next hidden state (aka coefficients representing the function, f(t))
        """

        c_k_list = []
        y_k_list = []

        c_k = c_0.copy()
        for i in range(f.shape[1]):
            c_k, y_k = jax.vmap(self.lsi_step, in_axes=(None, None, None, None, 0, 0))(
                A[i], B[i], C, D, c_k, f[:, i, :]
            )
            c_k_list.append((c_k.copy()).astype(dtype))
            y_k_list.append((y_k.copy()).astype(dtype))

        if self.verbose:
            return c_k_list, y_k_list
        else:
            return c_k_list[-1], y_k_list[-1]

    def lti_recurrence(self, A, B, C, D, c_0, f, alpha=0.5, dtype=jnp.float32):
        """
        This is for returning the discretized hidden state often needed for an RNN.
        Args:
            A (jnp.ndarray):
                shape: (N, N)
                the discretized A matrix

            B (jnp.ndarray):
                shape: (N, 1)
                the discretized B matrix

            C (jnp.ndarray):
                shape: (N, 1)
                the discretized C matrix

            D (jnp.ndarray):
                shape: (N, 1)
                the discretized C matrix

            f (jnp.ndarray):
                shape: (sequence length, 1)
                the input sequence

            c_0 (jnp.ndarray):
                shape: (batch size, input length, N)
                the initial hidden state

        Returns:
            the next hidden state (aka coefficients representing the function, f(t))
        """
        Ad, Bd = self.discretize(
            A=A, B=B, step=self.step_size, alpha=alpha, dtype=dtype
        )

        def lti_step(c_k_i, f_k):
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

            c_k = jnp.dot(c_k_i, Ad.T) + (Bd.T * f_k)
            y_k = jnp.dot(C, c_k) + (D * f_k)

            return c_k, (c_k, y_k)

        c_k, (c_s, y_s) = jax.vmap(jax.lax.scan, in_axes=(None, 0, 0))(lti_step, c_0, f)

        if self.verbose:
            return c_s, y_s
        else:
            return c_k, y_s

    def lsi_step(self, Ad, Bd, Cd, Dd, c_k_i, f_k):
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

        c_k = jnp.dot(c_k_i, Ad.T) + (Bd.T * f_k)
        y_k = jnp.dot(Cd, c_k) + (Dd * f_k)

        return c_k, y_k


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
