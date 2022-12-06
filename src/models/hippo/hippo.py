## import packages
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.numpy.linalg import inv
from scipy import special as ss


class HiPPO(nn.Module):
    """
    class that constructs HiPPO model using the defined measure.

    Args:
        N (int): order of the HiPPO projection, aka the number of coefficients to describe the matrix
        max_length (int): maximum sequence length to be input
        measure (str): the measure used to define which way to instantiate the HiPPO matrix
        step (float): step size used for descretization
        GBT_alpha (float): represents which descretization transformation to use based off the alpha value
        seq_L (int): length of the sequence to be used for training
        v (str): choice of vectorized or non-vectorized function instantiation
            - 'v': vectorized
            - 'nv': non-vectorized
        lambda_n (float): value associated with the tilt of legt
            - 1: tilt on legt
            - \sqrt(2n+1)(-1)^{N}: tilt associated with the legendre memory unit (LMU)
        fourier_type (str): choice of fourier measures
            - fru: fourier recurrent unit measure (FRU) - 'fru'
            - fout: truncated Fourier (FouT) - 'fout'
            - fourd: decaying fourier transform - 'fourd'
        alpha (float): The order of the Laguerre basis.
        beta (float): The scale of the Laguerre basis.
    """

    N: int
    max_length: int
    step: float
    GBT_alpha: float
    seq_L: int
    A: jnp.ndarray
    B: jnp.ndarray
    measure: str

    def setup(self):
        A = self.A
        B = self.B
        self.C = jnp.ones((self.N,))
        self.D = jnp.zeros((1,))

        self.basis(c=0.0, truncate_measure=True)

    def __call__(self, f, init_state=None, t_step=0, kernel=False):
        # print(f"u shape:\n{f.shape}")
        # print(f"u:\n{f}")
        if not kernel:
            if init_state is None:
                init_state = jnp.zeros((self.N, 1))

            # Ab, Bb, Cb, Db = self.collect_SSM_vars(
            #     self.A, self.B, self.C, self.D, f, t_step=t_step, alpha=self.GBT_alpha
            # )
            c_k, y_k, GBT_A, GBT_B = self.loop_SSM(
                A=self.A,
                B=self.B,
                C=self.C,
                D=self.D,
                c_0=init_state,
                f=f,
                alpha=self.GBT_alpha,
            )
            # c_k, y_k = self.scan_SSM(Ab=Ab, Bb=Bb, Cb=Cb, Db=Db, c_0=init_state, f=f)

        else:
            Ab, Bb, Cb, Db = self.discretize(
                self.A, self.B, self.C, self.D, step=self.step, alpha=self.GBT_alpha
            )
            c_k, y_k = self.causal_convolution(
                f, self.K_conv(Ab, Bb, Cb, Db, L=self.max_length)
            )

        return c_k, y_k, GBT_A, GBT_B

    def measure_fn(self, c=0.0):

        if self.measure == "legt":
            fn = lambda x: jnp.heaviside(x, 0.0) * jnp.heaviside(1.0 - x, 0.0)

        elif self.measure == "legs":
            fn = lambda x: jnp.heaviside(x, 1.0) * jnp.exp(-x)

        elif self.measure == "lagt":
            fn = lambda x: jnp.heaviside(x, 1.0) * jnp.exp(-x)

        elif self.measure in ["fourier"]:
            fn = lambda x: jnp.heaviside(x, 1.0) * jnp.heaviside(1.0 - x, 1.0)

        else:
            raise NotImplementedError

        fn_tilted = lambda x: jnp.exp(c * x) * fn(x)

        return fn_tilted

    def basis(self, c=0.0, truncate_measure=True):
        """
        vals: list of times (forward in time)
        returns: shape (T, N) where T is length of vals
        """
        L = self.max_length
        vals = jnp.linspace(0.0, 1.0, L)
        eval_matrix = None
        if self.measure == "legt":
            eval_matrix = jax.scipy.special.lpmn_values(
                m=(self.N - 1), n=(self.N - 1), z=(2 * _vals - 1), is_normalized=False
            ).T
            eval_matrix *= (2 * jnp.arange(self.N) + 1) ** 0.5 * (-1) ** jnp.arange(
                self.N
            )

        elif self.measure == "legs":
            _vals = jnp.exp(-vals)
            eval_matrix = jax.scipy.special.lpmn_values(
                m=(self.N - 1), n=(self.N - 1), z=(2 * _vals - 1), is_normalized=False
            ).T
            eval_matrix *= (2 * jnp.arange(self.N) + 1) ** 0.5 * (-1) ** jnp.arange(
                self.N
            )

        elif self.measure == "lagt":
            vals = vals[::-1]
            eval_matrix = ss.eval_genlaguerre(jnp.arange(self.N)[:, None], 0, vals)
            eval_matrix = eval_matrix * jnp.exp(-vals / 2)
            eval_matrix = eval_matrix.T

        elif self.measure == "fourier":
            cos = 2**0.5 * jnp.cos(
                2 * jnp.pi * jnp.arange(self.N // 2)[:, None] * (vals)
            )  # (N/2, T/dt)
            sin = 2**0.5 * jnp.sin(
                2 * jnp.pi * jnp.arange(self.N // 2)[:, None] * (vals)
            )  # (N/2, T/dt)
            cos[0] /= 2**0.5
            eval_matrix = jnp.stack([cos.T, sin.T], axis=-1).reshape(
                -1, self.N
            )  # (T/dt, N)
        #     print("eval_matrix shape", eval_matrix.shape)

        if truncate_measure:
            eval_matrix.at[self.measure_fn()(vals) == 0.0].set(0.0)  # TODO: fix this

        self.eval_matrix = eval_matrix

        p = eval_matrix

        self.p = p * jnp.exp(-c * vals)[:, None]  # [::-1, None]

    def reconstruct(self, c):
        """
        Uses coeffecients to reconstruct the signal

        Args:
            c (jnp.ndarray): coefficients of the HiPPO projection

        Returns:
            reconstructed signal
        """
        return (self.eval_matrix @ jnp.expand_dims(c, -1)).squeeze(-1)

    def discretize(self, A, B, C, D, step, alpha=0.5):
        """
        function used for discretizing the HiPPO matrix

        Args:
            A (jnp.ndarray): matrix to be discretized
            B (jnp.ndarray): matrix to be discretized
            C (jnp.ndarray): matrix to be discretized
            D (jnp.ndarray): matrix to be discretized
            step (float): step size used for discretization
            alpha (float, optional): used for determining which generalized bilinear transformation to use
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

        base_GBT_B = jnp.linalg.lstsq(part1, B, rcond=None)[0]
        GBT_B = step_size * base_GBT_B

        if alpha > 1:  # Zero-order Hold
            GBT_A = jax.scipy.linalg.expm(step_size * A)
            GBT_B = (jnp.linalg.inv(A) @ (jax.scipy.linalg.expm(step_size * A) - I)) @ B

        return (
            GBT_A.astype(jnp.float32),
            GBT_B.astype(jnp.float32),
            C.astype(jnp.float32),
            D.astype(jnp.float32),
        )

    def collect_SSM_vars(self, A, B, C, D, f, t_step=0, alpha=0.5):
        """
        turns the continuos HiPPO matrix components into discrete ones

        Args:
            A (jnp.ndarray): matrix to be discretized
            B (jnp.ndarray): matrix to be discretized
            C (jnp.ndarray): matrix to be discretized
            D (jnp.ndarray): matrix to be discretized
            f (jnp.ndarray): input signal
            alpha (float, optional): used for determining which generalized bilinear transformation to use

        Returns:
            Ab (jnp.ndarray): discrete form of the HiPPO matrix
            Bb (jnp.ndarray): discrete form of the HiPPO matrix
            Cb (jnp.ndarray): discrete form of the HiPPO matrix
            Db (jnp.ndarray): discrete form of the HiPPO matrix
        """
        N = A.shape[0]

        if t_step == 0:
            L = f.shape[0]  # seq_L, 1
            assert (
                L == self.seq_L
            ), f"sequence length must match, currently {L} != {self.seq_L}"
            assert N == self.N, f"Order number must match, currently {N} != {self.N}"
        else:
            L = t_step
            assert t_step >= 1, f"time step must be greater than 0, currently {t_step}"
            assert N == self.N, f"Order number must match, currently {N} != {self.N}"

        Ab, Bb, Cb, Db = self.discretize(A, B, C, D, step=L, alpha=alpha)

        return (
            Ab.astype(jnp.float32),
            Bb.astype(jnp.float32),
            Cb.astype(jnp.float32),
            Db.astype(jnp.float32),
        )

    def scan_SSM(self, Ad, Bd, Cd, Dd, c_0, f):
        """
        This is for returning the discretized hidden state often needed for an RNN.
        Args:
            Ab (jnp.ndarray): the discretized A matrix
            Bb (jnp.ndarray): the discretized B matrix
            Cb (jnp.ndarray): the discretized C matrix
            f (jnp.ndarray): the input sequence
            c_0 (jnp.ndarray): the initial hidden state
        Returns:
            the next hidden state (aka coefficients representing the function, f(t))
        """

        def step(c_k_1, f_k):
            """
            Get descretized coefficients of the hidden state by applying HiPPO matrix to input sequence, u_k, and previous hidden state, x_k_1.
            Args:
                c_k_1: previous hidden state
                f_k: output from function f at, descritized, time step, k.
                t:

            Returns:
                c_k: current hidden state
                y_k: current output of hidden state applied to Cb (sorry for being vague, I just dont know yet)
            """
            part1 = Ad @ c_k_1
            part2 = jnp.expand_dims((Bd @ f_k), -1)

            c_k = part1 + part2
            y_k = Cd @ c_k  # + (Db.T @ f_k)

            return c_k, y_k

        return jax.lax.scan(step, c_0, f)

    def loop_SSM(self, A, B, C, D, c_0, f, alpha=0.5):
        """
        This is for returning the discretized hidden state often needed for an RNN.
        Args:
            Ab (jnp.ndarray): the discretized A matrix
            Bb (jnp.ndarray): the discretized B matrix
            Cb (jnp.ndarray): the discretized C matrix
            f (jnp.ndarray): the input sequence
            c_0 (jnp.ndarray): the initial hidden state
        Returns:
            the next hidden state (aka coefficients representing the function, f(t))
        """
        GBT_A_lst = []
        GBT_B_lst = []
        c_k_list = []
        y_k_list = []

        c_k = c_0.copy()
        for i in range(1, f.shape[0] + 1):
            Ad_i, Bd_i, Cd_i, Dd_i = self.collect_SSM_vars(
                A=A, B=B, C=C, D=D, f=f, t_step=i, alpha=alpha
            )
            c_k, y_k = self.loop_step(
                Ad=Ad_i, Bd=Bd_i, Cd=Cd_i, Dd=Dd_i, c_k_i=c_k, f_k=f[i - 1][0]
            )
            c_k_list.append(c_k.copy())
            y_k_list.append(y_k.copy())
            GBT_A_lst.append(Ad_i.copy())
            GBT_B_lst.append(Bd_i.copy())

        return c_k_list, y_k_list, GBT_A_lst, GBT_B_lst

    def loop_step(self, Ad, Bd, Cd, Dd, c_k_i, f_k):
        """
        Get descretized coefficients of the hidden state by applying HiPPO matrix to input sequence, u_k, and previous hidden state, x_k_1.
        Args:
            c_k_i: previous hidden state
            f_k: output from function f at, descritized, time step, k.

        Returns:
            c_k: current hidden state
            y_k: current output of hidden state applied to Cb (sorry for being vague, I just dont know yet)
        """

        part1 = Ad @ c_k_i
        part2 = Bd * f_k
        c_k = part1 + part2
        y_k = Cd @ c_k  # + (Db.T @ f_k)

        return c_k.astype(jnp.float32), y_k.astype(jnp.float32)


# def init_fn(input_shape, seed=1701):
#     rng = jax.random.PRNGKey(seed)                                     # jr = jax.random
#     dummy_input = jnp.ones(*input_shape)
#     L = self.input_size

#     return HiPPO(
#             N=self.hidden_size,
#             max_length=L,
#             step=1.0 / L,
#             GBT_alpha=self.GBT_alpha,
#             seq_L=L,
#             A=A,
#             B=B,
#             measure=self.measure,
#         )


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
