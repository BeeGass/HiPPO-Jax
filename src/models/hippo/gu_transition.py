## import packages
import jax.numpy as jnp
from jax.numpy.linalg import inv
from scipy import special as ss
from src.models.hippo.unroll import *


class GuTransMatrix:
    def __init__(
        self, N, measure="legs", lambda_n=1.0, fourier_type="fru", alpha=0, beta=1
    ):
        """
        Instantiates the HiPPO matrix of a given order using a particular measure.
        Args:
            N (int): Order of coefficients to describe the orthogonal polynomial that is the HiPPO projection.
            v (str): choose between this repo's implementation or hazy research's implementation.
            measure (str):
                choose between
                    - HiPPO w/ Translated Legendre (LegT) - legt
                    - HiPPO w/ Translated Laguerre (LagT) - lagt
                    - HiPPO w/ Scaled Legendre (LegS) - legs
                    - HiPPO w/ Fourier basis - fourier
                        - FRU: Fourier Recurrent Unit
                        - FouT: Translated Fourier
            lambda_n (int): The amount of tilt applied to the HiPPO-LegS basis, determines between LegS and LMU.
            fourier_type (str): chooses between the following:
                - FRU: Fourier Recurrent Unit - fru
                - FouT: Translated Fourier - fout
                - FourD: Fourier Decay - fourd
            alpha (float): The order of the Laguerre basis.
            beta (float): The scale of the Laguerre basis.

        Returns:
            A (jnp.ndarray): The HiPPO matrix multiplied by -1.
            B (jnp.ndarray): The other corresponding state space matrix.

        """
        A = None
        B = None
        if measure == "legt":
            A, B = self.build_gu_LegT(N=N, lambda_n=lambda_n)

        elif measure == "lagt":
            A, B = self.build_gu_LagT(alpha=alpha, beta=beta, N=N)

        elif measure == "legs":
            A, B = self.build_gu_LegS(N=N)

        elif measure == "fourier":
            A, B = self.build_gu_Fourier(N=N, fourier_type=fourier_type)

        elif measure == "random":
            A = jnp.random.randn(N, N) / N
            B = jnp.random.randn(N, 1)

        elif measure == "diagonal":
            A = -jnp.diag(jnp.exp(jnp.random.randn(N)))
            B = jnp.random.randn(N, 1)

        else:
            raise ValueError("Invalid HiPPO type")

        self.A_matrix = A.copy()
        self.B_matrix = B.copy()

    # Scaled Legendre (LegS), non-vectorized
    @staticmethod
    def build_gu_LegS(N):
        """
        The, non-vectorized implementation of the, measure derived from the Scaled Legendre basis.

        Args:
            N (int): Order of coefficients to describe the orthogonal polynomial that is the HiPPO projection.

        Returns:
            A (jnp.ndarray): The A HiPPO matrix.
            B (jnp.ndarray): The B HiPPO matrix.

        """
        q = jnp.arange(
            N, dtype=jnp.float64
        )  # q represents the values 1, 2, ..., N each column has
        k, n = jnp.meshgrid(q, q)
        r = 2 * q + 1
        M = -(jnp.where(n >= k, r, 0) - jnp.diag(q))  # represents the state matrix M
        D = jnp.sqrt(
            jnp.diag(2 * q + 1)
        )  # represents the diagonal matrix D $D := \text{diag}[(2n+1)^{\frac{1}{2}}]^{N-1}_{n=0}$
        A = D @ M @ jnp.linalg.inv(D)
        B = jnp.diag(D)[:, None]
        B = (
            B.copy()
        )  # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)

        return A, B

    # Translated Legendre (LegT) - non-vectorized
    @staticmethod
    def build_gu_LegT(N, lambda_n=1.0):
        """
        The, non-vectorized implementation of the, measure derived from the translated Legendre basis

        Args:
            N (int): Order of coefficients to describe the orthogonal polynomial that is the HiPPO projection.
            legt_type (str): Choice between the two different tilts of basis.
                - legt: translated Legendre - 'legt'
                - lmu: Legendre Memory Unit - 'lmu'

        Returns:
            A (jnp.ndarray): The A HiPPO matrix.
            B (jnp.ndarray): The B HiPPO matrix.

        """
        Q = jnp.arange(N, dtype=jnp.float64)
        pre_R = 2 * Q + 1
        k, n = jnp.meshgrid(Q, Q)

        if lambda_n == 1.0:
            R = jnp.sqrt(pre_R)
            A = R[:, None] * jnp.where(n < k, (-1.0) ** (n - k), 1) * R[None, :]
            B = R[:, None]
            A = -A

            # Halve again for timescale correctness
            # A, B = A/2, B/2
            # A *= 0.5
            # B *= 0.5

        elif lambda_n == 2.0:
            R = pre_R[:, None]
            A = jnp.where(n < k, -1, (-1.0) ** (n - k + 1)) * R
            B = (-1.0) ** Q[:, None] * R

        return A, B

    # Translated Laguerre (LagT) - non-vectorized
    @staticmethod
    def build_gu_LagT(alpha, beta, N):
        """
        The, non-vectorized implementation of the, measure derived from the translated Laguerre basis.

        Args:
            alpha (float): The order of the Laguerre basis.
            beta (float): The scale of the Laguerre basis.
            N (int): Order of coefficients to describe the orthogonal polynomial that is the HiPPO projection.

        Returns:
            A (jnp.ndarray): The A HiPPO matrix.
            B (jnp.ndarray): The B HiPPO matrix.

        """
        A = -jnp.eye(N) * (1 + beta) / 2 - jnp.tril(jnp.ones((N, N)), -1)
        B = ss.binom(alpha + jnp.arange(N), jnp.arange(N))[:, None]

        L = jnp.exp(
            0.5
            * (ss.gammaln(jnp.arange(N) + alpha + 1) - ss.gammaln(jnp.arange(N) + 1))
        )
        A = (1.0 / L[:, None]) * A * L[None, :]
        B = (
            (1.0 / L[:, None])
            * B
            * jnp.exp(-0.5 * ss.gammaln(1 - alpha))
            * beta ** ((1 - alpha) / 2)
        )

        return A, B

    @staticmethod
    def build_gu_Fourier(N, fourier_type="fru"):
        """
        Non-vectorized measure implementations derived from fourier basis.

        Args:
            N (int): Order of coefficients to describe the orthogonal polynomial that is the HiPPO projection.
            fourier_type (str): The type of Fourier measure.
                - FRU: Fourier Recurrent Unit - fru
                - FouT: truncated Fourier - fout
                - fouD: decayed Fourier - foud

        Returns:
            A (jnp.ndarray): The A HiPPO matrix.
            B (jnp.ndarray): The B HiPPO matrix.

        """
        freqs = jnp.arange(N // 2)

        if fourier_type == "fru":  # Fourier Recurrent Unit (FRU) - non-vectorized
            d = jnp.stack([jnp.zeros(N // 2), freqs], axis=-1).reshape(-1)[1:]
            A = jnp.pi * (-jnp.diag(d, 1) + jnp.diag(d, -1))

            B = jnp.zeros(A.shape[1])
            B = B.at[0::2].set(jnp.sqrt(2))
            B = B.at[0].set(1)

            A = A - B[:, None] * B[None, :]
            B = B[:, None]

        elif fourier_type == "fout":  # truncated Fourier (FouT) - non-vectorized
            freqs *= 2
            d = jnp.stack([jnp.zeros(N // 2), freqs], axis=-1).reshape(-1)[1:]
            A = jnp.pi * (-jnp.diag(d, 1) + jnp.diag(d, -1))

            B = jnp.zeros(A.shape[1])
            B = B.at[0::2].set(jnp.sqrt(2))
            B = B.at[0].set(1)

            # Subtract off rank correction - this corresponds to the other endpoint u(t-1) in this case
            A = A - B[:, None] * B[None, :] * 2
            B = B[:, None] * 2

        elif fourier_type == "fourd":
            d = jnp.stack([jnp.zeros(N // 2), freqs], axis=-1).reshape(-1)[1:]
            A = jnp.pi * (-jnp.diag(d, 1) + jnp.diag(d, -1))

            B = jnp.zeros(A.shape[1])
            B = B.at[0::2].set(jnp.sqrt(2))
            B = B.at[0].set(1)

            # Subtract off rank correction - this corresponds to the other endpoint u(t-1) in this case
            A = A - 0.5 * B[:, None] * B[None, :]
            B = 0.5 * B[:, None]

        return A, B
