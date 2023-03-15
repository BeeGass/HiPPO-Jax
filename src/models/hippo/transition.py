## import packages
from typing import Any, Tuple

import jax.numpy as jnp
from jaxtyping import Array, Float
from scipy import special as ss


class TransMatrix:
    def __init__(
        self,
        N: int,
        measure: str = "legs",
        lambda_n: float = 1.0,
        alpha: float = 0.0,
        beta: float = 1.0,
        dtype: Any = jnp.float32,
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
                    - HiPPO w/ Fourier basis
                        - FRU: Fourier Recurrent Unit - fru
                        - FouT: Translated Fourier - fout
                        - FourD: Fourier Decay - fourd
            lambda_n (int): The amount of tilt applied to the HiPPO-LegS basis, determines between LegS and LMU.
            alpha (float): The order of the Laguerre basis.
            beta (float): The scale of the Laguerre basis.

        Returns:
            A (jnp.ndarray): The HiPPO matrix multiplied by -1.
            B (jnp.ndarray): The other corresponding state space matrix.

        """
        A = None
        B = None
        if measure in ["legt", "lmu"]:
            if measure == "legt":
                assert lambda_n == 1.0
            elif measure == "lmu":
                assert lambda_n == 2.0
            else:
                raise ValueError("Invalid lambda_n for HiPPO type 'legt' or 'lmu")

            A, B = self.build_LegT(N=N, lambda_n=lambda_n, dtype=dtype)

        elif measure == "lagt":
            A, B = self.build_LagT(alpha=alpha, beta=beta, N=N, dtype=dtype)

        elif measure == "legs":
            A, B = self.build_LegS(N=N, dtype=dtype)

        elif measure in ["fout", "fru", "foud"]:
            A, B = self.build_Fourier(N=N, fourier_type=measure, dtype=dtype)

        elif measure == "random":
            A = jnp.random.randn(N, N) / N
            B = jnp.random.randn(N, 1)

        elif measure == "diagonal":
            A = -jnp.diag(jnp.exp(jnp.random.randn(N)))
            B = jnp.random.randn(N, 1)

        else:
            raise ValueError("Invalid HiPPO type")

        self.A = (A.copy()).astype(dtype)
        self.B = (B.copy()).astype(dtype)

    # Translated Legendre (LegT) - vectorized
    @staticmethod
    def build_LegT(
        N: int, lambda_n: int = 1, dtype=jnp.float32
    ) -> Tuple[Float[Array, "N N"], Float[Array, "N 1"]]:
        """
        The, vectorized implementation of the, measure derived from the translated Legendre basis.

        Args:
            N (int): Order of coefficients to describe the orthogonal polynomial that is the HiPPO projection.
            legt_type (str): Choice between the two different tilts of basis.
                - legt: translated Legendre - 'legt'
                - lmu: Legendre Memory Unit - 'lmu'

        Returns:
            A (jnp.ndarray): The A HiPPO matrix.
            B (jnp.ndarray): The B HiPPO matrix.

        """
        q = jnp.arange(N, dtype=dtype)
        k, n = jnp.meshgrid(q, q)
        case = jnp.power(-1.0, (n - k))
        A = None
        B = None

        if lambda_n == 1:
            A_base = jnp.sqrt(2 * n + 1) * jnp.sqrt(2 * k + 1)
            pre_D = jnp.sqrt(jnp.diag(2 * q + 1))
            B = D = jnp.diag(pre_D)[:, None]
            A = jnp.where(
                k <= n, A_base, A_base * case
            )  # if n >= k, then case_2 * A_base is used, otherwise A_base

        elif lambda_n == 2:  # (jnp.sqrt(2*n+1) * jnp.power(-1, n)):
            A_base = 2 * n + 1
            B = jnp.diag((2 * q + 1) * jnp.power(-1, n))[:, None]
            A = jnp.where(
                k <= n, A_base * case, A_base
            )  # if n >= k, then case_2 * A_base is used, otherwise A_base

        return -A.astype(dtype), B.astype(dtype)

    # Translated Laguerre (LagT) - non-vectorized
    @staticmethod
    def build_LagT(
        alpha: float, beta: float, N: int, dtype=jnp.float32
    ) -> Tuple[Float[Array, "N N"], Float[Array, "N 1"]]:
        """
        The, vectorized implementation of the, measure derived from the translated Laguerre basis.

        Args:
            alpha (float): The order of the Laguerre basis.
            beta (float): The scale of the Laguerre basis.
            N (int): Order of coefficients to describe the orthogonal polynomial that is the HiPPO projection.

        Returns:
            A (jnp.ndarray): The A HiPPO matrix.
            B (jnp.ndarray): The B HiPPO matrix.

        """
        L = jnp.exp(
            0.5
            * (ss.gammaln(jnp.arange(N) + alpha + 1) - ss.gammaln(jnp.arange(N) + 1))
        )
        inv_L = 1.0 / L[:, None]
        pre_A = (jnp.eye(N) * ((1 + beta) / 2)) + jnp.tril(jnp.ones((N, N)), -1)
        pre_B = ss.binom(alpha + jnp.arange(N), jnp.arange(N))[:, None]

        A = -inv_L * pre_A * L[None, :]
        B = (
            jnp.exp(-0.5 * ss.gammaln(1 - alpha))
            * jnp.power(beta, (1 - alpha) / 2)
            * inv_L
            * pre_B
        )

        return A.astype(dtype), B.astype(dtype)

    # Scaled Legendre (LegS) vectorized
    @staticmethod
    def build_LegS(
        N: int, dtype=jnp.float32
    ) -> Tuple[Float[Array, "N N"], Float[Array, "N 1"]]:
        """
        The, vectorized implementation of the, measure derived from the Scaled Legendre basis.

        Args:
            N (int): Order of coefficients to describe the orthogonal polynomial that is the HiPPO projection.

        Returns:
            A (jnp.ndarray): The A HiPPO matrix.
            B (jnp.ndarray): The B HiPPO matrix.

        """
        q = jnp.arange(N, dtype=dtype)
        k, n = jnp.meshgrid(q, q)
        pre_D = jnp.sqrt(jnp.diag(2 * q + 1))
        B = D = jnp.diag(pre_D)[:, None]

        A_base = jnp.sqrt(2 * n + 1) * jnp.sqrt(2 * k + 1)

        A = jnp.where(n > k, A_base, jnp.where(n == k, n + 1, 0.0))

        return -A.astype(dtype), B.astype(dtype)

    # Fourier Basis OPs and functions - vectorized
    @staticmethod
    def build_Fourier(
        N: int, fourier_type: str = "fru", dtype=jnp.float32
    ) -> Tuple[Float[Array, "N N"], Float[Array, "N 1"]]:
        """
        Vectorized measure implementations derived from fourier basis.

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
        A = jnp.diag(
            jnp.stack([jnp.zeros(N // 2), jnp.zeros(N // 2)], axis=-1).reshape(-1)[1:],
            1,
        )
        B = jnp.zeros(A.shape[1], dtype=dtype)

        B = B.at[0::2].set(jnp.sqrt(2))
        B = B.at[0].set(1)

        q = jnp.arange(A.shape[1], dtype=dtype)
        k, n = jnp.meshgrid(q, q)

        n_odd = n % 2 == 0
        k_odd = k % 2 == 0

        case_1 = (n == k) & (n == 0)
        case_2_3 = ((k == 0) & (n_odd)) | ((n == 0) & (k_odd))
        case_4 = (n_odd) & (k_odd)
        case_5 = (n - k == 1) & (k_odd)
        case_6 = (k - n == 1) & (n_odd)

        if fourier_type == "fru":  # Fourier Recurrent Unit (FRU) - vectorized
            A = jnp.where(
                case_1,
                -1.0,
                jnp.where(
                    case_2_3,
                    -jnp.sqrt(2),
                    jnp.where(
                        case_4,
                        -2,
                        jnp.where(
                            case_5,
                            jnp.pi * (n // 2),
                            jnp.where(case_6, -jnp.pi * (k // 2), 0.0),
                        ),
                    ),
                ),
            )

        elif fourier_type == "fout":  # truncated Fourier (FouT) - vectorized
            A = jnp.where(
                case_1,
                -1.0,
                jnp.where(
                    case_2_3,
                    -jnp.sqrt(2),
                    jnp.where(
                        case_4,
                        -2,
                        jnp.where(
                            case_5,
                            jnp.pi * (n // 2),
                            jnp.where(case_6, -jnp.pi * (k // 2), 0.0),
                        ),
                    ),
                ),
            )

            A = 2 * A
            B = 2 * B

        elif fourier_type == "foud":
            A = jnp.where(
                case_1,
                -1.0,
                jnp.where(
                    case_2_3,
                    -jnp.sqrt(2),
                    jnp.where(
                        case_4,
                        -2,
                        jnp.where(
                            case_5,
                            2 * jnp.pi * (n // 2),
                            jnp.where(case_6, 2 * -jnp.pi * (k // 2), 0.0),
                        ),
                    ),
                ),
            )

            A = 0.5 * A
            B = 0.5 * B

        B = B[:, None]

        return A.astype(dtype), B.astype(dtype)
