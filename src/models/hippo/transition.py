## import packages
import jax.numpy as jnp
from jax.numpy.linalg import inv
from scipy import special as ss
from opt_einsum import contract


class TransMatrix:
    def __init__(
        self, N, measure="legs", lambda_n=1, fourier_type="fru", alpha=0, beta=1
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
            A, B = self.build_LegT(N=N, lambda_n=lambda_n)

        elif measure == "lagt":
            A, B = self.build_LagT(alpha=alpha, beta=beta, N=N)

        elif measure == "legs":
            A, B = self.build_LegS(N=N)

        elif measure == "fourier":
            A, B = self.build_Fourier(N=N, fourier_type=fourier_type)

        elif measure == "random":
            A = jnp.random.randn(N, N) / N
            B = jnp.random.randn(N, 1)

        elif measure == "diagonal":
            A = -jnp.diag(jnp.exp(jnp.random.randn(N)))
            B = jnp.random.randn(N, 1)

        else:
            raise ValueError("Invalid HiPPO type")

        self.A_matrix = (A.copy()).astype(jnp.float32)
        self.B_matrix = (B.copy()).astype(jnp.float32)

    # Translated Legendre (LegT) - vectorized
    @staticmethod
    def build_LegT(N, lambda_n=1):
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
        q = jnp.arange(N, dtype=jnp.float32)
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

        return -A, B

    # Translated Laguerre (LagT) - non-vectorized
    @staticmethod
    def build_LagT(alpha, beta, N):
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

        return A, B

    # Scaled Legendre (LegS) vectorized
    @staticmethod
    def build_LegS(N):
        """
        The, vectorized implementation of the, measure derived from the Scaled Legendre basis.

        Args:
            N (int): Order of coefficients to describe the orthogonal polynomial that is the HiPPO projection.

        Returns:
            A (jnp.ndarray): The A HiPPO matrix.
            B (jnp.ndarray): The B HiPPO matrix.

        """
        q = jnp.arange(N, dtype=jnp.float32)
        k, n = jnp.meshgrid(q, q)
        pre_D = jnp.sqrt(jnp.diag(2 * q + 1))
        B = D = jnp.diag(pre_D)[:, None]

        A_base = jnp.sqrt(2 * n + 1) * jnp.sqrt(2 * k + 1)

        A = jnp.where(n > k, A_base, jnp.where(n == k, n + 1, 0.0))

        return -A.astype(jnp.float32), B.astype(jnp.float32)

    # Fourier Basis OPs and functions - vectorized
    @staticmethod
    def build_Fourier(N, fourier_type="fru"):
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
        B = jnp.zeros(A.shape[1], dtype=jnp.float32)

        B = B.at[0::2].set(jnp.sqrt(2))
        B = B.at[0].set(1)

        q = jnp.arange(A.shape[1], dtype=jnp.float32)
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

        return A.astype(jnp.float32), B.astype(jnp.float32)


class LowRankMatrix:
    def __init__(
        self,
        N,
        rank,
        measure="legs",
        lambda_n=1,
        fourier_type="fru",
        alpha=0,
        beta=1,
        DPLR=True,
        dtype=jnp.float32,
    ):
        self.N = N
        self.measure = measure
        self.rank = rank
        _trans_matrix = TransMatrix(N, measure, lambda_n, fourier_type, alpha, beta)

        Lambda = None
        B = None
        V = None
        A, B, P, S = self.make_NPLR(trans_matrix=_trans_matrix, dtype=dtype)
        if DPLR:
            Lambda, P, B, V = self.make_DPLR(B=B, P=P, S=S)
            self.Lambda = (Lambda.copy()).astype(dtype)  # real eigenvalues
            self.V = (V.copy()).astype(dtype)  # imaginary (complex) eigenvalues

        self.A = (A.copy()).astype(dtype)  # HiPPO A Matrix (N x N)
        self.B = (B.copy()).astype(dtype)  # HiPPO B Matrix (N x 1)
        self.P = (P.copy()).astype(dtype)  # HiPPO rank correction matrix (N x rank)
        self.S = (S.copy()).astype(
            dtype
        )  # HiPPO normal (skew-symmetric) matrix (N x N)

    def make_NPLR(self, trans_matrix, dtype=jnp.float32):
        A = trans_matrix.A_matrix
        B = trans_matrix.B_matrix

        P = self.rank_correction(dtype=dtype)  # (r N)

        S = A + jnp.sum(
            jnp.expand_dims(P, -2) * jnp.expand_dims(P, -1), axis=-3
        )  # rank correct if rank > 1, summation happens in outer most dimension
        # S is nearly skew-symmetric

        return A, B, P, S

    def make_DPLR(self, B, P, S):
        """Diagonalize NPLR representation"""

        self.check_skew(S=S)

        # Check skew symmetry
        S_diag = jnp.diagonal(S)
        Lambda_real = jnp.mean(S_diag, -1, keepdims=True) * jnp.ones_like(
            S_diag
        )  # S itself is not skew-symmetric. It is skew-symmetric by: S + c * I. Extract the value c, c = mean(S_diag)

        # Diagonalize S to V \Lambda V^*
        Lambda_imaginary, V = jnp.linalg.eigh(S * -1j)
        Lambda = Lambda_real + 1j * Lambda_imaginary

        Lambda, V = self.fix_zeroed_eigvals(Lambda=Lambda, V=V)

        P = V.conj().transpose(-1, -2) @ P
        B = V.conj().transpose(-1, -2) @ B

        return Lambda, P, B, V

    def check_skew(self, S):
        """Check if a matrix is skew symmetric

        We require AP to be nearly skew-symmetric. To be clear, AP IS NOT skew-symmetric.
        However, it is skew-symmetric up to a small error. This function checks that error is within an acceptable tolerance.

        refer to:
        - https://www.cuemath.com/algebra/skew-symmetric-matrix/
        - https://en.wikipedia.org/wiki/Skew-symmetric_matrix

        """
        _S = S + S.transpose(
            -1, -2
        )  # ensure matrices are skew symmetric by assuming S is skew symmetric, adding two skew symmetric matrices results in a skew symmetric matrix
        if (
            err := jnp.sum((_S - _S[0, 0] * jnp.eye(self.N)) ** 2) / self.N
        ) > 1e-5:  # if not torch.allclose(_A - _A[0,0]*torch.eye(N), torch.zeros(N, N), atol=1e-5):
            print("WARNING: HiPPO matrix not skew symmetric", err)
            print(
                f"Transposed matrix:\n{_S.transpose(-1, -2)}\n\nUnchanged matrix:\n{-_S}"
            )  # the transpose of a skew symmetric matrix is equal to the negative of the matrix

    def fix_zeroed_eigvals(self, Lambda, V):

        # Only keep half of each conjugate pair
        imaginary_eigvals = Lambda.imag
        print(f"jax - imaginary eigvals: {imaginary_eigvals}")
        print(f"jax - idx of imaginary eigvals: {jnp.argsort(imaginary_eigvals)}")
        idx = jnp.argsort(imaginary_eigvals)
        Lambda_sorted = Lambda[idx]
        V_sorted = V[:, idx]

        # There is an edge case when eigenvalues can be 0, which requires some machinery to handle
        # We use a huge hack here: Assume only one pair is 0, and that it is the first row/column of A (only happens in Fourier case)
        V = V_sorted[:, : self.N // 2]
        Lambda = Lambda_sorted[: self.N // 2]
        if jnp.abs(Lambda[-1]) < 1e-4:
            # x = x.at[idx].set(y)
            V = V.at[:, -1].set(0.0)  # V[:, -1] = 0.0
            V = V.at[0, -1].set(2**-0.5)  # V[0, -1] = 2**-0.5
            V = V.at[1, -1].set(2**-0.5 * 1j)  # V[1, -1] = 2**-0.5 * 1j
        else:
            print(f"Lambdas:\n{Lambda[-1]}\n\n")
            raise ValueError("Only 1 zero eigenvalue allowed in diagonal part of A")

        _AP = V @ jnp.diag_embed(Lambda) @ V.conj().transpose(-1, -2)
        if (err := jnp.sum((2 * _AP.real - AP) ** 2) / self.N) > 1e-5:
            print(
                "Warning: Diagonalization of A matrix not numerically precise - error",
                err,
            )

        return Lambda, V

    def rank_correction(self, dtype=jnp.float32):
        """Return low-rank matrix L such that A + L is normal"""

        if self.measure == "legs":
            assert self.rank >= 1
            P = jnp.expand_dims(
                jnp.sqrt(0.5 + jnp.arange(self.N, dtype=dtype)), 0
            )  # (1 N)

        elif self.measure == "legt":
            assert self.rank >= 2
            P = jnp.sqrt(1 + 2 * jnp.arange(self.N, dtype=dtype))  # (N)
            P0 = P.clone()
            P0 = P0.at[0::2].set(0.0)  # P0[0::2] = 0.0
            P1 = P.clone()
            P1 = P1.at[1::2].set(0.0)  # P1[1::2] = 0.0
            P = jnp.stack([P0, P1], axis=0)  # (2 N)
            P = P * (
                2 ** (-0.5)
            )  # Halve the rank correct just like the original matrix was halved

        elif self.measure == "lagt":
            assert self.rank >= 1
            P = 0.5**0.5 * jnp.ones((1, self.N), dtype=dtype)

        elif self.measure in ["fourier", "fout"]:
            P = jnp.zeros(self.N)
            P = P.at[0::2].set(2**0.5)  # P[0::2] = 2**0.5
            P = P.at[0].set(1)  # P[0] = 1
            P = jnp.expand_dims(P, 0)

        elif self.measure == "fourier_decay":
            P = jnp.zeros(self.N)
            P = P.at[0::2].set(2**0.5)  # P[0::2] = 2**0.5
            P = P.at[0].set(1)  # P[0] = 1
            P = jnp.expand_dims(P, 0)
            P = P / 2**0.5

        elif self.measure == "fourier2":
            P = jnp.zeros(self.N)
            P = P.at[0::2].set(2**0.5)  # P[0::2] = 2**0.5
            P = P.at[0].set(1)  # P[0] = 1
            P = 2**0.5 * jnp.expand_dims(P, 0)

        elif self.measure in ["fourier_diag", "foud", "legsd"]:
            P = jnp.zeros((1, self.N), dtype=dtype)

        else:
            raise NotImplementedError

        d = jnp.size(P, axis=0)
        if self.rank > d:
            P = jnp.concatenate(
                [P, jnp.zeros((self.rank - d, self.N), dtype=dtype)], axis=0
            )  # (rank N)

        return P
