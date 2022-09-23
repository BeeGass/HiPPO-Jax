## import packages
import jax.numpy as jnp
from jax.numpy.linalg import inv
from scipy import special as ss


def make_HiPPO(
    N, v="nv", measure="legs", lambda_n=1, fourier_type="fru", alpha=0, beta=1
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
        if v == "nv":
            A, B = build_LegT(N=N, lambda_n=lambda_n)
        else:
            A, B = build_LegT_V(N=N, lambda_n=lambda_n)

    elif measure == "lagt":
        if v == "nv":
            A, B = build_LagT(alpha=alpha, beta=beta, N=N)
        else:
            A, B = build_LagT_V(alpha=alpha, beta=beta, N=N)

    elif measure == "legs":
        if v == "nv":
            A, B = build_LegS(N=N)
        else:
            A, B = build_LegS_V(N=N)

    elif measure == "fourier":
        if v == "nv":
            A, B = build_Fourier(N=N, fourier_type=fourier_type)
        else:
            A, B = build_Fourier_V(N=N, fourier_type=fourier_type)

    elif measure == "random":
        A = jnp.random.randn(N, N) / N
        B = jnp.random.randn(N, 1)

    elif measure == "diagonal":
        A = -jnp.diag(jnp.exp(jnp.random.randn(N)))
        B = jnp.random.randn(N, 1)

    else:
        raise ValueError("Invalid HiPPO type")

    A_copy = A.copy()
    B_copy = B.copy()

    return jnp.array(A_copy), B_copy


# Translated Legendre (LegT) - vectorized
def build_LegT_V(N, lambda_n=1):
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
    q = jnp.arange(N, dtype=jnp.float64)
    k, n = jnp.meshgrid(q, q)
    case = jnp.power(-1.0, (n - k))
    A = None
    B = None

    if lambda_n == 1:
        A_base = -jnp.sqrt(2 * n + 1) * jnp.sqrt(2 * k + 1)
        pre_D = jnp.sqrt(jnp.diag(2 * q + 1))
        B = D = jnp.diag(pre_D)[:, None]
        A = jnp.where(
            k <= n, A_base, A_base * case
        )  # if n >= k, then case_2 * A_base is used, otherwise A_base

    elif lambda_n == 2:  # (jnp.sqrt(2*n+1) * jnp.power(-1, n)):
        A_base = -(2 * n + 1)
        B = jnp.diag((2 * q + 1) * jnp.power(-1, n))[:, None]
        A = jnp.where(
            k <= n, A_base * case, A_base
        )  # if n >= k, then case_2 * A_base is used, otherwise A_base

    return A, B


# Translated Legendre (LegT) - non-vectorized
def build_LegT(N, legt_type="legt"):
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

    if legt_type == "legt":
        R = jnp.sqrt(pre_R)
        A = R[:, None] * jnp.where(n < k, (-1.0) ** (n - k), 1) * R[None, :]
        B = R[:, None]
        A = -A

        # Halve again for timescale correctness
        # A, B = A/2, B/2
        # A *= 0.5
        # B *= 0.5

    elif legt_type == "lmu":
        R = pre_R[:, None]
        A = jnp.where(n < k, -1, (-1.0) ** (n - k + 1)) * R
        B = (-1.0) ** Q[:, None] * R

    return A, B


# Translated Laguerre (LagT) - non-vectorized
def build_LagT_V(alpha, beta, N):
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
        0.5 * (ss.gammaln(jnp.arange(N) + alpha + 1) - ss.gammaln(jnp.arange(N) + 1))
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


# Translated Laguerre (LagT) - non-vectorized
def build_LagT(alpha, beta, N):
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
        0.5 * (ss.gammaln(jnp.arange(N) + alpha + 1) - ss.gammaln(jnp.arange(N) + 1))
    )
    A = (1.0 / L[:, None]) * A * L[None, :]
    B = (
        (1.0 / L[:, None])
        * B
        * jnp.exp(-0.5 * ss.gammaln(1 - alpha))
        * beta ** ((1 - alpha) / 2)
    )

    return A, B


# Scaled Legendre (LegS) vectorized
def build_LegS_V(N):
    """
    The, vectorized implementation of the, measure derived from the Scaled Legendre basis.

    Args:
        N (int): Order of coefficients to describe the orthogonal polynomial that is the HiPPO projection.

    Returns:
        A (jnp.ndarray): The A HiPPO matrix.
        B (jnp.ndarray): The B HiPPO matrix.

    """
    q = jnp.arange(N, dtype=jnp.float64)
    k, n = jnp.meshgrid(q, q)
    pre_D = jnp.sqrt(jnp.diag(2 * q + 1))
    B = D = jnp.diag(pre_D)[:, None]

    A_base = (-jnp.sqrt(2 * n + 1)) * jnp.sqrt(2 * k + 1)
    case_2 = (n + 1) / (2 * n + 1)

    A = jnp.where(n > k, A_base, 0.0)  # if n > k, then A_base is used, otherwise 0
    A = jnp.where(
        n == k, (A_base * case_2), A
    )  # if n == k, then A_base is used, otherwise A

    return A, B


# Scaled Legendre (LegS), non-vectorized
def build_LegS(N):
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


# Fourier Basis OPs and functions - vectorized
def build_Fourier_V(N, fourier_type="fru"):
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
        jnp.stack([jnp.zeros(N // 2), jnp.zeros(N // 2)], axis=-1).reshape(-1)[1:], 1
    )
    B = jnp.zeros(A.shape[1], dtype=jnp.float64)

    B = B.at[0::2].set(jnp.sqrt(2))
    B = B.at[0].set(1)

    q = jnp.arange(A.shape[1], dtype=jnp.float64)
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

    return A, B


def build_Fourier(N, fourier_type="fru"):
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

    elif fourier_type == "foud":
        d = jnp.stack([jnp.zeros(N // 2), freqs], axis=-1).reshape(-1)[1:]
        A = jnp.pi * (-jnp.diag(d, 1) + jnp.diag(d, -1))

        B = jnp.zeros(A.shape[1])
        B = B.at[0::2].set(jnp.sqrt(2))
        B = B.at[0].set(1)

        # Subtract off rank correction - this corresponds to the other endpoint u(t-1) in this case
        A = A - 0.5 * B[:, None] * B[None, :]
        B = 0.5 * B[:, None]

    return A, B
