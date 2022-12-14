## import packages
import jax.numpy as jnp
from jax.numpy.linalg import inv
from scipy import special as ss
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import special as ss
from einops import rearrange, repeat
from opt_einsum import contract
import math
from typing import Any
from src.models.hippo.unroll import *


class GuTransMatrix:
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
        if measure in ["legt", "lmu"]:
            A, B = self.build_gu_LegT(N=N, lambda_n=lambda_n)

        elif measure == "lagt":
            A, B = self.build_gu_LagT(alpha=alpha, beta=beta, N=N)

        elif measure == "legs":
            A, B = self.build_gu_LegS(N=N)

        elif measure in ["fout", "fru", "foud"]:
            A, B = self.build_gu_Fourier(N=N, fourier_type=measure)

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
            N, dtype=jnp.float32
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
        A = None
        B = None
        Q = jnp.arange(N, dtype=jnp.float32)
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
        A = None
        B = None
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
        else:
            raise ValueError("Invalid fourier_type")

        return A, B


class GuLowRankMatrix:
    def __init__(
        self,
        N: int,
        rank: int = 1,
        measure: str = "legs",
        lambda_n: float = 1.0,
        alpha: float = 0.0,
        beta: float = 1.0,
        DPLR: bool = True,
        dtype: Any = torch.float,
    ):
        self.N = N
        self.measure = measure
        self.rank = rank
        _trans_matrix = GuTransMatrix(
            N, measure, lambda_n, alpha, beta, dtype=jnp.float32
        )

        A, B, P, S = self.pre_nplr(trans_matrix=_trans_matrix, dtype=dtype)
        if DPLR:
            Lambda, P, B, V = self.dplr(
                trans_matrix=_trans_matrix,
                scaling="hippo",
                H=1,
                dtype=dtype,
                real_scale=1.0,
                imag_scale=1.0,
                random_real=False,
                random_imag=False,
                normalize=False,
                diagonal=True,
                random_B=False,
            )
            self.Lambda = Lambda  # real eigenvalues
            self.V = V  # imaginary (complex) eigenvalues

        self.A = A  # HiPPO A Matrix (N x N)
        self.B = B  # HiPPO B Matrix (N x 1)
        self.P = P  # HiPPO rank correction matrix (N x rank)
        self.S = S  # HiPPO normal (skew-symmetric) matrix (N x N)

    def rank_correction(self, dtype=torch.float):
        """Return low-rank matrix L such that A + L is normal"""

        if self.measure == "legs":
            assert self.rank >= 1
            P = torch.sqrt(0.5 + torch.arange(self.N, dtype=dtype)).unsqueeze(
                0
            )  # (1 N)

        elif self.measure == "legt":
            assert self.rank >= 2
            P = torch.sqrt(1 + 2 * torch.arange(self.N, dtype=dtype))  # (N)
            P0 = P.clone()
            P0[0::2] = 0.0
            P1 = P.clone()
            P1[1::2] = 0.0
            P = torch.stack([P0, P1], dim=0)  # (2 N)
            P *= 2 ** (
                -0.5
            )  # Halve the rank correct just like the original matrix was halved

        elif self.measure == "lagt":
            assert self.rank >= 1
            P = 0.5**0.5 * torch.ones(1, self.N, dtype=dtype)

        elif self.measure in ["fourier", "fout", "fru"]:
            P = torch.zeros(self.N)
            P[0::2] = 2**0.5
            P[0] = 1
            P = P.unsqueeze(0)

        elif self.measure == "fourier_decay":
            P = torch.zeros(self.N)
            P[0::2] = 2**0.5
            P[0] = 1
            P = P.unsqueeze(0)
            P = P / 2**0.5

        elif self.measure == "fourier2":
            P = torch.zeros(self.N)
            P[0::2] = 2**0.5
            P[0] = 1
            P = 2**0.5 * P.unsqueeze(0)

        elif self.measure in ["fourier_diag", "foud", "legsd"]:
            P = torch.zeros(1, self.N, dtype=dtype)

        else:
            raise NotImplementedError

        d = P.size(0)
        if self.rank > d:
            P = torch.cat(
                [P, torch.zeros((self.rank - d, self.N), dtype=dtype)], dim=0
            )  # (rank N)

        return P

    def pre_nplr(self, trans_matrix, dtype=torch.float):
        jnp_A = trans_matrix.A
        jnp_B = trans_matrix.B

        np_A = np.asarray(jnp_A)
        A = torch.from_numpy(np_A)  # (N, N)

        np_B = np.asarray(jnp_B)
        B = torch.from_numpy(np_B)  # [:, 0]  # (N,)

        P = self.rank_correction(dtype=dtype)  # (r N)
        AP = A + torch.sum(P.unsqueeze(-2) * P.unsqueeze(-1), dim=-3)

        return A, B, P, AP

    def check_skew(self, AP):
        """
        We require AP to be nearly skew-symmetric. To be clear, AP IS NOT skew-symmetric.
        However, it is skew-symmetric up to a small error. This function checks that error is within an acceptable tolerance.
        """
        _A = AP + AP.transpose(-1, -2)
        if (
            err := torch.sum((_A - _A[0, 0] * torch.eye(self.N)) ** 2) / self.N
        ) > 1e-5:  # if not torch.allclose(_A - _A[0,0]*torch.eye(N), torch.zeros(N, N), atol=1e-5):
            print("WARNING: HiPPO matrix not skew symmetric", err)

    def nplr(self, trans_matrix, diagonalize_precision=True, dtype=torch.float):
        """Return w, p, q, V, B such that
        (w - p q^*, B) is unitarily equivalent to the original HiPPO A, B by the matrix V
        i.e. A = V[w - p q^*]V^*, B = V B
        """
        assert dtype == torch.float or torch.double
        cdtype = torch.cfloat if dtype == torch.float else torch.cdouble

        A, pre_B, P, AP = self.pre_nplr(trans_matrix=trans_matrix, dtype=dtype)

        B = pre_B[:, 0]

        self.check_skew(AP)

        # Take advantage of identity + skew-symmetric form to calculate real and imaginary parts separately
        # Imaginary part can use eigh instead of eig
        w_re = torch.mean(torch.diagonal(AP), -1, keepdim=True)

        # Diagonalize in double precision
        if diagonalize_precision:
            AP = AP.to(torch.double)
        # w, V = torch.linalg.eig(AP) # (..., N) (..., N, N)
        w_im, V = torch.linalg.eigh(AP * -1j)  # (..., N) (..., N, N)
        if diagonalize_precision:
            w_im, V = w_im.to(cdtype), V.to(cdtype)
        w = w_re + 1j * w_im
        # Check: V w V^{-1} = A
        # print("check", V @ torch.diag_embed(w) @ V.conj().transpose(-1, -2))

        # Only keep half of each conjugate pair
        imaginary_eigvals = w.imag
        print(f"imaginary eigvals: {imaginary_eigvals}")
        print(f"idx of imaginary eigvals: {torch.sort(imaginary_eigvals)}")
        _, idx = torch.sort(w.imag)
        w_sorted = w[idx]
        V_sorted = V[:, idx]

        # There is an edge case when eigenvalues can be 0, which requires some machinery to handle
        # We use a huge hack here: Assume only one pair is 0, and that it is the first row/column of A (only happens in Fourier case)
        V = V_sorted[:, : self.N // 2]
        w = w_sorted[: self.N // 2]
        assert (
            w[-2].abs() > 1e-4
        ), "Only 1 zero eigenvalue allowed in diagonal part of A"
        if w[-1].abs() < 1e-4:
            V[:, -1] = 0.0
            V[0, -1] = 2**-0.5
            V[1, -1] = 2**-0.5 * 1j

        _AP = V @ torch.diag_embed(w) @ V.conj().transpose(-1, -2)
        if (err := torch.sum((2 * _AP.real - AP) ** 2) / self.N) > 1e-5:
            print(
                "Warning: Diagonalization of A matrix not numerically precise - error",
                err,
            )
        # print("check", V @ torch.diag_embed(w) @ V.conj().transpose(-1, -2))

        V_inv = V.conj().transpose(-1, -2)
        print(f"Lambda:\n{w}")
        print(f"V_inv:\n{V_inv}")
        print(f"P:\n{P}")
        print(f"B:\n{B}")
        print(f"V_inv  shape:\n{V_inv.shape}")
        print(f"P shape:\n{P.shape}")
        print(f"B shape:\n{B.shape}")

        # C = initial_C(measure, N, dtype=dtype)
        B = contract("ij, j -> i", V_inv, B.to(V))  # V^* B
        # C = contract('ij, j -> i', V_inv, C.to(V)) # V^* C
        P = contract("ij, ...j -> ...i", V_inv, P.to(V))  # V^* P

        print(f"gu_B after einsum:\n{B}")
        print(f"gu_P after einsum:\n{P}")

        # return w, P, B, C, V
        return w, P, B, V

    def dplr(
        self,
        trans_matrix,
        scaling="hippo",
        H=1,
        dtype=torch.float,
        real_scale=1.0,
        imag_scale=1.0,
        random_real=False,
        random_imag=False,
        normalize=False,
        diagonal=True,
        random_B=False,
    ):
        assert dtype == torch.float or torch.double
        dtype = torch.cfloat if dtype == torch.float else torch.cdouble

        pi = torch.tensor(math.pi)
        if random_real:
            real_part = torch.rand(H, self.N // 2)

        else:
            real_part = 0.5 * torch.ones(H, self.N // 2)

        if random_imag:
            imag_part = self.N // 2 * torch.rand(H, self.N // 2)

        else:
            imag_part = repeat(torch.arange(self.N // 2), "n -> h n", h=H)

        real_part = real_scale * real_part
        if scaling == "random":
            imag_part = torch.randn(H, self.N // 2)

        elif scaling == "real":
            imag_part = 0 * imag_part
            real_part = 1 + repeat(torch.arange(self.N // 2), "n -> h n", h=H)

        elif scaling in ["linear", "lin"]:
            imag_part = pi * imag_part

        elif scaling in [
            "inverse",
            "inv",
        ]:  # Based on asymptotics of the default HiPPO matrix
            imag_part = 1 / pi * self.N * (self.N / (1 + 2 * imag_part) - 1)

        elif scaling in ["inverse2", "inv2"]:
            imag_part = 1 / pi * self.N * (self.N / (1 + imag_part) - 1)

        elif scaling in ["quadratic", "quad"]:
            imag_part = 1 / pi * (1 + 2 * imag_part) ** 2

        elif scaling in ["legs", "hippo"]:
            w, _, _, _ = self.nplr(
                trans_matrix=trans_matrix, dtype=torch.float, diagonalize_precision=True
            )
            imag_part = w.imag

        else:
            raise NotImplementedError

        imag_part = imag_scale * imag_part
        w = -real_part + 1j * imag_part

        # Initialize B
        if random_B:
            B = torch.randn(H, self.N // 2, dtype=dtype)

        else:
            B = torch.ones(H, self.N // 2, dtype=dtype)

        if normalize:
            norm = (
                -B / w
            )  # (H, N) # Result if you integrate the kernel with constant 1 function
            zeta = 2 * torch.sum(
                torch.abs(norm) ** 2, dim=-1, keepdim=True
            )  # Variance with a random C vector
            B = B / zeta**0.5

        P = torch.randn(self.rank, H, self.N // 2, dtype=dtype)

        if diagonal:
            P = P * 0.0
        V = torch.eye(self.N, dtype=dtype)[:, : self.N // 2]  # Only used in testing
        V = repeat(V, "n m -> h n m", h=H)

        return w, P, B, V
