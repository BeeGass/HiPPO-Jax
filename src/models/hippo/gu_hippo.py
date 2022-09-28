import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from scipy import linalg as la
from scipy import signal
from scipy import special as ss

from src.models.hippo.gu_transition import GuTransMatrix
from src.models.hippo.transition import TransMatrix

import math


class HiPPO_LegS(nn.Module):
    """Vanilla HiPPO-LegS model (scale invariant instead of time invariant)"""

    def __init__(self, N, max_length=1024, measure="legs", discretization="bilinear"):
        """
        max_length: maximum sequence length
        """
        super().__init__()
        self.N = N
        legs_matrices = GuTransMatrix(N=N, measure=measure)
        A = legs_matrices.A_matrix
        B = legs_matrices.B_matrix
        # A, B = transition(measure, N)
        B = B.squeeze(-1)
        A_stacked = np.empty((max_length, N, N), dtype=A.dtype)
        B_stacked = np.empty((max_length, N), dtype=B.dtype)
        for t in range(1, max_length + 1):
            At = A / t
            Bt = B / t
            if discretization == "forward":
                A_stacked[t - 1] = np.eye(N) + At
                B_stacked[t - 1] = Bt
            elif discretization == "backward":
                A_stacked[t - 1] = la.solve_triangular(
                    np.eye(N) - At, np.eye(N), lower=True
                )
                B_stacked[t - 1] = la.solve_triangular(np.eye(N) - At, Bt, lower=True)
            elif discretization == "bilinear":
                alpha = 0.5
                A_stacked[t - 1] = np.linalg.lstsq(
                    np.eye(N) - (At * alpha), np.eye(N) + (At * alpha), rcond=None
                )[
                    0
                ]  # TODO: Referencing this: https://stackoverflow.com/questions/64527098/numpy-linalg-linalgerror-singular-matrix-error-when-trying-to-solve
                B_stacked[t - 1] = np.linalg.lstsq(
                    np.eye(N) - (At * alpha), Bt, rcond=None
                )[0]
            else:  # ZOH
                A_stacked[t - 1] = la.expm(A * (math.log(t + 1) - math.log(t)))
                B_stacked[t - 1] = la.solve_triangular(
                    A, A_stacked[t - 1] @ B - B, lower=True
                )
        self.A_stacked = torch.Tensor(A_stacked.copy())  # (max_length, N, N)
        self.B_stacked = torch.Tensor(B_stacked.copy())  # (max_length, N)
        vals = np.linspace(0.0, 1.0, max_length)
        self.eval_matrix = torch.from_numpy(
            np.asarray(
                ((B[:, None] * ss.eval_legendre(np.arange(N)[:, None], 2 * vals - 1)).T)
            )
        )

    def forward(self, inputs, fast=False):
        """
        inputs : (length, ...)
        output : (length, ..., N) where N is the order of the HiPPO projection
        """
        result = None

        L = inputs.shape[0]

        u = inputs.unsqueeze(-1)
        u = torch.transpose(u, 0, -2)
        u = u * self.B_stacked[:L]  # c_k = A @ c_{k-1} + B @ f_k
        print(f"u - Gu: {u}")
        my_b = torch.Tensor(
            [
                [6.6666657e-01],
                [5.7735050e-01],
                [1.4907140e-01],
                [-2.3096800e-07],
                [-2.7939677e-09],
                [2.9616058e-07],
                [-2.2817403e-08],
                [-8.1490725e-08],
            ]
        )
        u = torch.transpose(u, 0, -2)  # (length, ..., N)

        # print(f"A_stacked: {self.A_stacked[:L]}")
        # print(f"B_stacked: {self.B_stacked[:L]}")

        if fast:
            result = variable_unroll_matrix(self.A_stacked[:L], u)

        else:
            result = variable_unroll_matrix_sequential(self.A_stacked[:L], u)

        return result

    def reconstruct(self, c):
        a = self.eval_matrix @ c.unsqueeze(-1)
        return a.squeeze(-1)


class HiPPO_LegT(nn.Module):
    def __init__(self, N, dt=1.0, discretization="bilinear"):
        """
        N: the order of the HiPPO projection
        dt: discretization step size - should be roughly inverse to the length of the sequence
        """
        super().__init__()
        self.N = N
        # A, B = transition('lmu', N)
        legt_matrices = GuTransMatrix(N=N, measure="legt", lambda_n=1.0)
        A = legt_matrices.A_matrix
        B = legt_matrices.B_matrix
        C = np.ones((1, N))
        D = np.zeros((1,))
        # dt, discretization options
        A, B, _, _, _ = signal.cont2discrete((A, B, C, D), dt=dt, method=discretization)

        B = B.squeeze(-1)

        self.register_buffer("A", torch.Tensor(A))  # (N, N)
        self.register_buffer("B", torch.Tensor(B))  # (N,)

        # vals = np.linspace(0.0, 1.0, 1./dt)
        vals = np.arange(0.0, 1.0, dt)
        self.eval_matrix = torch.Tensor(
            ss.eval_legendre(np.arange(N)[:, None], 1 - 2 * vals).T
        )

    def forward(self, inputs):
        """
        inputs : (length, ...)
        output : (length, ..., N) where N is the order of the HiPPO projection
        """

        inputs = inputs.unsqueeze(-1)
        u = inputs * self.B  # (length, ..., N)

        c = torch.zeros(u.shape[1:])
        cs = []
        for f in inputs:
            c = F.linear(c, self.A) + self.B * f
            # print(f"f:\n{f}")
            cs.append(c)
        return torch.stack(cs, dim=0)

    def reconstruct(self, c):
        return (self.eval_matrix @ c.unsqueeze(-1)).squeeze(-1)
