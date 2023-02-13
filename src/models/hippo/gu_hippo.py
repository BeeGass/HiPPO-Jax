import math

import einops
import functorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg as la
from scipy import signal
from scipy import special as ss

from src.models.hippo.gu_transition import GuTransMatrix
from src.models.hippo.transition import TransMatrix
from src.models.hippo.unroll import (
    basis,
    measure,
    variable_unroll_matrix,
    variable_unroll_matrix_sequential,
)


class gu_HiPPO_LSI(nn.Module):
    """Vanilla HiPPO-LegS model (scale invariant instead of time invariant)"""

    def __init__(
        self,
        N,
        method="legs",
        max_length=1024,
        discretization=0.5,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
    ):
        """
        max_length: maximum sequence length
        """
        super().__init__()
        self.N = N
        matrices = GuTransMatrix(
            N=N, measure=method, lambda_n=lambda_n, alpha=alpha, beta=beta
        )
        A = np.asarray(matrices.A, dtype=np.float32)
        B = np.asarray(matrices.B, dtype=np.float32)
        # A, B = transition(method, N)
        B = B.squeeze(-1)
        A_stacked = np.empty((max_length, N, N), dtype=A.dtype)
        B_stacked = np.empty((max_length, N), dtype=B.dtype)
        for t in range(1, max_length + 1):
            At = A / t
            Bt = B / t
            if discretization == 0.0:  # forward
                A_stacked[t - 1] = np.eye(N) + At
                B_stacked[t - 1] = Bt
            elif discretization == 1.0:  # backward
                A_stacked[t - 1] = la.solve_triangular(
                    np.eye(N) - At, np.eye(N), lower=True
                )
                B_stacked[t - 1] = la.solve_triangular(np.eye(N) - At, Bt, lower=True)
            elif discretization == 0.5:  # bilinear
                # A_stacked[t - 1] = la.solve_triangular(
                #     np.eye(N) - At / 2, np.eye(N) + At / 2, lower=True
                # )
                # B_stacked[t - 1] = la.solve_triangular(
                #     np.eye(N) - At / 2, Bt, lower=True
                # )
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
                # A_stacked[t - 1] = la.expm(At)
                B_stacked[t - 1] = la.solve_triangular(
                    A, A_stacked[t - 1] @ B - B, lower=True
                )

                # A_stacked[t - 1] = la.expm(At)
                # B_stacked[t - 1] = la.inv(A) @ (la.expm(At) - np.eye(A.shape[0])) @ B

        # self.register_buffer('A_stacked', torch.Tensor(A_stacked)) # (max_length, N, N)
        # self.register_buffer('B_stacked', torch.Tensor(B_stacked)) # (max_length, N)

        self.A_stacked = torch.Tensor(A_stacked.copy())  # (max_length, N, N)
        self.B_stacked = torch.Tensor(B_stacked.copy())  # (max_length, N)

        vals = np.linspace(0.0, 1.0, max_length)
        self.eval_matrix = torch.from_numpy(
            np.asarray(
                ((B[:, None] * ss.eval_legendre(np.arange(N)[:, None], 2 * vals - 1)).T)
            )
        )

    def forward(self, inputs, fast=True):
        """
        inputs : (length, ...)
        output : (length, ..., N) where N is the order of the HiPPO projection
        """

        L = inputs.shape[0]

        inputs = inputs.unsqueeze(-1)
        u = torch.transpose(inputs, 0, -2)
        u = u * self.B_stacked[:L]
        # print(f"Gu - u * self.B_stacked[:L]: {u}")
        u = torch.transpose(u, 0, -2)  # (length, ..., N)

        if fast:
            result = variable_unroll_matrix(self.A_stacked[:L], u)
            return result

        c = torch.zeros(u.shape[1:]).to(inputs)
        cs = []
        for t, f in enumerate(inputs):

            batched_linear = functorch.vmap(F.linear, in_dims=(0, None))
            batched_hadamard = functorch.vmap(torch.mul, in_dims=(None, 0))

            part1 = batched_linear(c, self.A_stacked[t])
            part2 = batched_hadamard(self.B_stacked[t], f)

            c = part1 + part2

            cs.append(c)
        return torch.stack(cs, dim=0), c

    def reconstruct(self, c):
        eval_matrix = self.eval_matrix

        print(f"gu LSI eval_matrix.shape: {(eval_matrix).shape}")
        print(f"c.shape: {c.shape}")

        y = None
        if len(c.shape) == 3:
            c = einops.rearrange(c, "batch input_size N -> batch N input_size")
            y = functorch.vmap(torch.matmul, in_dims=(None, 0))(eval_matrix.to(c), c)
        elif len(c.shape) == 4:
            c = einops.rearrange(
                c, "seq_len batch input_size N -> batch seq_len N input_size"
            )
            time_dot = functorch.vmap(torch.matmul, in_dims=(None, 0))
            batch_time_dot = functorch.vmap(time_dot, in_dims=(None, 0))
            y = batch_time_dot(eval_matrix.to(c), c)
        else:
            raise ValueError(
                "c must be of shape (batch size, input length, N) or (batch seq_len input_size N)"
            )

        return y


class gu_HiPPO_LTI(nn.Module):
    """Linear time invariant x' = Ax + Bu"""

    def __init__(
        self,
        N,
        method="legt",
        dt=1.0,
        T=1.0,
        discretization=0.5,
        lambda_n=1.0,
        alpha=0.0,
        beta=1.0,
        c=0.0,
    ):
        """
        N: the order of the HiPPO projection
        dt: discretization step size - should be roughly inverse to the length of the sequence
        """
        super().__init__()

        self.method = method
        self.N = N
        self.dt = dt
        self.T = T
        self.c = c

        matrices = GuTransMatrix(
            N=N, measure=method, lambda_n=lambda_n, alpha=alpha, beta=beta
        )
        A = np.asarray(matrices.A, dtype=np.float32)
        B = np.asarray(matrices.B, dtype=np.float32)
        # A, B = transition(method, N)
        A = A + (np.eye(N) * c)
        self.A = A
        self.B = B.squeeze(-1)
        self.measure_fn = measure(method)

        C = np.ones((1, N))
        D = np.zeros((1,))
        if type(discretization) in [float, int]:
            dA, dB, _, _, _ = signal.cont2discrete(
                (A, B, C, D), dt=dt, method="gbt", alpha=discretization
            )
        else:
            dA, dB, _, _, _ = signal.cont2discrete((A, B, C, D), dt=dt, method="zoh")

        dB = dB.squeeze(-1)

        self.dA = torch.Tensor(dA.copy())  # (N, N)
        self.dB = torch.Tensor(dB.copy())  # (N, )

        self.vals = np.arange(0.0, T, dt)
        self.eval_matrix = basis(self.method, self.N, self.vals, c=self.c)  # (T/dt, N)
        self.measure = measure(self.method)(self.vals)

    def forward(self, inputs, fast=True):
        """
        inputs : (length, ...)
        output : (length, ..., N) where N is the order of the HiPPO projection
        """

        inputs = inputs.unsqueeze(-1)
        u = inputs * self.dB  # (length, ..., N)

        if fast:
            dA = einops.repeat(self.dA, "m n -> l m n", l=u.size(0))
            return variable_unroll_matrix(dA, u)

        c = torch.zeros(u.shape[1:]).to(inputs)
        cs = []
        for f in inputs:

            batched_linear = functorch.vmap(F.linear, in_dims=(0, None))
            batched_hadamard = functorch.vmap(torch.mul, in_dims=(None, 0))

            part1 = batched_linear(c, self.dA)
            part2 = batched_hadamard(self.dB, f)

            c = part1 + part2

            cs.append(c)
        return torch.stack(cs, dim=0), c

    def reconstruct(self, c, evals=None):
        """
        c: (..., N,) HiPPO coefficients (same as x(t) in S4 notation)
        output: (..., L,)
        """
        if evals is not None:
            eval_matrix = basis(self.method, self.N, evals)
        else:
            eval_matrix = self.eval_matrix

        m = self.measure[self.measure != 0.0]

        print(f"gu LTI eval_matrix.shape: {(eval_matrix).shape}")
        print(f"c.shape: {c.shape}")

        y = None
        if len(c.shape) == 3:
            c = einops.rearrange(c, "batch input_size N -> batch N input_size")
            y = functorch.vmap(torch.matmul, in_dims=(None, 0))(eval_matrix.to(c), c)
        elif len(c.shape) == 4:
            c = einops.rearrange(
                c, "seq_len batch input_size N -> batch seq_len N input_size"
            )
            time_dot = functorch.vmap(torch.matmul, in_dims=(None, 0))
            batch_time_dot = functorch.vmap(time_dot, in_dims=(None, 0))
            y = batch_time_dot(eval_matrix.to(c), c)
        else:
            raise ValueError(
                "c must be of shape (batch size, input length, N) or (batch seq_len input_size N)"
            )

        return y
