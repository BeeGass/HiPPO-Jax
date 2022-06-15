## import packages
import jax
import jax.numpy as jnp

from flax import linen as nn

from jax.nn.initializers import lecun_normal, uniform
from jax.numpy.linalg import eig, inv, matrix_power
from jax.scipy.signal import convolve

import os
import requests

from scipy import linalg as la
from scipy import signal
from scipy import special as ss

## setup JAX to use TPUs if available
try:
    url = 'http:' + os.environ['TPU_NAME'].split(':')[1] + ':8475/requestversion/tpu_driver_nightly'
    resp = requests.post(url)
    jax.config.FLAGS.jax_xla_backend = 'tpu_driver'
    jax.config.FLAGS.jax_backend_target = os.environ['TPU_NAME']
except:
    pass

# ----------------------------------------------------------------------------------------------------------------------
# ---------------- HiPPO -------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def make_HiPPO(N, v='nv', HiPPO_type="legs", lambda_n=1, fourier_type="FRU", alpha=0, beta=1):
    A = None
    B = None
    for k in range(1, N + 1):
        for n in range(1, N + 1):
            if HiPPO_type == "legt":
                if v == 'nv':
                    A, B = build_LegT(N=N, lambda_n=lambda_n)
                else:
                 A, B = build_LegT_V(N=N, lambda_n=lambda_n) 
                
            elif HiPPO_type == "lagt":
                if v == 'nv':
                    A, B = build_LagT(alpha=alpha, beta=beta, N=N)
                else:
                    A, B = build_LagT_V(alpha=alpha, beta=beta, N=N)
                
            elif HiPPO_type == "legs":
                if v == 'nv':
                    A, B = build_LegS(N=N)
                else:
                    A, B = build_LegS_V(N=N)
                
            elif HiPPO_type == "fourier":
                if v == 'nv':
                    A, B = build_Fourier(N=N, fourier_type=fourier_type)
                else:
                    A, B = build_Fourier_V(N=N, fourier_type=fourier_type)
                
            elif HiPPO_type == "random":
                A = jnp.random.randn(N, N) / N
                B = jnp.random.randn(N, 1)
                
            elif HiPPO_type == "diagonal":
                A = -jnp.diag(jnp.exp(jnp.random.randn(N)))
                B = jnp.random.randn(N, 1)
                
            else:
                raise ValueError("Invalid HiPPO type")
    
    return -jnp.array(A), B


# ----------------------------------------------------------------------------------------------------------------------
# Translated Legendre (LegT) - non-vectorized
def build_LegT(N, legt_type="legt"):
    Q = jnp.arange(N, dtype=jnp.float64)
    pre_R = (2*Q + 1)
    k, n = jnp.meshgrid(Q, Q)
        
    if legt_type == "legt":
        R = jnp.sqrt(pre_R)
        A = R[:, None] * jnp.where(n < k, (-1.)**(n-k), 1) * R[None, :]
        B = R[:, None]
        A = -A

        # Halve again for timescale correctness
        # A, B = A/2, B/2
        #A *= 0.5
        #B *= 0.5
        
    elif legt_type == "lmu":
        R = pre_R[:, None]
        A = jnp.where(n < k, -1, (-1.)**(n-k+1)) * R
        B = (-1.)**Q[:, None] * R
        
    return A, B

# Translated Legendre (LegT) - vectorized
def build_LegT_V(N, lambda_n=1):
    q = jnp.arange(N, dtype=jnp.float64)
    k, n = jnp.meshgrid(q, q)
    case = jnp.power(-1.0, (n-k))
    A = None
    B = None
    
    if lambda_n == 1:
        A_base = -jnp.sqrt(2*n+1) * jnp.sqrt(2*k+1)
        pre_D = jnp.sqrt(jnp.diag(2*q+1))
        B = D = jnp.diag(pre_D)[:, None]
        A = jnp.where(k <= n, A_base, A_base * case) # if n >= k, then case_2 * A_base is used, otherwise A_base
        
    elif lambda_n == 2: #(jnp.sqrt(2*n+1) * jnp.power(-1, n)):
        A_base = -(2*n+1)
        B = jnp.diag((2*q+1) * jnp.power(-1, n))[:, None]
        A = jnp.where(k <= n, A_base * case, A_base) # if n >= k, then case_2 * A_base is used, otherwise A_base

    return A, B

# ----------------------------------------------------------------------------------------------------------------------
# Translated Laguerre (LagT) - non-vectorized
def build_LagT(alpha, beta, N):
    A = -jnp.eye(N) * (1 + beta) / 2 - jnp.tril(jnp.ones((N, N)), -1)
    B = ss.binom(alpha + jnp.arange(N), jnp.arange(N))[:, None]

    L = jnp.exp(.5 * (ss.gammaln(jnp.arange(N)+alpha+1) - ss.gammaln(jnp.arange(N)+1)))
    A = (1./L[:, None]) * A * L[None, :]
    B = (1./L[:, None]) * B * jnp.exp(-.5 * ss.gammaln(1-alpha)) * beta**((1-alpha)/2)
    
    return A, B

# Translated Laguerre (LagT) - non-vectorized
def build_LagT_V(alpha, beta, N):
    L = jnp.exp(.5 * (ss.gammaln(jnp.arange(N)+alpha+1) - ss.gammaln(jnp.arange(N)+1)))
    inv_L = 1./L[:, None]
    pre_A = (jnp.eye(N) * ((1 + beta) / 2)) + jnp.tril(jnp.ones((N, N)), -1)
    pre_B = ss.binom(alpha + jnp.arange(N), jnp.arange(N))[:, None]
    
    A = -inv_L * pre_A * L[None, :]
    B =  jnp.exp(-.5 * ss.gammaln(1-alpha)) * jnp.power(beta, (1-alpha)/2) * inv_L * pre_B 
    
    return A, B

# ----------------------------------------------------------------------------------------------------------------------
#Scaled Legendre (LegS), non-vectorized
def build_LegS(N):
    q = jnp.arange(N, dtype=jnp.float64)  # q represents the values 1, 2, ..., N each column has
    k, n = jnp.meshgrid(q, q)
    r = 2 * q + 1
    M = -(jnp.where(n >= k, r, 0) - jnp.diag(q)) # represents the state matrix M 
    D = jnp.sqrt(jnp.diag(2 * q + 1)) # represents the diagonal matrix D $D := \text{diag}[(2n+1)^{\frac{1}{2}}]^{N-1}_{n=0}$
    A = D @ M @ jnp.linalg.inv(D)
    B = jnp.diag(D)[:, None]
    B = B.copy() # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
    
    return A, B

#Scaled Legendre (LegS) vectorized
def build_LegS_V(N):
    q = jnp.arange(N, dtype=jnp.float64)
    k, n = jnp.meshgrid(q, q)
    pre_D = jnp.sqrt(jnp.diag(2*q+1))
    B = D = jnp.diag(pre_D)[:, None]
    
    A_base = (-jnp.sqrt(2*n+1)) * jnp.sqrt(2*k+1)
    case_2 = (n+1)/(2*n+1)
    
    A = jnp.where(n > k, A_base, 0.0) # if n > k, then A_base is used, otherwise 0
    A = jnp.where(n == k, (A_base * case_2), A) # if n == k, then A_base is used, otherwise A
    
    return A, B

# ----------------------------------------------------------------------------------------------------------------------
def build_Fourier(N, fourier_type='FRU'):
    freqs = jnp.arange(N//2)
    
    if fourier_type == "FRU": # Fourier Recurrent Unit (FRU) - non-vectorized
        d = jnp.stack([jnp.zeros(N//2), freqs], axis=-1).reshape(-1)[1:]
        A = jnp.pi*(-jnp.diag(d, 1) + jnp.diag(d, -1))
        
        B = jnp.zeros(A.shape[1])
        B = B.at[0::2].set(jnp.sqrt(2))
        B = B.at[0].set(1)
        
        A = A - B[:, None] * B[None, :]
        B = B[:, None]

    elif fourier_type == "FouT": # truncated Fourier (FouT) - non-vectorized
        freqs *= 2
        d = jnp.stack([jnp.zeros(N//2), freqs], axis=-1).reshape(-1)[1:]
        A = jnp.pi*(-jnp.diag(d, 1) + jnp.diag(d, -1))
        
        B = jnp.zeros(A.shape[1])
        B = B.at[0::2].set(jnp.sqrt(2))
        B = B.at[0].set(1)

        # Subtract off rank correction - this corresponds to the other endpoint u(t-1) in this case
        A = A - B[:, None] * B[None, :] * 2
        B = B[:, None] * 2
        
    elif fourier_type == "fourier_decay":
        d = jnp.stack([jnp.zeros(N//2), freqs], axis=-1).reshape(-1)[1:]
        A = jnp.pi*(-jnp.diag(d, 1) + jnp.diag(d, -1))
        
        B = jnp.zeros(A.shape[1])
        B = B.at[0::2].set(jnp.sqrt(2))
        B = B.at[0].set(1)

        # Subtract off rank correction - this corresponds to the other endpoint u(t-1) in this case
        A = A - 0.5 * B[:, None] * B[None, :]
        B = 0.5 * B[:, None]
        
            
    return A, B

# Fourier Basis OPs and functions - vectorized
def build_Fourier_V(N, fourier_type='FRU'):    
    q = jnp.arange((N//2)*2, dtype=jnp.float64)
    k, n = jnp.meshgrid(q, q)
    
    n_odd = n % 2 == 0
    k_odd = k % 2 == 0
    
    case_1 = (n==k) & (n==0)
    case_2_3 = ((k==0) & (n_odd)) | ((n==0) & (k_odd))
    case_4 = (n_odd) & (k_odd)
    case_5 = (n-k==1) & (k_odd)
    case_6 = (k-n==1) & (n_odd)
    
    A = None
    B = None
    
    if fourier_type == "FRU": # Fourier Recurrent Unit (FRU) - vectorized
        A = jnp.diag(jnp.stack([jnp.zeros(N//2), jnp.zeros(N//2)], axis=-1).reshape(-1))
        B = jnp.zeros(A.shape[1], dtype=jnp.float64)
        q = jnp.arange((N//2)*2, dtype=jnp.float64)
        
        A = jnp.where(case_1, -1.0, 
                    jnp.where(case_2_3, -jnp.sqrt(2),
                                jnp.where(case_4, -2, 
                                        jnp.where(case_5, jnp.pi * (n//2), 
                                                    jnp.where(case_6, -jnp.pi * (k//2), 0.0)))))
        
        B = B.at[::2].set(jnp.sqrt(2))
        B = B.at[0].set(1)
        
    elif fourier_type == "FouT": # truncated Fourier (FouT) - vectorized
        A = jnp.diag(jnp.stack([jnp.zeros(N//2), jnp.zeros(N//2)], axis=-1).reshape(-1))
        B = jnp.zeros(A.shape[1], dtype=jnp.float64)
        k, n = jnp.meshgrid(q, q)
        n_odd = n % 2 == 0
        k_odd = k % 2 == 0
        
        A = jnp.where(case_1, -1.0, 
                    jnp.where(case_2_3, -jnp.sqrt(2),
                                jnp.where(case_4, -2, 
                                        jnp.where(case_5, jnp.pi * (n//2), 
                                                    jnp.where(case_6, -jnp.pi * (k//2), 0.0)))))
        
        B = B.at[::2].set(jnp.sqrt(2))
        B = B.at[0].set(1)
        
        A = 2 * A
        B = 2 * B
        
    elif fourier_type == "fourier_decay":
        A = jnp.diag(jnp.stack([jnp.zeros(N//2), jnp.zeros(N//2)], axis=-1).reshape(-1))
        B = jnp.zeros(A.shape[1], dtype=jnp.float64)
        
        A = jnp.where(case_1, -1.0, 
                    jnp.where(case_2_3, -jnp.sqrt(2),
                                jnp.where(case_4, -2, 
                                        jnp.where(case_5, 2 * jnp.pi * (n//2), 
                                                    jnp.where(case_6, 2 * -jnp.pi * (k//2), 0.0)))))
        
        B = B.at[::2].set(jnp.sqrt(2))
        B = B.at[0].set(1)
        
        A = 0.5 * A
        B = 0.5 * B
        
    
    
    B = B[:, None]
        
    return A, B


def discretize(A, B, C, step, alpha=0.5, ZOH_bool=False):
    '''
    - forward Euler corresponds to α = 0,
    - backward Euler corresponds to α = 1,
    - bilinear corresponds to α = 0.5,
    '''
    I = jnp.eye(A.shape[0])
    GBT = jnp.linalg.inv(I - ((step * alpha) * A))
    GBT_A = GBT @ (I + ((step * (1-alpha)) * A))
    GBT_B = (step * GBT) @ B
    
    if ZOH_bool: # Zero-order Hold
        GBT_A = jax.scipy.linalg.expm(step * A)
        GBT_B = (jnp.linalg.inv(A) @ (jax.scipy.linalg.expm(step * A) - I)) @ B 
    
    return GBT_A, GBT_B, C   

class HiPPO_LegS(nn.Module):
    """ Vanilla HiPPO-LegS model (scale invariant instead of time invariant) """
    N: int
    max_length: int = 1024
    measure: str
    step: float
    
    def setup(self):
        A, B = make_HiPPO(N=self.N, v='v', HiPPO_type="legs", lambda_n=1, fourier_type="FRU", alpha=0, beta=1)
        B = B.squeeze(-1)
        A, B, C = discretize(A, B, C, step, alpha=0.5, ZOH_bool=False)
        vals = jnp.linspace(0.0, 1.0, self.max_length)
        self.eval_matrix = (B[:, None] * ss.eval_legendre(jnp.arange(self.N)[:, None], 2 * vals - 1)).T

        
        
        
    def __call__():
        pass
    
    def ODE(self):
        pass
        
        
    def forward(self, inputs, fast=False):
        """
        inputs : (length, ...)
        output : (length, ..., N) where N is the order of the HiPPO projection
        """

        L = inputs.shape[0]

        inputs = inputs.unsqueeze(-1)
        u = torch.transpose(inputs, 0, -2)
        u = u * self.B_stacked[:L]
        u = torch.transpose(u, 0, -2) # (length, ..., N)

        if fast:
            result = unroll.variable_unroll_matrix(self.A_stacked[:L], u)
        else:
            result = unroll.variable_unroll_matrix_sequential(self.A_stacked[:L], u)
        return result

    def reconstruct(self, c):
        a = self.eval_matrix @ c.unsqueeze(-1)
        return a.squeeze(-1)
