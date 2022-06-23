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


def make_HiPPO(N, v='nv', measure="legs", lambda_n=1, fourier_type="fru", alpha=0, beta=1):
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
        if v == 'nv':
            A, B = build_LegT(N=N, lambda_n=lambda_n)
        else:
            A, B = build_LegT_V(N=N, lambda_n=lambda_n) 
        
    elif measure == "lagt":
        if v == 'nv':
            A, B = build_LagT(alpha=alpha, beta=beta, N=N)
        else:
            A, B = build_LagT_V(alpha=alpha, beta=beta, N=N)
        
    elif measure == "legs":
        if v == 'nv':
            A, B = build_LegS(N=N)
        else:
            A, B = build_LegS_V(N=N)
        
    elif measure == "fourier":
        if v == 'nv':
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


# ----------------------------------------------------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------------------------------------------------
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
    L = jnp.exp(.5 * (ss.gammaln(jnp.arange(N)+alpha+1) - ss.gammaln(jnp.arange(N)+1)))
    inv_L = 1./L[:, None]
    pre_A = (jnp.eye(N) * ((1 + beta) / 2)) + jnp.tril(jnp.ones((N, N)), -1)
    pre_B = ss.binom(alpha + jnp.arange(N), jnp.arange(N))[:, None]
    
    A = -inv_L * pre_A * L[None, :]
    B =  jnp.exp(-.5 * ss.gammaln(1-alpha)) * jnp.power(beta, (1-alpha)/2) * inv_L * pre_B 
    
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

    L = jnp.exp(.5 * (ss.gammaln(jnp.arange(N)+alpha+1) - ss.gammaln(jnp.arange(N)+1)))
    A = (1./L[:, None]) * A * L[None, :]
    B = (1./L[:, None]) * B * jnp.exp(-.5 * ss.gammaln(1-alpha)) * beta**((1-alpha)/2)
    
    return A, B

# ----------------------------------------------------------------------------------------------------------------------
#Scaled Legendre (LegS) vectorized
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
    pre_D = jnp.sqrt(jnp.diag(2*q+1))
    B = D = jnp.diag(pre_D)[:, None]
    
    A_base = (-jnp.sqrt(2*n+1)) * jnp.sqrt(2*k+1)
    case_2 = (n+1)/(2*n+1)
    
    A = jnp.where(n > k, A_base, 0.0) # if n > k, then A_base is used, otherwise 0
    A = jnp.where(n == k, (A_base * case_2), A) # if n == k, then A_base is used, otherwise A

    return A, B

#Scaled Legendre (LegS), non-vectorized
def build_LegS(N):
    """
        The, non-vectorized implementation of the, measure derived from the Scaled Legendre basis. 
        
        Args:
            N (int): Order of coefficients to describe the orthogonal polynomial that is the HiPPO projection.
            
        Returns:
            A (jnp.ndarray): The A HiPPO matrix.
            B (jnp.ndarray): The B HiPPO matrix.
    """
    q = jnp.arange(N, dtype=jnp.float64)  # q represents the values 1, 2, ..., N each column has
    k, n = jnp.meshgrid(q, q)
    r = 2 * q + 1
    M = -(jnp.where(n >= k, r, 0) - jnp.diag(q)) # represents the state matrix M 
    D = jnp.sqrt(jnp.diag(2 * q + 1)) # represents the diagonal matrix D $D := \text{diag}[(2n+1)^{\frac{1}{2}}]^{N-1}_{n=0}$
    A = D @ M @ jnp.linalg.inv(D)
    B = jnp.diag(D)[:, None]
    B = B.copy() # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
    
    return A, B

# ----------------------------------------------------------------------------------------------------------------------
# Fourier Basis OPs and functions - vectorized
def build_Fourier_V(N, fourier_type='fru'): 
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
    
    if fourier_type == "fru": # Fourier Recurrent Unit (FRU) - vectorized
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
        
    elif fourier_type == "fout": # truncated Fourier (FouT) - vectorized
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
        
    elif fourier_type == "fourd":
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

def build_Fourier(N, fourier_type='fru'):
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
    freqs = jnp.arange(N//2)
    
    if fourier_type == "fru": # Fourier Recurrent Unit (FRU) - non-vectorized
        d = jnp.stack([jnp.zeros(N//2), freqs], axis=-1).reshape(-1)[1:]
        A = jnp.pi*(-jnp.diag(d, 1) + jnp.diag(d, -1))
        
        B = jnp.zeros(A.shape[1])
        B = B.at[0::2].set(jnp.sqrt(2))
        B = B.at[0].set(1)
        
        A = A - B[:, None] * B[None, :]
        B = B[:, None]

    elif fourier_type == "fout": # truncated Fourier (FouT) - non-vectorized
        freqs *= 2
        d = jnp.stack([jnp.zeros(N//2), freqs], axis=-1).reshape(-1)[1:]
        A = jnp.pi*(-jnp.diag(d, 1) + jnp.diag(d, -1))
        
        B = jnp.zeros(A.shape[1])
        B = B.at[0::2].set(jnp.sqrt(2))
        B = B.at[0].set(1)

        # Subtract off rank correction - this corresponds to the other endpoint u(t-1) in this case
        A = A - B[:, None] * B[None, :] * 2
        B = B[:, None] * 2
        
    elif fourier_type == "fourd":
        d = jnp.stack([jnp.zeros(N//2), freqs], axis=-1).reshape(-1)[1:]
        A = jnp.pi*(-jnp.diag(d, 1) + jnp.diag(d, -1))
        
        B = jnp.zeros(A.shape[1])
        B = B.at[0::2].set(jnp.sqrt(2))
        B = B.at[0].set(1)

        # Subtract off rank correction - this corresponds to the other endpoint u(t-1) in this case
        A = A - 0.5 * B[:, None] * B[None, :]
        B = 0.5 * B[:, None]
        
            
    return A, B


class HiPPO(jnn.Module):
    '''
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
    '''
    N: int 
    max_length: int 
    measure: str 
    step: float 
    GBT_alpha: float 
    seq_L: int 
    v: str 
    lambda_n: float
    fourier_type: str
    alpha: float
    beta: float
    
    def setup(self):
        A, B = make_HiPPO(N=self.N,
                          v=self.v, 
                          measure=self.measure, 
                          lambda_n=self.lambda_n, 
                          fourier_type=self.fourier_type,
                          alpha=self.alpha,
                          beta=self.beta)
        
        self.A = A
        self.B = B #.squeeze(-1)
        self.C = jnp.ones((1, self.N)).squeeze(0)
        self.D = jnp.zeros((1,))
        
        if self.measure == "legt":
            L = self.seq_L
            vals = jnp.arange(0.0, 1.0, L)
            n = jnp.arange(self.N)[:, None]
            x = 1 - 2 * vals
            self.eval_matrix = ss.eval_legendre(n, x).T
            
        elif self.measure == "legs":
            L = self.max_length
            vals = jnp.linspace(0.0, 1.0, L)
            n = jnp.arange(self.N)[:, None]
            x =  2 * vals - 1
            self.eval_matrix = (B[:, None] * ss.eval_legendre(n, x)).T
        else:
            raise ValueError("invalid measure")
        
    def __call__(self, u, kernel=False):
        if not kernel:
            Ab, Bb, Cb, Db = self.collect_SSM_vars(self.A, self.B, self.C, self.D, u, alpha=self.GBT_alpha)
            c_k = self.scan_SSM(Ab, Bb, Cb, Db, u, x0=jnp.zeros((self.N, )))[1]
        else:
            Ab, Bb, Cb, Db = self.discretize(self.A, self.B, self.C, self.D, step=self.step, alpha=self.GBT_alpha)
            c_k = self.causal_convolution(u, self.K_conv(Ab, Bb, Cb, Db, L=self.max_length))
            
        return c_k
    
    def reconstruct(self, c):
        '''
        Uses coeffecients to reconstruct the signal
        
        Args: 
            c (jnp.ndarray): coefficients of the HiPPO projection
            
        Returns:
            reconstructed signal
        '''
        return (self.eval_matrix @ jnp.expand_dims(c, -1)).squeeze(-1)
    
    def discretize(self, A, B, C, D, step, alpha=0.5):
        '''
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
        '''
        I = jnp.eye(A.shape[0])
        GBT = jnp.linalg.inv(I - (step * alpha * A))
        GBT_A = GBT @ (I + (step * (1-alpha) * A))
        GBT_B = (step * GBT) @ B
        
        if alpha > 1: # Zero-order Hold
            GBT_A = jax.scipy.linalg.expm(step * A)
            GBT_B = (jnp.linalg.inv(A) @ (jax.scipy.linalg.expm(step * A) - I)) @ B 
        
        return GBT_A, GBT_B, C, D
    
    def collect_SSM_vars(self, A, B, C, D, u, alpha=0.5):
        '''
        turns the continuos HiPPO matrix components into discrete ones
        
        Args:
            A (jnp.ndarray): matrix to be discretized
            B (jnp.ndarray): matrix to be discretized
            C (jnp.ndarray): matrix to be discretized
            D (jnp.ndarray): matrix to be discretized
            u (jnp.ndarray): input signal
            alpha (float, optional): used for determining which generalized bilinear transformation to use
            
        Returns:
            Ab (jnp.ndarray): discrete form of the HiPPO matrix
            Bb (jnp.ndarray): discrete form of the HiPPO matrix
            Cb (jnp.ndarray): discrete form of the HiPPO matrix
            Db (jnp.ndarray): discrete form of the HiPPO matrix
        '''
        L = u.shape[0]
        N = A.shape[0]
        assert L == self.seq_L
        assert N == self.N

        Ab, Bb, Cb, Db = self.discretize(A, B, C, D, step=1.0/L, alpha=alpha)
    
        return Ab, Bb, Cb, Db
    
    def scan_SSM(self, Ab, Bb, Cb, Db, u, x0):
        '''
        This is for returning the discretized hidden state often needed for an RNN. 
        Args:
            Ab (jnp.ndarray): the discretized A matrix
            Bb (jnp.ndarray): the discretized B matrix
            Cb (jnp.ndarray): the discretized C matrix
            u (jnp.ndarray): the input sequence
            x0 (jnp.ndarray): the initial hidden state
        Returns:
            the next hidden state (aka coefficients representing the function, f(t))
        '''
        def step(x_k_1, u_k):
            '''
            Get descretized coefficients of the hidden state by applying HiPPO matrix to input sequence, u_k, and previous hidden state, x_k_1.
            Args:
                x_k_1: previous hidden state
                u_k: output from function f at, descritized, time step, k.
            
            Returns: 
                x_k: current hidden state
                y_k: current output of hidden state applied to Cb (sorry for being vague, I just dont know yet)
            '''

            x_k = (Ab @ x_k_1) + (Bb @ u_k)
            y_k = (Cb @ x_k) + (Db @ u_k)
            
            return x_k, y_k

        return jax.lax.scan(step, x0, u)
