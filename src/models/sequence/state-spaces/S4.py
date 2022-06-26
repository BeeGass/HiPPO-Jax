from functools import partial
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import lecun_normal, uniform
from jax.numpy.linalg import eig, inv, matrix_power
from jax.scipy.signal import convolve
from scipy import linalg as la
from scipy import signal
from scipy import special as ss
from numpy import ndarray


rng = jax.random.PRNGKey(1)

def random_SSM(rng, N):
    a_r, b_r, c_r = jax.random.split(rng, 3)
    A = jax.random.uniform(a_r, (N, N))
    B = jax.random.uniform(b_r, (N, 1))
    C = jax.random.uniform(c_r, (1, N))
    return A, B, C


def discretize(A, B, C, step):
    I = np.eye(A.shape[0])
    BL = inv(I - (step / 2.0) * A)
    Ab = BL @ (I + (step / 2.0) * A)
    Bb = (BL * step) @ B
    return Ab, Bb, C


def scan_SSM(Ab, Bb, Cb, u, x0):
    def step(x_k_1, u_k):
        x_k = Ab @ x_k_1 + Bb @ u_k
        y_k = Cb @ x_k
        return x_k, y_k

    return jax.lax.scan(step, x0, u)


def run_SSM(A, B, C, u):
    L = u.shape[0]
    N = A.shape[0]
    Ab, Bb, Cb = discretize(A, B, C, step=1.0 / L)

    # Run recurrence
    return scan_SSM(Ab, Bb, Cb, u[:, np.newaxis], np.zeros((N,)))[1]
    

def K_conv(Ab, Bb, Cb, L):
    return np.array(
        [(Cb @ matrix_power(Ab, l) @ Bb).reshape() for l in range(L)]
    ) # the kernal is L-length and is matrix C @ A^{i-1} @ B. This operation is done L times. 


def non_circular_convolution(u, K, nofft=False):
    if nofft:
        return convolve(u, K, mode="full")[: u.shape[0]]
    else:
        assert K.shape[0] == u.shape[0]
        ud = np.fft.rfft(np.pad(u, (0, K.shape[0])))
        Kd = np.fft.rfft(np.pad(K, (0, u.shape[0])))
        out = ud * Kd
        return np.fft.irfft(out)[: u.shape[0]]


def test_cnn_is_rnn(N=4, L=16, step=1.0 / 16):
    ssm = random_SSM(rng, N)
    u = jax.random.uniform(rng, (L,))
    jax.random.split(rng, 3)
    # RNN
    rec = run_SSM(*ssm, u)

    # CNN
    ssmb = discretize(*ssm, step=step)
    conv = non_circular_convolution(u, K_conv(*ssmb, L))

    # Check
    assert np.allclose(rec.ravel(), conv.ravel())


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


def example_legendre(N=8):
    # Random hidden state as coefficients
    import numpy as np
    import numpy.polynomial.legendre

    x = (np.random.rand(N) - 0.5) * 2
    t = np.linspace(-1, 1, 100)
    f = numpy.polynomial.legendre.Legendre(x)(t)

    # Plot
    import matplotlib.pyplot as plt
    import seaborn

    seaborn.set_context("talk")
    fig = plt.figure(figsize=(20, 10))
    ax = fig.gca(projection="3d")
    ax.plot(
        np.linspace(-25, (N - 1) * 100 + 25, 100),
        [0] * 100,
        zs=-1,
        zdir="x",
        color="black",
    )
    ax.plot(t, f, zs=N * 100, zdir="y", c="r")
    for i in range(N):
        coef = [0] * N
        coef[N - i - 1] = 1
        ax.set_zlim(-4, 4)
        ax.set_yticks([])
        ax.set_zticks([])
        # Plot basis function.
        f = numpy.polynomial.legendre.Legendre(coef)(t)
        ax.bar(
            [100 * i],
            [x[i]],
            zs=-1,
            zdir="x",
            label="x%d" % i,
            color="brown",
            fill=False,
            width=50,
        )
        ax.plot(t, f, zs=100 * i, zdir="y", c="b", alpha=0.5)
    ax.view_init(elev=40.0, azim=-45)
    fig.savefig("images/leg.png")



if False:
    example_legendre()


def log_step_initializer(dt_min=0.001, dt_max=0.1):
    def init(key, shape):
        return jax.random.uniform(key, shape) * (
            np.log(dt_max) - np.log(dt_min)
        ) + np.log(dt_min)

    return init



class SSMLayer(nn.Module):
    A: np.DeviceArray  # HiPPO
    N: int
    l_max: int
    decode: bool = False

    def setup(self):
        # SSM parameters
        self.B = self.param("B", lecun_normal(), (self.N, 1))
        self.C = self.param("C", lecun_normal(), (1, self.N))
        self.D = self.param("D", nn.initializers.ones, (1,))

        # Step parameter
        self.log_step = self.param("log_step", log_step_initializer(), (1,))

        step = np.exp(self.log_step)
        self.ssm = discretize(self.A, self.B, self.C, step=step)
        self.K = K_conv(*self.ssm, self.l_max)

        # RNN cache for long sequences
        self.x_k_1 = self.variable("cache", "cache_x_k", np.zeros, (self.N,))

    def __call__(self, u):
        if not self.decode:
            # CNN Mode
            return non_circular_convolution(u, self.K) + self.D * u
        else:
            # RNN Mode
            x_k, y_s = scan_SSM(*self.ssm, u[:, np.newaxis], self.x_k_1.value)
            if self.is_mutable_collection("cache"):
                self.x_k_1.value = x_k
            return y_s.reshape(-1).real + self.D * u


def cloneLayer(layer):
    return nn.vmap(
        layer,
        in_axes=1,
        out_axes=1,
        variable_axes={"params": 1, "cache": 1, "prime": 1},
        split_rngs={"params": True},
    )


def SSMInit(N):
    return partial(cloneLayer(SSMLayer), A=make_HiPPO(N), N=N)


class SequenceBlock(nn.Module):
    layer: nn.Module
    l_max: int
    dropout: float
    d_model: int
    training: bool = True
    decode: bool = False

    def setup(self):
        self.seq = self.layer(l_max=self.l_max, decode=self.decode)
        self.norm = nn.LayerNorm()
        self.out = nn.Dense(self.d_model)
        self.drop = nn.Dropout(
            self.dropout,
            broadcast_dims=[0],
            deterministic=not self.training,
        )

    def __call__(self, x):
        x2 = self.seq(x)
        z = self.drop(self.out(self.drop(nn.gelu(x2))))
        return self.norm(z + x)


class StackedModel(nn.Module):
    layer: nn.Module
    d_output: int
    d_model: int
    l_max: int
    n_layers: int
    dropout: float = 0.2
    training: bool = True
    classification: bool = False
    decode: bool = False

    def setup(self):
        self.encoder = nn.Dense(self.d_model)
        self.decoder = nn.Dense(self.d_output)
        self.layers = [
            SequenceBlock(
                layer=self.layer,
                d_model=self.d_model,
                dropout=self.dropout,
                training=self.training,
                l_max=self.l_max,
                decode=self.decode,
            )
            for _ in range(self.n_layers)
        ]

    def __call__(self, x):
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x)
        if self.classification:
            x = np.mean(x, axis=0)
        x = self.decoder(x)
        return nn.log_softmax(x, axis=-1)


BatchStackedModel = nn.vmap(
    StackedModel,
    in_axes=0,
    out_axes=0,
    variable_axes={"params": None, "dropout": None, "cache": 0, "prime": None},
    split_rngs={"params": False, "dropout": True},
)


def K_gen_simple(Ab, Bb, Cb, L):
    K = K_conv(Ab, Bb, Cb, L)

    def gen(z):
        return np.sum(K * (z ** np.arange(L)))

    return gen


def conv_from_gen(gen, L):
    # Evaluate at roots of unity
    # Generating function is (-)z-transform, so we evaluate at (-)root
    Omega_L = np.exp((-2j * np.pi) * (np.arange(L) / L))
    atRoots = jax.vmap(gen)(Omega_L)
    # Inverse FFT
    out = np.fft.ifft(atRoots, L).reshape(L)
    return out.real


def K_gen_inverse(Ab, Bb, Cb, L):
    I = np.eye(Ab.shape[0])
    Ab_L = matrix_power(Ab, L)
    Ct = Cb @ (I - Ab_L)
    return lambda z: (Ct.conj() @ inv(I - Ab * z) @ Bb).reshape()



def test_gen_inverse(L=16, N=4):
    ssm = random_SSM(rng, N)
    ssm = discretize(*ssm, 1.0 / L)
    b = K_conv(*ssm, L=L)

    a = conv_from_gen(K_gen_inverse(*ssm, L=L), L)
    assert np.allclose(a, b)



@partial(np.vectorize, signature="(c),(),(c)->()")
def cauchy_dot(v, omega, lambd):
    return (v / (omega - lambd)).sum()



def K_gen_DPLR(Lambda, p, q, B, Ct, step, unmat=False):
    aterm = (Ct.conj().ravel(), q.conj().ravel())
    bterm = (B.ravel(), p.ravel())

    def gen(o):
        g = (2.0 / step) * ((1.0 - o) / (1.0 + o))
        c = 2.0 / (1.0 + o)

        def k(a):
            # Checkpoint this calculation for memory efficiency.
            if unmat:
                return jax.remat(cauchy_dot)(a, g, Lambda)
            else:
                return cauchy_dot(a, g, Lambda)

        k00 = k(aterm[0] * bterm[0])
        k01 = k(aterm[0] * bterm[1])
        k10 = k(aterm[1] * bterm[0])
        k11 = k(aterm[1] * bterm[1])
        return c * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)

    return gen


def random_DPLR(rng, N):
    l_r, p_r, q_r, b_r, c_r = jax.random.split(rng, 5)
    Lambda = jax.random.uniform(l_r, (N,))
    p = jax.random.uniform(p_r, (N,))
    q = jax.random.uniform(q_r, (N,))
    B = jax.random.uniform(b_r, (N, 1))
    C = jax.random.uniform(c_r, (1, N))
    return Lambda, p, q, B, C



def test_gen_dplr(L=16, N=4):
    I = np.eye(4)

    # Create a DPLR A matrix and discretize
    _, _, _, B, _ = random_DPLR(rng, N)
    _, Lambda, p, q, V = make_NPLR_HiPPO(N)
    Vc = V.conj().T
    p = Vc @ p
    q = Vc @ q.conj()
    A = np.diag(Lambda) - p[:, np.newaxis] @ q[:, np.newaxis].conj().T
    B = Vc @ np.sqrt(1.0 + 2 * np.arange(N)).reshape(N, 1)
    _, _, C = random_SSM(rng, N)

    Ab, Bb, Cb = discretize(A, B, C, 1.0 / L)
    a = K_conv(Ab, Bb, Cb.conj(), L=L)

    # Compare to the DPLR generating function approach.
    Ct = (I - matrix_power(Ab, L)).conj().T @ Cb.ravel()
    b = conv_from_gen(K_gen_DPLR(Lambda, p, q, B, Ct, step=1.0 / L), L)
    assert np.allclose(a.real, b.real)



def discrete_DPLR(Lambda, p, q, B, Ct, step, L):
    N = Lambda.shape[0]
    A = np.diag(Lambda) - p[:, np.newaxis] @ q[:, np.newaxis].conj().T
    I = np.eye(N)

    # Forward Euler
    A0 = (2.0 / step) * I + A

    # Backward Euler
    D = np.diag(1.0 / ((2.0 / step) - Lambda))
    qc = q.conj().T.reshape(1, -1)
    p2 = p.reshape(-1, 1)
    A1 = D - (D @ p2 * (1.0 / (1 + (qc @ D @ p2))) * qc @ D)

    # A bar and B bar
    Ab = A1 @ A0
    Bb = 2 * A1 @ B

    # Recover Cbar from Ct
    Cb = Ct @ inv(I - matrix_power(Ab, L)).conj()
    return Ab, Bb, Cb.conj()



def make_NPLR_HiPPO(N):
    # Make -HiPPO
    nhippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    p = 0.5 * np.sqrt(2 * np.arange(1, N + 1) + 1.0)
    q = 2 * p
    S = nhippo + p[:, np.newaxis] * q[np.newaxis, :]

    # Diagonalize to S to V \Lambda V^*
    Lambda, V = jax.jit(eig, backend="cpu")(S)
    # Lambda, V = eig(jax.device_put(S, device=jax.devices("cpu")[0]))
    return nhippo, Lambda, p, q, V



def test_nplr(N=8):
    A2, Lambda, p, q, V = make_NPLR_HiPPO(N)
    p, q = p[:, np.newaxis], q[:, np.newaxis]
    Lambda = np.diag(Lambda)
    Vc = V.conj().T
    A3 = V @ (Lambda - (Vc @ p) @ (Vc @ q.conj()).conj().T) @ Vc
    A4 = V @ Lambda @ Vc - (p @ q.T)
    assert np.allclose(A2, A3, atol=1e-4, rtol=1e-4)
    assert np.allclose(A2, A4, atol=1e-4, rtol=1e-4)



def test_conversion(N=8, L=16):
    step = 1.0 / L
    # Compute a HiPPO NPLR matrix.
    _, Lambda, p, q, V = make_NPLR_HiPPO(N)
    Vc = V.conj().T
    p = Vc @ p
    q = Vc @ q.conj()
    B = lecun_normal(dtype=np.complex64)(rng, (N, 1))
    B = Vc @ B
    # Random complex Ct
    Ct = lecun_normal(dtype=np.complex64)(rng, (1, N))

    # CNN form.
    K_gen = K_gen_DPLR(Lambda, p, q, B, Ct, step)
    K = conv_from_gen(K_gen, L)

    # RNN form.
    Ab, Bb, Cb = discrete_DPLR(Lambda, p, q, B, Ct, step, L)
    K2 = K_conv(Ab, Bb, Cb, L=L)
    assert np.allclose(K.real, K2.real, atol=1e-5, rtol=1e-5)

    # Apply CNN
    u = np.arange(L) * 1.0
    y1 = non_circular_convolution(u, K.real)

    # Apply RNN
    _, y2 = scan_SSM(
        Ab, Bb, Cb, u[:, np.newaxis], np.zeros((N,)).astype(np.complex64)
    )
    assert np.allclose(y1, y2.reshape(-1).real, atol=1e-4, rtol=1e-4)



class S4Layer(nn.Module):
    A: np.DeviceArray
    Vc: np.DeviceArray
    p: np.DeviceArray
    q: np.DeviceArray
    Lambda: np.DeviceArray
    N: int
    l_max: int
    decode: bool = False

    def setup(self):
        # Learned Parameters (Ct is complex!)
        self.Ct = self.param("Ct", lecun_normal(), (1, self.N, 2))
        self.Ct = self.Ct[..., 0] + 1j * self.Ct[..., 1]
        self.B = self.Vc @ self.param("B", lecun_normal(), (self.N, 1))
        self.D = self.param("D", uniform(), (1,))
        self.step = np.exp(self.param("log_step", log_step_initializer(), (1,)))

        if not self.decode:
            # CNN mode, compute kernel.
            K_gen = K_gen_DPLR(
                self.Lambda,
                self.p,
                self.q,
                self.B,
                self.Ct,
                self.step[0],
                unmat=self.l_max > 1000,
            )
            self.K = conv_from_gen(K_gen, self.l_max)

        else:
            # RNN mode, discretize

            # Flax trick to cache discrete form during decoding.
            def init_discrete():
                return discrete_DPLR(
                    self.Lambda,
                    self.p,
                    self.q,
                    self.B,
                    self.Ct,
                    self.step[0],
                    self.l_max,
                )

            ssm_var = self.variable("prime", "ssm", init_discrete)
            if self.is_mutable_collection("prime"):
                ssm_var.value = init_discrete()
            self.ssm = ssm_var.value

            # RNN Cache
            self.x_k_1 = self.variable(
                "cache", "cache_x_k", np.zeros, (self.N,), np.complex64
            )

    def __call__(self, u):
        # This is identical to SSM Layer
        if not self.decode:
            # CNN Mode
            return non_circular_convolution(u, self.K) + self.D * u
        else:
            # RNN Mode
            x_k, y_s = scan_SSM(*self.ssm, u[:, np.newaxis], self.x_k_1.value)
            if self.is_mutable_collection("cache"):
                self.x_k_1.value = x_k
            return y_s.reshape(-1).real + self.D * u


S4Layer = cloneLayer(S4Layer)


def S4LayerInit(N):
    _, Lambda, p, q, V = make_NPLR_HiPPO(N)
    Vc = V.conj().T
    p = Vc @ p
    q = Vc @ q.conj()
    A = np.diag(Lambda) - p[:, np.newaxis] @ q[:, np.newaxis].conj().T
    return partial(S4Layer, N=N, A=A, Lambda=Lambda, p=p, q=q, Vc=Vc)



def sample(model, params, prime, cache, x, start, end, rng):
    def loop(i, cur):
        x, rng, cache = cur
        r, rng = jax.random.split(rng)
        out, vars = model.apply(
            {"params": params, "prime": prime, "cache": cache},
            x[:, np.arange(1, 2) * i],
            mutable=["cache"],
        )

        def update(x, out):
            p = jax.random.categorical(r, out[0])
            x = x.at[i + 1, 0].set(p)
            return x

        x = jax.vmap(update)(x, out)
        return x, rng, vars["cache"].unfreeze()

    return jax.lax.fori_loop(start, end, jax.jit(loop), (x, rng, cache))[0]



def init_from_checkpoint(model, checkpoint, init_x):
    from flax.training import checkpoints

    print("[*] Loading")
    state = checkpoints.restore_checkpoint(checkpoint, None)
    assert "params" in state
    print("[*] Initializing")
    variables = model.init(rng, init_x)
    vars = {
        "params": state["params"],
        "cache": variables["cache"].unfreeze(),
        "prime": variables["prime"].unfreeze(),
    }
    print("[*] Priming")
    _, prime_vars = model.apply(vars, init_x, mutable=["prime"])
    return vars["params"], prime_vars["prime"], vars["cache"]



def sample_checkpoint(path, model, length):
    start = np.zeros((1, length, 1))
    print("[*] Initializing from checkpoint %s" % path)
    params, prime, cache = init_from_checkpoint(model, path, start[:, :-1])
    print("[*] Sampling output")
    return sample(model, params, prime, cache, start, 0, length - 1, rng)


def sample_mnist_prefix(path, model, length):
    import matplotlib.pyplot as plt
    import numpy as onp
    from .data import Datasets

    BATCH = 32
    START = 300
    start = np.zeros((BATCH, length, 1))
    params, prime, init_cache = init_from_checkpoint(model, path, start[:, :-1])

    _, testloader, _, _, _ = Datasets["mnist"](bsz=BATCH)
    it = iter(testloader)
    for j, im in enumerate(it):
        image = im[0].numpy()

        cur = onp.array(image)
        cur[:, START + 1 :, 0] = 0
        cur = np.array(cur)

        # Cache the first `start` inputs.
        out, vars = model.apply(
            {"params": params, "prime": prime, "cache": init_cache},
            cur[:, np.arange(0, START)],
            mutable=["cache"],
        )
        cache = vars["cache"].unfreeze()
        out = sample(model, params, prime, cache, cur, START, length - 1, rng)
        print(j)

        # Visualization
        out = out.reshape(BATCH, 28, 28)
        final = onp.zeros((BATCH, 28, 28, 3))
        final2 = onp.zeros((BATCH, 28, 28, 3))
        final[:, :, :, 0] = out
        f = final.reshape(BATCH, 28 * 28, 3)
        i = image.reshape(BATCH, 28 * 28)
        f[:, :START, 1] = i[:, :START]
        f[:, :START, 2] = i[:, :START]
        f = final2.reshape(BATCH, 28 * 28, 3)
        f[:, :, 1] = i
        f[:, :START, 0] = i[:, :START]
        f[:, :START, 2] = i[:, :START]
        for k in range(BATCH):
            fig, (ax1, ax2) = plt.subplots(ncols=2)
            ax1.set_title("Sampled")
            ax1.imshow(final[k] / 256.0)
            ax2.set_title("True")
            ax1.axis("off")
            ax2.axis("off")
            ax2.imshow(final2[k] / 256.0)
            fig.savefig("im%d.%d.png" % (j, k))
            print(j)