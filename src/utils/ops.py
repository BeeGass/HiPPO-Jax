import jax.numpy as jnp
import jax
from jax import jit


def genlaguerre(n, alpha, x):

    assert alpha >= 0, "alpha must be non-negative"

    def step(p, i):
        Ln, Lnm1 = p

        # Recursive definition: L_n^alpha(x) = (2*n + alpha - x) / (n) * L_{n-1}^alpha(x) - (n + alpha - 1) / n * L_{n-2}^alpha(x)
        Lnp1 = (2 * i + alpha - x) / i * Ln - (i + alpha - 1) / i * Lnm1

        return Lnp1, Ln

    # Initial values: L_{-1}^alpha(x) = 0, L_0^alpha(x) = 1
    L_init = (jnp.zeros_like(x, dtype=jnp.float32), jnp.ones_like(x, dtype=jnp.float32))

    # Use lax.scan to compute L_1^alpha(x), L_2^alpha(x), ..., L_n^alpha(x)
    _, Ln = jax.lax.scan(step, L_init, jnp.arange(1, n + 1))

    return Ln


# def legendre_polynomial(n, x):
#     if n == 0:
#         return jnp.ones_like(x)
#     if n == 1:
#         return x
#     return (
#         (2 * n - 1) * x * legendre_polynomial(n - 1, x)
#         - (n - 1) * legendre_polynomial(n - 2, x)
#     ) / n


def legendre_polynomial(n, x):
    p_prev = jnp.ones_like(x, dtype=jnp.float32)
    p_curr = x.astype(jnp.float32)
    for i in range(1, n):
        p_next = ((2 * i + 1) * x * p_curr - i * p_prev) / (i + 1)
        p_prev = p_curr
        p_curr = p_next
    return p_curr
