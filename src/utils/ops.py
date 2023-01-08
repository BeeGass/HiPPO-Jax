import jax.numpy as jnp
import jax


def genlaguerre(n, alpha, x):
    assert alpha >= 0, "alpha must be non-negative"

    def step(i, p):
        Ln, Lnm1 = p

        # Recursive definition: L_n^alpha(x) = (2*n + alpha - x) / (n) * L_{n-1}^alpha(x) - (n + alpha - 1) / n * L_{n-2}^alpha(x)
        Lnp1 = (2 * i + alpha - x) / i * Ln - (i + alpha - 1) / i * Lnm1

        return Lnp1, Ln

    # Initial values: L_{-1}^alpha(x) = 0, L_0^alpha(x) = 1
    L_init = (jnp.zeros_like(x), jnp.ones_like(x))

    # Use lax.scan to compute L_1^alpha(x), L_2^alpha(x), ..., L_n^alpha(x)
    _, Ln = jax.lax.scan(step, jnp.arange(1, n + 1), L_init)

    return Ln
