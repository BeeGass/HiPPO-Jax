from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap

# pulled from https://github.com/google/jax/issues/3171
@partial(jit, static_argnums=(1,))
def moving_window(a, size: int):
    starts = jnp.arange(len(a) - size + 1)
    return vmap(lambda start: jax.lax.dynamic_slice(a, (start,), (size,)))(starts)


# pulled from https://github.com/google/jax/issues/3171
def rolling_window(a: jnp.ndarray, window: int):
    idx = jnp.arange(len(a) - window + 1)[:, None] + jnp.arange(window)[None, :]
    return a[idx]
