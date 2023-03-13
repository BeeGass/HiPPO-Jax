from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np

# pulled from https://github.com/google/jax/issues/3171
@partial(jit, static_argnums=(1,))
def moving_window(a, size: int):
    starts = jnp.arange(len(a) - size + 1)
    return vmap(lambda start: jax.lax.dynamic_slice(a, (start,), (size,)))(starts)


# pulled from https://github.com/google/jax/issues/3171
def rolling_window(a: jnp.ndarray, window: int):
    idx = jnp.arange(len(a) - window + 1)[:, None] + jnp.arange(window)[None, :]
    return a[idx]


def whitesignal(key, period, dt, freq, rms=0.5, batch_shape=()):
    """
    Produces output signal of length period / dt, band-limited to frequency freq
    Output shape (*batch_shape, period/dt)
    Adapted from the nengo library
    """

    if freq is not None and freq < 1.0 / period:
        raise ValueError(
            f"Make ``{freq=} >= 1. / {period=}`` to produce a non-zero signal",
        )

    nyquist_cutoff = 0.5 / dt
    if freq > nyquist_cutoff:
        raise ValueError(
            f"{freq} must not exceed the Nyquist frequency for the given dt ({nyquist_cutoff:0.3f})"
        )

    n_coefficients = int(jnp.ceil(period / dt / 2.0))
    shape = batch_shape + (n_coefficients + 1,)
    sigma = rms * jnp.sqrt(0.5)
    coefficients = 1j * jax.random.normal(key, shape) * sigma
    coefficients = jnp.array(coefficients)
    coefficients = coefficients.at[..., -1].set(0.0)
    coefficients += jax.random.normal(key, shape) * sigma
    coefficients = jnp.array(coefficients)
    coefficients = coefficients.at[..., 0].set(0.0)

    set_to_zero = jnp.fft.rfftfreq(2 * n_coefficients, d=dt) > freq
    coefficients *= 1 - set_to_zero
    power_correction = jnp.sqrt(
        1.0 - jnp.sum(set_to_zero, dtype=jnp.float32) / n_coefficients
    )
    if power_correction > 0.0:
        coefficients /= power_correction
    coefficients *= jnp.sqrt(2 * n_coefficients)
    signal = jnp.fft.irfft(coefficients, axis=-1)
    signal = signal - signal[..., :1]  # Start from 0
    return signal
