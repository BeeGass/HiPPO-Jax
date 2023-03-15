from typing import Any, Callable, Mapping, Optional
import functools
import hydra
import numpy as np
import jax
from jax import numpy as jnp
from omegaconf import DictConfig

# This is a modified version of hydra.utils.instantiate
# using https://github.com/aranku/hydra_example/blob/main/hydraexample/utils.py as a reference
def instantiate(config: Optional[DictConfig]) -> Any:
    # Case 1: no config
    if config is None:
        return None

    # Case 2: grab the desired callable from name
    _target_ = config.pop("_target_")

    # Case 2a: retrieve the right constructor automatically based on type
    if isinstance(_target_, str):
        fn = hydra.utils.get_method(path=_target_)
    elif isinstance(_target_, Callable):
        fn = _target_
    else:
        raise NotImplementedError("instantiate target must be string or callable")

    # Instantiate the object
    config = {
        k: instantiate(v) if isinstance(v, Mapping) and "_target_" in v else v
        for k, v in config.items()
    }
    obj = functools.partial(fn, **config)

    # Restore _target_
    if _target_ is not None:
        config["_target_"] = _target_

    return obj()


def legendre_recurrence(n, x, max_n):
    """
    Computes the Legendre polynomial of degree n at point x using the recurrence relation.

    Args:
    n: int, the degree of the Legendre polynomial.
    x: float, the point at which to evaluate the polynomial.
    max_n: int, the maximum degree of n in the batch.

    Returns:
    The value of the Legendre polynomial of degree n at point x.
    """
    # Initialize the array to store the Legendre polynomials for all degrees from 0 to max_n
    p = jnp.zeros((max_n + 1,) + x.shape)
    p = p.at[0].set(1.0)  # Set the 0th degree Legendre polynomial
    p = p.at[1].set(x)  # Set the 1st degree Legendre polynomial

    # Compute the Legendre polynomials for degrees 2 to max_n using the recurrence relation
    def body_fun(i, p):
        p_i = ((2 * i - 1) * x * p[i - 1] - (i - 1) * p[i - 2]) / i
        return p.at[i].set(p_i)

    p = jax.lax.fori_loop(2, max_n + 1, body_fun, p)

    return p[n]


def genlaguerre_recurrence(n, alpha, x, max_n):
    """
    Computes the generalized Laguerre polynomial of degree n with parameter alpha at point x using the recurrence relation.

    Args:
    n: int, the degree of the generalized Laguerre polynomial.
    alpha: float, the parameter of the generalized Laguerre polynomial.
    x: float, the point at which to evaluate the polynomial.
    max_n: int, the maximum degree of n in the batch.

    Returns:
    The value of the generalized Laguerre polynomial of degree n with parameter alpha at point x.
    """
    # Initialize the array to store the generalized Laguerre polynomials for all degrees from 0 to max_n
    p = jnp.zeros((max_n + 1,) + x.shape)
    p = p.at[0].set(1.0)  # Set the 0th degree generalized Laguerre polynomial

    # Compute the generalized Laguerre polynomials for degrees 1 to max_n using the recurrence relation
    def body_fun(i, p):
        p_i = ((2 * i + alpha - 1 - x) * p[i - 1] - (i + alpha - 1) * p[i - 2]) / i
        return p.at[i].set(p_i)

    p = jax.lax.fori_loop(1, max_n + 1, body_fun, p)

    return p[n]


def eval_legendre(n, x, out=None):
    """
    Evaluates the Legendre polynomials of degrees specified in the input array n at the points specified in the input array x.

    Args:
    n: array-like, the degrees of the Legendre polynomials.
    x: array-like, the points at which to evaluate the polynomials.
    out: optional, an output array to store the results.

    Returns:
    An array containing the Legendre polynomial values of the specified degrees at the specified points.
    """
    n = jnp.asarray(n)
    x = jnp.asarray(x)
    max_n = n.max()

    if n.ndim == 1 and x.ndim == 1:
        p = jax.vmap(
            lambda ni: jax.vmap(lambda xi: legendre_recurrence(ni, xi, max_n))(x)
        )(n)
        p = jnp.diagonal(
            p
        )  # Get the diagonal elements to match the scipy.special.eval_legendre output
    else:
        p = jax.vmap(
            lambda ni: jax.vmap(lambda xi: legendre_recurrence(ni, xi, max_n))(x)
        )(n)

    if out is not None:
        out = jnp.asarray(out)
        out = jnp.copy(p, out=out)
        return out
    else:
        return jnp.squeeze(p)

def eval_genlaguerre(n, alpha, x, out=None):
    """
    Evaluates the generalized Laguerre polynomials of degrees specified in the input array n with parameter alpha at the points specified in the input array x.

    Args:
    n: array-like, the degrees of the generalized Laguerre polynomials.
    alpha: float, the parameter of the generalized Laguerre polynomials.
    x: array-like, the points at which to evaluate the polynomials.
    out: optional, an output array to store the results.

    Returns:
    An array containing the generalized Laguerre polynomial values of the specified degrees with parameter alpha at the specified points.
    """
    n = jnp.asarray(n)
    x = jnp.asarray(x)
    max_n = n.max()

    if n.ndim == 1 and x.ndim == 1:
        p = jax.vmap(
            lambda ni: jax.vmap(
                lambda xi: genlaguerre_recurrence(ni, alpha, xi, max_n)
            )(x)
        )(n)
        p = jnp.diagonal(
            p
        )  # Get the diagonal elements to match the scipy.signal.eval_genlaguerre output
    else:
        p = jax.vmap(
            lambda ni: jax.vmap(
                lambda xi: genlaguerre_recurrence(ni, alpha, xi, max_n)
            )(x)
        )(n)

    if out is not None:
        out = jnp.asarray(out)
        out = jnp.copy(p, out=out)
        return out
    else:
        return jnp.squeeze(p)
