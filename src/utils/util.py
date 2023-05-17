from typing import Any, Callable, Mapping, Optional
import functools
import hydra
import numpy as np
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Int
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


@jax.jit
def normalize(x: Float[Array, "n"]) -> Float[Array, "n"]:
    """
    Normalize a JAX array to the range [-1, 1] using min-max normalization.

    The formula used for normalization is:
    normalized_x = (2 * (x - min_value) / (max_value - min_value)) - 1

    This function is just-in-time (JIT) compiled for improved performance.

    Args:
        x : jnp.ndarray
            The JAX array to be normalized.

    Returns:
        jnp.ndarray
            The normalized JAX array in the range [-1, 1].

    Raises:
        ZeroDivisionError
            If max_value equals min_value in the input data, a ZeroDivisionError is raised due to the division in the normalization formula.

    Note:
        Make sure the x doesn't contain extreme outliers as it can distort the normalized values.

    Example:
        >>> x = jnp.array([1, 2, 3, 4, 5])
        >>> print(normalize_x(x))
        DeviceArray([-1. , -0.5,  0. ,  0.5,  1. ], dtype=float32)
    """
    min_value = jnp.min(x)
    max_value = jnp.max(x)
    normalized_x = (2 * (x - min_value) / (max_value - min_value)) - 1
    return normalized_x


def legendre_recurrence(
    n: Int[Array, "n"], x: Float[Array, "m"], n_max: Int[Array, ""]
) -> Float[Array, "n m"]:
    """
    Compute the Legendre polynomials up to degree n_max at a given point or array of points x.

    The function employs the recurrence relation for Legendre polynomials. The Legendre polynomials
    are orthogonal on the interval [-1,1] and are used in a wide array of scientific and mathematical applications.
    This function returns a series of Legendre polynomials evaluated at the point(s) x, up to the degree n_max.

    Args:
        n_max (int): The highest degree of Legendre polynomial to compute. Must be a non-negative integer.
        x (jnp.ndarray): The point(s) at which the Legendre polynomials are to be evaluated. Can be a single
                        point (float) or an array of points.

    Returns:
        jnp.ndarray: A sequence of Legendre polynomial values of shape (n_max+1,) + x.shape, evaluated at point(s) x.
                    The i-th entry of the output array corresponds to the Legendre polynomial of degree i.

    Notes:
        The first two Legendre polynomials are initialized as P_0(x) = 1 and P_1(x) = x. The subsequent polynomials
        are computed using the recurrence relation:
        P_{n+1}(x) = ((2n + 1) * x * P_n(x) - n * P_{n-1}(x)) / (n + 1).
    """

    p_init = jnp.zeros((2,) + x.shape)
    p_init = p_init.at[0].set(1.0)  # Set the 0th degree Legendre polynomial
    p_init = p_init.at[1].set(x)  # Set the 1st degree Legendre polynomial

    def body_fun(carry, _):
        i, (p_im1, p_i) = carry
        p_ip1 = ((2 * i + 1) * x * p_i - i * p_im1) / (i + 1)

        return ((i + 1).astype(int), (p_i, p_ip1)), p_ip1

    (_, (_, _)), p_n = jax.lax.scan(
        f=body_fun, init=(1, (p_init[0], p_init[1])), xs=(None), length=(n_max - 1)
    )
    p_n = jnp.concatenate((p_init, p_n), axis=0)

    return p_n[n]


def legendre_recurrence_old(
    n: Int[Array, "n"], x: Float[Array, "m"], max_n: Int[Array, ""]
) -> Float[Array, "n m"]:
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

    p = jax.lax.fori_loop(lower=2, upper=(max_n + 1), body_fun=body_fun, init_val=p)

    return p[n]


def genlaguerre_recurrence(
    n: Int[Array, "n"],
    alpha: Float[Array, ""],
    x: Float[Array, "m"],
    max_n: Int[Array, ""],
) -> Float[Array, "n m"]:
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


def eval_legendre(n: Int[Array, "n"], x: Float[Array, "m"]) -> Float[Array, "n m"]:
    """
    Evaluate Legendre polynomials of specified degrees at provided point(s).

    This function makes use of a vectorized version of the Legendre polynomial recurrence relation to
    compute the necessary polynomials up to the maximum degree found in 'n'. It then selects and returns
    the values of the polynomials at the degrees specified in 'n' and evaluated at the points in 'x'.

    Parameters:
        n (jnp.ndarray): An array of integer degrees for which the Legendre polynomials are to be evaluated.
                        Each element must be a non-negative integer and the array can be of any shape.
        x (jnp.ndarray): The point(s) at which the Legendre polynomials are to be evaluated. Can be a single
                        point (float) or an array of points. The shape must be broadcastable to the shape of 'n'.

    Returns:
        jnp.ndarray: An array of Legendre polynomial values. The output has the same shape as 'n' and 'x' after broadcasting.
                    The i-th entry corresponds to the Legendre polynomial of degree 'n[i]' evaluated at point 'x[i]'.

    Notes:
        This function makes use of the vectorized map (vmap) functionality in JAX to efficiently compute and select
        the necessary Legendre polynomial values.
    """
    n = jnp.asarray(n)
    x = jnp.asarray(x)
    n_max = n.max()

    if n.ndim == 1 and x.ndim == 1:
        p = jax.vmap(
            lambda ni: jax.vmap(lambda xi: legendre_recurrence(ni, xi, n_max))(x)
        )(n)
        p = jnp.diagonal(
            p
        )  # Get the diagonal elements to match the scipy.special.eval_legendre output
    else:
        p = jax.vmap(
            lambda ni: jax.vmap(lambda xi: legendre_recurrence(ni, xi, n_max))(x)
        )(n)

    return jnp.squeeze(p)


def eval_legendre_old(
    n: Int[Array, "n"], x: Float[Array, "m"], out: Float[Array, "n m"] = None
) -> Float[Array, "n m"]:
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
    n_max = n.max()

    if n.ndim == 1 and x.ndim == 1:
        p = jax.vmap(
            lambda ni: jax.vmap(lambda xi: legendre_recurrence_old(ni, xi, n_max))(x)
        )(n)
        p = jnp.diagonal(
            p
        )  # Get the diagonal elements to match the scipy.special.eval_legendre output
    else:
        p = jax.vmap(
            lambda ni: jax.vmap(lambda xi: legendre_recurrence_old(ni, xi, n_max))(x)
        )(n)

    if out is not None:
        out = jnp.asarray(out)
        out = jnp.copy(p, out=out)
        return out
    else:
        return jnp.squeeze(p)


def eval_genlaguerre(
    n: Int[Array, "n"],
    alpha: Float[Array, ""],
    x: Float[Array, "m"],
    out: Float[Array, "n m"] = None,
) -> Float[Array, "n m"]:
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
