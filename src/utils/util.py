from typing import Any, Callable, Mapping, Optional
import functools
import hydra
import numpy as np
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


def eval_legendre(n: int, x: np.ndarray) -> np.ndarray:
    """
    Evaluates the first n Legendre polynomials at the points in x.

    Parameters
    ----------
    n : int
        The number of Legendre polynomials to evaluate. Must be a positive
        integer.
    x : np.ndarray
        The points at which to evaluate the Legendre polynomials.

    Returns
    -------
    L : np.ndarray
        An array of shape (n + 1, len(x)) containing the first n + 1 Legendre
        polynomials evaluated at x.

    Raises
    ------
    ValueError
        If n is not a positive integer.
    TypeError
        If n is not an integer.
    """
    # Check that n is a positive integer
    if n < 0:
        raise ValueError("n must be a positive integer")
    elif not isinstance(n, int):
        raise TypeError("n must be a positive integer")
    # If n is 0, return a 1D array of 1s
    elif n == 0:
        return np.ones_like(x)
    # If n is 1, return a 2D array with the first row containing 1s and the
    # second row containing the values in x
    elif n == 1:
        return np.vstack((np.ones_like(x), x))
    # Otherwise, use the recurrence relation to evaluate the Legendre
    # polynomials
    else:
        L = np.zeros((n + 1, len(x)))
        L[0] = np.ones_like(x)
        L[1] = x
        for i in range(2, n + 1):
            L[i] = ((2 * i - 1) * x * L[i - 1] - (i - 1) * L[i - 2]) / i
        return L


def eval_legendre(n: int, x: jnp.ndarray) -> jnp.ndarray:
    """
    This function evaluates the first n Legendre polynomials at the points x.

    Parameters
    ----------
    n : int
        Number of Legendre polynomials to evaluate
    x : jnp.ndarray
        Points at which to evaluate the Legendre polynomials

    Returns
    -------
    L : jnp.ndarray
        Array containing the first n Legendre polynomials evaluated at x
    """
    # Make sure n is a non-negative integer
    if not isinstance(n, int) or n < 0:
        raise TypeError("n must be a non-negative integer")
    # Make sure x is a real vector
    if x.ndim != 1:
        raise TypeError("x must be a real vector")
    # If n is 0, return a vector of ones
    if n == 0:
        return jnp.ones_like(x)
    # If n is 1, return a 2-by-n array consisting of ones and x
    elif n == 1:
        return jnp.vstack((jnp.ones_like(x), x))
    # If n is greater than 1, use the recurrence relation to construct the Legendre polynomials
    else:
        # Initialize an (n + 1) by n array of zeros
        L = jnp.zeros((n + 1, len(x)))
        # Set the first row to be ones
        L = L.at[0].set(jnp.ones_like(x))
        # Set the second row to be x
        L = L.at[1].set(x)
        # Use a for loop to compute the remaining rows of L
        for i in range(2, n + 1):
            L = L.at[i].set(((2 * i - 1) * x * L[i - 1] - (i - 1) * L[i - 2]) / i)
        return L
