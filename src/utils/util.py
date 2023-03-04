from typing import Callable, Mapping
import functools
import hydra

# This is a modified version of hydra.utils.instantiate
# using https://github.com/aranku/hydra_example/blob/main/hydraexample/utils.py as a reference
def instantiate(config):
    # Case 1: no config
    if config is None:
        return None
    # Case 2b: grab the desired callable from name
    else:
        _target_ = config.pop("_target_")

    # Retrieve the right constructor automatically based on type
    if isinstance(_target_, str):
        fn = hydra.utils.get_method(path=_target_)
    elif isinstance(_target_, Callable):
        fn = _target_
    else:
        raise NotImplementedError("instantiate target must be string or callable")

    config = {
        k: instantiate(v) if isinstance(v, Mapping) and "_target_" in v else v
        for k, v in config.items()
    }
    obj = functools.partial(fn, **config)

    # Restore _name_
    if _target_ is not None:
        config["_target_"] = _target_

    return obj()


def eval_legendre(n, x):
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return np.vstack((np.ones_like(x), x))
    else:
        L = np.zeros((n + 1, len(x)))
        L[0] = np.ones_like(x)
        L[1] = x
        for i in range(2, n + 1):
            L[i] = ((2 * i - 1) * x * L[i - 1] - (i - 1) * L[i - 2]) / i
        return L


def eval_legendre(n, x):
    if n == 0:
        return jnp.ones_like(x)
    elif n == 1:
        return jnp.vstack((jnp.ones_like(x), x))
    else:
        L = jnp.zeros((n + 1, len(x)))
        L = L.at[0].set(jnp.ones_like(x))
        L = L.at[1].set(x)
        for i in range(2, n + 1):
            L = L.at[i].set(((2 * i - 1) * x * L[i - 1] - (i - 1) * L[i - 2]) / i)
        return L
