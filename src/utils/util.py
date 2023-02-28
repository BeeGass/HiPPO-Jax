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
