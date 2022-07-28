from collections import defaultdict
import inspect
import pprint

GLOBAL_REGISTER = defaultdict(dict)


def register(name, type_, n_lock=False):
    """
    Register an object with a name.
    """

    def _register(obj):

        if name in GLOBAL_REGISTER[type_] and not n_lock:
            raise ValueError(
                f"Global registry cannot accept {name}: {type_}. The object has already been registered"
            )

        GLOBAL_REGISTER[name][obj.__name__] = obj
        return obj

    GLOBAL_REGISTER[name] = obj
    return obj
