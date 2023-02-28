from flax import linen as nn


class Model(nn.Module):
    """
    Base class for a model.
    Wraps flax.linen.Module.
    """

    def __call__(self, **args):
        raise NotImplementedError
