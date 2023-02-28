import jax
import optax
from jax import numpy as jnp
from jaxtyping import Array, Float

from src.loss.loss import Loss


class SoftmaxCrossEntropy(Loss):
    def __init__(self):
        self.loss_fn = jax.vmap(jnp.mean(optax.softmax_cross_entropy), in_axes=(0, 0))

    def apply(
        self, y_pred: Float[Array, "batch ..."], y_true: Float[Array, "batch ..."]
    ):
        """_summary_

        Args:
            y_pred (_type_): _description_
            y_true (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.loss_fn(y_pred, y_true)
