import jax
import optax
from jax import numpy as jnp
from jaxtyping import Array, Float

from src.loss.loss import Loss


class Huber(Loss):
    def __init__(self, delta: float = 1.0):
        self.delta = delta
        self.loss_fn = jax.vmap(optax.huber_loss, in_axes=(0, 0, None))

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
        return self.loss_fn(y_pred, y_true, self.delta).mean()
