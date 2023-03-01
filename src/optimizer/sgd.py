import optax

from src.optimizer.optim import Optimizer


class SGD(Optimizer):
    def __init__(self, lr: float, momentum: float = 0.0, nesterov: bool = False):
        super().__init__(lr=lr)
        self.momentum = momentum
        self.nesterov = nesterov
        self.tx = optax.sgd(
            learning_rate=self.lr, momentum=self.momentum, nesterov=self.nesterov
        )
