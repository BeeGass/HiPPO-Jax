import optax

from src.optimizer.optim import Optimizer


class Lamb(Optimizer):
    def __init__(self, lr: float, beta: float, weight_decay: float):
        super().__init__(lr=lr)
        self.beta = beta
        self.weight_decay = weight_decay
        self.tx = optax.lamb(self.lr, self.beta, self.weight_decay)
