from flax.training import train_state


class Trainer:
    """
    Class to manage model training and feature extraction.
    """

    def init_train_state(self, *args, **kwargs) -> train_state.TrainState:
        raise NotImplementedError()

    def update_model(self, *args, **kwargs):
        raise NotImplementedError()

    def train(self, *args, **kwargs):
        raise NotImplementedError()

    def epoch(self, *args, **kwargs):
        raise NotImplementedError()

    def step(self, *args, **kwargs):
        raise NotImplementedError()

    def eval(self, *args, **kwargs):
        raise NotImplementedError()

    def log_metrics(self, *args, **kwargs):
        raise NotImplementedError()
