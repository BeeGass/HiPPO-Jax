import numpy as np


class DataProcessor:
    """
    Class to manage data processing.
    """

    def __init__(self, seed: int, data: np.ndarray, target: str):
        self.seed = seed
        self.data = data
        self.target = target
