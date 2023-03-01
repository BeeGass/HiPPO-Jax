import numpy as np


class HiPPOPipeline(DataProcessor):
    """
    Class to manage data processing.
    """

    def __init__(self, seed: int, data: np.ndarray, target: str):
        super().__init__(seed, data, target)
        self.pipeline = []
