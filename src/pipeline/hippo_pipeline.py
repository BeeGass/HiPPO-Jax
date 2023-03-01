import numpy as np

from src.pipeline.data_processor import DataProcessor


class HiPPOPipeline(DataProcessor):
    """
    Class to manage data processing.
    """

    def __init__(self):
        self.pipeline = []
