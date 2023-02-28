from src.optimizer.optim import Optimizer
from src.datasets.dataset import Dataset
from src.models.model import Model
from dataclasses import dataclass


@dataclass
class Task:
    """
    Class to manage model training and feature extraction.
    """

    seed: int
    optimizer: Optimizer
    dataset: Dataset
    model: Model
