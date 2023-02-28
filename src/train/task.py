from dataclasses import dataclass

from src.datasets.dataset import Dataset
from src.loss.loss import Loss
from src.models.model import Model
from src.optimizer.optim import Optimizer


@dataclass
class Task:
    """
    Class to manage model training and feature extraction.
    """

    seed: int
    optimizer: Optimizer
    loss: Loss
    pipeline: DataProcessor
    dataset: Dataset
    model: Model
