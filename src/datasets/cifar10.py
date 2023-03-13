from src.datasets.dataset import Dataset
import torch
from torchvision import datasets, transforms


class CIFAR10Dataset(Dataset):
    def __init__(self, name: str, path: str, batch_size: int = 64):
        super().__init__(name, path)
        self.batch_size = batch_size
        train_dataset = datasets.CIFAR10(
            path,
            download=True,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),  # first, convert image to PyTorch tensor
                ]
            ),
        )
        test_dataset = datasets.CIFAR10(
            path,
            download=False,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),  # first, convert image to PyTorch tensor
                ]
            ),
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=True
        )