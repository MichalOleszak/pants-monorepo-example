from typing import Tuple

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset


class MNISTDataset(Dataset):
    def __init__(
        self, root: str, train: bool = True, transform: transforms.Compose = None
    ) -> None:
        super().__init__()
        self.dataset = datasets.MNIST(
            root=root, train=train, transform=transform, download=True
        )

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, label = self.dataset[index]
        return image, label

    def __len__(self) -> int:
        return len(self.dataset)
