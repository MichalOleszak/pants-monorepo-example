import unittest

import torch
from torchvision import transforms

from mnist.src.data import MNISTDataset


class TestMNISTDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = MNISTDataset(
            root="data/", train=True, transform=transforms.ToTensor()
        )

    def test_dataset_length(self):
        # Act
        length = len(self.dataset)

        # Assert
        self.assertEqual(length, 60000)

    def test_getitem(self):
        # Arrange
        index = 0

        # Act
        image, label = self.dataset[index]

        # Assert
        self.assertIsInstance(image, torch.Tensor)
        self.assertIsInstance(label, int)
        self.assertEqual(len(image.shape), 3)
        self.assertEqual(image.shape[0], 1)
        self.assertGreaterEqual(label, 0)
        self.assertLessEqual(label, 9)


if __name__ == "__main__":
    unittest.main()
