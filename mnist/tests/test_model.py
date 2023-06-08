import unittest
import torch

from mnist.src.model import MNISTModel


class TestMNISTModel(unittest.TestCase):
    def test_model_output_shape(self):
        # Arrange
        model = MNISTModel()
        input_tensor = torch.randn(1, 1, 28, 28)

        # Act
        output = model(input_tensor)

        # Assert
        expected_shape = torch.Size([1, 10])
        self.assertEqual(output.shape, expected_shape)

    def test_model_forward_pass(self):
        # Arrange
        model = MNISTModel()
        input_tensor = torch.randn(1, 1, 28, 28)

        # Act
        output = model(input_tensor)

        # Assert
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(len(output.shape), 2)
        self.assertEqual(output.shape[0], 1)
        self.assertEqual(output.shape[1], 10)


if __name__ == "__main__":
    unittest.main()
