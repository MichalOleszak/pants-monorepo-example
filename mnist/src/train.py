from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from mnist.src.data import MNISTDataset
from mnist.src.model import MNISTModel
from mnist.src.config import Config


if __name__ == "__main__":

    # Set device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the MNIST dataset
    train_dataset = MNISTDataset(
        root="data/", train=True, transform=transforms.ToTensor()
    )
    test_dataset = MNISTDataset(
        root="data/", train=False, transform=transforms.ToTensor()
    )

    # Create data loaders
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=Config.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=Config.batch_size, shuffle=False
    )

    # Initialize the model
    model = MNISTModel().to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)

    # Training loop
    for epoch in range(Config.num_epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device)
            targets = targets.to(device)

            # Forward pass
            scores = model(data)
            loss = criterion(scores, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Epoch {epoch+1}/{Config.num_epochs}, Step {batch_idx+1}/{len(train_loader)}, Loss: {loss.item()}"
                )

    # Evaluation on the test set
    model.eval()
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)

            scores = model(data)
            _, predictions = scores.max(1)
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)

    accuracy = float(num_correct) / float(num_samples) * 100
    print(f"Accuracy on the test set: {accuracy:.2f}%")
