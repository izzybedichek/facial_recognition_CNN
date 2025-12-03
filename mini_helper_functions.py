# calculating mean and std for Normalize()
# from https://saturncloud.io/blog/how-to-normalize-image-dataset-using-pytorch/

def get_mean_std(loader):
    mean = 0.
    std = 0.
    total_images = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)  # flatten H*W
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images

    return mean, std

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F


def train(model, dataset, device, epochs, lr):
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(loader):.4f}")

from torch.utils.data import Dataset
class AugmentedMinorityDataset(Dataset):
    """Creates N augmented copies of minority class samples"""

    def __init__(self, original_dataset, minority_indices, transform, n_copies=5):
        self.original_dataset = original_dataset
        self.minority_indices = minority_indices
        self.transform = transform
        self.n_copies = n_copies

    def __len__(self):
        return len(self.minority_indices) * self.n_copies

    def __getitem__(self, idx):
        original_idx = self.minority_indices[idx % len(self.minority_indices)]
        image, label = self.original_dataset[original_idx]
        # Apply random augmentation each time
        return self.transform(image), label


