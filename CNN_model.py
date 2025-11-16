# https://www.datacamp.com/tutorial/pytorch-cnn-tutorial
import torch
import yaml
from torch import nn
import torchvision
import torch.nn.functional as F
import torchmetrics
import CNN_data_loading
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

dataset_train = CNN_data_loading.dataset_train
config = CNN_data_loading.config
print("hello?")

class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):

        """
        Building blocks of convolutional neural network.

        Parameters:
            * in_channels: Number of channels in the input image (for grayscale images, 1)
            * num_classes: Number of classes to predict. In our problem, 10 (i.e digits from  0 to 9).
        """
        super(CNN, self).__init__()

        # 1st convolutional layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 2nd convolutional layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Fully connected layer
        self.fc1 = nn.Linear(64 * 12 * 12, num_classes)

    def forward(self, x):
        """
        Define the forward pass of the neural network.

        Parameters:
            x: Input tensor.

        Returns:
            torch.Tensor
                The output tensor after passing through the network.
        """
        x = F.relu(self.conv1(x))  # Apply first convolution and ReLU activation
        x = self.pool(x)           # Apply max pooling
        x = F.relu(self.conv2(x))  # Apply second convolution and ReLU activation
        x = self.pool(x)           # Apply max pooling
        x = x.reshape(x.shape[0], -1)  # Flatten the tensor
        x = self.fc1(x)            # Apply fully connected layer
        return x
        x = x.reshape(x.shape[0], -1)  # Flatten the tensor
        x = self.fc1(x)            # Apply fully connected layer
        return x

train_loader = DataLoader(dataset_train, batch_size= config['data']['batch_size'], shuffle=True, num_workers= config['data']['num_workers'])

print(f"Train samples: {len(dataset_train)}")
print(f"Number of classes: {len(dataset_train.classes)}")

model = CNN(in_channels=1, num_classes=len(CNN_data_loading.dataset_train.classes))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= config['training']['learning_rate'])


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate metrics
        total_loss += loss.item() * images.size(0)
        pred = outputs.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

        # Print progress every 10 batches
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(loader)}: loss={loss.item():.4f}")

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy



print("\n" + "="*50)
print("STARTING ONE EPOCH TEST")
print("="*50 + "\n")

print("Training...")
train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, config["device"])

# source: https://discuss.pytorch.org/t/plotting-loss-curve/42632
loss_values = []  # Store loss for each epoch

for epoch in range(1, 5):
    print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")

    # Train for one epoch
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, config["device"])

    # Store the loss
    loss_values.append(train_loss)

    print(f"Epoch {epoch + 1}: Loss={train_loss:.4f}, Acc={train_acc:.4f} ({train_acc * 100:.2f}%)")

print("\n" + "="*50)
print("RESULTS")
print("="*50)
print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} ({train_acc*100:.2f}%)")
print("="*50)
print("\n completed")

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(loss_values) + 1), loss_values, marker='o', linestyle='-', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Training Loss', fontsize=12)
plt.title('Training Loss Over Epochs', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()

print("\n" + "=" * 50)
print("TRAINING COMPLETED")
print("=" * 50)
print(f"Final Train Loss: {train_loss:.4f}")
print(f"Final Train Acc: {train_acc:.4f} ({train_acc * 100:.2f}%)")

