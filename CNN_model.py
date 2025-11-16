# https://www.datacamp.com/tutorial/pytorch-cnn-tutorial
import torch
import yaml
from torch import nn
import torchvision
import torch.nn.functional as F
import torchmetrics
import CNN_data_loading
import loss_graph
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

dataset_train = CNN_data_loading.train_loader
config = CNN_data_loading.config


print("hello?")

class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):

        """
        Building blocks of convolutional neural network.
        """
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels = 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 12 * 12, num_classes)

    def forward(self, x):
        """
        Define the forward pass of the neural network.
        """
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

model = CNN(in_channels= 1, num_classes= config["model"]["num_classes"])
model = model.to(config['device'])
criterion = nn.CrossEntropyLoss() # look into
optimizer = optim.Adam(model.parameters(), lr = config['training']['learning_rate'])

loss_values = loss_graph.train_and_plot(model, dataset_train, optimizer, criterion, config)

# train_loader = DataLoader(dataset_train, batch_size= config['data']['batch_size'], shuffle = True)
# images, labels = next(iter(train_loader))
# print('Batch shape:', images.shape, 'Labels shape:', labels.shape)
#
# print(f"Train samples: {len(dataset_train)}")
# print(f"Number of classes: {len(dataset_train.classes)}")
#
# model = CNN(in_channels= 1, num_classes= config["model"]["num_classes"])
#
# criterion = nn.CrossEntropyLoss() # look into
# optimizer = optim.Adam(model.parameters(), lr = config['training']['learning_rate'])
#
#
# def train_one_epoch(model, loader, optimizer, criterion, device):
#     model.train()
#     total_loss = 0.0
#     correct = 0
#     total = 0
#
#     for batch_idx, (images, labels) in enumerate(loader):
#         images, labels = images.to(device), labels.to(device)
#
#         # Forward pass
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#
#         # Backward pass
#         loss.backward()
#         optimizer.step()
#
#         # Calculate metrics
#         total_loss += loss.item() * images.size(0)
#         pred = outputs.argmax(dim=1)
#         correct += (pred == labels).sum().item()
#         total += labels.size(0)
#
#         # Print progress every 10 batches
#         if batch_idx % 10 == 0:
#             print(f"  Batch {batch_idx}/{len(loader)}: loss={loss.item():.4f}")
#
#     avg_loss = total_loss / total
#     accuracy = correct / total
#     return avg_loss, accuracy
#
# print("STARTING ONE EPOCH TEST")
#
# print("Training")
# train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, config["device"])
#
# print("Epoch results:")
#
# print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} ({train_acc*100:.2f}%)")
#
# print("\n Epoch completed")
#
# print("Training complete")
#
# print(f"Train Loss: {train_loss:.4f}")
# print(f"Train Accuracy: {train_acc:.4f} ({train_acc * 100:.2f}%)")

