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
dataset_val = CNN_data_loading.val_loader
config = CNN_data_loading.config

class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):

        """
        Building blocks of convolutional neural network.
        """
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels = 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # classifier layer
        self.fc1 = nn.Linear(128 * 6 * 6, num_classes)

    def forward(self, x):
        """
        Define the forward pass of the neural network.
        """
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.leaky_relu(self.conv3(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)

        # classification layer
        x = self.fc1(x)

        return x

model = CNN(in_channels= 1, num_classes= config["model"]["num_classes"])
model = model.to(config['device'])
criterion = nn.CrossEntropyLoss() # look into
optimizer = optim.Adam(model.parameters(), lr = config['training']['learning_rate'], weight_decay=1e-4)

loss_values = loss_graph.train_and_plot(model, dataset_train, dataset_val, optimizer, criterion, config)


