# https://www.datacamp.com/tutorial/pytorch-cnn-tutorial
import torch
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
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels = 32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels = 64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.avgpool = nn.AdaptiveMaxPool2d((6, 6))

        self.dropout = nn.Dropout(config["model"]["dropout"])

        self.fc1 = nn.Linear(128*6*6, 128)
        self.fc2 = nn.Linear(128, num_classes)


    def forward(self, x):
        """
        Define the forward pass of the neural network.
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

model = CNN(in_channels= 1, num_classes= config["model"]["num_classes"])
model = model.to(config['device'])

criterion = nn.CrossEntropyLoss() # look into

optimizer = optim.Adam(model.parameters(),
                       lr = config['training']['learning_rate'],
                       weight_decay= config["training"]["weight_decay"])

loss_values = loss_graph.train_and_plot(model,
                                        dataset_train,
                                        dataset_val,
                                        optimizer,
                                        criterion,
                                        config)


