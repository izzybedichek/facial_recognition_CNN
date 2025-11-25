# https://www.datacamp.com/tutorial/pytorch-cnn-tutorial
import torch
from torch import nn
import torch.nn.functional as F
from CNN_data_loading import train_loader, val_loader, config
import train_and_plot
import torch.optim as optim
from lion_pytorch import Lion
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts # new

# Train for 30-50 epochs
class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):

        """
        Building blocks of convolutional neural network.
        """
        # convolutional layer
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels = 32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # avg pool
        self.maxpool = nn.AdaptiveMaxPool2d((6, 6))

        # dropout
        self.dropout = nn.Dropout(config["model"]["dropout"])

        # fully connected/dense layer
        self.fc1 = nn.Linear(256*6*6, 256)
        self.fc2 = nn.Linear(256, num_classes)


    def forward(self, x):
        """
        Define the forward pass of the neural network.
        """
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))

        x = self.maxpool(x)

        x = torch.flatten(x, 1)

        x = F.leaky_relu(self.fc1(x))

        x = self.dropout(x)

        x = self.fc2(x)

        return x



def create_model(config):
    return CNN(in_channels=1, num_classes=config["model"]["num_classes"])

model = CNN(in_channels = 1, num_classes = config["model"]["num_classes"])
model = model.to(config['device'])

criterion = nn.CrossEntropyLoss() # look into

# optimizer = optim.AdamW(model.parameters(),
#                        lr = config['training']['learning_rate'],
#                        weight_decay= config["training"]["weight_decay"])

optimizer = Lion(model.parameters(),
                config['training']['learning_rate'],
                weight_decay=config["training"]["weight_decay"])

scheduler = CosineAnnealingWarmRestarts( # new
    optimizer,
    T_0=10,  # restart every 10 epochs
    T_mult=2
)

loss_values = train_and_plot.train_and_plot(model,
                                        train_loader,
                                        val_loader,
                                        optimizer,
                                        criterion,
                                        config,
                                        scheduler=scheduler)



