# https://www.datacamp.com/tutorial/pytorch-cnn-tutorial

import torch
from torch import nn
from torch.nn.functional import sigmoid
import torch.optim as optim
from CNN_data_loading import train_loader, val_loader, config, class_weights_tensor
from train_and_plot import train_and_plot, MulticlassSVMLoss
import torch.nn.functional as F
from lion_pytorch import Lion
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau  # new


class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):

        """
        Building blocks of convolutional neural network.
        """
        ### flow goes: convolutional layer --> batch norm x 3, then adaptive avg pool, the fc layers ###
        # convolutional layer
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels=32,
                               kernel_size=3,
                               stride=1,
                               padding=1)

        self.bn1 = nn.BatchNorm2d(32)

        self.conv1b = nn.Conv2d(in_channels=32, out_channels=32, # here in and out are the same (ResNet inspired)
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.bn1b = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels = 32, out_channels=64,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv2b = nn.Conv2d(in_channels=64, out_channels=64,  # here in and out are the same (ResNet inspired)
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.bn2b = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # max pool
        self.pool = nn.MaxPool2d(2, 2)

        # avg pool
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # dropout layer
        self.dropout = nn.Dropout(config["model"]["dropout"])

        # fully connected/dense layers
        self.fc1 = nn.Linear(128*6*6, 128)
        self.fc2 = nn.Linear(128, num_classes)


    def forward(self, x):
        """
        Define the forward pass of the neural network.
        """
        # convolutional
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = F.relu(self.bn1b(self.conv1b(x)))

        x = F.sigmoid(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = F.relu(self.bn2b(self.conv2b(x)))

        x = F.relu(self.bn3(self.conv3(x)))

        x = self.avgpool(x)

        # dense/connected
        x = torch.flatten(x, 1)

        x = self.dropout(x)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x

model = CNN(in_channels = 1,
            num_classes = config["model"]["num_classes"])

model = model.to(config['device'])

#criterion = nn.CrossEntropyLoss(label_smoothing=0.6) # the label smoothing makes it equivalently useful as the MulticlassSVM loss
criterion = MulticlassSVMLoss() # works better in most cases

# optimizer = optim.AdamW(model.parameters(),
#                        lr = config['training']['learning_rate'],
#                        weight_decay= config["training"]["weight_decay"])

optimizer = Lion(model.parameters(),
                 config['training']['learning_rate']/3,
                 weight_decay=config["training"]["weight_decay"])

scheduler = CosineAnnealingWarmRestarts( # I have found this to work better than ReduceLROnPlateau
    optimizer,
    T_0=10,  # restart every 10 epochs
    T_mult=2
)

# scheduler = ReduceLROnPlateau(
#     optimizer,
#     mode='min',
#     factor = 0.5,
#     patience = 3,
#     min_lr=.0000001
# )

loss_values = train_and_plot(model,
                             train_loader,
                             val_loader,
                             optimizer,
                             criterion,
                             config,
                             scheduler=scheduler)



