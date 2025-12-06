# https://www.datacamp.com/tutorial/pytorch-cnn-tutorial
import torch
from torch import nn
import torch.nn.functional as F
import CNN_data_loading
import train_and_plot
import torch.optim as optim
from lion_pytorch import Lion
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from focal_loss import FocalLoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


from train_and_plot import evaluate_testset
import CNN_SpatialAttention
from CNN_SpatialAttention import SpatialAttention
dataset_train = CNN_data_loading.train_loader
dataset_val = CNN_data_loading.val_loader
dataset_test = CNN_data_loading.test_loader
config = CNN_data_loading.config
device = CNN_data_loading.device

print("Starting CNN_model.py")

class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):

        """
        Building blocks of convolutional neural network.
        """
        # convolutional layer
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(in_channels = 32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1b = nn.BatchNorm2d(32)

        # max pool
        self.maxpool = nn.AdaptiveMaxPool2d((12, 12))

        self.conv2 = nn.Conv2d(in_channels = 32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # first spatial attention layer
        self.attention1 = SpatialAttention(128)

        # max pool
        self.maxpool2 = nn.AdaptiveMaxPool2d((6, 6))

 
        # convolutional layer block 2
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv4b = nn.Conv2d(in_channels = 256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4b = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(in_channels = 256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

        # ADD ATTENTION HERE - before final pooling
        self.attention2 = SpatialAttention(512)

        # max pool
        self.maxpool3 = nn.AdaptiveAvgPool2d((3, 3))

        # dropout
        self.dropout = nn.Dropout(config["model"]["dropout"])

        # fully connected/dense layer
        self.fc1 = nn.Linear(512*3*3, 512)
        self.fc2 = nn.Linear(512, num_classes)


    def forward(self, x):
        """
        Define the forward pass of the neural network.
        """
        
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn1b(self.conv1b(x)))

        x = self.maxpool(x)

        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))

        x = self.attention1(x)
        x = self.maxpool2(x)

        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = F.leaky_relu(self.bn4b(self.conv4b(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))

        x = self.attention2(x)
        x = self.maxpool3(x)

        x = torch.flatten(x, 1)

        x = F.leaky_relu(self.fc1(x))

        x = self.dropout(x)

        x = self.fc2(x)

        return x

model = CNN(in_channels = 1, num_classes = config["model"]["num_classes"])
model = model.to(config['device'])



#trying a weighted cross entropy loss

#get the targets from the training dataset loader
train_targets = []
for _, labels in dataset_train:
    train_targets.extend(labels.tolist())

#calculate normalized class weights using skikit learn
class_weights_bal = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_targets),
    y=train_targets
)

class_weights_bal = torch.tensor(class_weights_bal, dtype=torch.float).to(device)
# criterion = nn.CrossEntropyLoss(weight=class_weights_bal) # look into
criterion = nn.CrossEntropyLoss()

# class_weights_bal = torch.tensor(class_weights_bal, dtype=torch.float).to(device)
# criterion = FocalLoss(alpha=class_weights_bal, gamma=2.0, reduction='mean')

optimizer = optim.Adam(model.parameters(),
                       lr = config['training']['learning_rate'],
                       weight_decay= config["training"]["weight_decay"])

#optimizer = Lion(model.parameters(),
#                 config['training']['learning_rate'],
#                 weight_decay=config["training"]["weight_decay"])

scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=5,
    T_mult=2,
    eta_min=1e-6
)

loss_values = train_and_plot.train_and_plot(model,
                                        dataset_train,
                                        dataset_val,
                                        optimizer,
                                        criterion,
                                        config,
                                        scheduler=scheduler)


print("\n" + "="*60)
print("TRAINING COMPLETE - EVALUATING ON TEST SET")
print("="*60)

# Evaluate on the test dataset and display classification report
y_true, y_pred = evaluate_testset(model, dataset_test, device)
