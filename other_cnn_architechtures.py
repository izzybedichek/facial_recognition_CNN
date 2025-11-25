import torch
from torch import nn
import torch.nn.functional as F

class OtherCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # Add more conv blocks with residual-like connections
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2b = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2b = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv3b = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3b = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Block 1: 48x48 -> 24x24
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        # Block 2: 24x24 -> 12x12
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool(x)

        # Block 3: 12x12 -> 6x6
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))
        x = self.pool(x)

        # Block 4: 6x6 -> 3x3 -> 1x1
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.adaptive_pool(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        """
        in_channels: number of input channels
        growth_rate: how many channels each layer adds (typically 12-32)
        num_layers: how many conv layers in this block
        """
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(
                self._make_dense_layer(in_channels + i * growth_rate, growth_rate)
            )

    def _make_dense_layer(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            # Concatenate all previous features
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)

        # Return concatenation of all features
        return torch.cat(features, dim=1)


class TransitionLayer(nn.Module):
    """Reduces spatial dimensions and number of channels between dense blocks"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        x = self.bn(x)
        x = F.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x

class DenseCNN(nn.Module):
    def __init__(self, in_channels, num_classes, growth_rate=16):
        super().__init__()

        # Initial: 48x48x1 -> 48x48x32
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # Dense Block 1: 48x48, 3 layers
        self.dense1 = DenseBlock(32, growth_rate, num_layers=3)
        # 32 + 16*3 = 80 channels

        # Transition: 48x48 -> 24x24, 80 -> 40
        self.trans1 = TransitionLayer(80, 40)

        # Dense Block 2: 24x24, 4 layers
        self.dense2 = DenseBlock(40, growth_rate, num_layers=4)
        # 40 + 16*4 = 104 channels

        # Transition: 24x24 -> 12x12, 104 -> 52
        self.trans2 = TransitionLayer(104, 52)

        # Dense Block 3: 12x12, 4 layers
        self.dense3 = DenseBlock(52, growth_rate, num_layers=4)
        # 52 + 16*4 = 116 channels

        # Final
        self.bn_final = nn.BatchNorm2d(116)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(116, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.dense1(x)
        x = self.trans1(x)

        x = self.dense2(x)
        x = self.trans2(x)

        x = self.dense3(x)

        x = F.relu(self.bn_final(x))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
