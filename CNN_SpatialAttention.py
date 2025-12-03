import torch
from torch import nn
import torch.optim as optim
from CNN_data_loading import train_loader, val_loader, config
from train_and_plot import train_and_plot, MulticlassSVMLoss
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class SpatialAttention(nn.Module):
    """
    Spatial attention for 2D feature maps.
    Works directly on CNN feature maps before flattening.
    """

    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )

        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: [batch, channels, height, width]
        Returns:
            attention-weighted features [batch, channels, height, width]
        """
        # Channel attention
        channel_weight = self.channel_attention(x)
        x = x * channel_weight

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_weight = self.spatial_attention(spatial_input)
        x = x * spatial_weight

        return x


class CNN_SpatialAttention(nn.Module):
    """
    CNN with spatial attention on feature maps.
    Best for: emphasizing important spatial regions.
    """

    def __init__(self, in_channels, num_classes):
        super(CNN_SpatialAttention, self).__init__()


        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv1b = nn.Conv2d(in_channels=32, out_channels=32,
                                kernel_size=3, stride=1, padding=1)
        self.bn1b = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

        # Spatial Attention
        # Apply before final pooling
        self.spatial_attention = SpatialAttention(128)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.dropout = nn.Dropout(config["model"]["dropout"])

        # fully connected layers
        self.fc1 = nn.Linear(128 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = F.relu(self.bn1b(self.conv1b(x)))

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        # Apply spatial attention to feature maps
        x = self.spatial_attention(x)

        x = self.avgpool(x)

        # flatten/classify
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x



model = CNN_SpatialAttention(
    in_channels=1,
    num_classes=config["model"]["num_classes"]
)

model = model.to(config['device'])

criterion = MulticlassSVMLoss() # works best

optimizer = optim.AdamW(
    model.parameters(),
    lr=config['training']['learning_rate'],
    weight_decay=config["training"]["weight_decay"]
)

scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,
    T_mult=2
)

loss_values = train_and_plot(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    config,
    scheduler=scheduler
)