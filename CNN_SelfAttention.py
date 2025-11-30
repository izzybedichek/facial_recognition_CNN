import torch
from torch import nn
import torch.optim as optim
from CNN_data_loading import train_loader, val_loader, config
from train_and_plot import train_and_plot, MulticlassSVMLoss
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class SelfAttention(nn.Module):
    """
    Self-attention mechanism for capturing feature relationships.
    Works on flattened spatial features.
    """

    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        # Reduce dimensionality for efficiency
        self.query = nn.Linear(in_dim, in_dim // 8)
        self.key = nn.Linear(in_dim, in_dim // 8)
        self.value = nn.Linear(in_dim, in_dim)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable weight

    def forward(self, x):
        """
        Args:
            x: [batch_size, in_dim] flattened features
        Returns:
            attention-weighted features [batch_size, in_dim]
        """
        batch_size = x.size(0)

        # Generate Q, K, V
        Q = self.query(x)  # [batch, in_dim//8]
        K = self.key(x)  # [batch, in_dim//8]
        V = self.value(x)  # [batch, in_dim]

        # Attention scores
        attention = torch.matmul(Q, K.transpose(-2, -1))  # [batch, batch]
        attention = attention / (K.size(-1) ** 0.5)  # Scale
        attention = F.softmax(attention, dim=-1)

        # Apply attention to values
        out = torch.matmul(attention, V)  # [batch, in_dim]

        # Residual connection with learnable weight
        out = self.gamma * out + x

        return out


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


class CNN_SelfAttention(nn.Module):
    """
    CNN with self-attention on flattened features.
    Best for: capturing global relationships between features.
    """

    def __init__(self, in_channels, num_classes):
        super(CNN_SelfAttention, self).__init__()

        # ========== CNN Backbone ==========
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
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.dropout = nn.Dropout(config["model"]["dropout"])

        # ========== Self-Attention Layer ==========
        feature_dim = 128 * 6 * 6  # Flattened feature dimension
        self.self_attention = SelfAttention(feature_dim)

        # ========== Classifier ==========
        self.fc1 = nn.Linear(feature_dim, 128)
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

        x = self.avgpool(x)

        # Flatten
        x = torch.flatten(x, 1)

        x = self.dropout(x)

        # Apply self-attention
        x = self.self_attention(x)

        # Classification
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class CNN_SpatialAttention(nn.Module):
    """
    CNN with spatial attention on feature maps.
    Best for: emphasizing important spatial regions.
    """

    def __init__(self, in_channels, num_classes):
        super(CNN_SpatialAttention, self).__init__()

        # ========== CNN Backbone ==========
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

        # ========== Spatial Attention ==========
        # Apply after conv3 (before final pooling)
        self.spatial_attention = SpatialAttention(128)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.dropout = nn.Dropout(config["model"]["dropout"])

        # ========== Classifier ==========
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

        # Flatten and classify
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# ========== Choose Your Model ==========

# # Option 1: Self-attention on flattened features (lighter, faster)
# model = CNN_SelfAttention(
#     in_channels=1,
#     num_classes=config["model"]["num_classes"]
# )

# Option 2: Spatial attention on feature maps (more interpretable)
model = CNN_SpatialAttention(
    in_channels=1,
    num_classes=config["model"]["num_classes"]
)

model = model.to(config['device'])

criterion = MulticlassSVMLoss()

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