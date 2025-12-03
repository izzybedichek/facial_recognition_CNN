import torch
from torch import nn
import torch.optim as optim
from CNN_data_loading import train_loader, val_loader, config
from train_and_plot import train_and_plot, MulticlassSVMLoss
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class CNN_LSTM(nn.Module):
    def __init__(self, in_channels, num_classes, lstm_hidden=128, lstm_layers=1):
        """
        CNN-LSTM hybrid for spatial-temporal feature learning.

        Args:
            in_channels: Input channels (1 for grayscale)
            num_classes: Number of output classes
            lstm_hidden: LSTM hidden dimension (keep small for speed)
            lstm_layers: Number of LSTM layers (1-2 recommended)
        """
        super(CNN_LSTM, self).__init__()

        # ========== CNN Feature Extractor ==========
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
        self.dropout_cnn = nn.Dropout(config["model"]["dropout"])

        # ========== LSTM Layer ==========
        # Input: flattened CNN features treated as sequence
        self.lstm_input_size = 128 * 6 * 6
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers

        # Reshape CNN output into sequences for LSTM
        # We'll split the flattened features into chunks
        self.sequence_length = 36  # 6*6 spatial locations as sequence
        self.feature_dim = 128  # depth at each location

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.0 if lstm_layers == 1 else 0.2,  # dropout between LSTM layers
            bidirectional=False  # Set True for BiLSTM (doubles hidden_size)
        )

        self.dropout_lstm = nn.Dropout(0.3)

        # ========== Classifier ==========
        # Input size is lstm_hidden (not 128*6*6 anymore!)
        self.fc1 = nn.Linear(lstm_hidden, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Forward pass through CNN then LSTM.
        """
        batch_size = x.size(0)

        # ========== CNN Feature Extraction ==========
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = F.relu(self.bn1b(self.conv1b(x)))

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        x = self.avgpool(x)  # [batch, 128, 6, 6]

        # ========== Prepare for LSTM ==========
        # Reshape: [batch, channels, h, w] -> [batch, h*w, channels]
        # This treats each spatial location as a time step
        x = x.permute(0, 2, 3, 1)  # [batch, 6, 6, 128]
        x = x.reshape(batch_size, self.sequence_length, self.feature_dim)  # [batch, 36, 128]

        # ========== LSTM Processing ==========
        # LSTM output: [batch, seq_len, hidden_size]
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state for classification
        # h_n shape: [num_layers, batch, hidden_size]
        x = h_n[-1]  # Take last layer: [batch, hidden_size]

        # Alternative: use mean pooling over sequence
        # x = torch.mean(lstm_out, dim=1)  # [batch, hidden_size]

        x = self.dropout_lstm(x)

        # ========== Classification Head ==========
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# ========== Alternative: Simpler CNN-LSTM for faster runtime ==========
class CNN_LSTM_Fast(nn.Module):
    """Lightweight version with LSTM only on global features"""

    def __init__(self, in_channels, num_classes, lstm_hidden=64):
        super(CNN_LSTM_Fast, self).__init__()

        # Reuse your CNN backbone
        self.conv1 = nn.Conv2d(in_channels, 32,
                               3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32,
                                3, 1, 1)

        self.bn1b = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64,
                               3, 1, 1)

        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128,
                               3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global pooling
        self.dropout = nn.Dropout(config["model"]["dropout"])

        # Simple LSTM on global CNN features
        # Treat batch as having implicit temporal structure
        self.lstm = nn.LSTM(128, lstm_hidden, 1, batch_first=True)

        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        batch_size = x.size(0)

        # CNN
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.avgpool(x)  # [batch, 128, 1, 1]

        x = torch.flatten(x, 1)  # [batch, 128]
        x = self.dropout(x)

        # LSTM expects [batch, seq, features]
        x = x.unsqueeze(1)  # [batch, 1, 128] - single timestep
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # [batch, hidden]

        x = self.fc(x)
        return x


# ========== Training Setup ==========
# Choose model version
model = CNN_LSTM(
    in_channels=1,
    num_classes=config["model"]["num_classes"],
    lstm_hidden=128,  # Reduce to 64 for faster runtime
    lstm_layers=1  # Keep at 1 for speed, try 2 for accuracy
)

# For fastest runtime:
# model = CNN_LSTM_Fast(in_channels=1, num_classes=config["model"]["num_classes"])

model = model.to(config['device'])

criterion = MulticlassSVMLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=config['training']['learning_rate'] * 0.5,  # Reduce LR for LSTM stability
    weight_decay=config["training"]["weight_decay"]
)

scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,
    T_mult=2
)

# ========== Training ==========
loss_values = train_and_plot(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    config,
    scheduler=scheduler
)