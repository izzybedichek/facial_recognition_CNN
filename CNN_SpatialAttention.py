import torch
from torch import nn
import torch.optim as optim
from CNN_data_loading import train_loader, val_loader, config
from train_and_plot import train_and_plot, MulticlassSVMLoss, evaluate_testset
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import CNN_data_loading
from focal_loss import FocalLoss

dataset_train = CNN_data_loading.train_loader
dataset_val = CNN_data_loading.val_loader
dataset_test = CNN_data_loading.test_loader
config = CNN_data_loading.config
device = CNN_data_loading.device


class SpatialAttention(nn.Module):
    """
    Spatial attention for 2D feature maps.
    Works directly on CNN feature maps before flattening.
    """

    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()

        # Getting rid of channel attention did not affect metrics, so removed to parse down

        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention removed, see comment in Init

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_weight = self.spatial_attention(spatial_input)

        # Store BOTH versions
        self.spatial_attention_output = spatial_weight.detach().cpu()  # For viz
        self.spatial_attention_weights = spatial_weight  # For loss (has gradients)

        return x * spatial_weight


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
        x = self.spatial_attention1(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.bn1b(self.conv1b(x)))

        x = F.relu(self.bn2(self.conv2(x)))
        #x = self.spatial_attention2(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.bn3(self.conv3(x)))

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

criterion = nn.CrossEntropyLoss(label_smoothing=0.4)
# criterion = MulticlassSVMLoss() # works best


# Get normalized class weights for focal loss
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

# Focal Loss
# class_weights_bal = torch.tensor(class_weights_bal, dtype=torch.float).to(device)
# criterion = FocalLoss(alpha=class_weights_bal, gamma=2.0, reduction='mean')

optimizer = optim.AdamW(
    model.parameters(),
    lr=config['training']['learning_rate'],
    weight_decay=config["training"]["weight_decay"]
)

scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=8,
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

print("\n" + "="*60)
print("TRAINING COMPLETE - EVALUATING ON TEST SET")
print("="*60)

# Evaluate on the test dataset and display classification report
y_true, y_pred = evaluate_testset(model, dataset_test, device)
