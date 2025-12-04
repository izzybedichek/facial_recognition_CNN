# source: https://discuss.pytorch.org/t/plotting-loss-curve/42632
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from torch import nn
import numpy as np
import cv2
from types import MethodType

def train_and_plot(model, train_loader, val_loader, optimizer, criterion, config, scheduler=None):
    """
    Trains the model for multiple epochs and plots the training loss.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer
        criterion: Loss function
        config: Dictionary containing training configuration with keys:
                - "training": {"epochs": int}
                - "device": str

    Returns: loss_values: List of training losses for each epoch
    """
    loss_values_train = []  # Store loss for each epoch
    loss_values_val = []

    early_stopper = EarlyStopper(patience=3, min_delta=10)
    for epoch in range(config["training"]["epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")

        # Train for one epoch
        avg_loss, avg_acc, precision, recall, f1 = train_one_epoch(model,
                                                                   train_loader,
                                                                   optimizer,
                                                                   criterion,
                                                                   config["device"])

        avg_loss_val, avg_acc_val, precision_val, recall_val, f1_val = val_one_epoch(model,
                                                                 val_loader,
                                                                 criterion,
                                                                 config["device"])

        # Store the loss for training
        loss_values_train.append(avg_loss)
        loss_values_val.append(avg_loss_val)

        # Print current learning rate
        if scheduler is not None:
            scheduler.step(avg_loss_val)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning Rate: {current_lr:.6f}")

        if early_stopper.early_stop(avg_loss_val):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

        print(f"(Training) Loss={avg_loss:.4f}, "
              f"Acc={avg_acc:.4f} ({avg_acc * 100:.2f}%), "
              f"Precision={precision:.4f}, "
              f"Recall={recall:.4f}, "
              f"F1={f1:.4f}")

        print(f"(Validation) Loss={avg_loss_val:.4f}, "
              f"Acc={avg_acc_val:.4f} ({avg_acc_val * 100:.2f}%), "
              f"Precision={precision_val:.4f}, "
              f"Recall={recall_val:.4f}, "
              f"F1={f1_val:.4f}")


    # Plot the training loss
    plot_training_loss(loss_values_train, loss_values_val)

    return loss_values_train, loss_values_val


def plot_training_loss(loss_values_train, loss_values_val):
    """Plots training loss over epochs."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_values_train) + 1), loss_values_train, label = "Training Loss", marker='o', linestyle='-', linewidth=2)
    plt.plot(range(1, len(loss_values_val) + 1), loss_values_val, label = "Validation Loss", marker='o', linestyle='-', linewidth=2)
    plt.legend()
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Loss Over Epochs', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """
    Train the model for one epoch.

    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on ("cuda" or "cpu")

    Returns:
        avg_loss: Average loss for the epoch
        avg_acc: Average accuracy for the epoch
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Reset
        optimizer.zero_grad()

        # Forwards
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backwards
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Setting uo
        all_labels.append(labels.cpu())
        all_preds.append(predicted.cpu())

     # Metrics
    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()

    avg_loss = total_loss / len(train_loader)
    avg_acc = correct / total

    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return avg_loss, avg_acc, precision, recall, f1

# look into Lion v Adam

def val_one_epoch(model, val_loader, criterion, device):
    """
    Validate the model for one epoch.

    Args:
        model: PyTorch model
        val_loader: DataLoader for training data
        criterion: Loss function
        device: Device to train on ("cuda" or "cpu")

    Returns:
        avg_loss: Average loss for the epoch
        avg_acc: Average accuracy for the epoch
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Track metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    avg_acc = correct / total

    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    unique_classes = sorted(set(all_labels))

    print(classification_report(all_labels, all_preds,
                                labels=unique_classes,
                                target_names=[f"Class_{i}" for i in unique_classes],
                                zero_division=0))

    return avg_loss, avg_acc, precision, recall, f1

class MulticlassSVMLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, outputs, labels):
        batch_size = outputs.size(0) # gets samples per batch for later calculation

        # extracts the score of the correct class for each sample and reshapes into a column vector for calculating the margin
        correct_scores = outputs[torch.arange(batch_size), labels].unsqueeze(1)

        # calculates the margin violations: margin - (correct_score - wrong_scores)
        margins = self.margin - correct_scores + outputs

        # zeroes out loss at the position of the correct class so that correct guesses
        # aren't compared against themselves
        margins[torch.arange(batch_size), labels] = 0

        # clamps all elements in input into the range [ min, max ] (applying hinge max on: margin - (correct_score - wrong_scores))
        loss = torch.clamp(margins, min=0)

        # average
        return loss.sum() / batch_size


def visualize_attention(model, data_loader, config, device=None):
    """
    Visualizes spatial & channel attention for a single example.
    Call with:
        visualize_attention(model, val_loader, config)
    """

    if device is None:
        device = config.get("device", "cpu")

    # patch spatial attention to extract heatmap

    def patch_spatial_attention(model):
        for module in model.modules():
            if module.__class__.__name__ == "SpatialAttention":

                # keep original forward
                original_forward = module.forward

                # define wrapped forward
                def new_forward(self, x):
                    # channel attention
                    channel_weight = self.channel_attention(x)
                    self.channel_attention_output = channel_weight.detach().cpu()
                    x = x * channel_weight

                    # spatial attention
                    avg_out = torch.mean(x, dim=1, keepdim=True)
                    max_out, _ = torch.max(x, dim=1, keepdim=True)
                    spatial_input = torch.cat([avg_out, max_out], dim=1)
                    spatial_weight = self.spatial_attention(spatial_input)
                    self.spatial_attention_output = spatial_weight.detach().cpu()

                    return x * spatial_weight

                # bind dynamically
                module.forward = MethodType(new_forward, module)

    patch_spatial_attention(model)

    #overlay function

    def overlay_heatmap(img, attn_map):
        img = img.squeeze().cpu().numpy()
        attn = attn_map.cpu().numpy()

        attn = cv2.resize(attn, img.shape, interpolation=cv2.INTER_CUBIC)
        attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

        heatmap = cv2.applyColorMap((attn_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        img_rgb = np.stack([img, img, img], axis=2)
        overlay = 0.5 * heatmap + 0.5 * img_rgb
        overlay = overlay / overlay.max()
        return overlay

    # run sample thru model
    model.eval()
    sample = next(iter(data_loader))[0][0]  # first image of first batch
    sample = sample.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(sample)

    # retrieve attention maps
    spatial_attn = None
    channel_attn = None

    for module in model.modules():
        if module.__class__.__name__ == "SpatialAttention":
            spatial_attn = module.spatial_attention_output[0, 0]
            channel_attn = module.channel_attention_output[0, :, 0, 0]
            break

    # plotting
    plt.figure(figsize=(15, 4))

    # input
    plt.subplot(1, 4, 1)
    plt.title("Input Image")
    plt.imshow(sample[0, 0].cpu(), cmap='gray')
    plt.axis("off")

    # spatial attn
    plt.subplot(1, 4, 2)
    plt.title("Spatial Attention")
    plt.imshow(spatial_attn, cmap='jet')
    plt.axis("off")

    # overlay
    plt.subplot(1, 4, 3)
    plt.title("Overlay")
    plt.imshow(overlay_heatmap(sample[0, 0], spatial_attn))
    plt.axis("off")

    # channel weights
    plt.subplot(1, 4, 4)
    plt.title("Channel Attention Weights")
    plt.bar(range(len(channel_attn)), channel_attn.numpy())
    plt.xlabel("Channel")
    plt.tight_layout()

    plt.show()

    print("\nModel output logits:\n", out.cpu().numpy())



# from https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
# DIDNT END USING
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.best_model = None

    def early_stop(self, validation_loss, model=None):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            if model is not None:
                self.best_model = model.state_dict().copy()  # Save best weights
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False