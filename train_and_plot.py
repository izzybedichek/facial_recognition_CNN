# source: https://discuss.pytorch.org/t/plotting-loss-curve/42632
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from torch import nn

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

# from https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch

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
