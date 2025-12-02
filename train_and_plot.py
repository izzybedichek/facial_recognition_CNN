# source: https://discuss.pytorch.org/t/plotting-loss-curve/42632
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def train_and_plot(model, train_loader, val_loader, optimizer, criterion, config):
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

        if early_stopper.early_stop(avg_loss_val):
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
    plt.plot(range(1, len(loss_values_train) + 1), loss_values_val, label = "Validation Loss", marker='o', linestyle='-', linewidth=2)
    plt.plot(range(1, len(loss_values_train) + 1), loss_values_train, label = "Training Loss", marker='o', linestyle='-', linewidth=2)
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

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

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

    return avg_loss, avg_acc, precision, recall, f1

# from https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


#plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, emotions=EMOTIONS):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=emotions, yticklabels=emotions)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()