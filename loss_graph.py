# source: https://discuss.pytorch.org/t/plotting-loss-curve/42632
import matplotlib.pyplot as plt

def train_and_plot(model, train_loader, optimizer, criterion, config):
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
    loss_values = []  # Store loss for each epoch

    for epoch in range(config["training"]["epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")

        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, config["device"]
        )

        # Store the loss
        loss_values.append(train_loss)

        print(f"Loss={train_loss:.4f}, Acc={train_acc:.4f} ({train_acc * 100:.2f}%)")

    # Plot the training loss
    plot_training_loss(loss_values)

    return loss_values


def plot_training_loss(loss_values):
    """Plots training loss over epochs."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_values) + 1), loss_values, marker='o', linestyle='-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Training Loss Over Epochs', fontsize=14)
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

    avg_loss = total_loss / len(train_loader)
    avg_acc = correct / total

    return avg_loss, avg_acc