import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, Subset
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from dataset_loader import OralCancerDataset


def get_random_subset(dataset, subset_size):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return Subset(dataset, indices[:min(subset_size, len(dataset))])


def show_sample_predictions(model, loader, device, class_names, num_images=6):
    model.eval()
    images_shown = 0

    plt.figure(figsize=(15, 8))

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for i in range(images.size(0)):
                if images_shown == num_images:
                    break

                img = images[i].cpu().permute(1, 2, 0).numpy()

                # Normalize for display
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)

                plt.subplot(2, 3, images_shown + 1)
                plt.imshow(img)
                plt.title(f"Pred: {class_names[preds[i].item()]}\nTrue: {class_names[labels[i].item()]}")
                plt.axis("off")

                images_shown += 1

            if images_shown == num_images:
                break

    plt.tight_layout()
    plt.savefig("sample_predictions.png")
    plt.show()


def main():
    print("STEP 1: Starting training...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -----------------------------
    # Load datasets
    # -----------------------------
    print("Loading training dataset...")
    full_train_dataset = OralCancerDataset("Data/train")
    train_dataset = get_random_subset(full_train_dataset, 12000)
    print("Training samples:", len(train_dataset))

    print("Loading validation dataset...")
    full_val_dataset = OralCancerDataset("Data/val")
    val_dataset = get_random_subset(full_val_dataset, 3000)
    print("Validation samples:", len(val_dataset))

    print("Loading test dataset...")
    full_test_dataset = OralCancerDataset("Data/test")
    test_dataset = get_random_subset(full_test_dataset, 2000)
    print("Test samples:", len(test_dataset))

    # -----------------------------
    # DataLoaders
    # -----------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    print("DataLoaders ready")

    # -----------------------------
    # Model
    # -----------------------------
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epochs = 6
    best_val_loss = float('inf')

    # -----------------------------
    # Lists for graph plotting
    # -----------------------------
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # -----------------------------
    # Training Loop
    # -----------------------------
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            if batch_idx == 0:
                print("First training batch loaded")

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        train_losses.append(train_loss)
        train_accuracies.append(train_acc * 100)

        print(f"Training Loss: {train_loss:.4f}")
        print(f"Training Accuracy: {train_acc:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        val_losses.append(val_loss)
        val_accuracies.append(val_acc * 100)

        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_oral_cancer_model.pth")
            print("Best model saved!")

    print("\nTraining completed successfully.")

    # -----------------------------
    # Plot Accuracy Graph
    # -----------------------------
    epochs_range = range(1, epochs + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, train_accuracies, marker='o', label='Training Accuracy')
    plt.plot(epochs_range, val_accuracies, marker='o', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_graph.png")
    plt.show()

    # -----------------------------
    # Plot Loss Graph
    # -----------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, train_losses, marker='o', label='Training Loss')
    plt.plot(epochs_range, val_losses, marker='o', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_graph.png")
    plt.show()

    # -----------------------------
    # Load best model before testing
    # -----------------------------
    model.load_state_dict(torch.load("best_oral_cancer_model.pth"))
    model.eval()

    # -----------------------------
    # Final Testing
    # -----------------------------
    all_preds = []
    all_labels = []
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

    test_acc = 100 * test_correct / test_total
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Normal', 'OSCC'],
        yticklabels=['Normal', 'OSCC'],
        linewidths=0.5,
        linecolor='black',
        cbar=False
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()

    # -----------------------------
    # Classification Report
    # -----------------------------
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'OSCC']))

    # -----------------------------
    # Sample Predictions
    # -----------------------------
    show_sample_predictions(model, test_loader, device, ['Normal', 'OSCC'])

    print("\nAll result files generated successfully.")


if __name__ == "__main__":
    main()