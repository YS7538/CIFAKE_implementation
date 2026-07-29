import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms

from config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    TRAIN_SPLIT,
    NUM_WORKERS,
    CHECKPOINT_DIR,
)

from models.cnn_improved import ImprovedCNN
from models.cnn_paper import PaperCNN
from models.ResNet import ResNet18Model
from models.vit import ViTModel

from utilis.dataset import get_dataloaders
from utilis.train import train_model
from utilis.evaluate import evaluate_model
from utilis.visualize import plot_loss, plot_accuracy


def main():

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    # Data loaders
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        transform=transform,
        batch_size=BATCH_SIZE,
        train_split=TRAIN_SPLIT,
        num_workers=NUM_WORKERS,
    )

    print(f"Classes: {class_names}")

    # Model
    model = ViTModel().to(device)

    # Loss Function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # Train
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=EPOCHS,
        checkpoint_path=f"{CHECKPOINT_DIR}/Vit.pth",
    )

    # Evaluate
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
    )

    # Print Metrics
    print("\n===== Test Results =====")

    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1']:.4f}")

    print("\nConfusion Matrix")
    print(metrics["confusion_matrix"])

    print("\nClassification Report")
    print(metrics["classification_report"])

    # Visualizations
    plot_loss(history)
    plot_accuracy(history)


if __name__ == "__main__":
    main()
