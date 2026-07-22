import matplotlib.pyplot as plt


def plot_loss(history):
    """
    Plots training and validation loss.
    """

    plt.figure(figsize=(8, 5))

    plt.plot(
        history["train_loss"],
        label="Train Loss",
        marker="o"
    )

    plt.plot(
        history["val_loss"],
        label="Validation Loss",
        marker="o"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.show()


def plot_accuracy(history):
    """
    Plots validation accuracy.
    """

    plt.figure(figsize=(8, 5))

    plt.plot(
        history["val_acc"],
        label="Validation Accuracy",
        marker="o"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.grid(True)

    plt.show()