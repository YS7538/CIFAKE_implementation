import time

import torch


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    checkpoint_path,
    progress_interval=50,
):

    print(f"Training on: {next(model.parameters()).device}")

    train_loss_history = []
    val_loss_history = []
    val_acc_history = []

    best_val_loss = float("inf")

    for epoch in range(epochs):

        # =========================
        # Training
        # =========================

        model.train()
        epoch_start = time.perf_counter()
        total_train_batches = len(train_loader)
        print(
            f"Epoch [{epoch + 1}/{epochs}] started "
            f"({total_train_batches:,} training batches)",
            flush=True,
        )

        running_loss = 0.0

        for batch_index, (images, labels) in enumerate(train_loader, start=1):

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_index == 1 or batch_index % progress_interval == 0:
                # Synchronize before timing so CUDA's asynchronous work is
                # included in the elapsed time and ETA.
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - epoch_start
                batches_per_second = batch_index / elapsed
                remaining_seconds = (total_train_batches - batch_index) / batches_per_second
                print(
                    f"  Train batch {batch_index:,}/{total_train_batches:,} "
                    f"| loss: {loss.item():.4f} "
                    f"| {batches_per_second:.1f} batches/s "
                    f"| ETA: {remaining_seconds / 60:.1f} min",
                    flush=True,
                )

        avg_train_loss = running_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # =========================
        # Validation
        # =========================

        model.eval()

        running_val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_accuracy = 100 * correct / total

        val_loss_history.append(avg_val_loss)
        val_acc_history.append(val_accuracy)

        # Save Best Model

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.2f}%"
        )

    print("\nTraining Finished!")

    history = {
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "val_acc": val_acc_history,
    }

    return history
