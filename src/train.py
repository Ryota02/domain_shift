import copy
import torch
from tqdm import tqdm


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, epochs):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


def validate_one_epoch(model, loader, criterion, device, epoch, epochs):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


def fit_source_only(model, train_loader, val_loader, criterion, optimizer, device, epochs, scheduler=None,early_stopping_patience=None):
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        train_result = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            epochs=epochs,
        )

        val_result = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            epochs=epochs,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_result["loss"])
        history["train_accuracy"].append(train_result["accuracy"])
        history["val_loss"].append(val_result["loss"])
        history["val_accuracy"].append(val_result["accuracy"])
        history["lr"].append(current_lr)

        if scheduler is not None:
            scheduler.step(val_result["loss"])

        improved = val_result["loss"] < best_val_loss

        if improved: 
            best_val_loss = val_result["loss"]
            best_val_acc = val_result["accuracy"]
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else: 
            epochs_without_improvement += 1

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_result['loss']:.4f}, "
            f"Train Acc: {train_result['accuracy']:.4f}, "
            f"Val Loss: {val_result['loss']:.4f}, "
            f"Val Acc: {val_result['accuracy']:.4f}, "
            f"LR: {current_lr:.6f}"
        )

        if early_stopping_patience is not None:
            if epochs_without_improvement >= early_stopping_patience:
                print(
                    f"[INFO] Early stopping at epoch {epoch+1}. "
                    f"Best Val Loss: {best_val_loss:.4f}, "
                    f"Best Val Acc: {best_val_acc:.4f}"
                )
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return history, best_val_acc, best_val_loss