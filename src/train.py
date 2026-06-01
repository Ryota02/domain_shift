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


def fit_source_only(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    best_val_acc = 0.0
    best_state_dict = None

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

        history["train_loss"].append(train_result["loss"])
        history["train_accuracy"].append(train_result["accuracy"])
        history["val_loss"].append(val_result["loss"])
        history["val_accuracy"].append(val_result["accuracy"])

        if val_result["accuracy"] > best_val_acc:
            best_val_acc = val_result["accuracy"]
            best_state_dict = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_result['loss']:.4f}, "
            f"Train Acc: {train_result['accuracy']:.4f}, "
            f"Val Loss: {val_result['loss']:.4f}, "
            f"Val Acc: {val_result['accuracy']:.4f}"
        )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return history, best_val_acc