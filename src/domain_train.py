import copy

from src.domain_evaluate import evaluate_domain_classifier


def train_domain_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, domain_labels, _ in loader:
        images = images.to(device)
        domain_labels = domain_labels.to(device)

        logits = model(images)
        loss = criterion(logits, domain_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)

        total_loss += loss.item() * domain_labels.size(0)
        correct += (preds == domain_labels).sum().item()
        total += domain_labels.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


def fit_domain_classifier(
    model,
    train_loader,
    select_loader,
    criterion,
    optimizer,
    device,
    epochs,
    num_domains,
    domains,
    select_name="Val",
):
    history = {
        "train_loss": [],
        "train_domain_acc": [],
        "select_loss": [],
        "select_domain_acc": [],
        "select_macro_domain_acc": [],
    }

    best_select_acc = -1.0
    best_state_dict = None

    for epoch in range(1, epochs + 1):
        train_result = train_domain_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        history["train_loss"].append(train_result["loss"])
        history["train_domain_acc"].append(train_result["accuracy"])

        if select_loader is not None:
            select_result = evaluate_domain_classifier(
                model=model,
                loader=select_loader,
                criterion=criterion,
                device=device,
                num_domains=num_domains,
            )

            history["select_loss"].append(select_result["loss"])
            history["select_domain_acc"].append(select_result["domain_acc"])
            history["select_macro_domain_acc"].append(select_result["macro_domain_acc"])

            by_class_text = " ".join([
                f"{domains[d]}Acc={select_result['domain_acc_by_class'][d]:.4f}"
                for d in range(num_domains)
                if select_result["domain_acc_by_class"][d] is not None
            ])

            pred_ratio_text = " ".join([
                f"Pred{domains[d]}={select_result['pred_ratio'][d]:.4f}"
                for d in range(num_domains)
            ])

            print(
                f"Epoch [{epoch}/{epochs}] "
                f"TrainLoss={train_result['loss']:.4f} "
                f"TrainAcc={train_result['accuracy']:.4f} "
                f"{select_name}Loss={select_result['loss']:.4f} "
                f"{select_name}Acc={select_result['domain_acc']:.4f} "
                f"{select_name}MacroAcc={select_result['macro_domain_acc']:.4f} "
                f"{by_class_text} "
                f"{pred_ratio_text}"
            )

            if select_result["domain_acc"] > best_select_acc:
                best_select_acc = select_result["domain_acc"]
                best_state_dict = copy.deepcopy(model.state_dict())

        else:
            print(
                f"Epoch [{epoch}/{epochs}] "
                f"TrainLoss={train_result['loss']:.4f} "
                f"TrainAcc={train_result['accuracy']:.4f}"
            )

    if best_state_dict is None:
        best_state_dict = copy.deepcopy(model.state_dict())
        best_select_acc = None

    return history, best_state_dict, best_select_acc