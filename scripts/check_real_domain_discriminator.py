import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import torch
import torch.nn as nn

from src.config import load_config
from src.utils import set_seed, get_device
from src.data import build_datasets, build_loaders
from src.ada_model import SupervisedADAModel, DomainDiscriminator


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to ADA config yaml file.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to source_pretrained_model.pth or best_model.pth.",
    )
    
    return parser.parse_args()


def load_model(cfg, checkpoint_path, device):
    model = SupervisedADAModel(
        num_classes=len(cfg["classes"]),
        feature_dim=cfg["model"].get("feature_dim", 1920),
        pretrained=False,
        grl_lambda=cfg["ada"].get("grl_lambda", 1.0),
    )

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    model = model.to(device)

    return model


def reset_domain_discriminator(model, cfg, device):
    """
    ADA中に学習されたDomain Discriminatorは使わず，
    新しいDomain Discriminatorを初期化する。
    """

    feature_dim = cfg["model"].get("feature_dim", 1920)

    model.domain_discriminator = DomainDiscriminator(
        feature_dim=feature_dim,
    ).to(device)

    return model


def freeze_feature_extractor_and_classifier(model):
    """
    Feature ExtractorとClassifierは固定する。
    Domain Discriminatorだけを学習する。
    """

    for p in model.feature_extractor.parameters():
        p.requires_grad = False

    for p in model.classifier.parameters():
        p.requires_grad = False

    for p in model.domain_discriminator.parameters():
        p.requires_grad = True

    return model


def train_domain_discriminator_one_epoch(
    model,
    source_loader,
    target_loader,
    criterion,
    optimizer,
    device,
):
    model.eval()
    model.domain_discriminator.train()

    total_loss = 0.0
    correct = 0
    total = 0

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)

    num_batches = min(len(source_loader), len(target_loader))

    for _ in range(num_batches):
        try:
            source_images, _ = next(source_iter)
        except StopIteration:
            source_iter = iter(source_loader)
            source_images, _ = next(source_iter)

        try:
            target_images, _ = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            target_images, _ = next(target_iter)

        source_images = source_images.to(device)
        target_images = target_images.to(device)

        source_domain_labels = torch.zeros(
            source_images.size(0),
            dtype=torch.long,
            device=device,
        )

        target_domain_labels = torch.ones(
            target_images.size(0),
            dtype=torch.long,
            device=device,
        )

        images = torch.cat([source_images, target_images], dim=0)
        domain_labels = torch.cat(
            [source_domain_labels, target_domain_labels],
            dim=0,
        )

        with torch.no_grad():
            _, features = model.forward_class(images)

        # GRLは使わない
        domain_logits = model.domain_discriminator(features.detach())

        loss = criterion(domain_logits, domain_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = domain_logits.argmax(dim=1)

        total_loss += loss.item() * domain_labels.size(0)
        correct += (preds == domain_labels).sum().item()
        total += domain_labels.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


def evaluate_real_domain_discriminator(
    model,
    source_loader,
    target_loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    source_correct = 0
    source_total = 0

    target_correct = 0
    target_total = 0

    pred_source_count = 0
    pred_target_count = 0

    y_true_domain = []
    y_pred_domain = []
    y_proba_domain = []

    with torch.no_grad():
        # -------------------------
        # Source domain evaluation
        # -------------------------
        for images, _ in source_loader:
            images = images.to(device)

            domain_labels = torch.zeros(
                images.size(0),
                dtype=torch.long,
                device=device,
            )

            _, features = model.forward_class(images)

            # 評価時もGRLは使わない
            domain_logits = model.domain_discriminator(features)

            loss = criterion(domain_logits, domain_labels)

            domain_probs = torch.softmax(domain_logits, dim=1)
            domain_preds = domain_probs.argmax(dim=1)

            total_loss += loss.item() * domain_labels.size(0)

            source_correct += (domain_preds == domain_labels).sum().item()
            source_total += domain_labels.size(0)

            correct += (domain_preds == domain_labels).sum().item()
            total += domain_labels.size(0)

            pred_source_count += (domain_preds == 0).sum().item()
            pred_target_count += (domain_preds == 1).sum().item()

            y_true_domain.append(domain_labels.cpu())
            y_pred_domain.append(domain_preds.cpu())
            y_proba_domain.append(domain_probs.cpu())

        # -------------------------
        # Target domain evaluation
        # -------------------------
        for images, _ in target_loader:
            images = images.to(device)

            domain_labels = torch.ones(
                images.size(0),
                dtype=torch.long,
                device=device,
            )

            _, features = model.forward_class(images)

            # 評価時もGRLは使わない
            domain_logits = model.domain_discriminator(features)

            loss = criterion(domain_logits, domain_labels)

            domain_probs = torch.softmax(domain_logits, dim=1)
            domain_preds = domain_probs.argmax(dim=1)

            total_loss += loss.item() * domain_labels.size(0)

            target_correct += (domain_preds == domain_labels).sum().item()
            target_total += domain_labels.size(0)

            correct += (domain_preds == domain_labels).sum().item()
            total += domain_labels.size(0)

            pred_source_count += (domain_preds == 0).sum().item()
            pred_target_count += (domain_preds == 1).sum().item()

            y_true_domain.append(domain_labels.cpu())
            y_pred_domain.append(domain_preds.cpu())
            y_proba_domain.append(domain_probs.cpu())

    y_true_domain = torch.cat(y_true_domain).numpy()
    y_pred_domain = torch.cat(y_pred_domain).numpy()
    y_proba_domain = torch.cat(y_proba_domain).numpy()

    return {
        "loss": total_loss / total,
        "domain_acc": correct / total,
        "source_domain_acc": source_correct / source_total,
        "target_domain_acc": target_correct / target_total,
        "source_count": source_total,
        "target_count": target_total,
        "pred_source_count": pred_source_count,
        "pred_target_count": pred_target_count,
        "pred_source_ratio": pred_source_count / total,
        "pred_target_ratio": pred_target_count / total,
        "y_true_domain": y_true_domain.tolist(),
        "y_pred_domain": y_pred_domain.tolist(),
        "y_proba_domain": y_proba_domain.tolist(),
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)

    epochs = 20
    lr = 1e-4

    set_seed(42)

    device = get_device(require_cuda=True)

    output_dir = cfg["output_dir"]
    result_dir = output_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] experiment:", cfg["experiment_name"])
    print("[INFO] checkpoint:", args.checkpoint)
    print("[INFO] device:", device)

    datasets_dict = build_datasets(cfg)
    loaders = build_loaders(cfg, datasets_dict)

    model = load_model(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    model = reset_domain_discriminator(
        model=model,
        cfg=cfg,
        device=device,
    )

    model = freeze_feature_extractor_and_classifier(model)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.domain_discriminator.parameters(),
        lr=lr,
    )

    history = {
        "train_loss": [],
        "train_domain_acc": [],
        "eval_loss": [],
        "eval_domain_acc": [],
        "eval_source_domain_acc": [],
        "eval_target_domain_acc": [],
        "pred_source_ratio": [],
        "pred_target_ratio": [],
    }

    print("[INFO] Start training new Domain Discriminator")
    print("[INFO] Source domain label = 0")
    print("[INFO] Target domain label = 1")

    for epoch in range(1, epochs + 1):
        train_result = train_domain_discriminator_one_epoch(
            model=model,
            source_loader=loaders["source_train"],
            target_loader=loaders["target_adapt"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        eval_result = evaluate_real_domain_discriminator(
            model=model,
            source_loader=loaders["source_val"],
            target_loader=loaders["target_test"],
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(train_result["loss"])
        history["train_domain_acc"].append(train_result["accuracy"])
        history["eval_loss"].append(eval_result["loss"])
        history["eval_domain_acc"].append(eval_result["domain_acc"])
        history["eval_source_domain_acc"].append(eval_result["source_domain_acc"])
        history["eval_target_domain_acc"].append(eval_result["target_domain_acc"])
        history["pred_source_ratio"].append(eval_result["pred_source_ratio"])
        history["pred_target_ratio"].append(eval_result["pred_target_ratio"])

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"TrainLoss: {train_result['loss']:.4f} "
            f"TrainDomainAcc: {train_result['accuracy']:.4f} "
            f"EvalDomainAcc: {eval_result['domain_acc']:.4f} "
            f"SourceDomainAcc: {eval_result['source_domain_acc']:.4f} "
            f"TargetDomainAcc: {eval_result['target_domain_acc']:.4f} "
            f"PredSourceRatio: {eval_result['pred_source_ratio']:.4f} "
            f"PredTargetRatio: {eval_result['pred_target_ratio']:.4f}"
        )

    final_result = evaluate_real_domain_discriminator(
        model=model,
        source_loader=loaders["source_val"],
        target_loader=loaders["target_test"],
        criterion=criterion,
        device=device,
    )


    print("[RESULT]")
    print(f"Domain Acc: {final_result['domain_acc']:.4f}")
    print(f"Source Domain Acc: {final_result['source_domain_acc']:.4f}")
    print(f"Target Domain Acc: {final_result['target_domain_acc']:.4f}")
    print(f"Pred Source Ratio: {final_result['pred_source_ratio']:.4f}")
    print(f"Pred Target Ratio: {final_result['pred_target_ratio']:.4f}")


if __name__ == "__main__":
    main()