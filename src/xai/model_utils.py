from pathlib import Path

import torch


def load_checkpoint(
    checkpoint_path,
    device,
):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    return checkpoint


def extract_state_dict(
    checkpoint,
):
    if not isinstance(
        checkpoint,
        dict,
    ):
        return checkpoint

    for key in [
        "model_state_dict",
        "state_dict",
        "model",
    ]:
        if (
            key in checkpoint
            and isinstance(
                checkpoint[key],
                dict,
            )
        ):
            return checkpoint[
                key
            ]

    return checkpoint


def remove_module_prefix(
    state_dict,
):
    return {
        (
            key[len("module."):]
            if key.startswith(
                "module."
            )
            else key
        ): value
        for key, value
        in state_dict.items()
    }


def get_prediction_threshold(
    checkpoint,
    cfg,
):
    prediction_cfg = cfg.get(
        "prediction",
        {},
    )

    fallback = float(
        prediction_cfg.get(
            "fallback_threshold",
            0.5,
        )
    )

    use_checkpoint = bool(
        prediction_cfg.get(
            "use_checkpoint_threshold",
            True,
        )
    )

    if (
        use_checkpoint
        and isinstance(
            checkpoint,
            dict,
        )
    ):
        for key in [
            "threshold",
            "best_threshold",
        ]:
            if key in checkpoint:
                return float(
                    checkpoint[key]
                )

    return fallback


@torch.no_grad()
def predict(
    model,
    input_tensor,
    threshold,
):
    logits = model(
        input_tensor
    ).reshape(
        -1
    )

    probability = float(
        torch.sigmoid(
            logits[0]
        ).item()
    )

    predicted_class = int(
        probability
        >= threshold
    )

    return (
        predicted_class,
        probability,
    )