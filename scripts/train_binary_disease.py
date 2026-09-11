import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT_DIR = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT_DIR),
)


from src.binary_disease_data import (
    build_binary_disease_datasets,
    build_binary_disease_loaders,
    # calculate_pos_weight,
)

from src.config import load_config

from src.one_vs_rest_model import (
    build_one_vs_rest_model,
)

from src.one_vs_rest_train import (
    fit_one_vs_rest,
)

from src.one_vs_rest_plots import (
    plot_training_history,
    plot_roc_and_pr,
    plot_confusion,
)

from src.utils import (
    get_device,
    make_json_serializable,
    parse_args,
    set_seed,
)


def main():
    args = parse_args()

    cfg = load_config(
        args.config
    )

    seed = cfg["train"].get(
        "seed",
        42,
    )

    set_seed(seed)

    device = get_device(
        require_cuda=cfg[
            "train"
        ].get(
            "require_cuda",
            True,
        )
    )

    output_dir = Path(
        cfg["output_dir"]
    )

    checkpoint_dir = (
        output_dir
        / "checkpoints"
    )

    result_dir = (
        output_dir
        / "results"
    )

    figure_dir = (
        output_dir
        / "figures"
    )

    for directory in [
        checkpoint_dir,
        result_dir,
        figure_dir,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # -------------------------
    # Dataset
    # -------------------------

    datasets_dict = (
        build_binary_disease_datasets(
            cfg
        )
    )

    loaders = (
        build_binary_disease_loaders(
            cfg,
            datasets_dict,
        )
    )

    # -------------------------
    # Model
    # -------------------------

    model = (
        build_one_vs_rest_model(
            cfg
        ).to(device)
    )

    # -------------------------
    # Class imbalance
    # -------------------------

    if cfg.get(
        "loss",
        {},
    ).get(
        "use_pos_weight",
        True,
    ):
        pos_weight = (
            calculate_pos_weight(
                datasets_dict
            )
        )
    else:
        pos_weight = 1.0

    print(
        "[INFO] Device:",
        device,
    )

    print(
        "[INFO] Model:",
        cfg["model"]["backbone"],
    )

    print(
        "[INFO] Classes:",
        datasets_dict[
            "class_names"
        ],
    )

    print(
        "[INFO] pos_weight:",
        pos_weight,
    )

    # -------------------------
    # Training
    # -------------------------

    start_time = time.time()

    result = fit_one_vs_rest(
        model=model,
        loaders=loaders,
        cfg=cfg,
        device=device,
        pos_weight=pos_weight,
    )

    elapsed_time = (
        time.time()
        - start_time
    )

    # -------------------------
    # Checkpoint
    # -------------------------

    best_checkpoint = (
        checkpoint_dir
        / "best_model.pth"
    )

    torch.save(
        {
            "model_state_dict": (
                result[
                    "best_state_dict"
                ]
            ),
            "best_epoch": (
                result[
                    "best_epoch"
                ]
            ),
            "best_score": (
                result[
                    "best_score"
                ]
            ),
            "threshold": (
                result[
                    "threshold"
                ]
            ),
            "class_names": (
                datasets_dict[
                    "class_names"
                ]
            ),
            "class_to_idx": (
                datasets_dict[
                    "class_to_idx"
                ]
            ),
            "backbone": (
                cfg[
                    "model"
                ][
                    "backbone"
                ]
            ),
            "config": (
                make_json_serializable(
                    cfg
                )
            ),
        },
        best_checkpoint,
    )

    # -------------------------
    # Test results
    # -------------------------

    test_result = (
        result["test_result"]
    )

    test_metrics = (
        test_result["metrics"]
    )

    y_true = (
        test_result["y_true"]
    )

    y_probability = (
        test_result[
            "y_probability"
        ]
    )

    threshold = (
        result["threshold"]
    )

    y_pred = (
        np.asarray(
            y_probability
        )
        >= threshold
    ).astype(
        np.int64
    )

    # -------------------------
    # Plots
    # -------------------------

    plot_training_history(
        result["history"],
        output_dir=figure_dir,
    )

    plot_roc_and_pr(
        y_true,
        y_probability,
        figure_dir,
    )

    plot_confusion(
        y_true,
        y_probability,
        threshold,
        datasets_dict["class_names"],
        figure_dir / "confusion_matrix.png",
    )

    # -------------------------
    # JSON
    # -------------------------

    result_log = {
        "classes": (
            datasets_dict[
                "class_names"
            ]
        ),
        "class_to_idx": (
            datasets_dict[
                "class_to_idx"
            ]
        ),
        "model": (
            cfg[
                "model"
            ][
                "backbone"
            ]
        ),
        "best_epoch": (
            result[
                "best_epoch"
            ]
        ),
        "best_score": (
            result[
                "best_score"
            ]
        ),
        "threshold": threshold,
        "pos_weight": (
            pos_weight
        ),
        "validation_metrics": (
            result[
                "validation_result"
            ][
                "metrics"
            ]
        ),
        "test_metrics": (
            test_metrics
        ),
        "training_time_seconds": (
            elapsed_time
        ),
    }

    with (
        result_dir
        / "metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            make_json_serializable(
                result_log
            ),
            file,
            indent=2,
        )

    print("\n[RESULT]")

    for key, value in (
        test_metrics.items()
    ):
        print(
            f"{key}: {value}"
        )

    print(
        "[INFO] Checkpoint:",
        best_checkpoint,
    )


if __name__ == "__main__":
    main()