import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.binary_disease_data import (
    build_binary_disease_datasets,
)

from src.config import load_config

from src.one_vs_rest_model import (
    build_one_vs_rest_model,
)

from src.utils import (
    get_device,
    set_seed,
)

from src.xai.cam import (
    generate_cam,
)

from src.xai.image_utils import (
    denormalize_tensor,
    load_original_image,
    resize_heatmap,
)

from src.xai.lime_explainer import (
    generate_lime,
)

from src.xai.model_utils import (
    extract_state_dict,
    get_prediction_threshold,
    load_checkpoint,
    predict,
    remove_module_prefix,
)

from src.xai.plotting import (
    save_xai_figure,
)

from src.xai.shap_explainer import (
    build_background,
    generate_shap,
)

from src.xai.attention_rollout import (
    generate_attention_rollout, 
)


def load_yaml(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    args = parser.parse_args()

    # XAI config
    xai_cfg = load_yaml(args.config)

    # Training config
    cfg = load_config(
        xai_cfg["training_config"]
    )

    seed = int(
        cfg.get("train", {}).get(
            "seed",
            42,
        )
    )

    set_seed(seed)

    device = get_device(
        cfg.get("train", {}).get(
            "require_cuda",
            True,
        )
    )
    valid_mode = ["original", "lung_only", "moment_standardized"]
    mode = cfg.get("preprocessing", {}).get("mode", {})

    if mode not in valid_mode: 
        raise ValueError ("This modes is not supported: ", mode)

    # ============================================
    # Dataset
    # ============================================

    datasets = (build_binary_disease_datasets(cfg, seed))

    data_cfg = xai_cfg["data"]

    split = data_cfg.get(
        "split",
        "test",
    )

    dataset = datasets[split]

    indices_cfg = data_cfg.get("indices", [0])

    if indices_cfg == "all":
        indices = list(
            range(len(dataset))
        )

    else:
        indices = [
            int(x)
            for x in indices_cfg
        ]

    # ============================================
    # Model
    # ============================================

    model = build_one_vs_rest_model(cfg).to(device)
    
    checkpoint_dir = Path(cfg["output_dir"])
    seed_name = "seed" + str(seed)
    
    checkpoint_path = checkpoint_dir / mode / seed_name / "checkpoints" / "best_model.pth"
    checkpoint = load_checkpoint(
        checkpoint_path, 
        device,
    )

    state_dict = (
        remove_module_prefix(
            extract_state_dict(
                checkpoint
            )
        )
    )

    model.load_state_dict(state_dict)

    model.eval()

    threshold = (
        get_prediction_threshold(
            checkpoint,
            xai_cfg,
        )
    )

    backbone_name = (
        cfg["model"]["backbone"]
    )

    # ============================================
    # XAI configuration
    # ============================================

    methods = xai_cfg["methods"]

    shap_background = None

    if methods["shap"].get("enabled", False):
        shap_background = (
            build_background(
                datasets["val"],
                device,
                methods["shap"].get("background_samples",8),
            )
        )

    output_dir = Path(xai_cfg["output_dir"])
    output_dir = output_dir / mode / seed_name
    
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ============================================
    # Visualization
    # ============================================

    for index in indices:
        image_tensor, label = dataset[index]

        input_tensor = (
            image_tensor
            .unsqueeze(0)
            .to(
                device
            )
        )

        true_class = int(label.item())

        (predicted_class, probability) = predict(
            model,
            input_tensor,
            threshold,
        )

        target_mode = data_cfg.get(
            "target",
            "predicted",
        )

        if target_mode == "predicted":
            target_class = predicted_class
                
        elif target_mode == "true":
            target_class = true_class

        elif target_mode == "pneumonia":
            target_class = 1

        else:
            target_class = 0

        sample = dataset.samples[index]

        image_path = Path(sample["image"])

        model_input = denormalize_tensor(image_tensor)
            
        original = load_original_image(image_path)

        gradcam_image = None
        gradcam_pp_image = None
        rollout_map = None
        rollout_image = None
        lime_map = None
        shap_map = None

        # ========================================
        # Grad-CAM
        # ========================================

        if methods["gradcam"].get("enabled", False):
            _, gradcam_image = (
                generate_cam(
                    model,
                    input_tensor,
                    model_input,
                    target_class,
                    backbone_name,
                    "gradcam",
                )
            )

        # ========================================
        # Grad-CAM++
        # ========================================

        if methods["gradcam_plus_plus"].get("enabled",False):
            _, gradcam_pp_image = (
                generate_cam(
                    model,
                    input_tensor,
                    model_input,
                    target_class,
                    backbone_name,
                    "gradcam++",
                )
            )
        # ========================================
        # Attention Rollout
        # ========================================
        
        rollout_cfg = methods.get("rollout", {})
        
        if rollout_cfg.get("enabled", False):
            print("[INFO] Attention Rollout...")
            
            rollout_map, rollout_image = generate_attention_rollout(
                model=model,
                input_tensor=input_tensor,
                display_image=model_input,
                backbone_name=backbone_name,
                head_fusion=(rollout_cfg.get("head_fusion","mean")),
                discard_ratio=float(rollout_cfg.get("discard_ratio", 0.0)),
                start_layer=int(rollout_cfg.get("start_layer", 0)),
            )

        # ========================================
        # LIME
        # ========================================

        lime_cfg = methods["lime"]

        if lime_cfg.get("enabled", False):
            lime_map = generate_lime(
                model=model,
                image=model_input,
                target_class=target_class,
                device=device,
                num_samples=lime_cfg.get("num_samples", 1000),
                num_features=lime_cfg.get("num_features",10),
                batch_size=lime_cfg.get("batch_size", 32) 
            )

        # ========================================
        # SHAP
        # ========================================

        shap_cfg = methods["shap"]

        if shap_cfg.get("enabled", False,):
            shap_map = generate_shap(
                model=model,
                input_tensor=input_tensor,
                background=shap_background,
                target_class=target_class,
                nsamples=shap_cfg.get("nsamples", 100)
            )

            shap_map = resize_heatmap(
                shap_map,
                model_input.shape[:2]
            )

        # ========================================
        # Save
        # ========================================

        class_names = datasets["class_names"]

        title = (
            f"{image_path.name} | "
            f"True="
            f"{class_names[true_class]} | "
            f"Pred="
            f"{class_names[predicted_class]} | "
            f"P(Pneumonia)="
            f"{probability:.3f}"
        )

        output_path = (
            output_dir
            / (
                f"{index:04d}_"
                f"{image_path.stem}"
                f"_xai.png"
            )
        )

        save_xai_figure(
            original_image=original,
            model_input=model_input,
            gradcam_image=gradcam_image, 
            gradcam_pp_image=gradcam_pp_image, 
            rollout_image=rollout_image,
            lime_heatmap=lime_map,
            shap_heatmap=shap_map,
            output_path=output_path, 
            title=title,
            dpi=xai_cfg.get("save", {}).get("dpi", 200),
        )

        print("[SAVE]", output_path)


if __name__ == "__main__":
    main()