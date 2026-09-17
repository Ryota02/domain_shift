# src/xai/attention_rollout.py

import cv2
import numpy as np
import torch

from pytorch_grad_cam.utils.image import (
    show_cam_on_image,
)


# ============================================================
# Attention capture
# ============================================================

class ViTAttentionCapture:
    """
    torchvision ViTの各EncoderBlockにある
    MultiheadAttentionからattention weightsを取得する．

    torchvision ViTでは通常，
        need_weights=False
    でforwardされるため，
    一時的にTrueへ変更する．
    """

    def __init__(
        self,
        backbone,
    ):
        self.backbone = backbone

        self.attentions = []

        self.original_forwards = []


    def __enter__(
        self,
    ):
        self.attentions = []

        self.original_forwards = []

        encoder_layers = (
            self.backbone
            .encoder
            .layers
        )

        for block in encoder_layers:

            attention_module = (
                block.self_attention
            )

            original_forward = (
                attention_module.forward
            )

            self.original_forwards.append(
                (
                    attention_module,
                    original_forward,
                )
            )

            def make_forward(
                original_forward,
            ):
                def wrapped_forward(
                    *args,
                    **kwargs,
                ):
                    # ----------------------------------------
                    # Attention weightsを取得
                    # ----------------------------------------

                    kwargs[
                        "need_weights"
                    ] = True

                    kwargs[
                        "average_attn_weights"
                    ] = False

                    output, weights = (
                        original_forward(
                            *args,
                            **kwargs,
                        )
                    )

                    if weights is not None:

                        self.attentions.append(
                            weights.detach()
                        )

                    return (
                        output,
                        weights,
                    )

                return wrapped_forward

            attention_module.forward = (
                make_forward(
                    original_forward
                )
            )

        return self


    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        # ----------------------------------------
        # 必ず元のforwardへ戻す
        # ----------------------------------------

        for (attention_module, original_forward) in self.original_forwards:

            attention_module.forward = original_forward


# ============================================================
# Attention head fusion
# ============================================================

def fuse_attention_heads(
    attention,
    method="mean",
):
    """
    attention:
        [B, Heads, Tokens, Tokens]

    return:
        [B, Tokens, Tokens]
    """

    method = str(
        method
    ).lower()

    if method == "mean":

        return attention.mean(dim=1)

    if method == "max":

        return attention.max(dim=1).values

    if method == "min":

        return attention.min(dim=1).values

    raise ValueError(
        f"Unknown head fusion: "
        f"{method}. "
        "Use mean, max, or min."
    )


# ============================================================
# Low-attention filtering
# ============================================================

def discard_low_attention(
    attention,
    discard_ratio,
):
    """
    小さいattentionを除外するオプション．

    attention:
        [B, N, N]

    defaultではdiscard_ratio=0なので何もしない．
    """

    discard_ratio = float(discard_ratio)

    if discard_ratio <= 0:
        return attention

    if discard_ratio >= 1:
        raise ValueError(
            "discard_ratio must be "
            "smaller than 1.0"
        )

    result = attention.clone()

    batch_size = result.size(0)

    n_tokens = result.size(1)

    # ----------------------------------------
    # CLS token関連は残し，
    # patch -> patch部分のみ低attention除去
    # ----------------------------------------

    for batch_index in range(
        batch_size
    ):
        patch_attention = (
            result[
                batch_index,
                1:,
                1:,
            ]
        )

        flat = (patch_attention.reshape(-1))

        n_discard = int(flat.numel() * discard_ratio)

        if n_discard <= 0:
            continue

        threshold = torch.kthvalue(
            flat,
            k=n_discard,
        ).values

        patch_attention[
            patch_attention
            <= threshold
        ] = 0.0

        result[
            batch_index,
            1:,
            1:,
        ] = patch_attention

    return result


# ============================================================
# Attention rollout
# ============================================================

def compute_attention_rollout(
    attentions,
    head_fusion="mean",
    discard_ratio=0.0,
    start_layer=0,
):
    """
    Attention Rollout.

    各Transformer layerについて，

        A_hat = A + I

    としてresidual connectionを加え，
    行方向に正規化した後，

        A_L @ A_(L-1) @ ... @ A_1

    を計算する．

    最後に
        CLS -> patch token
    のattentionを取得する．
    """

    if not attentions:

        raise RuntimeError(
            "No attention matrices "
            "were captured."
        )

    start_layer = int(
        start_layer
    )

    if (start_layer < 0 or start_layer >= len(attentions)):
        raise ValueError(
            f"Invalid start_layer: "
            f"{start_layer}. "
            f"Number of layers: "
            f"{len(attentions)}"
        )

    joint_attention = None

    for attention in attentions[start_layer:]:

        # ----------------------------------------
        # [B,H,N,N]
        # ->
        # [B,N,N]
        # ----------------------------------------

        fused = fuse_attention_heads(
            attention,
            method=head_fusion,
        )

        fused = discard_low_attention(
            fused,
            discard_ratio=(
                discard_ratio
            ),
        )

        # ----------------------------------------
        # Residual connection
        # ----------------------------------------

        n_tokens = fused.size(-1)

        identity = torch.eye(
            n_tokens,
            device=fused.device,
            dtype=fused.dtype,
        ).unsqueeze(
            0
        )

        augmented = (fused + identity)

        # ----------------------------------------
        # Row normalization
        # ----------------------------------------

        augmented = (augmented / augmented.sum(dim=-1, keepdim=True).clamp_min(1e-8))

        # ----------------------------------------
        # Rollout
        # ----------------------------------------

        if joint_attention is None:

            joint_attention = augmented

        else:

            joint_attention = (torch.bmm(augmented, joint_attention))

    # ========================================================
    # CLS -> patch attention
    # ========================================================

    cls_to_patch = (
        joint_attention[
            :,
            0,
            1:,
        ]
    )

    # 今回はbatch size 1
    cls_to_patch = (cls_to_patch[0])

    n_patches = int(cls_to_patch.numel())

    side = int(np.sqrt(n_patches))

    if (side* side != n_patches):
        raise ValueError(
            "Number of patch tokens "
            "cannot be reshaped into "
            f"a square: {n_patches}"
        )

    rollout_map = (
        cls_to_patch
        .reshape(
            side,
            side,
        )
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    # ========================================================
    # Normalize 0-1
    # ========================================================

    minimum = float(rollout_map.min())

    maximum = float(rollout_map.max())

    denominator = (maximum - minimum)

    if denominator > 1e-8:
        rollout_map = (rollout_map - minimum) / denominator

    else:
        rollout_map = np.zeros_like(
            rollout_map,
            dtype=np.float32,
        )

    return rollout_map


# ============================================================
# Generate rollout
# ============================================================

def generate_attention_rollout(
    model,
    input_tensor,
    display_image,
    backbone_name,
    head_fusion="mean",
    discard_ratio=0.0,
    start_layer=0,
):
    """
    torchvision ViT用Attention Rollout．

    return:
        rollout_map
        rollout_visualization
    """

    name = str(
        backbone_name
    ).lower()

    if "vit" not in name:

        raise ValueError(
            "Attention Rollout is "
            "currently implemented only "
            "for torchvision ViT. "
            f"Backbone: {backbone_name}"
        )

    backbone = getattr(
        model,
        "backbone",
        model,
    )

    model.eval()

    # ========================================================
    # Capture attention
    # ========================================================

    with ViTAttentionCapture(
        backbone
    ) as capture:

        with torch.no_grad():
            _ = model(
                input_tensor
            )

    attentions = (
        capture.attentions
    )

    if not attentions:

        raise RuntimeError(
            "No ViT attention weights "
            "were captured. "
            "Check the torchvision ViT "
            "model structure."
        )

    print(
        "[INFO] Rollout attention layers:",
        len(
            attentions
        ),
    )

    # ========================================================
    # Rollout
    # ========================================================

    rollout_map = (
        compute_attention_rollout(
            attentions=attentions,
            head_fusion=head_fusion,
            discard_ratio=(
                discard_ratio
            ),
            start_layer=(
                start_layer
            ),
        )
    )

    # ========================================================
    # 14 x 14 -> model input size
    # ========================================================

    height = display_image.shape[0]

    width = display_image.shape[1]

    rollout_map = cv2.resize(
        rollout_map,
        (width,height,),
        interpolation=(cv2.INTER_CUBIC),
    )

    rollout_map = np.clip(
        rollout_map,
        0.0,
        1.0,
    ).astype(np.float32)

    display_image = np.clip(
        display_image,
        0.0,
        1.0,
    ).astype(np.float32)

    visualization = (
        show_cam_on_image(
            display_image,
            rollout_map,
            use_rgb=True,
        )
    )

    return (
        rollout_map,
        visualization,
    )