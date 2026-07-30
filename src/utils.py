import argparse
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(
    seed: int = 42,
    deterministic: bool = True,
):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_device(
    require_cuda: bool = True,
):
    """
    require_cuda=True：
        CUDAが使用できなければエラーにする．

    require_cuda=False：
        CUDAがあればCUDA，なければCPUを使用する．
    """
    if require_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available."
            )

        return torch.device("cuda")

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def make_json_serializable(value):
    if isinstance(value, dict):
        return {
            key: make_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_json_serializable(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()

    if isinstance(value, Path):
        return str(value)

    return value


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file.",
    )

    return parser.parse_args()


def resolve_path(
    path_value,
    base_dir=None,
):
    """
    絶対パスならそのまま使用し，
    相対パスならbase_dirを基準に解決する．
    """
    path = Path(path_value).expanduser()

    if path.is_absolute():
        return path.resolve()

    if base_dir is None:
        base_dir = Path.cwd()

    return (
        Path(base_dir) / path
    ).resolve()