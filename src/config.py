from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    workdir: Path = Path("/media/share/Member/ueki/datasets/")

    doha_name: str = "COVID-19_Radiography_binary_dataset_clean"
    nigeria_name: str = "nigerian_pneumonia_binary_dataset"
    china_name: str = "ZhangLabData_binary_dataset"

    batch_size: int = 128
    epochs: int = 30
    img_size: int = 224
    num_workers: int = 0

    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999

    classes: tuple = ("NORMAL", "PNEUMONIA")
    positive_class: str = "PNEUMONIA"

    output_dir: Path = Path("outputs")