from pathlib import Path
import yaml


def load_config(config_path):
    config_path = Path(config_path)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["workdir"] = Path(cfg["workdir"])
    cfg["output_dir"] = Path(cfg["output_dir"])

    return cfg