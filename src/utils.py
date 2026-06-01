import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(require_cuda: bool = True):
    if require_cuda:
        assert torch.cuda.is_available(), "CUDA is not available."
        return torch.device("cuda")

    return torch.device("cuda")