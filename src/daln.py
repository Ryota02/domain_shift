from typing import Optional, Any, Tuple
import numpy as np

import torch.nn as nn
from torch.autograd import Function
import torch
import torch.nn.functional as F

class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)

class WarmStartGradientReverseLayer(nn.Module):
    def __init__(
        self, 
            alpha: Optional[float] = 1.0, 
            lo: Optional[float] = 0.0, 
            hi: Optional[float] = 1.,
            max_iters: Optional[int] = 1000., 
            auto_step: Optional[bool] = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        coeff = float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        self.iter_num += 1

class NuclearWassersteinDiscrepancy(nn.Module):
    """
    DALNのNuclear-norm Wasserstein Discrepancy.

    classifier_head:
        feature -> class logits を出す分類器ヘッド。
        DALNではこの分類器をdiscriminatorとして再利用する。
    """
    def __init__(
            self, 
            classifier_head: nn.Module, 
            grl_alpha=1.0, 
            grl_lo=0.0, 
            grl_hi=1.0, 
            grl_max_iters=1000,):
        super(NuclearWassersteinDiscrepancy, self).__init__()
        self.grl = WarmStartGradientReverseLayer(
            alpha=grl_alpha, 
            lo=grl_lo, 
            hi=grl_hi, 
            max_iters=grl_max_iters, 
            auto_step=True)
        self.classifier_head = classifier_head

    @staticmethod
    def n_discrepancy(logits_s: torch.Tensor, logits_t: torch.Tensor) -> torch.Tensor:
        prob_s = F.softmax(logits_s, dim=1)
        prob_t = F.softmax(logits_t, dim=1)
        nuc_s = torch.linalg.matrix_norm(prob_s, ord="nuc") # nuclear norm source 
        nuc_t = torch.linalg.matrix_norm(prob_t, ord="nuc") # nuclear norm target
        loss = -nuc_t / logits_t.shape[0] + nuc_s / logits_s.shape[0] 
        return loss

    def forward(
        self, 
        source_features: torch.Tensor, 
        target_features: torch.Tensor
    ) -> torch.Tensor:
        num_source = source_features.size(0)
        all_features = torch.cat([source_features, target_features], dim=0)
        feature_grl = self.grl(all_features)
        logits = self.classifier_head(feature_grl)
        logits_s = logits[:num_source]
        logits_t = logits[num_source:]

        loss = self.n_discrepancy(logits_s, logits_t)
        return loss

def build_daln_discrepancy(model, cfg):
    grl_cfg = cfg.get("grl", {})

    return NuclearWassersteinDiscrepancy(
        classifier_head=model.classifier_head,
        grl_alpha=grl_cfg.get("alpha", 1.0),
        grl_lo=grl_cfg.get("lo", 0.0),
        grl_hi=grl_cfg.get("hi", 1.0),
        grl_max_iters=grl_cfg.get("max_iters", 1000),
    )