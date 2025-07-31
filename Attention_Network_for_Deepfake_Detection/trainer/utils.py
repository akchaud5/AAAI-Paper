# trainer/utils.py
import os
import sys
import time
from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F

try:
    # prefer sklearn's reference implementation if present
    from sklearn.metrics import roc_auc_score
    _HAS_SK = True
except Exception:
    _HAS_SK = False


LEGAL_METRIC = ["Acc", "AUC", "LogLoss"]


# -------------------------------------------------------------------------
# Misc helpers
# -------------------------------------------------------------------------
def exp_recons_loss(recons, x):
    """
    Reconstruction loss used in some experiments.
    Keeps original behavior; just adds the missing import F above.
    """
    x, y = x
    loss = torch.tensor(0.0, device=y.device)
    real_index = torch.where(1 - y)[0]
    for r in recons:
        if real_index.numel() > 0:
            real_x = torch.index_select(x, dim=0, index=real_index)
            real_rec = torch.index_select(r, dim=0, index=real_index)
            real_rec = F.interpolate(
                real_rec, size=x.shape[-2:], mode="bilinear", align_corners=True
            )
            loss += torch.mean(torch.abs(real_rec - real_x))
    return loss


def setup_for_distributed(local_rank: int):
    """
    Initialize default NCCL group; no-op if you don't use DDP.
    """
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.distributed.barrier()


def cleanup():
    """Destroy the process group if initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


def center_print(content: str, around: str = "*", repeat_around: int = 10):
    """Print centered text with decorations."""
    s = around * repeat_around
    print(f"{s} {content} {s}")


def reduce_tensor(t: torch.Tensor) -> torch.Tensor:
    """Average a tensor across processes if DDP is initialized; else return t."""
    if not dist.is_initialized():
        return t
    rt = t.clone()
    dist.all_reduce(rt)
    rt /= float(dist.get_world_size())
    return rt


def tensor2image(tensor: torch.Tensor) -> np.ndarray:
    """Convert [C,H,W] tensor to [H,W,C] uint8-like float image in [0,1]."""
    image = tensor.permute(1, 2, 0).detach().cpu().numpy()
    vmin, vmax = float(image.min()), float(image.max())
    if vmax == vmin:
        return np.zeros_like(image)
    return (image - vmin) / (vmax - vmin)


def state_dict(state_dict):
    """Remove 'module.' prefix from DDP state dictionaries."""
    weights = OrderedDict()
    for k, v in state_dict.items():
        weights[k.replace("module.", "")] = v
    return weights


def Timer():
    """
    Simple wall‑clock timer. Use as:
        t = Timer()
        ... do work ...
        print(t())     # seconds elapsed
    """
    from timeit import default_timer as timer
    start = timer()

    def elapsed():
        return timer() - start

    return elapsed


# -------------------------------------------------------------------------
# Running meters
# -------------------------------------------------------------------------
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n: int = 1):
        self.val = float(val)
        self.sum += float(val) * int(n)
        self.count += int(n)
        self.avg = self.sum / max(1, self.count)


class AccMeter:
    """
    Accumulates frame‑level accuracy (top‑1).
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        if predictions.dim() > 1:  # logits or probabilities
            predictions = torch.argmax(predictions, dim=1)
        if targets.dim() > 1:
            targets = torch.argmax(targets, dim=1)
        self.correct += int((predictions == targets).sum().item())
        self.total += int(targets.numel())

    def mean_acc(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


class AUCMeter:
    """
    Numerically correct AUC meter for **frame‑level** AUC:

      • Stores ALL frame probabilities and labels.
      • Computes a single ROC AUC when asked (not per‑batch average).
      • Uses sklearn if available; otherwise falls back to a stable manual ROC integration.
      • Returns NaN if only one class present (AUC undefined).
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.scores: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        predictions: [B,2] logits/probabilities (softmax over 2) or [B] binary logits
        targets:     [B] in {0,1}
        """
        with torch.no_grad():
            if predictions.ndim == 2 and predictions.size(1) == 2:
                # Convert logits to P(fake) if needed
                probs = torch.softmax(predictions, dim=1)[:, 1]
            elif predictions.ndim == 1:
                probs = torch.sigmoid(predictions)
            else:
                raise ValueError(f"Unsupported predictions shape for AUCMeter: {tuple(predictions.shape)}")
            self.scores.append(probs.detach().cpu())
            self.targets.append(targets.detach().cpu().int())

    def compute_auc(self) -> float:
        if not self.scores:
            return float("nan")
        s = torch.cat(self.scores).numpy()
        t = torch.cat(self.targets).numpy().astype(np.int64)
        if np.unique(t).size < 2:
            return float("nan")

        if _HAS_SK:
            try:
                return float(roc_auc_score(t, s))
            except Exception:
                pass  # fallback to manual

        # Manual ROC (stable and close to sklearn)
        order = np.argsort(-s, kind="mergesort")
        t_sorted = t[order]
        P = t_sorted.sum()
        N = len(t_sorted) - P
        if P == 0 or N == 0:
            return float("nan")
        tps = np.cumsum(t_sorted)          # true positives at each threshold
        fps = np.cumsum(1 - t_sorted)      # false positives at each threshold
        tpr = tps / P
        fpr = fps / N
        # Add endpoints
        tpr = np.concatenate(([0.0], tpr, [1.0]))
        fpr = np.concatenate(([0.0], fpr, [1.0]))
        return float(np.trapz(tpr, fpr))


class Logger:
    """
    Duplicates stdout into a logfile.
    """
    def __init__(self, filename: str):
        self.terminal = sys.stdout
        self.log = open(filename, "a", buffering=1)  # line-buffered

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()
