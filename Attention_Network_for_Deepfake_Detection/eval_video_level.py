#!/usr/bin/env python3
"""
CelebDF video-level evaluation

- Loads your trained Recce model from a checkpoint
- Runs batched inference on the validation/test split (frames)
- Aggregates frame predictions to one score per video
- Reports frame-level and video-level Acc/AUC (plus BalAcc, F1)
- (Optional) caps frames/video and picks them uniformly (strided)
  to match the trainer's sampling policy.

USAGE
------
(VAL) Find τ* on validation (e.g., maximize Balanced Accuracy)
  python3 eval_video_level.py \
    --yaml config/dataset/CelebDF.yml --branch val_cfg \
    --ckpt ./checkpoints_celebclean_k120/best_model_ema.pt \
    --device cuda:0 --batch 256 \
    --max-per-video 120 --strided \
    --search-metric balacc \
    --save-csv runs/val_video_scores.csv

(TEST) Apply the same τ* on test
  python3 eval_video_level.py \
    --yaml config/dataset/CelebDF.yml --branch test_cfg \
    --ckpt ./checkpoints_celebclean_k120/best_model_ema.pt \
    --device cuda:0 --batch 256 \
    --max-per-video 120 --strided \
    --apply-threshold 0.0476 \
    --save-csv runs/test_video_scores.csv
"""

import os
import re
import sys
import glob
import argparse
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# -------------------------
# Helpers
# -------------------------

def vid_from_name(path_or_name: str) -> str:
    """
    Extract video id from filename:
      id0_id16_0000.jpg -> id0_id16
      id5_0123.png      -> id5
      00001.jpg         -> 00001
      fallback: token before first underscore
    """
    base = os.path.splitext(os.path.basename(path_or_name))[0]
    for pat in (r'^(id\d+_id\d+)', r'^(id\d+)', r'^(\d+)', r'^([^_]+)'):
        m = re.match(pat, base)
        if m:
            return m.group(1)
    return re.sub(r'_\d+$', '', base)


def build_transform(size=(299, 299)):
    from torchvision import transforms as T
    return T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def iter_split_images(root: str):
    """Yield (path, label) tuples from a split root folder."""
    classes = [("Celeb-real", 0), ("Celeb-synthesis", 1)]
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    for cname, label in classes:
        cdir = os.path.join(root, cname)
        if not os.path.isdir(cdir):
            continue
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(cdir, ext)))
        # deterministic order
        files.sort()
        for p in files:
            yield p, label


def _unwrap_state_dict(sd: dict) -> dict:
    """Unwrap common containers to get the raw tensor dict."""
    if "model_state" in sd:        # format used by your trainer
        return sd["model_state"]
    if "state_dict" in sd:         # common torch convention
        return sd["state_dict"]
    return sd


def _clean_state_dict(raw_sd: dict) -> dict:
    """
    Normalize parameter keys to match the model definition:
    - remove DDP prefixes: 'module.'
    - remove torch.compile wrapper: '_orig_mod.'
    - strip generic wrappers like 'model.' or 'net.' if present
    """
    sd = _unwrap_state_dict(raw_sd)

    def strip_prefix(k: str, prefix: str) -> str:
        return k[len(prefix):] if k.startswith(prefix) else k

    cleaned = {}
    for k, v in sd.items():
        k = strip_prefix(k, "module.")
        if k.startswith("_orig_mod."):
            k = k.replace("_orig_mod.", "", 1)
        k = strip_prefix(k, "model.")
        k = strip_prefix(k, "net.")
        cleaned[k] = v
    return cleaned


def load_model(ckpt_path: str, device: torch.device):
    """Instantiate Recce(num_classes=2), load weights, move to device."""
    # ensure repo root on sys.path
    sys.path.append(os.path.dirname(__file__))

    from model.network import Recce

    print(f"[eval] Loading checkpoint: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu")
    state = _clean_state_dict(raw)

    model = Recce(num_classes=2)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[eval] Missing: {len(missing)} | Unexpected: {len(unexpected)}")
    if missing:
        print("       (first 10 missing):", missing[:10])
    if unexpected:
        print("       (first 10 unexpected):", unexpected[:10])

    model.to(device).eval()

    # H100-safe perf knobs (do not change behavior beyond TF32 tolerance)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    return model


def cap_frames_per_video(data_pairs, k: int, strided: bool):
    """
    data_pairs: list of (path, label)
    Returns a new list capped to k frames per (video, label) key.
    If strided=True -> pick uniformly spaced indices; else take first k.
    """
    buckets = defaultdict(list)
    for p, y in data_pairs:
        buckets[(vid_from_name(p), y)].append(p)

    capped = []
    for (vid, y), flist in buckets.items():
        flist.sort()  # temporal-ish order
        if len(flist) <= k:
            capped.extend((p, y) for p in flist)
        else:
            if strided:
                idx = np.linspace(0, len(flist) - 1, k, dtype=int)
                pick = [flist[i] for i in idx]
            else:
                pick = flist[:k]
            capped.extend((p, y) for p in pick)
    return capped


def batched_infer(model, paths, device, batch_size=256, channels_last=True, bf16=False):
    """Return per-frame probabilities P(fake) for each path."""
    tfm = build_transform()
    N = len(paths)
    probs = np.zeros(N, dtype=np.float32)

    def load_batch(bpaths):
        ims = []
        for p in bpaths:
            with Image.open(p) as im:
                im = im.convert("RGB")
                ims.append(tfm(im))
        x = torch.stack(ims).to(device, non_blocking=True)
        if channels_last and device.type == "cuda":
            x = x.to(memory_format=torch.channels_last)
        return x

    use_autocast = bf16 and device.type == "cuda"
    dtype = torch.bfloat16 if use_autocast else None

    for i in range(0, N, batch_size):
        bpaths = paths[i:i+batch_size]
        x = load_batch(bpaths)
        with torch.no_grad():
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    logits = model(x)
            else:
                logits = model(x)
            p_fake = torch.softmax(logits, dim=1)[:, 1].float().cpu().numpy()
            probs[i:i+len(bpaths)] = p_fake
        if (i // batch_size) % 50 == 0:
            print(f"[eval] processed {i+len(bpaths)}/{N}")
    return probs


def aggregate_by_video(paths, probs, labels):
    """Average frame probabilities by video id and return arrays + map."""
    by_vid = defaultdict(list)
    vid_label = {}
    for p, y, pf in zip(paths, labels, probs):
        v = vid_from_name(p)
        by_vid[v].append(pf)
        vid_label[v] = y   # assume consistent label per video

    vids = list(by_vid.keys())
    y_true = np.array([vid_label[v] for v in vids])
    y_score = np.array([np.mean(by_vid[v]) for v in vids])
    y_pred = (y_score >= 0.5).astype(int)
    return vids, y_true, y_score, y_pred


def _metric_for_pred(y_true, y_pred, metric="balacc"):
    metric = str(metric).lower()
    if metric == "balacc":
        return balanced_accuracy_score(y_true, y_pred)
    if metric == "f1":
        return f1_score(y_true, y_pred, zero_division=0)
    if metric == "acc":
        return accuracy_score(y_true, y_pred)
    if metric == "youden":
        # Youden's J = TPR - FPR
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        tpr = tp / (tp + fn + 1e-12)
        fpr = fp / (fp + tn + 1e-12)
        return tpr - fpr
    raise ValueError(f"Unknown metric '{metric}'")


def best_threshold(scores, labels, metric="balacc", steps=400):
    """
    Return threshold τ* that maximizes the chosen metric on (scores, labels).
    """
    if len(scores) == 0:
        return 0.5, 0.0
    cand = np.linspace(0.0, 1.0, steps)
    best_t, best_val = 0.5, -1e9
    for t in cand:
        pred = (scores >= t).astype(int)
        val = _metric_for_pred(labels, pred, metric=metric)
        if val > best_val:
            best_val, best_t = val, t
    return float(best_t), float(best_val)


def save_video_scores_csv(path_csv, vids, y_true, y_score):
    import csv
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    with open(path_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "label", "prob_fake"])
        for v, y, s in zip(vids, y_true, y_score):
            w.writerow([v, int(y), float(s)])
    print(f"[eval] saved per-video scores to: {path_csv}")


# -------------------------
# YAML helper (optional)
# -------------------------
def _get_branch_cfg_from_yaml(yaml_path: str, branch: str):
    import yaml
    with open(yaml_path, "r") as f:
        data_cfg = yaml.safe_load(f)
    if branch not in data_cfg:
        keys = ", ".join(data_cfg.keys())
        raise KeyError(
            f"Split '{branch}' not found in YAML '{yaml_path}'. "
            f"Available top-level keys: {keys}"
        )
    return data_cfg[branch]


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser("CelebDF video-level evaluation")
    # Either pass a root or a YAML+branch
    ap.add_argument("--val_root", help="e.g., ./celebdb_splits/{val|test}")
    ap.add_argument("--yaml", help="config/dataset/CelebDF.yml")
    ap.add_argument("--branch", default="val_cfg", help="branch in YAML (val_cfg|test_cfg)")
    ap.add_argument("--ckpt", required=True, help="e.g., ./checkpoints/best_model(.pt|_ema.pt)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--no_channels_last", action="store_true")
    ap.add_argument("--bf16", action="store_true", help="Faster eval on H100 (minor numeric diffs)")
    ap.add_argument("--max-per-video", type=int, default=0, help="cap frames/video to mimic train/val")
    ap.add_argument("--strided", action="store_true", help="use uniform strided selection when capping")
    # operating point control
    ap.add_argument("--search-metric", default="balacc",
                    choices=["balacc", "f1", "acc", "youden"],
                    help="metric to optimize when finding threshold on this split")
    ap.add_argument("--apply-threshold", type=float, default=None,
                    help="if given, do NOT search; evaluate at this fixed τ (use τ* from VAL)")
    ap.add_argument("--save-csv", default="",
                    help="optional path to save per-video scores (CSV)")
    args = ap.parse_args()

    # Resolve root from YAML if needed
    if not args.val_root:
        if not args.yaml:
            ap.error("Please provide either --val_root OR both --yaml and --branch")
        branch_cfg = _get_branch_cfg_from_yaml(args.yaml, args.branch)
        root = branch_cfg["root"]
        # allow per-video cap from yaml if present
        if args.max_per_video == 0:
            k_yaml = int(branch_cfg.get("max_frames_per_video", 0) or 0)
            args.max_per_video = k_yaml
    else:
        root = args.val_root

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1) Load model
    model = load_model(args.ckpt, device)

    # 2) Collect frames (VAL or TEST)
    data = list(iter_split_images(root))
    if not data:
        print(f"[eval] No images found under {root}")
        return

    # Optional cap/strided to mirror train/val sampling
    if args.max_per_video and args.max_per_video > 0:
        data = cap_frames_per_video(data, k=int(args.max_per_video), strided=args.strided)
        max_vid_str = str(args.max_per_video)
    else:
        max_vid_str = "no-cap"

    paths, labels = zip(*data)
    labels = np.array(labels)

    # 3) Frame predictions
    probs = batched_infer(
        model,
        list(paths),
        device,
        batch_size=args.batch,
        channels_last=not args.no_channels_last,
        bf16=args.bf16,
    )

    # 4) Frame-level metrics at fixed 0.5 (reference)
    frame_pred = (probs >= 0.5).astype(int)
    frame_acc  = accuracy_score(labels, frame_pred)
    frame_bal  = balanced_accuracy_score(labels, frame_pred)
    frame_f1   = f1_score(labels, frame_pred, zero_division=0)
    try:
        frame_auc = roc_auc_score(labels, probs)
    except ValueError:
        frame_auc = float("nan")  # e.g., if only one class in labels

    # 5) Video-level aggregation
    vids, y_true, y_score, y_pred = aggregate_by_video(paths, probs, labels)
    video_acc = accuracy_score(y_true, y_pred)
    video_bal = balanced_accuracy_score(y_true, y_pred)
    video_f1  = f1_score(y_true, y_pred, zero_division=0)
    try:
        video_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        video_auc = float("nan")

    # 6) Report (threshold=0.5)
    print("\n================ VIDEO-LEVEL EVAL ================")
    print(f"Split root: {root}")
    print(f"Frames: {len(labels)} | Videos: {len(vids)} | Max/vid: {max_vid_str}")
    print(
        f"Frame-level  Acc: {frame_acc:.4f}  AUC: {frame_auc:.4f}  "
        f"BalAcc: {frame_bal:.4f}  F1: {frame_f1:.4f}"
    )
    print(
        f"Video-level  Acc: {video_acc:.4f}  AUC: {video_auc:.4f}  "
        f"BalAcc: {video_bal:.4f}  F1: {video_f1:.4f}"
    )
    print(f"Videos — Real: {(y_true==0).sum()}  Fake: {(y_true==1).sum()}")

    # 7) Operating point
    print("\n---- Operating point selection ----")
    if args.apply_threshold is not None:
        t = float(args.apply_threshold)
        op_pred = (y_score >= t).astype(int)
        print(f"Applied threshold (from VAL): τ = {t:.4f}")
    else:
        # Find τ* on THIS split (use only for VAL; for TEST provide --apply-threshold)
        t, best_val = best_threshold(y_score, y_true, metric=args.search_metric)
        op_pred = (y_score >= t).astype(int)
        print(f"Best threshold on this split (metric={args.search_metric}): τ* = {t:.4f}")
        print(f"Best {args.search_metric}: {best_val:.4f}")

    print(
        f"Video Acc@τ: {accuracy_score(y_true, op_pred):.4f}  "
        f"BalAcc@τ: {balanced_accuracy_score(y_true, op_pred):.4f}  "
        f"F1@τ: {f1_score(y_true, op_pred, zero_division=0):.4f}"
    )
    # Per-class precision/recall (video)
    pr, rc, f1s, _ = precision_recall_fscore_support(y_true, op_pred, labels=[0,1], zero_division=0)
    print(f"Class 0 (Real)  P: {pr[0]:.4f}  R: {rc[0]:.4f}")
    print(f"Class 1 (Fake)  P: {pr[1]:.4f}  R: {rc[1]:.4f}")
    tn, fp, fn, tp = confusion_matrix(y_true, op_pred, labels=[0,1]).ravel()
    print(f"Confusion (video) | TN: {tn}  FP: {fp}  FN: {fn}  TP: {tp}")

    # 8) Save per-video scores if requested
    if args.save_csv:
        save_video_scores_csv(args.save_csv, vids, y_true, y_score)

    print("--------------------------------------------------")
    print("NOTE:")
    print(" • For fair paper numbers: choose τ* on the validation split,")
    print("   then re-run this script on TEST with --apply-threshold τ*.")
    print(" • AUC is threshold-free and should also be reported.")
    print("--------------------------------------------------")


if __name__ == "__main__":
    # H100-friendly defaults at process level
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    main()
