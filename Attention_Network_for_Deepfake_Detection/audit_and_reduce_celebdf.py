#!/usr/bin/env python3
import os, re, json, hashlib, argparse
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image
import numpy as np
import imagehash
import shutil
import csv
import matplotlib.pyplot as plt

CATEGORIES = ["Celeb-real","Celeb-synthesis","YouTube-real"]

def vid_from_name(fname):
    stem = os.path.splitext(os.path.basename(fname))[0]
    for pat in (r'^(id\d+_id\d+)', r'^(id\d+)', r'^(\d+)', r'^([^_]+)'):
        m = re.match(pat, stem)
        if m: return m.group(1)
    return stem

def frame_idx(p):
    m = re.search(r'_(\d+)\.\w+$', p)
    return int(m.group(1)) if m else 0

def audit_split(root):
    per_video = defaultdict(lambda: {"label":None,"frames":[]})
    class_counts = Counter()
    for cname in CATEGORIES:
        cdir = Path(root) / cname
        if not cdir.is_dir(): 
            continue
        lab = 0 if cname in ("Celeb-real","YouTube-real") else 1
        for f in cdir.glob("*.jpg"):
            v = vid_from_name(f.name)
            per_video[v]["label"] = lab
            per_video[v]["frames"].append(str(f.name))
            class_counts[lab]+=1
    for v in per_video:
        per_video[v]["frames"].sort(key=frame_idx)
    return per_video, class_counts

def stats_frames_per_video(per_video):
    lens = [len(per_video[v]["frames"]) for v in per_video]
    if not lens: return {}
    return {
        "min": int(np.min(lens)),
        "p50": float(np.percentile(lens,50)),
        "mean": float(np.mean(lens)),
        "p95": float(np.percentile(lens,95)),
        "max": int(np.max(lens))
    }

def plot_hist(lens, title, out_png):
    plt.figure(figsize=(5,3))
    plt.hist(lens, bins=40, color="steelblue")
    plt.title(title)
    plt.xlabel("frames/video")
    plt.ylabel("#videos")
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.close()

def phash_ok(prev_h, h, thr):
    return (prev_h is None) or ((h - prev_h) > thr)

def select_frames(frames, max_k=120, dedup=False, dup_thr=8):
    # frames are file names only; sorted
    if dedup:
        kept = []
        last = None
        for fn in frames:
            p = fn  # we will open with full path later
            try:
                h = imagehash.phash(Image.open(p).convert("RGB"))
            except Exception:
                continue
            if phash_ok(last, h, dup_thr):
                kept.append(fn)
                last = h
        frames = kept if kept else frames
    if len(frames) <= max_k:
        return frames
    idx = np.linspace(0, len(frames)-1, max_k, dtype=int)
    return [frames[i] for i in idx]

def symlink_selected(src_root, dst_root, selections):
    dst_root = Path(dst_root)
    for cname in CATEGORIES:
        (dst_root/cname).mkdir(parents=True, exist_ok=True)
    for cname, files in selections.items():
        for src in files:
            src_p = Path(src_root)/cname/src
            dst_p = dst_root/cname/src
            if not dst_p.exists():
                os.symlink(src_p.resolve(), dst_p)

def main():
    ap = argparse.ArgumentParser("CelebDF audit + reduce")
    ap.add_argument("--root", required=True, help="root holding train/val/test split dirs")
    ap.add_argument("--out", required=True, help="output dir for reduced splits and audit")
    ap.add_argument("--max-k", type=int, default=120, help="cap frames per video")
    ap.add_argument("--dedup", action="store_true", help="enable pHash de-dup per video")
    ap.add_argument("--dup-thr", type=int, default=8, help="hamming threshold for pHash")
    args = ap.parse_args()

    root = Path(args.root)
    out  = Path(args.out)
    audit_dir = out / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    overlap = {}
    lens_plots = {}

    selections = {"train":defaultdict(list), "val":defaultdict(list), "test":defaultdict(list)}

    for split in ("train","val","test"):
        split_root = root / split
        per_video, class_counts = audit_split(split_root)
        lens = [len(per_video[v]["frames"]) for v in per_video]
        # dump histogram
        if lens:
            png = audit_dir/f"{split}_frames_per_video.png"
            plot_hist(lens, f"{split} frames/video", png)
            lens_plots[split] = str(png)
        stats = stats_frames_per_video(per_video)
        n_frames = sum(lens)
        n_videos = len(per_video)
        n_real = sum(1 for v in per_video if per_video[v]["label"]==0)
        n_fake = n_videos - n_real

        summary[split] = dict(
            frames=n_frames, videos=n_videos,
            real=sum(1 for lab in per_video.values() for _ in lab["frames"] if lab["label"]==0),
            fake=sum(1 for lab in per_video.values() for _ in lab["frames"] if lab["label"]==1),
            dup_groups=0,  # placeholder
            stats=stats
        )

        # selection per video
        sel = defaultdict(list)
        for v, rec in per_video.items():
            lab = rec["label"]
            cdir = "YouTube-real" if lab==0 and "YouTube" in "".join(rec["frames"]) else ("Celeb-real" if lab==0 else "Celeb-synthesis")
            # build absolute file list
            abs_files = [str((split_root/cdir/f).resolve()) for f in rec["frames"]]
            chosen = select_frames(abs_files, max_k=args.max_k, dedup=args.dedup, dup_thr=args.dup_thr)
            # store relative names for symlink
            sel[cdir].extend([os.path.basename(p) for p in chosen])

        selections[split] = sel

    # simple overlap check by video id between splits
    def vids(split):
        per_video, _ = audit_split(root/split)
        return set(per_video.keys())
    overlap = {
        "train-val": len(vids("train") & vids("val")),
        "train-test": len(vids("train") & vids("test")),
        "val-test": len(vids("val") & vids("test")),
    }

    # write summary
    json.dump({"summary":summary, "overlap":overlap, "plots":lens_plots},
              open(audit_dir/"summary.json","w"), indent=2)
    print("\n=== SUMMARY ===")
    for s in ("train","val","test"):
        st = summary[s]
        stats = st["stats"]
        print(f"[{s}] frames={st['frames']}, videos={st['videos']}, "
              f"frames/video: min={stats.get('min',0)}, p50={stats.get('p50',0):.1f}, "
              f"mean={stats.get('mean',0):.1f}, p95={stats.get('p95',0):.1f}, max={stats.get('max',0)}")
    print("Overlap (video IDs):", overlap)
    print(f"\nAudit written to: {audit_dir}")

    # build reduced symlinked dataset
    red_root = out / f"{root.name}_reduced_k{args.max_k}{'_dedup' if args.dedup else ''}"
    for split in ("train","val","test"):
        dst = red_root / split
        dst.mkdir(parents=True, exist_ok=True)
        symlink_selected(root/split, dst, selections[split])

    print(f"\nSymlinked reduced dataset -> {red_root}")
    print("Point your config train/val/test roots to these reduced split directories.")
    print("Done.")
if __name__ == "__main__":
    main()
