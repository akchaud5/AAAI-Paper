#!/usr/bin/env python3
import os, re, json, csv, argparse, hashlib
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------- helpers ----------
CLASSES = {"Celeb-real": 0, "YouTube-real": 0, "Celeb-synthesis": 1}
SOURCES = ["Celeb-real", "YouTube-real", "Celeb-synthesis"]

VID_PATTERNS = [
    r"^(id\d+_id\d+)",   # id0_id16_00012.jpg -> id0_id16
    r"^(id\d+)",         # id0_00012.jpg      -> id0
    r"^(\d+)",           # 00001_0001.jpg     -> 00001
    r"^([^_]+)",         # fallback before first underscore
]

def extract_video_id(filename: str) -> str:
    name = os.path.splitext(os.path.basename(filename))[0]
    for pat in VID_PATTERNS:
        m = re.match(pat, name)
        if m:
            return m.group(1)
    return re.sub(r"_\d+$", "", name)

def img_size(path):
    try:
        with Image.open(path) as im:
            return im.size  # (W, H)
    except (UnidentifiedImageError, OSError):
        return None

def md5sum(path, blocksize=1<<20):
    m = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(blocksize), b""):
            m.update(chunk)
    return m.hexdigest()

def ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)

# ---------- analysis ----------
def scan_split(split_root, split_name, do_hash=False):
    """
    Returns:
      dict with:
        - counts, by_source, by_class
        - corrupt, unreadable
        - videos: {video_id: {'count': n, 'label': 0/1}}
        - frames: list of dict rows for CSV
        - size_stats
        - dupes (optional)
    """
    result = {
        "split": split_name,
        "total": 0,
        "by_class": Counter(),
        "by_source": Counter(),
        "videos": defaultdict(lambda: {"count": 0, "label": None}),
        "corrupt": [],
        "frames": [],
        "resolutions": [],
        "dupes": []  # list of groups
    }

    hash_map = defaultdict(list) if do_hash else None

    for src in SOURCES:
        src_dir = Path(split_root) / src
        if not src_dir.exists():
            continue

        files = [p for p in src_dir.iterdir()
                 if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        result["by_source"][src] += len(files)

        for p in files:
            result["total"] += 1
            cls = CLASSES[src]
            result["by_class"][cls] += 1

            vid = extract_video_id(p.name)
            result["videos"][vid]["count"] += 1
            result["videos"][vid]["label"] = cls

            size = img_size(p)
            if size is None:
                result["corrupt"].append(str(p))
            else:
                w, h = size
                result["resolutions"].append((w, h))

            if do_hash:
                try:
                    h = md5sum(p)
                    hash_map[h].append(str(p))
                except Exception:
                    pass

            result["frames"].append({
                "split": split_name,
                "source": src,
                "class": int(cls),
                "video_id": vid,
                "path": str(p)
            })

    # find duplicate groups
    if do_hash:
        for k, v in hash_map.items():
            if len(v) > 1:
                result["dupes"].append({"hash": k, "paths": v})

    # size stats
    if result["resolutions"]:
        ws, hs = zip(*result["resolutions"])
        result["size_stats"] = {
            "count": len(ws),
            "w_min": int(np.min(ws)), "w_p50": int(np.median(ws)),
            "w_mean": float(np.mean(ws)), "w_p95": int(np.percentile(ws, 95)),
            "w_max": int(np.max(ws)),
            "h_min": int(np.min(hs)), "h_p50": int(np.median(hs)),
            "h_mean": float(np.mean(hs)), "h_p95": int(np.percentile(hs, 95)),
            "h_max": int(np.max(hs)),
        }
    else:
        result["size_stats"] = {}

    return result

def hist_frames_per_video(split_name, videos, out_dir):
    counts = [v["count"] for v in videos.values()]
    if not counts:
        return
    plt.figure(figsize=(7,4))
    plt.hist(counts, bins=40, color="#3a7bd5", alpha=0.85)
    plt.title(f"Frames per video – {split_name} (n={len(counts)})")
    plt.xlabel("frames per video")
    plt.ylabel("videos")
    plt.grid(alpha=0.25)
    ensure_dir(out_dir)
    plt.tight_layout()
    plt.savefig(str(Path(out_dir)/f"frames_per_video_{split_name}.png"), dpi=140)
    plt.close()

def hist_resolutions(split_name, resolutions, out_dir):
    if not resolutions:
        return
    ws, hs = zip(*resolutions)
    plt.figure(figsize=(7,4))
    plt.hist(ws, bins=40, alpha=0.8, label="width")
    plt.hist(hs, bins=40, alpha=0.6, label="height")
    plt.title(f"Image resolution – {split_name} (n={len(ws)})")
    plt.xlabel("pixels")
    plt.ylabel("count")
    plt.legend()
    plt.grid(alpha=0.25)
    ensure_dir(out_dir)
    plt.tight_layout()
    plt.savefig(str(Path(out_dir)/f"resolution_{split_name}.png"), dpi=140)
    plt.close()

def summarize_split(split_stats):
    vids = split_stats["videos"]
    counts = [v["count"] for v in vids.values()] if vids else []
    if counts:
        fr = {
            "min": int(np.min(counts)),
            "p25": int(np.percentile(counts, 25)),
            "p50": int(np.median(counts)),
            "mean": float(np.mean(counts)),
            "p95": int(np.percentile(counts, 95)),
            "max": int(np.max(counts)),
        }
    else:
        fr = {}
    return {
        "total_frames": split_stats["total"],
        "class_0_real": int(split_stats["by_class"][0]),
        "class_1_fake": int(split_stats["by_class"][1]),
        "by_source": {k: int(v) for k, v in split_stats["by_source"].items()},
        "num_videos": len(vids),
        "frames_per_video": fr,
        "num_corrupt": len(split_stats["corrupt"]),
        "num_dupe_groups": len(split_stats.get("dupes", [])),
        "size_stats": split_stats.get("size_stats", {}),
    }

def main():
    ap = argparse.ArgumentParser(description="CelebDF split auditor")
    ap.add_argument("--root", type=str, default="celebdb_splits",
                    help="root containing train/val/test")
    ap.add_argument("--out", type=str, default="celebdb_audit",
                    help="output directory for report/plots")
    ap.add_argument("--hash", action="store_true",
                    help="compute md5 to detect duplicate frames (slower)")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    splits = ["train", "val", "test"]
    stats = {}
    frames_index = []

    print(f"🔍 Auditing splits under: {root.resolve()}")
    for sp in splits:
        sp_root = root / sp
        if not sp_root.exists():
            print(f"  ⚠️ split missing: {sp_root}")
            continue
        print(f"  → scanning {sp} ...")
        res = scan_split(sp_root, sp, do_hash=args.hash)
        stats[sp] = res
        frames_index.extend(res["frames"])
        # plots
        hist_frames_per_video(sp, res["videos"], out_dir)
        hist_resolutions(sp, res["resolutions"], out_dir)

    # overlap check across splits
    vid_sets = {sp: set(stats[sp]["videos"].keys()) for sp in stats.keys()}
    overlaps = {
        "train-val": list(vid_sets.get("train", set()) & vid_sets.get("val", set())),
        "train-test": list(vid_sets.get("train", set()) & vid_sets.get("test", set())),
        "val-test": list(vid_sets.get("val", set()) & vid_sets.get("test", set())),
    }

    # compact summary
    compact = {
        sp: summarize_split(stats[sp]) for sp in stats.keys()
    }
    compact["overlap"] = {k: len(v) for k, v in overlaps.items()}

    # write JSON report
    with open(out_dir/"split_stats.json", "w") as f:
        json.dump({
            "summary": compact,
            "overlap_lists": overlaps,  # keep full list in case you want to inspect
        }, f, indent=2)

    # write CSV index (one row per frame)
    with open(out_dir/"video_index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "source", "class", "video_id", "path"])
        w.writeheader()
        w.writerows(frames_index)

    # simple console summary
    print("\n================ SUMMARY ================")
    for sp in splits:
        if sp not in compact: 
            continue
        s = compact[sp]
        print(f"[{sp}] frames={s['total_frames']}, videos={s['num_videos']}, "
              f"real={s['class_0_real']}, fake={s['class_1_fake']}, "
              f"corrupt={s['num_corrupt']}, dup_groups={s['num_dupe_groups']}")
        if s["frames_per_video"]:
            fr = s["frames_per_video"]
            print(f"    frames/video: min={fr['min']}, p50={fr['p50']}, "
                  f"mean={fr['mean']:.1f}, p95={fr['p95']}, max={fr['max']}")
    print("Overlap (video IDs):", {k: v for k, v in compact["overlap"].items()})
    print(f"\n💾 Saved report to: {out_dir/'split_stats.json'}")
    print(f"🖼  Plots saved to: {out_dir.resolve()}")
    print(f"📑 Full index:      {out_dir/'video_index.csv'}")

    # ---- Suggestions based on distribution ----
    print("\n🧭 Suggestions:")
    for sp in splits:
        if sp not in compact: 
            continue
        s = compact[sp]
        if s["frames_per_video"]:
            p95 = s["frames_per_video"]["p95"]
            if p95 > 120:
                print(f"  • [{sp}] cap frames/video at ~{min(120,p95)} (sample per video) "
                      f"to reduce correlation & speed up training.")
        if s["class_0_real"] and s["class_1_fake"]:
            ratio = s["class_1_fake"] / max(s["class_0_real"],1)
            if ratio > 2.0 or ratio < 0.5:
                print(f"  • [{sp}] class imbalance detected (fake/real ≈ {ratio:.2f}). "
                      f"Use sampler or loss weighting.")
        if s["num_corrupt"] > 0:
            print(f"  • [{sp}] {s['num_corrupt']} corrupt images — remove or re-extract.")
    if compact["overlap"]["train-val"] or compact["overlap"]["train-test"] or compact["overlap"]["val-test"]:
        print("  • ❌ Overlap between splits found — re-create video‑level splits.")
    else:
        print("  • ✅ No overlap between splits (video‑level clean).")
    print("=========================================")


if __name__ == "__main__":
    main()
