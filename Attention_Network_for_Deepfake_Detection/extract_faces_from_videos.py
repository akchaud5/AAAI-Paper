#!/usr/bin/env python3
"""
Extract face crops from videos with:
 - uniform temporal sampling across each video
 - InsightFace (GPU) via shim_retinaface (fallbacks included)
 - optional perceptual-hash dedupe to avoid near-duplicate crops
 - resumable processing (skips videos that already have crops unless --force)

Examples
--------
# Resume run (skip processed), auto frames ~ every 0.5s, clamp 8..100 per video
python3 extract_faces_from_videos.py \
  -i . -o ./celebdb \
  --frames auto --interval_s 0.5 --min_frames 8 --max_frames 100 \
  --max_per_dir none --conf 0.6

# Reprocess everything (overwrite)
python3 extract_faces_from_videos.py ... --force
"""

from __future__ import annotations
import argparse
import glob
import math
import os
import sys
from collections import defaultdict
from typing import List, Tuple

import cv2
from PIL import Image
import imagehash

from shim_retinaface import RetinaFace


# ---------- utilities ----------

def list_videos(input_dir: str, max_per_dir: int | None) -> List[Tuple[str, str, str]]:
    """
    Return list of (video_path, rel_dir, video_name).
    If input_dir contains subfolders (e.g. Celeb-real), we keep their relative path.
    """
    videos = []
    if os.path.isfile(input_dir) and input_dir.lower().endswith(".mp4"):
        v = input_dir
        rel_dir = os.path.dirname(os.path.relpath(v, start=os.path.dirname(input_dir))) or "."
        videos.append((v, rel_dir, os.path.splitext(os.path.basename(v))[0]))
        return videos

    # If input_dir points to a specific class folder, just that folder.
    patterns = []
    if os.path.isdir(input_dir):
        # If exactly these folders exist, keep the structure; else just glob all mp4s
        special = ["Celeb-real", "Celeb-synthesis", "YouTube-real"]
        subdirs = [d for d in special if os.path.isdir(os.path.join(input_dir, d))]
        if subdirs:
            for d in subdirs:
                patterns.append(os.path.join(input_dir, d, "*.mp4"))
        else:
            patterns.append(os.path.join(input_dir, "*.mp4"))
    else:
        raise FileNotFoundError(f"Input path not found: {input_dir}")

    for patt in patterns:
        found = sorted(glob.glob(patt))
        if max_per_dir is not None and len(found) > max_per_dir:
            found = found[:max_per_dir]
        for v in found:
            rel_dir = os.path.relpath(os.path.dirname(v), start=input_dir)
            videos.append((v, rel_dir, os.path.splitext(os.path.basename(v))[0]))
    return videos


def already_processed(out_dir: str, video_name: str) -> bool:
    patt = os.path.join(out_dir, f"{video_name}-*.jpg")
    return len(glob.glob(patt)) > 0


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def evenly_spaced_indices(n_frames: int, keep: int) -> List[int]:
    """Return 'keep' indices from [0..n_frames-1] spaced as evenly as possible."""
    if keep <= 0:
        return []
    if keep >= n_frames:
        return list(range(n_frames))
    # linspace inclusive on both ends, then clamp & unique
    idxs = [int(round(i)) for i in
            [i * (n_frames - 1) / (keep - 1) for i in range(keep)]]
    # Guarantee sorted uniques within range
    seen, out = set(), []
    for i in idxs:
        i = min(max(i, 0), n_frames - 1)
        if i not in seen:
            out.append(i)
            seen.add(i)
    return out


def choose_kept_count(cap: cv2.VideoCapture, frames_arg: str, interval_s: float,
                      min_frames: int, max_frames: int) -> Tuple[int, bool]:
    """
    Decide how many frames to sample. If frames_arg == 'auto', compute from duration/interval_s.
    Returns (keep_count, auto_flag).
    """
    if frames_arg != "auto":
        try:
            keep = int(frames_arg)
            return max(min_frames, min(max_frames, keep)), False
        except Exception:
            raise ValueError("--frames must be 'auto' or an integer")

    # auto: approximate from duration
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = n_frames / max(fps, 1e-6)
    est = int(max(1, duration_s / max(interval_s, 1e-6)))
    keep = max(min_frames, min(max_frames, est))
    return keep, True


# ---------- main per-video processing ----------

def process_video(video_path: str, output_dir: str, video_name: str,
                  frames_arg: str, interval_s: float,
                  min_frames: int, max_frames: int,
                  conf: float, dedupe: bool, hash_size: int, hamming_thresh: int,
                  model) -> int:
    """
    Extract faces for one video.
    Returns number of crops saved.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] cannot open: {video_path}")
        return 0

    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n_total <= 0:
        cap.release()
        return 0

    # Decide how many frames to keep & choose indices uniformly across video
    keep_count, auto_flag = choose_kept_count(cap, frames_arg, interval_s, min_frames, max_frames)
    sample_idxs = evenly_spaced_indices(n_total, keep_count)

    # Build a map index->(pos) and jump via CAP_PROP_POS_FRAMES
    hashes = []  # for dedupe
    saved = 0
    ensure_dir(output_dir)

    for idx in sample_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        # detect faces
        dets = RetinaFace.detect_faces(frame, model=model, threshold=float(conf))
        if not isinstance(dets, dict) or not dets:
            continue

        # save each face
        for det in dets.values():
            x1, y1, x2, y2 = det.get("facial_area", [0, 0, 0, 0])
            # clamp & sanity
            h, w = frame.shape[:2]
            x1 = max(0, min(int(x1), w - 1)); y1 = max(0, min(int(y1), h - 1))
            x2 = max(0, min(int(x2), w));     y2 = max(0, min(int(y2), h))
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2, :]
            if crop.size == 0 or crop.shape[0] < 32 or crop.shape[1] < 32:
                continue

            # dedupe (per-video) using perceptual hash
            if dedupe:
                pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                ph = imagehash.phash(pil, hash_size=hash_size)
                is_dup = any(ph - h_ <= hamming_thresh for h_ in hashes)
                if is_dup:
                    continue
                hashes.append(ph)

            # save (match your earlier naming: videoName-XX.jpg)
            out_path = os.path.join(output_dir, f"{video_name}-{saved:02d}.jpg")
            cv2.imwrite(out_path, crop)
            saved += 1

    cap.release()
    print(f"[{video_name}] kept_frames={len(sample_idxs)}  crops_saved={saved}  "
          f"(auto_frames={frames_arg == 'auto'})")
    return saved


def process_directory(input_dir: str, output_root: str, frames_arg: str,
                      interval_s: float, min_frames: int, max_frames: int,
                      max_per_dir: int | None, conf: float, force: bool,
                      dedupe: bool, hash_size: int, hamming_thresh: int) -> int:
    """
    Walk input_dir and process videos (optionally limited per subdir).
    Returns total crops saved.
    """
    vids = list_videos(input_dir, max_per_dir)
    if not vids:
        print(f"[INFO] no videos found under {input_dir}")
        return 0

    # Build the detector once and reuse
    model = RetinaFace.build_model()

    total = 0
    for vpath, rel_dir, vname in vids:
        out_dir = os.path.join(output_root, rel_dir if rel_dir != "." else os.path.basename(os.path.dirname(vpath)) or ".")
        # If input_dir is exactly the parent of the file, keep rel_dir "." → put crops under output_root
        # If input_dir is the dataset root, rel_dir carries 'Celeb-real' etc.

        ensure_dir(out_dir)

        if not force and already_processed(out_dir, vname):
            print(f"Skipping already processed: {vname} in {out_dir}")
            continue

        total += process_video(
            video_path=vpath,
            output_dir=out_dir,
            video_name=vname,
            frames_arg=frames_arg,
            interval_s=interval_s,
            min_frames=min_frames,
            max_frames=max_frames,
            conf=conf,
            dedupe=dedupe,
            hash_size=hash_size,
            hamming_thresh=hamming_thresh,
            model=model,
        )

    return total


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Uniform, resumable face extraction (InsightFace GPU).")
    p.add_argument("-i", "--input_dir", required=True, help="Input directory (dataset root or class folder) or single .mp4")
    p.add_argument("-o", "--output_dir", required=True, help="Output root directory for crops")

    p.add_argument("--frames", default="auto", help="'auto' or integer frames per video")
    p.add_argument("--interval_s", type=float, default=0.5, help="When --frames=auto, sample ~every N seconds")
    p.add_argument("--min_frames", type=int, default=8, help="Minimum frames per video")
    p.add_argument("--max_frames", type=int, default=100, help="Maximum frames per video")

    p.add_argument("--max_per_dir", default="none",
                   help="Limit videos per subdir ('none' for unlimited)")
    p.add_argument("--conf", type=float, default=0.6, help="Detection threshold (0.6–0.7 is good for InsightFace)")
    p.add_argument("--force", action="store_true", help="Reprocess even if outputs already exist")

    p.add_argument("--no_dedupe", action="store_true", help="Disable perceptual-hash dedupe")
    p.add_argument("--hash_size", type=int, default=8, help="ImageHash size")
    p.add_argument("--hamming", type=int, default=5, help="Max Hamming distance considered duplicate")

    args = p.parse_args(argv)

    # normalize
    if args.frames != "auto":
        try:
            _ = int(args.frames)
        except Exception:
            p.error("--frames must be 'auto' or an integer")

    if args.max_per_dir is None or str(args.max_per_dir).lower() == "none":
        args.max_per_dir = None
    else:
        try:
            args.max_per_dir = int(args.max_per_dir)
        except Exception:
            p.error("--max_per_dir must be 'none' or an integer")

    return args


def main(argv=None):
    args = parse_args(argv)
    total = process_directory(
        input_dir=args.input_dir,
        output_root=args.output_dir,
        frames_arg=args.frames,
        interval_s=args.interval_s,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        max_per_dir=args.max_per_dir,
        conf=args.conf,
        force=args.force,
        dedupe=(not args.no_dedupe),
        hash_size=args.hash_size,
        hamming_thresh=args.hamming,
    )
    print(f"\n[Done] Total crops saved: {total}")


if __name__ == "__main__":
    main()
