#!/usr/bin/env python3
# extract_img.py  — uniform, auto frame count, de-dup; RetinaFace crops unchanged
import os
import cv2
import glob
import numpy as np
import argparse
from shim_retinaface import RetinaFace

# ----------------------------
# Face detection (unchanged)
# ----------------------------
def extract_faces(frame_bgr, model, conf=None):
    """
    Detect faces in BGR frame and return list of BGR crops.
    Matches original behavior: no padding, no resize.
    """
    faces = (RetinaFace.detect_faces(frame_bgr, model=model)
             if conf is None else
             RetinaFace.detect_faces(frame_bgr, model=model, threshold=float(conf)))
    face_images = []
    if isinstance(faces, dict):
        for _, v in faces.items():
            l, t, r, b = v["facial_area"]  # [x1,y1,x2,y2]
            crop = frame_bgr[t:b, l:r]
            if crop.size != 0:
                face_images.append(crop)
    return face_images


# ----------------------------
# Perceptual hash (aHash) for de-dup
# ----------------------------
def ahash_bits(img_bgr, size=8):
    """
    aHash -> tuple of bools (length size*size). Lower Hamming distance => more similar.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    mean = small.mean()
    bits = (small > mean)
    return tuple(bool(x) for x in bits.flatten())

def hamming_distance(h1, h2):
    return sum(a != b for a, b in zip(h1, h2)) if len(h1) == len(h2) else max(len(h1), len(h2))


# ----------------------------
# Frame access helpers
# ----------------------------
def grab_at_index(cap, idx, total_frames, neighbor_offsets):
    """
    Seek to frame index `idx`. If backend can’t land exactly, probe neighbors.
    Returns (ok, frame_bgr, actual_index).
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(idx)))
    ok, frame = cap.read()
    if ok and frame is not None:
        return True, frame, int(idx)

    for off in neighbor_offsets:
        nf = int(np.clip(idx + off, 0, max(total_frames - 1, 0)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, nf)
        ok, frame = cap.read()
        if ok and frame is not None:
            return True, frame, nf
    return False, None, -1


# ----------------------------
# Auto decide frames from duration
# ----------------------------
def auto_frames_for_video(total_frames, fps, interval_s, min_frames, max_frames):
    """
    Compute target number of uniformly spaced samples based on desired temporal spacing (sec).
    Clamps to [min_frames, max_frames] and never exceeds total_frames.
    """
    if total_frames <= 0 or fps <= 0:
        # Fallback: conservative default
        return min_frames
    duration_s = total_frames / fps
    est = int(round(max(1.0, duration_s / max(0.05, float(interval_s)))))  # guard tiny intervals
    est = max(min_frames, min(est, max_frames))
    est = min(est, total_frames)
    return max(est, 1)


# ----------------------------
# Core per-video processing
# ----------------------------
def process_video(
    video_path,
    output_dir,
    video_name,
    frames_arg="auto",
    interval_s=0.5,
    min_frames=8,
    max_frames=100,
    neighbor_offsets=(-2, -1, 1, 2, -3, 3, -5, 5),
    conf=None,
    digits=2,
    dedupe=True,
    hash_size=8,
    hamming_thresh=5,
):
    """
    Uniformly sample frames across the clip. If frames_arg == 'auto', compute
    count from duration using interval_s (sec). De-duplicate frames by aHash.
    Then run original RetinaFace cropping on the kept frames. Crops saved at
    original size, named: {video_name}-{NN}.jpg (NN zero-padded to `digits`).
    """
    model = RetinaFace.build_model()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return 0

    os.makedirs(output_dir, exist_ok=True)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)

    # Decide how many samples
    if isinstance(frames_arg, str) and frames_arg.lower() == "auto":
        frames_per_video = auto_frames_for_video(total_frames, fps, interval_s, min_frames, max_frames)
    else:
        frames_per_video = int(frames_arg)

    if frames_per_video <= 0:
        cap.release()
        print(f"Skipping {video_name}: frames_per_video={frames_per_video}")
        return 0

    # Build uniform targets
    if total_frames > 0:
        targets = np.linspace(0, max(total_frames - 1, 0), num=frames_per_video, dtype=int)
        targets = np.unique(targets).tolist()
    else:
        # Unknown count → use ratio-based approximations
        stride = max(1, int(round((fps or 30.0) * interval_s)))
        targets = list(range(0, frames_per_video * stride, stride))

    pad_fmt = f"{{:0{int(digits)}d}}"
    saved_crop_count = 0
    frame_kept = 0
    seen_hashes = []

    for t_idx in targets:
        ok, frame, actual_idx = grab_at_index(cap, int(t_idx), total_frames, neighbor_offsets)
        if not ok:
            continue

        # De-dup frames before detection to avoid near-identical training inputs
        if dedupe:
            cand_hash = ahash_bits(frame, size=hash_size)
            # Try small neighborhood if too similar
            similar = any(hamming_distance(cand_hash, h) <= hamming_thresh for h in seen_hashes)
            if similar:
                picked = False
                for off in neighbor_offsets:
                    if total_frames > 0:
                        nf = int(np.clip(actual_idx + off, 0, max(total_frames - 1, 0)))
                    else:
                        nf = max(0, int(t_idx + off))
                    ok2, frame2, _ = grab_at_index(cap, nf, total_frames, neighbor_offsets=())
                    if not ok2:
                        continue
                    cand_hash2 = ahash_bits(frame2, size=hash_size)
                    if not any(hamming_distance(cand_hash2, h) <= hamming_thresh for h in seen_hashes):
                        frame = frame2
                        cand_hash = cand_hash2
                        picked = True
                        break
                if not picked:
                    continue  # skip this target
            seen_hashes.append(cand_hash)

        # Detect faces on the kept frame; save crops at original size
        face_images = extract_faces(frame, model, conf=conf)
        for i, face_img in enumerate(face_images):
            out_name = f"{video_name}-{pad_fmt.format(saved_crop_count)}.jpg"
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, face_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_crop_count += 1

        frame_kept += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"[{video_name}] kept_frames={frame_kept}  crops_saved={saved_crop_count}  (auto_frames={frames_arg=='auto'})")
    return saved_crop_count


# ----------------------------
# Directory traversal
# ----------------------------
def check_processed(output_dir, video_name):
    return len(glob.glob(os.path.join(output_dir, f"{video_name}-*.jpg"))) > 0

def process_directory(
    dataset_dir,
    output_base_dir,
    frames="auto",
    interval_s=0.5,
    min_frames=8,
    max_frames=100,
    max_videos_per_folder=20,
    neighbor_offsets=(-2, -1, 1, 2, -3, 3, -5, 5),
    conf=None,
    digits=2,
    dedupe=True,
    hash_size=8,
    hamming_thresh=5,
    force=False,
):
    total_crops = 0
    for root, _, files in os.walk(dataset_dir):
        video_files = [f for f in files if f.lower().endswith(".mp4")]
        video_files.sort()
        if max_videos_per_folder is not None:
            video_files = video_files[:max_videos_per_folder]

        for video_file in video_files:
            video_path = os.path.join(root, video_file)
            relative_path = os.path.relpath(root, dataset_dir)
            output_dir = os.path.join(output_base_dir, relative_path)
            os.makedirs(output_dir, exist_ok=True)

            video_name = os.path.splitext(video_file)[0]
            if not force and check_processed(output_dir, video_name):
                print(f"Skipping already processed: {video_name} in {output_dir}")
                continue

            total_crops += process_video(
                video_path,
                output_dir,
                video_name,
                frames_arg=frames,
                interval_s=interval_s,
                min_frames=min_frames,
                max_frames=max_frames,
                neighbor_offsets=neighbor_offsets,
                conf=conf,
                digits=digits,
                dedupe=dedupe,
                hash_size=hash_size,
                hamming_thresh=hamming_thresh,
            )
    print(f"\nTOTAL crops saved: {total_crops}")


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Uniformly sample frames per video (auto count by duration), "
                    "de-duplicate near-identical frames, then save RetinaFace crops "
                    "with the same naming/dimensions as the original script."
    )
    p.add_argument("--dataset_dir", "-i", required=True, type=str,
                   help="Root of dataset (contains subfolders with .mp4 files).")
    p.add_argument("--output_base_dir", "-o", required=True, type=str,
                   help="Where to mirror output structure and save crops.")

    # Auto sampling/limits
    p.add_argument("--frames", default="auto",
                   help="'auto' (default) or an integer count per video.")
    p.add_argument("--interval_s", type=float, default=0.5,
                   help="Target seconds between samples in auto mode (default: 0.5s).")
    p.add_argument("--min_frames", type=int, default=8,
                   help="Min frames per video in auto mode (default: 8).")
    p.add_argument("--max_frames", type=int, default=100,
                   help="Max frames per video in auto mode (default: 100).")

    # RetinaFace / detection
    p.add_argument("--conf", type=float,
                   help="Optional RetinaFace confidence threshold; if omitted, use library default.")

    # Seek behavior & naming
    p.add_argument("--neighbor", nargs="*", type=int,
                   default=[-2, -1, 1, 2, -3, 3, -5, 5],
                   help="Neighbor offsets to probe if exact seek fails.")
    p.add_argument("--digits", type=int, default=2,
                   help="Zero-padding for output index (default: 2 → -00.jpg).")
    p.add_argument("--max_per_dir", type=lambda x: None if x.lower()=="none" else int(x),
                   default=20,
                   help="Max videos per folder (default: 20). Use 'none' to process all.")
    p.add_argument("--force", action="store_true",
                   help="Reprocess even if output files already exist.")

    # De-dup
    p.add_argument("--no_dedupe", action="store_true",
                   help="Disable aHash de-duplication.")
    p.add_argument("--hash_size", type=int, default=8,
                   help="aHash size (NxN). Larger is stricter but slower (default: 8).")
    p.add_argument("--hamming", type=int, default=5,
                   help="Max Hamming distance to consider 'similar' (default: 5).")

    return p.parse_args()


def main():
    args = parse_args()
    frames = args.frames if isinstance(args.frames, str) and args.frames.lower() == "auto" else int(args.frames)
    neighbor_offsets = tuple(args.neighbor) if args.neighbor else tuple()
    process_directory(
        dataset_dir=args.dataset_dir,
        output_base_dir=args.output_base_dir,
        frames=frames,
        interval_s=args.interval_s,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        max_videos_per_folder=args.max_per_dir,
        neighbor_offsets=neighbor_offsets,
        conf=args.conf,
        digits=args.digits,
        dedupe=(not args.no_dedupe),
        hash_size=args.hash_size,
        hamming_thresh=args.hamming,
        force=args.force,
    )


if __name__ == "__main__":
    main()
