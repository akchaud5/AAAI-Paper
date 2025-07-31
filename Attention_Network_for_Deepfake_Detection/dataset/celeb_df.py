# dataset/celeb_df.py
import os
import re
import random
from typing import List, Tuple, Dict
from collections import defaultdict, Counter

import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from os.path import join


class CelebDF(Dataset):
    """
    CelebDF frames dataset (video-aware).

    Key features (behaviour-safe for training objective):
      • Optional max_frames_per_video with uniform temporal sampling (reduces correlation).
      • Optional near-duplicate filtering using tiny grayscale thumbnails (very light).
      • Optional return_path for val/test to compute video‑level metrics in the trainer.
      • Reproducible seeded shuffle for train split.
      • Exposes sample_weights and class_weight_tensor for:
          - WeightedRandomSampler (balanced batches)
          - Class‑weighted CrossEntropyLoss
      • set_epoch(e): re-jitters the uniform sampling each epoch (if trainer calls it).

    Folder layout per split:
        <root>/
          Celeb-real/
          Celeb-synthesis/
          YouTube-real/
    """

    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    # ---------- filename parsing patterns ----------
    _VID_PATTERNS = (
        r"^(id\d+_id\d+)",  # id0_id16_0000.jpg -> id0_id16
        r"^(id\d+)",        # id0_0000.jpg      -> id0
        r"^(\d+)",          # 00000_0123.jpg    -> 00000
        r"^([^_]+)",        # fallback: token before first underscore
    )

    @staticmethod
    def _video_id_from_name(fname: str) -> str:
        stem = os.path.splitext(os.path.basename(fname))[0]
        for pat in CelebDF._VID_PATTERNS:
            m = re.match(pat, stem)
            if m:
                return m.group(1)
        return stem

    @staticmethod
    def _frame_index_from_path(path_or_name: str) -> int:
        m = re.search(r'_(\d+)\.\w+$', os.path.basename(path_or_name))
        return int(m.group(1)) if m else 0

    # ---------- init ----------
    def __init__(self, cfg: Dict):
        self.root = cfg["root"]
        self.split = cfg.get("split", "train")  # "train" | "val" | "test"
        self.return_path = bool(cfg.get("return_path", self.split != "train"))

        # seeding for reproducibility + small jitter
        self.seed = cfg.get("seed", None)
        self._base_seed = int(cfg.get("seed", 1337))
        self._rng = random.Random(self._base_seed)

        # per-video sampling options (all optional)
        self.max_per_video = cfg.get("max_frames_per_video", None)
        self.max_per_video = int(self.max_per_video) if self.max_per_video not in (None, "", 0) else None
        self.sampling_strategy = str(cfg.get("sampling_strategy", "strided")).lower()
        self.stride_hint = int(cfg.get("stride_hint", 5))
        self.drop_near_duplicates = bool(
            cfg.get("drop_near_duplicates", False) or (self.sampling_strategy == "strided_unique")
        )
        self.sim_threshold = float(cfg.get("sim_threshold", 0.985))  # cosine sim threshold
        self.sim_window = int(cfg.get("sim_window", 6))              # (kept for API symmetry)

        # transforms
        self.transforms = self.__get_transforms(cfg.get("transforms", []))

        # build the index of (relative_frame_path, label)
        self.images_ids: List[Tuple[str, int]] = self.__build_index()
        self.categories = {0: "Real", 1: "Fake"}

        # expose counts & weights for sampler / weighted CE
        self.class_counts: Counter = Counter(y for _, y in self.images_ids)
        total = sum(self.class_counts.values())
        self._class_w = {c: total / (self.class_counts.get(c, 1) + 1e-12) for c in (0, 1)}
        self.sample_weights = [self._class_w[y] for _, y in self.images_ids]

        print(
            f"[CelebDF:{self.split}] {len(self.images_ids)} frames | "
            f"videos≈{self._num_videos} | real={self.class_counts.get(0,0)}, "
            f"fake={self.class_counts.get(1,0)}"
        )

    # ---------- transforms ----------
    def __get_transforms(self, transforms_cfg):
        t_list = []
        saw_to_tensor = False

        for t in transforms_cfg:
            name = t.get("name")
            p = t.get("params", {}) or {}

            if name == "Resize":
                t_list.append(transforms.Resize((p["height"], p["width"])))
            elif name == "HorizontalFlip":
                t_list.append(transforms.RandomHorizontalFlip(p=p.get("p", 0.5)))
            elif name == "ColorJitter":
                t_list.append(
                    transforms.ColorJitter(
                        brightness=p.get("brightness", 0.0),
                        contrast=p.get("contrast", 0.0),
                        saturation=p.get("saturation", 0.0),
                        hue=p.get("hue", 0.0),
                    )
                )
            elif name == "Normalize":
                # Ensure ToTensor comes before Normalize
                t_list.append(transforms.ToTensor())
                saw_to_tensor = True
                t_list.append(transforms.Normalize(mean=p["mean"], std=p["std"]))

        if not saw_to_tensor:
            if not any(isinstance(tt, transforms.ToTensor) for tt in t_list):
                t_list.append(transforms.ToTensor())

        return transforms.Compose(t_list) if t_list else None

    # ---------- indexing & per-video sampling ----------
    def __build_index(self) -> List[Tuple[str, int]]:
        # Collect all frames grouped by video id + label
        label_map = {
            "Celeb-real": 0,
            "YouTube-real": 0,
            "Celeb-synthesis": 1,
        }

        per_video: Dict[Tuple[str, int], List[str]] = defaultdict(list)  # (vid, label) -> [rel_path...]
        for subdir, label in label_map.items():
            subdir_path = join(self.root, subdir)
            if not os.path.isdir(subdir_path):
                continue
            try:
                # use scandir to ensure files exist and are regular
                for e in os.scandir(subdir_path):
                    if not e.is_file():
                        continue
                    fname = e.name
                    if not fname.lower().endswith(self.IMG_EXTS):
                        continue
                    vid = self._video_id_from_name(fname)
                    per_video[(vid, label)].append(os.path.join(subdir, fname))
            except OSError as e:
                print(f"[CelebDF] Warning: cannot read {subdir_path}: {e}")

        # Sort within video by temporal index
        for key in per_video.keys():
            per_video[key].sort(key=self._frame_index_from_path)

        # Build final list with per-video cap & dedup; drop missing safely
        ids: List[Tuple[str, int]] = []
        missing_at_build = 0

        for (vid, lab), frames in per_video.items():
            if self.max_per_video is not None and len(frames) > self.max_per_video:
                keep = self._select_uniform_candidates(frames, self.max_per_video)
                if self.drop_near_duplicates:
                    keep = self._filter_near_duplicates(keep, target_k=self.max_per_video)
                # Still too many (rare) → cut evenly
                if len(keep) > self.max_per_video:
                    idx = np.linspace(0, len(keep) - 1, self.max_per_video, dtype=int)
                    keep = [keep[i] for i in idx]
            else:
                keep = frames

            # filter out any path that does not exist at build time
            exist_keep = []
            for rp in keep:
                if os.path.exists(join(self.root, rp)):
                    exist_keep.append(rp)
                else:
                    missing_at_build += 1

            ids.extend([(rp, lab) for rp in exist_keep])

        # Count videos for the log
        self._num_videos = len(per_video)
        if missing_at_build > 0:
            print(f"[CelebDF:{self.split}] dropped {missing_at_build} missing files at index build")

        # Reproducible shuffle on train; val/test keep order
        if self.split == "train":
            if self.seed is not None:
                rng = random.Random(self.seed)
                rng.shuffle(ids)
            else:
                random.shuffle(ids)

        return ids

    def _select_uniform_candidates(self, frames: List[str], k: int) -> List[str]:
        """
        Evenly spaced frame selection with tiny jitter so different epochs
        don't see exactly the same frames (if trainer calls set_epoch()).
        """
        n = len(frames)
        if n <= k:
            return list(frames)

        idx = np.linspace(0, n - 1, k, dtype=int)

        # Tiny ±1 jitter (train only)
        if self.split == "train":
            used = set()
            out = []
            for i in idx:
                j = i + self._rng.randint(-1, 1)
                j = min(n - 1, max(0, j))
                if j in used:
                    for d in (1, -1, 2, -2, 3, -3):
                        q = min(n - 1, max(0, i + d))
                        if q not in used:
                            j = q
                            break
                used.add(j)
                out.append(j)
            idx = np.array(sorted(out), dtype=int)

        return [frames[i] for i in idx]

    def _filter_near_duplicates(self, candidates: List[str], target_k: int) -> List[str]:
        """
        Remove near-duplicate frames among the candidate list using a tiny
        grayscale thumbnail and cosine similarity to the last kept frame.
        """
        if not candidates:
            return candidates

        kept = []
        last_feat = None

        for relp in candidates:
            full = join(self.root, relp)
            feat = self._thumb_feature(full)
            if feat is None:
                continue

            if last_feat is None:
                kept.append(relp)
                last_feat = feat
            else:
                cos = float(np.dot(last_feat, feat))
                if cos < self.sim_threshold:  # keep if sufficiently different
                    kept.append(relp)
                    last_feat = feat

            if len(kept) >= target_k:
                break

        # If dedup dropped too many, pad evenly from remaining candidates
        if len(kept) < target_k:
            remaining = [r for r in candidates if r not in set(kept)]
            need = target_k - len(kept)
            if remaining:
                idx = np.linspace(0, len(remaining) - 1, min(need, len(remaining)), dtype=int)
                kept.extend([remaining[i] for i in idx])

        return kept

    @staticmethod
    def _thumb_feature(path: str):
        """
        Tiny grayscale feature for near-duplicate filtering.
        Returns L2-normalized 256-dim vector (16x16), or None on failure.
        """
        try:
            with Image.open(path) as im:
                im = im.convert("L").resize((16, 16), Image.BILINEAR)
                v = np.asarray(im, dtype=np.float32).reshape(-1)
                v -= v.mean()
                n = np.linalg.norm(v) + 1e-8
                return (v / n)
        except Exception:
            return None

    # ---------- dataset API ----------
    def __len__(self):
        return len(self.images_ids)

    def __getitem__(self, idx):
        """
        Fail-soft loader: if the requested image is missing, try a few nearby
        indices (wrap around) before raising a clear error. This prevents training
        from crashing when a handful of files disappear.
        """
        trials = 5
        N = len(self.images_ids)
        k = 0
        last_err = None

        while k < trials:
            rel_path, label = self.images_ids[idx % N]
            full_path = join(self.root, rel_path)
            try:
                image = self.__load_image(full_path)
                if self.transforms:
                    image = self.transforms(image)
                if self.return_path:
                    return image, label, full_path
                return image, label
            except FileNotFoundError as e:
                # try next item
                last_err = e
                idx += 1
                k += 1
                continue

        # If we reach here, several consecutive items are missing
        raise last_err if last_err is not None else FileNotFoundError("Image not found")

    # ---------- IO ----------
    def __load_image(self, path):
        # Fail loud if image missing/corrupt; callers can catch if needed
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with Image.open(path) as im:
            return im.convert("RGB")

    # ---------- sampler & weighted CE helpers ----------
    def class_weight_tensor(self) -> np.ndarray:
        """Returns class weights proportional to inverse frequency, shape [2]."""
        tot = sum(self.class_counts.values())
        w0 = tot / (self.class_counts.get(0, 1) + 1e-12)
        w1 = tot / (self.class_counts.get(1, 1) + 1e-12)
        return np.array([w0, w1], dtype=np.float32)

    # ---------- epoch reseeding (optional) ----------
    def set_epoch(self, e: int):
        """
        Re-jitter uniform sampling each epoch (trainer may call this).
        Only affects train split.
        """
        if self.split != "train":
            return
        self._rng = random.Random(self._base_seed + int(e))
        # rebuild the index to move jittered picks
        self.images_ids = self.__build_index()
        # recompute counts and weights
        self.class_counts = Counter(y for _, y in self.images_ids)
        total = sum(self.class_counts.values())
        self._class_w = {c: total / (self.class_counts.get(c, 1) + 1e-12) for c in (0, 1)}
        self.sample_weights = [self._class_w[y] for _, y in self.images_ids]
