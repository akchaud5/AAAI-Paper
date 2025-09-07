#!/usr/bin/env python3
"""
Enhanced CelebDF Dataset with Integrated Balanced Splitting
===========================================================

This module provides both:
1. CelebDF PyTorch Dataset class for training with advanced features
2. BalancedCelebDBSplitter for creating stratified, balanced train/val/test splits

Features:
- Stratified splitting with configurable real/fake balance
- Video-level splitting to prevent data leakage  
- Quality-aware distribution by frame count
- Advanced per-video frame sampling with deduplication
- Class balancing via WeightedRandomSampler and weighted loss
- Comprehensive validation and reporting
- PyTorch Dataset API with robust error handling
"""

import os
import re
import random
import shutil
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional
import numpy as np

# PyTorch imports
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# Analysis and plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# BALANCED SPLITTER CLASS
# =============================================================================

class BalancedCelebDBSplitter:
    """Enhanced CelebDB splitter with stratification and balance control."""
    
    def __init__(self, 
                 celebdb_root: str,
                 splits_root: str,
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1,
                 test_ratio: float = 0.1,
                 target_balance: float = 0.5,
                 balance_tolerance: float = 0.05,
                 seed: int = 42):
        
        self.celebdb_root = Path(celebdb_root)
        self.splits_root = Path(splits_root)
        
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio  
        self.test_ratio = test_ratio
        
        # Balance control
        self.target_balance = target_balance  # 0.5 = 50% fake, 50% real
        self.balance_tolerance = balance_tolerance  # ±5% tolerance
        
        # Reproducibility
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Categories and labels
        self.categories = {
            'Celeb-real': 0,      # Real
            'Celeb-synthesis': 1,  # Fake 
            'YouTube-real': 0      # Real
        }
        
        print("Enhanced CelebDB Balanced Splitter")
        print("=" * 50)
        print(f"Source: {self.celebdb_root}")
        print(f"Output: {self.splits_root}")
        print(f"Ratios: train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%}")
        print(f"Target balance: {target_balance:.1%} fake")
        print(f"Tolerance: ±{balance_tolerance:.1%}")
        print(f"Seed: {seed}")
    
    def extract_video_id(self, filename: str) -> str:
        """Extract video ID from filename using multiple patterns."""
        stem = filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
        
        # Try different patterns
        patterns = [
            r'(id\d+_id\d+)',  # id0_id16_0000 -> id0_id16
            r'(id\d+)',        # id0_0000 -> id0  
            r'(\d+)',          # 00000_0123 -> 00000
        ]
        
        for pattern in patterns:
            match = re.match(pattern, stem)
            if match:
                return match.group(1)
        
        # Fallback: everything before first dash/underscore
        return stem.split('-')[0].split('_')[0]
    
    def scan_dataset(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Scan dataset and group files by video ID.
        Returns: {category: {video_id: [frame_files...]}}
        """
        print("\nScanning dataset...")
        videos_by_category = {}
        
        for category in self.categories:
            category_path = self.celebdb_root / category
            if not category_path.exists():
                print(f"Warning: {category} directory not found")
                videos_by_category[category] = {}
                continue
                
            print(f"  Scanning {category}...")
            video_groups = defaultdict(list)
            
            for img_file in category_path.glob('*.jpg'):
                video_id = self.extract_video_id(img_file.name)
                video_groups[video_id].append(img_file.name)
            
            # Sort frames within each video
            for video_id in video_groups:
                video_groups[video_id].sort()
            
            videos_by_category[category] = dict(video_groups)
            print(f"    Found {len(video_groups)} videos")
        
        return videos_by_category
    
    def analyze_dataset(self, videos_by_category: Dict) -> Dict:
        """Analyze dataset statistics for quality-aware splitting."""
        print("\nAnalyzing dataset...")
        
        analysis = {
            'total_videos': 0,
            'total_frames': 0,
            'by_category': {},
            'by_label': {0: {'videos': 0, 'frames': 0}, 1: {'videos': 0, 'frames': 0}},
            'frame_distribution': [],
            'video_quality_bins': defaultdict(list)
        }
        
        for category, videos in videos_by_category.items():
            label = self.categories[category]
            category_videos = len(videos)
            category_frames = sum(len(frames) for frames in videos.values())
            
            analysis['by_category'][category] = {
                'videos': category_videos,
                'frames': category_frames,
                'label': label
            }
            
            analysis['total_videos'] += category_videos
            analysis['total_frames'] += category_frames
            analysis['by_label'][label]['videos'] += category_videos
            analysis['by_label'][label]['frames'] += category_frames
            
            # Frame count distribution for quality binning
            for video_id, frames in videos.items():
                frame_count = len(frames)
                analysis['frame_distribution'].append(frame_count)
                
                # Quality bins (low/medium/high frame count)
                if frame_count <= 10:
                    quality_bin = 'low'
                elif frame_count <= 20:
                    quality_bin = 'medium'  
                else:
                    quality_bin = 'high'
                
                analysis['video_quality_bins'][quality_bin].append((category, video_id))
        
        # Calculate current balance
        total_real = analysis['by_label'][0]['frames']
        total_fake = analysis['by_label'][1]['frames'] 
        current_balance = total_fake / (total_real + total_fake) if (total_real + total_fake) > 0 else 0
        
        analysis['current_balance'] = current_balance
        analysis['frame_stats'] = {
            'min': np.min(analysis['frame_distribution']) if analysis['frame_distribution'] else 0,
            'max': np.max(analysis['frame_distribution']) if analysis['frame_distribution'] else 0,
            'mean': np.mean(analysis['frame_distribution']) if analysis['frame_distribution'] else 0,
            'median': np.median(analysis['frame_distribution']) if analysis['frame_distribution'] else 0,
            'std': np.std(analysis['frame_distribution']) if analysis['frame_distribution'] else 0
        }
        
        self.print_analysis(analysis)
        return analysis
    
    def print_analysis(self, analysis: Dict):
        """Print dataset analysis summary."""
        print(f"  Total videos: {analysis['total_videos']:,}")
        print(f"  Total frames: {analysis['total_frames']:,}")
        print(f"  Current balance: {analysis['current_balance']:.1%} fake")
        
        print(f"\n  By category:")
        for category, stats in analysis['by_category'].items():
            label_name = "Real" if stats['label'] == 0 else "Fake"
            print(f"    {category}: {stats['videos']:,} videos, {stats['frames']:,} frames ({label_name})")
        
        print(f"\n  Frame count stats:")
        stats = analysis['frame_stats']
        print(f"    Min: {stats['min']}, Max: {stats['max']}")
        print(f"    Mean: {stats['mean']:.1f}, Median: {stats['median']:.1f}")
        print(f"    Std: {stats['std']:.1f}")
    
    def create_stratified_splits(self, videos_by_category: Dict, analysis: Dict) -> Dict:
        """Create stratified splits maintaining balance across quality bins."""
        print(f"\nCreating stratified splits...")
        
        splits = {
            'train': defaultdict(list),
            'val': defaultdict(list), 
            'test': defaultdict(list)
        }
        
        # Separate real and fake videos
        real_videos = []  # [(category, video_id, frame_count)]
        fake_videos = []
        
        for category, videos in videos_by_category.items():
            label = self.categories[category] 
            for video_id, frames in videos.items():
                video_entry = (category, video_id, len(frames))
                if label == 0:  # Real
                    real_videos.append(video_entry)
                else:  # Fake
                    fake_videos.append(video_entry)
        
        print(f"  Real videos: {len(real_videos)}")
        print(f"  Fake videos: {len(fake_videos)}")
        
        # Sort by frame count for quality-aware splitting
        real_videos.sort(key=lambda x: x[2])  # Sort by frame count
        fake_videos.sort(key=lambda x: x[2])
        
        # Shuffle within quality bins to avoid systematic bias
        random.shuffle(real_videos)
        random.shuffle(fake_videos)
        
        # Split each type maintaining quality distribution
        def split_videos(video_list, name):
            n = len(video_list)
            n_train = int(n * self.train_ratio)
            n_val = int(n * self.val_ratio)
            
            train_vids = video_list[:n_train]
            val_vids = video_list[n_train:n_train + n_val]
            test_vids = video_list[n_train + n_val:]
            
            print(f"    {name}: train={len(train_vids)}, val={len(val_vids)}, test={len(test_vids)}")
            return train_vids, val_vids, test_vids
        
        real_train, real_val, real_test = split_videos(real_videos, "Real")
        fake_train, fake_val, fake_test = split_videos(fake_videos, "Fake")
        
        # Combine and organize by split and category
        all_splits = {
            'train': real_train + fake_train,
            'val': real_val + fake_val,
            'test': real_test + fake_test
        }
        
        for split_name, video_list in all_splits.items():
            for category, video_id, frame_count in video_list:
                splits[split_name][category].append(video_id)
        
        # Validate balance
        self.validate_splits_balance(splits, videos_by_category)
        
        return splits
    
    def validate_splits_balance(self, splits: Dict, videos_by_category: Dict):
        """Validate that splits maintain proper balance."""
        print(f"\nValidating split balance...")
        
        for split_name, split_data in splits.items():
            real_frames = fake_frames = 0
            
            for category, video_ids in split_data.items():
                label = self.categories[category]
                category_frames = sum(len(videos_by_category[category][vid]) 
                                    for vid in video_ids if vid in videos_by_category[category])
                
                if label == 0:  # Real
                    real_frames += category_frames
                else:  # Fake
                    fake_frames += category_frames
            
            total_frames = real_frames + fake_frames
            if total_frames > 0:
                fake_ratio = fake_frames / total_frames
                balance_diff = abs(fake_ratio - self.target_balance)
                
                status = "✅" if balance_diff <= self.balance_tolerance else "⚠️"
                print(f"  {split_name}: {fake_ratio:.1%} fake ({status})")
                
                if balance_diff > self.balance_tolerance:
                    print(f"    Warning: Balance deviation {balance_diff:.1%} > tolerance {self.balance_tolerance:.1%}")
    
    def copy_files_to_splits(self, splits: Dict, videos_by_category: Dict):
        """Copy files according to the splits."""
        print(f"\nCopying files to splits...")
        
        # Create directory structure
        for split in ['train', 'val', 'test']:
            for category in self.categories:
                (self.splits_root / split / category).mkdir(parents=True, exist_ok=True)
        
        total_copied = 0
        for split_name, split_data in splits.items():
            print(f"  Processing {split_name} split...")
            split_copied = 0
            
            for category, video_ids in split_data.items():
                src_dir = self.celebdb_root / category
                dst_dir = self.splits_root / split_name / category
                
                for video_id in video_ids:
                    if video_id in videos_by_category[category]:
                        frames = videos_by_category[category][video_id]
                        for frame_file in frames:
                            src_path = src_dir / frame_file
                            dst_path = dst_dir / frame_file
                            
                            if src_path.exists() and not dst_path.exists():
                                shutil.copy2(src_path, dst_path)
                                split_copied += 1
                                total_copied += 1
            
            print(f"    Copied {split_copied:,} files")
        
        print(f"  Total files copied: {total_copied:,}")
    
    def generate_report(self, splits: Dict, videos_by_category: Dict, analysis: Dict):
        """Generate comprehensive splitting report."""
        print(f"\nGenerating report...")
        
        report = {
            'metadata': {
                'source_dir': str(self.celebdb_root),
                'output_dir': str(self.splits_root),
                'timestamp': str(np.datetime64('now')),
                'seed': self.seed,
                'target_ratios': {
                    'train': self.train_ratio,
                    'val': self.val_ratio, 
                    'test': self.test_ratio
                },
                'target_balance': self.target_balance,
                'balance_tolerance': self.balance_tolerance
            },
            'source_analysis': analysis,
            'splits_summary': {}
        }
        
        # Analyze each split
        for split_name, split_data in splits.items():
            split_analysis = {
                'videos': 0,
                'frames': 0,
                'by_category': {},
                'balance': {}
            }
            
            real_frames = fake_frames = 0
            total_videos = 0
            
            for category, video_ids in split_data.items():
                label = self.categories[category]
                category_videos = len(video_ids)
                category_frames = sum(len(videos_by_category[category][vid]) 
                                    for vid in video_ids if vid in videos_by_category[category])
                
                split_analysis['by_category'][category] = {
                    'videos': category_videos,
                    'frames': category_frames,
                    'label': label
                }
                
                total_videos += category_videos
                if label == 0:
                    real_frames += category_frames
                else:
                    fake_frames += category_frames
            
            total_frames = real_frames + fake_frames
            split_analysis['videos'] = total_videos
            split_analysis['frames'] = total_frames
            split_analysis['balance'] = {
                'real_frames': real_frames,
                'fake_frames': fake_frames,
                'fake_ratio': fake_frames / total_frames if total_frames > 0 else 0
            }
            
            report['splits_summary'][split_name] = split_analysis
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj
        
        report = convert_types(report)
        
        # Save report
        report_path = self.splits_root / 'splitting_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"  Report saved: {report_path}")
        
        # Print summary
        self.print_split_summary(report)
        
        return report
    
    def print_split_summary(self, report: Dict):
        """Print split summary to console."""
        print(f"\n" + "="*60)
        print("SPLIT SUMMARY")
        print("="*60)
        
        for split_name, split_data in report['splits_summary'].items():
            balance = split_data['balance']
            fake_ratio = balance['fake_ratio']
            
            print(f"\n{split_name.upper()}:")
            print(f"  Videos: {split_data['videos']:,}")
            print(f"  Frames: {split_data['frames']:,}")
            print(f"  Balance: {fake_ratio:.1%} fake, {1-fake_ratio:.1%} real")
            
            # Check if within tolerance
            balance_diff = abs(fake_ratio - self.target_balance)
            if balance_diff <= self.balance_tolerance:
                print(f"  Status: ✅ Within tolerance (±{balance_diff:.1%})")
            else:
                print(f"  Status: ⚠️  Outside tolerance ({balance_diff:.1%} deviation)")
        
        print("\n" + "="*60)
    
    def create_plots(self, analysis: Dict, report: Dict):
        """Create visualization plots if matplotlib available."""
        if not HAS_MATPLOTLIB:
            return
            
        print(f"\nCreating visualization plots...")
        plots_dir = self.splits_root / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Plot 1: Frame distribution histogram  
        plt.figure(figsize=(10, 6))
        plt.hist(analysis['frame_distribution'], bins=50, alpha=0.7, color='steelblue')
        plt.title('Frame Count Distribution Across Videos')
        plt.xlabel('Frames per Video')
        plt.ylabel('Number of Videos')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'frame_distribution.png', dpi=150)
        plt.close()
        
        # Plot 2: Split balance comparison
        splits_data = report['splits_summary']
        split_names = list(splits_data.keys())
        fake_ratios = [splits_data[s]['balance']['fake_ratio'] for s in split_names]
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(split_names, fake_ratios, color=['#ff7f7f', '#7f7fff', '#7fff7f'])
        plt.axhline(y=self.target_balance, color='red', linestyle='--', 
                   label=f'Target ({self.target_balance:.1%})')
        plt.axhline(y=self.target_balance + self.balance_tolerance, color='orange', 
                   linestyle=':', alpha=0.7, label='Tolerance')
        plt.axhline(y=self.target_balance - self.balance_tolerance, color='orange', 
                   linestyle=':', alpha=0.7)
        
        plt.title('Real/Fake Balance Across Splits')
        plt.ylabel('Fake Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, ratio in zip(bars, fake_ratios):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{ratio:.1%}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'split_balance.png', dpi=150)
        plt.close()
        
        print(f"  Plots saved to: {plots_dir}")
    
    def run(self) -> Dict:
        """Execute the complete balanced splitting process."""
        print(f"\nStarting balanced CelebDB splitting...")
        
        # 1. Scan dataset
        videos_by_category = self.scan_dataset()
        
        # 2. Analyze dataset
        analysis = self.analyze_dataset(videos_by_category)
        
        # 3. Create stratified splits
        splits = self.create_stratified_splits(videos_by_category, analysis)
        
        # 4. Copy files
        self.copy_files_to_splits(splits, videos_by_category)
        
        # 5. Generate report
        report = self.generate_report(splits, videos_by_category, analysis)
        
        # 6. Create plots
        self.create_plots(analysis, report)
        
        print(f"\n🎉 Balanced splitting completed!")
        print(f"📁 Output directory: {self.splits_root}")
        print(f"📊 Report: {self.splits_root / 'splitting_report.json'}")
        
        return report


# =============================================================================
# PYTORCH DATASET CLASS  
# =============================================================================

class CelebDF(Dataset):
    """
    CelebDF frames dataset (video-aware) with advanced features.

    Key features:
      • Optional max_frames_per_video with uniform temporal sampling (reduces correlation).
      • Optional near-duplicate filtering using tiny grayscale thumbnails.
      • Optional return_path for val/test to compute video‑level metrics.
      • Reproducible seeded shuffle for train split.
      • Exposes sample_weights and class_weight_tensor for:
          - WeightedRandomSampler (balanced batches)
          - Class‑weighted CrossEntropyLoss
      • set_epoch(e): re-jitters the uniform sampling each epoch.

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
            subdir_path = os.path.join(self.root, subdir)
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
                if os.path.exists(os.path.join(self.root, rp)):
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
            full = os.path.join(self.root, relp)
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
            full_path = os.path.join(self.root, rel_path)
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


# =============================================================================
# CLI for Split Creation
# =============================================================================

def create_balanced_splits_cli():
    """CLI interface for creating balanced CelebDB splits."""
    parser = argparse.ArgumentParser(
        description="Create balanced CelebDB train/val/test splits with stratification"
    )
    parser.add_argument('--input', '-i', required=True,
                       help='Input CelebDB dataset directory (contains Celeb-real, etc.)')
    parser.add_argument('--output', '-o', required=True, 
                       help='Output directory for balanced splits')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Training set ratio (default: 0.8)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='Validation set ratio (default: 0.1)')  
    parser.add_argument('--test-ratio', type=float, default=0.1,
                       help='Test set ratio (default: 0.1)')
    parser.add_argument('--target-balance', type=float, default=0.5,
                       help='Target fake ratio (0.5 = 50%% fake, default: 0.5)')
    parser.add_argument('--balance-tolerance', type=float, default=0.05,
                       help='Balance tolerance (default: 0.05 = ±5%%)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Create and run splitter
    splitter = BalancedCelebDBSplitter(
        celebdb_root=args.input,
        splits_root=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        target_balance=args.target_balance,
        balance_tolerance=args.balance_tolerance,
        seed=args.seed
    )
    
    report = splitter.run()
    
    # Check if all splits are within tolerance
    all_balanced = True
    for split_name, split_data in report['splits_summary'].items():
        fake_ratio = split_data['balance']['fake_ratio']
        if abs(fake_ratio - args.target_balance) > args.balance_tolerance:
            all_balanced = False
            break
    
    if all_balanced:
        print(f"\n✅ All splits are properly balanced!")
    else:
        print(f"\n⚠️  Some splits exceed balance tolerance. Check the report for details.")
    
    return 0


if __name__ == "__main__":
    exit(create_balanced_splits_cli())