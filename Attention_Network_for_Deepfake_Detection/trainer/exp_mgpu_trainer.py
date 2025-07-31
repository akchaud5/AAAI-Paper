# trainer/exp_mgpu_trainer.py
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import re
import glob
import random

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from trainer.abstract_trainer import AbstractTrainer
from model.network import Recce
from dataset import CelebDF
from trainer.utils import AverageMeter, AccMeter, AUCMeter
import yaml


class ExpMultiGpuTrainer(AbstractTrainer):
    """
    Frame-based training control:
      - LR scheduler and early-stop are driven by a frame-level metric (default: frame_auc)
      - Video-level metrics are computed/printed for reference ONLY (no effect on LR/stop)
      - ReduceLROnPlateau, EMA-for-eval, balanced sampling, robust resume, deterministic val
    """

    def _initiated_settings(self, model_cfg, data_cfg, config_cfg):
        self.checkpoint_dir = config_cfg.get("save_dir", "./checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.debug = config_cfg.get("debug", False)
        self.resume = config_cfg.get("resume", False)
        self.device = torch.device(config_cfg.get("device", "cpu"))
        self.local_rank = config_cfg.get("local_rank", 0)
        log_dir = config_cfg.get("log_dir", "./logs/tensorboard")
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        # Seeds
        seed = int(config_cfg.get("seed", 42))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Data
        self.train_loader = self._get_data_loader(data_cfg, branch=data_cfg["train_branch"], stage="train")
        self.val_loader   = self._get_data_loader(data_cfg, branch=data_cfg["val_branch"],   stage="val")
        self.log_steps = config_cfg.get("log_steps", 100)

        # Stability toggles
        self.use_ema_eval = bool(config_cfg.get("use_ema_eval", False))
        self.deterministic_val = bool(config_cfg.get("deterministic_val", False))

        # Frame-based monitoring control
        self.monitor_key = str(config_cfg.get("monitor", "frame_auc")).lower()
        self._monitor_mode = "min" if self.monitor_key == "val_loss" else "max"

    def _train_settings(self, model_cfg, data_cfg, config_cfg):
        model_cfg = {k: v for k, v in model_cfg.items() if k != "name"}
        self.model = self.load_model(self.model_name)(**model_cfg)

        if config_cfg.get("channels_last", False):
            self.model = self.model.to(device=self.device, memory_format=torch.channels_last)
        else:
            self.model = self.model.to(self.device)

        if config_cfg.get("compile_model", False):
            try:
                self.model = torch.compile(self.model, mode="max-autotune")
                print("✅ Model compiled for H100 optimization")
            except Exception as e:
                print(f"⚠️ Model compilation failed: {e}")

        if torch.cuda.is_available() and "cuda" in str(self.device):
            if torch.distributed.is_initialized():
                self.model = DDP(self.model, device_ids=[self.local_rank])

        # Optimizer
        if config_cfg.get("fused_adamw", False):
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=config_cfg["optimizer"]["lr"],
                weight_decay=config_cfg["optimizer"]["weight_decay"],
                fused=True,
            )
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config_cfg["optimizer"]["lr"])

        # Class-weighted CE from train distribution
        try:
            labels = [y for _, y in self.train_loader.dataset.images_ids]
            cnt = np.bincount(labels, minlength=2).astype(float)
            cnt[cnt == 0.0] = 1.0
            class_w = (cnt.sum() / (2.0 * cnt))  # inverse-freq normalized
            self._class_weights = torch.as_tensor(class_w, dtype=torch.float32, device=self.device)
            self.criterion = torch.nn.CrossEntropyLoss(weight=self._class_weights)
            print(f"[loss] class weights = {class_w.tolist()}")
        except Exception:
            self.criterion = torch.nn.CrossEntropyLoss()

        # Scheduler
        sch_cfg = config_cfg.get("scheduler", {})
        if str(sch_cfg.get("name", "StepLR")).lower() == "reducelronplateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=sch_cfg.get("mode", "max"),
                factor=sch_cfg.get("factor", 0.5),
                patience=sch_cfg.get("patience", 2),
                threshold=sch_cfg.get("threshold", 0.001),
                threshold_mode=sch_cfg.get("threshold_mode", "abs"),
                min_lr=sch_cfg.get("min_lr", 1e-6),
            )
            self._use_plateau = True
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sch_cfg.get("step_size", 15000),
                gamma=sch_cfg.get("gamma", 0.5),
            )
            self._use_plateau = False

        # EMA model for eval only
        self.ema_model = AveragedModel(self.model) if self.use_ema_eval else None

        # Resume
        self.start_step = 0
        self.start_epoch = 0
        self.best_metric = -1e9 if self._monitor_mode == "max" else 1e9
        if self.resume:
            self._load_ckpt(best=config_cfg.get("resume_best", False), train=True)

        # Early-stop config
        es_cfg = self.config["config"].get("early_stop", {})
        self._es_mode = es_cfg.get("mode", "max")
        self._es_patience = int(es_cfg.get("patience", 4))
        self._es_min_delta = float(es_cfg.get("min_delta", 0.002))
        self._es_k = int(es_cfg.get("smooth_k", 3))
        self._es_hist = []
        self._es_bad_epochs = 0
        self._should_stop = False

        # Perf knobs
        if bool(self.config["config"].get("allow_tf32", True)):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        if bool(self.config["config"].get("cudnn_benchmark", True)):
            torch.backends.cudnn.benchmark = True

    def _test_settings(self, model_cfg, data_cfg, config_cfg):
        model_cfg = {k: v for k, v in model_cfg.items() if k != "name"}
        self.model = self.load_model(self.model_name)(**model_cfg).to(self.device)
        self._load_ckpt(best=True)

    def _save_ckpt(self, step, epoch, best=False):
        state = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "step": step,
            "epoch": epoch,
            "best_metric": self.best_metric,
        }
        filename = os.path.join(self.checkpoint_dir, f"step_{step}_model.pt" if not best else "best_model.pt")
        torch.save(state, filename)

    def _load_ckpt(self, best=False, train=False):
        if not best:
            filename = self.config.get("ckpt", "")
            if filename:
                checkpoint_file = filename
            else:
                checkpoint_file = max(
                    glob.glob(os.path.join(self.checkpoint_dir, "*.pt")),
                    key=os.path.getctime,
                    default=None,
                )
        else:
            checkpoint_file = os.path.join(self.checkpoint_dir, "best_model.pt")

        if not checkpoint_file or not os.path.exists(checkpoint_file):
            if train:
                print(f"No checkpoint found at {checkpoint_file}. Starting training from scratch.")
            return

        print(f"Loading checkpoint from {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])

        if train:
            if "optimizer_state" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            if "scheduler_state" in checkpoint:
                try:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state"])
                except Exception:
                    print("Warning: 'scheduler_state' could not be loaded. Scheduler will start from scratch.")
            self.start_step = checkpoint.get("step", 0) + 1
            self.start_epoch = checkpoint.get("epoch", 0)
            # keep best_metric if present; else leave initialized
            self.best_metric = checkpoint.get("best_metric", self.best_metric)

        print(f"Checkpoint loaded. Resuming from epoch {self.start_epoch} and step {self.start_step}.")

    def train(self):
        print("********** Training begins...... **********")
        loss_meter = AverageMeter()
        acc_meter = AccMeter()
        auc_meter = AUCMeter()

        steps_per_epoch = len(self.train_loader)
        base_epochs = self.config["config"].get("epochs", 25)
        if self.resume and self.config["config"].get("resume_best", False):
            extra = int(self.config["config"].get("extra_epochs", 0))
            total_epochs = self.start_epoch + max(extra, 0)
            print(f"[trainer] Resuming from best at epoch {self.start_epoch}; training {extra} more (to {total_epochs}).")
        else:
            total_epochs = base_epochs

        target_step = total_epochs * steps_per_epoch
        if self.start_step >= target_step:
            print(f"Training already completed. Target steps ({target_step}) reached.")
            return

        global_step = self.start_step
        processed_epochs = self.start_step // steps_per_epoch
        start_epoch_loop = max(self.start_epoch, processed_epochs)

        for epoch in range(start_epoch_loop, total_epochs):
            if hasattr(self.train_loader.dataset, "set_epoch"):
                self.train_loader.dataset.set_epoch(epoch)

            self.model.train()
            auc_meter.reset()
            loss_meter.reset()
            acc_meter.reset()
            print(f"Starting epoch {epoch + 1}...")

            for step, (I, Y) in enumerate(self.train_loader):
                global_step = epoch * steps_per_epoch + step
                if global_step < self.start_step:
                    continue

                if hasattr(self, "config") and self.config["config"].get("channels_last", False):
                    in_I = I.to(self.device, non_blocking=True, memory_format=torch.channels_last)
                else:
                    in_I = I.to(self.device, non_blocking=True)
                Y = Y.to(self.device, non_blocking=True).long()

                Y_pre = self.model(in_I)
                loss = self.criterion(Y_pre, Y)
                loss_meter.update(loss.item(), I.size(0))

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                if self.ema_model is not None:
                    self.ema_model.update_parameters(self.model)

                acc_meter.update(Y_pre, Y)
                auc_meter.update(Y_pre, Y)

                if global_step % self.log_steps == 0:
                    self.writer.add_scalar("Loss/train", loss_meter.avg, global_step)
                    self.writer.add_scalar("Accuracy/train", acc_meter.mean_acc(), global_step)
                    self.writer.add_scalar("AUC/train", auc_meter.compute_auc(), global_step)
                    self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], global_step)
                    print(
                        f"Epoch [{epoch+1}/{total_epochs}], Step [{step}/{steps_per_epoch}], "
                        f"Global Step [{global_step}], "
                        f"Loss: {loss_meter.avg:.4f}, Acc: {acc_meter.mean_acc():.4f}, "
                        f"AUC: {auc_meter.compute_auc():.4f}, "
                        f"LR: {self.optimizer.param_groups[0]['lr']:.6g}"
                    )

                if global_step % 1000 == 0:
                    self._save_ckpt(global_step, epoch)

            if not self._use_plateau:
                self.scheduler.step()

            print(f"Epoch {epoch + 1}/{total_epochs} finished. Starting validation...")
            self.validate(epoch, global_step)
            print(f"Validation for Epoch {epoch + 1} completed.")
            self._save_ckpt(global_step, epoch)
            self.start_step = 0

            if getattr(self, "_should_stop", False):
                print("[early-stop] stopping training due to convergence.")
                break

    def validate(self, epoch, step):
        print("********** Validation begins...... **********")
        prev_bench = torch.backends.cudnn.benchmark
        if self.deterministic_val:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass

        eval_model = self.ema_model if self.ema_model is not None else self.model
        eval_model.eval()

        loss_meter = AverageMeter()
        acc_meter = AccMeter()
        auc_meter = AUCMeter()
        auc_meter.reset()

        all_logits, all_labels, all_paths = [], [], []

        with torch.no_grad():
            for val_step, batch in enumerate(self.val_loader, 1):
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    I, Y, P = batch
                else:
                    I, Y = batch
                    P = None

                in_I, Y = self.to_device((I, Y))
                Y = Y.long()
                Y_pre = eval_model(in_I)
                loss = self.criterion(Y_pre, Y)

                loss_meter.update(loss.item(), I.size(0))
                acc_meter.update(Y_pre, Y)
                auc_meter.update(Y_pre, Y)

                all_logits.append(Y_pre.cpu())
                all_labels.append(Y.cpu())
                if P is not None:
                    all_paths.extend(list(P))

                self.writer.add_scalar("Loss/val", loss_meter.avg, step + val_step)
                self.writer.add_scalar("Accuracy/val", acc_meter.mean_acc(), step + val_step)
                self.writer.add_scalar("AUC/val", auc_meter.compute_auc(), step + val_step)

        logits = torch.cat(all_logits)
        y_true = torch.cat(all_labels).numpy()
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        preds = (probs >= 0.5).astype(int)

        # Frame-level metrics
        frame_acc = accuracy_score(y_true, preds)
        try:
            frame_auc = roc_auc_score(y_true, probs)
        except ValueError:
            frame_auc = float("nan")
        bal_acc = balanced_accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, zero_division=0)

        # Metrics dict for monitoring
        metrics = {
            "val_loss": float(loss_meter.avg),
            "frame_acc": float(frame_acc),
            "frame_auc": float(frame_auc) if np.isfinite(frame_auc) else float("nan"),
        }

        # Video-level metrics (side info only)
        video_auc = float("nan")
        video_acc = float("nan")
        if len(all_paths) == len(y_true) and len(all_paths) > 0:
            def vid_from_name(p):
                b = os.path.splitext(os.path.basename(p))[0]
                for pat in (r"^(id\d+_id\d+)", r"^(id\d+)", r"^(\d+)", r"^([^_]+)"):
                    m = re.match(pat, b)
                    if m:
                        return m.group(1)
                return re.sub(r"_\d+$", "", b)

            from collections import defaultdict
            bucket = defaultdict(list)
            vlabel = {}
            for p, y, pr in zip(all_paths, y_true, probs):
                v = vid_from_name(p)
                bucket[v].append(pr)
                vlabel[v] = y

            vids = list(bucket.keys())
            vy_true = np.array([vlabel[v] for v in vids])
            vy_score = np.array([np.mean(bucket[v]) for v in vids])
            vy_pred = (vy_score >= 0.5).astype(int)

            video_acc = accuracy_score(vy_true, vy_pred)
            try:
                video_auc = roc_auc_score(vy_true, vy_score)
            except ValueError:
                video_auc = float("nan")

        print(f"Validation: Epoch [{epoch}], Step [{step}]")
        print(
            f"Loss: {loss_meter.avg:.4f} | "
            f"Frame Acc: {frame_acc:.4f} AUC: {frame_auc:.4f} | "
            f"Video Acc: {video_acc:.4f} AUC: {video_auc:.4f} | "
            f"Balanced Acc: {bal_acc:.4f} F1: {f1:.4f}"
        )

        # Choose monitored metric (strictly frame-based as configured)
        monitored = metrics.get(self.monitor_key, float("nan"))
        if not np.isfinite(monitored):
            for key in ("frame_auc", "frame_acc", "val_loss"):
                if np.isfinite(metrics.get(key, float("nan"))):
                    monitored = metrics[key]
                    break

        # LR schedule
        if self._use_plateau and np.isfinite(monitored):
            self.scheduler.step(monitored)

        # Early stopping on smoothed monitored metric
        self._es_hist.append(monitored if np.isfinite(monitored) else (-1e9 if self._monitor_mode=="max" else 1e9))
        smoothed = float(np.median(self._es_hist[-self._es_k :]))
        improved = (
            smoothed > self.best_metric + self._es_min_delta if self._monitor_mode == "max"
            else smoothed < self.best_metric - self._es_min_delta
        )
        if improved:
            self.best_metric = smoothed
            self.best_step = epoch
            self._save_ckpt(step, epoch, best=True)
            if self.ema_model is not None:
                ema_state = self.ema_model.module.state_dict() if hasattr(self.ema_model, "module") else self.ema_model.state_dict()
                torch.save({"model_state": ema_state, "epoch": epoch, "best_metric": self.best_metric},
                           os.path.join(self.checkpoint_dir, "best_model_ema.pt"))
            self._es_bad_epochs = 0
        else:
            self._es_bad_epochs += 1
            print(
                f"[early-stop] no improvement ({self._es_bad_epochs}/{self._es_patience}) "
                f"smoothed {smoothed:.4f} best {self.best_metric:.4f} (monitor={self.monitor_key})"
            )

        min_epochs = int(self.config["config"].get("early_stop", {}).get("min_epochs", 0))
        self._should_stop = (epoch + 1) >= min_epochs and (self._es_bad_epochs >= self._es_patience)

        if self.deterministic_val:
            torch.backends.cudnn.benchmark = prev_bench

    def load_model(self, model_name):
        models = {"Recce": Recce}
        return models[model_name]

    def test(self):
        print("********** Testing begins...... **********")
        eval_model = self.ema_model if self.ema_model is not None else self.model
        eval_model.eval()
        loss_meter = AverageMeter()
        acc_meter = AccMeter()
        auc_meter = AUCMeter()

        with torch.no_grad():
            for test_step, batch in enumerate(self.test_loader, 1):
                I, Y = batch[:2]
                in_I, Y = self.to_device((I, Y))
                Y_pre = eval_model(in_I)
                loss = self.criterion(Y_pre, Y)
                loss_meter.update(loss.item(), I.size(0))
                acc_meter.update(Y_pre, Y)
                auc_meter.update(Y_pre, Y)

        print(f"Test: Loss: {loss_meter.avg:.4f}, Acc: {acc_meter.mean_acc():.4f}, AUC: {auc_meter.compute_auc():.4f}")

    def _get_data_loader(self, cfg, branch, stage):
        with open(cfg["file"], "r") as file:
            data_cfg = yaml.safe_load(file)

        branch_cfg = dict(data_cfg[branch])
        branch_cfg["split"] = stage

        print(f"Loading data for {stage}...")
        dataset = CelebDF(branch_cfg)
        batch_size = cfg[f"{stage}_batch_size"]

        # deterministic worker seeds
        seed = int(self.config["config"].get("seed", 42))
        def _worker_init_fn(worker_id):
            import numpy as _np, random as _rnd
            _rnd.seed(seed + worker_id)
            _np.random.seed(seed + worker_id)

        sampler = None
        shuffle = (stage == "train")
        if stage == "train" and branch_cfg.get("balance", False):
            labels = [y for _, y in dataset.images_ids]
            cnt = np.bincount(labels, minlength=2).astype(float)
            cnt[cnt == 0.0] = 1.0
            inv = 1.0 / cnt
            weights = np.array([inv[y] for y in labels], dtype=np.float64)
            sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
            shuffle = False

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=cfg.get("num_workers", 4),
            pin_memory=cfg.get("pin_memory", True),
            persistent_workers=cfg.get("persistent_workers", True),
            prefetch_factor=cfg.get("prefetch_factor", 4),
            worker_init_fn=_worker_init_fn,
            generator=torch.Generator().manual_seed(seed),
            drop_last=False,
        )
        print(f"Data for {stage} loaded. Batch size: {batch_size} (sampler={'balanced' if sampler is not None else 'none'})")
        return loader
