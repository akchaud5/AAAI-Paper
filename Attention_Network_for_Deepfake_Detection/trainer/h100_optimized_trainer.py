import os
import torch
import glob
import time
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from trainer.abstract_trainer import AbstractTrainer
from model.network import Recce
from dataset import CelebDF
from torch.utils.data import DataLoader
from trainer.utils import AverageMeter, AccMeter, AUCMeter
from torch.utils.tensorboard import SummaryWriter
import yaml

class H100OptimizedTrainer(AbstractTrainer):
    def _initiated_settings(self, model_cfg, data_cfg, config_cfg):
        self.checkpoint_dir = "./checkpoints/h100_optimized"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.debug = config_cfg.get('debug', False)
        self.resume = config_cfg.get('resume', False)
        self.device = torch.device(config_cfg.get('device', 'cuda:0'))
        self.local_rank = config_cfg.get('local_rank', 0)
        self.writer = SummaryWriter(log_dir="./logs/h100_tensorboard")
        
        # H100 optimizations
        self.mixed_precision = config_cfg.get('mixed_precision', True)
        self.compile_model = config_cfg.get('compile_model', True)
        self.channels_last = config_cfg.get('channels_last', True)
        self.gradient_clipping = config_cfg.get('gradient_clipping', 1.0)
        self.checkpoint_freq = config_cfg.get('checkpoint_freq', 1000)
        
        # Initialize mixed precision scaler
        if self.mixed_precision:
            self.scaler = GradScaler()
        
        # Data loaders with H100 optimizations
        self.train_loader = self._get_optimized_data_loader(data_cfg, branch=data_cfg['train_branch'], stage='train')
        self.val_loader = self._get_optimized_data_loader(data_cfg, branch=data_cfg['val_branch'], stage='val')
        self.log_steps = config_cfg.get('log_steps', 50)

    def _train_settings(self, model_cfg, data_cfg, config_cfg):
        model_cfg = {k: v for k, v in model_cfg.items() if k != 'name'}
        self.model = self.load_model(self.model_name)(**model_cfg)
        
        # Enable channels_last memory format for H100
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        
        self.model = self.model.to(self.device)
        
        # Compile model for PyTorch 2.0+ on H100
        if self.compile_model and hasattr(torch, 'compile'):
            print("Compiling model for H100 optimization...")
            self.model = torch.compile(self.model)
        
        # Distributed training setup
        if torch.cuda.is_available() and torch.distributed.is_initialized():
            self.model = DDP(self.model, device_ids=[self.local_rank])
        
        # Optimizer with H100-specific settings
        optimizer_cfg = config_cfg['optimizer']
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=optimizer_cfg['lr'],
            weight_decay=optimizer_cfg['weight_decay'],
            betas=optimizer_cfg.get('betas', [0.9, 0.999]),
            eps=optimizer_cfg.get('eps', 1e-8),
            amsgrad=optimizer_cfg.get('amsgrad', False)
        )
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=config_cfg['scheduler']['step_size'], 
            gamma=config_cfg['scheduler']['gamma']
        )
        
        self.start_step = 0
        self.start_epoch = 0
        self.best_metric = 0
        
        if self.resume:
            self._load_ckpt(best=config_cfg.get("resume_best", False), train=True)

    def _get_optimized_data_loader(self, cfg, branch, stage):
        """Create optimized data loader for H100"""
        with open(cfg['file'], 'r') as file:
            data_cfg = yaml.safe_load(file)

        branch_cfg = data_cfg[branch]
        branch_cfg['split'] = stage

        print(f"Loading optimized data for {stage}...")
        dataset = CelebDF(branch_cfg)
        shuffle = stage == 'train'
        batch_size = cfg[f'{stage}_batch_size']
        
        # H100 optimized DataLoader settings
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=cfg.get('num_workers', 12),
            pin_memory=cfg.get('pin_memory', True),
            persistent_workers=cfg.get('persistent_workers', True),
            prefetch_factor=cfg.get('prefetch_factor', 4),
            drop_last=True if stage == 'train' else False
        )
        
        print(f"Optimized data loader created for {stage}. Batch size: {batch_size}, Workers: {cfg.get('num_workers', 12)}")
        return loader

    def train(self):
        print("********** H100 Optimized Training begins...... **********")
        
        # Performance monitoring
        start_time = time.time()
        total_samples = 0
        
        loss_meter = AverageMeter()
        acc_meter = AccMeter()
        auc_meter = AUCMeter()
        steps_per_epoch = len(self.train_loader)
        total_epochs = 10

        if self.start_step >= total_epochs * steps_per_epoch:
            print(f"Training already completed. Target steps reached.")
            return

        global_step = self.start_step

        for epoch in range(self.start_epoch, total_epochs):
            if epoch * steps_per_epoch + self.start_step >= total_epochs * steps_per_epoch:
                print(f"Training completed up to epoch {epoch}.")
                break

            self.model.train()
            epoch_start_time = time.time()
            auc_meter.reset()
            loss_meter.reset()
            acc_meter.reset()

            print(f"Starting epoch {epoch+1}/{total_epochs}...")

            for step, (I, Y) in enumerate(self.train_loader):
                global_step = epoch * steps_per_epoch + step

                if global_step < self.start_step:
                    continue

                # Move data to device with channels_last if enabled
                if self.channels_last:
                    I = I.to(self.device, memory_format=torch.channels_last)
                else:
                    I = I.to(self.device)
                Y = Y.to(self.device).long()

                # Mixed precision training
                if self.mixed_precision:
                    with autocast():
                        Y_pre = self.model(I)
                        loss = self.criterion(Y_pre, Y)
                    
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    if self.gradient_clipping > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard training
                    Y_pre = self.model(I)
                    loss = self.criterion(Y_pre, Y)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    if self.gradient_clipping > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
                    
                    self.optimizer.step()

                # Update metrics
                loss_meter.update(loss.item(), I.size(0))
                acc_meter.update(Y_pre, Y)
                auc_meter.update(Y_pre, Y)
                total_samples += I.size(0)

                # Logging
                if global_step % self.log_steps == 0:
                    elapsed_time = time.time() - start_time
                    samples_per_sec = total_samples / elapsed_time if elapsed_time > 0 else 0
                    
                    self.writer.add_scalar('Loss/train', loss_meter.avg, global_step)
                    self.writer.add_scalar('Accuracy/train', acc_meter.mean_acc(), global_step)
                    self.writer.add_scalar('AUC/train', auc_meter.compute_auc(), global_step)
                    self.writer.add_scalar('Performance/samples_per_sec', samples_per_sec, global_step)

                    print(f"Epoch [{epoch+1}/{total_epochs}], Step [{step}/{steps_per_epoch}], "
                          f"Global Step [{global_step}], Loss: {loss_meter.avg:.4f}, "
                          f"Acc: {acc_meter.mean_acc():.4f}, AUC: {auc_meter.compute_auc():.4f}, "
                          f"Samples/sec: {samples_per_sec:.1f}")

                # Checkpointing
                if global_step % self.checkpoint_freq == 0:
                    self._save_ckpt(global_step, epoch)

            # End of epoch
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1} completed in {epoch_time:.1f}s")
            
            self.scheduler.step()
            print(f"Epoch {epoch+1}/{total_epochs} finished. Starting validation...")
            self.validate(epoch, global_step)
            print(f"Validation for Epoch {epoch+1} completed.")
            
            # Save checkpoint at end of epoch
            self._save_ckpt(global_step, epoch)
            self.start_step = 0

        total_training_time = time.time() - start_time
        print(f"********** Training completed in {total_training_time/3600:.2f} hours **********")
        print(f"Average throughput: {total_samples/total_training_time:.1f} samples/sec")

    def validate(self, epoch, step):
        print("********** H100 Optimized Validation begins...... **********")
        self.model.eval()
        loss_meter = AverageMeter()
        acc_meter = AccMeter()
        auc_meter = AUCMeter()
        auc_meter.reset()

        val_start_time = time.time()
        total_val_samples = 0

        with torch.no_grad():
            for val_step, (I, Y) in enumerate(self.val_loader, 1):
                if self.channels_last:
                    I = I.to(self.device, memory_format=torch.channels_last)
                else:
                    I = I.to(self.device)
                Y = Y.to(self.device).long()

                if self.mixed_precision:
                    with autocast():
                        Y_pre = self.model(I)
                        loss = self.criterion(Y_pre, Y)
                else:
                    Y_pre = self.model(I)
                    loss = self.criterion(Y_pre, Y)

                loss_meter.update(loss.item(), I.size(0))
                acc_meter.update(Y_pre, Y)
                auc_meter.update(Y_pre, Y)
                total_val_samples += I.size(0)

        val_time = time.time() - val_start_time
        val_samples_per_sec = total_val_samples / val_time if val_time > 0 else 0

        print(f"Validation: Epoch [{epoch}], Step [{step}]")
        print(f"Loss: {loss_meter.avg:.4f}, Acc: {acc_meter.mean_acc():.4f}, "
              f"AUC: {auc_meter.compute_auc():.4f}")
        print(f"Validation completed in {val_time:.1f}s, {val_samples_per_sec:.1f} samples/sec")

        # TensorBoard logging
        self.writer.add_scalar('Loss/val', loss_meter.avg, step)
        self.writer.add_scalar('Accuracy/val', acc_meter.mean_acc(), step)
        self.writer.add_scalar('AUC/val', auc_meter.compute_auc(), step)
        self.writer.add_scalar('Performance/val_samples_per_sec', val_samples_per_sec, step)

        if acc_meter.mean_acc() > self.best_metric:
            self.best_metric = acc_meter.mean_acc()
            self.best_step = step
            self._save_ckpt(step, epoch, best=True)
            print(f"New best model saved with accuracy: {self.best_metric:.4f}")

    def _save_ckpt(self, step, epoch, best=False):
        """Save optimized checkpoint"""
        model_state = self.model.state_dict()
        if hasattr(self.model, 'module'):  # DDP
            model_state = self.model.module.state_dict()
            
        state = {
            'model_state': model_state,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'step': step,
            'epoch': epoch,
            'best_metric': self.best_metric,
            'scaler_state': self.scaler.state_dict() if self.mixed_precision else None
        }
        
        filename = os.path.join(self.checkpoint_dir, f'step_{step}_model.pt' if not best else 'best_model.pt')
        torch.save(state, filename)
        print(f"Checkpoint saved: {filename}")

    def _load_ckpt(self, best=False, train=False):
        """Load optimized checkpoint"""
        if not best:
            checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, 'step_*_model.pt'))
            if checkpoint_files:
                checkpoint_file = max(checkpoint_files, key=os.path.getctime)
            else:
                checkpoint_file = None
        else:
            checkpoint_file = os.path.join(self.checkpoint_dir, 'best_model.pt')

        if not checkpoint_file or not os.path.exists(checkpoint_file):
            if train:
                print(f"No checkpoint found. Starting training from scratch.")
            return

        print(f"Loading checkpoint from {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        
        # Load model state
        if hasattr(self.model, 'module'):  # DDP
            self.model.module.load_state_dict(checkpoint['model_state'])
        else:
            self.model.load_state_dict(checkpoint['model_state'])

        if train:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            if 'scheduler_state' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            if self.mixed_precision and 'scaler_state' in checkpoint and checkpoint['scaler_state']:
                self.scaler.load_state_dict(checkpoint['scaler_state'])
            
            self.start_step = checkpoint['step'] + 1
            self.start_epoch = checkpoint.get('epoch', 0)
            self.best_metric = checkpoint['best_metric']

        print(f"Checkpoint loaded. Resuming from epoch {self.start_epoch} and step {self.start_step}.")

    def load_model(self, model_name):
        models = {
            'Recce': Recce
        }
        return models[model_name]