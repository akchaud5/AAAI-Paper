#!/usr/bin/env python3
"""
Production-Ready H100 Training Script for CelebDB-v2
Uses the advanced exp_mgpu_trainer.py with full H100 optimizations
"""

import os
import sys
import yaml
import argparse
from trainer.exp_mgpu_trainer import ExpMultiGpuTrainer

def parse_arguments():
    parser = argparse.ArgumentParser(description="H100 Optimized CelebDB Training")
    parser.add_argument("--config", type=str, 
                       default="./config/H100_CelebDF_optimized.yml",
                       help="Configuration file path")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint")
    parser.add_argument("--resume_best", action="store_true",
                       help="Resume from best checkpoint")
    parser.add_argument("--local_rank", type=int, default=0,
                       help="Local rank for distributed training")
    return parser.parse_args()

def validate_environment():
    """Validate H100 environment and optimizations"""
    import torch
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    
    device_name = torch.cuda.get_device_name()
    if "H100" not in device_name:
        print(f"⚠️  Warning: Not running on H100 (detected: {device_name})")
        print("   Performance may not be optimal")
    else:
        print(f"✅ Running on {device_name}")
    
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU Memory: {memory_gb:.1f} GB")
    
    # Check PyTorch version for compile support
    if hasattr(torch, 'compile'):
        print("✅ PyTorch 2.0+ compile support available")
    else:
        print("⚠️  PyTorch compile not available (need PyTorch 2.0+)")

def load_config(config_path, args):
    """Load and update configuration"""
    print(f"Loading configuration: {config_path}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with command line arguments
    if 'config' not in config:
        config['config'] = {}
    
    config['config']['resume'] = args.resume
    config['config']['resume_best'] = args.resume_best
    config['config']['local_rank'] = args.local_rank
    
    return config

def main():
    print("="*80)
    print("H100 OPTIMIZED CELEBDB-V2 DEEPFAKE DETECTION TRAINING")
    print("Using Advanced ExpMultiGpuTrainer with Full H100 Optimizations")
    print("="*80)
    
    args = parse_arguments()
    
    # Validate environment
    validate_environment()
    
    # Load configuration
    config = load_config(args.config, args)
    
    # Print training summary
    print(f"\n📋 Training Configuration:")
    print(f"   Model: {config['model']['name']}")
    print(f"   Batch Size: {config['data']['train_batch_size']}")
    print(f"   Learning Rate: {config['config']['optimizer']['lr']}")
    print(f"   Epochs: {config['config']['epochs']}")
    print(f"   Workers: {config['data']['num_workers']}")
    print(f"   Mixed Precision: {config['config'].get('use_ema_eval', 'N/A')}")
    print(f"   Model Compile: {config['config'].get('compile_model', 'N/A')}")
    print(f"   EMA Evaluation: {config['config'].get('use_ema_eval', 'N/A')}")
    print(f"   Monitor Metric: {config['config'].get('monitor', 'N/A')}")
    
    # Initialize trainer
    print(f"\n🚀 Initializing ExpMultiGpuTrainer...")
    trainer = ExpMultiGpuTrainer(config, stage="Train")
    
    print(f"\n🎯 Starting H100-optimized training...")
    print("="*80)
    
    try:
        trainer.train()
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()