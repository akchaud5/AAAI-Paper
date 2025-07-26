#!/usr/bin/env python3
"""
H100 Optimized Training Script for CelebDF-v2
Optimized for maximum throughput and efficiency on H100 GPUs
"""

import os
import yaml
import argparse
import torch
import torch.distributed as dist
from trainer.h100_optimized_trainer import H100OptimizedTrainer
from trainer.utils import setup_for_distributed, cleanup

def parse_arguments():
    parser = argparse.ArgumentParser(description="H100 Optimized Training for Deepfake Detection")
    parser.add_argument("--config", type=str, 
                       default="./config/H100_CelebDF_optimized.yml",
                       help="Path to the H100 optimized configuration file")
    parser.add_argument("--local_rank", type=int, default=0,
                       help="Local rank for distributed training")
    parser.add_argument("--world_size", type=int, default=1,
                       help="World size for distributed training")
    parser.add_argument("--master_addr", type=str, default="localhost",
                       help="Master address for distributed training")
    parser.add_argument("--master_port", type=str, default="12355",
                       help="Master port for distributed training")
    parser.add_argument("--mixed_precision", action="store_true", default=True,
                       help="Enable mixed precision training")
    parser.add_argument("--compile", action="store_true", default=True,
                       help="Enable torch.compile for model optimization")
    parser.add_argument("--profile", action="store_true", default=False,
                       help="Enable profiling for performance analysis")
    return parser.parse_args()

def setup_distributed_training(args):
    """Setup distributed training environment"""
    if args.world_size > 1:
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = args.master_port
        os.environ['RANK'] = str(args.local_rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        
        # Initialize distributed training
        dist.init_process_group(
            backend='nccl',
            world_size=args.world_size,
            rank=args.local_rank
        )
        
        # Set CUDA device for this process
        torch.cuda.set_device(args.local_rank)
        
        print(f"Distributed training initialized: rank {args.local_rank}/{args.world_size}")
    else:
        print("Single GPU training")

def optimize_cuda_settings():
    """Optimize CUDA settings for H100"""
    if torch.cuda.is_available():
        # Enable TensorFloat-32 (TF32) for improved performance on H100
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cudnn benchmarking for consistent input sizes
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Optimize memory allocation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        print("CUDA optimizations enabled:")
        print(f"  - TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"  - cuDNN benchmark: {torch.backends.cudnn.benchmark}")
        print(f"  - Available GPUs: {torch.cuda.device_count()}")
        print(f"  - Current device: {torch.cuda.current_device()}")
        print(f"  - Device name: {torch.cuda.get_device_name()}")
        print(f"  - Device memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

def load_and_validate_config(config_path):
    """Load and validate configuration"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Load dataset-specific configuration
    if 'data' in config and 'file' in config['data']:
        dataset_config_path = config['data']['file']
        if not os.path.exists(dataset_config_path):
            print(f"Warning: Dataset config file not found: {dataset_config_path}")
            print("Please update the path in the config file")
        else:
            with open(dataset_config_path, 'r') as file:
                dataset_config = yaml.safe_load(file)
            config.update(dataset_config)
    
    return config

def setup_profiling(args):
    """Setup PyTorch profiler if enabled"""
    if args.profile:
        from torch.profiler import profile, record_function, ProfilerActivity
        
        profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        return profiler
    return None

def main():
    args = parse_arguments()
    
    print("="*70)
    print("H100 Optimized Training for Deepfake Detection")
    print("="*70)
    
    # Setup distributed training if needed
    setup_distributed_training(args)
    
    # Optimize CUDA settings for H100
    optimize_cuda_settings()
    
    # Load and validate configuration
    print(f"Loading configuration from: {args.config}")
    config = load_and_validate_config(args.config)
    
    # Override config with command line arguments
    config.setdefault('config', {})
    config['config']['local_rank'] = args.local_rank
    config['config']['mixed_precision'] = args.mixed_precision
    config['config']['compile_model'] = args.compile
    
    # Ensure device is set correctly
    device = f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu'
    config['config']['device'] = device
    
    print(f"Training configuration:")
    print(f"  - Model: {config['model']['name']}")
    print(f"  - Device: {device}")
    print(f"  - Mixed Precision: {args.mixed_precision}")
    print(f"  - Model Compilation: {args.compile}")
    print(f"  - Batch Size: {config['data']['train_batch_size']}")
    print(f"  - Learning Rate: {config['config']['optimizer']['lr']}")
    print(f"  - Workers: {config['data'].get('num_workers', 12)}")
    
    # Setup profiling if enabled
    profiler = setup_profiling(args)
    
    try:
        # Initialize H100 optimized trainer
        trainer = H100OptimizedTrainer(config, stage="Train")
        
        print("\nStarting H100 optimized training...")
        print("="*70)
        
        if profiler:
            profiler.start()
        
        # Start training
        trainer.train()
        
        if profiler:
            profiler.stop()
            print("Profiling data saved to ./profiler_logs")
        
        print("="*70)
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        if profiler:
            profiler.stop()
        
        # Cleanup distributed training
        if args.world_size > 1:
            cleanup()
        raise e
    
    # Cleanup distributed training
    if args.world_size > 1:
        cleanup()
    
    print("H100 optimized training session completed.")

if __name__ == "__main__":
    main()