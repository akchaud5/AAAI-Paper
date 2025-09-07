#!/bin/bash

# H100 Optimized CelebDB-v2 Training Launch Script
# Maximum performance without compromising results

set -e  # Exit on error

echo "========================================================================"
echo "H100 OPTIMIZED CELEBDB-V2 DEEPFAKE DETECTION TRAINING"
echo "========================================================================"

# Environment setup
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64,roundup_power2_divisions:16"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# Check if running on H100
python3 -c "import torch; print('GPU:', torch.cuda.get_device_name() if torch.cuda.is_available() else 'No GPU')"

# Create necessary directories
mkdir -p checkpoints/h100_celebdb
mkdir -p logs/h100_celebdb_tensorboard

echo "========================================================================"
echo "LAUNCHING H100 TRAINING..."
echo "========================================================================"

# Launch training with optimized settings
python3 train_celebdb_h100.py \
    --config ./config/H100_CelebDF_optimized.yml \
    2>&1 | tee logs/h100_training_$(date +%Y%m%d_%H%M%S).log

echo "========================================================================"
echo "TRAINING COMPLETED!"
echo "========================================================================"

# Display results summary
echo "Results saved to:"
echo "  - Checkpoints: checkpoints/"
echo "  - Logs: logs/h100_celebdb_tensorboard"
echo "  - Training log: logs/h100_training_*.log"

echo ""
echo "To monitor training progress:"
echo "  tensorboard --logdir=logs/h100_celebdb_tensorboard --port=6006"
echo ""
echo "To resume training:"
echo "  python3 train_celebdb_h100.py --config ./config/H100_CelebDF_optimized.yml --resume"