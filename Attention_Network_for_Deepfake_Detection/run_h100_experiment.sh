#!/bin/bash

# H100 Optimized Training and Testing Pipeline
# Complete experiment: Train on CelebDF-v2, Test on FaceForensics++

set -e  # Exit on any error

echo "============================================================"
echo "H100 Optimized Deepfake Detection Experiment"
echo "Train: CelebDF-v2 -> Test: FaceForensics++"
echo "============================================================"

# Configuration
EXPERIMENT_NAME="CelebDF_to_FF++_H100"
BASE_DIR="$(pwd)"
CHECKPOINT_DIR="./checkpoints/h100_optimized"
RESULTS_DIR="./results/${EXPERIMENT_NAME}"
LOG_DIR="./logs/${EXPERIMENT_NAME}"

# Create directories
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${RESULTS_DIR}"
mkdir -p "${LOG_DIR}"

# Validate paths - UPDATE THESE TO YOUR ACTUAL DATASET PATHS
CELEBDF_PATH="/path/to/CelebDF_extracted"  # UPDATE THIS
FF_PATH="/path/to/FaceForensics++_extracted"  # UPDATE THIS

echo "Validating dataset paths..."
if [ ! -d "$CELEBDF_PATH" ]; then
    echo "ERROR: CelebDF dataset not found at: $CELEBDF_PATH"
    echo "Please update CELEBDF_PATH in this script"
    exit 1
fi

if [ ! -d "$FF_PATH" ]; then
    echo "ERROR: FaceForensics++ dataset not found at: $FF_PATH"
    echo "Please update FF_PATH in this script"
    exit 1
fi

# Update config files with correct paths
echo "Updating configuration files with dataset paths..."
sed -i.bak "s|/path/to/CelebDF_extracted|${CELEBDF_PATH}|g" ./config/dataset/CelebDF_H100.yml
sed -i.bak "s|/path/to/FaceForensics++_extracted|${FF_PATH}|g" ./config/dataset/FF++_test.yml

echo "Configuration updated successfully"

# System information
echo "============================================================"
echo "System Information"
echo "============================================================"
nvidia-smi
echo ""
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""

# Check dependencies
echo "Checking Python dependencies..."
python3 -c "
import torch
import torchvision
import yaml
import numpy as np
import sklearn
import matplotlib
import seaborn
print('All dependencies verified!')
"

# Stage 1: Training on CelebDF-v2
echo "============================================================"
echo "Stage 1: Training on CelebDF-v2 (H100 Optimized)"
echo "============================================================"

TRAIN_LOG="${LOG_DIR}/training.log"
echo "Starting training... (log: ${TRAIN_LOG})"

# Run training with optimizations
python3 train_h100.py \
    --config ./config/H100_CelebDF_optimized.yml \
    --mixed_precision \
    --compile \
    2>&1 | tee "${TRAIN_LOG}"

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "✓ Training completed successfully!"
    
    # Find the best model
    BEST_MODEL="${CHECKPOINT_DIR}/best_model.pt"
    if [ -f "$BEST_MODEL" ]; then
        echo "✓ Best model found: ${BEST_MODEL}"
    else
        echo "⚠ Best model not found, using latest checkpoint"
        BEST_MODEL=$(ls -t ${CHECKPOINT_DIR}/step_*_model.pt | head -1)
        echo "Using: ${BEST_MODEL}"
    fi
else
    echo "✗ Training failed! Check log: ${TRAIN_LOG}"
    exit 1
fi

# Stage 2: Testing on FaceForensics++
echo "============================================================"
echo "Stage 2: Testing on FaceForensics++ (H100 Optimized)"
echo "============================================================"

TEST_LOG="${LOG_DIR}/testing.log"
TEST_RESULTS="${RESULTS_DIR}/ff++_results"

echo "Starting FaceForensics++ testing... (log: ${TEST_LOG})"

# Run FaceForensics++ testing
python3 test_ff++.py \
    --model_path "${BEST_MODEL}" \
    --config ./config/dataset/FF++_test.yml \
    --output_dir "${TEST_RESULTS}" \
    --batch_size 128 \
    --num_workers 12 \
    --mixed_precision \
    --save_predictions \
    --visualize \
    --test_methods Deepfakes Face2Face FaceSwap NeuralTextures \
    2>&1 | tee "${TEST_LOG}"

# Check if testing completed successfully
if [ $? -eq 0 ]; then
    echo "✓ FaceForensics++ testing completed successfully!"
else
    echo "✗ Testing failed! Check log: ${TEST_LOG}"
    exit 1
fi

# Stage 3: Generate comprehensive report
echo "============================================================"
echo "Stage 3: Generating Comprehensive Report"
echo "============================================================"

REPORT_FILE="${RESULTS_DIR}/experiment_report.txt"

cat > "${REPORT_FILE}" << EOF
============================================================
H100 Optimized Deepfake Detection Experiment Report
============================================================

Experiment: ${EXPERIMENT_NAME}
Date: $(date)
Duration: Training + Testing

CONFIGURATION:
- Training Dataset: CelebDF-v2
- Testing Dataset: FaceForensics++
- Model: Attention Network with Cross-Modal Fusion
- Optimization: H100 with Mixed Precision, Torch Compile
- Training Batch Size: 64
- Testing Batch Size: 128

SYSTEM INFO:
$(nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv,noheader,nounits)

TRAINING RESULTS:
$(tail -20 "${TRAIN_LOG}" | grep -E "(Acc|AUC|Loss|Training completed)")

TESTING RESULTS:
$(cat "${TEST_RESULTS}/ff++_results_summary.csv" 2>/dev/null || echo "Results summary not found")

FILES GENERATED:
- Training logs: ${TRAIN_LOG}
- Testing logs: ${TEST_LOG}
- Model checkpoint: ${BEST_MODEL}
- Test results: ${TEST_RESULTS}/
- Visualizations: ${TEST_RESULTS}/visualizations/

PERFORMANCE ANALYSIS:
$(python3 -c "
import json
import os
try:
    with open('${TEST_RESULTS}/ff++_test_summary.json', 'r') as f:
        results = json.load(f)
    
    print('Cross-dataset Performance (CelebDF → FF++):')
    for result in results:
        print(f'  {result[\"method\"]:<15}: Acc={result[\"accuracy\"]:.4f}, AUC={result[\"auc_roc\"]:.4f}')
    
    avg_acc = sum(r['accuracy'] for r in results) / len(results)
    avg_auc = sum(r['auc_roc'] for r in results) / len(results)
    print(f'  Average Performance: Acc={avg_acc:.4f}, AUC={avg_auc:.4f}')
    
except Exception as e:
    print(f'Could not analyze results: {e}')
")

============================================================
EOF

echo "Comprehensive report generated: ${REPORT_FILE}"

# Stage 4: Performance summary
echo "============================================================"
echo "EXPERIMENT COMPLETED SUCCESSFULLY!"
echo "============================================================"

echo "📊 Results Summary:"
if [ -f "${TEST_RESULTS}/ff++_test_summary.json" ]; then
    python3 -c "
import json
with open('${TEST_RESULTS}/ff++_test_summary.json', 'r') as f:
    results = json.load(f)

print('\nCross-dataset Performance (CelebDF-v2 → FaceForensics++):')
print('-' * 60)
for result in results:
    print(f'{result[\"method\"]:<15}: Acc={result[\"accuracy\"]:.4f} | AUC={result[\"auc_roc\"]:.4f}')

avg_acc = sum(r['accuracy'] for r in results) / len(results)
avg_auc = sum(r['auc_roc'] for r in results) / len(results)
print('-' * 60)
print(f'{'Average':<15}: Acc={avg_acc:.4f} | AUC={avg_auc:.4f}')
print()
"
fi

echo "📁 Generated Files:"
echo "   • Training logs: ${TRAIN_LOG}"
echo "   • Testing logs: ${TEST_LOG}"
echo "   • Model checkpoint: ${BEST_MODEL}"
echo "   • Results: ${TEST_RESULTS}/"
echo "   • Report: ${REPORT_FILE}"

echo ""
echo "✅ H100 optimized experiment completed successfully!"
echo "   Your cross-dataset evaluation results are ready for the AAAI paper."

# Cleanup backup files
rm -f ./config/dataset/*.bak

echo "============================================================"