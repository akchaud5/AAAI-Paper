#!/usr/bin/env python3
"""
Final DFDC Cross-Dataset Test - Evaluating CelebDF-trained model on DFDC
"""

import torch
import yaml
import time
import numpy as np
from torch.utils.data import DataLoader
from dataset import load_dataset
from model.network import Recce
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

def load_trained_model(model_path):
    """Load the CelebDF-trained model with GlobalFilter fix"""
    checkpoint = torch.load(model_path, map_location='cpu')
    model_state = checkpoint['model_state']
    
    model = Recce(num_classes=2, drop_rate=0.2)
    
    # Fix GlobalFilter dimensions to match trained model
    filter_shape = model_state['filter.filter.complex_weight'].shape
    model.filter.filter.h = filter_shape[0]
    model.filter.filter.w = (filter_shape[1] - 1) * 2
    model.filter.filter.complex_weight = torch.nn.Parameter(
        torch.randn(filter_shape[0], filter_shape[1], 728, 2, dtype=torch.float32) * 0.02
    )
    
    model.load_state_dict(model_state, strict=True)
    return model

def test_dfdc_cross_dataset():
    print("="*80)
    print("CROSS-DATASET EVALUATION: CelebDF-trained Model → DFDC Test Set")  
    print("="*80)
    
    # Load DFDC test dataset
    with open('config/dataset/H100_DFDC_dataset.yml', 'r') as f:
        config = yaml.safe_load(f)
    test_config = config['test_cfg']
    
    print(f"Loading DFDC dataset from: {test_config['root']}")
    dataset = load_dataset('DFDC')(test_config)
    print(f"DFDC Test samples: {len(dataset)}")
    
    # Load CelebDF-trained model
    model_path = 'checkpoints/best_model.pt'
    print(f"Loading CelebDF-trained model: {model_path}")
    model = load_trained_model(model_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()
    
    # H100 optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=16, 
        shuffle=False, 
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Test the model
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    print(f"Testing on {len(dataloader)} batches...")
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(dataloader):
            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(data)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
    
    test_time = time.time() - start_time
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    
    # For AUC, we need probabilities of the positive class
    if probabilities.shape[1] == 2:
        auc = roc_auc_score(labels, probabilities[:, 1])
    else:
        auc = 0.0
    
    # Class distribution
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    unique_preds, pred_counts = np.unique(predictions, return_counts=True)
    
    print("="*80)
    print("CROSS-DATASET EVALUATION RESULTS")
    print("="*80)
    print(f"Dataset: DFDC Test Set")
    print(f"Model: CelebDF-trained Recce")
    print(f"Test samples: {len(labels):,}")
    print(f"Test time: {test_time:.2f} seconds")
    print(f"Samples/sec: {len(labels)/test_time:.1f}")
    print()
    print("PERFORMANCE METRICS:")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"AUC: {auc:.4f}")
    print()
    print("CLASS DISTRIBUTION:")
    print("Ground Truth:")
    for label, count in zip(unique_labels, label_counts):
        print(f"  Class {label}: {count:,} samples ({count/len(labels)*100:.1f}%)")
    print("Predictions:")
    for pred, count in zip(unique_preds, pred_counts):
        print(f"  Class {pred}: {count:,} samples ({count/len(predictions)*100:.1f}%)")
    print()
    
    # Confusion Matrix
    print("CONFUSION MATRIX:")
    cm = confusion_matrix(labels, predictions)
    print(cm)
    print()
    
    # Classification Report
    print("CLASSIFICATION REPORT:")
    report = classification_report(labels, predictions, target_names=['Real', 'Fake'])
    print(report)
    
    # Save results
    results = {
        'accuracy': float(accuracy),
        'auc': float(auc),
        'total_samples': int(len(labels)),
        'test_time': float(test_time),
        'samples_per_sec': float(len(labels)/test_time),
        'confusion_matrix': cm.tolist(),
        'predictions': predictions.tolist(),
        'ground_truth': labels.tolist(),
        'probabilities': probabilities.tolist()
    }
    
    import json
    with open('dfdc_cross_dataset_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: dfdc_cross_dataset_results.json")
    print("="*80)
    
    return accuracy, auc

if __name__ == "__main__":
    try:
        accuracy, auc = test_dfdc_cross_dataset()
        print(f"\n🎯 FINAL CROSS-DATASET RESULTS:")
        print(f"   CelebDF → DFDC Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   CelebDF → DFDC AUC: {auc:.4f}")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()