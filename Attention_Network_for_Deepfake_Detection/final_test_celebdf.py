#!/usr/bin/env python3
"""
Final CelebDF Test - Direct Validation
"""

import torch
import yaml
import time
import numpy as np
from torch.utils.data import DataLoader
from dataset import load_dataset
from model.network import Recce
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

def load_trained_model(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    model_state = checkpoint['model_state']
    
    model = Recce(num_classes=2, drop_rate=0.2)
    
    # Fix GlobalFilter dimensions
    filter_shape = model_state['filter.filter.complex_weight'].shape
    model.filter.filter.h = filter_shape[0]
    model.filter.filter.w = (filter_shape[1] - 1) * 2
    model.filter.filter.complex_weight = torch.nn.Parameter(
        torch.randn(filter_shape[0], filter_shape[1], 728, 2, dtype=torch.float32) * 0.02
    )
    
    model.load_state_dict(model_state, strict=True)
    return model

def test_celebdf_final():
    print("="*70)
    print("FINAL CELEBDF OFFICIAL TEST - CORRECTED METRICS")  
    print("="*70)
    
    # Load dataset
    with open('config/dataset/H100_CelebDF_dataset.yml', 'r') as f:
        config = yaml.safe_load(f)
    test_config = config['test_cfg']
    dataset = load_dataset('CelebDF')(test_config)
    
    # Load model
    model = load_trained_model('checkpoints/best_model.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"Testing {len(dataset)} samples...")
    print(f"Real samples: 741, Fake samples: 4680")
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            if len(batch_data) == 3:
                images, labels, _ = batch_data
            else:
                images, labels = batch_data
            
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Handle potential squeeze for single sample batches
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(0)
            
            # Get predictions and probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy().tolist())
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Processed {(batch_idx + 1) * 32} samples...")
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # AUC calculation (using probabilities for positive class)
    try:
        auc_score = roc_auc_score(all_labels, all_probabilities[:, 1])
    except:
        auc_score = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Classification report
    report = classification_report(all_labels, all_predictions, 
                                 target_names=['Real', 'Fake'], 
                                 output_dict=True)
    
    # Manual validation 
    correct_real = np.sum((all_labels == 0) & (all_predictions == 0))
    correct_fake = np.sum((all_labels == 1) & (all_predictions == 1))
    total_real = np.sum(all_labels == 0)
    total_fake = np.sum(all_labels == 1)
    
    print(f"\\nTotal time: {time.time() - start_time:.1f}s")
    print(f"\\n🎯 FINAL RESULTS:")
    print(f"{'='*50}")
    print(f"📊 Overall Performance:")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   AUC-ROC:  {auc_score:.4f}")
    
    print(f"\\n📈 Per-Class Performance:")
    print(f"   Real samples  - Correct: {correct_real:4d}/{total_real:4d} = {correct_real/total_real*100:.2f}%")
    print(f"   Fake samples  - Correct: {correct_fake:4d}/{total_fake:4d} = {correct_fake/total_fake*100:.2f}%")
    
    print(f"\\n📋 Confusion Matrix:")
    print(f"              Predicted")
    print(f"           Real    Fake")
    print(f"    Real   {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"    Fake   {cm[1,0]:4d}    {cm[1,1]:4d}")
    
    print(f"\\n📊 Detailed Classification Report:")
    for class_name in ['Real', 'Fake']:
        metrics = report[class_name]
        print(f"   {class_name:4s}: Precision={metrics['precision']:.4f}, "
              f"Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")
    
    # Sample predictions check
    print(f"\\n🔍 Sample Predictions Verification:")
    for i in [0, 100, 741, 1000, 3000, 5000]:
        if i < len(all_predictions):
            pred_class = "Real" if all_predictions[i] == 0 else "Fake" 
            true_class = "Real" if all_labels[i] == 0 else "Fake"
            conf = max(all_probabilities[i])
            status = "✓" if all_predictions[i] == all_labels[i] else "✗"
            print(f"   Sample {i:4d}: {status} Predicted={pred_class:4s}, True={true_class:4s}, Conf={conf:.3f}")

if __name__ == "__main__":
    test_celebdf_final()