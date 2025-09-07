#!/usr/bin/env python3
"""
Wild Deepfake Dataset Cross-Dataset Test
Test CelebDF-trained model on Wild deepfake dataset
"""

import torch
import time
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from model.network import Recce
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import json

class WildDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.images = []
        self.labels = []
        
        # Load real samples (label = 0)
        real_dir = os.path.join(root_dir, 'real')
        if os.path.exists(real_dir):
            for img in os.listdir(real_dir):
                if img.endswith(('.jpg', '.png', '.jpeg')):
                    self.images.append(os.path.join(real_dir, img))
                    self.labels.append(0)
        
        # Load fake samples (label = 1)
        fake_dir = os.path.join(root_dir, 'fake')
        if os.path.exists(fake_dir):
            for img in os.listdir(fake_dir):
                if img.endswith(('.jpg', '.png', '.jpeg')):
                    self.images.append(os.path.join(fake_dir, img))
                    self.labels.append(1)
        
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Wild Dataset loaded: {len(self.images)} samples")
        print(f"Real: {sum(1 for l in self.labels if l == 0)}, Fake: {sum(1 for l in self.labels if l == 1)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a zero tensor if image fails to load
            return torch.zeros((3, 299, 299)), label

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

def test_wild_dataset():
    print("="*80)
    print("CROSS-DATASET EVALUATION: CelebDF-trained Model → Wild Deepfake Dataset")
    print("="*80)
    
    # Load Wild dataset
    dataset = WildDataset('wild_deepfake_data/test')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Load trained model
    model = load_trained_model('checkpoints/best_model.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    print(f"Testing on {len(dataset)} samples...")
    start_time = time.time()
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if (i + 1) % 50 == 0:
                print(f"Processed {(i + 1) * dataloader.batch_size} samples...")
            
            images = images.to(device)
            outputs = model(images)
            predictions = torch.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(predictions, dim=1)
            
            all_predictions.extend(predicted_classes.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    test_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Calculate AUC if we have both classes
    auc = None
    if len(set(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_predictions)
    
    # Results
    print("="*80)
    print("WILD DEEPFAKE CROSS-DATASET EVALUATION RESULTS")
    print("="*80)
    print(f"Dataset: Wild Deepfake Test Set")
    print(f"Model: CelebDF-trained Recce") 
    print(f"Test samples: {len(dataset):,}")
    print(f"Test time: {test_time:.2f} seconds")
    print(f"Samples/sec: {len(dataset)/test_time:.1f}")
    print()
    print("PERFORMANCE METRICS:")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    if auc:
        print(f"AUC: {auc:.4f}")
    else:
        print("AUC: nan (single class dataset)")
    print()
    
    # Class distribution
    real_count = sum(1 for l in all_labels if l == 0)
    fake_count = sum(1 for l in all_labels if l == 1)
    pred_real = sum(1 for p in all_predictions if p == 0)
    pred_fake = sum(1 for p in all_predictions if p == 1)
    
    print("CLASS DISTRIBUTION:")
    print("Ground Truth:")
    print(f"  Real: {real_count} samples ({real_count/len(all_labels)*100:.1f}%)")
    print(f"  Fake: {fake_count} samples ({fake_count/len(all_labels)*100:.1f}%)")
    print("Predictions:")
    print(f"  Real: {pred_real} samples ({pred_real/len(all_predictions)*100:.1f}%)")
    print(f"  Fake: {pred_fake} samples ({pred_fake/len(all_predictions)*100:.1f}%)")
    print()
    
    print("CONFUSION MATRIX:")
    print(cm)
    print()
    
    print("CLASSIFICATION REPORT:")
    target_names = ['Real', 'Fake']
    print(classification_report(all_labels, all_predictions, target_names=target_names))
    
    # Save results
    results = {
        'dataset': 'Wild_Deepfake',
        'accuracy': float(accuracy),
        'auc': float(auc) if auc else None,
        'total_samples': len(dataset),
        'test_time': test_time,
        'samples_per_sec': len(dataset) / test_time,
        'confusion_matrix': cm.tolist(),
        'class_distribution': {
            'real_gt': int(real_count),
            'fake_gt': int(fake_count),
            'real_pred': int(pred_real),
            'fake_pred': int(pred_fake)
        }
    }
    
    with open('wild_cross_dataset_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Results saved to: wild_cross_dataset_results.json")
    print("="*80)
    print()
    print(f"🎯 FINAL CROSS-DATASET RESULTS:")
    print(f"   CelebDF → Wild Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    if auc:
        print(f"   CelebDF → Wild AUC: {auc:.4f}")

if __name__ == "__main__":
    test_wild_dataset()