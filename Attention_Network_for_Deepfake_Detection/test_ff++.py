#!/usr/bin/env python3
"""
FaceForensics++ Testing Script
Optimized for H100 inference on FaceForensics++ dataset
Tests trained CelebDF model on FF++ for cross-dataset evaluation
"""

import os
import yaml
import argparse
import torch
import time
import json
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from model.network import Recce
from dataset.faceforensics import create_faceforensics_dataset
from trainer.utils import AverageMeter, AccMeter, AUCMeter
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def parse_arguments():
    parser = argparse.ArgumentParser(description="FaceForensics++ Testing with H100 Optimization")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model checkpoint")
    parser.add_argument("--config", type=str, 
                       default="./config/dataset/FF++_test.yml",
                       help="Path to FaceForensics++ test configuration")
    parser.add_argument("--output_dir", type=str, default="./ff++_results",
                       help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=128,
                       help="Batch size for testing (optimized for H100)")
    parser.add_argument("--num_workers", type=int, default=12,
                       help="Number of data loading workers")
    parser.add_argument("--mixed_precision", action="store_true", default=True,
                       help="Enable mixed precision inference")
    parser.add_argument("--test_methods", nargs='+', 
                       default=['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'],
                       help="Manipulation methods to test")
    parser.add_argument("--save_predictions", action="store_true", default=True,
                       help="Save detailed predictions")
    parser.add_argument("--visualize", action="store_true", default=True,
                       help="Generate visualization plots")
    return parser.parse_args()

def setup_h100_inference():
    """Setup optimizations for H100 inference"""
    if torch.cuda.is_available():
        # Enable TensorFloat-32 for improved inference speed
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        print("H100 inference optimizations enabled:")
        print(f"  - Device: {torch.cuda.get_device_name()}")
        print(f"  - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  - TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")

def load_model(model_path, device):
    """Load trained model with optimizations"""
    print(f"Loading model from: {model_path}")
    
    # Initialize model
    model = Recce(num_classes=2, drop_rate=0.2)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state' in checkpoint:
        model_state = checkpoint['model_state']
    else:
        model_state = checkpoint
    
    # Handle DDP wrapped models
    if any(key.startswith('module.') for key in model_state.keys()):
        model_state = {key.replace('module.', ''): value for key, value in model_state.items()}
    
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()
    
    # Compile model for H100 optimization
    if hasattr(torch, 'compile'):
        print("Compiling model for H100 inference optimization...")
        model = torch.compile(model)
    
    print("Model loaded and optimized successfully")
    return model

def create_optimized_dataloader(dataset, batch_size, num_workers):
    """Create optimized dataloader for H100 inference"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle for testing
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False
    )

def test_method(model, dataset, device, method_name, mixed_precision=True):
    """Test model on specific manipulation method"""
    print(f"\n{'='*50}")
    print(f"Testing on {method_name}")
    print(f"{'='*50}")
    
    # Create dataloader
    dataloader = create_optimized_dataloader(dataset, batch_size=128, num_workers=12)
    
    # Initialize metrics
    loss_meter = AverageMeter()
    acc_meter = AccMeter()
    auc_meter = AUCMeter()
    auc_meter.reset()
    
    criterion = torch.nn.CrossEntropyLoss()
    
    # Storage for predictions
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    start_time = time.time()
    total_samples = 0
    
    print(f"Testing {len(dataset)} samples...")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device, memory_format=torch.channels_last)
            labels = labels.to(device).long()
            
            if mixed_precision:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Get predictions and probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            # Update metrics
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(outputs, labels)
            auc_meter.update(outputs, labels)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            total_samples += images.size(0)
            
            # Progress reporting
            if (batch_idx + 1) % 50 == 0:
                elapsed = time.time() - start_time
                samples_per_sec = total_samples / elapsed
                print(f"Batch [{batch_idx+1}/{len(dataloader)}], "
                      f"Samples: {total_samples}, "
                      f"Speed: {samples_per_sec:.1f} samples/sec")
    
    total_time = time.time() - start_time
    samples_per_sec = total_samples / total_time
    
    # Calculate final metrics
    accuracy = acc_meter.mean_acc()
    auc_score = auc_meter.compute_auc()
    avg_loss = loss_meter.avg
    
    # Additional metrics
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    class_report = classification_report(all_labels, all_predictions, 
                                       target_names=['Fake', 'Real'], 
                                       output_dict=True)
    
    results = {
        'method': method_name,
        'accuracy': float(accuracy),
        'auc_roc': float(auc_score),
        'avg_loss': float(avg_loss),
        'total_samples': total_samples,
        'inference_time': total_time,
        'samples_per_sec': samples_per_sec,
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities
    }
    
    print(f"\nResults for {method_name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC-ROC: {auc_score:.4f}")
    print(f"  Avg Loss: {avg_loss:.4f}")
    print(f"  Total Time: {total_time:.1f}s")
    print(f"  Throughput: {samples_per_sec:.1f} samples/sec")
    print(f"  Total Samples: {total_samples}")
    
    return results

def create_visualizations(all_results, output_dir):
    """Create visualization plots"""
    if not all_results:
        return
    
    print("\nGenerating visualizations...")
    
    # Create output directory
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. Performance comparison bar chart
    methods = [r['method'] for r in all_results]
    accuracies = [r['accuracy'] for r in all_results]
    auc_scores = [r['auc_roc'] for r in all_results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    bars1 = ax1.bar(methods, accuracies, color='skyblue', alpha=0.7)
    ax1.set_title('Accuracy by Manipulation Method', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # AUC comparison
    bars2 = ax2.bar(methods, auc_scores, color='lightcoral', alpha=0.7)
    ax2.set_title('AUC-ROC by Manipulation Method', fontsize=14, fontweight='bold')
    ax2.set_ylabel('AUC-ROC', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, auc in zip(bars2, auc_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, result in enumerate(all_results):
        if idx >= 4:  # Only show first 4 methods
            break
            
        cm = np.array(result['confusion_matrix'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Fake', 'Real'],
                   yticklabels=['Fake', 'Real'],
                   ax=axes[idx])
        axes[idx].set_title(f'{result["method"]}\nAcc: {result["accuracy"]:.3f}')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to: {viz_dir}")

def save_detailed_results(all_results, output_dir):
    """Save detailed results to files"""
    print("\nSaving detailed results...")
    
    # Create summary results (without predictions for file size)
    summary_results = []
    for result in all_results:
        summary = {k: v for k, v in result.items() 
                  if k not in ['predictions', 'labels', 'probabilities']}
        summary_results.append(summary)
    
    # Save summary
    with open(os.path.join(output_dir, 'ff++_test_summary.json'), 'w') as f:
        json.dump(summary_results, f, indent=2)
    
    # Save detailed results for each method
    for result in all_results:
        method_file = os.path.join(output_dir, f'{result["method"]}_detailed.json')
        with open(method_file, 'w') as f:
            json.dump(result, f, indent=2)
    
    # Create CSV summary
    import pandas as pd
    
    df_data = []
    for result in summary_results:
        row = {
            'Method': result['method'],
            'Accuracy': f"{result['accuracy']:.4f}",
            'AUC-ROC': f"{result['auc_roc']:.4f}",
            'Avg Loss': f"{result['avg_loss']:.4f}",
            'Samples': result['total_samples'],
            'Time (s)': f"{result['inference_time']:.1f}",
            'Samples/sec': f"{result['samples_per_sec']:.1f}"
        }
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    df.to_csv(os.path.join(output_dir, 'ff++_results_summary.csv'), index=False)
    
    print(f"Results saved to: {output_dir}")

def main():
    args = parse_arguments()
    
    print("="*70)
    print("FaceForensics++ Testing with H100 Optimization")
    print("="*70)
    
    # Setup H100 optimizations
    setup_h100_inference()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    print(f"Loading test configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Test on each manipulation method
    all_results = []
    
    for method in args.test_methods:
        print(f"\nPreparing dataset for {method}...")
        
        # Get method-specific config
        method_config = config.get(f'{method.lower()}_cfg', config['test_cfg'])
        
        # Create dataset
        try:
            dataset = create_faceforensics_dataset(method_config, method)
            print(f"Dataset created: {len(dataset)} samples")
            
            # Test the method
            results = test_method(model, dataset, device, method, args.mixed_precision)
            all_results.append(results)
            
        except Exception as e:
            print(f"Error testing {method}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_results:
        print("No successful tests completed!")
        return
    
    # Calculate overall statistics
    print("\n" + "="*70)
    print("OVERALL RESULTS SUMMARY")
    print("="*70)
    
    total_samples = sum(r['total_samples'] for r in all_results)
    avg_accuracy = np.mean([r['accuracy'] for r in all_results])
    avg_auc = np.mean([r['auc_roc'] for r in all_results])
    total_time = sum(r['inference_time'] for r in all_results)
    avg_throughput = np.mean([r['samples_per_sec'] for r in all_results])
    
    print(f"Methods tested: {len(all_results)}")
    print(f"Total samples: {total_samples:,}")
    print(f"Average accuracy: {avg_accuracy:.4f}")
    print(f"Average AUC-ROC: {avg_auc:.4f}")
    print(f"Total inference time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Average throughput: {avg_throughput:.1f} samples/sec")
    
    # Best and worst performing methods
    best_acc = max(all_results, key=lambda x: x['accuracy'])
    worst_acc = min(all_results, key=lambda x: x['accuracy'])
    
    print(f"\nBest performance: {best_acc['method']} (Acc: {best_acc['accuracy']:.4f})")
    print(f"Worst performance: {worst_acc['method']} (Acc: {worst_acc['accuracy']:.4f})")
    
    # Save results
    if args.save_predictions:
        save_detailed_results(all_results, args.output_dir)
    
    # Generate visualizations
    if args.visualize:
        create_visualizations(all_results, args.output_dir)
    
    print(f"\nTesting completed! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()