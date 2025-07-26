#!/usr/bin/env python3
"""
Dataset Preparation Script for H100 Deepfake Detection
Helps organize and validate datasets for training and testing
"""

import os
import argparse
import requests
import zipfile
import tarfile
from pathlib import Path
import cv2
import json
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset preparation for deepfake detection")
    parser.add_argument("--celebdf_path", type=str, required=True,
                       help="Path where CelebDF-v2 should be stored")
    parser.add_argument("--ff_path", type=str, required=True,
                       help="Path where FaceForensics++ should be stored")
    parser.add_argument("--extract_faces", action="store_true",
                       help="Extract faces from videos (requires face detection)")
    parser.add_argument("--validate_only", action="store_true",
                       help="Only validate existing datasets")
    return parser.parse_args()

def validate_celebdf_structure(path):
    """Validate CelebDF-v2 dataset structure"""
    print(f"Validating CelebDF-v2 at: {path}")
    
    required_dirs = [
        "Celeb-real",
        "Celeb-synthesis", 
        "YouTube-real"
    ]
    
    issues = []
    total_files = 0
    
    for dir_name in required_dirs:
        dir_path = os.path.join(path, dir_name)
        if not os.path.exists(dir_path):
            issues.append(f"Missing directory: {dir_name}")
        else:
            file_count = len([f for f in os.listdir(dir_path) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            total_files += file_count
            print(f"  ✓ {dir_name}: {file_count} images")
    
    if issues:
        print("❌ CelebDF-v2 validation failed:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"✅ CelebDF-v2 validation passed: {total_files} total images")
    
    return len(issues) == 0

def validate_ff_structure(path):
    """Validate FaceForensics++ dataset structure"""
    print(f"Validating FaceForensics++ at: {path}")
    
    required_structure = {
        "original_sequences/youtube/c23/images": "original",
        "manipulated_sequences/Deepfakes/c23/images": "Deepfakes", 
        "manipulated_sequences/Face2Face/c23/images": "Face2Face",
        "manipulated_sequences/FaceSwap/c23/images": "FaceSwap",
        "manipulated_sequences/NeuralTextures/c23/images": "NeuralTextures"
    }
    
    issues = []
    total_files = 0
    
    for rel_path, method_name in required_structure.items():
        full_path = os.path.join(path, rel_path)
        if not os.path.exists(full_path):
            issues.append(f"Missing directory: {rel_path}")
        else:
            # Count images in subdirectories (train/val/test)
            method_count = 0
            for split in ['train', 'val', 'test']:
                split_path = os.path.join(full_path, split)
                if os.path.exists(split_path):
                    for video_dir in os.listdir(split_path):
                        video_path = os.path.join(split_path, video_dir)
                        if os.path.isdir(video_path):
                            file_count = len([f for f in os.listdir(video_path)
                                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                            method_count += file_count
            
            total_files += method_count
            print(f"  ✓ {method_name}: {method_count} images")
    
    if issues:
        print("❌ FaceForensics++ validation failed:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"✅ FaceForensics++ validation passed: {total_files} total images")
    
    return len(issues) == 0

def print_dataset_download_instructions():
    """Print instructions for downloading datasets"""
    print("\n" + "="*70)
    print("DATASET DOWNLOAD INSTRUCTIONS")
    print("="*70)
    
    print("\n📥 CelebDF-v2 Dataset:")
    print("1. Visit: https://github.com/yuezunli/celeb-deepfakeforensics")
    print("2. Fill out the request form")
    print("3. Download the dataset")
    print("4. Extract faces from videos using RetinaFace")
    print("5. Organize in the required structure")
    
    print("\n📥 FaceForensics++ Dataset:")
    print("1. Visit: https://github.com/ondyari/FaceForensics")
    print("2. Fill out the request form")
    print("3. Get the download script")  
    print("4. Run: python download_faceforensics.py -d /path -c c23 -t videos")
    print("5. Extract faces from videos")
    
    print("\n⚠️  Both datasets require:")
    print("- Academic email for access request")
    print("- Face extraction (RetinaFace recommended)")
    print("- Proper directory organization")

def create_directory_structure(path, structure_type):
    """Create directory structure for datasets"""
    print(f"Creating directory structure for {structure_type} at: {path}")
    
    if structure_type == "celebdf":
        dirs = ["Celeb-real", "Celeb-synthesis", "YouTube-real"]
    elif structure_type == "faceforensics":
        dirs = [
            "original_sequences/youtube/c23/images/train",
            "original_sequences/youtube/c23/images/val", 
            "original_sequences/youtube/c23/images/test",
            "manipulated_sequences/Deepfakes/c23/images/train",
            "manipulated_sequences/Deepfakes/c23/images/val",
            "manipulated_sequences/Deepfakes/c23/images/test",
            "manipulated_sequences/Face2Face/c23/images/train",
            "manipulated_sequences/Face2Face/c23/images/val",
            "manipulated_sequences/Face2Face/c23/images/test",
            "manipulated_sequences/FaceSwap/c23/images/train",
            "manipulated_sequences/FaceSwap/c23/images/val", 
            "manipulated_sequences/FaceSwap/c23/images/test",
            "manipulated_sequences/NeuralTextures/c23/images/train",
            "manipulated_sequences/NeuralTextures/c23/images/val",
            "manipulated_sequences/NeuralTextures/c23/images/test"
        ]
    
    for dir_path in dirs:
        full_path = os.path.join(path, dir_path)
        os.makedirs(full_path, exist_ok=True)
        print(f"  ✓ Created: {dir_path}")

def check_face_detection_requirements():
    """Check if face detection libraries are available"""
    try:
        import face_recognition
        import dlib
        print("✅ Face detection libraries available")
        return True
    except ImportError:
        print("❌ Face detection libraries missing")
        print("Install with: pip install face-recognition dlib")
        return False

def extract_faces_from_video(video_path, output_dir, max_frames=100):
    """Extract faces from a video file"""
    if not check_face_detection_requirements():
        return False
    
    import face_recognition
    
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    extracted_count = 0
    
    print(f"Extracting faces from: {os.path.basename(video_path)}")
    
    while cap.read()[0] and extracted_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Extract faces every 10 frames to avoid redundancy
        if frame_count % 10 != 0:
            continue
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find face locations
        face_locations = face_recognition.face_locations(rgb_frame)
        
        for i, (top, right, bottom, left) in enumerate(face_locations):
            # Extract face with some padding
            padding = 20
            top = max(0, top - padding)
            right = min(rgb_frame.shape[1], right + padding)
            bottom = min(rgb_frame.shape[0], bottom + padding)
            left = max(0, left - padding)
            
            face_image = rgb_frame[top:bottom, left:right]
            
            # Resize to standard size
            face_image = cv2.resize(face_image, (299, 299))
            
            # Convert back to BGR for saving
            face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
            
            # Save face
            output_path = os.path.join(output_dir, f"frame_{frame_count:06d}_face_{i}.jpg")
            cv2.imwrite(output_path, face_image)
            extracted_count += 1
            
            if extracted_count >= max_frames:
                break
    
    cap.release()
    print(f"  Extracted {extracted_count} faces")
    return extracted_count > 0

def generate_setup_script():
    """Generate a setup script for easy dataset preparation"""
    script_content = '''#!/bin/bash

# Dataset Setup Script for H100 Deepfake Detection
echo "=========================================="
echo "Dataset Setup for H100 Deepfake Detection"
echo "=========================================="

# Set your paths here
CELEBDF_PATH="/path/to/CelebDF_extracted"  # UPDATE THIS
FF_PATH="/path/to/FaceForensics++_extracted"  # UPDATE THIS

echo "Setting up dataset directories..."

# Create CelebDF structure
mkdir -p "$CELEBDF_PATH"/{Celeb-real,Celeb-synthesis,YouTube-real}

# Create FaceForensics++ structure  
mkdir -p "$FF_PATH"/original_sequences/youtube/c23/images/{train,val,test}
mkdir -p "$FF_PATH"/manipulated_sequences/{Deepfakes,Face2Face,FaceSwap,NeuralTextures}/c23/images/{train,val,test}

echo "Directory structure created!"
echo ""
echo "Next steps:"
echo "1. Download CelebDF-v2 from: https://github.com/yuezunli/celeb-deepfakeforensics"
echo "2. Download FaceForensics++ from: https://github.com/ondyari/FaceForensics"
echo "3. Extract faces and organize in the created directories"
echo "4. Run validation: python prepare_datasets.py --celebdf_path $CELEBDF_PATH --ff_path $FF_PATH --validate_only"
'''
    
    with open('setup_datasets.sh', 'w') as f:
        f.write(script_content)
    
    os.chmod('setup_datasets.sh', 0o755)
    print("✅ Generated setup_datasets.sh script")

def main():
    args = parse_args()
    
    print("Dataset Preparation for H100 Deepfake Detection")
    print("="*50)
    
    if args.validate_only:
        print("Validation mode - checking existing datasets...")
        
        celebdf_valid = False
        ff_valid = False
        
        if os.path.exists(args.celebdf_path):
            celebdf_valid = validate_celebdf_structure(args.celebdf_path)
        else:
            print(f"❌ CelebDF path does not exist: {args.celebdf_path}")
        
        if os.path.exists(args.ff_path):
            ff_valid = validate_ff_structure(args.ff_path)
        else:
            print(f"❌ FaceForensics++ path does not exist: {args.ff_path}")
        
        if celebdf_valid and ff_valid:
            print("\n🎉 All datasets validated successfully!")
            print("You can now run the H100 training pipeline.")
        else:
            print("\n❌ Dataset validation failed.")
            print("Please download and organize datasets properly.")
            print_dataset_download_instructions()
        
        return celebdf_valid and ff_valid
    
    # Create directory structures
    print("Creating directory structures...")
    create_directory_structure(args.celebdf_path, "celebdf")
    create_directory_structure(args.ff_path, "faceforensics")
    
    # Generate setup script
    generate_setup_script()
    
    # Print download instructions
    print_dataset_download_instructions()
    
    print("\n" + "="*70)
    print("SETUP COMPLETE")
    print("="*70)
    print("✅ Directory structures created")
    print("✅ Setup script generated: setup_datasets.sh")
    print("")
    print("⚠️  IMPORTANT: You still need to:")
    print("1. Download the actual dataset files (requires registration)")
    print("2. Extract faces from videos")
    print("3. Organize files in the created directories")
    print("4. Run validation with --validate_only flag")

if __name__ == "__main__":
    main()