#!/bin/bash

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
