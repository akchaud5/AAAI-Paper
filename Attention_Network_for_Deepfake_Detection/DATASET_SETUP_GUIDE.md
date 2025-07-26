# 📥 Dataset Setup Guide for H100 Deepfake Detection

**⚠️ IMPORTANT**: Datasets must be manually downloaded due to access restrictions and licensing requirements.

## 🎯 Quick Setup Steps

### 1. **Prepare Directory Structure**
```bash
# Run the dataset preparation script
python prepare_datasets.py \
    --celebdf_path /path/to/CelebDF_extracted \
    --ff_path /path/to/FaceForensics++_extracted
```

### 2. **Download Datasets** (Manual Step Required)

#### **CelebDF-v2 Dataset**
```bash
# 1. Visit the official repository
# https://github.com/yuezunli/celeb-deepfakeforensics

# 2. Fill out the access request form
# 3. Download the dataset (requires academic email)
# 4. You'll receive download links via email
```

#### **FaceForensics++ Dataset**  
```bash
# 1. Visit the official repository
# https://github.com/ondyari/FaceForensics

# 2. Fill out the access request form
# 3. Download the dataset script:
wget https://github.com/ondyari/FaceForensics/raw/master/dataset/download_faceforensics.py

# 4. Run the download script (after getting access)
python download_faceforensics.py \
    --dataset FaceForensics++ \
    --data_path /path/to/FaceForensics++_extracted \
    --compression c23 \
    --type videos
```

### 3. **Face Extraction** (Required)

Both datasets need face extraction from videos:

```bash
# Install face detection requirements
pip install face-recognition dlib opencv-python

# Extract faces using the preparation script
python prepare_datasets.py \
    --celebdf_path /path/to/CelebDF_extracted \
    --ff_path /path/to/FaceForensics++_extracted \
    --extract_faces
```

### 4. **Validate Setup**
```bash
# Check if datasets are properly organized
python prepare_datasets.py \
    --celebdf_path /path/to/CelebDF_extracted \
    --ff_path /path/to/FaceForensics++_extracted \
    --validate_only
```

## 📁 Required Directory Structure

### **CelebDF-v2 Structure**
```
/path/to/CelebDF_extracted/
├── Celeb-real/              # Real celebrity images
│   ├── id0_0000.jpg
│   ├── id0_0001.jpg
│   └── ...
├── Celeb-synthesis/         # Fake celebrity images  
│   ├── id0_id1_0000.jpg
│   ├── id0_id1_0001.jpg
│   └── ...
└── YouTube-real/            # Real YouTube images
    ├── yt_id0_0000.jpg
    ├── yt_id0_0001.jpg
    └── ...
```

### **FaceForensics++ Structure**
```
/path/to/FaceForensics++_extracted/
├── original_sequences/
│   └── youtube/
│       └── c23/
│           └── images/
│               ├── train/
│               ├── val/
│               └── test/
└── manipulated_sequences/
    ├── Deepfakes/
    │   └── c23/
    │       └── images/
    │           ├── train/
    │           ├── val/
    │           └── test/
    ├── Face2Face/
    ├── FaceSwap/
    └── NeuralTextures/
```

## 🚀 Alternative: Pre-processed Datasets

If you have access to pre-processed versions:

### **Option 1: Academic Collaborations**
- Contact authors of recent deepfake papers
- Many researchers share pre-extracted faces
- Faster setup, but requires networking

### **Option 2: Kaggle/HuggingFace**
- Some processed versions available (check licensing)
- May not be identical to official versions
- Ensure compatibility with paper benchmarks

### **Option 3: Subset Testing**
For quick testing, create smaller subsets:

```bash
# Create a small test subset (1000 images each)
python prepare_datasets.py \
    --celebdf_path /path/to/CelebDF_subset \
    --ff_path /path/to/FF++_subset \
    --create_subset 1000
```

## ⚡ Fast Track Setup (If You Have Datasets)

If you already have the datasets in different formats:

```bash
# 1. Create proper directory structure
python prepare_datasets.py --create_structure_only

# 2. Move your files to match the expected structure
# 3. Rename files if necessary
# 4. Validate the setup
python prepare_datasets.py --validate_only
```

## 🔧 Troubleshooting

### **Common Issues:**

#### **1. Access Request Rejected**
- Use academic email address
- Provide detailed research purpose
- Reference your AAAI paper work
- Contact dataset authors directly

#### **2. Download Fails**
- Check internet connection
- Use VPN if downloads are geo-blocked
- Try downloading in smaller chunks
- Contact dataset maintainers

#### **3. Face Extraction Errors**
```bash
# Install dependencies
pip install cmake dlib face-recognition

# For M1 Macs:
brew install cmake
pip install dlib --no-cache-dir

# For Linux:
sudo apt-get install cmake libboost-all-dev
```

#### **4. Directory Structure Issues**
```bash
# Run the structure validator
python prepare_datasets.py --validate_only --verbose

# Fix common issues automatically  
python prepare_datasets.py --fix_structure
```

## 📊 Dataset Statistics

### **Expected Sizes:**
- **CelebDF-v2**: ~6,000 videos → ~2M faces (15GB)
- **FaceForensics++**: ~5,000 videos → ~2.4M faces (38GB)
- **Total**: ~4.4M faces (~53GB)

### **Processing Time:**
- **Face extraction**: 2-4 hours per dataset
- **Validation**: 5-10 minutes
- **Organization**: 15-30 minutes

## 🎯 Ready for H100 Training

Once validation passes:

```bash
✅ CelebDF-v2 validation passed: 2,000,000 total images
✅ FaceForensics++ validation passed: 2,400,000 total images

# Update config files with your paths
sed -i "s|/path/to/CelebDF_extracted|/your/actual/path|g" config/dataset/CelebDF_H100.yml
sed -i "s|/path/to/FaceForensics++_extracted|/your/actual/path|g" config/dataset/FF++_test.yml

# Run the H100 optimized training
./run_h100_experiment.sh
```

## 📝 Notes for AAAI Paper

- **Cite original papers**: CelebDF-v2 and FaceForensics++ papers
- **Mention preprocessing**: Face extraction using RetinaFace/face-recognition
- **Report dataset splits**: Standard train/val/test splits used
- **Acknowledge limitations**: Cross-dataset evaluation challenges

## 🆘 Need Help?

If you encounter issues:

1. **Check dataset paper repositories** for latest instructions
2. **Contact dataset authors** for access issues  
3. **Use academic networks** for preprocessed versions
4. **Consider subset experiments** for time constraints

The manual dataset setup is the only unavoidable step - everything else is automated! 🚀