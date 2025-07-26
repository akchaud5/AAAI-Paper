import os
import random
from os import listdir
from os.path import join
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class FaceForensics(Dataset):
    """
    FaceForensics++ dataset implementation optimized for H100 testing
    Supports all 4 manipulation methods: Deepfakes, Face2Face, FaceSwap, NeuralTextures
    """
    def __init__(self, cfg):
        self.root = cfg['root']
        self.split = cfg['split']  # train/val/test
        self.compression = cfg.get('compression', 'c23')  # c0, c23, c40
        self.manipulation_types = cfg.get('manipulation_types', ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'])
        self.transforms = self.__get_transforms(cfg.get('transforms', []))
        self.images_ids = self.__get_images_ids()
        self.categories = {0: 'Fake', 1: 'Real'}

    def __get_transforms(self, transforms_cfg):
        transform_list = []
        for transform in transforms_cfg:
            if transform['name'] == 'Resize':
                transform_list.append(transforms.Resize((transform['params']['height'], transform['params']['width'])))
            elif transform['name'] == 'HorizontalFlip':
                transform_list.append(transforms.RandomHorizontalFlip(p=transform['params']['p']))
            elif transform['name'] == 'ColorJitter':
                params = transform['params']
                transform_list.append(transforms.ColorJitter(
                    brightness=params.get('brightness', 0),
                    contrast=params.get('contrast', 0),
                    saturation=params.get('saturation', 0),
                    hue=params.get('hue', 0)
                ))
            elif transform['name'] == 'Normalize':
                transform_list.append(transforms.ToTensor())
                transform_list.append(transforms.Normalize(mean=transform['params']['mean'], std=transform['params']['std']))
        
        if not any(isinstance(t, transforms.ToTensor) for t in transform_list):
            transform_list.insert(0, transforms.ToTensor())
        
        return transforms.Compose(transform_list)

    def __get_images_ids(self, limit=None):
        """
        Load FaceForensics++ dataset structure
        Expected structure:
        root/
        ├── original_sequences/
        │   └── youtube/
        │       └── c23/
        │           └── videos/
        ├── manipulated_sequences/
        │   ├── Deepfakes/
        │   │   └── c23/
        │   │       └── videos/
        │   ├── Face2Face/
        │   ├── FaceSwap/
        │   └── NeuralTextures/
        """
        images_ids = []
        
        try:
            # Load original (real) images
            original_path = join(self.root, 'original_sequences', 'youtube', self.compression, 'images')
            if os.path.exists(original_path):
                for split_dir in ['train', 'val', 'test']:
                    split_path = join(original_path, split_dir)
                    if os.path.exists(split_path) and split_dir == self.split:
                        for video_dir in listdir(split_path):
                            video_path = join(split_path, video_dir)
                            if os.path.isdir(video_path):
                                for img_file in listdir(video_path):
                                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                        rel_path = os.path.join('original_sequences', 'youtube', self.compression, 'images', split_dir, video_dir, img_file)
                                        images_ids.append((rel_path, 1))  # 1 for real
                                        if limit and len(images_ids) >= limit // 2:
                                            break
                            if limit and len(images_ids) >= limit // 2:
                                break
                    if limit and len(images_ids) >= limit // 2:
                        break

            # Load manipulated (fake) images for each manipulation type
            fake_per_method = (limit - len(images_ids)) // len(self.manipulation_types) if limit else None
            
            for manipulation in self.manipulation_types:
                manip_path = join(self.root, 'manipulated_sequences', manipulation, self.compression, 'images')
                if os.path.exists(manip_path):
                    method_count = 0
                    for split_dir in ['train', 'val', 'test']:
                        split_path = join(manip_path, split_dir)
                        if os.path.exists(split_path) and split_dir == self.split:
                            for video_dir in listdir(split_path):
                                video_path = join(split_path, video_dir)
                                if os.path.isdir(video_path):
                                    for img_file in listdir(video_path):
                                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                            rel_path = os.path.join('manipulated_sequences', manipulation, self.compression, 'images', split_dir, video_dir, img_file)
                                            images_ids.append((rel_path, 0))  # 0 for fake
                                            method_count += 1
                                            if fake_per_method and method_count >= fake_per_method:
                                                break
                                    if fake_per_method and method_count >= fake_per_method:
                                        break
                            if fake_per_method and method_count >= fake_per_method:
                                break
                        if fake_per_method and method_count >= fake_per_method:
                            break

        except OSError as e:
            print(f"Error accessing FaceForensics++ directories: {e}")
            print(f"Expected structure: {self.root}/[original_sequences|manipulated_sequences]/...")
            raise

        if not images_ids:
            raise ValueError(f"No images found in {self.root}. Please check the dataset structure and paths.")

        print(f"Loaded {len(images_ids)} images from FaceForensics++ ({len([x for x in images_ids if x[1] == 1])} real, {len([x for x in images_ids if x[1] == 0])} fake)")
        
        # Shuffle for better training dynamics
        random.shuffle(images_ids)
        return images_ids

    def __len__(self):
        return len(self.images_ids)

    def __getitem__(self, idx):
        image_path, label = self.images_ids[idx]
        full_path = join(self.root, image_path)
        
        try:
            image = self.__load_image(full_path)
            if self.transforms:
                image = self.transforms(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {full_path}: {e}")
            # Return a dummy image in case of error
            dummy_image = Image.new('RGB', (299, 299), color='black')
            if self.transforms:
                dummy_image = self.transforms(dummy_image)
            return dummy_image, label

    def __load_image(self, path):
        """Load image with error handling"""
        try:
            return Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Error opening image {path}: {e}")
            # Return a black dummy image
            return Image.new('RGB', (299, 299), color='black')

    def get_method_specific_data(self, method):
        """Get data for a specific manipulation method"""
        if method not in self.manipulation_types and method != 'original':
            raise ValueError(f"Method {method} not supported. Available: {self.manipulation_types + ['original']}")
        
        if method == 'original':
            return [(path, label) for path, label in self.images_ids if label == 1]
        else:
            return [(path, label) for path, label in self.images_ids if label == 0 and method in path]


class FaceForensicsMethodSpecific(FaceForensics):
    """
    FaceForensics++ dataset for testing specific manipulation methods
    """
    def __init__(self, cfg, method='Deepfakes'):
        self.method = method
        super().__init__(cfg)
        
        # Filter images for specific method
        if method == 'original':
            self.images_ids = [(path, label) for path, label in self.images_ids if label == 1]
        else:
            # Include all real images + specific fake method
            real_images = [(path, label) for path, label in self.images_ids if label == 1]
            fake_images = [(path, label) for path, label in self.images_ids if label == 0 and method in path]
            self.images_ids = real_images + fake_images
        
        print(f"Method-specific dataset created for {method}: {len(self.images_ids)} images")


# Factory function for easy dataset creation
def create_faceforensics_dataset(cfg, method=None):
    """
    Factory function to create FaceForensics++ dataset
    
    Args:
        cfg: Configuration dictionary
        method: Specific method to test ('Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'original')
                If None, returns all methods
    """
    if method:
        return FaceForensicsMethodSpecific(cfg, method)
    else:
        return FaceForensics(cfg)