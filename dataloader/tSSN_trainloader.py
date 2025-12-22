import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# =============================================================================
# TRANSFORMATION UTILITIES (AUGMENTATION STRATEGY)
# =============================================================================

def get_pretraining_transforms(input_shape=(224, 224)):
    """
    Generates a robust data augmentation pipeline for signature pre-training.
    
    This pipeline includes geometric and photometric transformations to induce 
    invariance to rotation, scale, stroke thickness, and lighting conditions.
    
    Args:
        input_shape (tuple): Target input resolution (H, W). Default is (224, 224) for ResNet.
        
    Returns:
        torchvision.transforms.Compose: The composition of transforms.
    """
    return transforms.Compose([
        # Resize inputs to the standard resolution expected by the backbone (e.g., ResNet34)
        transforms.Resize(input_shape),
        
        # 1. Geometric Invariance: Random Rotation
        # Signatures are rarely perfectly aligned. +/- 10 degrees simulates natural alignment noise.
        transforms.RandomRotation(degrees=10),
        
        # 2. Stroke & Perspective Invariance: Random Affine
        # Simulates different pen holding angles and slight distortions.
        # shear=5 alters the slant of the signature.
        transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.95, 1.05), shear=5),
        
        # 3. Environmental Invariance: Color Jitter
        # Randomizes brightness, contrast, and saturation to prevent the model from 
        # relying on specific ink colors or paper whiteness.
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        
        # 4. Quality Invariance: Gaussian Blur
        # Simulates low-resolution scanning artifacts or ink bleeding. 
        # Crucial for Cross-Domain generalization (e.g., matching CEDAR vs BHSig quality).
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        
        # Convert to Tensor (C, H, W) in range [0, 1]
        transforms.ToTensor(),
        
        # Normalize using ImageNet statistics (Mean and Std)
        # This is mandatory for initializing with pre-trained weights.
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# =============================================================================
# DATASET CLASS WITH HARD MINING
# =============================================================================

class SignaturePretrainDataset(Dataset):
    """
    A PyTorch Dataset class for Triplet-based Pre-training with Online Hard Negative Mining.
    
    This dataset generates triplets (Anchor, Positive, Negative) dynamically.
    It prioritizes 'Hard Negatives' (skilled forgeries of the same user) 
    over 'Easy Negatives' (random signatures from other users) to accelerate convergence.
    """
    
    def __init__(self, org_dir, forg_dir, transform=None, user_list=None):
        """
        Initializes the dataset by indexing all signature files.

        Args:
            org_dir (str): Path to the directory containing genuine signatures.
            forg_dir (str): Path to the directory containing forged signatures.
            transform (callable, optional): Transformations to apply to the images.
            user_list (list, optional): Filter specific user IDs (used for splitting Train/Val).
        """
        self.transform = transform
        self.org_images = []
        self.forg_images = []
        
        # --- File Indexing Strategy ---
        # Recursively search for image files to handle nested directory structures 
        # (common in BHSig and CEDAR datasets).
        valid_exts = ('.png', '.tif', '.jpg', '.jpeg')
        
        for root, _, files in os.walk(org_dir):
            for file in files:
                if file.lower().endswith(valid_exts):
                     self.org_images.append(os.path.join(root, file))
        
        for root, _, files in os.walk(forg_dir):
            for file in files:
                 if file.lower().endswith(valid_exts):
                     self.forg_images.append(os.path.join(root, file))

        # --- User Filtering ---
        if user_list is not None:
            user_list = set(str(u) for u in user_list)
            self.org_images = [x for x in self.org_images if self._get_user_id(os.path.basename(x)) in user_list]
            self.forg_images = [x for x in self.forg_images if self._get_user_id(os.path.basename(x)) in user_list]
        
        # Create a mapping: UserID -> List of Genuine Signature Paths
        self.user_genuine_map = {}
        for path in self.org_images:
            uid = self._get_user_id(os.path.basename(path))
            if uid not in self.user_genuine_map:
                self.user_genuine_map[uid] = []
            self.user_genuine_map[uid].append(path)
            
        self.users = list(self.user_genuine_map.keys())
        self.triplets = []
        self.on_epoch_end() # Initial triplet generation

    def _get_user_id(self, filename):
        """
        Extracts User ID from filename. 
        Assumes format like 'original_1_1.png' or '001_01.png'.
        Standardizes extraction using Regex.
        """
        # Matches the first sequence of digits found in the filename
        import re
        match = re.search(r'\d+', filename)
        if match:
            number = str(int(match.group(0)))
            if 'H-' in filename:
                return f"H-{number}"
            elif 'B-' in filename:
                return f"B-{number}"
            else:
                return number
        return "unknown"

    def on_epoch_end(self):
        """
        Regenerates triplets at the end of each epoch.
        This randomizes the pairings to prevent the model from overfitting to specific triplets.
        """
        self.triplets = []
        all_user_ids = list(self.user_genuine_map.keys())

        for anchor_path in self.org_images:
            anchor_uid = self._get_user_id(os.path.basename(anchor_path))
            
            # 1. Select Positive (Another genuine signature from the same user)
            positives = self.user_genuine_map.get(anchor_uid, [])
            # Need at least 2 genuine samples to form a pair
            if len(positives) < 2: 
                continue 
            
            # Ensure Positive is not the same file as Anchor
            possible_pos = [p for p in positives if p != anchor_path]
            if not possible_pos:
                continue
            positive_path = random.choice(possible_pos)

            # 2. Select Negative (Hard Mining Logic)
            # Strategy: 
            # - Hard Negative: A skilled forgery of the SAME user.
            # - Easy Negative: A genuine signature of a DIFFERENT user.
            
            current_forgeries = [f for f in self.forg_images if self._get_user_id(os.path.basename(f)) == anchor_uid]
            
            # Probability threshold: 70% chance to pick a Hard Negative (if available)
            is_hard_mining = (random.random() < 0.7) and (len(current_forgeries) > 0)
            
            if is_hard_mining:
                negative_path = random.choice(current_forgeries)
            else:
                # Pick a random user that is NOT the anchor user
                other_uid = random.choice([u for u in all_user_ids if u != anchor_uid])
                negatives_from_other = self.user_genuine_map.get(other_uid, [])
                if not negatives_from_other: continue
                negative_path = random.choice(negatives_from_other)
            
            self.triplets.append((anchor_path, positive_path, negative_path))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        """
        Retrieves a triplet item.
        Crucial: Converts images to RGB to match ResNet backbone requirements.
        """
        anchor_path, pos_path, neg_path = self.triplets[idx]
        
        # Load images
        # CONVERT TO RGB: This is critical for ResNet (expects 3 channels)
        anchor_img = Image.open(anchor_path).convert('RGB')
        pos_img = Image.open(pos_path).convert('RGB')
        neg_img = Image.open(neg_path).convert('RGB')

        # Apply Transforms (Augmentation)
        if self.transform:
            anchor = self.transform(anchor_img)
            pos = self.transform(pos_img)
            neg = self.transform(neg_img)
            
        # Return Triplet and a dummy label (TripletLoss doesn't use explicit labels)
        return anchor, pos, neg, torch.tensor([1], dtype=torch.float32)