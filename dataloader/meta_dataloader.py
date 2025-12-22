import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import json
import random

# =============================================================================
# DATASET CLASS FOR META-LEARNING (EPISODIC TRAINING)
# =============================================================================

class SignatureEpisodeDataset(Dataset):
    """
    A PyTorch Dataset for N-Way K-Shot Meta-Learning tasks in Signature Verification.
    
    This dataset generates 'episodes' (tasks). Each episode consists of:
    1. Support Set: K genuine signatures (and optionally forgeries) for reference.
    2. Query Set: Signatures to be classified (Genuine vs Forgery).
    
    Attributes:
        split_file (str): Path to the JSON file containing the user split (train/val/test).
        n_ways (int): Number of classes (users) per episode (typically 1 for verification).
        k_shot (int): Number of reference signatures (support samples) per user.
        augment (bool): If True, applies data augmentation to the Support Set.
    """

    def __init__(self, split_file, root_dir=None, mode='train', k_shot=1, 
                 n_query_genuine=1, n_query_forgery=1, augment=False, use_full_path=False):
        """
        Initializes the meta-learning dataset.

        Args:
            split_file (str): Path to JSON file defining the data split.
            root_dir (str, optional): Base directory for images if paths are relative.
            mode (str): 'train' (meta-training) or 'test' (meta-validation).
            k_shot (int): Number of support samples (reference signatures).
            n_query_genuine (int): Number of genuine queries per episode.
            n_query_forgery (int): Number of forgery queries per episode.
            augment (bool): Whether to apply augmentation to the Support Set.
            use_full_path (bool): If True, treats paths in JSON as absolute.
        """
        self.k_shot = k_shot
        self.n_query_genuine = n_query_genuine
        self.n_query_forgery = n_query_forgery
        self.augment = augment
        self.use_full_path = use_full_path
        self.base_data_dir = root_dir if root_dir else ""
        self.mode = mode
        
        # Load user data from JSON split file
        with open(split_file, 'r') as f:
            raw_data = json.load(f)
        
        if mode in ['train', 'meta-train']:
            if 'meta-train' in raw_data:
                self.data = raw_data['meta-train']
                print(f" > [Dataset] Loaded subset 'meta-train' with {len(self.data)} users.")
            elif 'train' in raw_data:
                self.data = raw_data['train']
                print(f" > [Dataset] Loaded subset 'train' with {len(self.data)} users.")
            else:
                self.data = raw_data
                print(f" > [Dataset] Loaded flat dataset (Train mode) with {len(self.data)} users.")

        elif mode in ['val', 'test', 'meta-test', 'meta-val']:
            if 'meta-test' in raw_data:
                self.data = raw_data['meta-test']
                print(f" > [Dataset] Loaded subset 'meta-test' with {len(self.data)} users.")
            elif 'val' in raw_data:
                self.data = raw_data['val']
                print(f" > [Dataset] Loaded subset 'val' with {len(self.data)} users.")
            elif 'test' in raw_data:
                self.data = raw_data['test']
                print(f" > [Dataset] Loaded subset 'test' with {len(self.data)} users.")
            else:
                self.data = raw_data
                print(f" > [Dataset] Loaded flat dataset (Val mode) with {len(self.data)} users.")
        
        self.users = list(self.data.keys())
        if len(self.users) == 0:
             raise ValueError(f"Dataset rỗng! Kiểm tra lại file JSON {split_file} và mode {mode}")
             
        first_val = self.data[self.users[0]]
        if not isinstance(first_val, dict):
             raise ValueError(f"Cấu trúc JSON không hợp lệ. User {self.users[0]} không chứa dict ảnh.")

        # =========================================================================
        # TRANSFORMATION PIPELINES (Standardized for ResNet Input: 224x224)
        # =========================================================================
        
        # ImageNet Normalization Constants
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225])
        
        # 1. Support Set Transform (Training Data) -> Apply Augmentation if enabled
        # This helps the meta-learner generalize better from few examples.
        self.augment_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(15),  # Increased rotation for robustness
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            self.normalize
        ])
        
        # 2. Query Set Transform (Validation/Test Data) -> Deterministic
        # Only Resize and Normalize. No random distortions.
        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            self.normalize
        ])

    def __len__(self):
        # In meta-learning, an 'epoch' is arbitrary. We define it as the number of users.
        return len(self.users)

    def __getitem__(self, idx):
        """
        Generates one episode (task) for a specific user.
        
        Returns:
            dict: Contains 'support_images', 'query_images', 'query_labels', and 'user_id'.
        """
        user_id = self.users[idx]
        user_data = self.data[user_id]
        
        gen_key = next((k for k in user_data.keys() if k.lower() == 'genuine'), None)
        forg_key = next((k for k in user_data.keys() if k.lower() in ['forged', 'forgeries']), None)

        if not gen_key or not forg_key:
            return self.__getitem__(random.randint(0, len(self.users)-1))

        genuine_paths = user_data[gen_key]
        forgery_paths = user_data[forg_key]
        
        # --- SAMPLING STRATEGY ---
        
        # 1. Sample Support Set (Reference Signatures)
        # We need K genuine signatures.
        if len(genuine_paths) < self.k_shot:
            # If not enough samples, reuse with replacement
            support_paths = random.choices(genuine_paths, k=self.k_shot)
        else:
            support_paths = random.sample(genuine_paths, self.k_shot)
            
        # 2. Sample Query Set (Genuine)
        # Remaining genuine signatures not in support set
        remaining_gen = [p for p in genuine_paths if p not in support_paths]
        
        if len(remaining_gen) < self.n_query_genuine:
             # Fallback: sample from all genuine if we run out (rare case)
             query_gen_paths = random.choices(genuine_paths, k=self.n_query_genuine)
        else:
             query_gen_paths = random.sample(remaining_gen, self.n_query_genuine)
             
        # 3. Sample Query Set (Forgery)
        if len(forgery_paths) < self.n_query_forgery:
             query_forg_paths = random.choices(forgery_paths, k=self.n_query_forgery)
        else:
             query_forg_paths = random.sample(forgery_paths, self.n_query_forgery)

        # --- LOAD IMAGES ---
        
        # Support images: Apply augmentation only if self.augment is True (Training phase)
        support_imgs = self._load_batch(support_paths, augment=self.augment)
        
        # Query images: NEVER apply augmentation (Testing phase)
        query_imgs_gen = self._load_batch(query_gen_paths, augment=False)
        query_imgs_forg = self._load_batch(query_forg_paths, augment=False)
        
        # --- CONSTRUCT TENSORS ---
        
        # Concatenate Genuine and Forgery queries
        if len(query_imgs_forg) > 0:
            query_imgs = torch.cat([query_imgs_gen, query_imgs_forg], dim=0)
            # Labels: 1 for Genuine, 0 for Forgery
            labels_gen = torch.ones(len(query_imgs_gen), dtype=torch.float32)
            labels_forg = torch.zeros(len(query_imgs_forg), dtype=torch.float32)
            query_labels = torch.cat([labels_gen, labels_forg], dim=0)
        else:
            # Case: Only genuine queries (rare)
            query_imgs = query_imgs_gen
            query_labels = torch.ones(len(query_imgs_gen), dtype=torch.float32)

        return {
            'support_images': support_imgs,  # Shape: (K, C, H, W)
            'query_images': query_imgs,      # Shape: (N_Query, C, H, W)
            'query_labels': query_labels,    # Shape: (N_Query,)
            'user_id': str(user_id)
        }

    def _load_batch(self, paths, augment=False):
        """
        Helper function to load and process a batch of image paths.
        """
        images = []
        for path in paths:
            # Handle Path Resolution
            if self.use_full_path or os.path.isabs(path):
                full_path = path
            else:
                full_path = os.path.join(self.base_data_dir, path)
            
            try:
                # CRITICAL: Convert to RGB for ResNet compatibility
                img = Image.open(full_path).convert('RGB')
                
                # Apply appropriate transform
                if augment:
                    tensor = self.augment_transform(img)
                else:
                    tensor = self.base_transform(img)
                    
                images.append(tensor)
            except Exception as e:
                # In production, logging this error is better than printing
                # print(f"Error loading {full_path}: {e}")
                pass 
        
        if len(images) > 0:
            return torch.stack(images)
        else:
            # Return empty tensor if loading failed
            return torch.empty(0)