import torch
from torch.utils.data import Dataset
import os
import random
import re
from PIL import Image

class SignatureEpisodeDataset(Dataset):
    """
    Dataset for Few-Shot Learning (Meta-Learning) specifically designed to prevent Data Leakage.
    
    It generates 'Episodes' (Tasks) consisting of:
    - Support Set: k_shot genuine signatures from a specific user.
    - Query Set: q_query genuine signatures + q_query forged signatures from the same user.
    
    Key Feature:
    - specific_users: A list of User IDs allowed in this dataset. Ideally used to enforce
      strict splitting (e.g., Users 1-39 for Train, 40-44 for Val).
    """
    def __init__(self, org_dir, forg_dir, n_way=2, k_shot=5, q_query=5, specific_users=None, transform=None):
        """
        Args:
            org_dir (str): Path to the directory containing genuine signatures.
            forg_dir (str): Path to the directory containing forged signatures.
            n_way (int): Number of classes (usually 2 for Genuine vs Forgery, though Metric Learning often treats it as One-Class).
            k_shot (int): Number of genuine samples in Support Set.
            q_query (int): Number of samples per class in Query Set (e.g., 5 gen + 5 forg).
            specific_users (list[int]): List of allowed User IDs. If None, uses all found users.
            transform (callable): PyTorch transforms to apply to images.
        """
        self.org_dir = org_dir
        self.forg_dir = forg_dir
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.transform = transform
        self.specific_users = specific_users

        # Dictionary to store image paths: {user_id: {'org': [], 'forg': []}}
        self.user_data = {}
        
        # Supported image extensions
        self.valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.bmp')

        print(f"Initializing SignatureEpisodeDataset...")
        if specific_users:
            print(f"Filter active: Loading data ONLY for {len(specific_users)} specific users.")
        else:
            print("Warning: No user filter applied. Loading all users found.")

        # --- Load Genuine Signatures ---
        self._load_images(org_dir, 'org')
        
        # --- Load Forged Signatures ---
        self._load_images(forg_dir, 'forg')

        # Filter out users who don't have enough samples for k_shot + q_query
        self.users = []
        min_samples_required = k_shot + q_query
        
        for uid, data in self.user_data.items():
            n_org = len(data['org'])
            n_forg = len(data['forg'])
            
            # Condition: User must have enough genuine samples and at least some forgeries
            if n_org >= min_samples_required and n_forg >= q_query:
                self.users.append(uid)
            else:
                # Optional: Warning for skipped users
                # print(f"Skipping User {uid}: Insufficient data (Org: {n_org}, Forg: {n_forg})")
                pass
        
        self.users.sort()
        print(f"Dataset Loaded. Valid Users: {len(self.users)}")

    def _get_user_id(self, filename):
        """
        Extracts User ID from CEDAR filename format (e.g., 'original_1_1.png' -> 1).
        Adjust regex if using a different dataset naming convention.
        """
        # Regex for CEDAR/BHSig format usually involves underscores or hyphens
        # CEDAR: original_1_24.png -> User 1
        match = re.search(r'_(\d+)_', filename)
        if match:
            return int(match.group(1))
        
        # Fallback for BHSig or other formats if needed (e.g., BH-Sig uses hyphens)
        match = re.search(r'-(\d+)-', filename)
        if match:
             return int(match.group(1))
             
        return None

    def _load_images(self, directory, key):
        """Helper to crawl directory and populate self.user_data"""
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")

        for f in os.listdir(directory):
            if f.lower().endswith(self.valid_exts):
                uid = self._get_user_id(f)
                
                # Critical Logic: Filter by specific_users
                if uid is not None:
                    if self.specific_users is not None and uid not in self.specific_users:
                        continue # Skip this file if user is not in allowed list

                    if uid not in self.user_data:
                        self.user_data[uid] = {'org': [], 'forg': []}
                    
                    full_path = os.path.join(directory, f)
                    self.user_data[uid][key].append(full_path)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        """
        Generates one Episode (Support Set + Query Set) for a specific user.
        """
        user_id = self.users[idx]
        data = self.user_data[user_id]
        
        org_paths = data['org']
        forg_paths = data['forg']

        # 1. Sample Genuine Images (Support + Query)
        # We need k_shot + q_query distinct genuine images
        # random.sample guarantees unique elements
        selected_orgs = random.sample(org_paths, self.k_shot + self.q_query)
        
        support_paths = selected_orgs[:self.k_shot]
        query_org_paths = selected_orgs[self.k_shot:]
        
        # 2. Sample Forged Images (Query only)
        # We need q_query forged images
        selected_forgs = random.sample(forg_paths, self.q_query)
        query_forg_paths = selected_forgs

        # 3. Load Images and Apply Transforms
        support_imgs = self._load_batch(support_paths)
        query_org_imgs = self._load_batch(query_org_paths)
        query_forg_imgs = self._load_batch(query_forg_paths)

        # 4. Construct Query Set and Labels
        # Query Set structure: [Genuine Queries ..., Forged Queries ...]
        query_imgs = torch.cat([query_org_imgs, query_forg_imgs], dim=0)
        
        # Labels: 1 for Genuine, 0 for Forgery
        # Note: In Metric Learning with Triplet Loss, we might just use embeddings,
        # but for evaluation (Accuracy/AUC), we need these labels.
        org_labels = torch.ones(len(query_org_imgs), dtype=torch.long)
        forg_labels = torch.zeros(len(query_forg_imgs), dtype=torch.long)
        query_labels = torch.cat([org_labels, forg_labels], dim=0)

        return {
            'user_id': user_id,
            'support_images': support_imgs, # Shape: (k_shot, C, H, W)
            'query_images': query_imgs,     # Shape: (2*q_query, C, H, W)
            'query_labels': query_labels    # Shape: (2*q_query,)
        }

    def _load_batch(self, paths):
        """Loads a list of image paths into a tensor batch."""
        imgs = []
        for p in paths:
            try:
                img = Image.open(p).convert('L') # Ensure Grayscale
                if self.transform:
                    img = self.transform(img)
                imgs.append(img)
            except Exception as e:
                print(f"Error loading {p}: {e}")
                # Handling corruption: Create a black image as fallback to avoid crashing
                # In production, you might want to retry or skip.
                fallback = torch.zeros((1, 220, 150)) # Assuming rough size, transform usually resizes
                if self.transform: 
                    # Create dummy PIL image to pass through transform
                    fallback_pil = Image.new('L', (150, 220))
                    fallback = self.transform(fallback_pil)
                imgs.append(fallback)
                
        return torch.stack(imgs)