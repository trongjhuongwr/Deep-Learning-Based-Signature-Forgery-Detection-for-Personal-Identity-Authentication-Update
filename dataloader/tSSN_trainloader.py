from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import os
import re
import numpy as np

class SignaturePretrainDataset(Dataset):
    """
    A PyTorch Dataset class for creating (Anchor, Positive, Negative) triplets
    for pre-training the signature feature extractor using standard Triplet Loss.

    This dataloader samples triplets based on user identity derived from filenames.
    """
    def __init__(self, org_dir, forg_dir, transform=None, specific_users=None):
        """
        Args:
            org_dir (str): Path to genuine signatures.
            forg_dir (str): Path to forged signatures.
            transform (callable): Augmentations/Transforms.
            specific_users (list of int): List of allowed User IDs (e.g., [1, 2, ..., 39]).
                                          If None, uses all found users (NOT recommended for strict splitting).
        """
        self.transform = transform
        self.specific_users = specific_users

        # 1. Load and Filter Images based on User ID
        self.org_images = self._load_and_filter(org_dir, specific_users)
        self.forg_images = self._load_and_filter(forg_dir, specific_users)

        print(f"Pre-training Dataset Stats: {len(self.org_images)} genuine, {len(self.forg_images)} forgeries.")
        
        if specific_users:
            print(f"Filtered for {len(specific_users)} specific users.")

        # 2. Generate Triplets
        self.triplets = self._create_triplets()
        print(f"Generated {len(self.triplets)} triplets.")

    def _get_user_id_from_filename(self, filename):
        """Extracts user ID from CEDAR filename (e.g., 'original_1_1.png' -> 1)."""
        match = re.search(r'_(\d+)_', filename)
        return int(match.group(1)) if match else None

    def _load_and_filter(self, directory, allowed_users):
        """Loads image paths and keeps only those belonging to allowed_users."""
        valid_paths = []
        supported_ext = ('.png', '.jpg', '.jpeg', '.tif', '.bmp')
        
        for f in os.listdir(directory):
            if f.lower().endswith(supported_ext):
                uid = self._get_user_id_from_filename(f)
                # Logic: If specific_users is defined, keep only if uid is in list.
                if allowed_users is None or (uid is not None and uid in allowed_users):
                    valid_paths.append(os.path.join(directory, f))
        return sorted(valid_paths)

    def _create_triplets(self):
        triplets = []
        user_genuine_map = {} 

        # Group genuine images by user
        for img_path in self.org_images:
            uid = self._get_user_id_from_filename(os.path.basename(img_path))
            if uid not in user_genuine_map: user_genuine_map[uid] = []
            user_genuine_map[uid].append(img_path)

        # Generate triplets
        for anchor_path in self.org_images:
            anchor_uid = self._get_user_id_from_filename(os.path.basename(anchor_path))
            
            # Positive: different genuine sample from same user
            pos_candidates = [p for p in user_genuine_map.get(anchor_uid, []) if p != anchor_path]
            if not pos_candidates: continue
            positive_path = random.choice(pos_candidates)

            # Negative: forged of same user OR genuine of different user
            # (Prioritize Hard Negatives: Forgeries of the same user)
            neg_candidates = []
            
            # 1. Forgeries (Hardest)
            current_user_forgeries = [
                f for f in self.forg_images 
                if self._get_user_id_from_filename(os.path.basename(f)) == anchor_uid
            ]
            neg_candidates.extend(current_user_forgeries)

            # 2. Other Users (Easier)
            other_users = [u for u in user_genuine_map.keys() if u != anchor_uid]
            if other_users:
                random_other_user = random.choice(other_users)
                neg_candidates.extend(user_genuine_map[random_other_user])

            if not neg_candidates: continue
            negative_path = random.choice(neg_candidates)

            triplets.append((anchor_path, positive_path, negative_path))
            
        return triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor, pos, neg = self.triplets[idx]
        try:
            a_img = Image.open(anchor).convert('L')
            p_img = Image.open(pos).convert('L')
            n_img = Image.open(neg).convert('L')
            if self.transform:
                a_img = self.transform(a_img)
                p_img = self.transform(p_img)
                n_img = self.transform(n_img)
            return a_img, p_img, n_img
        except Exception as e:
            return None