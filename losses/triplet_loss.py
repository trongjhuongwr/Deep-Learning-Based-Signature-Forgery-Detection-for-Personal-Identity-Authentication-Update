import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """
    Standard Triplet Loss with optimization to ignore zero-loss triplets in mean calculation.
    """
    def __init__(self, margin=1.0, mode='euclidean'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.mode = mode.lower()

    def forward(self, anchor, positive, negative):
        if self.mode == 'euclidean':
            dist_pos = F.pairwise_distance(anchor, positive, p=2)
            dist_neg = F.pairwise_distance(anchor, negative, p=2)
        elif self.mode == 'cosine':
            dist_pos = 1 - F.cosine_similarity(anchor, positive)
            dist_neg = 1 - F.cosine_similarity(anchor, negative)
        
        # Basic Hinge Loss
        losses = F.relu(dist_pos - dist_neg + self.margin)
        
        # --- OPTIMIZATION: Filter out zero losses before averaging ---
        # This prevents the gradient from being diluted by easy triplets
        valid_triplets = losses[losses > 1e-16]
        
        if len(valid_triplets) > 0:
            return torch.mean(valid_triplets)
        else:
            # If all are easy, return mean (which is 0) to keep graph connected
            return torch.mean(losses)