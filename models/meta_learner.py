import torch
import torch.nn as nn
import torch.nn.functional as F

class MetricGenerator(nn.Module):
    """
    Implements a Deep Relation Network for Learnable Metric Similarity in One-Shot Learning.

    Research Context:
    Traditional Few-Shot Learning approaches, such as Prototypical Networks, rely on 
    pre-defined distance metrics (e.g., Euclidean or Cosine distance) in the embedding space. 
    However, in fine-grained tasks like Offline Signature Verification, the manifold of 
    genuine signatures versus forgeries is often complex and non-linearly separable.

    This module implements a 'Relation Network' (based on Sung et al., CVPR 2018). 
    Instead of assuming a fixed metric, it learns a non-linear similarity function 
    (via a Multi-Layer Perceptron) that takes a pair of feature vectors (Support and Query) 
    and outputs a learnable relation score. This allows the model to capture subtle 
    intra-class variations (e.g., slant, stroke width) that rigid metrics might miss.

    Mathematical Formalism:
        Let f(x_s) and f(x_q) be the feature embeddings of the support and query images, respectively.
        The Relation Module g(.) computes the similarity score r as:
            r = g( Concat( f(x_s), f(x_q) ) )
        
        The network is trained end-to-end to maximize r for genuine pairs and minimize r for forged pairs.

    Attributes:
        embedding_dim (int): Dimensionality of the concatenated input features (Support + Query). 
                             Typically 2 * Backbone_Output_Dim.
        hidden_dim (int): Dimensionality of the hidden latent layer.
    """

    def __init__(self, embedding_dim=1024, hidden_dim=256, dropout=0.3):
        """
        Initializes the Relation Network architecture.

        Args:
            embedding_dim (int): The size of the combined feature vector. 
                                 For ResNet34 backbone (512 dim), this should be 512 + 512 = 1024.
            hidden_dim (int): The size of the hidden interaction layer. Default: 256.
            dropout (float): The dropout probability for regularization during training.
        """
        super(MetricGenerator, self).__init__()
        
        # Deep Relation Module (MLP)
        self.relation_module = nn.Sequential(
            # 1. Feature Interaction Layer
            # Projects the high-dimensional concatenated vector into a latent interaction space.
            nn.Linear(embedding_dim, hidden_dim),
            
            # 2. Normalization Strategy
            # LayerNorm is utilized instead of BatchNorm. In Meta-Learning scenarios (N-way K-shot),
            # batch sizes are often small or consist of episodic data, making BatchNorm statistics unstable.
            nn.LayerNorm(hidden_dim),
            
            # 3. Non-Linear Activation
            # ReLU introduces non-linearity, enabling the approximation of complex decision boundaries.
            nn.ReLU(),
            
            # 4. Regularization
            # Dropout is applied to prevent the relation module from overfitting to specific artifacts
            # in the training support sets, thereby improving cross-domain generalization.
            nn.Dropout(dropout),
            
            # 5. Scalar Scoring Layer
            # Projects the latent representation to a single scalar relation score (logit).
            # Note: No Sigmoid activation is applied here as BCEWithLogitsLoss is used 
            # in the training loop for better numerical stability.
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, combined_features):
        """
        Performs the forward pass to compute the similarity score (relation) between feature pairs.

        Args:
            combined_features (Tensor): The concatenated feature vectors of support and query images.
                                        Shape: [Batch_Size, embedding_dim]
                                        (e.g., [32, 1024])

        Returns:
            similarity_logits (Tensor): The raw similarity scores (logits) indicating the likelihood 
                                        that the pair belongs to the same identity.
                                        Shape: [Batch_Size, 1]
        """
        # Compute the non-linear relation score
        similarity_logits = self.relation_module(combined_features)
        
        return similarity_logits