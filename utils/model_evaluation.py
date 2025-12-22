import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score
)
from tqdm.notebook import tqdm

# Set aesthetic style for all plots
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 11})

def compute_metrics(y_true, y_scores):
    """
    Computes a comprehensive set of biometric verification metrics.
    
    Args:
        y_true (array): Ground truth labels (1 for Genuine, 0 for Forged).
        y_scores (array): Similarity scores from the model (Higher = More Similar).
                          Range should be [0, 1] (Sigmoid output).
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores) 
    
    # 1. ROC Curve
    # pos_label=1 means we treat Genuine as the Positive class
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    roc_auc = auc(fpr, tpr)
    
    # 2. EER Calculation (Intersection of FAR and FRR)
    # EER is where False Positive Rate (FPR) == False Negative Rate (FNR)
    eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer = fpr[eer_idx]
    optimal_threshold = thresholds[eer_idx]
    
    # 3. Binary Classification Metrics at Optimal EER Threshold
    y_pred = (y_scores >= optimal_threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        'eer': eer,
        'auc': roc_auc,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'threshold': optimal_threshold,
        'fpr': fpr,
        'tpr': tpr,
        'y_true': y_true,
        'y_scores': y_scores,
        'y_pred': y_pred
    }

def evaluate_and_plot(feature_extractor, metric_generator, dataloader, device, save_dir):
    """
    Main evaluation pipeline for Relation Network.
    Executes the forward pass, computes metrics, and generates visualizations.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    feature_extractor.eval()
    metric_generator.eval()
    
    all_labels = []
    all_scores = []
    
    print(" > [Evaluation] Starting inference on Test Set...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            # 1. Unpack Data
            support_imgs = batch['support_images'].squeeze(1).to(device) # [B, C, H, W] (assuming K=1)
            query_imgs = batch['query_images'].to(device)                # [B, N_Q, C, H, W]
            labels = batch['query_labels'].to(device)                    # [B, N_Q]
            
            # 2. Handle Dimensions (Batch * N_Query)
            B, N_Q, C, H, W = query_imgs.shape
            
            # Flatten Query images
            query_imgs_flat = query_imgs.view(B * N_Q, C, H, W)
            labels_flat = labels.view(B * N_Q).unsqueeze(1)
            
            # Repeat Support images to match Query (Broadcasting logic)
            # [B, C, H, W] -> [B, 1, C, H, W] -> [B, N_Q, C, H, W] -> [B*N_Q, C, H, W]
            support_imgs_flat = support_imgs.unsqueeze(1).expand(-1, N_Q, -1, -1, -1).reshape(B * N_Q, C, H, W)
            
            # 3. Feature Extraction
            s_feats = feature_extractor(support_imgs_flat) # [B*N_Q, 512]
            q_feats = feature_extractor(query_imgs_flat)   # [B*N_Q, 512]
            
            # 4. Relation Network Logic (Concatenation)
            # Input dim becomes 1024 (512 + 512)
            combined_feats = torch.cat((s_feats, q_feats), dim=1)
            
            # 5. Metric Generation (Similarity Scoring)
            logits = metric_generator(combined_feats) # [B*N_Q, 1]
            probs = torch.sigmoid(logits)             # Convert to Probability [0, 1]
            
            # 6. Accumulate Results
            all_scores.extend(probs.cpu().numpy().flatten())
            all_labels.extend(labels_flat.cpu().numpy().flatten())
            
    # --- Compute Metrics ---
    results = compute_metrics(all_labels, all_scores)
    
    # --- Print Report ---
    print(f"\n{'='*10} FINAL EVALUATION REPORT {'='*10}")
    print(f"EER (Equal Error Rate) : {results['eer']:.2%}")
    print(f"AUC (Area Under Curve) : {results['auc']:.4f}")
    print(f"Optimal Threshold      : {results['threshold']:.4f}")
    print(f"Accuracy (at EER)      : {results['accuracy']:.2%}")
    print(f"Precision              : {results['precision']:.2%}")
    print(f"Recall                 : {results['recall']:.2%}")
    print(f"F1-Score               : {results['f1']:.2%}")
    print("="*40)
    
    # --- Visualizations ---
    _plot_roc_curve(results, save_dir)
    _plot_score_distribution(results, save_dir)
    _plot_confusion_matrix(results, save_dir)
    
    return results

def _plot_roc_curve(results, save_dir):
    plt.figure(figsize=(8, 6))
    plt.plot(results['fpr'], results['tpr'], color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {results["auc"]:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Mark EER Point
    plt.scatter([results['eer']], [1-results['eer']], color='red', s=100, zorder=5, 
                label=f'EER = {results["eer"]:.2%}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('True Positive Rate (TAR)')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    path = os.path.join(save_dir, 'roc_curve.png')
    plt.savefig(path)
    print(f" > Saved ROC Plot to: {path}")
    plt.close()

def _plot_score_distribution(results, save_dir):
    """Plots histogram of similarity scores for Genuine vs Forged pairs."""
    plt.figure(figsize=(10, 6))
    
    y_true = results['y_true']
    scores = results['y_scores']
    
    gen_scores = scores[y_true == 1]
    forg_scores = scores[y_true == 0]
    
    sns.histplot(gen_scores, color='green', label='Genuine Pairs', kde=True, stat="density", element="step", alpha=0.5)
    sns.histplot(forg_scores, color='red', label='Forged Pairs', kde=True, stat="density", element="step", alpha=0.5)
    
    # Draw Threshold Line
    plt.axvline(x=results['threshold'], color='blue', linestyle='--', linewidth=2, 
                label=f'Threshold = {results["threshold"]:.2f}')
    
    plt.xlabel('Similarity Score (0.0 = Different, 1.0 = Same)')
    plt.ylabel('Density')
    plt.title('Similarity Score Distribution')
    plt.legend()
    
    path = os.path.join(save_dir, 'score_distribution.png')
    plt.savefig(path)
    print(f" > Saved Distribution Plot to: {path}")
    plt.close()

def _plot_confusion_matrix(results, save_dir):
    cm = confusion_matrix(results['y_true'], results['y_pred'])
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Pred: Forged', 'Pred: Genuine'],
                yticklabels=['True: Forged', 'True: Genuine'])
    plt.title('Confusion Matrix (at EER Threshold)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(path)
    print(f" > Saved Confusion Matrix to: {path}")
    plt.close()

def visualize_hard_examples(feature_extractor, metric_generator, dataloader, device, save_dir, top_k=5):
    """
    Finds and saves the most confusing examples (False Positives & False Negatives).
    """
    os.makedirs(save_dir, exist_ok=True)
    feature_extractor.eval()
    metric_generator.eval()
    
    hard_positives = [] # Should be Same (1), but Score is Low (False Negative)
    hard_negatives = [] # Should be Diff (0), but Score is High (False Positive)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Mining Hard Examples", leave=False):
            # ... Data Unpacking (Same as evaluate_and_plot) ...
            support_imgs = batch['support_images'].squeeze(1).to(device)
            query_imgs = batch['query_images'].to(device)
            labels = batch['query_labels'].to(device)
            user_ids = batch['user_id'] # Assuming dataloader returns user_ids
            
            B, N_Q, C, H, W = query_imgs.shape
            
            # Prepare Logic
            query_flat = query_imgs.view(B * N_Q, C, H, W)
            labels_flat = labels.view(B * N_Q)
            support_flat = support_imgs.unsqueeze(1).expand(-1, N_Q, -1, -1, -1).reshape(B * N_Q, C, H, W)
            
            # Forward
            s_feats = feature_extractor(support_flat)
            q_feats = feature_extractor(query_flat)
            combined = torch.cat((s_feats, q_feats), dim=1)
            scores = torch.sigmoid(metric_generator(combined)).squeeze(1) # [B*N_Q]
            
            # --- Mining Logic ---
            for i in range(len(scores)):
                score = scores[i].item()
                label = labels_flat[i].item()
                
                # Get User ID for logging (Logic to map back from flattened index)
                # flattened_idx i -> batch_idx = i // N_Q
                batch_idx = i // N_Q
                uid = user_ids[batch_idx]
                
                # Image Tensors (CPU for saving)
                s_img = support_flat[i].cpu()
                q_img = query_flat[i].cpu()
                
                info = (score, label, uid, s_img, q_img)
                
                # False Negative: Label=1 (Genuine), Score Low
                if label == 1:
                    hard_positives.append(info)
                    
                # False Positive: Label=0 (Forged), Score High
                if label == 0:
                    hard_negatives.append(info)
    
    # Sort and Save Top-K
    # Hard Positives: Sort by Score Ascending (Lowest score is worst error)
    hard_positives.sort(key=lambda x: x[0])
    _save_example_images(hard_positives[:top_k], "FalseNegative", save_dir)
    
    # Hard Negatives: Sort by Score Descending (Highest score is worst error)
    hard_negatives.sort(key=lambda x: x[0], reverse=True)
    _save_example_images(hard_negatives[:top_k], "FalsePositive", save_dir)

def _save_example_images(example_list, prefix, save_dir):
    # Helper to save tensor images
    inv_normalize = lambda t: t * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    
    for idx, (score, label, uid, s_img, q_img) in enumerate(example_list):
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        
        # Un-normalize for display
        s_disp = inv_normalize(s_img).permute(1, 2, 0).numpy()
        q_disp = inv_normalize(q_img).permute(1, 2, 0).numpy()
        s_disp = np.clip(s_disp, 0, 1)
        q_disp = np.clip(q_disp, 0, 1)
        
        ax[0].imshow(s_disp)
        ax[0].set_title(f"Support (Ref)\nUser: {uid}")
        ax[0].axis('off')
        
        ax[1].imshow(q_disp)
        type_str = "Genuine" if label==1 else "Forged"
        ax[1].set_title(f"Query ({type_str})\nModel Score: {score:.4f}")
        ax[1].axis('off')
        
        plt.suptitle(f"Error Type: {prefix} (Confidence: {score:.2%})")
        plt.savefig(os.path.join(save_dir, f"{prefix}_{idx+1}_uid{uid}.png"))
        plt.close()
    print(f" > Saved {len(example_list)} hard examples for {prefix}.")