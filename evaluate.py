import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from dataloader import get_dataloaders
from model_setup import build_model

def load_checkpoint_weights(model, checkpoint_path, device):
    """Loads weights robustly, handling 'model.' prefixes from Lightning."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("model.", "") 
        new_state_dict[name] = v
        
    if 'fc.weight' in new_state_dict:
        ckpt_classes = new_state_dict['fc.weight'].shape[0]
        current_classes = model.fc.out_features
        if ckpt_classes != current_classes:
            print(f"⚠️ Dimension Mismatch: Checkpoint has {ckpt_classes} outputs, Architecture has {current_classes}.")
            print(f"   -> Re-initializing model output layer to match checkpoint ({ckpt_classes}).")
            import torch.nn as nn
            model.fc = nn.Linear(model.fc.in_features, ckpt_classes)

    # Load weights
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"Note: {len(missing)} missing keys (likely optimizer/loss states, safe to ignore for inference).")
    print("Weights loaded successfully.")
    return model

def evaluate():
    # 1. Setup
    print("--- Initializing Evaluation ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Load Data
    _, _, test_loader, classes = get_dataloaders()
    print(f"Classes detected: {classes}")

    # 3. Initialize Model Architecture
    model, _, _ = build_model(num_classes=len(classes), gray_scale=False, freeze_backbone=False, lr=0.001)
    model = model.to(device)

    # 4. Load Trained Weights
    model = load_checkpoint_weights(model, 'pneumonia_model_with_hparams.pth', device)
    model.eval()

    # 5. Inference Loop
    all_preds = []
    all_labels = []
    all_probs = []

    print("Running Inference on Test Set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 6. Metrics & Visualization
    print("\n--- Results ---")
    print(classification_report(all_labels, all_preds, target_names=classes))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Saved: confusion_matrix.png")

    # ROC-AUC (Only calculated if exactly 2 classes found in checkpoint)
    if model.fc.out_features == 2:
        # Assuming index 1 is the positive class (PNEUMONIA)
        pos_probs = [p[1] for p in all_probs]
        auc_score = roc_auc_score(all_labels, pos_probs)
        print(f"ROC-AUC Score: {auc_score:.4f}")

        fpr, tpr, _ = roc_curve(all_labels, pos_probs)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig('roc_curve.png')
        print("Saved: roc_curve.png")

if __name__ == "__main__":
    evaluate()