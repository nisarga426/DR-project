import torch, cv2, numpy as np, pandas as pd, os, sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (cohen_kappa_score, roc_auc_score,
                              classification_report, confusion_matrix)
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm, torch.nn as nn

sys.path.insert(0, os.path.expanduser('~/dr_project'))

DEVICE = torch.device("cpu")
GRADES = {0:"No DR", 1:"Mild", 2:"Moderate", 3:"Severe", 4:"Proliferative"}

# Model definition
class DRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4", pretrained=False, num_classes=0)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 5))
    def forward(self, x): return self.head(self.backbone(x))

# Dataset
class DRDataset(Dataset):
    def __init__(self, df, img_dir):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tfm = A.Compose([
            A.Resize(300, 300),
            A.Normalize(mean=[0.485,0.456,0.406],
                       std=[0.229,0.224,0.225]),
            ToTensorV2()])
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(f"{self.img_dir}/{row.id_code}.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.tfm(image=img)["image"]
        return img, torch.tensor(row.diagnosis, dtype=torch.long), row.id_code

def run_evaluation():
    print("Loading model...")
    model = DRModel().to(DEVICE)
    model.load_state_dict(torch.load(
        os.path.expanduser('~/dr_project/best_model.pth'),
        map_location=DEVICE))
    model.eval()
    print("Model loaded!")

    # Load validation data
    DATA_DIR = os.path.expanduser(
        '~/dr_project/data/raw/aptos2019')
    df = pd.read_csv(f"{DATA_DIR}/train.csv")
    from sklearn.model_selection import train_test_split
    _, val_df = train_test_split(df, test_size=0.15,
                                 stratify=df.diagnosis, random_state=42)
    print(f"Validation images: {len(val_df)}")

    ds = DRDataset(val_df, f"{DATA_DIR}/train_images")
    loader = DataLoader(ds, batch_size=8, shuffle=False)

    all_preds, all_labels, all_probs = [], [], []
    print("Running inference on validation set...")
    with torch.no_grad():
        for i, (imgs, labels, _) in enumerate(loader):
            logits = model(imgs.to(DEVICE))
            probs  = torch.softmax(logits, dim=1)
            preds  = probs.argmax(1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.numpy())
            if i % 10 == 0:
                print(f"  Processed {i*8}/{len(val_df)} images...")

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    # ── Metrics ──────────────────────────────────────────
    kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")

    # Sensitivity & Specificity at referable threshold (grade >= 2)
    ref_true = (all_labels >= 2).astype(int)
    ref_pred = (all_preds  >= 2).astype(int)
    tp = ((ref_true==1)&(ref_pred==1)).sum()
    fn = ((ref_true==1)&(ref_pred==0)).sum()
    tn = ((ref_true==0)&(ref_pred==0)).sum()
    fp = ((ref_true==0)&(ref_pred==1)).sum()
    sensitivity = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)

    # AUC
    try:
        auc = roc_auc_score(ref_true, all_probs[:,2:].sum(axis=1))
    except:
        auc = 0.0

    print("\n" + "="*55)
    print("       CLINICAL EVALUATION REPORT")
    print("="*55)
    print(f"  Quadratic Weighted Kappa (QWK): {kappa:.4f}")
    print(f"  Target: ≥ 0.80  →  {'✅ PASSED' if kappa>=0.80 else '❌ FAILED'}")
    print(f"\n  Sensitivity (referable DR):     {sensitivity:.4f}")
    print(f"  Target: ≥ 0.90  →  {'✅ PASSED' if sensitivity>=0.90 else '⚠️  CHECK'}")
    print(f"\n  Specificity:                    {specificity:.4f}")
    print(f"\n  AUC-ROC (referable vs not):     {auc:.4f}")
    print("="*55)

    print("\nPer-class breakdown:")
    print(classification_report(all_labels, all_preds,
          target_names=list(GRADES.values())))

    # ── Confusion Matrix ──────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(7,6))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(list(GRADES.values()), rotation=45, ha='right')
    ax.set_yticklabels(list(GRADES.values()))
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix  (QWK={kappa:.4f})')
    plt.colorbar(im)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(cm[i,j]),
                   ha='center', va='center',
                   color='white' if cm[i,j]>cm.max()//2 else 'black')
    plt.tight_layout()
    out = os.path.expanduser('~/dr_project/confusion_matrix.png')
    plt.savefig(out, dpi=150)
    print(f"\nConfusion matrix saved to: {out}")
    return kappa, sensitivity, specificity

if __name__ == "__main__":
    run_evaluation()
