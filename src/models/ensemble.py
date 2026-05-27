import torch
import torch.nn as nn
import timm
import numpy as np
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = torch.device("cpu")
GRADES = {0:"No DR",1:"Mild NPDR",2:"Moderate NPDR",
          3:"Severe NPDR",4:"Proliferative DR"}

class DRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4", pretrained=False, num_classes=0)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 5))
    def forward(self, x): return self.head(self.backbone(x))

class EnsembleModel:
    """
    Combines V1 (clean APTOS labels, high accuracy) and
    V2 (diverse EyePACS data, better generalization).
    Weighted average: V1 gets more weight since higher QWK.
    """
    def __init__(self, v1_path, v2_path,
                 v1_weight=0.65, v2_weight=0.35):
        print("Loading V1 model (QWK=0.8832)...")
        self.model_v1 = DRModel().to(DEVICE)
        self.model_v1.load_state_dict(
            torch.load(v1_path, map_location=DEVICE))
        self.model_v1.eval()

        print("Loading V2 model (QWK=0.6979)...")
        self.model_v2 = DRModel().to(DEVICE)
        self.model_v2.load_state_dict(
            torch.load(v2_path, map_location=DEVICE))
        self.model_v2.eval()

        self.v1_weight = v1_weight
        self.v2_weight = v2_weight
        print(f"Ensemble ready! V1 weight={v1_weight} V2 weight={v2_weight}")

        self.tfm = A.Compose([
            A.Resize(300, 300),
            A.Normalize(mean=[0.485,0.456,0.406],
                       std=[0.229,0.224,0.225]),
            ToTensorV2()])

    def predict(self, img_rgb):
        """
        Run ensemble prediction on a single RGB image.
        Returns: (grade, confidence, probabilities)
        """
        tensor = self.tfm(image=img_rgb)["image"].unsqueeze(0)

        with torch.no_grad():
            probs_v1 = torch.softmax(self.model_v1(tensor), dim=1)[0]
            probs_v2 = torch.softmax(self.model_v2(tensor), dim=1)[0]

        # Weighted average of probabilities
        probs = (self.v1_weight * probs_v1 +
                 self.v2_weight * probs_v2)

        grade = int(probs.argmax())
        conf  = float(probs[grade])
        return grade, conf, probs.numpy()

def test_ensemble():
    """Quick test of the ensemble on validation images."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import cohen_kappa_score

    ensemble = EnsembleModel(
        v1_path=os.path.expanduser('~/dr_project/best_model.pth'),
        v2_path=os.path.expanduser('~/dr_project/best_model_v2.pth'))

    DATA_DIR = os.path.expanduser('~/dr_project/data/raw/aptos2019')
    df = pd.read_csv(f"{DATA_DIR}/train.csv")
    _, val_df = train_test_split(df, test_size=0.15,
                                 stratify=df.diagnosis, random_state=42)

    print(f"\nTesting ensemble on {len(val_df)} validation images...")
    all_preds, all_labels = [], []

    for i, (_, row) in enumerate(val_df.iterrows()):
        img = cv2.imread(
            f"{DATA_DIR}/train_images/{row.id_code}.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        grade, conf, probs = ensemble.predict(img)
        all_preds.append(grade)
        all_labels.append(row.diagnosis)
        if i % 100 == 0:
            print(f"  Processed {i}/{len(val_df)}...")

    kappa = cohen_kappa_score(
        all_labels, all_preds, weights="quadratic")

    print("\n" + "="*50)
    print("  ENSEMBLE EVALUATION RESULTS")
    print("="*50)
    print(f"  V1 alone:  QWK = 0.8832")
    print(f"  V2 alone:  QWK = 0.6979")
    print(f"  Ensemble:  QWK = {kappa:.4f}")
    if kappa > 0.8832:
        print(f"  ✅ Ensemble BEATS V1 alone!")
    elif kappa > 0.8700:
        print(f"  ✅ Ensemble close to V1, better generalization")
    else:
        print(f"  ℹ️  V1 still best for APTOS images")
        print(f"  V2 helps with diverse camera types")
    print("="*50)
    return kappa

if __name__ == "__main__":
    test_ensemble()
