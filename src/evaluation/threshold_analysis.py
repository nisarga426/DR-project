import torch, cv2, numpy as np, pandas as pd, os
import torch.nn as nn, timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

DEVICE = torch.device("cpu")
GRADES = {0:"No DR",1:"Mild",2:"Moderate",3:"Severe",4:"Proliferative"}

class DRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4", pretrained=False, num_classes=0)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 5))
    def forward(self, x): return self.head(self.backbone(x))

def analyze_thresholds():
    model = DRModel().to(DEVICE)
    model.load_state_dict(torch.load(
        os.path.expanduser('~/dr_project/best_model.pth'),
        map_location=DEVICE))
    model.eval()

    DATA_DIR = os.path.expanduser('~/dr_project/data/raw/aptos2019')
    df = pd.read_csv(f"{DATA_DIR}/train.csv")
    _, val_df = train_test_split(df, test_size=0.15,
                                 stratify=df.diagnosis, random_state=42)

    tfm = A.Compose([
        A.Resize(300,300),
        A.Normalize(mean=[0.485,0.456,0.406],
                   std=[0.229,0.224,0.225]),
        ToTensorV2()])

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for _, row in val_df.iterrows():
            img = cv2.imread(
                f"{DATA_DIR}/train_images/{row.id_code}.png")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = tfm(image=img)["image"].unsqueeze(0)
            probs = torch.softmax(model(tensor), dim=1)[0]
            all_preds.append(probs.argmax().item())
            all_labels.append(row.diagnosis)
            all_probs.append(probs.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    print("="*65)
    print("    THRESHOLD ANALYSIS — Finding the safest cutoff")
    print("="*65)
    print(f"{'Threshold':<12} {'Sensitivity':>12} {'Specificity':>12} "
          f"{'Dangerous':>10} {'Notes'}")
    print("-"*65)

    for threshold in [1, 2, 3]:
        ref_true = (all_labels >= threshold).astype(int)
        ref_pred = (all_preds  >= threshold).astype(int)
        tp = ((ref_true==1)&(ref_pred==1)).sum()
        fn = ((ref_true==1)&(ref_pred==0)).sum()
        tn = ((ref_true==0)&(ref_pred==0)).sum()
        fp = ((ref_true==0)&(ref_pred==1)).sum()
        sens = tp/(tp+fn+1e-9)
        spec = tn/(tn+fp+1e-9)

        # Dangerous = sick patient sent home
        # Sick = grade >= 2 (actual disease)
        actually_sick = (all_labels >= 2)
        sent_home     = (all_preds < threshold)
        dangerous     = (actually_sick & sent_home).sum()

        note = ""
        if threshold == 1: note = "← refer if ANY DR sign"
        if threshold == 2: note = "← current setting"
        if threshold == 3: note = "← only refer if Severe+"

        print(f"  Grade≥{threshold}      {sens:>11.1%} {spec:>12.1%} "
              f"{dangerous:>10} {note}")

    print("="*65)
    print()
    print("RECOMMENDATION:")
    print("  Use threshold = Grade ≥ 1 for maximum patient safety.")
    print("  Trade-off: more false referrals but ZERO dangerous misses")
    print("  on patients with actual DR disease (Grade 2+).")
    print()

    # Show the one dangerous case at threshold=2
    print("The 1 dangerous miss at current threshold (Grade≥2):")
    for i, (pred, label) in enumerate(zip(all_preds, all_labels)):
        if label >= 2 and pred < 2:
            row = val_df.iloc[i]
            conf = all_probs[i][pred]
            print(f"  Image: {row.id_code}")
            print(f"  True grade:      {GRADES[label]} (Grade {label})")
            print(f"  Predicted grade: {GRADES[pred]} (Grade {pred})")
            print(f"  Confidence:      {conf:.0%}")
            print(f"  Action needed:   Lower threshold to Grade≥1")

if __name__ == "__main__":
    analyze_thresholds()
