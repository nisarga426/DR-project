import torch, cv2, numpy as np, pandas as pd, os, sys
import torch.nn as nn, timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

DEVICE = torch.device("cpu")
GRADES = {0:"No DR", 1:"Mild", 2:"Moderate", 3:"Severe", 4:"Proliferative"}

class DRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4", pretrained=False, num_classes=0)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 5))
    def forward(self, x): return self.head(self.backbone(x))

def analyze():
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
        A.Resize(300, 300),
        A.Normalize(mean=[0.485,0.456,0.406],
                   std=[0.229,0.224,0.225]),
        ToTensorV2()])

    print("Analyzing Severe DR cases (Grade 3)...")
    print("="*60)
    severe_df = val_df[val_df.diagnosis == 3]
    correct, wrong = 0, 0

    for _, row in severe_df.iterrows():
        img = cv2.imread(
            f"{DATA_DIR}/train_images/{row.id_code}.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = tfm(image=img)["image"].unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0]
            pred  = probs.argmax().item()
            conf  = probs[pred].item()
        status = "✅ CORRECT" if pred == 3 else f"❌ MISSED → predicted {GRADES[pred]}"
        print(f"  {row.id_code}: {status} (confidence: {conf:.0%})")
        if pred == 3: correct += 1
        else: wrong += 1

    print("="*60)
    print(f"Severe DR: {correct}/{len(severe_df)} correctly detected")
    print(f"Missed:    {wrong}/{len(severe_df)} cases")
    print()
    print("When Severe is missed, what does the model predict?")
    print("(If it predicts Moderate/Proliferative = still referable ✅)")
    print("(If it predicts No DR/Mild = DANGEROUS ❌)")

if __name__ == "__main__":
    analyze()
