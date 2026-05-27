import torch, cv2, numpy as np, os, base64
import torch.nn as nn, timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

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

def smart_decision(pred_grade, confidence):
    if pred_grade == 0 and confidence >= 0.75:
        return "CLEAR", "No DR detected. Return for annual screening.", "low"
    if pred_grade >= 2:
        urgency = "URGENT" if pred_grade >= 3 else "ROUTINE"
        return "REFER", f"{GRADES[pred_grade]} detected. Refer to specialist.", urgency
    if pred_grade == 1 and confidence >= 0.75:
        return "MONITOR", "Mild DR signs. Follow up in 6 months.", "low"
    if confidence < 0.75:
        return "HUMAN_REVIEW", \
               f"Uncertain ({confidence:.0%} confidence). Human review required.", "medium"
    return "MONITOR", "Borderline. Follow up in 6 months.", "low"

print("Loading DR model...")
model = DRModel().to(DEVICE)
model.load_state_dict(torch.load(
    os.path.expanduser('~/dr_project/best_model.pth'),
    map_location=DEVICE))
model.eval()
print("Model ready!")

TFM = A.Compose([
    A.Resize(300, 300),
    A.Normalize(mean=[0.485,0.456,0.406],
               std=[0.229,0.224,0.225]),
    ToTensorV2()])

app = FastAPI(title="AutoDR API")

# ── CORS fix ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    image_b64: str

@app.get("/")
def root():
    return {"message": "AutoDR API running",
            "qwk": 0.8832, "sensitivity": 0.9103}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(req: PredictRequest):
    try:
        img_bytes = base64.b64decode(req.image_b64)
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Cannot decode image")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise HTTPException(status_code=400,
                           detail=f"Image error: {str(e)}")

    tensor = TFM(image=img)["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    grade = int(probs.argmax())
    conf  = float(probs[grade])
    action, message, urgency = smart_decision(grade, conf)

    return {
        "grade":        grade,
        "grade_label":  GRADES[grade],
        "confidence":   round(conf, 4),
        "action":       action,
        "message":      message,
        "urgency":      urgency,
        "referable":    grade >= 2 or action == "HUMAN_REVIEW",
        "probabilities":{GRADES[i]: round(float(p), 4)
                        for i, p in enumerate(probs)}
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
