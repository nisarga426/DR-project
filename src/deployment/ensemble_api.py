import torch, cv2, numpy as np, os, base64, io
import torch.nn as nn, timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
from datetime import datetime
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, white
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, Image as RLImage)
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
import uuid

DEVICE = torch.device("cpu")
GRADES = {0:"No DR",1:"Mild NPDR",2:"Moderate NPDR",
          3:"Severe NPDR",4:"Proliferative DR"}
GRADE_COLORS = {0:"#2e7d32",1:"#1565c0",2:"#e65100",
                3:"#b71c1c",4:"#4a0000"}
GRADE_DESC = {
    0:"No signs of diabetic retinopathy detected. Eye appears healthy.",
    1:"Mild NPDR. A few microaneurysms present. Monitor annually.",
    2:"Moderate NPDR. More microaneurysms, hemorrhages and exudates visible.",
    3:"Severe NPDR. Extensive hemorrhages, venous beading. High progression risk.",
    4:"Proliferative DR. New abnormal vessel growth detected. Urgent care needed."
}

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
    return "HUMAN_REVIEW", \
           f"Uncertain ({confidence:.0%} confidence). Human review required.", "medium"

# Load both models
print("Loading V1 model (QWK=0.8832)...")
model_v1 = DRModel().to(DEVICE)
model_v1.load_state_dict(torch.load(
    os.path.expanduser('~/dr_project/best_model.pth'),
    map_location=DEVICE))
model_v1.eval()

print("Loading V2 model (QWK=0.6979)...")
model_v2 = DRModel().to(DEVICE)
model_v2.load_state_dict(torch.load(
    os.path.expanduser('~/dr_project/best_model_v2.pth'),
    map_location=DEVICE))
model_v2.eval()
print("Ensemble ready! (Expected QWK=0.9519)")

TFM = A.Compose([
    A.Resize(300, 300),
    A.Normalize(mean=[0.485,0.456,0.406],
               std=[0.229,0.224,0.225]),
    ToTensorV2()])

target_layer = model_v1.backbone.conv_head
cam = GradCAM(model=model_v1, target_layers=[target_layer])

app = FastAPI(
    title="AutoDR Ensemble API v3.0",
    description="Ensemble DR Detection — QWK=0.9519")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                  allow_credentials=True,
                  allow_methods=["*"], allow_headers=["*"])

class PredictRequest(BaseModel):
    image_b64:    str
    patient_name: str = "Anonymous"
    patient_age:  str = "N/A"
    patient_id:   str = "N/A"
    generate_pdf: bool = True

def ensemble_predict(tensor):
    with torch.no_grad():
        probs_v1 = torch.softmax(model_v1(tensor), dim=1)[0]
        probs_v2 = torch.softmax(model_v2(tensor), dim=1)[0]
    probs = 0.65 * probs_v1 + 0.35 * probs_v2
    grade = int(probs.argmax())
    conf  = float(probs[grade])
    return grade, conf, probs

def make_gradcam(img_rgb, tensor):
    grayscale_cam = cam(input_tensor=tensor)[0]
    img_resized   = cv2.resize(img_rgb, (300, 300))
    img_float     = np.float32(img_resized) / 255.0
    heatmap       = show_cam_on_image(
        img_float, grayscale_cam, use_rgb=True)
    _, buf = cv2.imencode('.png',
                          cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode(), heatmap

def make_pdf(patient_name, patient_age, patient_id,
             grade, grade_label, confidence, action,
             message, probabilities, img_rgb, heatmap):
    rid      = str(uuid.uuid4())[:8].upper()
    ts       = datetime.now().strftime("%B %d, %Y at %H:%M")
    pdf_path = f"/tmp/AutoDR_{rid}.pdf"
    doc      = SimpleDocTemplate(pdf_path, pagesize=A4,
                                 rightMargin=2*cm, leftMargin=2*cm,
                                 topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    E = []

    hdr = Table([[
        Paragraph("<font size=20><b>AutoDR</b></font><br/>"
                 "<font size=9>Diabetic Retinopathy Screening Report</font>",
                 ParagraphStyle('hl', alignment=TA_LEFT,
                               textColor=white)),
        Paragraph(f"<font size=8>Report ID: {rid}<br/>"
                 f"Date: {ts}<br/>"
                 f"Ensemble QWK: 0.9519</font>",
                 ParagraphStyle('hr', alignment=TA_RIGHT,
                               textColor=white))
    ]], colWidths=[10*cm, 7*cm])
    hdr.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1),HexColor('#0f6e56')),
        ('PADDING',(0,0),(-1,-1),14),
    ]))
    E.append(hdr)
    E.append(Spacer(1, 0.4*cm))

    E.append(Paragraph("<b>Patient Information</b>",
                       ParagraphStyle('sh', fontSize=11,
                                     textColor=HexColor('#333'))))
    E.append(Spacer(1,0.2*cm))
    pt = Table([
        ["Patient Name", patient_name, "Patient ID", patient_id],
        ["Age", patient_age, "Date",
         datetime.now().strftime("%d/%m/%Y")]
    ], colWidths=[4*cm,6*cm,4*cm,3*cm])
    pt.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(0,-1),HexColor('#f0f4f8')),
        ('BACKGROUND',(2,0),(2,-1),HexColor('#f0f4f8')),
        ('FONTSIZE',(0,0),(-1,-1),10),
        ('PADDING',(0,0),(-1,-1),8),
        ('GRID',(0,0),(-1,-1),0.5,HexColor('#ddd')),
    ]))
    E.append(pt)
    E.append(Spacer(1,0.4*cm))

    E.append(Paragraph("<b>AI Ensemble Diagnosis</b>",
                       ParagraphStyle('sh', fontSize=11,
                                     textColor=HexColor('#333'))))
    E.append(Spacer(1,0.2*cm))
    action_labels = {
        "CLEAR":"✓ CLEAR — No action needed",
        "MONITOR":"◉ MONITOR — Follow up in 6 months",
        "REFER":"⚠ REFER TO SPECIALIST",
        "HUMAN_REVIEW":"⚑ HUMAN REVIEW REQUIRED"
    }
    action_col = {
        "CLEAR":"#2e7d32","MONITOR":"#1565c0",
        "REFER":"#e65100","HUMAN_REVIEW":"#b71c1c"
    }
    diag = Table([
        ["DR Grade",
         Paragraph(f"<font color='{GRADE_COLORS[grade]}'>"
                  f"<b>Grade {grade} — {grade_label}</b></font>",
                  styles['Normal'])],
        ["Confidence", f"{confidence:.1%}"],
        ["Action",
         Paragraph(f"<font color='{action_col.get(action,'#333')}'>"
                  f"<b>{action_labels.get(action,action)}</b></font>",
                  styles['Normal'])],
        ["Recommendation", message],
        ["Description", GRADE_DESC[grade]],
        ["Model", "Ensemble (V1 + V2) — QWK 0.9519"],
    ], colWidths=[5*cm, 12*cm])
    diag.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(0,-1),HexColor('#f0f4f8')),
        ('FONTSIZE',(0,0),(-1,-1),10),
        ('PADDING',(0,0),(-1,-1),10),
        ('GRID',(0,0),(-1,-1),0.5,HexColor('#ddd')),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('FONTNAME',(0,0),(0,-1),'Helvetica-Bold'),
    ]))
    E.append(diag)
    E.append(Spacer(1,0.4*cm))

    E.append(Paragraph("<b>Grade Probability Distribution</b>",
                       ParagraphStyle('sh', fontSize=11,
                                     textColor=HexColor('#333'))))
    E.append(Spacer(1,0.2*cm))
    risk = {0:"None",1:"Low",2:"Moderate",3:"High",4:"Critical"}
    prob_data = [["DR Grade","Probability","Risk"]] + [
        [f"Grade {i} — {GRADES[i]}", f"{p:.1%}", risk[i]]
        for i,(_, p) in enumerate(probabilities.items())]
    prob_t = Table(prob_data, colWidths=[7*cm,5*cm,5*cm])
    prob_t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),HexColor('#0f6e56')),
        ('TEXTCOLOR',(0,0),(-1,0),white),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,-1),10),
        ('PADDING',(0,0),(-1,-1),8),
        ('GRID',(0,0),(-1,-1),0.5,HexColor('#ddd')),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),
         [HexColor('#fff'),HexColor('#f9f9f9')]),
        ('BACKGROUND',(0,grade+1),(-1,grade+1),
         HexColor('#fff3e0')),
    ]))
    E.append(prob_t)
    E.append(Spacer(1,0.4*cm))

    E.append(Paragraph("<b>Retinal Images — Grad-CAM Explainability</b>",
                       ParagraphStyle('sh', fontSize=11,
                                     textColor=HexColor('#333'))))
    E.append(Spacer(1,0.2*cm))
    orig_p = f"/tmp/orig_{rid}.png"
    heat_p = f"/tmp/heat_{rid}.png"
    cv2.imwrite(orig_p, cv2.cvtColor(
        cv2.resize(img_rgb,(300,300)), cv2.COLOR_RGB2BGR))
    cv2.imwrite(heat_p, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
    img_t = Table([[
        RLImage(orig_p, width=7*cm, height=7*cm),
        RLImage(heat_p, width=7*cm, height=7*cm)
    ],[
        Paragraph("<i>Original retinal image</i>",
                 ParagraphStyle('c',alignment=TA_CENTER,fontSize=9)),
        Paragraph("<i>Grad-CAM: Red = AI focus areas</i>",
                 ParagraphStyle('c',alignment=TA_CENTER,fontSize=9))
    ]], colWidths=[8.5*cm,8.5*cm])
    img_t.setStyle(TableStyle([
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('PADDING',(0,0),(-1,-1),6),
    ]))
    E.append(img_t)
    E.append(Spacer(1,0.4*cm))

    E.append(Paragraph(
        "<i>Disclaimer: This AI report is for clinical decision support "
        "only and does not replace a qualified ophthalmologist. "
        "AutoDR Ensemble v3.0 | QWK=0.9519 | Sensitivity=91%</i>",
        ParagraphStyle('disc', fontSize=8,
                      textColor=HexColor('#888'),
                      alignment=TA_CENTER)))
    doc.build(E)
    return pdf_path, rid

@app.get("/")
def root():
    return {"message":"AutoDR Ensemble API v3.0",
            "ensemble_qwk": 0.9519,
            "v1_qwk": 0.8832,
            "v2_qwk": 0.6979,
            "sensitivity": 0.9103}

@app.get("/health")
def health():
    return {"status":"ok","models":["V1","V2"],"ensemble":True}

@app.post("/predict")
async def predict(req: PredictRequest):
    try:
        img_bytes = base64.b64decode(req.image_b64)
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Cannot decode image")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise HTTPException(400, f"Image error: {str(e)}")

    tensor = TFM(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)
    grade, conf, probs = ensemble_predict(tensor)
    action, message, urgency = smart_decision(grade, conf)
    probabilities = {GRADES[i]: round(float(p),4)
                    for i,p in enumerate(probs)}

    heatmap_b64, heatmap_img = make_gradcam(img_rgb, tensor)

    pdf_path, rid = None, None
    if req.generate_pdf:
        pdf_path, rid = make_pdf(
            patient_name=req.patient_name,
            patient_age=req.patient_age,
            patient_id=req.patient_id,
            grade=grade, grade_label=GRADES[grade],
            confidence=conf, action=action,
            message=message, probabilities=probabilities,
            img_rgb=img_rgb, heatmap=heatmap_img)

    return {
        "grade":         grade,
        "grade_label":   GRADES[grade],
        "confidence":    round(conf,4),
        "action":        action,
        "message":       message,
        "urgency":       urgency,
        "referable":     grade >= 2 or action == "HUMAN_REVIEW",
        "probabilities": probabilities,
        "gradcam_b64":   heatmap_b64,
        "report_id":     rid,
        "pdf_url":       f"/report/{rid}" if rid else None,
        "model":         "Ensemble V1+V2 (QWK=0.9519)"
    }

@app.get("/report/{report_id}")
async def get_report(report_id: str):
    pdf_path = f"/tmp/AutoDR_{report_id}.pdf"
    if not os.path.exists(pdf_path):
        raise HTTPException(404, "Report not found")
    return FileResponse(pdf_path,
                       media_type="application/pdf",
                       filename=f"AutoDR_Report_{report_id}.pdf")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
