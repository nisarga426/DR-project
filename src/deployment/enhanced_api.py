import torch, cv2, numpy as np, os, base64, io, json
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
from reportlab.lib.units import inch, cm
from reportlab.lib.colors import HexColor, black, white
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, Image as RLImage)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import tempfile, uuid

DEVICE = torch.device("cpu")
GRADES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR"
}
GRADE_COLORS = {
    0: "#2e7d32",
    1: "#1565c0",
    2: "#e65100",
    3: "#b71c1c",
    4: "#4a0000"
}
GRADE_DESC = {
    0: "No signs of diabetic retinopathy detected.",
    1: "Mild non-proliferative DR. A few microaneurysms present.",
    2: "Moderate NPDR. More microaneurysms, hemorrhages, and exudates.",
    3: "Severe NPDR. Extensive hemorrhages, venous beading. High risk.",
    4: "Proliferative DR. New vessel growth. Risk of vision loss."
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

# GradCAM setup
target_layer = model.backbone.conv_head
cam = GradCAM(model=model, target_layers=[target_layer])

app = FastAPI(
    title="AutoDR Enhanced API",
    description="DR Detection with Grad-CAM + PDF Reports",
    version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    image_b64: str
    patient_name: str = "Anonymous"
    patient_age: str = "N/A"
    patient_id: str = "N/A"
    generate_pdf: bool = True

def generate_gradcam(img_rgb, tensor):
    """Generate Grad-CAM heatmap and return as base64."""
    grayscale_cam = cam(input_tensor=tensor)[0]
    img_resized = cv2.resize(img_rgb, (300, 300))
    img_float = np.float32(img_resized) / 255.0
    heatmap = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
    _, buffer = cv2.imencode('.png',
                             cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode(), heatmap

def generate_pdf_report(patient_name, patient_age, patient_id,
                        grade, grade_label, confidence, action,
                        message, urgency, probabilities,
                        original_img_rgb, heatmap_img):
    """Generate a professional PDF clinical report."""
    report_id = str(uuid.uuid4())[:8].upper()
    timestamp = datetime.now().strftime("%B %d, %Y at %H:%M")
    pdf_path = f"/tmp/AutoDR_Report_{report_id}.pdf"

    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                           rightMargin=2*cm, leftMargin=2*cm,
                           topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    elements = []

    # ── Header ───────────────────────────────────────────
    header_data = [[
        Paragraph("<font size=22><b>AutoDR</b></font><br/>"
                 "<font size=10 color='#666666'>Diabetic Retinopathy Detection System</font>",
                 ParagraphStyle('h', alignment=TA_LEFT)),
        Paragraph(f"<font size=9 color='#666666'>Report ID: {report_id}<br/>"
                 f"Generated: {timestamp}<br/>"
                 f"Model Version: v1.0 (QWK=0.8832)</font>",
                 ParagraphStyle('r', alignment=TA_RIGHT))
    ]]
    header_table = Table(header_data, colWidths=[10*cm, 7*cm])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), HexColor('#0f6e56')),
        ('TEXTCOLOR', (0,0), (-1,-1), white),
        ('PADDING', (0,0), (-1,-1), 14),
        ('ROUNDEDCORNERS', [8]),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 0.4*cm))

    # ── Patient Info ─────────────────────────────────────
    elements.append(Paragraph("<b>Patient Information</b>",
                             ParagraphStyle('s', fontSize=12,
                                           textColor=HexColor('#333333'))))
    elements.append(Spacer(1, 0.2*cm))
    patient_data = [
        ["Patient Name", patient_name,
         "Patient ID", patient_id],
        ["Age", patient_age,
         "Examination Date", datetime.now().strftime("%d/%m/%Y")],
    ]
    pt = Table(patient_data, colWidths=[4*cm, 6*cm, 4*cm, 3*cm])
    pt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), HexColor('#f5f5f5')),
        ('BACKGROUND', (2,0), (2,-1), HexColor('#f5f5f5')),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('PADDING', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, HexColor('#dddddd')),
    ]))
    elements.append(pt)
    elements.append(Spacer(1, 0.4*cm))

    # ── Diagnosis Result ─────────────────────────────────
    grade_color = HexColor(GRADE_COLORS[grade])
    elements.append(Paragraph("<b>AI Diagnosis Result</b>",
                             ParagraphStyle('s', fontSize=12,
                                           textColor=HexColor('#333333'))))
    elements.append(Spacer(1, 0.2*cm))

    action_icons = {
        "CLEAR": "✓ CLEAR",
        "MONITOR": "◉ MONITOR",
        "REFER": "⚠ REFER TO SPECIALIST",
        "HUMAN_REVIEW": "⚑ HUMAN REVIEW REQUIRED"
    }
    action_colors = {
        "CLEAR": "#2e7d32",
        "MONITOR": "#1565c0",
        "REFER": "#e65100",
        "HUMAN_REVIEW": "#b71c1c"
    }

    diag_data = [
        [Paragraph(f"<b>DR Grade</b>", styles['Normal']),
         Paragraph(f"<font color='{GRADE_COLORS[grade]}'>"
                  f"<b>Grade {grade} — {grade_label}</b></font>",
                  styles['Normal'])],
        [Paragraph("<b>Confidence</b>", styles['Normal']),
         Paragraph(f"<b>{confidence:.1%}</b>", styles['Normal'])],
        [Paragraph("<b>Clinical Action</b>", styles['Normal']),
         Paragraph(f"<font color='{action_colors.get(action, '#333')}'>"
                  f"<b>{action_icons.get(action, action)}</b></font>",
                  styles['Normal'])],
        [Paragraph("<b>Recommendation</b>", styles['Normal']),
         Paragraph(message, styles['Normal'])],
        [Paragraph("<b>Description</b>", styles['Normal']),
         Paragraph(GRADE_DESC[grade], styles['Normal'])],
    ]
    dt = Table(diag_data, colWidths=[5*cm, 12*cm])
    dt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), HexColor('#f0f4f8')),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('PADDING', (0,0), (-1,-1), 10),
        ('GRID', (0,0), (-1,-1), 0.5, HexColor('#dddddd')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    elements.append(dt)
    elements.append(Spacer(1, 0.4*cm))

    # ── Probability Table ─────────────────────────────────
    elements.append(Paragraph("<b>Grade Probability Distribution</b>",
                             ParagraphStyle('s', fontSize=12,
                                           textColor=HexColor('#333333'))))
    elements.append(Spacer(1, 0.2*cm))
    prob_header = [["DR Grade", "Probability", "Risk Level"]]
    risk_levels = {0:"None", 1:"Low", 2:"Moderate", 3:"High", 4:"Critical"}
    prob_rows = [[f"Grade {i} — {GRADES[i]}",
                  f"{prob:.1%}",
                  risk_levels[i]]
                 for i, (_, prob) in enumerate(probabilities.items())]
    prob_data = prob_header + prob_rows
    prob_table = Table(prob_data, colWidths=[7*cm, 5*cm, 5*cm])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), HexColor('#0f6e56')),
        ('TEXTCOLOR', (0,0), (-1,0), white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('PADDING', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, HexColor('#dddddd')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [HexColor('#ffffff'), HexColor('#f9f9f9')]),
        ('BACKGROUND', (0, grade+1), (-1, grade+1),
         HexColor('#fff3e0')),
    ]))
    elements.append(prob_table)
    elements.append(Spacer(1, 0.4*cm))

    # ── Images ───────────────────────────────────────────
    elements.append(Paragraph("<b>Retinal Analysis — Grad-CAM Explainability</b>",
                             ParagraphStyle('s', fontSize=12,
                                           textColor=HexColor('#333333'))))
    elements.append(Spacer(1, 0.2*cm))

    # Save images temporarily
    orig_path = f"/tmp/orig_{report_id}.png"
    heat_path = f"/tmp/heat_{report_id}.png"
    cv2.imwrite(orig_path,
               cv2.cvtColor(cv2.resize(original_img_rgb, (300,300)),
                           cv2.COLOR_RGB2BGR))
    cv2.imwrite(heat_path,
               cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2BGR))

    img_data = [[
        RLImage(orig_path, width=7*cm, height=7*cm),
        RLImage(heat_path, width=7*cm, height=7*cm)
    ],[
        Paragraph("<i>Original retinal image</i>",
                 ParagraphStyle('c', alignment=TA_CENTER, fontSize=9)),
        Paragraph("<i>Grad-CAM: Red areas show where AI focused</i>",
                 ParagraphStyle('c', alignment=TA_CENTER, fontSize=9))
    ]]
    img_table = Table(img_data, colWidths=[8.5*cm, 8.5*cm])
    img_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    elements.append(img_table)
    elements.append(Spacer(1, 0.4*cm))

    # ── Disclaimer ───────────────────────────────────────
    elements.append(Paragraph(
        "<i>Disclaimer: This report is generated by an AI screening tool "
        "and is intended for clinical decision support only. It does not "
        "replace the judgment of a qualified ophthalmologist. All findings "
        "should be confirmed by a licensed medical professional before "
        "clinical action is taken. AutoDR v1.0 | QWK=0.8832 | "
        "Sensitivity=91.03% | AUC=0.9825</i>",
        ParagraphStyle('d', fontSize=8, textColor=HexColor('#888888'),
                      alignment=TA_CENTER)))

    doc.build(elements)
    return pdf_path, report_id

@app.get("/")
def root():
    return {"message": "AutoDR Enhanced API v2.0",
            "features": ["diagnosis", "grad-cam", "pdf-report"],
            "qwk": 0.8832, "sensitivity": 0.9103}

@app.get("/health")
def health():
    return {"status": "ok", "model": "EfficientNet-B4"}

@app.post("/predict")
async def predict(req: PredictRequest):
    # Decode image
    try:
        img_bytes = base64.b64decode(req.image_b64)
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Cannot decode image")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise HTTPException(400, f"Image error: {str(e)}")

    # Preprocess
    tensor = TFM(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    grade = int(probs.argmax())
    conf  = float(probs[grade])
    action, message, urgency = smart_decision(grade, conf)
    probabilities = {GRADES[i]: round(float(p), 4)
                    for i, p in enumerate(probs)}

    # Generate Grad-CAM
    heatmap_b64, heatmap_img = generate_gradcam(img_rgb, tensor)

    # Generate PDF report
    pdf_path, report_id = None, None
    if req.generate_pdf:
        pdf_path, report_id = generate_pdf_report(
            patient_name=req.patient_name,
            patient_age=req.patient_age,
            patient_id=req.patient_id,
            grade=grade,
            grade_label=GRADES[grade],
            confidence=conf,
            action=action,
            message=message,
            urgency=urgency,
            probabilities=probabilities,
            original_img_rgb=img_rgb,
            heatmap_img=heatmap_img
        )

    return {
        "grade":         grade,
        "grade_label":   GRADES[grade],
        "confidence":    round(conf, 4),
        "action":        action,
        "message":       message,
        "urgency":       urgency,
        "referable":     grade >= 2 or action == "HUMAN_REVIEW",
        "probabilities": probabilities,
        "gradcam_b64":   heatmap_b64,
        "report_id":     report_id,
        "pdf_available": pdf_path is not None,
        "pdf_url":       f"/report/{report_id}" if report_id else None
    }

@app.get("/report/{report_id}")
async def get_report(report_id: str):
    pdf_path = f"/tmp/AutoDR_Report_{report_id}.pdf"
    if not os.path.exists(pdf_path):
        raise HTTPException(404, "Report not found or expired")
    return FileResponse(pdf_path,
                       media_type="application/pdf",
                       filename=f"AutoDR_Report_{report_id}.pdf")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
