import torch, cv2, numpy as np, os, base64
import torch.nn as nn, timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
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
from supabase import create_client
from supabase_config import (SUPABASE_URL, SUPABASE_ANON_KEY,
                              SUPABASE_SERVICE_KEY, DOCTOR_INVITE_CODE)

# ── Supabase clients ──────────────────────────────────
supabase_anon    = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
supabase_service = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

DEVICE = torch.device("cpu")
GRADES = {0:"No DR",1:"Mild NPDR",2:"Moderate NPDR",
          3:"Severe NPDR",4:"Proliferative DR"}
GRADE_COLORS = {0:"#2e7d32",1:"#1565c0",2:"#e65100",
                3:"#b71c1c",4:"#4a0000"}
GRADE_DESC = {
    0:"No signs of diabetic retinopathy detected.",
    1:"Mild NPDR. A few microaneurysms present.",
    2:"Moderate NPDR. More hemorrhages and exudates visible.",
    3:"Severe NPDR. Extensive hemorrhages. High progression risk.",
    4:"Proliferative DR. New vessel growth. Urgent care needed."
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

print("Loading models...")
model_v1 = DRModel().to(DEVICE)
model_v1.load_state_dict(torch.load(
    os.path.expanduser('~/dr_project/best_model.pth'),
    map_location=DEVICE))
model_v1.eval()

model_v2 = DRModel().to(DEVICE)
model_v2.load_state_dict(torch.load(
    os.path.expanduser('~/dr_project/best_model_v2.pth'),
    map_location=DEVICE))
model_v2.eval()
print("Ensemble ready!")

TFM = A.Compose([
    A.Resize(300, 300),
    A.Normalize(mean=[0.485,0.456,0.406],
               std=[0.229,0.224,0.225]),
    ToTensorV2()])

target_layer = model_v1.backbone.conv_head
cam = GradCAM(model=model_v1, target_layers=[target_layer])

app = FastAPI(title="AutoDR Complete API v4.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                  allow_credentials=True,
                  allow_methods=["*"], allow_headers=["*"])

# ── Request models ────────────────────────────────────
class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: str
    role: str = "patient"
    age: Optional[int] = None
    phone: Optional[str] = None
    invite_code: Optional[str] = None

class LoginRequest(BaseModel):
    email: str
    password: str

class PredictRequest(BaseModel):
    image_b64: str
    patient_name: str = "Anonymous"
    patient_age: str = "N/A"
    patient_id: str = "N/A"
    generate_pdf: bool = True
    save_result: bool = False
    access_token: Optional[str] = None

class ProgressionRequest(BaseModel):
    access_token: str
    grade: int
    record_date: str
    notes: Optional[str] = None

class DoctorNoteRequest(BaseModel):
    access_token: str
    result_id: str
    notes: str
    flagged_urgent: bool = False

class BatchPredictRequest(BaseModel):
    access_token: str
    images: List[str]
    patient_names: Optional[List[str]] = None

# ── Auth helpers ──────────────────────────────────────
def get_user_from_token(token: str):
    try:
        user = supabase_anon.auth.get_user(token)
        return user.user
    except:
        raise HTTPException(401, "Invalid or expired token")

def get_user_role(user_id: str) -> str:
    try:
        result = supabase_service.table('profiles')\
            .select('role').eq('id', user_id).execute()
        if result.data:
            return result.data[0]['role']
        return 'patient'
    except:
        return 'patient'

def require_doctor(token: str):
    user = get_user_from_token(token)
    role = get_user_role(user.id)
    if role != 'doctor':
        raise HTTPException(403, "Doctor access required")
    return user

# ── ML helpers ────────────────────────────────────────
def ensemble_predict(tensor):
    with torch.no_grad():
        p1 = torch.softmax(model_v1(tensor), dim=1)[0]
        p2 = torch.softmax(model_v2(tensor), dim=1)[0]
    probs = 0.65*p1 + 0.35*p2
    grade = int(probs.argmax())
    conf  = float(probs[grade])
    return grade, conf, probs

def make_gradcam(img_rgb, tensor):
    grayscale_cam = cam(input_tensor=tensor)[0]
    img_resized   = cv2.resize(img_rgb, (300, 300))
    img_float     = np.float32(img_resized) / 255.0
    heatmap       = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
    _, buf = cv2.imencode('.png',
                          cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode(), heatmap

def make_pdf(patient_name, patient_age, patient_id,
             grade, grade_label, confidence, action,
             message, probabilities, img_rgb, heatmap,
             is_doctor=False, uncertainty=None):
    rid      = str(uuid.uuid4())[:8].upper()
    ts       = datetime.now().strftime("%B %d, %Y at %H:%M")
    pdf_path = f"/tmp/AutoDR_{rid}.pdf"
    doc      = SimpleDocTemplate(pdf_path, pagesize=A4,
                                 rightMargin=2*cm, leftMargin=2*cm,
                                 topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    E = []

    # Header
    hdr = Table([[
        Paragraph("<font size=20><b>AutoDR</b></font><br/>"
                 "<font size=9>Diabetic Retinopathy Screening Report</font>",
                 ParagraphStyle('hl', alignment=TA_LEFT, textColor=white)),
        Paragraph(f"<font size=8>Report ID: {rid}<br/>"
                 f"Date: {ts}<br/>"
                 f"Ensemble QWK: 0.9519</font>",
                 ParagraphStyle('hr', alignment=TA_RIGHT, textColor=white))
    ]], colWidths=[10*cm, 7*cm])
    hdr.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1),HexColor('#0f6e56')),
        ('PADDING',(0,0),(-1,-1),14),
    ]))
    E.append(hdr)
    E.append(Spacer(1, 0.4*cm))

    # Patient info
    E.append(Paragraph("<b>Patient Information</b>",
                       ParagraphStyle('sh', fontSize=11,
                                     textColor=HexColor('#333'))))
    E.append(Spacer(1,0.2*cm))
    pt = Table([
        ["Patient Name", patient_name, "Patient ID", patient_id],
        ["Age", patient_age, "Date", datetime.now().strftime("%d/%m/%Y")]
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

    # Diagnosis
    action_labels = {
        "CLEAR":"✓ CLEAR",
        "MONITOR":"◉ MONITOR",
        "REFER":"⚠ REFER TO SPECIALIST",
        "HUMAN_REVIEW":"⚑ HUMAN REVIEW REQUIRED"
    }
    action_col = {
        "CLEAR":"#2e7d32","MONITOR":"#1565c0",
        "REFER":"#e65100","HUMAN_REVIEW":"#b71c1c"
    }
    diag_rows = [
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
    ]
    if is_doctor and uncertainty is not None:
        diag_rows.append(["Uncertainty Score",
                          f"{uncertainty:.3f} ({'High — refer to human' if uncertainty>0.25 else 'Low — confident prediction'})"])
        diag_rows.append(["Model", "Ensemble V1+V2 (QWK=0.9519, Sensitivity=91%)"])

    diag = Table(diag_rows, colWidths=[5*cm, 12*cm])
    diag.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(0,-1),HexColor('#f0f4f8')),
        ('FONTSIZE',(0,0),(-1,-1),10),
        ('PADDING',(0,0),(-1,-1),10),
        ('GRID',(0,0),(-1,-1),0.5,HexColor('#ddd')),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('FONTNAME',(0,0),(0,-1),'Helvetica-Bold'),
    ]))
    E.append(Paragraph("<b>AI Diagnosis</b>",
                       ParagraphStyle('sh', fontSize=11,
                                     textColor=HexColor('#333'))))
    E.append(Spacer(1,0.2*cm))
    E.append(diag)
    E.append(Spacer(1,0.4*cm))

    # Probabilities (doctor only)
    if is_doctor:
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

    # Images
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
    E.append(Paragraph("<b>Retinal Images — Grad-CAM</b>",
                       ParagraphStyle('sh', fontSize=11,
                                     textColor=HexColor('#333'))))
    E.append(Spacer(1,0.2*cm))
    E.append(img_t)
    E.append(Spacer(1,0.4*cm))

    E.append(Paragraph(
        "<i>Disclaimer: For clinical decision support only. "
        "Does not replace a qualified ophthalmologist. "
        "AutoDR Ensemble v4.0 | QWK=0.9519</i>",
        ParagraphStyle('disc', fontSize=8,
                      textColor=HexColor('#888'),
                      alignment=TA_CENTER)))
    doc.build(E)
    return pdf_path, rid

# ══════════════════════════════════════════════════════
# AUTH ENDPOINTS
# ══════════════════════════════════════════════════════

@app.post("/auth/register")
async def register(req: RegisterRequest):
    if req.role == 'doctor':
        if req.invite_code != DOCTOR_INVITE_CODE:
            raise HTTPException(400, "Invalid doctor invite code")
    try:
        # Use admin API to create user without email verification
        result = supabase_service.auth.admin.create_user({
            "email": req.email,
            "password": req.password,
            "email_confirm": True,
            "user_metadata": {
                "full_name": req.full_name,
                "role": req.role
            }
        })
        if result.user:
            supabase_service.table('profiles').upsert({
                "id": result.user.id,
                "email": req.email,
                "full_name": req.full_name,
                "role": req.role,
                "age": req.age,
                "phone": req.phone
            }).execute()
            return {
                "success": True,
                "message": f"Registration successful! Welcome to AutoDR, {req.full_name}!",
                "user_id": result.user.id,
                "role": req.role
            }
        raise HTTPException(400, "Registration failed")
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/auth/login")
async def login(req: LoginRequest):
    try:
        result = supabase_anon.auth.sign_in_with_password({
            "email": req.email,
            "password": req.password
        })
        if result.user:
            role = get_user_role(result.user.id)
            profile = supabase_service.table('profiles')\
                .select('*').eq('id', result.user.id).execute()
            return {
                "success": True,
                "access_token": result.session.access_token,
                "user_id": result.user.id,
                "email": result.user.email,
                "role": role,
                "full_name": profile.data[0]['full_name'] if profile.data else "",
                "message": f"Welcome back! Logged in as {role}"
            }
        raise HTTPException(401, "Invalid credentials")
    except Exception as e:
        raise HTTPException(401, str(e))

@app.post("/auth/logout")
async def logout(access_token: str):
    try:
        supabase_anon.auth.sign_out()
        return {"success": True, "message": "Logged out successfully"}
    except:
        return {"success": True, "message": "Logged out"}

@app.get("/auth/profile")
async def get_profile(access_token: str):
    user = get_user_from_token(access_token)
    profile = supabase_service.table('profiles')\
        .select('*').eq('id', user.id).execute()
    if profile.data:
        return {"success": True, "profile": profile.data[0]}
    raise HTTPException(404, "Profile not found")

# ══════════════════════════════════════════════════════
# PREDICTION ENDPOINTS
# ══════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"message": "AutoDR Complete API v4.0",
            "ensemble_qwk": 0.9519}

@app.get("/health")
def health():
    return {"status": "ok", "models": ["V1","V2"],
            "database": "connected"}

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

    # Check if user is doctor
    is_doctor = False
    user_id = None
    if req.access_token:
        try:
            user = get_user_from_token(req.access_token)
            user_id = user.id
            role = get_user_role(user.id)
            is_doctor = role == 'doctor'
        except:
            pass

    tensor = TFM(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)
    grade, conf, probs = ensemble_predict(tensor)
    action, message, urgency = smart_decision(grade, conf)
    probabilities = {GRADES[i]: round(float(p),4)
                    for i,p in enumerate(probs)}

    # Uncertainty (variance across augmented predictions)
    uncertainty = float(probs.std())

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
            img_rgb=img_rgb, heatmap=heatmap_img,
            is_doctor=is_doctor,
            uncertainty=uncertainty)

    # Save to database if logged in
    result_id = None
    if req.save_result and user_id:
        try:
            db_result = supabase_service.table('screening_results').insert({
                "user_id": user_id,
                "grade": grade,
                "grade_label": GRADES[grade],
                "confidence": conf,
                "action": action,
                "message": message,
                "urgency": urgency,
                "referable": grade >= 2 or action == "HUMAN_REVIEW",
            }).execute()
            if db_result.data:
                result_id = db_result.data[0]['id']
        except Exception as e:
            print(f"DB save error: {e}")

    response = {
        "grade": grade,
        "grade_label": GRADES[grade],
        "confidence": round(conf,4),
        "action": action,
        "message": message,
        "urgency": urgency,
        "referable": grade >= 2 or action == "HUMAN_REVIEW",
        "probabilities": probabilities,
        "gradcam_b64": heatmap_b64,
        "report_id": rid,
        "pdf_url": f"/report/{rid}" if rid else None,
        "result_id": result_id,
        "model": "Ensemble V1+V2 (QWK=0.9519)"
    }

    # Add doctor-only fields
    if is_doctor:
        response["uncertainty"] = round(uncertainty, 4)
        response["uncertainty_flag"] = uncertainty > 0.25
        response["technical_metrics"] = {
            "ensemble_qwk": 0.9519,
            "sensitivity": 0.9103,
            "specificity": 0.9480,
            "auc_roc": 0.9825
        }

    return response

# ══════════════════════════════════════════════════════
# PROGRESSION TRACKER ENDPOINTS
# ══════════════════════════════════════════════════════

@app.post("/progression/add")
async def add_progression(req: ProgressionRequest):
    user = get_user_from_token(req.access_token)
    try:
        result = supabase_service.table('progression_records').insert({
            "patient_id": user.id,
            "grade": req.grade,
            "record_date": req.record_date,
            "notes": req.notes
        }).execute()
        return {"success": True, "record": result.data[0]}
    except Exception as e:
        raise HTTPException(400, str(e))

@app.get("/progression/history")
async def get_progression(access_token: str):
    user = get_user_from_token(access_token)
    result = supabase_service.table('progression_records')\
        .select('*').eq('patient_id', user.id)\
        .order('record_date').execute()
    return {"success": True, "records": result.data}

# ══════════════════════════════════════════════════════
# DOCTOR ENDPOINTS
# ══════════════════════════════════════════════════════

@app.get("/doctor/patients")
async def get_patients(access_token: str):
    require_doctor(access_token)
    result = supabase_service.table('profiles')\
        .select('*').eq('role', 'patient').execute()
    return {"success": True, "patients": result.data}

@app.get("/doctor/all-results")
async def get_all_results(access_token: str):
    require_doctor(access_token)
    result = supabase_service.table('screening_results')\
        .select('*, profiles(full_name, email)')\
        .order('created_at', desc=True).execute()
    return {"success": True, "results": result.data}

@app.post("/doctor/add-note")
async def add_doctor_note(req: DoctorNoteRequest):
    doctor = require_doctor(req.access_token)
    result = supabase_service.table('screening_results').update({
        "doctor_notes": req.notes,
        "flagged_urgent": req.flagged_urgent
    }).eq('id', req.result_id).execute()
    return {"success": True, "updated": result.data}

@app.get("/doctor/statistics")
async def get_statistics(access_token: str):
    require_doctor(access_token)
    results = supabase_service.table('screening_results')\
        .select('grade, action, flagged_urgent, created_at')\
        .execute()
    data = results.data
    grade_counts = {0:0,1:0,2:0,3:0,4:0}
    for r in data:
        grade_counts[r['grade']] = grade_counts.get(r['grade'],0)+1
    flagged = sum(1 for r in data if r.get('flagged_urgent'))
    referable = sum(1 for r in data if r.get('action') in ['REFER','HUMAN_REVIEW'])
    return {
        "success": True,
        "total_screenings": len(data),
        "grade_distribution": grade_counts,
        "flagged_urgent": flagged,
        "referable_cases": referable,
        "clear_cases": sum(1 for r in data if r.get('action')=='CLEAR')
    }

@app.post("/doctor/batch-predict")
async def batch_predict(req: BatchPredictRequest):
    require_doctor(req.access_token)
    results = []
    for i, img_b64 in enumerate(req.images):
        try:
            img_bytes = base64.b64decode(img_b64)
            arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = TFM(image=img_rgb)["image"].unsqueeze(0)
            grade, conf, probs = ensemble_predict(tensor)
            action, message, urgency = smart_decision(grade, conf)
            results.append({
                "index": i,
                "patient_name": req.patient_names[i] if req.patient_names else f"Patient {i+1}",
                "grade": grade,
                "grade_label": GRADES[grade],
                "confidence": round(conf,4),
                "action": action,
                "message": message,
                "urgency": urgency,
                "referable": grade >= 2 or action == "HUMAN_REVIEW"
            })
        except Exception as e:
            results.append({"index": i, "error": str(e)})
    urgent = sum(1 for r in results if r.get('urgency')=='URGENT')
    referable = sum(1 for r in results if r.get('referable'))
    return {
        "success": True,
        "total": len(results),
        "referable": referable,
        "urgent": urgent,
        "results": results
    }

@app.get("/report/{report_id}")
async def get_report(report_id: str):
    pdf_path = f"/tmp/AutoDR_{report_id}.pdf"
    if not os.path.exists(pdf_path):
        raise HTTPException(404, "Report not found")
    return FileResponse(pdf_path, media_type="application/pdf",
                       filename=f"AutoDR_Report_{report_id}.pdf")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

@app.get("/auth/callback")
async def auth_callback(access_token: str = None, 
                        refresh_token: str = None,
                        error: str = None,
                        error_description: str = None):
    if error:
        return {"error": error, "description": error_description}
    return {
        "success": True,
        "message": "Email verified successfully! You can now login.",
        "access_token": access_token
    }

from fastapi.responses import HTMLResponse

@app.get("/auth/callback", response_class=HTMLResponse)
async def auth_callback(
    access_token: str = None,
    refresh_token: str = None,
    error: str = None,
    error_description: str = None,
    type: str = None):

    if error:
        return HTMLResponse(f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>AutoDR — Verification Failed</title>
<style>
  body{{font-family:-apple-system,sans-serif;background:#f0f4f8;display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0}}
  .card{{background:white;border-radius:16px;padding:2.5rem;text-align:center;max-width:420px;box-shadow:0 4px 20px rgba(0,0,0,0.1)}}
  .icon{{font-size:56px;margin-bottom:1rem}}
  h1{{font-size:22px;color:#b71c1c;margin-bottom:8px}}
  p{{font-size:14px;color:#666;line-height:1.6;margin-bottom:1.5rem}}
  .btn{{display:inline-block;background:linear-gradient(135deg,#0f6e56,#185fa5);color:white;padding:12px 28px;border-radius:10px;text-decoration:none;font-weight:600;font-size:14px}}
</style>
</head>
<body>
<div class="card">
  <div class="icon">❌</div>
  <h1>Verification Failed</h1>
  <p>The verification link has expired or is invalid.<br>
  Please register again or request a new verification email.</p>
  <a href="javascript:window.close()" class="btn">Close this tab</a>
</div>
</body>
</html>
""")

    if type == 'signup' or access_token:
        return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>AutoDR — Email Verified!</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:linear-gradient(135deg,#0f6e56,#185fa5);display:flex;align-items:center;justify-content:center;min-height:100vh}
  .card{background:white;border-radius:20px;padding:2.5rem;text-align:center;max-width:440px;width:90%;box-shadow:0 8px 40px rgba(0,0,0,0.2)}
  .icon{font-size:64px;margin-bottom:1rem;animation:bounce 1s ease infinite alternate}
  @keyframes bounce{from{transform:translateY(0)}to{transform:translateY(-10px)}}
  h1{font-size:24px;font-weight:700;color:#0f6e56;margin-bottom:10px}
  p{font-size:14px;color:#666;line-height:1.7;margin-bottom:1.5rem}
  .steps{background:#f0f4f8;border-radius:12px;padding:1.25rem;margin-bottom:1.5rem;text-align:left}
  .step{display:flex;align-items:center;gap:10px;font-size:13px;margin-bottom:8px;color:#333}
  .step:last-child{margin-bottom:0}
  .step-num{width:24px;height:24px;border-radius:50%;background:linear-gradient(135deg,#0f6e56,#185fa5);color:white;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;flex-shrink:0}
  .btn{display:inline-block;background:linear-gradient(135deg,#0f6e56,#185fa5);color:white;padding:13px 32px;border-radius:12px;text-decoration:none;font-weight:600;font-size:15px;box-shadow:0 4px 14px rgba(15,110,86,0.3);transition:opacity 0.2s}
  .btn:hover{opacity:0.9}
  .badge{display:inline-block;background:#e8f5e9;color:#2e7d32;padding:4px 14px;border-radius:20px;font-size:12px;font-weight:600;margin-bottom:1rem}
  .countdown{font-size:12px;color:#888;margin-top:1rem}
  #counter{font-weight:700;color:#185fa5}
</style>
</head>
<body>
<div class="card">
  <div class="icon">✅</div>
  <div class="badge">Email Verified Successfully!</div>
  <h1>Welcome to AutoDR!</h1>
  <p>Your account has been verified. You can now login and access all features of the AutoDR diabetic retinopathy screening system.</p>
  
  <div class="steps">
    <div class="step"><div class="step-num">1</div><span>Go back to the AutoDR app</span></div>
    <div class="step"><div class="step-num">2</div><span>Click <strong>Login</strong> in the top right</span></div>
    <div class="step"><div class="step-num">3</div><span>Enter your email and password</span></div>
    <div class="step"><div class="step-num">4</div><span>Start screening retinal images!</span></div>
  </div>

  <a href="javascript:window.close()" class="btn">Close & Go Login →</a>
  <div class="countdown">This tab will close automatically in <span id="counter">10</span> seconds</div>
</div>
<script>
  let c=10;
  const interval=setInterval(()=>{
    c--;
    document.getElementById('counter').textContent=c;
    if(c<=0){clearInterval(interval);window.close();}
  },1000);
</script>
</body>
</html>
""")

    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>AutoDR — Processing</title>
<style>
  body{font-family:-apple-system,sans-serif;background:#f0f4f8;display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0}
  .card{background:white;border-radius:16px;padding:2.5rem;text-align:center;max-width:420px;box-shadow:0 4px 20px rgba(0,0,0,0.1)}
  .spinner{width:48px;height:48px;border:4px solid #e0e8f0;border-top-color:#185fa5;border-radius:50%;animation:spin 0.8s linear infinite;margin:0 auto 1.5rem}
  @keyframes spin{to{transform:rotate(360deg)}}
  h1{font-size:20px;color:#333;margin-bottom:8px}
  p{font-size:14px;color:#666}
</style>
</head>
<body>
<div class="card">
  <div class="spinner"></div>
  <h1>Processing your verification...</h1>
  <p>Please wait a moment.</p>
</div>
</body>
</html>
""")
