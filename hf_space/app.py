import streamlit as st
import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="AutoDR — Retinopathy Screening",
    page_icon="👁",
    layout="wide"
)

GRADES = {
    0: ("No DR",            "#27ae60", "✅ No diabetic retinopathy. Rescreen in 12 months."),
    1: ("Mild NPDR",        "#f39c12", "⚠️ Mild changes. Monitor closely."),
    2: ("Moderate NPDR",    "#e67e22", "🔴 Referable. Ophthalmologist consult recommended."),
    3: ("Severe NPDR",      "#e74c3c", "🚨 Urgent referral recommended."),
    4: ("Proliferative DR", "#8e44ad", "🚨 URGENT: Risk of vision loss. Immediate referral.")
}

class DRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4", pretrained=False, num_classes=0
        )
        self.head = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 5)
        )
    def forward(self, x):
        return self.head(self.backbone(x))

@st.cache_resource
def load_model():
    model = DRModel()
    state = torch.load("best_model.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

def preprocess(img_array):
    img = cv2.resize(img_array, (380, 380))
    b, g, r = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = cv2.merge([b, clahe.apply(g), r])
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0).float()

st.title("👁 AutoDR — Diabetic Retinopathy Screening")
st.markdown(
    "AI screening using **EfficientNet-B4** trained on APTOS 2019 · "
    "**QWK 0.8688** · exceeds WHO clinical screening target"
)
st.divider()

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Upload Fundus Image")
    uploaded = st.file_uploader(
        "Colour fundus photograph (JPG/PNG)",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded:
        img_pil = Image.open(uploaded).convert("RGB")
        st.image(img_pil, caption="Uploaded image", use_column_width=True)

with col2:
    st.subheader("AI Analysis")
    if not uploaded:
        st.info("Upload a fundus image on the left to begin.")
    else:
        with st.spinner("Analysing..."):
            img_np = np.array(img_pil)
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            tensor = preprocess(img_cv)
            model  = load_model()
            with torch.no_grad():
                logits = model(tensor)
                probs  = torch.softmax(logits, dim=1)[0]
                grade  = probs.argmax().item()
                conf   = probs.max().item()

        label, color, advice = GRADES[grade]
        st.markdown(f"### Grade {grade} — {label}")
        st.markdown(
            f'<div style="background:{color}22;border-left:4px solid {color};'
            f'padding:12px;border-radius:6px;font-size:15px;margin:8px 0">'
            f'{advice}</div>', unsafe_allow_html=True
        )
        col_a, col_b = st.columns(2)
        col_a.metric("Confidence", f"{conf*100:.1f}%")
        col_b.metric("Referable DR", "Yes 🔴" if grade >= 2 else "No ✅")

        st.markdown("#### Grade probabilities")
        for g, (lbl, col, _) in GRADES.items():
            p = float(probs[g])
            st.progress(p, text=f"Grade {g} — {lbl}: {p*100:.1f}%")

with st.sidebar:
    st.markdown("### Model Info")
    st.markdown("**Architecture:** EfficientNet-B4")
    st.markdown("**Dataset:** APTOS 2019 (3,662 images)")
    st.markdown("**QWK:** 0.8688")
    st.markdown("**Training:** Kaggle T4 GPU")
    st.divider()
    st.caption("⚠️ Research only. Not a certified medical device.")
