# 👁 AutoDR — Diabetic Retinopathy Detection

[![Live Demo](https://img.shields.io/badge/Live_Demo-Hugging_Face-yellow?logo=huggingface)](https://huggingface.co/spaces/nisarga426/autodr-retinopathy)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> Automated diabetic retinopathy screening from fundus photographs
> using EfficientNet-B4 deep learning. Achieves **QWK 0.8688** on
> APTOS 2019 — exceeding the WHO clinical screening target of 0.80.

## 🔴 Live Demo
Upload any fundus image → AI grades DR severity (0–4) + confidence

**[▶ Try it here](https://huggingface.co/spaces/nisarga426/autodr-retinopathy)**

---

## 📊 Results

| Metric | Score | Clinical Target | Status |
|--------|-------|-----------------|--------|
| Quadratic Weighted Kappa (QWK) | **0.8688** | ≥ 0.80 | ✅ Passed |
| Training Epochs | 4 | — | — |
| Dataset | APTOS 2019 | 3,662 images | — |
| Training Hardware | Kaggle T4 GPU | — | — |

---

## 🏗 Architecture

- **Backbone:** EfficientNet-B4 (ImageNet pretrained, fine-tuned)
- **Head:** Linear(1792→512) → ReLU → Dropout(0.3) → Linear(512→5)
- **Loss:** Focal Loss (γ=2.0) — handles 73% grade-0 class imbalance
- **Augmentation:** CLAHE, random flips, rotations, colour jitter
- **Optimizer:** AdamW · Cosine LR schedule

## DR Grading Scale

| Grade | Label | Clinical Action |
|-------|-------|-----------------|
| 0 | No DR | Rescreen in 12 months |
| 1 | Mild NPDR | Monitor closely |
| 2 | Moderate NPDR | **Refer to ophthalmologist** |
| 3 | Severe NPDR | **Urgent referral** |
| 4 | Proliferative DR | **Emergency referral** |

---

## 🛠 Tech Stack

| Area | Tools |
|------|-------|
| Model | PyTorch · timm · EfficientNet-B4 |
| Training | Kaggle T4 GPU · Focal Loss · AdamW |
| Preprocessing | OpenCV · CLAHE · Albumentations |
| Backend | FastAPI · Uvicorn |
| Database | Supabase |
| Deployment | Streamlit · Hugging Face Spaces |
| Evaluation | scikit-learn · QWK · AUC-ROC |

---

## 🚀 Run Locally

```bash
git clone https://github.com/nisarga426/DR-project.git
cd DR-project
pip install -r requirements.txt

# Copy config template and fill in your keys
cp supabase_config.example.py supabase_config.py
# Edit supabase_config.py with your real Supabase keys

# Run the API
cd src/deployment
python auth_api.py

# In another terminal, run the frontend
streamlit run app.py
```

---

## ⚠️ Disclaimer
Research prototype only. Not a certified medical device.
Always consult a qualified ophthalmologist for clinical decisions.
