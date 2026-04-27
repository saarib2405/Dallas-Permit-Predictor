# Dallas Building Permit Value Predictor

> **ASDS 6302 Final Project** — Predict construction permit values for the City of Dallas using a Two-Stage Hurdle Model with XGBoost, deployed as a REST API with an AI-powered Telegram chatbot.

**Live API:** [https://dallas-permit-predictor.onrender.com](https://dallas-permit-predictor.onrender.com)

---

## Project Overview

This project builds an end-to-end ML pipeline that predicts the declared construction value of Dallas building permits. It addresses a key challenge in the data: **23.7% of permits have $0 value** (administrative permits like signs and demolitions). A standard regression model can't handle this — so we use a **Two-Stage Hurdle Model**:

| Stage | Task | Model | Features | Performance |
|-------|------|-------|----------|-------------|
| **Stage 1** | Classify zero vs. positive value | XGBoost Classifier | 63 | Accuracy: 97.7%, F1: 0.965 |
| **Stage 2** | Predict log(Value) for positive permits | XGBoost Regressor | 62 | R²: 0.866, MAPE: 43.1% |

> Stage 2 uses 62 features (excludes `area_value_ratio`) to prevent data leakage — since `area_value_ratio = Area / Value` and Value is the prediction target.

---

## Project Structure

```
Dallas-Permit-Predictor/
│
├── app.py                    # Flask REST API (production server)
├── requirements.txt          # Python dependencies
├── Procfile                  # Render deployment config
├── runtime.txt               # Python version (3.11)
├── README.md
├── .gitignore
│
├── models/                   # Trained model artifacts
│   ├── stage1_classifier.joblib    # XGBoost binary classifier
│   ├── stage2_final_model.joblib   # XGBoost regressor
│   ├── scaler_cls.joblib           # StandardScaler for Stage 1
│   ├── scaler_reg.joblib           # StandardScaler for Stage 2
│   └── encoders.json               # Feature encoding maps
│
├── plots/                    # SHAP & clustering visualizations
│   ├── shap_beeswarm.png
│   ├── shap_bar_summary.png
│   ├── shap_waterfall_*.png        # Local explanations (3 files)
│   ├── shap_dependence_*.png       # Dependence plots (3 files)
│   ├── rf_feature_importance.png
│   ├── cluster_pca_scatter.png
│   ├── cluster_mean_value.png
│   └── kmeans_elbow_silhouette.png
│
├── src/                      # Pipeline source code
│   ├── data_cleaning.py            # Phase 1A: Raw CSV → cleaned dataset
│   ├── feature_engineering.py      # Phase 1B: Cleaned → 63-feature matrix
│   ├── modeling_pipeline.py        # Phase 2: Train/evaluate all models + SHAP
│   ├── save_encoders.py            # Export encoding maps for deployment
│   └── test_api.py                 # API integration tests
│
├── n8n/                      # AI Agent workflow
│   └── workflow.json               # Importable n8n workflow (Telegram bot)
│
└── docs/                     # Documentation
    ├── project_log.md              # 600-line decision log (WHAT + WHY)
    ├── cleaning_log.md             # Data cleaning audit trail
    ├── execution_guide.md          # Original project execution guide
    └── telegram_n8n_guide.md       # Telegram + n8n setup instructions
```

---

## Quick Start

### Run the API Locally
```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

### Make a Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "permit_type": "Building (BU) Commercial New Construction",
    "zip_code": "75201",
    "square_footage": 15000,
    "occupancy_type": "OFFICE BUILDING",
    "contractor_city": "DALLAS",
    "issue_month": 6,
    "issue_year": 2020
  }'
```

### Reproduce the Pipeline
```bash
cd src
python data_cleaning.py          # Step 1: Clean raw CSV
python feature_engineering.py    # Step 2: Engineer 63 features
python modeling_pipeline.py      # Step 3: Train models + SHAP + clustering
python save_encoders.py          # Step 4: Export encoders for API
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Interactive prediction form |
| `GET` | `/health` | Health check + model metadata |
| `POST` | `/predict` | Predict permit value (JSON input → JSON output) |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Models | XGBoost 2.0 + Optuna |
| Interpretability | SHAP (TreeExplainer) |
| API | Flask 3.1 + gunicorn |
| Hosting | Render (Free Tier) |
| AI Agent | n8n + Google Gemini 1.5 Flash |
| Chat Interface | Telegram Bot API |
| Tunnel | ngrok (for local webhook dev) |

---

## Dataset

- **Source:** City of Dallas Open Data Portal
- **Size:** 126,840 building permits (2018–2020)
- **Target:** Declared construction value (USD)
- **Key Challenge:** 23.7% of permits have $0 value (administrative permits)

---

## Key Results

- **Stage 1 Classifier:** 97.7% accuracy, 0.99 AUC-ROC — near-perfect separation of $0 vs. positive-value permits
- **Stage 2 Regressor:** R² = 0.866 — explains 86.6% of variance in construction costs
- **Top SHAP Features:** Area (sqft), permit_code_freq, work_type_Alteration
- **9 Permit Archetypes:** Identified via K-Means clustering (from low-value residential to mega-projects)

---

## Author

**Saarib** — ASDS 6302, University of Texas at Dallas
