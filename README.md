# 🌡️ Urban Heat Island Predictor
### ISRO × Hack2Skill 2026 — Problem Statement: Optimizing Urban Heat Mitigation via AI/ML

A geospatial AI/ML system that identifies urban heat stress hotspots, quantifies key drivers of urban heating, and generates optimized scenario-based cooling interventions — backed by physics-informed decision-making.

**Live Demo:** []

---

## 🏙️ Cities Covered
| City | Climate Zone | Data Period |
|------|-------------|-------------|
| Hyderabad | Semi-arid | Apr–Jun 2025 |
| Delhi | Hot semi-arid | Apr–Jun 2025 |
| Mumbai | Tropical wet | Mar–May 2025 |
| Chennai | Tropical wet-dry | Apr–Jun 2025 |

---

## 🔧 System Architecture

### Layer 1 — Data Pipeline (Google Earth Engine)
- **Landsat 8/9 C02 T1 L2** — Land Surface Temperature (ST_B10), spectral bands
- **ECMWF ERA5-Land** — Air temperature per city (monthly aggregates)
- Cloud masking via QA_PIXEL bitmask, scale factors applied per USGS spec
- 5,000 sampled points per city with spatial coordinates preserved

### Layer 2 — Feature Engineering
| Feature | Description | Source |
|---------|-------------|--------|
| LST | Land Surface Temperature (°C) | Landsat thermal |
| NDVI | Vegetation index | Landsat B5/B4 |
| NDWI | Water index | Landsat B3/B5 |
| NDBI | Built-up index | Landsat B6/B5 |
| MNDWI | Modified water index (SWIR) | Landsat B3/B6 |
| Albedo | Surface shortwave reflectance | Liang (2001) formula |

### Layer 3 — ML Models

#### XGBoost Driver Model
- Trained on **20,164 samples** across all 4 cities
- **R² = 0.713 | RMSE = 1.93°C**
- Per-city stratified spatial block split (prevents data leakage from neighboring pixels)
- SHAP values for feature attribution per prediction

#### Physics-Informed Neural Network (PINN)
- Architecture: 64 → 32 → 16 → 1 (ReLU activations)
- Physics constraint: evaporative cooling penalty (Penman–Monteith approximation)
- Trained weights: `pinn_model.pth` | **R² = 0.599 | RMSE = 3.02°C**
- Intentionally trades accuracy for physical consistency — penalizes predictions that violate energy balance constraints

### Layer 4 — Physics-Informed Constraints
- Surface energy balance equation: Rn = H + LE + G
- Evaporative cooling modeled via NDVI-based constraint
- PINN loss = Data MSE + 0.1 × Physics violation penalty

### Layer 5 — Scenario Engine
Three pre-defined cooling interventions with quantified ΔT:
- 🌳 **Plant Urban Forest** — NDVI +0.15, NDBI −0.05
- 🏠 **Install Cool Roofs** — NDWI +0.05, NDBI −0.10
- 💧 **Restore Water Body** — NDWI +0.12, NDBI −0.03

Custom scenario slider for arbitrary feature combinations.

### Layer 6 — Heat Stress Assessment
- **Heat Stress Index** — LST-adjusted approximation combining air temperature and radiant heat load
- Thresholds based on general outdoor heat stress guidelines:
  - 🟢 < 28°C — Low risk
  - 🟡 28–32°C — Moderate risk
  - 🟠 32–35°C — High risk
  - 🔴 > 35°C — Extreme risk

---

## 📈 UHI Temporal Trend (2017–2025)
| City | Trend | R² | p-value | Significant? |
|------|-------|----|---------|--------------|
| Hyderabad | −0.565°C/yr | 0.895 | 0.015 | ✅ Yes |
| Delhi | −0.733°C/yr | 0.597 | 0.126 | ❌ No |
| Mumbai | −0.200°C/yr | 0.256 | 0.494 | ❌ No |
| Chennai | −0.026°C/yr | 0.003 | 0.932 | ❌ No |

Hyderabad's statistically significant cooling trend likely reflects HMDA urban forestry programmes and green initiatives (2019–2025).

---

## 🚀 Deployment

### Requirements
```
streamlit
pandas
numpy
scikit-learn
xgboost
shap
plotly
joblib
folium
streamlit-folium
torch
scipy
```

### Run Locally
```bash
git clone <repo-url>
cd isro-uhi-predictor
pip install -r requirements.txt
streamlit run app.py
```

### Required Files
```
app.py
xgb_model.pkl
scaler.pkl
model_metadata.pkl
model_metadata.json (optional)
X_test_sample.pkl
pinn_model.pth
pinn_artifacts.pkl
uhi_trend_hyderabad.csv / _meta.json
uhi_trend_delhi.csv / _meta.json
uhi_trend_mumbai.csv / _meta.json
uhi_trend_chennai.csv / _meta.json
assets/ (folder containing UI images)
```

---

## 📁 Repository Structure
```
├── assets/                       # Background/Capsule image assets
├── app.py                        # Streamlit dashboard
├── FINALISRO.ipynb               # GEE data pipeline + model training
├── xgb_model.pkl                 # Trained XGBoost model
├── scaler.pkl                    # StandardScaler (spectral features only)
├── model_metadata.pkl/json       # Feature stats, city encodings, metrics
├── X_test_sample.pkl             # Scaled test sample for SHAP background
├── pinn_model.pth                # Trained PINN weights
├── pinn_artifacts.pkl            # PINN scalers and architecture info
├── uhi_trend_*.csv               # Per-city temporal trend data
├── uhi_trend_*_meta.json         # Per-city trend statistics
└── requirements.txt
```

---

## 🔬 Validation Notes
- Train/test split uses **per-city spatial block GroupShuffleSplit** — entire spatial blocks go to either train or test, never split across both, preventing leakage from spatially correlated pixels
- XGBoost trained on **scaled spectral features only** — `city_encoded` passed raw (categorical integer, not scaled)
- SHAP TreeExplainer background uses scaled data matching inference-time inputs
- PINN uses separate StandardScaler fitted only on training split

---

## 👨‍💻 Author
**Team-Antariksh Vision** —ISRO × Hack2Skill 2026 Submission
