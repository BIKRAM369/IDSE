# 🏭 Industrial Predictive Maintenance — Interactive Web Platform

An interactive, guided web application for exploring the full data-science pipeline
for **industrial milling machine failure prediction**, built on the
[UCI AI4I 2020 dataset](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset).

The app runs the **actual Python ML pipeline** on a Flask backend and renders
**live, interactive Plotly.js charts** in a dark glassmorphism UI — no static images.

---

## ✨ Features

- **11 guided sections** mirroring the ML pipeline: Problem → Load → Preprocess → EDA → Split → Train → Tune → Evaluate → Features → Errors → Code Viewer
- **Per-section ▶ Run buttons** plus a top-level **Run All Pipeline** orchestrator
- **Live pipeline progress** indicator in the sidebar
- **Interactive Plotly charts**: donut, box plots, correlation heatmap, PR curves, confusion matrix, feature importance bars
- **Syntax-highlighted Python code viewer** (Prism.js) with copy button
- **Leakage-aware preprocessing**: removes `TWF/HDF/PWF/OSF/RNF` sub-failure flags and engineers physics-inspired features (`Temp_Diff_K`, `Power_W`, `Overstrain_Index`)
- **3 models** (Logistic Regression, Random Forest, Gradient Boosting) + GridSearchCV tuning on the best

---

## 🚀 Quick Start

### 1. Install backend dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Flask API

```bash
python app.py
```

The API runs at `http://localhost:5000` and serves the frontend automatically.
Open **http://localhost:5000** in your browser.

> ⚠️ **First run takes ~2–3 minutes**: the dataset (`ai4i2020.csv`, ~10k rows)
is downloaded from UCI on first use and cached locally, then all models are
trained. A loading overlay keeps you informed.

### Alternative: static frontend only

You can also open `templates/index.html` via any static server, but the Run
buttons require the Flask backend on `localhost:5000`.

---

## 📁 Project Structure

```
IDSE/
├── backend-equivalent files (flat layout)
│   ├── app.py            # Flask REST API — all /api/run/* endpoints
│   ├── analysis.py       # Stateful ML pipeline (load → … → errors)
│   └── requirements.txt
├── templates/
│   └── index.html        # SPA shell (sidebar + 11 sections)
├── static/
│   ├── css/style.css     # Dark glassmorphism theme
│   └── js/
│       ├── api.js        # Fetch wrappers for the REST API
│       ├── charts.js     # Plotly.js chart renderers
│       └── app.js        # SPA router, state, run orchestration, code viewer
└── README.md
```

---

## 🔌 API Endpoints

| Method | Endpoint              | Description                                  |
|--------|-----------------------|----------------------------------------------|
| GET    | `/api/status`         | Health check + pipeline step status          |
| POST   | `/api/run/load`       | Load & audit dataset (shape, dtypes, balance)|
| POST   | `/api/run/preprocess` | Leakage removal + feature engineering        |
| POST   | `/api/run/eda`        | EDA chart data (box plots, correlation, …)   |
| POST   | `/api/run/split`      | Stratified 70/15/15 split summary            |
| POST   | `/api/run/train`      | Train LR / RF / GBM, return CV + val metrics |
| POST   | `/api/run/tune`       | GridSearchCV on Gradient Boosting            |
| POST   | `/api/run/evaluate`   | PR curves, confusion matrix, report          |
| POST   | `/api/run/features`   | Feature importance data                      |
| POST   | `/api/run/errors`     | False negative / positive analysis           |
| POST   | `/api/run/all`        | Full pipeline end-to-end                     |

Steps are **stateful and ordered** — each endpoint requires the previous steps
to have run (the UI enforces this automatically).

---

## 🧠 Pipeline Highlights

- **Target**: `Machine failure` (binary, ~3.4% positive — heavily imbalanced)
- **Leakage removal**: `TWF, HDF, PWF, OSF, RNF` are logical-OR components of the target
- **Feature engineering**:
  - `Temp_Diff_K = Process temp − Air temp` (heat dissipation margin)
  - `Power_W = Torque × (RPM × 2π / 60)` (true mechanical power)
  - `Overstrain_Index = Tool wear × Torque` (cumulative strain)
- **Validation**: 5-fold stratified CV, F1 scoring; final metrics on held-out test set
- **Class imbalance** handled via `class_weight="balanced"` and PR-AUC evaluation

---

## 🛠 Tech Stack

| Layer    | Technology                                   |
|----------|----------------------------------------------|
| Backend  | Python, Flask, Flask-CORS                    |
| ML       | pandas, NumPy, scikit-learn                  |
| Frontend | Vanilla HTML/CSS/JS, Plotly.js, Prism.js     |
| Design   | Dark glassmorphism, teal/purple accents      |
