# Insulin Matrix AI

**Diabetes screening & classification platform (demonstration only — not a medical diagnosis)**

Full-stack local app for VS Code / local development:

- **Frontend:** React, TypeScript, Vite, Tailwind CSS, Recharts, Lucide
- **Backend:** FastAPI, Pandas, scikit-learn, TensorFlow/Keras ANN, XGBoost (when installed), Joblib
- **Reports:** ReportLab PDF, OpenPyXL Excel
- **Storage:** SQLite (`data/insulin_matrix.sqlite`)

The bundled workbook under `data/sample/` is **synthetic**. Every prediction UI and PDF includes a medical disclaimer.

Target classes: **No Diabetes · Prediabetes · Type 1 Diabetes · Type 2 Diabetes**

> Type 1 is extremely rare in the sample (a handful of rows). Metrics for that class are unstable — this is expected.

---

## Quick start (VS Code / local)

Use two terminals (or the VS Code tasks in `.vscode/tasks.json`).

### 1. Backend (FastAPI + uvicorn)

```bash
cd /path/to/insulin-matrix

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt

# Run uvicorn from backend/ (avoids clashing with any root-level app.py)
cd backend
export PYTHONPATH="$PWD"
export INSULIN_MATRIX_HOME="$(cd .. && pwd)"   # keeps data/sample + artifacts at repo root
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Or from the repo root:

```bash
./scripts/dev-backend.sh
```

API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)  
Health: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

On first start the API **auto-loads** `data/sample/Insulin_Matrix_Synthetic_Dataset_10000_Updated.xlsx` (toggle in Settings).

### 2. Frontend (Vite)

```bash
cd frontend
npm install
npm run dev
```

Or:

```bash
./scripts/dev-frontend.sh
```

App: [http://127.0.0.1:5173](http://127.0.0.1:5173)  
Vite proxies `/api` → `http://127.0.0.1:8000`.

### 3. VS Code

1. Open the repository folder in VS Code.
2. Select the `.venv` Python interpreter (`Python: Select Interpreter`).
3. Run task **Insulin Matrix: Backend** and **Insulin Matrix: Frontend** (Terminal → Run Task), or use the launch configs under **Run and Debug**.
4. Open the Vite URL from the frontend terminal.

Optional: install the official Python and ESLint extensions.

---

## Typical workflow

1. **Overview / Dataset** — confirm the synthetic sample is loaded (or upload Excel/CSV).
2. **Preprocessing** — fit imputation / scaling / encoding on the training split only.
3. **Training** — train LR, Decision Tree, Random Forest, SVM, KNN, Gradient Boosting, XGBoost (if available), and the ANN.
4. **Comparison / ANN Analytics** — review held-out metrics and learning curves (live backend data).
5. **Patient Assessment / Batch** — score individuals or spreadsheets; explanations + class probabilities.
6. **Reports / History** — download PDF/Excel; browse SQLite prediction & experiment history.

No fake accuracy and no hardcoded predictions. Identifier / target / criteria-flag columns are excluded from features (no target leakage).

---

## Project layout

```
backend/app/          FastAPI app, ML pipeline, SQLite, PDF/Excel
frontend/             React + Vite dashboard
data/sample/          Default synthetic Excel (committed)
scripts/              Local dev helpers for VS Code
artifacts/            Trained model files (gitignored)
.vscode/              Tasks + launch configs
```

Legacy Streamlit research modules under `src/` / `pages/` / `legacy_streamlit_app.py` are **not** the primary app. Prefer `backend/` + `frontend/` for local VS Code development.

---

## Environment notes

| Item | Detail |
|------|--------|
| Python | 3.11+ recommended (3.12 tested) |
| Node | 20+ for the Vite frontend |
| GPU | Not required; TensorFlow runs on CPU |
| `INSULIN_MATRIX_HOME` | Optional override for writable data/artifacts |

Root `requirements.txt` mirrors `backend/requirements.txt` for simple `pip install -r requirements.txt`.

---

## Smoke test

With the backend running:

```bash
./scripts/e2e-smoke.sh
```

This loads the sample (if needed), preprocesses, trains a small model set + ANN, runs a prediction, and downloads PDF/Excel.

---

## Disclaimer

Demonstration / model-development use only. Outputs are **not** a medical diagnosis, treatment advice, or insulin dosing guidance.
