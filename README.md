# Insulin Matrix

**AI-Powered Insulin and Metabolic Research Platform**

Insulin Matrix is a local research prototype for inspecting insulin and glucose spreadsheets, calculating documented metabolic indices, and training scikit-learn models. It is a Streamlit application written in Python. There is no separate JavaScript frontend.

Calculated indices and model outputs are not measurements, diagnoses, treatment recommendations, or insulin doses. HOMA2 is a different model and is not calculated here.

## Run it in Cursor

Open a terminal in this project folder.

This environment has `python3` and does not provide a `python` command. If `python` is missing on your computer, use `python3` in every command below. If `python --version` prints 3.12 or newer, you can use `python` instead.

Install the pinned dependencies once:

```bash
python3 -m pip install --user -r requirements.txt
```

Start the application:

```bash
python3 -m streamlit run app.py
```

The documented launch command is `python -m streamlit run app.py`. Use that form when `python` points at Python 3.12 or a compatible version.

Streamlit prints a local URL, usually `http://localhost:8501`. In a remote Cursor workspace, open that port with the workspace preview or port-forwarding control. The application does not send research data to an external service.

Optional packages are not required. XGBoost appears in the model list only when it is already installed. PyTorch is not part of the core application.

## First session

1. Open **Research Dashboard**.
2. Click **Load synthetic demonstration dataset**. The table is labeled synthetic everywhere it is used.
3. Review the worksheet in **Excel Dataset Explorer**.
4. Confirm column mapping and units in **Variable Mapping and Validation**, then validate.
5. Calculate indices in **Physiological Indices**.
6. Choose features in **Data Processing**.
7. Train a model in **Model Training** and read the held-out metrics in **Model Evaluation**.
8. Score new rows in **Prediction Lab**.
9. Compare or delete runs in **Experiment History**.

The in-app page **Documentation and Research Notes** restates file formats, units, formulas, the training workflow, and the scientific limits.

## Excel

Supported inputs are `.xlsx`, `.xlsm` (cached cell values only; macros are never executed), and `.csv`. Legacy `.xls` is not supported. Reading `.xls` would require the separate `xlrd` package, which is not installed.

When a workbook has several worksheets, choose one. Sheets are never combined, and the original file is never overwritten. Uploads are limited to 50 MB and 250,000 rows. Files are stored under generated identifiers, not the original filename.

Exports are `.xlsx` workbooks named `Insulin_Matrix_*.xlsx`. A workbook only contains sheets for work that has been completed, plus `Export_Notes`. Synthetic results are labeled on that sheet.

## Indices

Implemented calculations, with units and assumptions shown in the app:

- HOMA1-IR: fasting glucose (mmol/L) × fasting insulin (µU/mL) / 22.5. Glucose in mg/dL uses the matching denominator 405. The conversion used inside the app is 18 mg/dL per mmol/L, so the two forms agree.
- HOMA1-B: `20 × insulin / (glucose − 3.5)` with glucose in mmol/L, or `360 × insulin / (glucose − 63)` with glucose in mg/dL. The result is undefined at or below that glucose offset.
- Insulinogenic index: `(insulin 30 − insulin 0) / (glucose 30 − glucose 0)`, reported as (µU/mL) per (mmol/L). It is not calculated when glucose does not rise.
- Matsuda index: `10000 / sqrt(G0 × I0 × Gmean × Imean)` after converting glucose to mg/dL and insulin to µU/mL. The mean uses paired times among 0, 30, 60, 90, and 120 minutes.
- Glucose and insulin area under the curve: linear trapezoid in the measurement’s original unit × minutes.
- Oral disposition index: insulinogenic index / fasting insulin (Utzschneider). This is not the product of the insulinogenic index and the Matsuda index.

Insulin in pmol/L is converted with 6.945 pmol/L per µU/mL. mIU/L is treated as the same scale as µU/mL. Units are never guessed.

## Models

Regression: linear regression, ridge regression, random forest, gradient boosting. Classification: logistic regression, random forest, gradient boosting.

When a participant identifier is mapped, every row for a participant stays in one partition. Grouped cross-validation, when it can be run, uses the training partition only. Preprocessing is a scikit-learn pipeline fitted on the training participants. Participant identifiers are excluded from model inputs.

Training is refused when the target is invalid, too few independent participants remain, or a meaningful holdout cannot be built. The dashboard shows zeros until a dataset or experiment actually exists.

## Synthetic data

The demonstration table is deterministic (NumPy seed 42): 80 participants, 24 of them with a second visit (104 rows). Fasting and oral glucose tolerance test glucose and insulin, age, sex, BMI, and waist are included. `synthetic_metabolic_score` is a documented function of HOMA1-IR, age, and BMI plus noise. `synthetic_risk_group` is a median split of that score. Neither column is a disease label, and performance on them is not clinical evidence.

Do not combine the synthetic table with a real workbook unless you do that yourself and keep the synthetic label visible.

## Storage

SQLite metadata lives at `data/insulin_matrix.sqlite`. Uploads are under `data/uploads/`, processed copies under `data/processed/`, and model files under `artifacts/models/`. These paths are gitignored.

Deleting a dataset removes its upload and processed copy. Experiment metrics remain, with the dataset label copied onto the experiment. Deleting an experiment removes the history row and the model file together.

Set `INSULIN_MATRIX_HOME` to keep that storage outside the project folder. Tests do this automatically.

## Tests

```bash
python3 -m pytest
```

The tests use synthetic fixtures. They do not need patient records, a network connection, or an API key.

## Project layout

`app.py` is the Streamlit entry point. Scientific code is in `src/` and can be imported without starting the interface. Pages are in `pages/`. Tests are in `tests/`.
