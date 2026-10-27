"""In-application research notes. Wording here is the user-facing documentation."""

from __future__ import annotations

from src.config import DISCLAIMER, HOMA2_STATEMENT
from src.indices import FORMULAS

SECTIONS = [
    {
        "title": "What this application is",
        "paragraphs": [
            "Insulin Matrix is a local research workspace for insulin and glucose spreadsheets, calculated metabolic indices, and supervised models.",
            DISCLAIMER,
            "Nothing in the application is a medical device output. Do not use a prediction, an index, or a chart to choose a treatment or an insulin dose.",
        ],
    },
    {
        "title": "Supported files",
        "paragraphs": [
            "Upload .xlsx, .xlsm, or .csv. The first row of the chosen worksheet must contain column names.",
            "Macros in .xlsm files are never executed. Formula cells are read as their cached values. If a formula has never been calculated in Excel, the cached value may be empty.",
            "Legacy .xls files are not supported. Converting them would require the optional xlrd package, which is not installed. Save the workbook as .xlsx instead.",
            "Worksheets are never combined. A workbook with more than one sheet waits until you choose one.",
            "Files larger than 50 MB, or worksheets with more than 250,000 rows, are rejected. The original upload is stored under a generated name and is not overwritten.",
        ],
    },
    {
        "title": "Variable mapping",
        "paragraphs": [
            "You map spreadsheet columns onto canonical names such as participant ID, fasting glucose, and the 30, 60, 90, and 120 minute OGTT samples.",
            "Wide sheets put each OGTT time in its own column. Long sheets put time in one column and glucose or insulin in a single column. If both layouts are mapped, the wide columns are used and a warning is shown.",
            "Name suggestions are not applied by themselves. Units are never inferred from the numbers. The synthetic demonstration dataset is the exception: its mapping and units are taken from the generator specification and shown before use.",
            "Participant IDs and observation IDs are kept out of model inputs.",
        ],
    },
    {
        "title": "Units",
        "paragraphs": [
            "Glucose may be mmol/L or mg/dL. Insulin may be µU/mL, mIU/L, or pmol/L. mIU/L is the same numeric scale as µU/mL.",
            "HOMA1 converts glucose with 18 mg/dL per mmol/L, so glucose × insulin / 22.5 (mmol/L) matches glucose × insulin / 405 (mg/dL).",
            "Insulin in pmol/L is divided by 6.945 to reach µU/mL.",
            "The insulinogenic index and the oral disposition index are reported after conversion to µU/mL and mmol/L. The Matsuda index is reported with the conventional mg/dL and µU/mL constant of 10000. Areas under the curve stay in the unit you supplied, multiplied by minutes.",
        ],
    },
    {
        "title": "Physiological indices",
        "paragraphs": [
            FORMULAS["homa_ir"],
            FORMULAS["homa_b"],
            HOMA2_STATEMENT,
            FORMULAS["insulinogenic_index"],
            FORMULAS["matsuda_index"],
            FORMULAS["glucose_auc"],
            FORMULAS["insulin_auc"],
            FORMULAS["oral_disposition_index"],
            "A different product, the insulinogenic index multiplied by the Matsuda index, is sometimes also called a disposition index. Insulin Matrix does not calculate that product.",
            "Missing inputs, non-positive denominators, glucose that does not rise, and time series with fewer than two points are left blank and explained. They are not filled with zero.",
            "Indices are calculated quantities. They are not direct measurements and they are not diagnoses.",
        ],
    },
    {
        "title": "Models and evaluation",
        "paragraphs": [
            "Regression models are ordinary least squares, ridge, random forest, and gradient boosting. Classification models are logistic regression, random forest, and gradient boosting. XGBoost appears only when that package is installed. PyTorch is not required.",
            "Choose the target and the features yourself. Identifier columns are rejected. If the target is also selected as a feature, training stops.",
            "Imputation and scaling are fitted on the training participants only, inside a scikit-learn pipeline. Derived indices are row-wise formulas and do not use other participants.",
            "When a participant ID is present, every row from that participant stays in one partition. Grouped cross-validation, when it can be run, is limited to the training partition. The numbers shown as the main result are the held-out test metrics.",
            "If there are fewer than 8 participants, or a class cannot appear in both partitions, training stops and the reason is shown. The application does not invent a score for that case.",
            "Row-wise random splitting is used only after you confirm that each row is an independent participant.",
        ],
    },
    {
        "title": "Predictions",
        "paragraphs": [
            "The Prediction Lab loads a saved pipeline and applies it. It does not refit the model.",
            "New rows must contain the stored feature names. Confirm the units. A mismatch stops scoring unless you run the explicit conversion step, which shows the change before you continue.",
            "Predicted values are research outputs. They are not diagnoses and they do not imply an insulin dose.",
        ],
    },
    {
        "title": "Storage and deletion",
        "paragraphs": [
            "The SQLite file in data/ stores dataset labels, mappings, units, and experiment metrics. Cell-level tables stay in the uploaded file and in a processed CSV copy. Model pipelines are joblib files under artifacts/models/.",
            "Names on disk are generated identifiers, not the original filename.",
            "Deleting an experiment removes its history row and its model file together. If the file is already gone, the history row can still be removed.",
            "Deleting a dataset removes the uploaded file and the processed copy. Experiment metrics are kept so the history remains readable. Dataset labels are copied onto the experiment when it is trained.",
            "Insulin Matrix does not upload research tables to an external service. Streamlit usage statistics are disabled in the project configuration.",
            "Do not commit uploads, the SQLite file, or model artifacts to git. Those paths are gitignored.",
        ],
    },
    {
        "title": "Synthetic demonstration data",
        "paragraphs": [
            "The demonstration set is generated locally with NumPy seed 42. It contains synthetic participant IDs, fasting and OGTT glucose and insulin, age, sex, BMI, waist circumference, a synthetic score, and a higher/lower median split of that score.",
            "The score equals 1.4 × HOMA1-IR + 0.015 × age + 0.02 × (BMI − 25) + a small normal noise term. The higher/lower label is a median split of the score. It is not a disease class.",
            "Twenty-four of the eighty participants have a second visit so that participant-level splitting can be checked.",
            "Models that use fasting glucose or fasting insulin can recover this generator. Report that as a software check, not as clinical performance.",
            "Synthetic and user-supplied tables are not merged. Loading one replaces the active table.",
        ],
    },
    {
        "title": "Scientific limitations",
        "paragraphs": [
            "HOMA1 is not HOMA2. OGTT areas change when samples are missing. The Matsuda mean uses the paired times that are present, not a continuous curve.",
            "The application does not implement glucose clamps, insulin-dose calculators, or diagnostic thresholds.",
            "A high test-set R² or accuracy on the synthetic score only means the model followed the documented generator.",
            "Real research data can contain identifiers. Review exports before sharing them. The history stores the original filename as a label on this computer and does not send it elsewhere.",
        ],
    },
]
