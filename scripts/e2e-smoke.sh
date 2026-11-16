#!/usr/bin/env bash
# End-to-end API smoke test against a running backend (default :8000).
set -euo pipefail
BASE="${INSULIN_MATRIX_API:-http://127.0.0.1:8000}"

echo "== health =="
curl -sf "$BASE/api/health" | python3 -m json.tool >/tmp/im_health.json
cat /tmp/im_health.json

echo "== load default dataset =="
curl -sf -X POST "$BASE/api/dataset/load-default" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['n_rows'] if 'n_rows' in d else d.get('profile',{}).get('n_rows'), d.get('class_counts'))"

echo "== preprocess =="
curl -sf -X POST "$BASE/api/preprocess" -H 'Content-Type: application/json' -d '{}' | python3 -c "import sys,json; d=json.load(sys.stdin); print('train', d['n_train'], 'test', d['n_test'], 'stratified', d['stratified'])"

echo "== train classical (subset) =="
curl -sf -X POST "$BASE/api/train" -H 'Content-Type: application/json' \
  -d '{"model_names":["Logistic Regression","Random Forest","Decision Tree","KNN"]}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print([(x['model_name'], round(x['metrics']['accuracy'],4), round(x['metrics']['f1_macro'],4)) for x in d['trained']])"

echo "== train ANN =="
curl -sf -X POST "$BASE/api/ann/train" -H 'Content-Type: application/json' \
  -d '{"epochs":8,"hidden_layers":[32,16],"batch_size":128}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['backend'], round(d['metrics']['accuracy'],4), round(d['metrics']['f1_macro'],4))"

echo "== predict =="
PRED=$(curl -sf -X POST "$BASE/api/predict" -H 'Content-Type: application/json' -d '{
  "model_name":"Random Forest",
  "features":{
    "Age_years":55,"Sex":"Male","Height_cm":175,"Weight_kg":88,"BMI_kg_m2":28.7,
    "Waist_Circumference_cm":98,"Family_History_Diabetes":"Yes","Physical_Activity_Level":"Low",
    "Fasting_Glucose_mg_dL":128,"HbA1c_percent":6.6,"C_Peptide_ng_mL":2.1,
    "Fasting_Insulin_uIU_mL":16,"Systolic_BP_mmHg":138,"Diastolic_BP_mmHg":84,
    "Triglycerides_mg_dL":180,"HDL_mg_dL":38
  }
}')
echo "$PRED" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['predicted_class'], d['prediction_id']); open('/tmp/im_pred_id.txt','w').write(d['prediction_id'])"
PID=$(cat /tmp/im_pred_id.txt)

echo "== pdf =="
curl -sf -X POST "$BASE/api/reports/pdf" -H 'Content-Type: application/json' \
  -d "{\"prediction_id\":\"$PID\",\"include_comparison\":true}" \
  -o /tmp/Insulin_Matrix_Report.pdf
python3 -c "import pathlib; p=pathlib.Path('/tmp/Insulin_Matrix_Report.pdf'); print(p, p.stat().st_size, 'bytes', p.read_bytes()[:4])"

echo "== excel =="
curl -sf "$BASE/api/reports/excel" -o /tmp/Insulin_Matrix_Comparison.xlsx
python3 -c "import pathlib; p=pathlib.Path('/tmp/Insulin_Matrix_Comparison.xlsx'); print(p, p.stat().st_size, 'bytes')"

echo "== comparison =="
curl -sf "$BASE/api/comparison" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d['models']), 'models')"

echo "E2E smoke OK"
