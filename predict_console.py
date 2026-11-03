"""
predict_console.py - question & answer version (works without Streamlit).
Run:  python predict_console.py
"""
import joblib
import pandas as pd
from common import QUESTIONS, LABELS, evaluate

bundle = joblib.load("model.joblib")
features = bundle["features"]

print("\n=== Student Performance Predictor ===")
print("Answer the following questions about the student.\n")

answers = {}
for q in QUESTIONS:
    if q["key"] not in features:
        continue
    if q["type"] == "int":
        while True:
            raw = input(f"{q['label']} [{q['min']}-{q['max']}] (Enter = {q['default']}): ").strip()
            if raw == "":
                value = q["default"]; break
            if raw.isdigit() and q["min"] <= int(raw) <= q["max"]:
                value = int(raw); break
            print("  Invalid value, try again.")
    else:
        keys = list(q["options"].keys())
        print(f"{q['label']}:")
        for i, k in enumerate(keys, 1):
            print(f"   {i}. {q['options'][k]}")
        default_idx = keys.index(q["default"]) + 1 if q.get("default") in keys else 1
        while True:
            raw = input(f"  Choose 1-{len(keys)} (Enter = {default_idx}): ").strip()
            if raw == "":
                value = keys[default_idx - 1]; break
            if raw.isdigit() and 1 <= int(raw) <= len(keys):
                value = keys[int(raw) - 1]; break
            print("  Invalid choice, try again.")
    answers[q["key"]] = value

row = pd.DataFrame([answers])[features]
result = evaluate(bundle["model"].predict(row)[0])

print("\n========== PREDICTION RESULT ==========")
print(f"Predicted Final Score : {result['score20']} / 20")
print(f"Percentage            : {result['percentage']} / 100")
print(f"Letter Grade          : {result['letter']}")
print(f"SGPA (10-point scale) : {result['sgpa']} / 10")
print(f"Risk Level            : {result['risk']}")
print("Top factors affecting grades (overall):")
for k, v in bundle["top_factors"]:
    print(f"   - {LABELS[k]}")
print(f"Model used            : {bundle['model_name']}")
