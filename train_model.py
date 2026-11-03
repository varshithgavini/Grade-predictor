"""
train_model.py
Trains Linear Regression, Decision Tree and Random Forest on the UCI
Student Performance dataset, compares them, saves charts and saves the
best model to model.joblib (used by app.py / predict_console.py).

Run:  python train_model.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # save charts as images (no pop-up windows)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from common import QUESTIONS, LABELS

# ------------------------- SETTINGS -------------------------
DATA_FILE = "student-mat.csv"
USE_PAST_GRADES = True     # True  -> use G1 & G2 (higher accuracy)
                           # False -> early prediction without past grades (harder)
RANDOM_STATE = 42
# ------------------------------------------------------------

Path("charts").mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")

# 1. LOAD DATA ----------------------------------------------------
df = pd.read_csv(DATA_FILE, sep=None, engine="python")   # auto-detects ; or ,
df.columns = df.columns.str.strip()
print("Dataset shape:", df.shape)
print("Missing values:", int(df.isnull().sum().sum()))
if "G3" not in df.columns:
    raise SystemExit("G3 column not found - check that you loaded student-mat.csv")

features = [q["key"] for q in QUESTIONS if USE_PAST_GRADES or not q.get("past_grade")]
missing = [f for f in features if f not in df.columns]
if missing:
    raise SystemExit(f"These columns are missing from your CSV: {missing}")

X = df[features].copy()
y = df["G3"]
categorical = [c for c in features if not pd.api.types.is_numeric_dtype(X[c])]
numeric = [c for c in features if c not in categorical]
print("Features used:", len(features), "| with past grades:", USE_PAST_GRADES)

# 2. EDA CHARTS ---------------------------------------------------
plt.figure(figsize=(6, 4))
sns.histplot(y, kde=True, bins=20)
plt.title("Distribution of Final Grade (G3)")
plt.xlabel("G3 (out of 20)")
plt.tight_layout(); plt.savefig("charts/1_g3_distribution.png", dpi=150); plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric + ["G3"]].corr(), cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout(); plt.savefig("charts/2_correlation_heatmap.png", dpi=150); plt.close()

plt.figure(figsize=(6, 4))
sns.boxplot(x=df["studytime"], y=y)
plt.title("Study Time vs Final Grade")
plt.xlabel("Study time (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)")
plt.tight_layout(); plt.savefig("charts/3_studytime_vs_grade.png", dpi=150); plt.close()

# 3. TRAIN / TEST SPLIT ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE)

def make_pipeline(model):
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numeric),
    ])
    return Pipeline([("prep", prep), ("model", model)])

models = {
    "Linear Regression": make_pipeline(LinearRegression()),
    "Decision Tree": make_pipeline(DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE)),
    "Random Forest": make_pipeline(RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE)),
}

# 4. TRAIN + EVALUATE ALL 3 MODELS -------------------------------
rows, predictions = [], {}
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    pred = np.clip(pipe.predict(X_test), 0, 20)
    predictions[name] = pred
    cv_r2 = cross_val_score(pipe, X, y, cv=kf, scoring="r2").mean()
    rows.append({
        "Model": name,
        "MAE": round(mean_absolute_error(y_test, pred), 2),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, pred)), 2),
        "R2 (test)": round(r2_score(y_test, pred), 3),
        "R2 (5-fold CV)": round(cv_r2, 3),
    })

results = pd.DataFrame(rows)
print("\n=========== MODEL COMPARISON ===========")
print(results.to_string(index=False))
results.to_csv("model_comparison.csv", index=False)

best_name = results.sort_values("RMSE").iloc[0]["Model"]
print(f"\nBest model (lowest RMSE): {best_name}")

# 5. COMPARISON CHARTS -------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.barplot(data=results, x="Model", y="R2 (test)", ax=ax[0])
ax[0].set_title("R2 Score (higher is better)")
sns.barplot(data=results, x="Model", y="RMSE", ax=ax[1])
ax[1].set_title("RMSE (lower is better)")
for a in ax: a.tick_params(axis="x", rotation=15)
plt.tight_layout(); plt.savefig("charts/4_model_comparison.png", dpi=150); plt.close()

plt.figure(figsize=(5.5, 5.5))
plt.scatter(y_test, predictions[best_name], alpha=0.6)
plt.plot([0, 20], [0, 20], "r--")
plt.xlabel("Actual G3"); plt.ylabel("Predicted G3")
plt.title(f"Predicted vs Actual ({best_name})")
plt.tight_layout(); plt.savefig("charts/5_predicted_vs_actual.png", dpi=150); plt.close()

# 6. FEATURE IMPORTANCE (from Random Forest) ----------------------
rf = models["Random Forest"]
names = rf.named_steps["prep"].get_feature_names_out()
importances = rf.named_steps["model"].feature_importances_
agg = {f: 0.0 for f in features}
for n, v in zip(names, importances):
    n = n.split("__", 1)[1]
    if n in agg:
        agg[n] += v
    else:
        for c in categorical:
            if n.startswith(c + "_"):
                agg[c] += v
                break
imp = pd.Series(agg).sort_values(ascending=False)
print("\nTop 10 influential factors (Random Forest):")
print(imp.head(10).round(3).to_string())

plt.figure(figsize=(7, 5))
top = imp.head(10)[::-1]
plt.barh([LABELS[k] for k in top.index], top.values)
plt.title("Top 10 Factors Affecting Final Grade")
plt.xlabel("Importance")
plt.tight_layout(); plt.savefig("charts/6_feature_importance.png", dpi=150); plt.close()

# 7. SAVE BEST MODEL ----------------------------------------------
joblib.dump({
    "model": models[best_name],
    "model_name": best_name,
    "features": features,
    "results": results,
    "top_factors": [(k, float(v)) for k, v in imp.head(5).items()],
}, "model.joblib")
print("\nSaved: model.joblib, model_comparison.csv and charts/ folder")
