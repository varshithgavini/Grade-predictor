"""
app.py - Web interface (Streamlit)
Run:  streamlit run app.py
"""
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

from common import QUESTIONS, LABELS, evaluate

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")
st.title("🎓 Student Academic Performance Predictor")
st.caption("Answer the questions below to predict the student's final grade using Machine Learning.")

if not Path("model.joblib").exists():
    st.error("model.joblib not found. Run `python train_model.py` first.")
    st.stop()


@st.cache_resource
def load_bundle():
    return joblib.load("model.joblib")


bundle = load_bundle()
features = bundle["features"]
questions = [q for q in QUESTIONS if q["key"] in features]

# ---------------------------- INPUT FORM ----------------------------
answers = {}
with st.form("student_form"):
    current_section = None
    cols = None
    count = 0
    for q in questions:
        if q["section"] != current_section:
            current_section = q["section"]
            st.subheader(current_section)
            cols = st.columns(2)
            count = 0
        box = cols[count % 2]
        count += 1
        if q["type"] == "int":
            answers[q["key"]] = box.number_input(
                q["label"], min_value=q["min"], max_value=q["max"],
                value=q["default"], step=1, key=q["key"])
        else:
            keys = list(q["options"].keys())
            idx = keys.index(q["default"]) if q.get("default") in keys else 0
            answers[q["key"]] = box.selectbox(
                q["label"], options=keys, index=idx,
                format_func=lambda k, opts=q["options"]: opts[k], key=q["key"])
    submitted = st.form_submit_button("Predict Performance", use_container_width=True)

# ---------------------------- RESULT --------------------------------
if submitted:
    row = pd.DataFrame([answers])[features]
    result = evaluate(bundle["model"].predict(row)[0])

    st.divider()
    st.header("Prediction Result")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Score", f"{result['score20']} / 20")
    c2.metric("Percentage", f"{result['percentage']} / 100")
    c3.metric("SGPA (10-point)", f"{result['sgpa']} / 10")
    st.write(f"**Letter Grade:** {result['letter']}")
    st.progress(min(int(result["percentage"]), 100) / 100)

    if result["risk"] == "High Risk":
        st.error("Risk Level: HIGH RISK - predicted below the pass mark. Early support is recommended.")
    elif result["risk"] == "Medium Risk":
        st.warning("Risk Level: MEDIUM RISK - passing, but there is room for improvement.")
    else:
        st.success("Risk Level: LOW RISK - the student is on track.")

    st.subheader("Key factors influencing grades (overall)")
    for k, _ in bundle["top_factors"]:
        st.write(f"- {LABELS[k]}")

    with st.expander("Model comparison (Linear Regression vs Decision Tree vs Random Forest)"):
        st.dataframe(bundle["results"], hide_index=True)
        st.caption(f"Model used for this prediction: **{bundle['model_name']}**")
