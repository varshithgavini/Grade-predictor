# Student Academic Performance Prediction (Machine Learning)

Predicts a student's final grade (G3, out of 20) from academic, family and lifestyle
factors using Linear Regression, Decision Tree and Random Forest, then converts the
result to a percentage (/100), a letter grade and an Indian 10-point SGPA.

## Files
- common.py          questions asked + score conversion (edit GRADE_TABLE for your university)
- train_model.py     trains/compares the 3 models, saves charts + model.joblib
- app.py             web interface (Streamlit)
- predict_console.py same Q&A in the terminal (backup if Streamlit fails)

## Run
1. pip install -r requirements.txt
2. Put student-mat.csv in this folder (UCI Student Performance dataset)
3. python train_model.py
4. streamlit run app.py        (or: python predict_console.py)
