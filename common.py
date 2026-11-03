"""
common.py - shared settings used by train_model.py, app.py and predict_console.py
Edit GRADE_TABLE below if your university uses a different 10-point scale.
"""

# ------------------------------------------------------------------
# Questions asked to the user (also decide which columns the model uses)
# type "int"    -> number box
# type "choice" -> dropdown; the key is the value stored in the dataset
# past_grade    -> only used when USE_PAST_GRADES = True in train_model.py
# ------------------------------------------------------------------
RATING = {1: "1 - Very low", 2: "2 - Low", 3: "3 - Medium", 4: "4 - High", 5: "5 - Very high"}
YES_NO = {"yes": "Yes", "no": "No"}
EDU = {0: "None", 1: "Primary (up to 4th grade)", 2: "5th - 9th grade",
       3: "Secondary", 4: "Higher education"}

QUESTIONS = [
    # ---- Personal ----
    {"key": "age", "label": "Age", "section": "Personal details",
     "type": "int", "min": 15, "max": 22, "default": 17},
    {"key": "sex", "label": "Sex", "section": "Personal details",
     "type": "choice", "options": {"F": "Female", "M": "Male"}},
    {"key": "address", "label": "Home address type", "section": "Personal details",
     "type": "choice", "options": {"U": "Urban", "R": "Rural"}},
    {"key": "health", "label": "Current health status", "section": "Personal details",
     "type": "choice", "default": 3,
     "options": {1: "1 - Very bad", 2: "2 - Bad", 3: "3 - Average", 4: "4 - Good", 5: "5 - Very good"}},

    # ---- Family ----
    {"key": "Medu", "label": "Mother's education", "section": "Family background",
     "type": "choice", "options": EDU, "default": 2},
    {"key": "Fedu", "label": "Father's education", "section": "Family background",
     "type": "choice", "options": EDU, "default": 2},
    {"key": "famrel", "label": "Quality of family relationships", "section": "Family background",
     "type": "choice", "default": 4,
     "options": {1: "1 - Very bad", 2: "2 - Bad", 3: "3 - Average", 4: "4 - Good", 5: "5 - Excellent"}},
    {"key": "famsup", "label": "Family educational support", "section": "Family background",
     "type": "choice", "options": YES_NO, "default": "yes"},

    # ---- Study & lifestyle ----
    {"key": "studytime", "label": "Weekly study time", "section": "Study habits & lifestyle",
     "type": "choice", "default": 2,
     "options": {1: "Less than 2 hours", 2: "2 to 5 hours", 3: "5 to 10 hours", 4: "More than 10 hours"}},
    {"key": "failures", "label": "Number of past class failures", "section": "Study habits & lifestyle",
     "type": "int", "min": 0, "max": 3, "default": 0},
    {"key": "absences", "label": "Number of school absences", "section": "Study habits & lifestyle",
     "type": "int", "min": 0, "max": 75, "default": 4},
    {"key": "schoolsup", "label": "Extra educational support from school", "section": "Study habits & lifestyle",
     "type": "choice", "options": YES_NO, "default": "no"},
    {"key": "internet", "label": "Internet access at home", "section": "Study habits & lifestyle",
     "type": "choice", "options": YES_NO, "default": "yes"},
    {"key": "higher", "label": "Wants to pursue higher education", "section": "Study habits & lifestyle",
     "type": "choice", "options": YES_NO, "default": "yes"},
    {"key": "freetime", "label": "Free time after school", "section": "Study habits & lifestyle",
     "type": "choice", "options": RATING, "default": 3},
    {"key": "goout", "label": "Going out with friends", "section": "Study habits & lifestyle",
     "type": "choice", "options": RATING, "default": 3},
    {"key": "Dalc", "label": "Alcohol consumption on weekdays", "section": "Study habits & lifestyle",
     "type": "choice", "options": RATING, "default": 1},
    {"key": "Walc", "label": "Alcohol consumption on weekends", "section": "Study habits & lifestyle",
     "type": "choice", "options": RATING, "default": 1},

    # ---- Previous grades (optional, controlled by USE_PAST_GRADES) ----
    {"key": "G1", "label": "First term grade (G1, out of 20)", "section": "Previous grades",
     "type": "int", "min": 0, "max": 20, "default": 10, "past_grade": True},
    {"key": "G2", "label": "Second term grade (G2, out of 20)", "section": "Previous grades",
     "type": "int", "min": 0, "max": 20, "default": 10, "past_grade": True},
]

LABELS = {q["key"]: q["label"] for q in QUESTIONS}

# ------------------------------------------------------------------
# Score conversion: dataset score (0-20) -> percentage -> Indian 10-point grade
# ------------------------------------------------------------------
# (minimum percentage, letter grade, grade point / SGPA)
GRADE_TABLE = [
    (90, "O  (Outstanding)", 10),
    (80, "A+ (Excellent)", 9),
    (70, "A  (Very Good)", 8),
    (60, "B+ (Good)", 7),
    (50, "B  (Above Average)", 6),
    (45, "C  (Average)", 5),
    (40, "P  (Pass)", 4),
    (0,  "F  (Fail)", 0),
]
PASS_PERCENT = 40


def to_percentage(score_out_of_20):
    return round(score_out_of_20 / 20 * 100, 1)


def grade_info(percentage):
    for min_pct, letter, point in GRADE_TABLE:
        if percentage >= min_pct:
            return letter, point
    return GRADE_TABLE[-1][1], GRADE_TABLE[-1][2]


def risk_level(percentage):
    if percentage < PASS_PERCENT:
        return "High Risk"
    if percentage < 60:
        return "Medium Risk"
    return "Low Risk"


def evaluate(predicted_score_out_of_20):
    """Turn the model's predicted G3 into everything we show the user."""
    score = max(0.0, min(20.0, float(predicted_score_out_of_20)))
    pct = to_percentage(score)
    letter, point = grade_info(pct)
    return {
        "score20": round(score, 1),
        "percentage": pct,
        "letter": letter,
        "sgpa": point,
        "risk": risk_level(pct),
    }
