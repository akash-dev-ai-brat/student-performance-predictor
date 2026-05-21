import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import os

# ─── Generate realistic student dataset ────────────────────────
np.random.seed(42)
n = 2000

study_hours     = np.random.normal(5, 2, n).clip(0, 14)
attendance      = np.random.normal(75, 15, n).clip(30, 100)
sleep_hours     = np.random.normal(7, 1.5, n).clip(3, 12)
prev_score      = np.random.normal(65, 15, n).clip(20, 100)
assignments     = np.random.normal(75, 20, n).clip(0, 100)
stress_level    = np.random.randint(1, 11, n)   # 1-10
extra_curricular= np.random.randint(0, 4, n)    # 0-3 activities
internet_hours  = np.random.normal(4, 2, n).clip(0, 12)
family_support  = np.random.randint(1, 6, n)    # 1-5
part_time_job   = np.random.randint(0, 2, n)    # 0 or 1

# ─── Calculate performance score ───────────────────────────────
score = (
    study_hours     * 4.0 +
    attendance      * 0.3 +
    sleep_hours     * 1.5 +
    prev_score      * 0.4 +
    assignments     * 0.2 +
    family_support  * 2.0 -
    stress_level    * 1.5 -
    internet_hours  * 1.0 -
    part_time_job   * 3.0 +
    extra_curricular* 1.0 +
    np.random.normal(0, 5, n)
)

# ─── Convert score to grade ─────────────────────────────────────
def score_to_grade(s):
    if s >= 75:   return "A"
    elif s >= 60: return "B"
    elif s >= 45: return "C"
    elif s >= 30: return "D"
    else:         return "F"

grades = [score_to_grade(s) for s in score]

# ─── Build DataFrame ───────────────────────────────────────────
df = pd.DataFrame({
    "study_hours":      study_hours,
    "attendance":       attendance,
    "sleep_hours":      sleep_hours,
    "prev_score":       prev_score,
    "assignments":      assignments,
    "stress_level":     stress_level,
    "extra_curricular": extra_curricular,
    "internet_hours":   internet_hours,
    "family_support":   family_support,
    "part_time_job":    part_time_job,
    "grade":            grades
})

# ─── Encode labels ─────────────────────────────────────────────
le = LabelEncoder()
df["grade_encoded"] = le.fit_transform(df["grade"])

features = ["study_hours","attendance","sleep_hours","prev_score",
            "assignments","stress_level","extra_curricular",
            "internet_hours","family_support","part_time_job"]

X = df[features]
y = df["grade_encoded"]

# ─── Train model ───────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"✅ Model trained! Accuracy: {acc*100:.1f}%")

# ─── Save model and encoder ────────────────────────────────────
joblib.dump(model, "model.pkl")
joblib.dump(le,    "label_encoder.pkl")
joblib.dump(features, "features.pkl")
print("✅ Model saved as model.pkl")