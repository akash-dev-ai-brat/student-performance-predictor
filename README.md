<div align="center">

<img src="assets/demo.gif" alt="Student Performance Predictor Demo" width="800"/>

# 📊 Student Performance Predictor

### ML-powered Academic Outcome Forecasting System

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Student Performance Predictor** uses machine learning to forecast a student's academic performance based on study habits, attendance, parental involvement, and socioeconomic indicators — giving educators and students actionable insights before exams.

[Live Demo](#) · [Report Bug](https://github.com/akash-dev-ai-brat/student-performance-predictor/issues) · [View Notebook](notebooks/)

</div>

---

## 📌 Table of Contents
- [About the Project](#about)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Screenshots](#screenshots)
- [Future Improvements](#future-improvements)

---

## 🎯 About the Project <a name="about"></a>

Early identification of at-risk students is a critical challenge in education. This project builds a supervised ML pipeline that predicts student exam scores (regression) and pass/fail outcomes (classification) using features like study time, absences, parental education level, and extracurricular participation.

The end product is an interactive Streamlit dashboard where educators can input a student profile and receive an instant prediction with feature importance explanations.

---

## ✨ Features <a name="features"></a>

- 🔮 **Dual prediction** — predicts both exam score (regression) and pass/fail (classification)
- 📈 **Feature importance visualization** — see which factors most impact performance using SHAP or permutation importance
- 🧪 **Multiple model comparison** — Random Forest, XGBoost, and Logistic Regression benchmarked side by side
- 📋 **Interactive input form** — educators enter student data and receive instant predictions
- 📓 **Fully documented EDA notebook** — with charts, correlation heatmaps, and distribution plots

---

## 🛠️ Tech Stack <a name="tech-stack"></a>

| Layer | Technology |
|-------|-----------|
| ML Models | scikit-learn, XGBoost |
| EDA & Analysis | Pandas, NumPy, Matplotlib, Seaborn |
| Explainability | SHAP / Feature Importance |
| App UI | Streamlit |
| Notebook | Jupyter |
| Dataset | UCI Student Performance / Kaggle |

---

## 📊 Model Performance <a name="model-performance"></a>

| Model | Accuracy | F1 Score | RMSE (Score) |
|-------|----------|----------|--------------|
| Random Forest | 87.3% | 0.86 | 4.2 |
| XGBoost | 88.9% | 0.88 | 3.8 |
| Logistic Regression | 82.1% | 0.81 | — |

> 📌 **Action item:** Replace with your actual metrics after running your models.

---

## 📁 Project Structure <a name="project-structure"></a>

```
student-performance-predictor/
├── app.py                          # Streamlit prediction dashboard
├── notebooks/
│   └── eda_and_modeling.ipynb      # Full EDA + model training notebook
├── src/
│   ├── train.py                    # Model training pipeline
│   ├── predict.py                  # Prediction logic
│   └── preprocess.py               # Feature engineering & encoding
├── models/
│   └── best_model.pkl              # Serialized best model
├── data/
│   └── student_data.csv            # Dataset (or link to source)
├── assets/
│   └── demo.gif
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start <a name="quick-start"></a>

**1. Clone the repository**
```bash
git clone https://github.com/akash-dev-ai-brat/student-performance-predictor.git
cd student-performance-predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. (Optional) Retrain the model**
```bash
python src/train.py
```

**4. Launch the app**
```bash
streamlit run app.py
```

---

## 📂 Dataset <a name="dataset"></a>

This project uses the [Student Performance Dataset](https://www.kaggle.com/) from Kaggle / UCI ML Repository.

**Key features used:**
- Study hours per week
- Number of absences
- Parental education level
- Tutoring sessions attended
- Extracurricular activities
- Previous exam scores

---

## 📸 Screenshots <a name="screenshots"></a>


---

## 🔮 Future Improvements <a name="future-improvements"></a>

- [ ] Add SHAP waterfall plots per individual prediction
- [ ] Deploy on Streamlit Cloud with shareable link
- [ ] Extend dataset to include real institutional data (with consent)
- [ ] Add early warning email alert system for teachers

---

<div align="center">
  Made with ❤️ by <a href="https://github.com/akash-dev-ai-brat">Akash Nath</a> · 
  <a href="https://www.linkedin.com/in/akash-nath-5aa816293/">LinkedIn</a>
</div>
