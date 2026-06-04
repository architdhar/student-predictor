# 🎓 Student Performance Predictor

> Random Forest classifier predicting student pass/fail outcomes from the UCI Student Performance dataset — with a teacher-facing prediction form and model analytics dashboard.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?logo=streamlit)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-blue)](https://plotly.com)

---

## 🔍 What it does

| Feature | Detail |
|---|---|
| Dataset | UCI Student Performance (649 students, 33 features) |
| Model | Random Forest (200 trees, max depth 8) |
| Accuracy | **88%** on held-out test set |
| F1-Score | **0.85** (5-fold cross-validation) |
| Interface | Teacher-facing input form + prediction + top-3 factors |

---

## 💡 Key Insights from EDA

- **Study time** is the strongest positive predictor of passing
- **Prior failures** are the strongest negative predictor
- Students with internet access at home show ~12% higher pass rates
- Feature importance plots communicate findings clearly to non-technical stakeholders

---

## 🚀 Run Locally

```bash
git clone https://github.com/archit-dhar/student-performance-predictor
cd student-performance-predictor
pip install -r requirements.txt
streamlit run app.py
```

---

## 🏗️ Architecture

```
UCI Dataset (649 students, 33 features)
        │
        ▼
  Data Cleaning + EDA
  (study time, failures, absences, etc.)
        │
        ▼
  Random Forest Classifier
  (train/test split 80/20 + 5-fold CV)
        │
        ├── Confusion Matrix + F1/Accuracy Metrics
        ├── Feature Importance Plot
        └── Teacher Input Form → Prediction + Top-3 Factors
```

---

## 📁 Project Structure

```
student-performance-predictor/
├── app.py              # Streamlit app (predict + analytics views)
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

`Python` · `scikit-learn` · `Random Forest` · `Pandas` · `NumPy` · `Plotly` · `Streamlit`

---

## 👤 Author

**Archit Dhar** · [LinkedIn](https://linkedin.com/in/archit-dhar) · B.E. AI & Data Science, VESIT Mumbai 2025
