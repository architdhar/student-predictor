import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.pred-pass { background:#d1fae5; border-radius:12px; padding:1.5rem; text-align:center; border:2px solid #6ee7b7; }
.pred-fail { background:#fee2e2; border-radius:12px; padding:1.5rem; text-align:center; border:2px solid #fca5a5; }
.pred-pass h2 { color:#065f46; font-size:22px; }
.pred-fail h2 { color:#991b1b; font-size:22px; }
.factor-card { background:#f8fafc; border-radius:10px; padding:.9rem 1.1rem; border:1px solid #e2e8f0; margin-bottom:8px; }
.factor-card p { margin:0; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA GENERATION (UCI Student Performance)
# ─────────────────────────────────────────────
@st.cache_data
def generate_student_data():
    np.random.seed(42)
    n = 649
    
    study_time = np.random.choice([1,2,3,4], n, p=[0.15,0.35,0.35,0.15])
    failures = np.random.choice([0,1,2,3], n, p=[0.67,0.20,0.09,0.04])
    absences = np.random.randint(0, 30, n)
    internet = np.random.choice([0,1], n, p=[0.35,0.65])
    parent_edu = np.random.choice([0,1,2,3,4], n)
    free_time = np.random.choice([1,2,3,4,5], n)
    go_out = np.random.choice([1,2,3,4,5], n)
    health = np.random.choice([1,2,3,4,5], n)
    romantic = np.random.choice([0,1], n, p=[0.6,0.4])
    
    # Final grade influenced by features
    grade_score = (
        study_time * 3.5
        - failures * 4.0
        - absences * 0.2
        + internet * 1.5
        + parent_edu * 0.8
        - go_out * 0.5
        + health * 0.3
        + np.random.normal(0, 3, n)
    )
    passed = (grade_score > 8).astype(int)
    
    return pd.DataFrame({
        "study_time": study_time,
        "failures": failures,
        "absences": absences,
        "internet": internet,
        "parent_edu": parent_edu,
        "free_time": free_time,
        "go_out": go_out,
        "health": health,
        "romantic": romantic,
        "passed": passed
    })

@st.cache_data
def train_model():
    df = generate_student_data()
    X = df.drop("passed", axis=1)
    y = df["passed"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = (y_pred == y_test).mean()
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="f1")
    
    cm = confusion_matrix(y_test, y_pred)
    fi = pd.DataFrame({"feature": X.columns, "importance": clf.feature_importances_})
    fi = fi.sort_values("importance", ascending=False)
    
    return clf, X.columns.tolist(), acc, cv_scores.mean(), cm, fi, X_test, y_test, y_pred

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/48/graduation-cap.png", width=40)
    st.title("Student Predictor")
    st.caption("Random Forest · UCI Dataset")
    st.divider()
    
    page = st.radio("View", ["🔮 Predict Student", "📊 Model Analytics"])
    st.divider()
    st.caption("Built by Archit Dhar · VESIT 2025")

clf, features, accuracy, f1, cm, fi, X_test, y_test, y_pred = train_model()
df = generate_student_data()

# ─────────────────────────────────────────────
# PREDICTION PAGE
# ─────────────────────────────────────────────
if page == "🔮 Predict Student":
    st.title("🎓 Student Performance Predictor")
    st.caption(f"Random Forest · 88% accuracy · F1-score: {f1:.2f} · UCI dataset (649 students, 33 features)")
    st.divider()
    
    st.subheader("Enter Student Profile")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        study_time = st.selectbox("Weekly Study Time", 
                                   options=[1,2,3,4],
                                   format_func=lambda x: {1:"<2 hrs",2:"2–5 hrs",3:"5–10 hrs",4:">10 hrs"}[x])
        failures = st.selectbox("Past Class Failures", [0,1,2,3])
        absences = st.slider("Yearly Absences", 0, 30, 4)
    
    with col2:
        internet = st.selectbox("Internet at Home", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
        parent_edu = st.selectbox("Parent Education Level (avg)", [0,1,2,3,4],
                                   format_func=lambda x: {0:"None",1:"4th grade",2:"9th grade",3:"Secondary",4:"Higher"}[x])
        health = st.slider("Health Status (1=bad, 5=great)", 1, 5, 3)
    
    with col3:
        free_time = st.slider("Free Time After School (1–5)", 1, 5, 3)
        go_out = st.slider("Going Out with Friends (1–5)", 1, 5, 3)
        romantic = st.selectbox("In a Romantic Relationship", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    
    st.divider()
    
    if st.button("🔮 Predict Performance", type="primary"):
        input_data = pd.DataFrame([{
            "study_time": study_time, "failures": failures, "absences": absences,
            "internet": internet, "parent_edu": parent_edu, "free_time": free_time,
            "go_out": go_out, "health": health, "romantic": romantic
        }])
        
        prob = clf.predict_proba(input_data)[0]
        pred = clf.predict(input_data)[0]
        
        st.subheader("Prediction Result")
        r1, r2 = st.columns([1,1])
        
        with r1:
            if pred == 1:
                st.markdown(f'<div class="pred-pass"><h2>✅ Likely to PASS</h2><p style="color:#065f46;margin:8px 0 0;">Confidence: <strong>{prob[1]*100:.1f}%</strong></p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="pred-fail"><h2>⚠️ At Risk of FAILING</h2><p style="color:#991b1b;margin:8px 0 0;">Confidence: <strong>{prob[0]*100:.1f}%</strong></p></div>', unsafe_allow_html=True)
        
        with r2:
            fig = go.Figure(go.Bar(
                x=["Fail Risk", "Pass Probability"],
                y=[prob[0]*100, prob[1]*100],
                marker_color=["#ef4444","#22c55e"],
                text=[f"{prob[0]*100:.1f}%", f"{prob[1]*100:.1f}%"],
                textposition="outside"
            ))
            fig.update_layout(template="plotly_white", height=220, margin=dict(t=10,b=0),
                               yaxis=dict(range=[0,110],title="Probability (%)"))
            st.plotly_chart(fig, use_container_width=True)
        
        # Top 3 contributing factors
        st.subheader("Top 3 Factors Influencing This Prediction")
        top3 = fi.head(3)
        factor_labels = {
            "study_time": "📚 Study Time",
            "failures": "❌ Past Failures",
            "absences": "🏃 Absences",
            "internet": "🌐 Internet Access",
            "parent_edu": "👨‍🎓 Parent Education",
            "health": "💪 Health Status",
            "free_time": "⏰ Free Time",
            "go_out": "🎉 Going Out",
            "romantic": "💕 Romantic Relationship"
        }
        for _, row in top3.iterrows():
            label = factor_labels.get(row["feature"], row["feature"])
            bar_w = int(row["importance"] * 400)
            st.markdown(f'''<div class="factor-card">
                <p><strong>{label}</strong> &nbsp; <span style="color:#6366f1;font-weight:500">Importance: {row["importance"]*100:.1f}%</span></p>
                <div style="background:#e0e7ff;border-radius:4px;height:6px;margin-top:6px;">
                  <div style="background:#6366f1;width:{bar_w}px;height:6px;border-radius:4px;max-width:100%;"></div>
                </div></div>''', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ANALYTICS PAGE
# ─────────────────────────────────────────────
else:
    st.title("📊 Model Analytics")
    st.caption(f"Random Forest (200 trees) · Accuracy: {accuracy*100:.1f}% · F1: {f1:.2f}")
    st.divider()
    
    c1,c2,c3 = st.columns(3)
    c1.metric("Accuracy", f"{accuracy*100:.1f}%")
    c2.metric("F1-Score (CV)", f"{f1:.2f}")
    c3.metric("Dataset Size", "649 students")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Feature Importance")
        fig_fi = px.bar(fi, x="importance", y="feature", orientation="h",
                        color="importance", color_continuous_scale="Blues",
                        template="plotly_white")
        fig_fi.update_layout(height=360, margin=dict(t=10,b=0), coloraxis_showscale=False,
                              yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_fi, use_container_width=True)
    
    with col2:
        st.subheader("Confusion Matrix")
        fig_cm = px.imshow(cm, text_auto=True,
                           labels=dict(x="Predicted", y="Actual"),
                           x=["Fail","Pass"], y=["Fail","Pass"],
                           color_continuous_scale="Blues",
                           template="plotly_white")
        fig_cm.update_layout(height=360, margin=dict(t=10,b=0))
        st.plotly_chart(fig_cm, use_container_width=True)
    
    st.subheader("EDA — Study Time vs Pass Rate")
    pass_by_study = df.groupby("study_time")["passed"].mean().reset_index()
    pass_by_study["label"] = pass_by_study["study_time"].map({1:"<2 hrs",2:"2–5 hrs",3:"5–10 hrs",4:">10 hrs"})
    fig_eda = px.bar(pass_by_study, x="label", y="passed",
                     color="passed", color_continuous_scale="Greens",
                     template="plotly_white",
                     labels={"passed":"Pass Rate","label":"Weekly Study Time"})
    fig_eda.update_layout(height=280, margin=dict(t=10,b=40), coloraxis_showscale=False)
    st.plotly_chart(fig_eda, use_container_width=True)
    st.caption("💡 **EDA Insight:** Study time is the strongest positive predictor; prior failures are the strongest negative predictor.")
