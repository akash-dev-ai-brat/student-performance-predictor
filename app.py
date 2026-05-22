import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os
# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor | Akash Nath",
    page_icon="📊",
    layout="wide"
)

# ─── Load Model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists("model.pkl"):
        import model as m  # auto-train if pkl not found
    model    = joblib.load("model.pkl")
    le       = joblib.load("label_encoder.pkl")
    features = joblib.load("features.pkl")
    return model, le, features
model, le, features = load_model()

feat_labels = [
    "Study Hours", "Attendance", "Sleep Hours", "Previous Score",
    "Assignments", "Stress Level", "Extra Curricular",
    "Internet Hours", "Family Support", "Part Time Job"
]

# ─── Grade Config ──────────────────────────────────────────────
GRADE_CONFIG = {
    "A": {"color": "#00D4AA", "emoji": "🏆", "msg": "Outstanding! Keep it up!",      "bg": "#00D4AA18"},
    "B": {"color": "#4FC3F7", "emoji": "⭐", "msg": "Great performance!",             "bg": "#4FC3F718"},
    "C": {"color": "#FFD700", "emoji": "📚", "msg": "Good, but room to improve!",     "bg": "#FFD70018"},
    "D": {"color": "#FF9800", "emoji": "⚠️", "msg": "Need more effort!",              "bg": "#FF980018"},
    "F": {"color": "#FF4444", "emoji": "❌", "msg": "At risk — take action now!",     "bg": "#FF444418"},
}

# ─── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #0f0f1a, #1a1a3e, #0f3460);
    border-radius: 20px; padding: 40px; text-align: center;
    margin-bottom: 28px; border: 1px solid #ffffff15;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.hero-title {
    font-size: 2.8em; font-weight: 700;
    background: linear-gradient(90deg, #00D4AA, #4FC3F7, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub { color: #aaaacc; font-size: 1.05em; margin-top: 8px; }
.badge {
    display: inline-block;
    background: linear-gradient(90deg, #00D4AA, #4FC3F7);
    color: #0f0f1a; padding: 4px 16px; border-radius: 20px;
    font-size: 0.82em; font-weight: 700; margin-top: 12px;
}
.card {
    background: #1a1a2e; border-radius: 16px; padding: 24px;
    border: 1px solid #ffffff15; margin-bottom: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}
.result-card {
    border-radius: 16px; padding: 28px; text-align: center;
    margin: 12px 0; border: 1px solid #ffffff20;
}
.grade-letter {
    font-size: 5em; font-weight: 800; letter-spacing: 4px;
    line-height: 1;
}
.grade-msg { font-size: 1.1em; margin-top: 8px; opacity: 0.9; }
.metric-box {
    background: #0f0f1a; border-radius: 12px; padding: 16px;
    text-align: center; border: 1px solid #ffffff10;
}
.metric-val { font-size: 1.8em; font-weight: 700; color: #00D4AA; }
.metric-lbl { font-size: 0.8em; color: #888; margin-top: 4px; }
.tip {
    background: #0f3460; border-radius: 10px; padding: 12px 16px;
    color: #aaaacc; font-size: 0.87em;
    border-left: 4px solid #00D4AA; margin-top: 10px;
}
.stButton > button {
    background: linear-gradient(90deg, #00D4AA, #4FC3F7) !important;
    color: #0f0f1a !important; border: none !important;
    border-radius: 10px !important; padding: 12px 28px !important;
    font-weight: 700 !important; font-size: 1.05em !important;
    width: 100% !important;
}
.section-title {
    color: #00D4AA; font-size: 1em; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1px;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# ─── HERO ──────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <div class='hero-title'>📊 Student Performance Predictor</div>
    <div class='hero-sub'>AI-powered grade prediction using Machine Learning</div>
    <div class='badge'>Built by Akash Nath · AI & Data Science</div>
</div>
""", unsafe_allow_html=True)

# ─── TABS ──────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🎯  Predict My Grade", "📈  Insights & Tips"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — PREDICTOR
# ══════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📚 Academic Inputs</div>", unsafe_allow_html=True)

        study_hours  = st.slider("📖 Daily Study Hours",        0.0, 14.0, 5.0, 0.5)
        attendance   = st.slider("🏫 Attendance (%)",           30,  100,  75)
        prev_score   = st.slider("📝 Previous Exam Score (%)",  20,  100,  65)
        assignments  = st.slider("✅ Assignment Completion (%)", 0,   100,  75)

        st.markdown("<div class='section-title' style='margin-top:20px'>🧠 Lifestyle Inputs</div>", unsafe_allow_html=True)

        sleep_hours      = st.slider("😴 Daily Sleep Hours",          3.0, 12.0, 7.0, 0.5)
        stress_level     = st.slider("😰 Stress Level (1=low, 10=high)", 1, 10, 5)
        internet_hours   = st.slider("📱 Daily Internet/Social Media Hours", 0.0, 12.0, 4.0, 0.5)
        extra_curricular = st.slider("⚽ Extra-curricular Activities",  0, 3, 1)
        family_support   = st.slider("👨‍👩‍👧 Family Support (1=low, 5=high)", 1, 5, 3)
        part_time_job    = st.radio("💼 Do you have a part-time job?", [0, 1],
                                     format_func=lambda x: "Yes" if x == 1 else "No",
                                     horizontal=True)

        st.markdown("</div>", unsafe_allow_html=True)
        predict_btn = st.button("🚀 Predict My Grade!")

    with col2:
        if predict_btn:
            input_data = pd.DataFrame([[
                study_hours, attendance, sleep_hours, prev_score,
                assignments, stress_level, extra_curricular,
                internet_hours, family_support, part_time_job
            ]], columns=features)

            pred_encoded = model.predict(input_data)[0]
            pred_proba   = model.predict_proba(input_data)[0]
            grade        = le.inverse_transform([pred_encoded])[0]
            cfg          = GRADE_CONFIG[grade]

            # ── Result card ────────────────────────────────────
            st.markdown(f"""
            <div class='result-card' style='background:{cfg["bg"]}; border-color:{cfg["color"]}44;'>
                <div style='font-size:2em'>{cfg["emoji"]}</div>
                <div class='grade-letter' style='color:{cfg["color"]}'>{grade}</div>
                <div class='grade-msg' style='color:{cfg["color"]}'>{cfg["msg"]}</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Confidence metrics ─────────────────────────────
            top_idx   = np.argsort(pred_proba)[::-1][:3]
            top_grades= le.inverse_transform(top_idx)
            top_probs = pred_proba[top_idx]

            cols = st.columns(3)
            for i, (g, p) in enumerate(zip(top_grades, top_probs)):
                c = GRADE_CONFIG[g]
                with cols[i]:
                    st.markdown(f"""
                    <div class='metric-box'>
                        <div class='metric-val' style='color:{c["color"]}'>{g}</div>
                        <div style='font-size:1.1em; color:{c["color"]}'>{p*100:.1f}%</div>
                        <div class='metric-lbl'>probability</div>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Probability bar chart ──────────────────────────
            all_grades = le.inverse_transform(range(len(pred_proba)))
            colors_list= [GRADE_CONFIG[g]["color"] for g in all_grades]

            fig = go.Figure(go.Bar(
                x=list(all_grades),
                y=[p*100 for p in pred_proba],
                marker_color=colors_list,
                text=[f"{p*100:.1f}%" for p in pred_proba],
                textposition="outside"
            ))
            fig.update_layout(
                title="Grade Probability Distribution",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccccee"),
                yaxis=dict(title="Probability (%)", gridcolor="#333355"),
                xaxis=dict(title="Grade"),
                showlegend=False,
                height=320,
                margin=dict(t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Smart tips ─────────────────────────────────────
            tips = []
            if study_hours < 4:
                tips.append("📖 Try to study at least 4-5 hours daily for better results")
            if attendance < 75:
                tips.append("🏫 Attendance below 75% significantly impacts your grade")
            if sleep_hours < 6:
                tips.append("😴 Sleep at least 7 hours — it boosts memory and focus")
            if stress_level > 7:
                tips.append("🧘 High stress hurts performance — try meditation or exercise")
            if internet_hours > 6:
                tips.append("📱 Reduce social media time — it's eating into your study hours")
            if assignments < 60:
                tips.append("✅ Complete more assignments — they boost your final score")

            if tips:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<div class='section-title'>💡 Personalized Tips</div>", unsafe_allow_html=True)
                for tip in tips:
                    st.markdown(f"<div class='tip'>{tip}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='tip'>🌟 Your habits look great! Keep maintaining this routine!</div>", unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class='card' style='text-align:center; padding:80px 20px; color:#555577;'>
                <div style='font-size:4em'>🎯</div>
                <div style='margin-top:16px; font-size:1.1em'>Adjust the sliders and click</div>
                <div style='color:#00D4AA; font-weight:700; font-size:1.2em; margin-top:8px'>
                    "Predict My Grade!"
                </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 2 — INSIGHTS
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📊 What factors matter most for your grade?</div>", unsafe_allow_html=True)

    importance = model.feature_importances_
    
    sorted_idx = np.argsort(importance)[::-1]

    fig2 = go.Figure(go.Bar(
        x=[importance[i]*100 for i in sorted_idx],
        y=[feat_labels[i] for i in sorted_idx],
        orientation='h',
        marker=dict(
            color=[importance[i]*100 for i in sorted_idx],
            colorscale=[[0, "#4FC3F7"], [0.5, "#00D4AA"], [1, "#a78bfa"]],
            showscale=False
        ),
        text=[f"{importance[i]*100:.1f}%" for i in sorted_idx],
        textposition="outside"
    ))
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccccee"),
        xaxis=dict(title="Importance (%)", gridcolor="#333355"),
        yaxis=dict(autorange="reversed"),
        height=420,
        margin=dict(t=20, b=20, l=140)
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Study tips grid ────────────────────────────────────────
    st.markdown("<div class='section-title'>🌟 Top Study Tips to Get Grade A</div>", unsafe_allow_html=True)
    tips_data = [
        ("📖", "Study 6+ Hours",        "Consistent daily study is the #1 factor in performance"),
        ("🏫", "90%+ Attendance",        "Never miss class — attendance strongly predicts success"),
        ("😴", "Sleep 7-8 Hours",        "Good sleep improves memory retention by up to 40%"),
        ("✅", "Submit All Assignments", "Assignments build understanding AND boost scores"),
        ("📱", "Limit Screen Time",      "Less than 3hrs social media keeps focus sharp"),
        ("🧘", "Manage Stress",          "Exercise, breaks and hobbies keep stress in check"),
    ]
    c1, c2, c3 = st.columns(3)
    for i, (emoji, title, desc) in enumerate(tips_data):
        col = [c1, c2, c3][i % 3]
        with col:
            st.markdown(f"""
            <div class='card' style='text-align:center; padding:20px'>
                <div style='font-size:2em'>{emoji}</div>
                <div style='font-weight:700; color:#00D4AA; margin:8px 0'>{title}</div>
                <div style='color:#aaaacc; font-size:0.87em'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ─── Footer ────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; color:#555577; font-size:0.85em; padding:30px 0 10px'>
    📊 Student Performance Predictor · Built with Python, scikit-learn & Streamlit ·
    <b style='color:#00D4AA'>Akash Nath</b> · AI & Data Science
</div>
""", unsafe_allow_html=True)