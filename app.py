import streamlit as st
import pickle
import re
import string
import numpy as np
from textblob import TextBlob
import plotly.graph_objects as go
import os

# =========================
# GROQ SETUP
# =========================
try:
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    USE_AI = True
except:
    USE_AI = False

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Essay Analyzer", layout="wide")

# =========================
# DARK MODE
# =========================
dark_mode = st.toggle("🌙 Dark Mode")

bg = "#0f172a" if dark_mode else "white"
text = "white" if dark_mode else "black"

st.markdown(f"""
<style>
body {{
    background-color: {bg};
    color: {text};
}}

.stButton>button {{
    background: linear-gradient(45deg, #ff512f, #dd2476);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 220px;
}}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
try:
    model = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
except:
    st.error("❌ model.pkl or vectorizer.pkl not found")
    st.stop()

# =========================
# FUNCTIONS
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\n', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# 👉 FIXED: convert classification → score
def content_score(text):
    try:
        if not text.strip():
            return 0

        vec = vectorizer.transform([clean_text(text)])
        pred = model.predict(vec)[0]

        # Convert 0/1 → score
        score = 8 if pred == 1 else 4

        return score
    except:
        return 0

def grammar_score(text):
    blob = TextBlob(text)
    corrected = blob.correct()
    words = len(text.split())

    if words == 0:
        return 0, text

    errors = sum(1 for o, c in zip(text.split(), str(corrected).split()) if o != c)
    score = max(0, 10 - (errors / words) * 50)

    return round(score, 2), corrected

def structure_score(text):
    sentences = [s for s in text.split('.') if s.strip()]
    if not sentences:
        return 0

    avg_len = np.mean([len(s.split()) for s in sentences])
    readability = max(0, 100 - (avg_len * 2))

    score = (avg_len / 20) + (readability / 100) * 5

    return round(max(0, min(10, score)), 2)

def percentile(score):
    return max(1, min(99, int((score / 10) * 100)))

# =========================
# AI FEEDBACK
# =========================
def ai_feedback(text, score):
    if USE_AI:
        try:
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "You are an expert essay grader."},
                    {"role": "user", "content": f"""
Analyze this essay and give:

1. Strengths
2. Weaknesses
3. Improvements

Essay:
{text}

Score: {score}/10
"""}
                ]
            )
            return response.choices[0].message.content
        except:
            return fallback_feedback(score)
    return fallback_feedback(score)

def fallback_feedback(score):
    if score > 8:
        return "Excellent work. Add deeper insights."
    elif score > 5:
        return "Good effort. Improve grammar and structure."
    else:
        return "Needs improvement. Focus on clarity and grammar."

def ai_rewrite(text):
    if USE_AI:
        try:
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": f"Rewrite this essay better:\n{text}"}]
            )
            return response.choices[0].message.content
        except:
            return text
    return text

# =========================
# EVALUATION
# =========================
def evaluate(text):
    c = content_score(text)
    g, corrected = grammar_score(text)
    s = structure_score(text)
    final = round((c*0.5 + g*0.25 + s*0.25), 2)
    return c, g, s, final, corrected

# =========================
# UI
# =========================
st.title("✨ AI Essay Analyzer")

essay = st.text_area("✍️ Enter your essay:", height=250)

if st.button("🚀 Evaluate Essay"):

    if essay.strip() == "":
        st.warning("Enter essay")
    else:
        c, g, s, final, corrected = evaluate(essay)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Content", c)
        col2.metric("Grammar", g)
        col3.metric("Structure", s)
        col4.metric("Final", final)

        p = percentile(final)
        st.info(f"📊 Better than {p}% students")

        st.progress(int(final * 10))

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[c, g, s, c],
            theta=["Content", "Grammar", "Structure", "Content"],
            fill='toself'
        ))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 10])))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🧠 Feedback")
        st.write(ai_feedback(essay, final))

        st.subheader("✍️ Corrected Essay")
        st.write(corrected)

        st.subheader("🚀 Improved Version")
        improved = ai_rewrite(essay)
        st.write(improved)

        new_score = content_score(improved)
        st.success(f"🔥 Improved Score: {new_score}/10")