import streamlit as st
import pickle
import re
import string
import numpy as np
import textstat
from textblob import TextBlob
import plotly.graph_objects as go
import os

# =========================
# GROQ SETUP 🔥
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
st.set_page_config(page_title="AI Essay Grader", layout="wide")

# =========================
# DARK MODE 🌙
# =========================
dark_mode = st.toggle("🌙 Dark Mode")

# =========================
# CSS (ANIMATED UI)
# =========================
st.markdown("""
<style>
body {
    background: linear-gradient(270deg, #00c6ff, #0072ff, #00c6ff);
    background-size: 600% 600%;
    animation: gradient 12s ease infinite;
}

@keyframes gradient {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.stButton>button {
    background: linear-gradient(45deg, #ff512f, #dd2476);
    color: white;
    border-radius: 15px;
    height: 3.5em;
    width: 260px;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# =========================
# FUNCTIONS
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\n', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def content_score(text):
    vec = vectorizer.transform([clean_text(text)])
    return round(model.predict(vec)[0], 2)

def grammar_score(text):
    blob = TextBlob(text)
    corrected = blob.correct()
    words = len(text.split())
    errors = sum(1 for o, c in zip(text.split(), str(corrected).split()) if o != c)
    score = max(0, 10 - (errors/words)*50) if words else 0
    return round(score,2), corrected

def structure_score(text):
    sentences = [s for s in text.split('.') if s.strip()]
    if not sentences:
        return 0

    avg_len = np.mean([len(s.split()) for s in sentences])

    # custom readability score
    words = len(text.split())
    readability = max(0, 100 - (avg_len * 2))

    score = (avg_len / 20) + (readability / 100) * 5
    return round(max(0, min(10, score)), 2)

# =========================
# 🔥 GROQ AI REWRITE
# =========================
def ai_rewrite(text):
    if USE_AI:
        try:
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "You are an expert academic essay writer."},
                    {"role": "user", "content": f"""
Rewrite the following essay to achieve a perfect academic score (10/10).

- Improve CONTENT (add depth, examples)
- Improve STRUCTURE (intro, body, conclusion)
- Fix GRAMMAR completely
- Make it formal and high-quality

Essay:
{text}
"""}
                ]
            )
            return response.choices[0].message.content
        except:
            return fallback_rewrite(text)
    else:
        return fallback_rewrite(text)

# =========================
# FALLBACK (NO API)
# =========================
def fallback_rewrite(text):
    blob = TextBlob(text)
    corrected = str(blob.correct())

    intro = "This essay provides a comprehensive discussion on the topic. "
    body = corrected + " This topic plays a crucial role in modern society and influences various aspects of life."
    conclusion = " In conclusion, the topic is highly significant and requires further attention."

    return intro + "\n\n" + body + "\n\n" + conclusion

# =========================
# EVALUATION
# =========================
def evaluate(text):
    c = content_score(text)
    g, corrected = grammar_score(text)
    s = structure_score(text)
    final = round((c*0.5 + g*0.25 + s*0.25),2)
    return c,g,s,final,corrected

# =========================
# UI
# =========================
st.title("✨ AI Essay Grader (Groq Powered)")
essay = st.text_area("✍️ Enter your essay:", height=250)

if st.button("🚀 Evaluate Essay"):
    
    if essay.strip() == "":
        st.warning("Please enter essay")
    
    else:
        c,g,s,final,corrected = evaluate(essay)

        # SCORES
        col1, col2, col3 = st.columns(3)
        col1.metric("📚 Content", c)
        col2.metric("🧠 Grammar", g)
        col3.metric("🏗 Structure", s)

        st.success(f"🎯 Final Score: {final}/10")

        # GRAPH
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Content","Grammar","Structure"],
            y=[c,g,s]
        ))
        st.plotly_chart(fig)

        # CORRECTED
        st.subheader("✍️ Grammar Corrected")
        st.write(corrected)

        # 🔥 AI REWRITE
        st.subheader("🚀 AI High-Score Essay (Groq)")
        improved = ai_rewrite(essay)
        st.write(improved)

        # NEW SCORE
        new_score = content_score(improved)
        st.success(f"🔥 Improved Score: {new_score}/10")