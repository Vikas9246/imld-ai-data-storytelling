import streamlit as st
import pandas as pd
import re
import difflib
from google import genai
from google.genai import types
from textstat import flesch_reading_ease
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------- Page config ----------
st.set_page_config(page_title="Data Storytelling", layout="wide")
st.title("AI‑Powered Data Storytelling")

# ---------- Secure API key ----------
client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

# ---------- Dataset ----------
@st.cache_data
def load_data():
    df = pd.read_csv("data/data.csv")
    return df

df = load_data()

def build_data_summary(df):
    summary = "Road traffic death rates per 100,000 population.\n"
    for country in sorted(df["country"].unique()):
        cdf = df[df["country"] == country]
        rates = ", ".join(
            [f"{row['year']} – {row['road_traffic_deaths_per_100k']}" for _, row in cdf.iterrows()]
        )
        summary += f"{country}: {rates}.\n"
    return summary

data_summary = build_data_summary(df)

# ---------- Human story ----------
human_story = """
**A Tale of Three Countries**  
Road traffic injuries are a leading cause of death worldwide, but the burden is not shared equally. In 2019, Brazil recorded 19.7 deaths per 100,000 people – nearly five times the German rate of 3.7. India stood at 15.6, still dangerously high.  

The pandemic briefly emptied roads in 2020: deaths dropped everywhere. Germany reached a low of 3.2, India fell to 13.7, and Brazil to 17.1. Many hoped this was a turning point.  

It wasn’t. By 2021, the numbers had crept back up. India rose to 14.2, Brazil to 18.3. Only Germany continued a small decline (2.9). The data tells us that without sustained safety policies, progress is fragile – and lives remain at risk.
"""

# ---------- Evaluation helpers ----------
analyzer = SentimentIntensityAnalyzer()

def readability_score(text):
    return round(flesch_reading_ease(text), 1)

def sentiment_scores(text):
    return analyzer.polarity_scores(text)["compound"]

def extract_numbers(text):
    return re.findall(r"\d+\.?\d*", text)

def factual_check(story, data_summary):
    story_nums = set(extract_numbers(story))
    data_nums = set(extract_numbers(data_summary))
    extra = story_nums - data_nums
    return len(extra) == 0, extra

def highlight_diff(original, modified):
    """Return HTML string with changes highlighted."""
    diff = difflib.ndiff(original.split(), modified.split())
    html_parts = []
    for word in diff:
        if word.startswith("+ "):
            html_parts.append(f"<span style='background-color:#a5d6a7; padding:0 2px;'>{word[2:]}</span>")
        elif word.startswith("- "):
            html_parts.append(f"<span style='background-color:#ffab91; padding:0 2px;'>{word[2:]}</span>")
        elif word.startswith("? "):
            continue
        else:
            html_parts.append(word[2:])
    return " ".join(html_parts)

# ---------- LLM helpers ----------
def generate_llm_story(summary):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"Dataset summary:\n{summary}\n\nWrite a data story of about 150-200 words.",
        config=types.GenerateContentConfig(
            system_instruction="You are a data journalist. Write a short, factual data story based **only** on the provided dataset. Use the numbers accurately. Do not add any information that is not in the data. The tone should be neutral and informative."
        )
    )
    return response.text

def enhance_story(original_story, summary, emotion="empathetic"):
    emotion_map = {
        "empathetic": "more empathetic and caring, as if you are telling a human story behind the numbers",
        "alarming": "more alarming and urgent, emphasizing the risk and danger",
        "hopeful": "more hopeful and positive, highlighting progress and solutions",
        "neutral": "more neutral and balanced, sticking closely to the facts without emotional loading"
    }
    instruction = emotion_map.get(emotion, emotion_map["empathetic"])

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"Original story:\n{original_story}\n\nDataset facts (for reference):\n{summary}\n\nRewrite with the requested emotional tone.",
        config=types.GenerateContentConfig(
            system_instruction=f"You are an expert in emotional data storytelling. Rewrite the given story to be {instruction} while keeping **every single number, statistic, and factual claim exactly as it appears**. Do not add new data points. Do not change any number."
        )
    )
    return response.text

# ---------- Session state ----------
if "llm_story" not in st.session_state:
    st.session_state.llm_story = ""
if "enhanced_story" not in st.session_state:
    st.session_state.enhanced_story = ""
if "show_diff" not in st.session_state:
    st.session_state.show_diff = False

# ---------- UI: Column layout ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("✍️ Human‑Written Story")
    st.markdown(human_story)
    st.caption(f"**Readability (Flesch):** {readability_score(human_story)}  |  **Sentiment (compound):** {sentiment_scores(human_story):.2f}")

with col2:
    st.subheader("🤖 LLM‑Generated Story")
    if st.session_state.llm_story:
        st.markdown(st.session_state.llm_story)
        st.caption(f"**Readability:** {readability_score(st.session_state.llm_story)}  |  **Sentiment:** {sentiment_scores(st.session_state.llm_story):.2f}")
        ok, extra_nums = factual_check(st.session_state.llm_story, data_summary)
        if ok:
            st.success("✅ Fact check: all numbers match dataset")
        else:
            st.warning(f"⚠️ Extra numbers not in dataset: {extra_nums}")
    else:
        st.info("Click the button below to generate a story.")

# ---------- Action buttons ----------
col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
with col_btn1:
    if st.button("Generate LLM Story"):
        with st.spinner("Generating with Gemini..."):
            st.session_state.llm_story = generate_llm_story(data_summary)
            st.session_state.enhanced_story = ""
            st.session_state.show_diff = False
        st.rerun()
with col_btn2:
    if st.button("Reset"):
        st.session_state.llm_story = ""
        st.session_state.enhanced_story = ""
        st.session_state.show_diff = False
        st.rerun()

# ---------- Agentic enhancement ----------
if st.session_state.llm_story:
    st.markdown("---")
    st.subheader("🎭 Agentic Emotional Enhancement")
    emotion = st.selectbox("Choose emotional tone:", ["empathetic", "alarming", "hopeful", "neutral"], key="emotion")
    col_enhance, col_reset_enhance = st.columns([2, 1])
    with col_enhance:
        if st.button("Enhance Story"):
            with st.spinner("Agent rewriting..."):
                enhanced = enhance_story(st.session_state.llm_story, data_summary, emotion)
                st.session_state.enhanced_story = enhanced
                st.session_state.show_diff = True
            st.rerun()
    with col_reset_enhance:
        if st.button("Clear Enhancement"):
            st.session_state.enhanced_story = ""
            st.session_state.show_diff = False
            st.rerun()

    if st.session_state.enhanced_story:
        st.markdown("### Enhanced Story")
        st.markdown(st.session_state.enhanced_story)
        st.caption(f"**Readability:** {readability_score(st.session_state.enhanced_story)}  |  **Sentiment:** {sentiment_scores(st.session_state.enhanced_story):.2f}")
        ok_en, extra_en = factual_check(st.session_state.enhanced_story, data_summary)
        if ok_en:
            st.success("✅ Fact check: all numbers match dataset")
        else:
            st.warning(f"⚠️ Extra numbers found: {extra_en}")

        # Diff view
        if st.session_state.show_diff:
            with st.expander("🔍 See what changed (diff view)"):
                diff_html = highlight_diff(st.session_state.llm_story, st.session_state.enhanced_story)
                st.markdown(diff_html, unsafe_allow_html=True)

# ---------- Dataset chart ----------
st.markdown("---")
st.subheader("📊 Dataset Overview")
chart_data = df.pivot(index="year", columns="country", values="road_traffic_deaths_per_100k")
st.line_chart(chart_data)