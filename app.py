import streamlit as st
import pandas as pd
import numpy as np
import re
import html
import difflib
import plotly.graph_objects as go
from google import genai
from google.genai import types
from textstat import flesch_reading_ease
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI-Powered Data Storytelling – Student Depression",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  Custom CSS  – clean, readable, presentation-ready
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .story-box {
        background: #1e1e2e;
        border-left: 4px solid #6c8ebf;
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.5rem;
        color: #e0e0e0;
        line-height: 1.75;
    }
    .metric-row {
        display: flex;
        gap: 1.2rem;
        margin: 0.4rem 0 0.8rem 0;
        font-size: 0.82rem;
        color: #9e9e9e;
    }
    .metric-pill {
        background: #2a2a3e;
        padding: 2px 10px;
        border-radius: 20px;
    }
    .section-label {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #6c8ebf;
        margin-bottom: 0.3rem;
    }
    .fact-ok   { color: #81c784; font-size: 0.82rem; }
    .fact-warn { color: #ffb74d; font-size: 0.82rem; }
    .story-box p { margin: 0 0 0.7rem 0; }
    .story-box p:last-child { margin-bottom: 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Linked-visualisation: topics, keywords & colours
# ─────────────────────────────────────────────
# Each chart maps to one "topic". Clicking a bar highlights every sentence
# in every story that mentions that topic (keyword match).
TOPIC_KEYWORDS = {
    "academic_pressure": ["academic", "pressure", "study", "grade", "cgpa"],
    "financial_stress":  ["financial", "stress", "money", "rent", "fee"],
    "sleep":             ["sleep", "hours", "rest", "tired", "battery"],
}
# Light tints (chart-matched) used as the highlight background; dark text on top.
TOPIC_HL_COLORS = {
    "academic_pressure": "#fff3b0",   # amber/yellow
    "financial_stress":  "#ffd1d1",   # red tint
    "sleep":             "#cdeccd",   # green tint
}
TOPIC_LABELS = {
    "academic_pressure": "Academic Pressure",
    "financial_stress":  "Financial Stress",
    "sleep":             "Sleep",
}

def split_sentences(text):
    """Split a paragraph into sentences on . ! ? boundaries (keeps punctuation)."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]

def sentence_matches_topic(sentence, topic):
    kws = TOPIC_KEYWORDS.get(topic, [])
    low = sentence.lower()
    return any(kw in low for kw in kws)

def _md_inline_to_html(s):
    """Minimal markdown → HTML for inline bold so stories render properly inside a div."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)

def render_story_html(text, highlight_topic=None):
    """Render a story inside .story-box, optionally highlighting sentences
    that match the given topic. Returns an HTML string for st.markdown."""
    paragraphs = [p for p in text.strip().split("\n\n") if p.strip()]
    out_paras = []
    for para in paragraphs:
        rendered = []
        for sent in split_sentences(para):
            h = _md_inline_to_html(sent)
            if highlight_topic and sentence_matches_topic(sent, highlight_topic):
                color = TOPIC_HL_COLORS.get(highlight_topic, "#fff3b0")
                rendered.append(
                    f"<span style='background:{color};color:#1a1a1a;"
                    f"border-radius:3px;padding:1px 3px;'>{h}</span>"
                )
            else:
                rendered.append(h)
        out_paras.append(" ".join(rendered))
    body = "".join(f"<p>{p}</p>" for p in out_paras)
    return f"<div class='story-box'>{body}</div>"

# ─────────────────────────────────────────────
#  Gemini client
# ─────────────────────────────────────────────
client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

# ─────────────────────────────────────────────
#  Load & cache dataset
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/data.csv")
    # Normalise column names – strip whitespace, title-case
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# ─────────────────────────────────────────────
#  Dataset statistics (computed once, cached)
# ─────────────────────────────────────────────
@st.cache_data
def compute_stats(df):
    stats = {}

    # Overall depression rate
    stats["total"] = len(df)
    stats["depression_rate"] = round(df["Depression"].mean() * 100, 1)

    # Academic pressure → depression rate (scale 1-5)
    ap = (
        df[df["Academic Pressure"] > 0]
        .groupby("Academic Pressure")["Depression"]
        .mean()
        .mul(100)
        .round(1)
    )
    stats["academic_pressure"] = ap.to_dict()

    # Financial stress → depression rate (scale 1-5)
    fs = (
        df[df["Financial Stress"] > 0]
        .groupby("Financial Stress")["Depression"]
        .mean()
        .mul(100)
        .round(1)
    )
    stats["financial_stress"] = fs.to_dict()

    # Sleep duration → depression rate
    sleep_order = ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"]
    sleep_map = {}
    for s in sleep_order:
        subset = df[df["Sleep Duration"] == s]
        if len(subset) > 0:
            sleep_map[s] = round(subset["Depression"].mean() * 100, 1)
    stats["sleep"] = sleep_map

    # Gender breakdown
    gender_stats = {}
    for g in df["Gender"].dropna().unique():
        sub = df[df["Gender"] == g]
        gender_stats[g] = round(sub["Depression"].mean() * 100, 1)
    stats["gender"] = gender_stats

    # Key numbers for story and fact-check
    ap_dict = stats["academic_pressure"]
    fs_dict = stats["financial_stress"]
    sl_dict = stats["sleep"]

    stats["ap_low"]  = ap_dict.get(1.0, ap_dict.get(min(ap_dict.keys())))
    stats["ap_high"] = ap_dict.get(5.0, ap_dict.get(max(ap_dict.keys())))
    stats["fs_low"]  = fs_dict.get(1.0, fs_dict.get(min(fs_dict.keys())))
    stats["fs_high"] = fs_dict.get(5.0, fs_dict.get(max(fs_dict.keys())))
    stats["sleep_worst"] = sl_dict.get("Less than 5 hours", "N/A")
    stats["sleep_best"]  = sl_dict.get("More than 8 hours", "N/A")
    stats["sleep_56"]    = sl_dict.get("5-6 hours", "N/A")
    stats["sleep_78"]    = sl_dict.get("7-8 hours", "N/A")

    return stats

STATS = compute_stats(df)

# ─────────────────────────────────────────────
#  Build data summary for LLM (text)
# ─────────────────────────────────────────────
def build_data_summary(df, stats, gender_filter=None, sleep_filter=None):
    """Build a concise factual summary passed to the LLM."""
    sub = df.copy()
    applied = []
    if gender_filter and gender_filter != "All":
        sub = sub[sub["Gender"] == gender_filter]
        applied.append(f"gender={gender_filter}")
    if sleep_filter and sleep_filter != "All":
        sub = sub[sub["Sleep Duration"] == sleep_filter]
        applied.append(f"sleep='{sleep_filter}'")

    n = len(sub)
    dep_rate = round(sub["Depression"].mean() * 100, 1) if n > 0 else "N/A"

    lines = [
        f"Dataset: Student Depression Dataset (source: Kaggle, N={stats['total']:,} students)",
        f"Subset applied: {', '.join(applied) if applied else 'none (full dataset)'}",
        f"Subset size: {n:,} students | Depression rate in subset: {dep_rate}%",
        "",
        "Key findings (full dataset):",
        f"- Overall depression prevalence: {stats['depression_rate']}%",
        f"- Academic Pressure (scale 1-5): depression rate rises from {stats['ap_low']}% (level 1) to {stats['ap_high']}% (level 5)",
        f"- Financial Stress (scale 1-5): depression rate rises from {stats['fs_low']}% (level 1) to {stats['fs_high']}% (level 5)",
        f"- Sleep duration: 'Less than 5 hours' → {stats['sleep_worst']}% depressed; '5-6 hours' → {stats['sleep_56']}%; '7-8 hours' → {stats['sleep_78']}%; 'More than 8 hours' → {stats['sleep_best']}%",
    ]
    if stats.get("gender"):
        g_lines = [f"  {g}: {r}%" for g, r in stats["gender"].items()]
        lines.append("- Depression by gender: " + "; ".join(g_lines))
    return "\n".join(lines)

# ─────────────────────────────────────────────
#  Human-written story (team's "Zero Battery") – now filter-responsive (change #7)
# ─────────────────────────────────────────────
# Default persona: Arjun. Female-filter persona: Priya. Same structure & statistics.
_STORY_ARJUN = """**Zero Battery**

It's 1 AM. Arjun, a 23-year-old Computer Applications student, stares at his laptop screen — another assignment deadline looming, another month's rent overdue. He hasn't slept more than four hours in weeks. He wonders, quietly, if anyone else feels this hollow.

He's not alone. Across a dataset of nearly 28,000 students, one pattern cuts through the noise with devastating clarity: the students who face the most academic pressure are over **four times more likely** to be depressed than those who feel the least. Among students rating their academic burden at the maximum level, **{ap_high}%** report depression. Among those at the lowest level, just **{ap_low}%** do.

Financial stress tells the same story. Students struggling most financially show a depression rate of **{fs_high}%** — more than double the **{fs_low}%** seen among those with the least financial worry.

And sleep? Students sleeping fewer than five hours face a depression rate of **{fs_worst}%**. Those sleeping more than eight hours: just **{sleep_best}%**.

These aren't abstract statistics. They are Arjun's story — and the story of millions of students worldwide, caught between the weight of ambition and the fragility of mental health. The data doesn't tell us what to do. But it tells us clearly: we can't afford to keep looking away.
"""

_STORY_PRIYA = """**Zero Battery**

It's 1 AM. Priya, a 23-year-old Computer Applications student, stares at her laptop screen — another assignment deadline looming, another month's rent overdue. She hasn't slept more than four hours in weeks. She wonders, quietly, if anyone else feels this hollow.

She's not alone. Across a dataset of nearly 28,000 students, one pattern cuts through the noise with devastating clarity: the students who face the most academic pressure are over **four times more likely** to be depressed than those who feel the least. Among students rating their academic burden at the maximum level, **{ap_high}%** report depression. Among those at the lowest level, just **{ap_low}%** do.

Financial stress tells the same story. Students struggling most financially show a depression rate of **{fs_high}%** — more than double the **{fs_low}%** seen among those with the least financial worry.

And sleep? Students sleeping fewer than five hours face a depression rate of **{fs_worst}%**. Those sleeping more than eight hours: just **{sleep_best}%**.

These aren't abstract statistics. They are Priya's story — and the story of millions of students worldwide, caught between the weight of ambition and the fragility of mental health. The data doesn't tell us what to do. But it tells us clearly: we can't afford to keep looking away.
"""

# Extra opening sentence used when the sleep filter isolates the <5-hour group.
_SLEEP_LEAD = ("Tonight, like most nights this month, sleep simply won't come — "
               "fewer than five hours again, and the exhaustion has become a second self.")

def build_human_story(gender_filter=None, sleep_filter=None):
    """Return (story_text, variant_label) responding to the active filters."""
    fmt = dict(
        ap_high=STATS["ap_high"], ap_low=STATS["ap_low"],
        fs_high=STATS["fs_high"], fs_low=STATS["fs_low"],
        fs_worst=STATS["sleep_worst"],  # intentionally sleep_worst under fs label
        sleep_best=STATS["sleep_best"],
    )
    female = gender_filter == "Female"
    sleep_focus = sleep_filter == "Less than 5 hours"

    story = (_STORY_PRIYA if female else _STORY_ARJUN).format(**fmt)

    if sleep_focus:
        # Insert the extra lead sentence right after the title line.
        title, _, rest = story.partition("\n\n")
        story = f"{title}\n\n{_SLEEP_LEAD} {rest}"

    parts = []
    if female:
        parts.append("female variant")
    if sleep_focus:
        parts.append("sleep-focus variant")
    label = " + ".join(parts) if parts else "default story"
    return story, label

# ─────────────────────────────────────────────
#  Evaluation helpers
# ─────────────────────────────────────────────
analyzer = SentimentIntensityAnalyzer()

def readability_score(text):
    clean = re.sub(r"\*\*|__|\*|_|#+", "", text)
    return round(flesch_reading_ease(clean), 1)

def sentiment_score(text):
    return round(analyzer.polarity_scores(text)["compound"], 3)

def extract_numbers(text):
    return set(re.findall(r"\b\d+\.?\d*\b", text))

def factual_check(story, data_summary):
    """Check if numbers in story all appear in data summary."""
    story_nums   = extract_numbers(story)
    allowed_nums = extract_numbers(data_summary)
    # Always allow common non-data numbers: years (2019-2030), 100, 1000 etc.
    ignore = {"100", "1000", "5", "8", "28", "23", "4", "1", "2", "3", "0"}
    extra = story_nums - allowed_nums - ignore
    return len(extra) == 0, extra

def highlight_diff(original, modified):
    diff = difflib.ndiff(original.split(), modified.split())
    parts = []
    for token in diff:
        if token.startswith("+ "):
            parts.append(
                f"<span style='background:#1b5e20;color:#c8e6c9;padding:0 3px;border-radius:3px;'>"
                f"{token[2:]}</span>"
            )
        elif token.startswith("- "):
            parts.append(
                f"<span style='background:#b71c1c;color:#ffcdd2;padding:0 3px;border-radius:3px;text-decoration:line-through;'>"
                f"{token[2:]}</span>"
            )
        elif not token.startswith("? "):
            parts.append(token[2:])
    return " ".join(parts)

def _sentiment_color(compound):
    if compound >= 0.05:
        return "#66bb6a"   # green = positive
    if compound <= -0.05:
        return "#ef5350"   # red = negative
    return "#9e9e9e"        # grey = neutral

def render_sentiment_arc(text):
    """Render a per-sentence 'emotional tone arc': one coloured segment per
    sentence (VADER compound), with the sentence as a hover tooltip."""
    flat = re.sub(r"\s+", " ", re.sub(r"\*\*|__|\*|_|#+", "", text)).strip()
    sentences = split_sentences(flat)
    if not sentences:
        return
    segments = []
    for sent in sentences:
        comp = analyzer.polarity_scores(sent)["compound"]
        color = _sentiment_color(comp)
        tip = html.escape(f"{sent}  (compound {comp:+.2f})", quote=True)
        segments.append(
            f"<div title=\"{tip}\" style='flex:1 1 20px;max-width:34px;height:8px;"
            f"background:{color};border-radius:2px;'></div>"
        )
    bar = (
        "<div style='display:flex;gap:2px;margin:0.1rem 0 0.2rem 0;'>"
        + "".join(segments)
        + "</div>"
    )
    legend = (
        "<div style='font-size:0.72rem;color:#9e9e9e;margin-bottom:0.6rem;'>"
        "<b style='letter-spacing:0.05em;'>EMOTIONAL TONE ARC (SENTENCE-LEVEL)</b> &nbsp;·&nbsp; "
        "<span style='color:#ef5350;'>■</span> negative &nbsp; "
        "<span style='color:#9e9e9e;'>■</span> neutral &nbsp; "
        "<span style='color:#66bb6a;'>■</span> positive"
        "</div>"
    )
    st.markdown(legend + bar, unsafe_allow_html=True)

def render_metrics(text, label=""):
    r = readability_score(text)
    s = sentiment_score(text)
    sentiment_label = "positive" if s > 0.05 else ("negative" if s < -0.05 else "neutral")
    st.markdown(
        f"<div class='metric-row'>"
        f"<span class='metric-pill'>📖 Flesch: <b>{r}</b></span>"
        f"<span class='metric-pill'>💬 Sentiment: <b>{s}</b> ({sentiment_label})</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
#  Comparison "hero" cards  (change #3)
# ─────────────────────────────────────────────
def _badge(text, color):
    return (f"<span style='background:{color};color:#0e1117;font-weight:700;"
            f"padding:1px 9px;border-radius:10px;font-size:0.8rem;'>{text}</span>")

def _readability_color(r):
    return "#66bb6a" if r > 60 else ("#ffb74d" if r >= 40 else "#ef5350")

def _sentiment_badge_color(s):
    return "#66bb6a" if s > 0.05 else ("#ef5350" if s < -0.05 else "#9e9e9e")

def render_tone_shift(llm_text, enhanced_text, tone_name):
    """Horizontal −1…+1 gauge with two markers (LLM vs Enhanced VADER compound),
    connected by a coloured line showing the direction of the tone shift."""
    llm_c = sentiment_score(llm_text)
    enh_c = sentiment_score(enhanced_text)
    pos = lambda c: (c + 1) / 2 * 100
    p1, p2 = pos(llm_c), pos(enh_c)
    left, right = sorted([p1, p2])
    line_color = "#66bb6a" if enh_c > llm_c else ("#ef5350" if enh_c < llm_c else "#9e9e9e")
    direction = ("▲ warmer / more positive" if enh_c > llm_c
                 else ("▼ darker / more negative" if enh_c < llm_c else "→ unchanged"))
    block = f"""
    <div style='margin:0.4rem 0 0.9rem 0;'>
      <div style='font-size:0.86rem;color:#bdbdbd;margin-bottom:0.55rem;'>
        <b>Tone shift: LLM → Enhanced ({tone_name})</b>
        &nbsp; <span style='color:{line_color};font-weight:600;'>{direction}</span>
        &nbsp; <span style='color:#9e9e9e;'>({llm_c:+.3f} → {enh_c:+.3f})</span>
      </div>
      <div style='position:relative;height:30px;'>
        <div style='position:absolute;top:12px;left:0;right:0;height:6px;border-radius:3px;
             background:linear-gradient(to right,#ef5350,#9e9e9e,#66bb6a);opacity:0.35;'></div>
        <div style='position:absolute;top:13px;left:{left}%;width:{right-left}%;height:4px;
             background:{line_color};border-radius:2px;'></div>
        <div title='LLM ({llm_c:+.3f})' style='position:absolute;top:6px;left:calc({p1}% - 7px);
             width:14px;height:14px;border-radius:50%;background:#ce93d8;border:2px solid #0e1117;'></div>
        <div title='Enhanced ({enh_c:+.3f})' style='position:absolute;top:6px;left:calc({p2}% - 7px);
             width:14px;height:14px;border-radius:50%;background:#80cbc4;border:2px solid #0e1117;'></div>
      </div>
      <div style='display:flex;justify-content:space-between;font-size:0.7rem;color:#777;margin-top:2px;'>
        <span>−1 negative</span><span>0 neutral</span><span>+1 positive</span>
      </div>
      <div style='font-size:0.72rem;color:#9e9e9e;margin-top:3px;'>
        <span style='color:#ce93d8;'>●</span> LLM &nbsp;
        <span style='color:#80cbc4;'>●</span> Enhanced
      </div>
    </div>
    """
    st.markdown(block, unsafe_allow_html=True)

def render_comparison_card(label, story, data_summary, accent="#6c8ebf"):
    r = readability_score(story)
    s = sentiment_score(story)
    wc = len(story.split())
    ok, extra = factual_check(story, data_summary + "\n" + str(STATS))
    fact = "✅ Pass" if ok else f"⚠️ {len(extra)} flag(s)"
    fact_color = "#81c784" if ok else "#ffb74d"

    def row(lbl, value_html):
        return (f"<div style='display:flex;justify-content:space-between;align-items:center;"
                f"margin:0.4rem 0;font-size:0.85rem;color:#bdbdbd;'>"
                f"<span>{lbl}</span>{value_html}</div>")

    card = (
        f"<div style='background:#1e1e2e;border-top:3px solid {accent};border-radius:8px;"
        f"padding:0.9rem 1.1rem;height:100%;'>"
        f"<div style='font-weight:700;color:{accent};margin-bottom:0.5rem;font-size:0.95rem;'>{label}</div>"
        + row("📖 Readability", _badge(r, _readability_color(r)))
        + row("💬 Sentiment", _badge(s, _sentiment_badge_color(s)))
        + row("📝 Word count", f"<b style='color:#e0e0e0;'>{wc}</b>")
        + row("🔍 Fact-check", f"<span style='color:{fact_color};font-weight:600;'>{fact}</span>")
        + "</div>"
    )
    st.markdown(card, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LLM functions
# ─────────────────────────────────────────────
def generate_llm_story(summary):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=(
            f"Dataset summary:\n{summary}\n\n"
            "Write a data story of about 180–220 words about student depression based strictly on the numbers above. "
            "Start with a compelling hook. Mention at least 3 specific statistics. End with a reflective conclusion."
        ),
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are a data journalist writing about student mental health. "
                "Use ONLY the numbers provided in the dataset summary — do not invent or approximate any statistics. "
                "Write in clear, accessible prose. No bullet points. "
                "Format: plain paragraphs only."
            )
        ),
    )
    return response.text.strip()


EMOTION_MAP = {
    "empathetic": (
        "warm, empathetic, and compassionate — as if you deeply care about the students "
        "behind these numbers. Use 'we' language and humanising details."
    ),
    "alarming": (
        "urgent and alarming — emphasise the scale of the crisis, use strong language "
        "to convey danger and the need for immediate action."
    ),
    "hopeful": (
        "hopeful and solution-focused — acknowledge the problem but emphasise what can "
        "be done, highlight protective factors (e.g. sleep), and end on a positive note."
    ),
    "neutral": (
        "strictly neutral and analytical — report only facts without any emotional loading. "
        "Use passive voice where natural. Avoid any value judgements."
    ),
}

def enhance_story(original_story, summary, emotion="empathetic"):
    instruction = EMOTION_MAP.get(emotion, EMOTION_MAP["empathetic"])
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=(
            f"Original story:\n{original_story}\n\n"
            f"Dataset facts for reference:\n{summary}\n\n"
            "Rewrite the story with the requested emotional tone. "
            "Keep EVERY number and statistic exactly as written."
        ),
        config=types.GenerateContentConfig(
            system_instruction=(
                f"You are an expert in emotional data storytelling. "
                f"Rewrite the story to be {instruction} "
                f"Do NOT change any number, percentage, or factual claim. "
                f"Do NOT add data points not present in the original story or dataset summary. "
                f"Output plain paragraphs only — no bullet points, no headers."
            )
        ),
    )
    return response.text.strip()

# ─────────────────────────────────────────────
#  Visualisations  (interactive Plotly – linked to story highlighting)
# ─────────────────────────────────────────────
PLOT_BG   = "#1e1e2e"
PAPER_BG  = "#0e1117"
TEXT_CLR  = "#e0e0e0"
GRID_CLR  = "#2a2a3e"

def _style_fig(fig, title, xtitle, selected):
    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT_CLR, size=13)),
        xaxis_title=xtitle,
        yaxis_title="Depression Rate (%)",
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_CLR, size=11),
        margin=dict(l=10, r=10, t=46, b=10),
        height=320,
        clickmode="event+select",
        dragmode=False,
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, color=TEXT_CLR)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_CLR, color=TEXT_CLR, zeroline=False)
    return fig

def _bar_fig(labels, values, topic, base_color, title, xtitle, selected_topic):
    """One bar chart. Every bar carries the same topic in customdata so a click
    anywhere on the chart selects that topic. Selected chart gets a bright accent."""
    is_sel = selected_topic == topic
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            customdata=[topic] * len(labels),
            marker_color=base_color,
            marker_line_width=0,
            text=[f"{v}%" for v in values],
            textposition="outside",
            textfont=dict(color=TEXT_CLR, size=11),
            hovertemplate="%{x}<br>Depression: %{y}%<extra>click to highlight</extra>",
        )
    )
    ymax = min(100, (max(values) if values else 0) + 15)
    fig.update_yaxes(range=[0, ymax])
    title_marker = "  ●" if is_sel else ""
    return _style_fig(fig, title + title_marker, xtitle, is_sel)

def make_academic_fig(stats, selected_topic=None):
    d = {k: v for k, v in sorted(stats["academic_pressure"].items())}
    return _bar_fig([str(int(k)) for k in d], list(d.values()), "academic_pressure",
                    "#6c8ebf", "Academic Pressure vs Depression",
                    "Academic Pressure Level (1=Low, 5=High)", selected_topic)

def make_financial_fig(stats, selected_topic=None):
    d = {k: v for k, v in sorted(stats["financial_stress"].items())}
    return _bar_fig([str(int(k)) for k in d], list(d.values()), "financial_stress",
                    "#ef9a9a", "Financial Stress vs Depression",
                    "Financial Stress Level (1=Low, 5=High)", selected_topic)

def make_sleep_fig(stats, selected_topic=None):
    order  = ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"]
    labels = ["<5 hrs", "5-6 hrs", "7-8 hrs", ">8 hrs"]
    vals   = [stats["sleep"].get(s, 0) for s in order]
    return _bar_fig(labels, vals, "sleep", "#66bb6a",
                    "Sleep Duration vs Depression", "Sleep Duration", selected_topic)

# ─────────────────────────────────────────────
#  Sidebar – filters & info
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Filters")
    st.caption("Adjust what the LLM story focuses on.")

    gender_options = ["All"] + sorted(df["Gender"].dropna().unique().tolist())
    gender_filter = st.selectbox("Gender", gender_options)

    sleep_options = ["All", "Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"]
    sleep_filter = st.selectbox("Sleep Duration", sleep_options)

    # ── Subgroup snapshot (change #5) ───────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 Subgroup snapshot")

    _sub = df.copy()
    _applied = []
    if gender_filter != "All":
        _sub = _sub[_sub["Gender"] == gender_filter]
        _applied.append(gender_filter)
    if sleep_filter != "All":
        _sub = _sub[_sub["Sleep Duration"] == sleep_filter]
        _applied.append(sleep_filter)

    _full_rate = STATS["depression_rate"]
    if _applied:
        _n = len(_sub)
        _rate = round(_sub["Depression"].mean() * 100, 1) if _n else 0.0
        _delta = round(_rate - _full_rate, 1)
        st.metric("Subgroup size", f"{_n:,} students")
        st.metric(
            "Subgroup depression rate",
            f"{_rate}%",
            delta=f"{_delta:+.1f}pp vs average",
            delta_color="inverse",   # higher depression = worse → shown red
        )
        _word = "higher than" if _delta > 0 else ("lower than" if _delta < 0 else "equal to")
        st.caption(
            f"**{' · '.join(_applied)}** — {abs(_delta):.1f}pp {_word} the "
            f"full-dataset average of {_full_rate}%."
        )
    else:
        st.metric("Depression rate (full dataset)", f"{_full_rate}%")
        st.caption("No filter active — showing the full dataset. "
                   "Pick a filter to compare a subgroup against this average.")

    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    st.metric("Total Students", f"{STATS['total']:,}")
    st.metric("Overall Depression Rate", f"{STATS['depression_rate']}%")
    st.markdown("---")
    st.caption(
        "**Dataset:** Student Depression Dataset  \n"
        "[Kaggle →](https://www.kaggle.com/datasets/hopesb/student-depression-dataset)  \n"
        "N = 27,901 anonymised student records"
    )
    st.markdown("---")
    st.caption(
        "**Project:** IMLD Visual Computing Team Project SS 2026  \n"
        "TU Dresden · Prof. Dachselt"
    )

# ─────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────
st.title("🧠 AI-Powered Data Storytelling")
st.caption(
    "Comparing human-written, LLM-generated, and agentic-enhanced stories "
    "about student depression — IMLD Team Project SS 2026"
)
st.markdown("---")

# ─────────────────────────────────────────────
#  Data visualisation  (interactive – click a bar to highlight stories)
# ─────────────────────────────────────────────
st.subheader("📊 Dataset Overview")
st.caption(
    "💡 **Click any bar** to highlight the sentences about that topic across all "
    "three stories below. Click again or use *Clear* to remove the highlight."
)

# Track the currently selected topic and a per-chart selection signature so we
# can detect which chart the user interacted with most recently.
for key, default in [("selected_topic", None),
                     ("sig_academic_pressure", None),
                     ("sig_financial_stress", None),
                     ("sig_sleep", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

chart_specs = [
    ("academic_pressure", make_academic_fig),
    ("financial_stress",  make_financial_fig),
    ("sleep",             make_sleep_fig),
]

cc = st.columns(3, gap="small")
events = {}
for (topic, builder), col in zip(chart_specs, cc):
    with col:
        fig = builder(STATS, st.session_state.selected_topic)
        events[topic] = st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"chart_{topic}",
            on_select="rerun",
            selection_mode="points",
        )

# Arbitrate: whichever chart's selection changed since last run wins.
def _selection_sig(ev):
    try:
        pts = ev["selection"]["points"]
    except (TypeError, KeyError):
        return None
    return tuple(sorted(p.get("point_index", p.get("x")) for p in pts)) if pts else None

newly_clicked = None
for topic in TOPIC_KEYWORDS:
    sig = _selection_sig(events.get(topic))
    if sig != st.session_state[f"sig_{topic}"]:
        st.session_state[f"sig_{topic}"] = sig
        if sig:                      # non-empty selection => this chart was clicked
            newly_clicked = topic
        elif st.session_state.selected_topic == topic:
            newly_clicked = "__clear__"

if newly_clicked == "__clear__":
    st.session_state.selected_topic = None
    st.rerun()
elif newly_clicked is not None and newly_clicked != st.session_state.selected_topic:
    st.session_state.selected_topic = newly_clicked
    st.rerun()

# Active-highlight banner + clear button
if st.session_state.selected_topic:
    t = st.session_state.selected_topic
    bcol, _ = st.columns([3, 7])
    st.markdown(
        f"<div style='background:{TOPIC_HL_COLORS[t]};color:#1a1a1a;display:inline-block;"
        f"padding:4px 12px;border-radius:6px;font-size:0.85rem;font-weight:600;'>"
        f"🔦 Highlighting: {TOPIC_LABELS[t]} — see matching sentences in the stories below"
        f"</div>",
        unsafe_allow_html=True,
    )
    if st.button("✖ Clear highlight", key="clear_highlight"):
        st.session_state.selected_topic = None
        for topic in TOPIC_KEYWORDS:
            st.session_state[f"sig_{topic}"] = None
        st.rerun()

st.markdown("---")

# Build live data summary based on sidebar filters
data_summary = build_data_summary(df, STATS, gender_filter, sleep_filter)

# Human story now responds to the active filters (change #7)
HUMAN_STORY, HUMAN_VARIANT = build_human_story(gender_filter, sleep_filter)

# ─────────────────────────────────────────────
#  Session state init
# ─────────────────────────────────────────────
for key, default in [
    ("llm_story", ""),
    ("enhanced_story", ""),
    ("show_diff", False),
    ("rating_human", None),
    ("rating_llm", None),
    ("rating_enhanced", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
#  Comparison "hero" cards  (change #3) – shown once a story exists
# ─────────────────────────────────────────────
if st.session_state.llm_story:
    st.subheader("📊 Story Comparison at a Glance")
    versions = [
        ("✍️ Human-Written", HUMAN_STORY, "#6c8ebf"),
        ("🤖 LLM-Generated", st.session_state.llm_story, "#ce93d8"),
    ]
    if st.session_state.enhanced_story:
        tone = st.session_state.get("emotion_select", "enhanced")
        versions.append((f"🎭 Enhanced · {tone}", st.session_state.enhanced_story, "#80cbc4"))

    ccols = st.columns(3, gap="medium")
    for (label, story, accent), ccol in zip(versions, ccols):
        with ccol:
            render_comparison_card(label, story, data_summary, accent)
    st.caption(
        "Readability badge: green > 60 (easy), yellow 40–60 (standard), red < 40 (hard). "
        "Sentiment badge: green positive · grey neutral · red negative."
    )
    st.markdown("---")

# ─────────────────────────────────────────────
#  Contextual guidance (change #6) – collapsed by default
# ─────────────────────────────────────────────
with st.expander("ℹ️ How to read this — what am I comparing?"):
    st.markdown(
        "- **✍️ Human-Written** — a story crafted by the team (\"Zero Battery\"). "
        "It uses narrative craft and a named protagonist. Look at how it builds empathy.\n"
        "- **🤖 LLM-Generated** — written by Gemini *strictly* from the dataset numbers. "
        "Compare its accuracy and structure against the human version.\n"
        "- **🎭 Agentic-Enhanced** — the LLM story rewritten by an agent in a chosen emotional "
        "tone while keeping every statistic. Look at *what the tone changed* (and the tone-shift gauge).\n"
        "\n"
        "**📖 Flesch readability** measures how easy the text is to read (0–100, higher = easier): "
        "90–100 ≈ very easy (5th grade) · 60–70 ≈ standard newspaper level · below 30 ≈ very hard (academic).\n\n"
        "**💬 Sentiment** is the VADER *compound* score from −1 (very negative) to +1 (very positive); "
        "near 0 is emotionally neutral. The **tone arc** below each story shows this sentence by sentence."
    )

# ─────────────────────────────────────────────
#  Story columns
# ─────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("<div class='section-label'>✍️ Human-Written Story</div>", unsafe_allow_html=True)
    st.caption(f"Showing: **{HUMAN_VARIANT}**"
               + ("" if HUMAN_VARIANT == "default story" else "  ·  adapts to the active filter"))
    st.markdown(render_story_html(HUMAN_STORY, st.session_state.selected_topic),
                unsafe_allow_html=True)
    render_metrics(HUMAN_STORY)
    render_sentiment_arc(HUMAN_STORY)

    # Fact check human story
    ok_h, extra_h = factual_check(HUMAN_STORY, data_summary + "\n" + str(STATS))
    if ok_h:
        st.markdown("<span class='fact-ok'>✅ Fact check passed</span>", unsafe_allow_html=True)
    else:
        st.markdown(
            f"<span class='fact-warn'>⚠️ Numbers to verify: {extra_h}</span>",
            unsafe_allow_html=True,
        )

    rating_human = st.feedback("stars", key="rating_human")
    if st.session_state.rating_human is not None:
        st.caption(f"You rated this {st.session_state.rating_human + 1}/5 ⭐")

with col2:
    st.markdown("<div class='section-label'>🤖 LLM-Generated Story</div>", unsafe_allow_html=True)
    if st.session_state.llm_story:
        st.markdown(
            render_story_html(st.session_state.llm_story, st.session_state.selected_topic),
            unsafe_allow_html=True,
        )
        render_metrics(st.session_state.llm_story)
        render_sentiment_arc(st.session_state.llm_story)

        ok_l, extra_l = factual_check(st.session_state.llm_story, data_summary)
        if ok_l:
            st.markdown("<span class='fact-ok'>✅ Fact check passed</span>", unsafe_allow_html=True)
        else:
            st.markdown(
                f"<span class='fact-warn'>⚠️ Numbers to verify: {extra_l}</span>",
                unsafe_allow_html=True,
            )

        rating_llm = st.feedback("stars", key="rating_llm")
        if st.session_state.rating_llm is not None:
            st.caption(f"You rated this {st.session_state.rating_llm + 1}/5 ⭐")
    else:
        st.info("👇 Click **Generate LLM Story** below to produce a Gemini-generated story.")

# ─────────────────────────────────────────────
#  Action buttons
# ─────────────────────────────────────────────
st.markdown("---")
bcol1, bcol2, bcol3 = st.columns([3, 1, 1])

with bcol1:
    if st.button("⚡ Generate LLM Story", type="primary", use_container_width=True):
        with st.spinner("Gemini is writing a story…"):
            st.session_state.llm_story    = generate_llm_story(data_summary)
            st.session_state.enhanced_story = ""
            st.session_state.show_diff     = False
        st.rerun()

with bcol2:
    if st.button("🔄 Reset All", use_container_width=True):
        st.session_state.llm_story       = ""
        st.session_state.enhanced_story  = ""
        st.session_state.show_diff       = False
        st.rerun()

# ─────────────────────────────────────────────
#  Agentic emotional enhancement
# ─────────────────────────────────────────────
if st.session_state.llm_story:
    st.markdown("---")
    st.subheader("🎭 Agentic Emotional Enhancement")
    st.caption(
        "The agentic step rewrites the LLM story with a chosen emotional tone "
        "while preserving every factual claim."
    )

    ecol1, ecol2 = st.columns([3, 1])
    with ecol1:
        emotion = st.selectbox(
            "Select emotional tone:",
            ["empathetic", "alarming", "hopeful", "neutral"],
            key="emotion_select",
        )
    with ecol2:
        st.write("")  # vertical spacer
        if st.button("✨ Enhance Story", type="primary", use_container_width=True):
            with st.spinner(f"Agent rewriting as '{emotion}'…"):
                st.session_state.enhanced_story = enhance_story(
                    st.session_state.llm_story, data_summary, emotion
                )
                st.session_state.show_diff = True
            st.rerun()

    if st.session_state.enhanced_story:
        st.markdown("---")
        ecol_a, ecol_b = st.columns(2, gap="large")

        with ecol_a:
            st.markdown("<div class='section-label'>🤖 Original LLM Story</div>", unsafe_allow_html=True)
            st.markdown(
                render_story_html(st.session_state.llm_story, st.session_state.selected_topic),
                unsafe_allow_html=True,
            )
            render_metrics(st.session_state.llm_story)
            render_sentiment_arc(st.session_state.llm_story)

        with ecol_b:
            st.markdown(
                f"<div class='section-label'>🎭 Enhanced Story ({emotion})</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                render_story_html(st.session_state.enhanced_story, st.session_state.selected_topic),
                unsafe_allow_html=True,
            )
            render_metrics(st.session_state.enhanced_story)
            render_sentiment_arc(st.session_state.enhanced_story)

            ok_e, extra_e = factual_check(st.session_state.enhanced_story, data_summary)
            if ok_e:
                st.markdown("<span class='fact-ok'>✅ Fact check passed</span>", unsafe_allow_html=True)
            else:
                st.markdown(
                    f"<span class='fact-warn'>⚠️ Numbers to verify: {extra_e}</span>",
                    unsafe_allow_html=True,
                )

            rating_enhanced = st.feedback("stars", key="rating_enhanced")
            if st.session_state.rating_enhanced is not None:
                st.caption(f"You rated this {st.session_state.rating_enhanced + 1}/5 ⭐")

        # Tone shift indicator (change #4) – directly above the diff view
        st.markdown("---")
        render_tone_shift(
            st.session_state.llm_story,
            st.session_state.enhanced_story,
            st.session_state.get("emotion_select", "enhanced"),
        )

        # Diff view
        if st.session_state.show_diff:
            with st.expander("🔍 Show word-level diff (LLM → Enhanced)"):
                diff_html = highlight_diff(
                    st.session_state.llm_story,
                    st.session_state.enhanced_story,
                )
                st.markdown(diff_html, unsafe_allow_html=True)

        if st.button("🗑️ Clear Enhancement"):
            st.session_state.enhanced_story = ""
            st.session_state.show_diff = False
            st.rerun()

# ─────────────────────────────────────────────
#  Download report
# ─────────────────────────────────────────────
st.markdown("---")
st.subheader("📥 Download Report")

if st.button("📄 Generate Report"):
    lines = [
        "AI-Powered Data Storytelling – Story Report",
        "=" * 60,
        f"Dataset: Student Depression Dataset (N={STATS['total']:,})",
        f"Filters: gender={gender_filter}, sleep={sleep_filter}",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "─" * 60,
        "HUMAN-WRITTEN STORY",
        f"Readability (Flesch): {readability_score(HUMAN_STORY)}",
        f"Sentiment: {sentiment_score(HUMAN_STORY)}",
        f"Rating: {(st.session_state.rating_human + 1) if st.session_state.rating_human is not None else 'N/A'}/5",
        "",
        HUMAN_STORY,
        "",
        "─" * 60,
        "LLM-GENERATED STORY",
    ]
    if st.session_state.llm_story:
        lines += [
            f"Readability (Flesch): {readability_score(st.session_state.llm_story)}",
            f"Sentiment: {sentiment_score(st.session_state.llm_story)}",
            f"Rating: {(st.session_state.rating_llm + 1) if st.session_state.rating_llm is not None else 'N/A'}/5",
            "",
            st.session_state.llm_story,
        ]
    else:
        lines.append("(not yet generated)")

    lines += ["", "─" * 60, "AGENTIC-ENHANCED STORY"]
    if st.session_state.enhanced_story:
        lines += [
            f"Tone: {st.session_state.get('emotion_select', 'N/A')}",
            f"Readability (Flesch): {readability_score(st.session_state.enhanced_story)}",
            f"Sentiment: {sentiment_score(st.session_state.enhanced_story)}",
            f"Rating: {(st.session_state.rating_enhanced + 1) if st.session_state.rating_enhanced is not None else 'N/A'}/5",
            "",
            st.session_state.enhanced_story,
        ]
    else:
        lines.append("(not yet generated)")

    report_text = "\n".join(lines)
    st.download_button(
        label="⬇️ Download as .txt",
        data=report_text,
        file_name="student_depression_story_report.txt",
        mime="text/plain",
    )
