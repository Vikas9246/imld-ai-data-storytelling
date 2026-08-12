import streamlit as st
import pandas as pd
import numpy as np
import re
import html
import difflib
import io
import plotly.graph_objects as go
from google import genai
from google.genai import types
from textstat import flesch_reading_ease
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Table,
    TableStyle,
    Spacer,
    Image,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet

from reportlab.lib import colors

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI-Powered Data Storytelling – Student Depression",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  Theme-aware presentation layer
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    :root {
        --kyoto-dusk: #5B5E9D;
        --matcha-cream: #9BB29E;
        --roasted-terracotta: #DA6B51;
        --vanilla-foam: #F1DCBA;
        --charcoal-brew: #484149;
        --page-canvas: color-mix(in srgb, var(--vanilla-foam) 18%, transparent);
        --control-border: color-mix(in srgb, var(--kyoto-dusk) 46%, transparent);
        --control-hover: color-mix(in srgb, var(--kyoto-dusk) 12%, transparent);
        --control-selected: color-mix(in srgb, var(--kyoto-dusk) 24%, transparent);
    }

    .stApp {
        color: var(--text-color, inherit);
        background-color: var(--page-canvas) !important;
        background-image: none !important;
    }

    [data-testid="stAppViewContainer"],
    section[data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    .block-container {
        color: inherit;
        background-color: transparent !important;
        background-image: none !important;
    }

    .block-container {
        padding-top: 0;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    section[data-testid="stSidebar"] > div {
        background-color: color-mix(in srgb, var(--page-canvas) 92%, var(--matcha-cream) 8%);
    }

    [data-testid="stAppViewContainer"] > section,
    section[data-testid="stMain"] {
        scroll-snap-type: y mandatory;
        scroll-behavior: smooth;
    }

    .cover-section-marker,
    .compare-section-marker {
        display: none;
    }

    .workspace-section-marker {
        width: 1px;
        height: 500vh;
        position: absolute;
        display: block;
        pointer-events: none;
        scroll-snap-align: start;
        scroll-snap-stop: normal;
    }

    [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .cover-section-marker),
    [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .compare-section-marker) {
        scroll-snap-align: start;
        scroll-snap-stop: always;
        scroll-margin-top: 0.35rem;
    }

    [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .cover-section-marker),
    [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .compare-section-marker) {
        min-height: clamp(40rem, calc(100vh - 3.5rem), 64rem);
    }

    [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .cover-section-marker) {
        width: calc(100% + clamp(4rem, 10vw, 10rem));
        max-width: none;
        margin-inline: calc(clamp(2rem, 5vw, 5rem) * -1);
    }

    .cover-section {
        min-height: clamp(40rem, calc(100vh - 3.5rem), 64rem);
        display: grid;
        grid-template-columns: minmax(0, 58fr) minmax(20rem, 42fr);
        overflow: hidden;
        position: relative;
        isolation: isolate;
        border: 0;
        border-radius: 0;
        box-shadow: none;
        background: transparent;
    }

    .cover-copy-panel {
        min-width: 0;
        padding: clamp(3rem, 7vh, 5.75rem) clamp(2rem, 4.5vw, 5rem);
        display: flex;
        flex-direction: column;
        position: relative;
        z-index: 2;
        background: transparent;
    }

    .cover-geometry {
        min-width: 0;
        position: relative;
        overflow: hidden;
        background: var(--kyoto-dusk);
        pointer-events: none;
    }

    .poster-module {
        width: clamp(13rem, 22vw, 27rem);
        aspect-ratio: 1;
        position: absolute;
        right: clamp(-7rem, -4vw, -2.5rem);
        display: block;
    }

    .poster-module-one {
        top: clamp(-6rem, -7vh, -2.5rem);
        border-radius: 0 0 999px 999px;
        background: var(--roasted-terracotta);
    }

    .poster-module-two {
        top: 33%;
        right: clamp(4rem, 5vw, 7rem);
        border-radius: 999px 999px 0 0;
        background: var(--matcha-cream);
    }

    .poster-module-three {
        bottom: clamp(-7rem, -8vh, -3rem);
        border-radius: 999px 999px 0 0;
        background: var(--vanilla-foam);
    }

    .project-header {
        width: min(100%, 52rem);
        position: relative;
        z-index: 1;
    }

    [data-testid="stMarkdownContainer"] h1.project-title {
        margin: 0;
        color: currentColor;
        font-size: clamp(72px, 6.2vw, 132px);
        font-weight: 850;
        letter-spacing: -0.055em;
        line-height: 0.86;
        text-wrap: nowrap;
    }

    .project-title-line {
        display: block;
        white-space: nowrap;
    }

    .project-subtitle {
        width: min(82%, 35rem);
        margin: clamp(1.15rem, 2.5vh, 1.8rem) 0 0;
        color: color-mix(in srgb, currentColor 82%, var(--matcha-cream));
        font-size: clamp(0.95rem, 1.3vw, 1.2rem);
        line-height: 1.55;
        text-wrap: balance;
    }

    .cover-footer {
        margin-top: auto;
    }

    .cover-labels {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: clamp(0.8rem, 1.4vw, 1.35rem);
    }

    .cover-label {
        display: grid;
        grid-template-rows: auto auto;
        align-content: end;
        gap: 0.18rem;
        min-width: 0;
        padding-bottom: 0.55rem;
        border-bottom: 3px solid var(--label-accent);
        color: currentColor;
        font-size: clamp(0.68rem, 0.72vw, 0.76rem);
        font-weight: 750;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        white-space: nowrap;
    }

    .cover-label-number {
        color: var(--label-accent);
        font-weight: 850;
    }

    .cover-label-text { overflow: hidden; text-overflow: clip; }

    .cover-scroll-cue {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        position: absolute;
        right: clamp(1.5rem, 3vw, 3rem);
        bottom: clamp(1.5rem, 3.5vh, 2.5rem);
        z-index: 3;
        color: var(--vanilla-foam);
        font-size: 0.7rem;
        font-weight: 750;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        white-space: nowrap;
    }

    .cover-scroll-line {
        width: 1px;
        height: 2.7rem;
        position: relative;
        background: currentColor;
    }

    .cover-scroll-line::after {
        content: "";
        width: 0.42rem;
        height: 0.42rem;
        position: absolute;
        left: 50%;
        bottom: 0;
        border-right: 1px solid currentColor;
        border-bottom: 1px solid currentColor;
        transform: translateX(-50%) rotate(45deg);
    }

    .narrative-section {
        padding: clamp(2.5rem, 6vh, 4.5rem) 0;
    }

    .chapter-kicker {
        margin: 0 0 0.45rem;
        color: var(--kyoto-dusk);
        font-size: 0.75rem;
        font-weight: 750;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .chapter-title {
        margin: 0;
        color: var(--text-color, currentColor);
        font-size: clamp(1.85rem, 3.4vw, 3rem);
        font-weight: 750;
        letter-spacing: -0.025em;
        line-height: 1.12;
    }

    .chapter-lead {
        max-width: 52rem;
        margin: 0.7rem 0 1.75rem;
        color: color-mix(in srgb, currentColor 70%, var(--matcha-cream));
        font-size: clamp(0.95rem, 1.25vw, 1.08rem);
        line-height: 1.55;
    }

    .method-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: clamp(0.8rem, 1.6vw, 1.25rem);
    }

    .method-card,
    .measure-card {
        border: 1px solid color-mix(in srgb, var(--matcha-cream) 48%, transparent);
        border-radius: 14px;
        background: color-mix(in srgb, currentColor 4%, transparent);
    }

    .method-card {
        min-height: 13rem;
        padding: 1.15rem 1.2rem 1.25rem;
        border-top: 4px solid var(--method-accent);
    }

    .method-number {
        color: var(--method-accent);
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.1em;
    }

    .method-title,
    .measure-title {
        color: var(--text-color, currentColor);
        font-weight: 720;
    }

    .method-title {
        margin: 1rem 0 0.6rem;
        font-size: 1.1rem;
    }

    .method-copy,
    .measure-copy,
    .measure-range {
        color: color-mix(in srgb, currentColor 70%, var(--matcha-cream));
        line-height: 1.55;
    }

    .method-copy { font-size: 0.9rem; }

    .measure-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: clamp(0.8rem, 1.6vw, 1.25rem);
        margin-top: 1rem;
    }

    .measure-card {
        min-height: 12.25rem;
        padding: 1rem 1.15rem 1.05rem;
        display: grid;
        grid-template-rows: auto auto auto 1fr auto;
        border: 0;
        border-radius: 2px;
        border-top: 5px solid var(--measure-accent);
        background: color-mix(in srgb, currentColor 5%, transparent);
    }

    .measure-heading {
        display: flex;
        justify-content: space-between;
        color: color-mix(in srgb, currentColor 74%, var(--matcha-cream));
        font-size: 0.7rem;
        font-weight: 850;
        letter-spacing: 0.13em;
    }

    .measure-number { color: var(--measure-accent); }
    .measure-value {
        margin-top: 0.4rem;
        color: var(--text-color, currentColor);
        font-size: clamp(2rem, 3vw, 3.15rem);
        font-weight: 820;
        letter-spacing: -0.045em;
        line-height: 1;
    }
    .measure-name {
        margin-top: 0.15rem;
        color: var(--text-color, currentColor);
        font-size: 0.9rem;
        font-weight: 720;
    }
    .measure-copy { margin-top: 0.35rem; font-size: 0.78rem; }
    .metric-scale { margin-top: 0.7rem; }
    .scale-labels,
    .scale-values {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        align-items: end;
        font-size: 0.6rem;
        font-weight: 760;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .scale-labels span:nth-child(2), .scale-values span:nth-child(2) { text-align: center; }
    .scale-labels span:last-child, .scale-values span:last-child { text-align: right; }
    .scale-labels .negative { color: var(--roasted-terracotta); }
    .scale-labels .neutral { color: var(--kyoto-dusk); }
    .scale-labels .positive { color: var(--matcha-cream); }
    .scale-track {
        height: 4px;
        margin: 0.35rem 0 0.25rem;
        display: grid;
        grid-template-columns: 40fr 20fr 40fr;
        background: currentColor;
    }
    .scale-track span:nth-child(1) { background: var(--roasted-terracotta); }
    .scale-track span:nth-child(2) { background: var(--kyoto-dusk); }
    .scale-track span:nth-child(3) { background: var(--matcha-cream); }
    .sentiment-scale .scale-track { grid-template-columns: repeat(2, 1fr); position: relative; }
    .sentiment-scale .scale-track::before,
    .sentiment-scale .scale-track::after,
    .sentiment-scale .scale-midpoint {
        content: "";
        width: 0.55rem;
        height: 0.55rem;
        position: absolute;
        top: 50%;
        border: 2px solid var(--background-color, #fff);
        border-radius: 50%;
        transform: translate(-50%, -50%);
    }
    .sentiment-scale .scale-track::before { left: 0; background: var(--roasted-terracotta); }
    .sentiment-scale .scale-track::after { left: 100%; background: var(--matcha-cream); }
    .sentiment-scale .scale-midpoint { left: 50%; background: var(--kyoto-dusk); }

    .workspace-heading {
        padding-top: clamp(2.5rem, 6vh, 4.5rem);
    }

    .workspace-heading .chapter-lead { margin-bottom: 0.5rem; }

    .section-heading,
    .workspace-column-title {
        color: var(--text-color, currentColor);
        font-weight: 720;
        line-height: 1.2;
    }

    .section-heading {
        margin: 0.35rem 0 0.65rem;
        font-size: clamp(1.08rem, 1.6vw, 1.35rem);
    }

    .workspace-column-header {
        min-height: 5.65rem;
        display: grid;
        grid-template-rows: 3rem 2.65rem;
        margin: 0 0 0.45rem;
    }

    .workspace-column-title {
        min-height: 3rem;
        margin: 0;
        display: flex;
        align-items: center;
        padding-top: 0.5rem;
        border-top: 3px solid var(--column-accent, var(--kyoto-dusk));
        font-size: clamp(1.05rem, 1.35vw, 1.25rem);
    }

    .workspace-column-header.human { --column-accent: var(--matcha-cream); }
    .workspace-column-header.dataset { --column-accent: var(--kyoto-dusk); }
    .workspace-column-header.gemini { --column-accent: var(--kyoto-dusk); }

    .workspace-column-meta {
        min-height: 2.65rem;
        margin: 0;
        padding-top: 0.3rem;
        color: color-mix(in srgb, currentColor 68%, var(--matcha-cream));
        font-size: 0.82rem;
        line-height: 1.4;
    }

    .story-card-marker,
    .dataset-controls-marker,
    .agent-controls-marker,
    .report-controls-marker,
    .human-card-marker,
    .gemini-card-marker {
        display: none;
    }

    [data-testid="stVerticalBlockBorderWrapper"]:has(.human-card-marker),
    [data-testid="stVerticalBlockBorderWrapper"]:has(.gemini-card-marker),
    [data-testid="stVerticalBlockBorderWrapper"]:has(.story-card-marker) {
        border-radius: 14px;
        border: 1px solid color-mix(in srgb, var(--matcha-cream) 58%, transparent);
        background: color-mix(in srgb, currentColor 4%, transparent);
        box-shadow: 0 8px 22px color-mix(in srgb, #000 12%, transparent);
        transition: background-color 160ms ease, border-color 160ms ease, box-shadow 160ms ease;
    }

    [data-testid="stVerticalBlockBorderWrapper"]:has(.human-card-marker) {
        border-top: 3px solid var(--matcha-cream);
    }

    [data-testid="stVerticalBlockBorderWrapper"]:has(.gemini-card-marker) {
        border-top: 3px solid var(--kyoto-dusk);
    }

    .story-scroll {
        height: clamp(36rem, 66vh, 56rem);
        overflow-y: auto;
        padding-right: 0.55rem;
        scrollbar-gutter: stable;
    }

    .story-box {
        color: var(--text-color, currentColor);
        line-height: 1.72;
    }

    .story-box p { margin: 0 0 0.9rem; }
    .story-box p:last-child { margin-bottom: 0; }

    .story-empty {
        height: clamp(36rem, 66vh, 56rem);
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 1.25rem;
        border-radius: 10px;
        color: color-mix(in srgb, currentColor 70%, var(--matcha-cream));
        background: color-mix(in srgb, currentColor 3%, transparent);
        text-align: center;
        line-height: 1.55;
    }

    .metric-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin: 0.45rem 0 0.8rem;
        color: color-mix(in srgb, currentColor 75%, var(--matcha-cream));
        font-size: 0.82rem;
    }

    .metric-pill {
        padding: 0.2rem 0.65rem;
        border: 1px solid color-mix(in srgb, var(--matcha-cream) 46%, transparent);
        border-radius: 999px;
        background: color-mix(in srgb, currentColor 5%, transparent);
    }

    .comparison-card {
        height: 100%;
        padding: 0.9rem 1rem;
        border: 1px solid color-mix(in srgb, var(--matcha-cream) 48%, transparent);
        border-top: 3px solid var(--card-accent);
        border-radius: 12px;
        background: color-mix(in srgb, currentColor 4%, transparent);
    }

    .comparison-card-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.65rem;
        margin: 0.42rem 0;
        color: color-mix(in srgb, currentColor 72%, var(--matcha-cream));
        font-size: 0.85rem;
    }

    /* These two navigation groups are centered independently of header actions. */
    [data-testid="stElementContainer"].st-key-topic_view,
    [data-testid="stElementContainer"].st-key-agent_page {
        width: 100% !important;
        margin-block: 0.15rem 0.75rem;
    }

    .st-key-topic_view [data-testid="stButtonGroup"],
    .st-key-agent_page [data-testid="stButtonGroup"] {
        display: flex;
        flex-wrap: nowrap;
        width: min(100%, 25rem) !important;
        margin-inline: auto;
        background: transparent !important;
    }

    .st-key-topic_view [data-testid="stButtonGroup"] > div,
    .st-key-agent_page [data-testid="stButtonGroup"] > div {
        display: flex;
        width: 100% !important;
        max-width: none !important;
        flex: 1 1 100% !important;
    }

    .st-key-topic_view [data-testid="stButtonGroup"] button[data-variant="segmented_control"],
    .st-key-agent_page [data-testid="stButtonGroup"] button[data-variant="segmented_control"] {
        flex: 1 1 0;
        min-width: 6.5rem;
        padding-inline: 0.8rem;
        white-space: nowrap;
        text-overflow: clip;
        border-color: var(--control-border) !important;
        background: transparent !important;
        color: inherit !important;
    }

    .st-key-topic_view [role="radio"][aria-checked="true"],
    .st-key-agent_page [role="radio"][aria-checked="true"] {
        border-color: var(--kyoto-dusk);
        background: var(--control-selected) !important;
        color: inherit !important;
    }

    .st-key-topic_view [role="radio"]:hover,
    .st-key-agent_page [role="radio"]:hover {
        border-color: color-mix(in srgb, var(--kyoto-dusk) 68%, transparent);
        background: var(--control-hover) !important;
    }

    /* Preserve the existing full-width report selector independently. */
    .st-key-report_choice [data-testid="stButtonGroup"],
    .st-key-report_choice [data-testid="stButtonGroup"] > div {
        display: flex;
        flex-wrap: nowrap;
        width: 100% !important;
        max-width: none !important;
        flex: 1 1 100% !important;
        background: transparent !important;
    }

    .st-key-report_choice [data-testid="stButtonGroup"] button[data-variant="segmented_control"] {
        flex: 1 1 0;
        min-width: 0;
        white-space: nowrap;
        border-color: var(--control-border) !important;
        background: transparent !important;
        color: inherit !important;
    }

    .st-key-report_choice [role="radio"][aria-checked="true"] {
        border-color: var(--kyoto-dusk);
        background: var(--control-selected) !important;
        color: inherit !important;
    }

    .st-key-report_choice [role="radio"]:hover {
        border-color: color-mix(in srgb, var(--kyoto-dusk) 68%, transparent);
        background: var(--control-hover) !important;
    }

    .agentic-story-scroll {
        max-height: clamp(24rem, 48vh, 32rem);
        overflow-y: auto;
        padding-right: 0.55rem;
        scrollbar-gutter: stable;
    }

    div[data-testid="stButton"] button[kind="secondary"],
    div[data-testid="stDownloadButton"] button[kind="secondary"] {
        border-color: var(--control-border) !important;
        background: transparent !important;
        color: inherit !important;
    }

    div[data-testid="stButton"] button[kind="secondary"]:hover,
    div[data-testid="stDownloadButton"] button[kind="secondary"]:hover {
        border-color: var(--kyoto-dusk) !important;
        background: var(--control-hover) !important;
    }

    div[data-testid="stButton"] button[kind="primary"],
    div[data-testid="stDownloadButton"] button[kind="primary"] {
        border-color: var(--roasted-terracotta);
        background: var(--roasted-terracotta);
        color: #fff;
    }

    [data-testid="stSelectbox"] [role="group"],
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        border-color: var(--control-border) !important;
        background: transparent !important;
        color: inherit !important;
    }

    [data-testid="stPlotlyChart"],
    [data-testid="stPlotlyChart"] .js-plotly-plot,
    [data-testid="stPlotlyChart"] .plot-container,
    [data-testid="stPlotlyChart"] .svg-container,
    [data-testid="stPlotlyChart"] .modebar,
    [data-testid="stPlotlyChart"] .modebar-group {
        background: transparent !important;
    }

    button:focus-visible,
    [role="radiogroup"] button:focus-visible,
    a:focus-visible {
        outline: 3px solid color-mix(in srgb, var(--kyoto-dusk) 72%, white);
        outline-offset: 2px;
    }

    button:disabled {
        opacity: 0.55;
        cursor: not-allowed;
    }

    @media (max-width: 900px) {
        [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .cover-section-marker) {
            width: calc(100% + 2rem);
            margin-inline: -1rem;
        }
        .cover-section {
            min-height: clamp(40rem, calc(100vh - 3.5rem), 64rem);
            grid-template-columns: minmax(0, 64fr) minmax(12rem, 36fr);
        }
        .cover-copy-panel { padding: 3rem 1.5rem 2rem; }
        .project-header { width: 100%; }
        [data-testid="stMarkdownContainer"] h1.project-title {
            font-size: clamp(52px, 7.4vw, 72px);
        }
        .project-subtitle { width: 100%; }
        .poster-module { width: clamp(12rem, 25vw, 18rem); }
        .method-grid { grid-template-columns: 1fr; }
        .measure-grid { grid-template-columns: 1fr; }
        .method-card { min-height: auto; }
        .story-scroll, .story-empty { height: 36rem; }
    }

    @media (max-width: 640px) {
        .cover-labels { grid-template-columns: 1fr; }
        .cover-label { white-space: normal; }
    }

    @media (prefers-reduced-motion: reduce) {
        [data-testid="stAppViewContainer"] > section,
        section[data-testid="stMain"] { scroll-behavior: auto; }
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
#  Linked-visualisation: topics, keywords & colours
# ─────────────────────────────────────────────
# Each chart maps to one "topic". Clicking a bar highlights every sentence
# in every story that mentions that topic (keyword match).
TOPIC_KEYWORDS = {
    "academic_pressure": ["academic", "pressure", "study", "grade", "cgpa"],
    "financial_stress":  ["financial", "stress", "money", "rent", "fees"],
    "sleep":             ["sleep", "hours", "rest", "tired"],
}
# Light tints (chart-matched) used as the highlight background; dark text on top.
TOPIC_HL_COLORS = {
    "academic_pressure": "#E8E3C3",   # pale sand/cream
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
    """True if the sentence mentions any keyword for this topic.

    Matches on a leading word boundary so that suffixed forms (e.g. "pressures",
    "grades") still match, while substrings inside unrelated words (e.g. "fee"
    in "feels", "rent" in "parent") do not.
    """
    kws = TOPIC_KEYWORDS.get(topic, [])
    if not kws:
        return False
    pattern = r"\b(" + "|".join(re.escape(k) for k in kws) + r")"
    return re.search(pattern, sentence.lower()) is not None

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
_STORY_ARJUN = """

It's 1 AM. Arjun, a 23-year-old Computer Applications student, stares at his laptop screen — another assignment deadline looming, another month's rent overdue. He hasn't slept more than four hours in weeks. He wonders, quietly, if anyone else feels this hollow.

He's not alone. Across a dataset of nearly 28,000 students, one pattern cuts through the noise with devastating clarity: the students who face the most academic pressure are over **four times more likely** to be depressed than those who feel the least. Among students rating their academic burden at the maximum level, **{ap_high}%** report depression. Among those at the lowest level, just **{ap_low}%** do.

Financial stress tells the same story. Students struggling most financially show a depression rate of **{fs_high}%** — more than double the **{fs_low}%** seen among those with the least financial worry.

And sleep? Students sleeping fewer than five hours face a depression rate of **{fs_worst}%**. Those sleeping more than eight hours: just **{sleep_best}%**.

These aren't abstract statistics. They are Arjun's story — and the story of millions of students worldwide, caught between the weight of ambition and the fragility of mental health. The data doesn't tell us what to do. But it tells us clearly: we can't afford to keep looking away.
"""

_STORY_PRIYA = """

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
    ignore = {"100", "1000", "000", "5", "8", "28", "23", "4", "1", "2", "3", "0"}
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
        "<div style='font-size:0.72rem;color:color-mix(in srgb,currentColor 68%,#9BB29E);margin-bottom:0.6rem;'>"
        "<b style='letter-spacing:0.05em;'>EMOTIONAL TONE ARC (SENTENCE-LEVEL)</b> &nbsp;·&nbsp; "
        "negative &nbsp; neutral &nbsp; positive"
        "</div>"
    )
    st.markdown(legend + bar, unsafe_allow_html=True)

def render_metrics(text, label=""):
    r = readability_score(text)
    s = sentiment_score(text)
    sentiment_label = "positive" if s > 0.05 else ("negative" if s < -0.05 else "neutral")
    st.markdown(
        f"<div class='metric-row'>"
        f"<span class='metric-pill'>Flesch: <b>{r}</b></span>"
        f"<span class='metric-pill'>Sentiment: <b>{s}</b> ({sentiment_label})</span>"
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
    direction = ("warmer / more positive" if enh_c > llm_c
                 else ("darker / more negative" if enh_c < llm_c else "unchanged"))
    block = f"""
    <div style='margin:0.4rem 0 0.9rem 0;'>
      <div style='font-size:0.86rem;color:color-mix(in srgb,currentColor 72%,#9BB29E);margin-bottom:0.55rem;'>
        <b>Tone shift: LLM to Enhanced ({tone_name})</b>
        &nbsp; <span style='color:{line_color};font-weight:600;'>{direction}</span>
        &nbsp; <span>({llm_c:+.3f} to {enh_c:+.3f})</span>
      </div>
      <div style='position:relative;height:30px;'>
        <div style='position:absolute;top:12px;left:0;right:0;height:6px;border-radius:3px;
             background:linear-gradient(to right,#ef5350,#9e9e9e,#66bb6a);opacity:0.35;'></div>
        <div style='position:absolute;top:13px;left:{left}%;width:{right-left}%;height:4px;
             background:{line_color};border-radius:2px;'></div>
        <div title='LLM ({llm_c:+.3f})' style='position:absolute;top:6px;left:calc({p1}% - 7px);
             width:14px;height:14px;border-radius:50%;background:#5B5E9D;border:2px solid currentColor;'></div>
        <div title='Enhanced ({enh_c:+.3f})' style='position:absolute;top:6px;left:calc({p2}% - 7px);
             width:14px;height:14px;border-radius:50%;background:#DA6B51;border:2px solid currentColor;'></div>
      </div>
      <div style='display:flex;justify-content:space-between;font-size:0.7rem;color:#777;margin-top:2px;'>
        <span>−1 negative</span><span>0 neutral</span><span>+1 positive</span>
      </div>
      <div style='font-size:0.72rem;color:color-mix(in srgb,currentColor 68%,#9BB29E);margin-top:3px;'>
        LLM &nbsp; Enhanced
      </div>
    </div>
    """
    st.markdown(block, unsafe_allow_html=True)

def render_comparison_card(label, story, data_summary, accent="#fff3b0"):
    r = readability_score(story)
    s = sentiment_score(story)
    wc = len(story.split())
    ok, extra = factual_check(story, data_summary + "\n" + str(STATS))
    fact = "Pass" if ok else f"{len(extra)} flag(s)"
    fact_color = "#81c784" if ok else "#ffb74d"

    def row(lbl, value_html):
        return (f"<div class='comparison-card-row'>"
                f"<span>{lbl}</span>{value_html}</div>")

    card = (
        f"<div class='comparison-card' style='--card-accent:{accent};'>"
        f"<div style='font-weight:700;color:{accent};margin-bottom:0.5rem;font-size:0.95rem;'>{label}</div>"
        + row("Readability", _badge(r, _readability_color(r)))
        + row("Sentiment", _badge(s, _sentiment_badge_color(s)))
        + row("Word count", f"<b>{wc}</b>")
        + row("Fact-check", f"<span style='color:{fact_color};font-weight:600;'>{fact}</span>")
        + "</div>"
    )
    st.markdown(card, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Report download function
# ─────────────────────────────────────────────
@st.cache_data
def export_report_charts():
    academic_fig = make_academic_fig(STATS)
    academic_fig.write_image(
        "academic.png",
        width=420,
        height=280
    )

    financial_fig = make_financial_fig(STATS)
    financial_fig.write_image(
        "financial.png",
        width=420,
        height=280
    )

    sleep_fig = make_sleep_fig(STATS)
    sleep_fig.write_image(
        "sleep.png",
        width=420,
        height=280
    )

def build_pdf(report_choice):
    lines = [
    f"Dataset: Student Depression Dataset (N={STATS['total']:,})",
    f"Filters: gender={gender_filter}, sleep={sleep_filter}",
    f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
    ]
    info_text = "<br/>".join(lines)

    export_report_charts()

    metrics_text_human = (
        f"Readability (Flesch): {readability_score(HUMAN_STORY)}<br/>"
        f"Sentiment: {sentiment_score(HUMAN_STORY)}<br/>"
        f"Rating: {(st.session_state.rating_human + 1) if st.session_state.rating_human is not None else 'N/A'}/5<br/>"
    )    

    metrics_text_llm = (
        f"Readability (Flesch): {readability_score(st.session_state.llm_story)}<br/>"
        f"Sentiment: {sentiment_score(st.session_state.llm_story)}<br/>"
        f"Rating: {(st.session_state.rating_llm + 1) if st.session_state.rating_llm is not None else 'N/A'}/5<br/>"
    )

    metrics_text_agentic = (
        f"Tone: {st.session_state.get('enhanced_tone') or 'N/A'}<br/>"
        f"Readability (Flesch): {readability_score(st.session_state.enhanced_story)}<br/>"
        f"Sentiment: {sentiment_score(st.session_state.enhanced_story)}<br/>"
        f"Rating: {(st.session_state.rating_enhanced + 1) if st.session_state.rating_enhanced is not None else 'N/A'}/5<br/>"
    )

    if report_choice == "Human":
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer)

        elements = []

        styles = getSampleStyleSheet()
        elements.append(Paragraph("AI-Powered Data Storytelling", styles["Title"]))
        elements.append(Paragraph("Human-Written Story", styles["Heading2"]))        
        elements.append(Paragraph(info_text, styles["BodyText"]))
        elements.append(Image("academic.png"))
        elements.append(Image("financial.png"))
        elements.append(Image("sleep.png"))
        elements.append(Paragraph(HUMAN_STORY, styles["BodyText"]))
        elements.append(Paragraph(metrics_text_human, styles["BodyText"]))

        doc.build(elements)

    elif report_choice == "Gemini":
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer)

        elements = []

        styles = getSampleStyleSheet()
        elements.append(Paragraph("AI-Powered Data Storytelling", styles["Title"]))
        elements.append(Paragraph("LLM Story", styles["Heading2"]))
        elements.append(Paragraph(info_text, styles["BodyText"]))
        elements.append(Image("academic.png"))
        elements.append(Image("financial.png"))
        elements.append(Image("sleep.png"))
        elements.append(Paragraph(st.session_state.llm_story, styles["BodyText"]))
        elements.append(Paragraph(metrics_text_llm, styles["BodyText"]))

        doc.build(elements)    

    elif report_choice == "Agentic":
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer)

        elements = []

        styles = getSampleStyleSheet()
        elements.append(Paragraph("AI-Powered Data Storytelling", styles["Title"]))
        elements.append(Paragraph("Agentic Story", styles["Heading2"]))
        elements.append(Paragraph(info_text, styles["BodyText"]))
        elements.append(Image("academic.png"))
        elements.append(Image("financial.png"))
        elements.append(Image("sleep.png"))
        elements.append(Paragraph(st.session_state.enhanced_story, styles["BodyText"]))
        elements.append(Paragraph(metrics_text_agentic, styles["BodyText"]))

        doc.build(elements)

    elif report_choice == "All(Story Comparison)":
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer)

        elements = []

        styles = getSampleStyleSheet()
        elements.append(Paragraph("AI-Powered Data Storytelling", styles["Title"]))
        elements.append(Paragraph("Story Comparison", styles["Heading2"]))
        elements.append(Paragraph(info_text, styles["BodyText"]))
        elements.append(Image("academic.png"))
        elements.append(Image("financial.png"))
        elements.append(Image("sleep.png"))
        elements.append(Paragraph("Human-Written Story", styles["Heading3"]))
        elements.append(Paragraph(HUMAN_STORY, styles["BodyText"]))
        elements.append(Paragraph("LLM Story", styles["Heading3"]))
        elements.append(Paragraph(st.session_state.llm_story, styles["BodyText"]))
        elements.append(Paragraph("Agentic Story", styles["Heading3"]))
        elements.append(Paragraph(st.session_state.enhanced_story, styles["BodyText"]))

        left = Paragraph(metrics_text_human, styles["BodyText"])
        middle = Paragraph(metrics_text_llm, styles["BodyText"])
        right = Paragraph(metrics_text_agentic, styles["BodyText"])

        table = Table(
            [[left, middle, right]],
            colWidths=[170, 170, 170],
        )

        table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BOX", (0, 0), (-1, -1), 1, colors.black),
                    ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                    ("PADDING", (0, 0), (-1, -1), 10),
                ]
            )
        )

        elements.append(table)
        doc.build(elements)
    return buffer.getvalue()

# ─────────────────────────────────────────────
#  LLM functions
# ─────────────────────────────────────────────
def generate_llm_story(summary):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=(
            f"Dataset summary:\n{summary}\n\n"
            "Write a data story of about 180–220 words about student depression based strictly on the numbers above. "
            "If a subgroup filter is active in the dataset summary, make that subgroup the explicit focus of the story. "
            "Clearly mention the active gender and/or sleep-duration subgroup in the story instead of writing only about students in general. "
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
    # response.text is None when the candidate is empty or blocked by a safety filter.
    return (response.text or "").strip()


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
    # response.text is None when the candidate is empty or blocked by a safety filter.
    return (response.text or "").strip()

# ─────────────────────────────────────────────
#  Visualisations  (interactive Plotly – linked to story highlighting)
# ─────────────────────────────────────────────

def _style_fig(fig, title, xtitle, selected):
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        xaxis_title=xtitle,
        yaxis_title="Depression Rate (%)",
        font=dict(size=11),
        margin=dict(l=10, r=10, t=46, b=10),
        height=320,
        dragmode=False,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, zeroline=False)
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
            textfont=dict(size=11),
            hovertemplate="%{x}<br>Depression: %{y}%<extra></extra>",
        )
    )
    ymax = min(100, (max(values) if values else 0) + 15)
    fig.update_yaxes(range=[0, ymax])
    title_marker = " (selected)" if is_sel else ""
    return _style_fig(fig, title + title_marker, xtitle, is_sel)

def make_academic_fig(stats, selected_topic=None):
    d = {k: v for k, v in sorted(stats["academic_pressure"].items())}
    return _bar_fig([str(int(k)) for k in d], list(d.values()), "academic_pressure",
                    "#E8E3C3", "Academic Pressure vs Depression",
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

def sync_topic_from_view():
    mapping = {
        "Academic": "academic_pressure",
        "Financial": "financial_stress",
        "Sleep": "sleep",
    }

    view = st.session_state.get("topic_view")

    if view:
        st.session_state.selected_topic = mapping[view]

# ─────────────────────────────────────────────
#  Sidebar – filters & info
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Filters")
    st.caption("Adjust what the LLM story focuses on.")

    gender_options = ["All"] + sorted(df["Gender"].dropna().unique().tolist())
    gender_filter = st.selectbox("Gender", gender_options)

    sleep_options = ["All", "Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"]
    sleep_filter = st.selectbox("Sleep Duration", sleep_options)

    # ── Subgroup snapshot (change #5) ───────────────────────────────
    st.markdown("---")
    st.markdown("### Subgroup snapshot")

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
    st.markdown("### Dataset Info")
    st.metric("Total Students", f"{STATS['total']:,}")
    st.metric("Overall Depression Rate", f"{STATS['depression_rate']}%")
    st.markdown("---")
    st.caption(
        "**Dataset:** Student Depression Dataset  \n"
        "[Kaggle dataset](https://www.kaggle.com/datasets/hopesb/student-depression-dataset)  \n"
        "N = 27,901 anonymised student records"
    )
    st.markdown("---")
    st.caption(
        "**Project:** IMLD Visual Computing Team Project SS 2026  \n"
        "TU Dresden · Prof. Dachselt"
    )

@st.dialog("About this project")
def show_intro():
    st.markdown(
        """
        ### AI-Powered Data Storytelling 

        This project compares three approaches to data storytelling:

        - Human-written storytelling: crafted by the team (\"Zero Battery\")
        - LLM-generated storytelling
        - Agentic AI storytelling

        Explore how the same student-depression dataset
        can lead to different narratives and interpretations.
        """
    )

    if st.button("Explore the project", type="primary"):
        st.session_state.intro_seen = True
        st.rerun()


# # Track the currently selected topic and a per-chart selection signature so we
# # can detect which chart the user interacted with most recently.
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

def _selection_sig(ev):
    try:
        pts = ev["selection"]["points"]
    except (TypeError, KeyError):
        return None
    return tuple(sorted(p.get("point_index", p.get("x")) for p in pts)) if pts else None


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
    # Tone actually used for the current enhanced_story. Deliberately NOT the
    # selectbox's widget key: that key is dropped whenever the Setup tab stops
    # rendering, so the label would drift away from the story it describes.
    ("enhanced_tone", ""),
    ("show_diff", False),
    ("rating_human", None),
    ("rating_llm", None),
    ("rating_enhanced", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

if "intro_seen" not in st.session_state:
    st.session_state.intro_seen = False

with st.container():
    st.markdown("<div class='cover-section-marker'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <section class="cover-section">
            <div class="cover-copy-panel">
                <header class="project-header">
                    <h1 class="project-title">
                        <span class="project-title-line">AI-Powered</span>
                        <span class="project-title-line">Data Storytelling</span>
                    </h1>
                    <p class="project-subtitle">Comparing human-written, LLM-generated, and agentic narratives from student depression data</p>
                </header>
                <footer class="cover-footer">
                    <div class="cover-labels">
                        <div class="cover-label" style="--label-accent:#9BB29E;"><span class="cover-label-number">01</span><span class="cover-label-text">Human-Written</span></div>
                        <div class="cover-label" style="--label-accent:#5B5E9D;"><span class="cover-label-number">02</span><span class="cover-label-text">Gemini-Generated</span></div>
                        <div class="cover-label" style="--label-accent:#DA6B51;"><span class="cover-label-number">03</span><span class="cover-label-text">Agentic-Enhanced</span></div>
                    </div>
                </footer>
            </div>
            <div class="cover-geometry" aria-hidden="true">
                <span class="poster-module poster-module-one"></span>
                <span class="poster-module poster-module-two"></span>
                <span class="poster-module poster-module-three"></span>
            </div>
            <div class="cover-scroll-cue"><span class="cover-scroll-line" aria-hidden="true"></span>Scroll to explore</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

with st.container():
    st.markdown("<div class='compare-section-marker'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <section class="narrative-section compare-section">
            <div class="chapter-kicker">02 — How to Compare</div>
            <h2 class="chapter-title">How to Compare</h2>
            <p class="chapter-lead">Three approaches. One dataset. Different ways of turning evidence into narrative.</p>
            <div class="method-grid">
                <article class="method-card" style="--method-accent:#9BB29E;">
                    <div class="method-number">01 — HUMAN-WRITTEN</div>
                    <div class="method-title">Human-Written</div>
                    <div class="method-copy">Crafted by the team around a named protagonist. Focus on empathy and narrative structure.</div>
                </article>
                <article class="method-card" style="--method-accent:#5B5E9D;">
                    <div class="method-number">02 — GEMINI-GENERATED</div>
                    <div class="method-title">Gemini-Generated</div>
                    <div class="method-copy">Generated directly from the selected dataset. Focus on accuracy, coverage, and structure.</div>
                </article>
                <article class="method-card" style="--method-accent:#DA6B51;">
                    <div class="method-number">03 — AGENTIC-ENHANCED</div>
                    <div class="method-title">Agentic-Enhanced</div>
                    <div class="method-copy">Rewrites the Gemini story in a selected emotional tone while preserving its factual claims. Focus on what the tone changes.</div>
                </article>
            </div>
            <div class="measure-grid">
                <article class="measure-card" style="--measure-accent:#5B5E9D;">
                    <div class="measure-heading"><span>READABILITY</span><span class="measure-number">01</span></div>
                    <div class="measure-value">0–100</div>
                    <div class="measure-name">Flesch Reading Ease</div>
                    <div class="measure-copy">Higher scores mean easier reading.</div>
                    <div class="metric-scale readability-scale">
                        <div class="scale-labels"><span class="negative">Difficult</span><span class="neutral">Standard</span><span class="positive">Easier</span></div>
                        <div class="scale-track"><span></span><span></span><span></span></div>
                        <div class="scale-values"><span>0</span><span>40&nbsp;&nbsp;&nbsp;60</span><span>100</span></div>
                    </div>
                </article>
                <article class="measure-card" style="--measure-accent:#DA6B51;">
                    <div class="measure-heading"><span>SENTIMENT</span><span class="measure-number">02</span></div>
                    <div class="measure-value">−1&nbsp;&nbsp;&nbsp;0&nbsp;&nbsp;&nbsp;+1</div>
                    <div class="measure-name">VADER Compound Score</div>
                    <div class="measure-copy">Shows the emotional direction of each story.</div>
                    <div class="metric-scale sentiment-scale">
                        <div class="scale-labels"><span class="negative">Negative</span><span class="neutral">Neutral</span><span class="positive">Positive</span></div>
                        <div class="scale-track"><span></span><span></span><i class="scale-midpoint"></i></div>
                        <div class="scale-values"><span>−1</span><span>0</span><span>+1</span></div>
                    </div>
                </article>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

with st.container():
    st.markdown("<div class='workspace-section-marker'></div>", unsafe_allow_html=True)
    workspace_title_col, workspace_action_col = st.columns(
        [7, 1.25],
        vertical_alignment="bottom",
    )
    with workspace_title_col:
        st.markdown(
            """
            <section class="workspace-heading">
                <div class="chapter-kicker">03 — Explore and Compare</div>
                <h2 class="chapter-title">Explore and Compare</h2>
                <p class="chapter-lead">Select a subgroup, inspect the data, and compare how each approach constructs its story.</p>
            </section>
            """,
            unsafe_allow_html=True,
        )

    with workspace_action_col:
        if st.button("Reset All", use_container_width=True):
            st.session_state.llm_story       = ""
            st.session_state.enhanced_story  = ""
            st.session_state.enhanced_tone   = ""
            st.session_state.show_diff       = False
            st.rerun()


# Symmetric story comparison with a wider analytical centre.
left_col, center_col, right_col = st.columns([1.15, 1.7, 1.15], gap="large")

with left_col:
    human_variant_note = (
        HUMAN_VARIANT
        if HUMAN_VARIANT == "default story"
        else f"{HUMAN_VARIANT} · adapts to the active filter"
    )
    st.markdown(
        f"""
        <div class="workspace-column-header human">
            <div class="workspace-column-title">Human-Written Story</div>
            <div class="workspace-column-meta">Showing: <b>{human_variant_note}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    with st.container(border=True):
        st.markdown(
            "<div class='human-card-marker'></div>",
            unsafe_allow_html=True
        )
        human_story_html = render_story_html(
            HUMAN_STORY,
            st.session_state.selected_topic,
        )
        st.markdown(
            f"<div class='story-scroll'>{human_story_html}</div>",
            unsafe_allow_html=True,
        )

    rating_human = st.feedback(
    "stars",
    key="rating_human"
    )

    if st.session_state.rating_human is not None:
        st.caption(
            f"You rated this {st.session_state.rating_human + 1}/5"
        )

with center_col:
    st.markdown(
        """
        <div class="workspace-column-header dataset">
            <div class="workspace-column-title">Dataset Overview</div>
            <div class="workspace-column-meta" aria-hidden="true">&nbsp;</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown("<div class='dataset-controls-marker'></div>", unsafe_allow_html=True)
        topic_view = st.segmented_control(
            "Dataset view",
            ["Academic", "Financial", "Sleep"],
            default=None,
            key="topic_view",
            on_change=sync_topic_from_view,
            label_visibility="collapsed",
        )

    if st.session_state.selected_topic:
        clear_left, clear_center, clear_right = st.columns([1.2, 1, 1.2])
        with clear_center:
            if st.button("Clear Highlight", key="clear_highlight", use_container_width=True):
                st.session_state.selected_topic = None
                st.rerun()

    view_for_chart = topic_view or "Academic"

    if view_for_chart == "Academic":
        current_topic = "academic_pressure"
        fig = make_academic_fig(
            STATS,
            st.session_state.selected_topic
        )

    elif view_for_chart == "Financial":
        current_topic = "financial_stress"
        fig = make_financial_fig(
            STATS,
            st.session_state.selected_topic
        )

    else:
        current_topic = "sleep"
        fig = make_sleep_fig(
            STATS,
            st.session_state.selected_topic
        )


    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"chart_{current_topic}",
        theme="streamlit",
    )

    # ─────────────────────────────────────────────
    #  Agentic emotional enhancement
    # ─────────────────────────────────────────────
    agent_title_col, agent_action_col = st.columns(
        [4, 1.55],
        gap="small",
        vertical_alignment="top",
    )
    with agent_title_col:
        st.markdown(
            "<div class='section-heading'>Agentic Emotional Enhancement</div>",
            unsafe_allow_html=True,
        )
    with agent_action_col:
        if st.session_state.enhanced_story:
            if st.button("Clear Enhancement", use_container_width=True):
                st.session_state.enhanced_story = ""
                st.session_state.enhanced_tone = ""
                st.session_state.show_diff = False
                st.rerun()

    with st.container():
        st.markdown("<div class='agent-controls-marker'></div>", unsafe_allow_html=True)
        agent_page = st.segmented_control(
            "Agentic page",
            ["Setup", "Story", "Analysis"],
            default="Setup",
            key="agent_page",
            label_visibility="collapsed",
        )

    if agent_page == "Setup":
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
            if st.session_state.llm_story:
                if st.button("Enhance Story", type="primary", use_container_width=True):
                    # See note on the Generate handler: st.rerun() signals via
                    # RerunException (a BaseException), so `except Exception` is safe.
                    try:
                        with st.spinner(f"Agent rewriting as '{emotion}'…"):
                            text = enhance_story(
                                st.session_state.llm_story, data_summary, emotion
                            )
                        if not text:
                            st.error("The agent returned an empty response (this can "
                                     "happen when a safety filter blocks the output). "
                                     "Please try again.")
                        else:
                            st.session_state.enhanced_story = text
                            # Pin the tone that produced this text, not whatever the
                            # selectbox happens to show later.
                            st.session_state.enhanced_tone = emotion
                            st.session_state.show_diff = True
                            st.rerun()
                    except Exception as e:
                        st.error(f"Tone rewrite failed: {type(e).__name__}. "
                                 f"Check the API key, quota, and network connection.")
    elif agent_page == "Story":
        if st.session_state.enhanced_story:
            with st.container(border=True):
                st.markdown(
                    "<div class='story-card-marker'></div>",
                    unsafe_allow_html=True
                )
                strory_html = render_story_html(
                    st.session_state.enhanced_story,
                    st.session_state.selected_topic
                )
                st.markdown(
                    f"""<div class="agentic-story-scroll">
                    {strory_html}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            rating_enhanced = st.feedback("stars", key="rating_enhanced")
            if st.session_state.rating_enhanced is not None:
                st.caption(f"You rated this {st.session_state.rating_enhanced + 1}/5")




    elif agent_page == "Analysis":
        if st.session_state.enhanced_story:

            # Tone shift indicator (change #4) – directly above the diff view
            with st.container(height=350):
                render_metrics(st.session_state.enhanced_story)
                render_sentiment_arc(st.session_state.enhanced_story)          
                    
                render_tone_shift(
                    st.session_state.llm_story,
                    st.session_state.enhanced_story,
                    st.session_state.get("enhanced_tone") or "enhanced",
                )

                # Diff view
                if st.session_state.show_diff:
                        diff_html = highlight_diff(
                            st.session_state.llm_story,
                            st.session_state.enhanced_story,
                        )

                        st.markdown("#### Word-level diff")
                        st.markdown(diff_html, unsafe_allow_html=True)                

with right_col:
    st.markdown(
        """
        <div class="workspace-column-header gemini">
            <div class="workspace-column-title">Gemini-Generated Story</div>
            <div class="workspace-column-meta">Generated from the active filters</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        st.markdown(
            "<div class='gemini-card-marker'></div>",
            unsafe_allow_html=True
        )

        if st.session_state.llm_story:
                llm_story_html = render_story_html(
                    st.session_state.llm_story,
                    st.session_state.selected_topic
                )

                st.markdown(
                    f"<div class='story-scroll'>{llm_story_html}</div>",
                    unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='story-empty'>Generate a Gemini story to compare it with the human-written narrative.</div>",
                unsafe_allow_html=True,
            )

    if st.session_state.llm_story:

        rating_llm = st.feedback(
        "stars",
        key="rating_llm"
        )

        if st.session_state.rating_llm is not None:
            st.caption(
                f"You rated this {st.session_state.rating_llm + 1}/5"
            )

    if st.button("Generate Gemini Story", type="primary", use_container_width=True):
        # st.rerun() signals via RerunException, which subclasses BaseException and so
        # is not caught by `except Exception` — safe to call inside the try.
        try:
            with st.spinner("Gemini is writing a story…"):
                text = generate_llm_story(data_summary)
            if not text:
                st.error("Gemini returned an empty response (this can happen when "
                         "a safety filter blocks the output). Please try again.")
            else:
                st.session_state.llm_story = text
                st.session_state.enhanced_story = ""
                st.session_state.enhanced_tone = ""
                st.session_state.show_diff = False
                st.rerun()
        except Exception as e:
            st.error(f"Story generation failed: {type(e).__name__}. "
                     f"Check the API key, quota, and network connection.")


# ─────────────────────────────────────────────
#  Comparison "hero" cards  (change #3) – shown once a story exists
# ─────────────────────────────────────────────
if st.session_state.llm_story:
    st.subheader("Story Comparison at a Glance")

    versions = [
        ("Human-Written", HUMAN_STORY, "#9BB29E"),
        ("Gemini-Generated", st.session_state.llm_story, "#5B5E9D"),
    ]
    if st.session_state.enhanced_story:
        tone = st.session_state.get("enhanced_tone") or "enhanced"
        versions.append((f"Agentic Enhanced · {tone}", st.session_state.enhanced_story, "#DA6B51"))

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
#  Download report
# ─────────────────────────────────────────────
if st.session_state.llm_story:
    st.subheader("Download Report")

    if st.session_state.get("report_choice") == "All(Story Comparison)":
        st.session_state.report_choice = "Full Comparison"

    report_choice_col, report_action_col = st.columns(
        [5.5, 1.5],
        vertical_alignment="center",
    )

    with report_choice_col:
        st.markdown("<div class='report-controls-marker'></div>", unsafe_allow_html=True)
        report_choice = st.segmented_control(
            "Include in report",
            ["Full Comparison", "Human", "Gemini", "Agentic"],
            default=None,
            key="report_choice",
            label_visibility="collapsed",
        )

    with report_action_col:
        if report_choice is not None:
            pdf_report_choice = (
                "All(Story Comparison)"
                if report_choice == "Full Comparison"
                else report_choice
            )
            # Chart export needs kaleido, and kaleido 1.x is incompatible with
            # plotly < 6.1. Degrade to a notice rather than a traceback so the app
            # stays usable in an environment where that pairing is wrong.
            try:
                pdf_data = build_pdf(pdf_report_choice)
            except Exception:
                pdf_data = None
                st.warning(
                    "Report export is unavailable in this environment (chart image "
                    "export requires kaleido). The on-screen comparison above shows "
                    "the same metrics."
                )

            if pdf_data is not None:
                st.download_button(
                    "Download PDF",
                    data=pdf_data,
                    file_name="story_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

st.markdown("---")
about_col, _ = st.columns([1, 7])
with about_col:
    if st.button("About", use_container_width=True):
        show_intro()
