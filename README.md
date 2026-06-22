# AI-Powered Data Storytelling

**IMLD Visual Computing Team Project – Summer Semester 2026**  
Technische Universität Dresden | CMS Master's Programme  
Supervisors: Prof. Dr.-Ing. Raimund Dachselt · Susmita Khadse M.Sc. · Julián Méndez M.Sc.  
Course page: <https://imld.de/en/study/teaching/ss_26/cms-tea_26/>

---

## 👥 Team

| Name | Matriculation | Role |
|---|---|---|
| Vikash Yadav | 5317730 | Developer / Coordinator |
| Tongtong Li | 5191050 | Literature Review / Report |
| Xueyi Bai | 5317603 | Interface Design / Slides |
| Xintong Yang | 5330580 | Data Analysis / Human Story |

---

## 📖 Project Overview

This repository contains a browser-based prototype that compares three versions of a data story generated from the same dataset:

1. **Human-written story** – *"Zero Battery"*, authored by the team after analysing the Student Depression Dataset using a persona-based approach.
2. **LLM-generated story** – produced by Google Gemini 2.5 Flash from a structured dataset summary.
3. **Agentic-enhanced story** – the LLM story is emotionally refined by an agentic workflow (empathetic / alarming / hopeful / neutral tone) while preserving every factual claim.

The interface also evaluates all three versions automatically using readability, sentiment, and factual-consistency metrics.

---

## 🗂️ Dataset

**Student Depression Dataset** — [Kaggle](https://www.kaggle.com/datasets/hopesb/student-depression-dataset)

- **N = 27,901** anonymised student records
- Key variables: `Academic Pressure` (1–5), `Financial Stress` (1–5), `Sleep Duration` (categorical), `Gender`, `CGPA`, `City`, `Dietary Habits`, `Depression` (binary 0/1)
- Three primary findings from our data analysis:
  - Academic pressure: depression rate rises from **19.4%** (level 1) → **86.1%** (level 5)
  - Financial stress: depression rate rises from **31.9%** (level 1) → **81.3%** (level 5)
  - Sleep: students sleeping fewer than 5 hours show **64.5%** depression rate; 8+ hours drops to **28.5%**

Download the CSV from Kaggle and save it as `data/data.csv` before running.

---

## ✨ Features

### Data Visualisation
- Three interactive **Plotly bar charts** (academic pressure, financial stress, sleep duration vs depression rate)
- Click any bar to **highlight matching sentences** across all three story columns simultaneously

### Story Comparison
- Side-by-side display of human-written, LLM-generated, and agentic-enhanced stories
- **Sentence-level sentiment arc** — coloured segments (red / grey / green) showing emotional structure per sentence
- **Word-level diff view** — highlights exactly what the agentic step changed

### Agentic Tone Enhancement
- Dropdown selector: `empathetic` / `alarming` / `hopeful` / `neutral`
- Agent rewrites the LLM story to the chosen tone while preserving every number and statistic
- **Tone-shift gauge** showing VADER compound score before and after enhancement

### Evaluation Metrics
- **Flesch Reading Ease** — readability score (0–100, higher = easier)
- **VADER Compound Sentiment** — from −1 (most negative) to +1 (most positive)
- **Regex fact-check** — flags numbers in the story not present in the dataset summary
- **Comparison card** — readability, sentiment, word count, and fact-check status across all versions
- **Star ratings** (1–5) per story version for optional user study collection

### Filters & Personalisation
- Sidebar filters: **Gender** and **Sleep Duration**
- Subgroup snapshot: depression rate delta vs full dataset shown with `st.metric`
- Human story adapts to filters: female variant (Priya) and sleep-deprivation variant

### Download
- **Report export** (.txt) with all story versions, metrics, filter state, and ratings

---

## 🚀 Setup

### Prerequisites

- Python (tested with Anaconda base, Python 3.11+)
- A Google Gemini API key ([get one free](https://aistudio.google.com/app/apikey))
- The Student Depression Dataset CSV saved at `data/data.csv`

### Installation

```bash
git clone https://github.com/Vikas9246/imld-ai-data-storytelling.git
cd imld-ai-data-storytelling
pip install -r requirements.txt
```

### API Key

Create `.streamlit/secrets.toml`:

```toml
GOOGLE_API_KEY = "your-key-here"
```

This file is in `.gitignore` and is never committed.

### Run

```bash
# Standard Python
streamlit run app.py

# Anaconda
C:\Users\vikash\anaconda3\python.exe -m streamlit run app.py
```

Open <http://localhost:8501> in your browser.

---

## 📁 Repository Structure

```
imld-ai-data-storytelling/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── data/
│   └── data.csv              # Student Depression Dataset (not committed — download from Kaggle)
├── .streamlit/
│   └── secrets.toml          # API key (not committed)
└── .gitignore
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend / App | Streamlit |
| LLM | Google Gemini 2.5 Flash (`google-genai` SDK) |
| Visualisation | Plotly |
| Sentiment Analysis | VADER (`vaderSentiment`) |
| Readability | Textstat (Flesch Reading Ease) |
| Data Processing | Pandas, NumPy |

---

## 📚 Project Context

This project is part of the IMLD Visual Computing Team Project (SS 2026) at TU Dresden. The research question we investigate:

> *How does emotional tone in AI-generated data stories affect reader trust, engagement, and comprehension — compared to human-authored stories on the same data?*

The project is grounded in the literature on data storytelling (Segel & Heer, 2010), agentic AI narrative generation (Islam et al., 2024; Shen et al., 2024), and the emotional impact of AI authorship on readers (Shen et al., 2024; Zhao et al., 2025).

The scope was confirmed with supervisor Susmita Khadse in May 2026: the agentic step refines the LLM story in place rather than generating a separate third story from scratch, which reduces computational cost while preserving the comparative structure.

---

## 📅 Timeline

| Milestone | Date | Status |
|---|---|---|
| Interim report | June 21, 2026 | ✅ Done |
| Interim presentation | June 24, 2026 | ✅ Done |
| Interface polish + HCI design | July 2026 | 🔄 In progress |
| User study (if time permits) | July–August 2026 | ⏳ Planned |
| Final presentation | August 10, 2026 | ⏳ Planned |
| Final submission | August 16, 2026 | ⏳ Planned |
