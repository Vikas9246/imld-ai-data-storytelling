# AI-Powered Data Storytelling

**IMLD Visual Computing Team Project – Summer Semester 2026**  
Technische Universität Dresden | CMS Master 
Supervisors: Prof. Dr.-Ing. Raimund Dachselt, Susmita Khadse M.Sc., Julián Méndez M.Sc.
Course page: [https://imld.de/en/study/teaching/ss_26/cms-tea_26/](https://imld.de/en/study/teaching/ss_26/cms-tea_26/)


## 👥 Team

| Name | Matriculation | Role |
|------|---------------|------|
| Vikash Yadav | 5317730  | Developer |
| Tongtong Li | 5191050  | ... |
| Xueyi Bai | 5317603 | ... |
| Xintong Yang | 5330580 | ... |


## 📖 Project Overview

This repository contains a browser‑based prototype that compares three versions of a data story generated from the same public‑health dataset:

1. **Human‑written story** – composed by the team after analysing the data.  
2. **LLM‑generated story** – produced by a general Large Language Model (Google Gemini 2.5 Flash).  
3. **Agentic‑enhanced story** – the LLM story is emotionally moderated by an agentic workflow (dropdown selection for tone: empathetic, alarming, hopeful, neutral) while preserving factual accuracy.

The interface displays visualisations, side‑by‑side story comparison, and automatic evaluation metrics (readability, sentiment, factual consistency).  
Optionally, a small user study can be conducted to validate the results.


## ✨ Features

- 📊 Interactive data visualisation (line chart)  
- ✍️ Pre‑written human data story  
- 🤖 LLM‑generated story with one click  
- 🎭 Agentic emotional tone enhancement via dropdown  
- 🔍 Automatic evaluation metrics per story version  
- 🔐 API key kept secure using Streamlit secrets  
- 📦 Clean, modular codebase ready for collaboration


## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Vikas9246/imld-ai-data-storytelling.git
cd imld-ai-data-storytelling
```
### 2. Install dependencies 
-Make sure you have Python 3.10+ installed. Then:

```bash
pip install -r requirements.txt
```

### 3. Set up the API key
- Create a `.streamlit/secrets.toml` file **inside the project folder**.
- Add your Google Gemini API key (get one for free at [Google AI Studio](https://aistudio.google.com/)):

```toml
GOOGLE_API_KEY = "your-api-key-here"
```

> ⚠️ Never commit this file! The `.gitignore` already excludes it.

### 4. Run the app
```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`.


## 📊 Dataset

**WHO Road Traffic Deaths per 100,000 population** (selected countries, 2019–2021).

| Country | 2019 | 2020 | 2021 |
|---------|------|------|------|
| India   | 15.6 | 13.7 | 14.2 |
| Germany | 3.7  | 3.2  | 2.9  |
| Brazil  | 19.7 | 17.1 | 18.3 |

*Source: World Health Organization – Global Health Observatory (or Kaggle public subset).*


## 🧪 Evaluation

The app supports two evaluation modes:

1. **Automatic Metrics** (implemented)  
   - **Readability:** Flesch Reading Ease  
   - **Sentiment:** VADER compound polarity  
   - **Factual Consistency:** Regex‑based extraction of numbers vs. dataset  

2. **User Study** (optional, time‑permitting)  
   - Questionnaire with Likert scales (readability, trust, emotional engagement, comprehension)  
   - Results analysed via descriptive statistics / t‑tests.


## 📋 Deliverables Checklist

- [x] Task sheet (who did what) – maintained throughout  
- [x] AI tools analysis report – literature review of LLMs, storytelling platforms, agentic AI  
- [x] Git repository (this repo)  
- [ ] Final report with evaluation (due August 16)  
- [ ] Project presentation slides (interim: June 22, final: August 10)


## 🛠️ Technology Stack

- **Frontend / Interface:** Streamlit  
- **LLM:** Google Gemini 2.5 Flash (via `google-genai` SDK)  
- **Data Handling:** Pandas, Matplotlib  
- **Evaluation:** Textstat, VADER Sentiment, regex  
- **Version Control:** Git & GitHub


## 📝 License

This project is created for academic purposes at TU Dresden. All rights reserved unless otherwise noted.


## 📧 Contact

For questions about this repository, contact the team via the course supervisors or open an issue on GitHub.
```