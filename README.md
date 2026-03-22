# Data Science Assistant 🧠

An AI-powered data science assistant that analyzes any CSV dataset and guides you through understanding, cleaning, and modeling your data — powered by Claude Opus 4.6.

## Live Demo

**[data-scientist-ai.vercel.app](https://data-scientist-ai.vercel.app)**

Upload any CSV, optionally add a question or description, and get a full AI-powered analysis in seconds.

---

## What It Does

- **Dataset Understanding** — infers column meanings, data types, and missing values
- **Problem Framing** — identifies whether it's a classification, regression, or clustering task
- **Data Quality & Risks** — flags leakage, imbalanced classes, outliers, and multicollinearity
- **EDA Insights** — surfaces key patterns and correlations
- **Cleaning Plan** — step-by-step recommendations for preprocessing
- **Modeling Strategy** — recommends baseline and advanced models with reasoning
- **Evaluation Guidance** — suggests the right metrics for your problem
- **Code on Request** — ask for Python code and it will generate it

---

## Project Structure

```
data_science_assistant/
├── assistant.py        # CLI tool — analyze any CSV from the terminal
├── app.py              # Streamlit frontend (local)
├── web/                # Next.js web app (deployed on Vercel)
│   ├── app/
│   │   ├── page.tsx              # Main UI
│   │   └── api/analyze/route.ts  # API route (streams Claude response)
│   └── package.json
├── titanic.csv         # Sample dataset for testing
└── requirements.txt    # Python dependencies
```

---

## Running Locally

### CLI

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key
python assistant.py titanic.csv -q "predict survival"
```

### Streamlit App

```bash
python -m streamlit run app.py
```

### Web App (Next.js)

```bash
cd web
npm install
echo "ANTHROPIC_API_KEY=your_key" > .env.local
npm run dev
```

---

## Tech Stack

- **AI** — Claude Opus 4.6 (Anthropic) with adaptive thinking
- **Frontend** — Next.js 15, Tailwind CSS, React
- **CLI / Data** — Python, pandas, numpy, scikit-learn
- **Deployment** — Vercel
