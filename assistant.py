#!/usr/bin/env python3
"""
Data Science Assistant — Analyze any CSV dataset with Claude.

Usage:
    python assistant.py <csv_file> [--question "predict churn"] [--description "desc"]

Requirements:
    ANTHROPIC_API_KEY environment variable must be set.
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Load .env file if present
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import anthropic
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert AI Data Scientist and ML Engineer.

Your task is to act as an intelligent data assistant that can analyze raw datasets (especially messy, real-world CSV files) and guide the user through understanding, cleaning, and modeling the data.

The system must handle BOTH:
1. Fully raw datasets (no context)
2. Datasets with optional descriptions or metadata

---

### STEP 0: PROCESS OPTIONAL DESCRIPTION (VERY IMPORTANT)

If a dataset description is provided:
* Extract: Column meanings, Business context, Target variable hints, Constraints or assumptions
* Use this to guide ALL downstream steps

If description conflicts with data:
* Flag inconsistencies
* Ask clarification questions

If NO description is provided:
* Explicitly state assumptions while inferring meaning

---

### STEP 1: UNDERSTAND THE DATASET

* Inspect all columns and infer: Data types (numerical, categorical, datetime, text), Possible meaning of each column
* If description exists: Align inferred meanings with provided descriptions
* If column names are unclear: Infer meaning from values, Suggest better column names
* Detect: Missing values, Duplicates, Outliers, Inconsistent formats

---

### STEP 2: DETERMINE THE PROBLEM TYPE

* Use BOTH: Dataset structure, User query (if any), Description (if provided)
* Identify possible ML tasks: Classification, Regression, Time series forecasting, Clustering
* If user did not specify: Suggest 2–3 well-defined problem statements

---

### STEP 3: DATA QUALITY & RISKS

Critically analyze:
* Data leakage risks (VERY IMPORTANT)
* Imbalanced classes
* Small sample size issues
* Correlated features / multicollinearity
* Temporal leakage (if time-related data)

If description exists:
* Use it to detect subtle leakage (e.g., future info hidden in columns)

Explain WHY each issue matters.

---

### STEP 4: EDA (EXPLORATORY ANALYSIS)

Generate:
* Summary statistics
* Feature distributions
* Correlation insights
* Key patterns

If description exists:
* Relate insights to business meaning

Explain insights clearly and simply.

---

### STEP 5: DATA CLEANING PLAN

Provide a clear step-by-step plan:
* Handle missing values
* Encode categorical variables
* Normalize/scale features if needed
* Remove or cap outliers

If description exists:
* Respect domain-specific constraints

---

### STEP 6: MODELING APPROACH

Recommend:
* Suitable models (with reasoning)
* Baseline model first
* Then advanced models

Tie choices to:
* Data characteristics
* Business context (if provided)

---

### STEP 7: CODE GENERATION

**Only generate code if the user explicitly asks for it** (e.g., "give me the code", "show me the implementation", "write the code").

By default, skip this section entirely. Instead, briefly mention:
* What libraries would be used (pandas, sklearn, xgboost)
* That the user can ask for code at any time

---

### STEP 8: EVALUATION

* Suggest appropriate metrics:
  * Classification → Accuracy, F1, ROC-AUC
  * Regression → RMSE, MAE
* If description exists: Align metrics with business goals
* Explain what "good performance" means.

---

### STEP 9: FOLLOW-UP QUESTIONS

Ask smart follow-ups such as:
* "What is your target variable?"
* "Do you care more about precision or recall?"
* "Is this time-dependent data?"
* "Can you provide more context about this column?"

---

### STYLE REQUIREMENTS

* Be concise but insightful
* Avoid generic answers
* Think step-by-step like a real data scientist
* Highlight uncertainties clearly
* Do NOT hallucinate facts

---

### OUTPUT FORMAT

Structure your response clearly with sections:
1. Dataset Understanding
2. Use of Description (if provided)
3. Problem Framing
4. Risks & Issues
5. Key Insights (EDA)
6. Cleaning Plan
7. Modeling Strategy
8. Code
9. Evaluation
10. Questions for User

Your goal is to behave like a senior data scientist who can handle both raw data AND business context intelligently."""

# ---------------------------------------------------------------------------
# Data loading & profiling
# ---------------------------------------------------------------------------

def _safe_float(val):
    """Convert numpy scalar to Python float, handling NaN/inf."""
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 6)
    except (TypeError, ValueError):
        return None


def load_and_profile(filepath: str) -> tuple[pd.DataFrame, dict]:
    """Load a CSV file and return (DataFrame, profile_dict)."""
    df = pd.read_csv(filepath)

    profile: dict = {
        "filename": os.path.basename(filepath),
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": {},
    }

    for col in df.columns:
        series = df[col]
        null_count = int(series.isnull().sum())
        null_pct = round(null_count / len(df) * 100, 2) if len(df) > 0 else 0.0

        info: dict = {
            "dtype": str(series.dtype),
            "null_count": null_count,
            "null_pct": null_pct,
            "unique_count": int(series.nunique(dropna=True)),
        }

        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if len(non_null) > 0:
                info["min"] = _safe_float(non_null.min())
                info["max"] = _safe_float(non_null.max())
                info["mean"] = _safe_float(non_null.mean())
                info["median"] = _safe_float(non_null.median())
                info["std"] = _safe_float(non_null.std())
                # Simple outlier check: values beyond 3 std
                z_outliers = int(((non_null - non_null.mean()).abs() > 3 * non_null.std()).sum())
                info["outliers_3std"] = z_outliers
        else:
            top = series.value_counts().head(5)
            info["top_values"] = {str(k): int(v) for k, v in top.items()}
            info["sample_values"] = [str(v) for v in series.dropna().head(5).tolist()]

        profile["columns"][col] = info

    return df, profile


# ---------------------------------------------------------------------------
# Build user message
# ---------------------------------------------------------------------------

def build_user_message(
    df: pd.DataFrame,
    profile: dict,
    question: str | None,
    description: str | None,
) -> str:
    parts: list[str] = []

    # ---- Optional context ----
    if description:
        parts.append(f"## Dataset Description\n{description.strip()}")

    if question:
        parts.append(f"## User Question\n{question.strip()}")

    # ---- Basic info ----
    parts.append(
        f"## Dataset: {profile['filename']}\n"
        f"Shape: {profile['shape']['rows']} rows × {profile['shape']['columns']} columns\n"
        f"Duplicate rows: {profile['duplicate_rows']}"
    )

    # ---- Column summary ----
    col_lines = []
    for col, info in profile["columns"].items():
        null_tag = f"{info['null_pct']}% missing" if info["null_count"] > 0 else "no missing"
        col_lines.append(f"  - {col} ({info['dtype']}): {info['unique_count']} unique, {null_tag}")
    parts.append("## Columns Overview\n" + "\n".join(col_lines))

    # ---- Sample rows ----
    parts.append(f"## Sample Rows (first 5)\n```\n{df.head(5).to_string(index=False)}\n```")

    # ---- Descriptive statistics ----
    parts.append(f"## Descriptive Statistics\n```\n{df.describe(include='all').to_string()}\n```")

    # ---- Per-column details ----
    col_details: list[str] = []
    for col, info in profile["columns"].items():
        lines = [f"**{col}** ({info['dtype']})"]
        if "mean" in info:
            lines.append(
                f"  range=[{info['min']}, {info['max']}]  mean={info['mean']}  "
                f"std={info['std']}  outliers(3σ)={info['outliers_3std']}"
            )
        if "top_values" in info:
            lines.append(f"  top values: {info['top_values']}")
        if "sample_values" in info:
            lines.append(f"  sample: {info['sample_values']}")
        col_details.append("\n".join(lines))
    parts.append("## Detailed Column Profiles\n" + "\n\n".join(col_details))

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(filepath: str, question: str | None = None, description: str | None = None) -> None:
    print(f"\nLoading: {filepath}")

    try:
        df, profile = load_and_profile(filepath)
    except Exception as exc:
        print(f"Error loading dataset: {exc}", file=sys.stderr)
        sys.exit(1)

    rows, cols = profile["shape"]["rows"], profile["shape"]["columns"]
    print(f"Loaded: {rows} rows × {cols} columns")
    print("Sending to Claude Opus 4.6 (adaptive thinking)...\n")
    print("=" * 80)

    user_message = build_user_message(df, profile, question, description)

    client = anthropic.Anthropic()

    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)

    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Data Science Assistant — analyze any CSV with Claude Opus 4.6",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python assistant.py data.csv
  python assistant.py data.csv -q "predict customer churn"
  python assistant.py data.csv -q "predict price" -d "Airbnb listings in NYC"
        """,
    )
    parser.add_argument("csv_file", help="Path to the CSV file")
    parser.add_argument("--question", "-q", help="ML question or goal (e.g. 'predict churn')")
    parser.add_argument("--description", "-d", help="Dataset description or business context")

    args = parser.parse_args()

    if not os.path.isfile(args.csv_file):
        print(f"Error: file not found: {args.csv_file}", file=sys.stderr)
        sys.exit(1)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    analyze(args.csv_file, args.question, args.description)


if __name__ == "__main__":
    main()
