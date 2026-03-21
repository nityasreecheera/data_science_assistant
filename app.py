"""
Data Science Assistant — Streamlit frontend.
Run with: streamlit run app.py
"""

import os
import tempfile
from pathlib import Path

import streamlit as st

# Load .env
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# Import after env is loaded
import anthropic
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from assistant import load_and_profile, build_user_message, SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Data Science Assistant",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Data Science Assistant")
st.caption("Upload a CSV, ask a question, and get a full AI-powered data science analysis.")

# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Your key is never stored beyond this session.",
    )

    st.divider()
    st.header("Dataset")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    question = st.text_area(
        "Your Question (optional)",
        placeholder="e.g. predict customer churn",
        height=80,
    )

    description = st.text_area(
        "Dataset Description (optional)",
        placeholder="e.g. E-commerce orders dataset. target column is 'churn'.",
        height=120,
    )

    run_btn = st.button("▶ Run Analysis", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

if not uploaded_file and not run_btn:
    st.info("Upload a CSV file in the sidebar to get started.")
    st.stop()

if run_btn:
    # Validate inputs
    if not uploaded_file:
        st.error("Please upload a CSV file.")
        st.stop()

    if not api_key:
        st.error("Please enter your Anthropic API key in the sidebar.")
        st.stop()

    os.environ["ANTHROPIC_API_KEY"] = api_key

    # ---- Load & profile data ----
    with st.spinner("Loading dataset..."):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            df, profile = load_and_profile(tmp_path)
            profile["filename"] = uploaded_file.name
        except Exception as e:
            st.error(f"Failed to load CSV: {e}")
            st.stop()
        finally:
            os.unlink(tmp_path)

    rows, cols = profile["shape"]["rows"], profile["shape"]["columns"]

    # ---- Data preview ----
    st.subheader("📋 Data Preview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{rows:,}")
    col2.metric("Columns", cols)
    col3.metric("Duplicate Rows", profile["duplicate_rows"])

    st.dataframe(df.head(10), use_container_width=True)

    # Missing values summary
    missing = {k: v for k, v in profile["columns"].items() if v["null_count"] > 0}
    if missing:
        with st.expander(f"⚠️ Missing Values ({len(missing)} columns)"):
            miss_df = pd.DataFrame([
                {"Column": k, "Missing": v["null_count"], "Missing %": v["null_pct"]}
                for k, v in missing.items()
            ])
            st.dataframe(miss_df, use_container_width=True)

    st.divider()

    # ---- Stream analysis ----
    st.subheader("🤖 AI Analysis")

    user_message = build_user_message(df, profile, question or None, description or None)
    client = anthropic.Anthropic(api_key=api_key)

    output_area = st.empty()
    full_text = ""

    try:
        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            for chunk in stream.text_stream:
                full_text += chunk
                output_area.markdown(full_text)

        st.success("Analysis complete!")

        # Download button
        st.download_button(
            label="⬇️ Download Analysis",
            data=full_text,
            file_name=f"{uploaded_file.name}_analysis.md",
            mime="text/markdown",
        )

    except anthropic.AuthenticationError:
        st.error("Invalid API key. Please check your key in the sidebar.")
    except anthropic.BadRequestError as e:
        st.error(f"API error: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

elif uploaded_file:
    # Show preview while user hasn't clicked Run yet
    try:
        df_preview = pd.read_csv(uploaded_file)
        st.subheader("📋 Data Preview")
        rows, cols = df_preview.shape
        c1, c2 = st.columns(2)
        c1.metric("Rows", f"{rows:,}")
        c2.metric("Columns", cols)
        st.dataframe(df_preview.head(10), use_container_width=True)
        st.info("Fill in the sidebar and click **▶ Run Analysis** to start.")
    except Exception as e:
        st.error(f"Could not preview file: {e}")
