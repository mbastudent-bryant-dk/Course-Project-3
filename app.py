"""
ESG Startup Classifier + Trading Strategy — Streamlit App
=========================================================
Two-tab interactive web app:

1. ESG Classifier — wraps the Project 1 Gemini-based classification notebook.
2. ESG Trading Strategy — Moving Average crossover backtest comparing an
   ESG-aligned ETF (default ESGU) against a broad-market benchmark (default
   SPY). Satisfies the course-required "Moving Average Trading Strategy" task
   while keeping the ESG theme.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
import os
import time
from datetime import date, timedelta
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="ESG Startup Classifier",
    page_icon="LEAF",
    layout="wide",
)

DEFAULT_CSV_URL = (
    "https://raw.githubusercontent.com/hkuangedu/PythonProgramming/"
    "main/Project1_LLM_ESG_Classification.csv"
)
MODEL_NAME = "gemini-2.5-flash"

# ----------------------------------------------------------------------------
# Prompt templates (strictness modes)
# ----------------------------------------------------------------------------
BASE_RULES = """You are a strict ESG classification analyst. Classify a startup as
ESG (label = 1) only when its CORE product or service directly relates to
Environmental, Social, or Governance themes.

Categories:
- Environmental: direct reduction of pollution, resource conservation, clean energy,
  energy efficiency, sustainable alternatives to harmful products, waste/water
  infrastructure, ecosystem protection.
- Social: direct expansion of essential services to underserved populations,
  reduction of systemic inequality, work addressing poverty, housing, education
  gaps, or access barriers.
- Governance: corporate transparency, anti-corruption, ethical compliance,
  accountability, auditing, governance-risk controls.

Generic SaaS, analytics, marketplaces, regular medical devices, wellness products,
civic engagement tools, workplace tools, and charitable giving platforms should NOT
count unless the core business specifically targets an ESG problem.
"""

STRICTNESS_INSTRUCTIONS = {
    "Conservative": (
        "Be extra strict. If there is any doubt, label as 0 (non-ESG). "
        "Only classify as ESG when the description explicitly and unambiguously "
        "shows direct environmental, social, or governance impact."
    ),
    "Balanced": (
        "Default to non-ESG when uncertain, but accept clear cases such as "
        "plant-based foods, clean energy products, or accessibility services "
        "for underserved populations as ESG."
    ),
    "Inclusive": (
        "Be more permissive. If the core business plausibly delivers an "
        "environmental, social, or governance benefit, classify as ESG even "
        "if the language is somewhat indirect."
    ),
}

OUTPUT_FORMAT_INSTRUCTION = """Return ONLY a valid JSON array. Each element must follow:
{
  "firm_id": "the firm_id provided",
  "label": 0 or 1,
  "confidence": 0.0 to 1.0,
  "category": "Environmental" | "Social" | "Governance" | "Mixed" | "Non-ESG",
  "explanation": "one clear sentence reason"
}
Do not include markdown, code fences, or any text outside the JSON array."""


def build_system_prompt(strictness: str) -> str:
    return f"{BASE_RULES}\n\nStrictness mode: {strictness}.\n{STRICTNESS_INSTRUCTIONS[strictness]}\n\n{OUTPUT_FORMAT_INSTRUCTION}"


# ----------------------------------------------------------------------------
# Data loading helpers
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_default_csv(url: str = DEFAULT_CSV_URL) -> pd.DataFrame:
    """Loads the Project 1 reference CSV from GitHub."""
    df = pd.read_csv(url, header=None)
    # The original file has no header; first row is actually the header.
    if df.iloc[0].astype(str).str.contains("firm", case=False).any():
        df.columns = df.iloc[0].astype(str).str.strip().str.lower()
        df = df.iloc[1:].reset_index(drop=True)
    else:
        cols = ["firm_id", "bus_description"]
        if df.shape[1] >= 3:
            cols.append("esg_dummy")
        df.columns = cols + list(df.columns[len(cols):])
    df = _normalize_columns(df)
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerces column names to lower snake_case and casts esg_dummy to int."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    rename_map = {
        "id": "firm_id",
        "firmid": "firm_id",
        "description": "bus_description",
        "business_description": "bus_description",
        "bus_desc": "bus_description",
        "esg": "esg_dummy",
        "label": "esg_dummy",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if "esg_dummy" in df.columns:
        df["esg_dummy"] = pd.to_numeric(df["esg_dummy"], errors="coerce")
    if "firm_id" in df.columns:
        df["firm_id"] = df["firm_id"].astype(str)
    return df


def validate_uploaded_csv(df: pd.DataFrame) -> Tuple[bool, str]:
    df = _normalize_columns(df)
    missing = [c for c in ["firm_id", "bus_description"] if c not in df.columns]
    if missing:
        return False, f"Missing required column(s): {', '.join(missing)}"
    return True, "OK"


# ----------------------------------------------------------------------------
# Gemini client
# ----------------------------------------------------------------------------
def get_api_key() -> str | None:
    """Reads the API key from environment, Streamlit secrets, or session state."""
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        try:
            key = st.secrets.get("GOOGLE_API_KEY") or st.secrets.get("GEMINI_API_KEY")
        except Exception:
            key = None
    if not key:
        key = st.session_state.get("api_key_input")
    return key or None


def get_genai_client(api_key: str):
    """Lazy-imports google-genai to avoid import errors when not installed."""
    from google import genai  # type: ignore
    return genai.Client(api_key=api_key)


# ----------------------------------------------------------------------------
# Classification core
# ----------------------------------------------------------------------------
def build_user_prompt(batch: pd.DataFrame) -> str:
    lines = ["Classify each startup below:\n"]
    for _, row in batch.iterrows():
        desc = str(row.get("bus_description", "")).strip().replace("\n", " ")
        lines.append(f"firm_id: {row['firm_id']}\nDescription: {desc}\n")
    return "\n".join(lines)


def parse_response_text(text: str | None) -> List[Dict[str, Any]]:
    if not text:
        raise ValueError("Empty response from model (likely blocked or rate-limited).")
    text = text.strip()
    # strip code fences if the model adds them
    if text.startswith("```"):
        # remove first fence line (```json or ```)
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[: -3]
        text = text.strip()
    # Try to locate the first JSON array if there's surrounding text.
    # If the model returned a single object instead of an array, wrap it.
    arr_start, arr_end = text.find("["), text.rfind("]")
    obj_start, obj_end = text.find("{"), text.rfind("}")
    if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
        text = text[arr_start : arr_end + 1]
    elif obj_start != -1 and obj_end != -1 and obj_end > obj_start:
        text = "[" + text[obj_start : obj_end + 1] + "]"
    parsed = json.loads(text)
    # Some responses wrap the array under a key like {"results": [...]}
    if isinstance(parsed, dict):
        for v in parsed.values():
            if isinstance(v, list):
                return v
        return [parsed]
    return parsed


def _build_genai_config(system_prompt: str):
    """Builds a GenerateContentConfig. Falls back to dict for older SDKs."""
    try:
        from google.genai import types  # type: ignore

        return types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            temperature=0.0,
        )
    except Exception:  # noqa: BLE001
        return {"system_instruction": system_prompt, "response_mime_type": "application/json"}


def _extract_text(response) -> str | None:
    """Extracts text from a genai response, defending against None / blocked content."""
    text = getattr(response, "text", None)
    if text:
        return text
    # Fallback: walk candidates -> content.parts -> text
    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) if content is not None else None
        if parts:
            chunks = [getattr(p, "text", None) for p in parts]
            joined = "".join(c for c in chunks if c)
            if joined:
                return joined
    return None


def classify_batch(
    client,
    batch: pd.DataFrame,
    system_prompt: str,
    show_explanations: bool,
) -> List[Dict[str, Any]]:
    """Sends one batch of firms to Gemini and returns parsed list of dicts."""
    user_prompt = build_user_prompt(batch)
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            config=_build_genai_config(system_prompt),
            contents=user_prompt,
        )
        text = _extract_text(response)
        # Stash last raw response for debugging in the UI
        st.session_state["last_raw_response"] = text or "<empty>"
        results = parse_response_text(text)
        # Ensure every firm_id from this batch has a row
        returned_ids = {str(r.get("firm_id")) for r in results}
        for _, row in batch.iterrows():
            if str(row["firm_id"]) not in returned_ids:
                results.append(
                    {
                        "firm_id": str(row["firm_id"]),
                        "label": None,
                        "confidence": None,
                        "category": "Unknown",
                        "explanation": "Model did not return this firm.",
                    }
                )
        # Optionally hide explanations
        if not show_explanations:
            for r in results:
                r["explanation"] = ""
        return results
    except Exception as exc:  # noqa: BLE001
        return [
            {
                "firm_id": str(row["firm_id"]),
                "label": None,
                "confidence": None,
                "category": "Error",
                "explanation": f"PARSE/API ERROR: {type(exc).__name__}: {exc}",
            }
            for _, row in batch.iterrows()
        ]


def run_classification(
    df: pd.DataFrame,
    strictness: str,
    batch_size: int,
    show_explanations: bool,
    api_key: str,
) -> pd.DataFrame:
    """Runs classification across the whole dataframe in batches with progress."""
    client = get_genai_client(api_key)
    system_prompt = build_system_prompt(strictness)

    all_rows: List[Dict[str, Any]] = []
    total = len(df)
    progress = st.progress(0.0, text="Starting classification...")
    status = st.empty()

    for start in range(0, total, batch_size):
        batch = df.iloc[start : start + batch_size]
        status.write(f"Classifying firms {start + 1}–{min(start + batch_size, total)} of {total}...")
        rows = classify_batch(client, batch, system_prompt, show_explanations)
        all_rows.extend(rows)
        progress.progress(min(1.0, (start + batch_size) / total))
        # Gentle pacing to avoid quota bursts
        if start + batch_size < total:
            time.sleep(0.5)

    progress.progress(1.0, text="Done.")
    status.empty()
    out = pd.DataFrame(all_rows)
    if "firm_id" in out.columns:
        out["firm_id"] = out["firm_id"].astype(str)
    return out


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def compute_metrics(df_eval: pd.DataFrame) -> Dict[str, Any]:
    y_true = df_eval["esg_dummy"].astype(int)
    y_pred = df_eval["label"].astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


# ----------------------------------------------------------------------------
# Rendering helpers
# ----------------------------------------------------------------------------
def render_summary_cards(df_results: pd.DataFrame, low_conf_thresh: float) -> None:
    n_total = len(df_results)
    valid = df_results.dropna(subset=["label"])
    n_esg = int((valid["label"] == 1).sum())
    n_non = int((valid["label"] == 0).sum())
    pct_esg = (n_esg / len(valid) * 100) if len(valid) else 0.0
    if "confidence" in df_results.columns:
        conf_numeric = pd.to_numeric(df_results["confidence"], errors="coerce")
        n_low = int((conf_numeric < low_conf_thresh).sum())
    else:
        n_low = 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total firms", n_total)
    c2.metric("ESG (1)", n_esg)
    c3.metric("Non-ESG (0)", n_non)
    c4.metric("% ESG", f"{pct_esg:.1f}%")
    c5.metric("Low-confidence", n_low)


def render_charts(df_results: pd.DataFrame) -> None:
    valid = df_results.dropna(subset=["label"]).copy()
    if valid.empty:
        st.info("No valid predictions to chart yet.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**ESG vs. Non-ESG counts**")
        valid["label_text"] = valid["label"].map({1: "ESG", 0: "Non-ESG"})
        counts = valid["label_text"].value_counts().reset_index()
        counts.columns = ["label", "count"]
        fig = px.bar(
            counts,
            x="label",
            y="count",
            color="label",
            color_discrete_map={"ESG": "#2E7D32", "Non-ESG": "#9E9E9E"},
            text="count",
        )
        fig.update_layout(showlegend=False, height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Category distribution**")
        if "category" in valid.columns:
            cats = valid["category"].fillna("Unknown").value_counts().reset_index()
            cats.columns = ["category", "count"]
            fig2 = px.pie(cats, names="category", values="count", hole=0.45)
            fig2.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Categories not available.")

    if "confidence" in valid.columns:
        st.markdown("**Confidence distribution**")
        conf = pd.to_numeric(valid["confidence"], errors="coerce").dropna()
        if not conf.empty:
            fig3 = px.histogram(conf, nbins=20, labels={"value": "confidence"})
            fig3.update_layout(showlegend=False, height=300, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig3, use_container_width=True)


def render_confusion_matrix(metrics: Dict[str, Any]) -> None:
    cm = [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Pred Non-ESG", "Pred ESG"],
            y=["Actual Non-ESG", "Actual ESG"],
            text=cm,
            texttemplate="%{text}",
            colorscale="Greens",
            showscale=False,
        )
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)


def render_results_table(df_results: pd.DataFrame, df_input: pd.DataFrame) -> None:
    merged = df_input.merge(df_results, on="firm_id", how="left")
    merged["description_preview"] = (
        merged["bus_description"].astype(str).str.slice(0, 160) + "..."
    )
    cols = ["firm_id", "description_preview", "label", "category", "confidence", "explanation"]
    display_cols = [c for c in cols if c in merged.columns]
    st.dataframe(merged[display_cols], use_container_width=True, height=420)
    csv_buf = io.StringIO()
    merged[display_cols].to_csv(csv_buf, index=False)
    st.download_button(
        "Download results CSV",
        data=csv_buf.getvalue(),
        file_name="esg_results.csv",
        mime="text/csv",
    )


def render_decision_insight(df_results: pd.DataFrame, metrics: Dict[str, Any] | None) -> None:
    valid = df_results.dropna(subset=["label"])
    if valid.empty:
        return
    pct_esg = (valid["label"] == 1).mean() * 100
    bullets: List[str] = []
    if pct_esg < 25:
        bullets.append(
            f"Only {pct_esg:.0f}% of firms were flagged as ESG — most descriptions look like generic SaaS or marketplaces rather than direct ESG outcomes."
        )
    elif pct_esg > 60:
        bullets.append(
            f"{pct_esg:.0f}% of firms were flagged as ESG — consider tightening the strictness setting to Conservative if many of these are ambiguous."
        )
    else:
        bullets.append(
            f"{pct_esg:.0f}% of firms were classified as ESG. The split looks reasonable; spot-check the borderline rows."
        )

    conf = pd.to_numeric(valid.get("confidence"), errors="coerce")
    if conf is not None and conf.notna().any():
        low = (conf < 0.6).sum()
        if low:
            bullets.append(
                f"{low} firm(s) have confidence below 0.6 — review these manually before treating the label as reliable."
            )

    if metrics is not None:
        if metrics["recall"] < 0.7:
            bullets.append(
                "Recall is below 0.7 — the model is missing real ESG firms. Try the Inclusive strictness setting and review false negatives."
            )
        if metrics["precision"] < 0.7:
            bullets.append(
                "Precision is below 0.7 — the model is over-flagging. Try the Conservative strictness setting and review false positives."
            )

    st.markdown("### Decision insight")
    for b in bullets:
        st.markdown(f"- {b}")


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
def sidebar_settings() -> Dict[str, Any]:
    st.sidebar.header("Settings")

    with st.sidebar.expander("API key", expanded=False):
        st.caption(
            "Set the `GOOGLE_API_KEY` environment variable, add it to "
            "`.streamlit/secrets.toml`, or paste it here for this session only."
        )
        st.text_input(
            "Gemini API key (session only)",
            key="api_key_input",
            type="password",
            help="Stored only in this Streamlit session — never written to disk.",
        )

    strictness = st.sidebar.radio(
        "Strictness mode",
        options=["Conservative", "Balanced", "Inclusive"],
        index=1,
        help="Controls how aggressive the model is when labeling firms as ESG.",
    )
    batch_size = st.sidebar.slider(
        "Batch size",
        min_value=1,
        max_value=25,
        value=10,
        help="Number of firm descriptions sent per Gemini call.",
    )
    conf_threshold = st.sidebar.slider(
        "Low-confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="Predictions below this confidence are flagged for manual review.",
    )
    show_explanations = st.sidebar.checkbox("Show explanations", value=True)
    eval_against_truth = st.sidebar.checkbox(
        "Evaluate against ground truth (if available)", value=True
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Model: `{MODEL_NAME}`")

    return dict(
        strictness=strictness,
        batch_size=batch_size,
        conf_threshold=conf_threshold,
        show_explanations=show_explanations,
        eval_against_truth=eval_against_truth,
    )


def header_section() -> None:
    st.title("ESG Startup Classifier + Trading Strategy")
    st.write(
        "Two tools in one app. **ESG Classifier**: decide whether a startup's "
        "core business is directly tied to Environmental, Social, or Governance "
        "outcomes using Gemini-2.5-Flash on free-text descriptions. **ESG "
        "Trading Strategy**: backtest a moving-average crossover on an "
        "ESG-aligned ETF (default ESGU) versus a broad-market benchmark "
        "(default SPY) to test whether ESG investing has held up under a "
        "simple momentum rule."
    )


def input_section() -> Tuple[pd.DataFrame | None, str]:
    """Returns the input dataframe and the source mode used."""
    tab_manual, tab_csv, tab_sample = st.tabs(
        ["Single firm", "Upload CSV", "Sample dataset"]
    )

    df: pd.DataFrame | None = None
    source = ""

    with tab_manual:
        col_a, col_b = st.columns([1, 3])
        firm_id = col_a.text_input("Firm ID", value="firm_001")
        description = col_b.text_area(
            "Business description",
            value="",
            height=140,
            placeholder="Paste the startup's business description here...",
        )
        if firm_id and description.strip():
            df = pd.DataFrame(
                [{"firm_id": str(firm_id).strip(), "bus_description": description.strip()}]
            )
            source = "manual"

    with tab_csv:
        st.caption("CSV must contain at least `firm_id` and `bus_description`. "
                   "An optional `esg_dummy` column enables ground-truth evaluation.")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            try:
                df_up = pd.read_csv(uploaded)
            except Exception:  # noqa: BLE001
                uploaded.seek(0)
                df_up = pd.read_csv(uploaded, header=None)
                df_up.columns = ["firm_id", "bus_description"] + (
                    ["esg_dummy"] if df_up.shape[1] >= 3 else []
                ) + list(df_up.columns[3:])
            ok, msg = validate_uploaded_csv(df_up)
            if not ok:
                st.error(msg)
            else:
                df = _normalize_columns(df_up)
                source = "upload"

    with tab_sample:
        st.caption("Loads the Project 1 reference dataset from GitHub.")
        st.code(DEFAULT_CSV_URL, language="text")
        if st.button("Load sample dataset"):
            try:
                df = load_default_csv()
                st.session_state["sample_df"] = df
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not load sample CSV: {exc}")
        if "sample_df" in st.session_state and df is None:
            df = st.session_state["sample_df"]
        if df is not None and source == "":
            source = "sample"

    if df is not None:
        st.success(f"Loaded {len(df)} firm(s).")
        with st.expander("Preview input"):
            st.dataframe(df.head(20), use_container_width=True)

    return df, source


def classifier_tab(settings: Dict[str, Any]) -> None:
    st.markdown("### 1. Provide input")
    df_input, source = input_section()

    st.markdown("### 2. Run classification")
    run = st.button(
        "Classify firms",
        type="primary",
        disabled=df_input is None or df_input.empty,
    )

    if run:
        api_key = get_api_key()
        if not api_key:
            st.error(
                "No Gemini API key found. Set `GOOGLE_API_KEY` in your environment, "
                "add it to `.streamlit/secrets.toml`, or paste it into the sidebar."
            )
            return
        if len(df_input) > 500:
            st.warning(
                f"You're about to classify {len(df_input)} firms — this may use significant API quota."
            )

        with st.spinner("Calling Gemini..."):
            df_results = run_classification(
                df=df_input,
                strictness=settings["strictness"],
                batch_size=settings["batch_size"],
                show_explanations=settings["show_explanations"],
                api_key=api_key,
            )
        st.session_state["df_results"] = df_results
        st.session_state["df_input"] = df_input

    # Render results if any are present
    if "df_results" in st.session_state and "df_input" in st.session_state:
        df_results = st.session_state["df_results"]
        df_input = st.session_state["df_input"]

        # Surface API/parse errors prominently
        err_rows = df_results[df_results["category"].isin(["Error", "Unknown"])] if "category" in df_results.columns else pd.DataFrame()
        if not err_rows.empty:
            first_err = ""
            if "explanation" in err_rows.columns and not err_rows["explanation"].empty:
                first_err = str(err_rows["explanation"].iloc[0])
            st.error(
                f"{len(err_rows)} firm(s) failed to classify. First error: {first_err}"
            )
            with st.expander("Debug: raw model response from last batch"):
                st.code(st.session_state.get("last_raw_response", "<none>"))

        st.markdown("### 3. Results summary")
        render_summary_cards(df_results, settings["conf_threshold"])

        st.markdown("### 4. Charts")
        render_charts(df_results)

        st.markdown("### 5. Results table")
        render_results_table(df_results, df_input)

        # Evaluation if ground truth available
        metrics: Dict[str, Any] | None = None
        if (
            settings["eval_against_truth"]
            and "esg_dummy" in df_input.columns
            and df_input["esg_dummy"].notna().any()
        ):
            st.markdown("### 6. Evaluation against ground truth")
            df_eval = df_input.merge(df_results, on="firm_id", how="inner")
            df_eval = df_eval.dropna(subset=["esg_dummy", "label"])
            if df_eval.empty:
                st.info("No overlapping rows with ground truth and a valid prediction.")
            else:
                df_eval["esg_dummy"] = df_eval["esg_dummy"].astype(int)
                df_eval["label"] = df_eval["label"].astype(int)
                metrics = compute_metrics(df_eval)
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    m1, m2 = st.columns(2)
                    m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                    m2.metric("Precision", f"{metrics['precision']:.3f}")
                    m3, m4 = st.columns(2)
                    m3.metric("Recall", f"{metrics['recall']:.3f}")
                    m4.metric("F1", f"{metrics['f1']:.3f}")
                with col_b:
                    render_confusion_matrix(metrics)

                st.markdown("#### False positives & false negatives")
                fp_df = df_eval[(df_eval["label"] == 1) & (df_eval["esg_dummy"] == 0)]
                fn_df = df_eval[(df_eval["label"] == 0) & (df_eval["esg_dummy"] == 1)]
                with st.expander(f"False positives ({len(fp_df)}) — model said ESG, but wasn't"):
                    st.dataframe(
                        fp_df[
                            [c for c in ["firm_id", "bus_description", "category", "confidence", "explanation"] if c in fp_df.columns]
                        ],
                        use_container_width=True,
                    )
                with st.expander(f"False negatives ({len(fn_df)}) — model said non-ESG, but was"):
                    st.dataframe(
                        fn_df[
                            [c for c in ["firm_id", "bus_description", "category", "confidence", "explanation"] if c in fn_df.columns]
                        ],
                        use_container_width=True,
                    )

        st.markdown("### 7. Insights")
        render_decision_insight(df_results, metrics)


# ============================================================================
# ESG Trading Strategy tab — Moving Average crossover backtest
# ============================================================================

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Pulls Close prices via yfinance. Returns a single-column 'close' DataFrame."""
    import yfinance as yf

    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
    )
    if df is None or df.empty:
        raise ValueError(f"No price data returned for ticker '{ticker}'.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df.columns:
        raise ValueError(f"Unexpected response shape for '{ticker}': {list(df.columns)}")
    return df[["Close"]].rename(columns={"Close": "close"}).dropna()


def compute_ma_signals(prices: pd.DataFrame, short: int, long: int) -> pd.DataFrame:
    """Adds short/long SMA, position, daily return, strategy return, equity curves."""
    if short >= long:
        raise ValueError("Short MA window must be smaller than long MA window.")
    df = prices.copy()
    df["sma_short"] = df["close"].rolling(short).mean()
    df["sma_long"] = df["close"].rolling(long).mean()
    df["signal"] = (df["sma_short"] > df["sma_long"]).astype(int)
    # Position is yesterday's signal — we can only act on signals after they form.
    df["position"] = df["signal"].shift(1).fillna(0)
    df["daily_ret"] = df["close"].pct_change().fillna(0.0)
    df["strategy_ret"] = df["position"] * df["daily_ret"]
    df["equity_strategy"] = (1.0 + df["strategy_ret"]).cumprod()
    df["equity_bh"] = (1.0 + df["daily_ret"]).cumprod()
    df["trade"] = df["position"].diff().fillna(0)
    return df


def backtest_metrics(df: pd.DataFrame, capital: float) -> Dict[str, float]:
    """Computes total return, CAGR, Sharpe, max drawdown, # trades, final equity."""
    eq = df["equity_strategy"]
    if eq.empty:
        return {}
    total_ret = float(eq.iloc[-1] - 1)
    days = len(df)
    cagr = float(eq.iloc[-1] ** (252.0 / max(days, 1)) - 1) if days > 0 else 0.0
    daily = df["strategy_ret"]
    sharpe = float((daily.mean() / daily.std()) * np.sqrt(252)) if daily.std() > 0 else 0.0
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = float(dd.min()) if not dd.empty else 0.0
    n_trades = int((df["trade"].abs() > 0).sum())
    bh_ret = float(df["equity_bh"].iloc[-1] - 1)
    return {
        "total_return": total_ret,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_trades": n_trades,
        "final_equity": float(capital * eq.iloc[-1]),
        "buy_hold_return": bh_ret,
    }


def render_strategy_chart(df: pd.DataFrame, ticker: str, short: int, long: int) -> None:
    """Price + MAs + buy/sell markers."""
    buys = df[df["trade"] > 0]
    sells = df[df["trade"] < 0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["close"], name="Close", line=dict(color="#37474F", width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df["sma_short"], name=f"SMA {short}", line=dict(color="#1976D2", width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df["sma_long"], name=f"SMA {long}", line=dict(color="#E53935", width=1)))
    fig.add_trace(go.Scatter(
        x=buys.index, y=buys["close"], mode="markers", name="Buy",
        marker=dict(symbol="triangle-up", color="#2E7D32", size=10),
    ))
    fig.add_trace(go.Scatter(
        x=sells.index, y=sells["close"], mode="markers", name="Sell",
        marker=dict(symbol="triangle-down", color="#C62828", size=10),
    ))
    fig.update_layout(
        title=f"{ticker} — price, MA crossover signals",
        height=360,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_equity_curves(esg_df: pd.DataFrame, bench_df: pd.DataFrame, esg_t: str, bench_t: str) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=esg_df.index, y=esg_df["equity_strategy"], name=f"{esg_t} MA strategy", line=dict(color="#2E7D32", width=2)))
    fig.add_trace(go.Scatter(x=esg_df.index, y=esg_df["equity_bh"], name=f"{esg_t} buy & hold", line=dict(color="#2E7D32", width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=bench_df.index, y=bench_df["equity_strategy"], name=f"{bench_t} MA strategy", line=dict(color="#1565C0", width=2)))
    fig.add_trace(go.Scatter(x=bench_df.index, y=bench_df["equity_bh"], name=f"{bench_t} buy & hold", line=dict(color="#1565C0", width=1, dash="dot")))
    fig.update_layout(
        title="Equity curves (growth of $1)",
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def trading_decision_insight(esg_m: Dict[str, float], bench_m: Dict[str, float], esg_t: str, bench_t: str) -> None:
    bullets: List[str] = []
    esg_strat_vs_bh = esg_m["total_return"] - esg_m["buy_hold_return"]
    bench_strat_vs_bh = bench_m["total_return"] - bench_m["buy_hold_return"]

    if esg_strat_vs_bh > 0:
        bullets.append(f"On {esg_t}, the MA strategy beat buy-and-hold by {esg_strat_vs_bh*100:.1f} percentage points.")
    else:
        bullets.append(f"On {esg_t}, the MA strategy underperformed buy-and-hold by {-esg_strat_vs_bh*100:.1f} percentage points — typical of MA crossovers in trending markets.")

    if esg_m["total_return"] > bench_m["total_return"]:
        bullets.append(f"The ESG strategy ({esg_t}) returned {esg_m['total_return']*100:.1f}% vs the benchmark's {bench_m['total_return']*100:.1f}% — ESG outperformed over this window.")
    else:
        bullets.append(f"The ESG strategy ({esg_t}) returned {esg_m['total_return']*100:.1f}% vs the benchmark's {bench_m['total_return']*100:.1f}% — the benchmark outperformed over this window.")

    if esg_m["max_drawdown"] > bench_m["max_drawdown"]:
        bullets.append(f"ESG had a shallower max drawdown ({esg_m['max_drawdown']*100:.1f}% vs {bench_m['max_drawdown']*100:.1f}%) — lower downside risk.")
    else:
        bullets.append(f"ESG had a deeper max drawdown ({esg_m['max_drawdown']*100:.1f}% vs {bench_m['max_drawdown']*100:.1f}%).")

    if esg_m["sharpe"] > bench_m["sharpe"]:
        bullets.append(f"ESG had a higher Sharpe ({esg_m['sharpe']:.2f} vs {bench_m['sharpe']:.2f}) — better risk-adjusted return.")

    bullets.append(f"Total trades: {esg_t}={esg_m['n_trades']}, {bench_t}={bench_m['n_trades']}. Many trades implies higher transaction costs that this backtest does not include.")

    st.markdown("### Decision insight")
    for b in bullets:
        st.markdown(f"- {b}")


def trading_tab() -> None:
    st.markdown(
        "### Moving Average crossover backtest"
    )
    st.write(
        "Backtest a classic MA crossover strategy on an ESG-aligned ETF and "
        "compare it head-to-head with a broad-market benchmark. Go long when "
        "the short SMA is above the long SMA; otherwise hold cash. The chart "
        "marks every buy and sell signal so you can sanity-check the logic."
    )

    c1, c2 = st.columns(2)
    esg_ticker = c1.text_input(
        "ESG ticker",
        value="ESGU",
        help="Default: iShares ESG Aware MSCI USA ETF. Other options: SUSA, ICLN, ESGV.",
    )
    bench_ticker = c2.text_input(
        "Benchmark ticker",
        value="SPY",
        help="Default: SPDR S&P 500 ETF.",
    )

    c3, c4 = st.columns(2)
    short_window = c3.slider("Short MA window (days)", 5, 100, 50, step=5)
    long_window = c4.slider("Long MA window (days)", 50, 300, 200, step=10)

    today = date.today()
    default_start = today - timedelta(days=365 * 5)
    c5, c6 = st.columns(2)
    start_d = c5.date_input("Start date", value=default_start, max_value=today)
    end_d = c6.date_input("End date", value=today, max_value=today)

    capital = st.number_input(
        "Initial capital ($)", min_value=100.0, value=10_000.0, step=1000.0
    )

    run_bt = st.button("Run backtest", type="primary")

    if run_bt:
        if short_window >= long_window:
            st.error("Short MA window must be smaller than long MA window.")
            return
        if start_d >= end_d:
            st.error("Start date must be before end date.")
            return

        try:
            with st.spinner("Fetching prices..."):
                esg_prices = fetch_prices(esg_ticker.strip().upper(), str(start_d), str(end_d))
                bench_prices = fetch_prices(bench_ticker.strip().upper(), str(start_d), str(end_d))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not fetch prices: {exc}")
            return

        try:
            esg_df = compute_ma_signals(esg_prices, short_window, long_window)
            bench_df = compute_ma_signals(bench_prices, short_window, long_window)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Backtest error: {exc}")
            return

        st.session_state["bt"] = {
            "esg_df": esg_df,
            "bench_df": bench_df,
            "esg_t": esg_ticker.upper(),
            "bench_t": bench_ticker.upper(),
            "short": short_window,
            "long": long_window,
            "capital": capital,
        }

    if "bt" in st.session_state:
        bt = st.session_state["bt"]
        esg_df, bench_df = bt["esg_df"], bt["bench_df"]
        esg_t, bench_t = bt["esg_t"], bt["bench_t"]
        capital = bt["capital"]

        esg_metrics = backtest_metrics(esg_df, capital)
        bench_metrics = backtest_metrics(bench_df, capital)

        st.markdown("#### Performance summary")
        summary = pd.DataFrame(
            {
                esg_t: [
                    f"{esg_metrics['total_return']*100:.2f}%",
                    f"{esg_metrics['cagr']*100:.2f}%",
                    f"{esg_metrics['sharpe']:.2f}",
                    f"{esg_metrics['max_drawdown']*100:.2f}%",
                    esg_metrics["n_trades"],
                    f"${esg_metrics['final_equity']:,.0f}",
                    f"{esg_metrics['buy_hold_return']*100:.2f}%",
                ],
                bench_t: [
                    f"{bench_metrics['total_return']*100:.2f}%",
                    f"{bench_metrics['cagr']*100:.2f}%",
                    f"{bench_metrics['sharpe']:.2f}",
                    f"{bench_metrics['max_drawdown']*100:.2f}%",
                    bench_metrics["n_trades"],
                    f"${bench_metrics['final_equity']:,.0f}",
                    f"{bench_metrics['buy_hold_return']*100:.2f}%",
                ],
            },
            index=[
                "Strategy total return",
                "Strategy CAGR",
                "Sharpe (strategy)",
                "Max drawdown (strategy)",
                "Number of trades",
                "Final equity",
                "Buy & hold return",
            ],
        )
        st.dataframe(summary, use_container_width=True)

        st.markdown("#### Price + MA crossover signals")
        render_strategy_chart(esg_df, esg_t, bt["short"], bt["long"])
        render_strategy_chart(bench_df, bench_t, bt["short"], bt["long"])

        st.markdown("#### Equity curves")
        render_equity_curves(esg_df, bench_df, esg_t, bench_t)

        trading_decision_insight(esg_metrics, bench_metrics, esg_t, bench_t)


# ============================================================================
# Top-level entry
# ============================================================================

def main() -> None:
    settings = sidebar_settings()
    header_section()

    tab_class, tab_strat = st.tabs(["ESG Classifier", "ESG Trading Strategy"])
    with tab_class:
        classifier_tab(settings)
    with tab_strat:
        trading_tab()


if __name__ == "__main__":
    main()
