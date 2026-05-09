# ESG Startup Classifier + Trading Strategy

An interactive Streamlit web app with **two tabs** that fulfills the Project 3 vibe-coding requirement (Moving Average Trading Strategy task) while preserving the ESG analysis from Project 1.

## What the app does

### Tab 1 — ESG Classifier
- Lets users classify a single firm or upload a CSV of firm descriptions.
- Sends batches of descriptions to Gemini-2.5-Flash using a strict ESG analyst prompt.
- Returns label, confidence, category, and a one-sentence explanation per firm.
- When ground-truth labels are present (`esg_dummy` column), it computes accuracy, precision, recall, F1, and a confusion matrix, and lists false positives / negatives.
- Renders interactive charts (ESG vs. non-ESG, category breakdown, confidence distribution) and a plain-English insight panel.

### Tab 2 — ESG Trading Strategy (Moving Average crossover)
- Pulls historical prices for any two tickers via `yfinance` (default: **ESGU** vs **SPY**).
- Backtests a classic SMA crossover: long when short SMA > long SMA, otherwise cash.
- User-tunable inputs: ticker symbols, short/long MA windows, date range, initial capital.
- Outputs: side-by-side performance summary (total return, CAGR, Sharpe, max drawdown, # trades, final equity, buy & hold return), price chart with buy/sell markers, equity curves comparing strategy vs. buy-and-hold for both tickers, and a plain-English decision insight panel.
- Decision question: "Has an ESG-aligned ETF held up against the broad market under a simple momentum rule, and is the strategy worth running over buy-and-hold?"

## Decision question

> "Is this startup's core business directly related to Environmental, Social, or Governance themes, or is it just a general business/technology company with positive wording?"

## User-controllable inputs

- **Strictness mode** — Conservative / Balanced / Inclusive (changes the system prompt).
- **Batch size** — number of firms per Gemini call (1–25).
- **Low-confidence threshold** — flags uncertain predictions for manual review.
- **Show explanations** toggle.
- **Evaluate against ground truth** toggle (auto-detects `esg_dummy`).

## Setup

```bash
pip install -r requirements.txt
```

## API key

The app reads the Gemini API key in this priority order:

1. Environment variable `GOOGLE_API_KEY` (or `GEMINI_API_KEY`).
2. `.streamlit/secrets.toml` entry `GOOGLE_API_KEY = "..."`.
3. Sidebar text box (kept only in the running session — never written to disk).

```bash
# macOS / Linux
export GOOGLE_API_KEY=your_key_here

# Windows PowerShell
$env:GOOGLE_API_KEY = "your_key_here"

# Windows cmd
set GOOGLE_API_KEY=your_key_here
```

## Run

```bash
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`).

## CSV format

The CSV must contain these columns (case-insensitive; common aliases like `id`, `description`, `label` are auto-renamed):

| column            | required | description                                          |
| ----------------- | -------- | ---------------------------------------------------- |
| `firm_id`         | yes      | Unique identifier for the firm                       |
| `bus_description` | yes      | Free-text business description                       |
| `esg_dummy`       | no       | Ground-truth ESG label (0/1) — enables evaluation     |

A sample dataset is preloaded inside the **Sample dataset** tab — it pulls `Project1_LLM_ESG_Classification.csv` from your course GitHub repo.

## Project structure

```
FinalProject_Programming/
├─ app.py                              # Streamlit app
├─ requirements.txt                    # Python dependencies
├─ README.md                           # this file
├─ untitled6.py                        # original notebook (kept for reference)
├─ project instructions.txt            # course instructions
├─ ESG_Vibe_Coding_Prompt.txt          # design prompt used to build this app
└─ ESG_App_Prompt_and_Design_Document.docx
```

## Notes

- API failures or malformed JSON responses do not crash the app — affected rows are marked `Error` with the exception message.
- Batches are paced with a short sleep to be quota-friendly.
- All results can be downloaded as a CSV from the results table.
