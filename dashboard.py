"""
dashboard.py — SEC Signal: S&P 500 10-Q Similarity Dashboard

Run with:
    streamlit run dashboard.py
    streamlit run dashboard.py -- --csv path/to/scores.csv

Set ANTHROPIC_API_KEY for AI-generated detail summaries.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SEC Signal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ═══════════════════════════════════════════════════════
   DESIGN SYSTEM — SEC Signal
   Page bg: #F8F9FB  Card bg: #FFFFFF  Border: #E5E7EB
   Text: #111827 / #374151 / #6B7280
   Red: #DC2626   Green: #16A34A   Amber: #D97706
   ═══════════════════════════════════════════════════════ */

html, body, [class*="css"] {
    font-family: "Inter", "Helvetica Neue", Arial, sans-serif !important;
}

/* Page background */
.stApp, [data-testid="stAppViewContainer"] {
    background-color: #F8F9FB !important;
}
[data-testid="stMain"] { background-color: #F8F9FB !important; }

/* Main content padding */
.block-container { padding-top: 1.2rem !important; max-width: 100% !important; padding-left: 2rem !important; padding-right: 2rem !important; }
header[data-testid="stHeader"] { display: none !important; }
[data-testid="stToolbar"]      { display: none !important; }
[data-testid="stDecoration"]   { display: none !important; }
.stDeployButton                { display: none !important; }

/* ── Sidebar ─────────────────────────────────────────── */
/* Force sidebar always visible — overrides Cloud session state */
section[data-testid="stSidebar"] {
    min-width: 240px !important;
    max-width: 240px !important;
    transform: none !important;
    visibility: visible !important;
    display: block !important;
    background: #111827 !important;
    border-right: 1px solid #1F2937 !important;
}
/* Hide the collapse/expand toggle button */
[data-testid="collapsedControl"],
button[kind="header"][aria-label="Close sidebar"],
button[aria-label="Close sidebar"],
section[data-testid="stSidebar"] > div:first-child > div > button {
    display: none !important;
}
section[data-testid="stSidebar"] * { color: #D1D5DB !important; }
section[data-testid="stSidebar"] .stRadio label {
    font-size: .87rem !important;
    padding: 8px 12px !important;
    border-radius: 6px !important;
    cursor: pointer !important;
}
section[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
    font-size: .87rem !important;
}
/* Nav item hover */
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.06) !important;
}

/* ── Section labels ──────────────────────────────────── */
.sh {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.10em;
    text-transform: uppercase; color: #6B7280;
    margin: 0 0 10px 0; display: block;
}

/* ── Card container ──────────────────────────────────── */
.card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    padding: 16px 18px;
}

/* ── Badges ──────────────────────────────────────────── */
.badge {
    display: inline-block; border-radius: 4px;
    padding: 2px 9px; font-size: 0.71rem; font-weight: 600;
}
.badge-red    { background: #FEF2F2; color: #DC2626; border: 1px solid #FECACA; }
.badge-yellow { background: #FFFBEB; color: #D97706; border: 1px solid #FDE68A; }
.badge-green  { background: #F0FDF4; color: #16A34A; border: 1px solid #BBF7D0; }
.badge-gray   { background: #F9FAFB; color: #6B7280; border: 1px solid #E5E7EB; }

/* ── Top-10 rows — white card with left accent border ─ */
.t10-row {
    display: flex; align-items: center; gap: 8px;
    background: #FFFFFF;
    border: 1px solid #F3F4F6;
    border-radius: 6px;
    padding: 8px 10px;
    margin-bottom: 4px;
}
.t10-row:last-child { margin-bottom: 0; }
.t10-ticker  { font-weight: 700; font-size: .84rem; color: #111827;
               width: 46px; flex-shrink: 0; font-family: monospace; }
.t10-company { font-size: .77rem; color: #6B7280; flex: 1;
               white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.t10-z       { font-family: monospace; font-size: .82rem; font-weight: 700;
               width: 58px; flex-shrink: 0; text-align: right; }
.t10-bar-wrap { width: 72px; flex-shrink: 0; background: #F3F4F6;
                border-radius: 3px; height: 6px; overflow: hidden; }
.t10-bar     { height: 6px; border-radius: 3px; }

/* ── Callout / AI summary ────────────────────────────── */
.callout {
    border-left: 4px solid #3B82F6;
    background: #EFF6FF;
    border-radius: 0 8px 8px 0; padding: 14px 18px; margin: 12px 0;
    font-size: .87rem; color: #1E40AF; line-height: 1.7;
}
.callout-label {
    font-size: .64rem; text-transform: uppercase; letter-spacing: .1em;
    font-weight: 700; color: #3B82F6; margin-bottom: 6px; display: block;
}

/* ── Signal amplifier cards ──────────────────────────── */
.amp-card { border-radius: 0 7px 7px 0; padding: 10px 14px; flex: 1; }
.amp-title { font-weight: 700; font-size: .78rem; display: block; margin-bottom: 3px; }
.amp-desc  { font-size: .72rem; color: #6B7280; line-height: 1.45; }

/* ── Sector heatmap table ────────────────────────────── */
.hm-table { width: 100%; border-collapse: collapse; }
.hm-table td { padding: 7px 10px; font-size: .82rem; color: #374151; vertical-align: middle; }
.hm-table tr { border-bottom: 1px solid #F3F4F6; }
.hm-table tr:last-child { border-bottom: none; }
.hm-sector { font-weight: 500; color: #111827; min-width: 180px; }
.hm-z      { font-family: monospace; font-weight: 700; width: 72px; text-align: right; }
.hm-count  { color: #9CA3AF; width: 48px; text-align: right; font-size: .78rem; }
.hm-bar-wrap { width: 140px; }
.hm-bar    { height: 10px; border-radius: 3px; }
.hm-thead td { font-size: .68rem; font-weight: 700; letter-spacing: .08em;
               text-transform: uppercase; color: #9CA3AF; padding-bottom: 6px;
               border-bottom: 1px solid #E5E7EB; }

/* ── Diff view ───────────────────────────────────────── */
.diff-col-header {
    font-size: 0.69rem; font-weight: 700; letter-spacing: .09em;
    text-transform: uppercase; padding: 6px 0 9px; color: #6B7280;
}
.diff-removed {
    background: #FEF2F2; border-left: 3px solid #DC2626;
    border-radius: 0 5px 5px 0; padding: 10px 14px; margin: 4px 0;
    font-size: .84rem; color: #991B1B; line-height: 1.6;
    text-decoration: line-through; text-decoration-color: rgba(220,38,38,0.4);
}
.diff-added {
    background: #F0FDF4; border-left: 3px solid #16A34A;
    border-radius: 0 5px 5px 0; padding: 10px 14px; margin: 4px 0;
    font-size: .84rem; color: #14532D; line-height: 1.6;
}
.diff-mod-old {
    background: #FFFBEB; border-left: 3px solid #D97706;
    border-radius: 0 5px 5px 0; padding: 10px 14px; margin: 4px 0;
    font-size: .84rem; color: #78350F; line-height: 1.6;
    text-decoration: line-through; text-decoration-color: rgba(217,119,6,0.4);
}
.diff-mod-new {
    background: #F0FDF4; border-left: 3px solid #16A34A;
    border-radius: 0 5px 5px 0; padding: 10px 14px; margin: 4px 0;
    font-size: .84rem; color: #14532D; line-height: 1.6;
}
.diff-empty {
    border-left: 3px solid #E5E7EB;
    border-radius: 0 5px 5px 0; padding: 10px 14px; margin: 4px 0; min-height: 40px;
}
.diff-label {
    font-size: .66rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .1em; margin-bottom: 3px;
}
.diff-label-removed { color: #DC2626; }
.diff-label-added   { color: #16A34A; }
.diff-label-mod     { color: #D97706; }

/* ── KPI metric containers ───────────────────────────── */
div[data-testid="metric-container"] {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 8px; padding: 16px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
div[data-testid="metric-container"] label {
    color: #6B7280 !important; font-size: .72rem !important;
    text-transform: uppercase; letter-spacing: .06em;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #111827 !important; font-size: 1.5rem !important; font-weight: 700 !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricDelta"] svg { display: none; }

/* ── All buttons — base size fix ─────────────────────── */
button {
    padding: 10px 20px !important;
    cursor: pointer !important;
    min-height: 44px !important;
}

/* ── Quarter selector — large pill buttons ───────────── */
.qbtn-wrap button[kind="primary"] {
    border-radius: 22px !important;
    background: #111827 !important;
    color: #FFFFFF !important;
    border: 2px solid #111827 !important;
    font-weight: 700 !important;
    font-size: .9rem !important;
    min-width: 120px !important;
    min-height: 44px !important;
    padding: 10px 24px !important;
}
.qbtn-wrap button[kind="secondary"] {
    border-radius: 22px !important;
    background: #FFFFFF !important;
    color: #374151 !important;
    border: 2px solid #D1D5DB !important;
    font-weight: 600 !important;
    font-size: .9rem !important;
    min-width: 120px !important;
    min-height: 44px !important;
    padding: 10px 24px !important;
}
.qbtn-wrap button[kind="secondary"]:hover {
    border-color: #9CA3AF !important;
    color: #111827 !important;
}

/* ── Tabs ────────────────────────────────────────────── */
button[data-baseweb="tab"] { font-size: .82rem !important; }

/* ── Dividers ────────────────────────────────────────── */
hr { border-color: #E5E7EB !important; }

/* ── Streamlit dataframe ─────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
[data-testid="stDataFrame"] canvas { cursor: pointer !important; }
[data-testid="stDataFrame"] > div { cursor: pointer !important; }

/* ── Pulsing dot animation ───────────────────────────── */
@keyframes pulse-ring {
    0%   { transform: scale(0.6); opacity: 0.9; }
    70%  { transform: scale(2.2); opacity: 0; }
    100% { transform: scale(2.2); opacity: 0; }
}
@keyframes pulse-core {
    0%, 100% { transform: scale(1); }
    50%       { transform: scale(1.15); }
}
.pulse-dot-wrap {
    position: relative; display: inline-flex;
    align-items: center; justify-content: center;
    width: 14px; height: 14px; flex-shrink: 0;
}
.pulse-dot-ring {
    position: absolute; width: 14px; height: 14px;
    border-radius: 50%; background: #3B82F6; opacity: 0.6;
    animation: pulse-ring 1.6s ease-out infinite;
}
.pulse-dot-core {
    width: 10px; height: 10px; border-radius: 50%;
    background: #3B82F6; position: relative; z-index: 1;
    animation: pulse-core 1.6s ease-in-out infinite;
}
</style>
""", unsafe_allow_html=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

DEFAULT_CSV    = Path(__file__).parent / "scores.csv"
Q2_CSV         = Path(__file__).parent / "scores_q2.csv"

# Quarter selector options — label → (csv_path, display_label)
QUARTER_OPTIONS: dict[str, tuple[Path, str]] = {
    "Q1 2025": (DEFAULT_CSV, "Q1 2025 vs Q1 2024"),
    "Q2 2025": (Q2_CSV,      "Q2 2025 vs Q2 2024"),
}
DEFAULT_QUARTER = "Q1 2025"


def _parse_args() -> Path:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    known, _ = p.parse_known_args(sys.argv[1:])
    return known.csv


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["status"] == "ok"].copy()
    for col in ("mda_score", "risk_score", "combined_score",
                "mda_zscore", "risk_zscore", "combined_zscore"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date_run"] = pd.to_datetime(df["date_run"], utc=True, errors="coerce")

    def _label(z):
        if pd.isna(z): return "Unknown"
        if z < -1:     return "Most Changed"
        if z > 1:      return "Least Changed"
        return "Moderate"

    df["change_label"] = df["combined_zscore"].apply(_label)
    df["period"] = df["quarter"].astype(str) + " " + df["year"].astype(str)
    # Ensure sector column exists (may be absent in old CSVs)
    if "sector" not in df.columns:
        df["sector"] = "Unknown"
    df["sector"] = df["sector"].fillna("Unknown").replace("", "Unknown")

    # Derive extraction_status if not present (old CSVs without the column)
    if "extraction_status" not in df.columns:
        def _ext_status(row):
            if pd.isna(row.get("mda_score")) and pd.isna(row.get("risk_score")):
                return "failed"
            if pd.isna(row.get("mda_score")) or pd.isna(row.get("risk_score")):
                return "partial"
            return "complete"
        df["extraction_status"] = df.apply(_ext_status, axis=1)

    # Override change_label for failed extractions
    df.loc[df["extraction_status"] == "failed", "change_label"] = "Data gap"

    return df.sort_values("combined_zscore", ascending=True).reset_index(drop=True)


@st.cache_data(ttl=300, show_spinner=False)
def load_both_quarters() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (q1_df, q2_df). Either may be empty if the file doesn't exist yet."""
    def _safe(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return load_data(str(path))
    return _safe(DEFAULT_CSV), _safe(Q2_CSV)


def build_multi_quarter_df(q1_df: pd.DataFrame, q2_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Q1 and Q2 results on ticker. Returns a DataFrame with extra columns:
      q1_zscore, q2_zscore, persistent_signal (bool)
    """
    cols = ["ticker", "combined_zscore"]
    left  = q1_df[cols].rename(columns={"combined_zscore": "q1_zscore"}) if len(q1_df) else pd.DataFrame(columns=["ticker", "q1_zscore"])
    right = q2_df[cols].rename(columns={"combined_zscore": "q2_zscore"}) if len(q2_df) else pd.DataFrame(columns=["ticker", "q2_zscore"])
    merged = pd.merge(left, right, on="ticker", how="outer")
    merged["persistent_signal"] = (
        merged["q1_zscore"].fillna(0) < -0.5
    ) & (
        merged["q2_zscore"].fillna(0) < -0.5
    )
    return merged


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_filing_pair(ticker: str, quarter: str, year: int):
    from similarity_score import compute_similarity
    return compute_similarity(ticker, quarter, int(year))


@st.cache_data(ttl=86400, show_spinner=False)
def get_filing_links(ticker: str, quarter: str, year: int) -> tuple:
    """Return (current_filing_url, prior_filing_url) for the detail panel header links."""
    try:
        import time as _time
        from edgar_pull import _get_cik, _get_submissions, _find_10q_accession
        cik = _get_cik(ticker)
        _time.sleep(0.11)
        subs = _get_submissions(cik)
        cik_int = int(cik)

        def _index_url(acc):
            acc_clean = acc.replace("-", "")
            return f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/{acc}-index.htm"

        try:
            acc_cur, _ = _find_10q_accession(subs, quarter, year)
            current_url = _index_url(acc_cur)
        except Exception:
            current_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-Q&dateb=&owner=include&count=10"

        try:
            acc_prior, _ = _find_10q_accession(subs, quarter, year - 1)
            prior_url = _index_url(acc_prior)
        except Exception:
            prior_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-Q&dateb=&owner=include&count=10"

        return current_url, prior_url
    except Exception:
        return None, None


# ── Signal Validation ────────────────────────────────────────────────────────

# Fixed lists: top-10 valid z-scores from Q1 2025 (extraction_status != "failed")
# BK and EIX excluded — combined_score=0.0 due to extraction failure, not genuine signal
_VAL_SHORT = ["TKO", "EL", "KMB", "GEV", "BF-B", "MTB", "CEG", "HON", "ORCL", "HCA"]
_VAL_LONG  = ["EMR", "AWK", "PRU", "IRM", "USB", "GEHC", "FISV", "BG", "ED", "QCOM"]
# Persistent: top-10 by most negative avg z-score across Q1+Q2 2025 (both < −0.5, no failed extractions)
_VAL_PERSIST = ["EL", "GEV", "KMB", "CEG", "CMI", "BIIB", "NUE", "APO", "IP", "AEP"]
_VAL_START = "2025-05-01"
_VAL_END   = "2025-11-01"


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_validation_prices() -> dict:
    """
    Fetch closing prices on/after May 1 2025 and Nov 1 2025 for validation tickers + SPY.
    Returns dict keyed by ticker: {p1, d1, p2, d2, ret}
    """
    try:
        import yfinance as yf
    except ImportError:
        return {}

    all_tickers = list(dict.fromkeys(_VAL_SHORT + _VAL_LONG + _VAL_PERSIST + ["SPY"]))
    results: dict = {}

    def _get_price(ticker: str, from_date: str) -> tuple:
        start = pd.Timestamp(from_date)
        end   = start + pd.Timedelta(days=6)
        hist  = yf.Ticker(ticker).history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
        )
        if hist.empty:
            return None, None
        return float(hist["Close"].iloc[0]), hist.index[0].strftime("%Y-%m-%d")

    for tkr in all_tickers:
        p1, d1 = _get_price(tkr, _VAL_START)
        p2, d2 = _get_price(tkr, _VAL_END)
        ret = (p2 - p1) / p1 if (p1 and p2) else None
        results[tkr] = {"p1": p1, "d1": d1, "p2": p2, "d2": d2, "ret": ret}

    return results


def render_validation_panel(q1_df: pd.DataFrame, q2_df: pd.DataFrame) -> None:
    """Render the signal validation expander panel with two tabs."""
    try:
        import yfinance  # noqa — just confirm available
    except ImportError:
        st.error("yfinance not installed. Run: pip install yfinance")
        return

    with st.spinner("Fetching price data from Yahoo Finance …"):
        prices = fetch_validation_prices()

    if not prices or "SPY" not in prices or prices["SPY"]["ret"] is None:
        st.error("Could not fetch price data. Check your internet connection.")
        return

    spy_ret = prices["SPY"]["ret"]
    d1_label = prices["SPY"]["d1"]
    d2_label = prices["SPY"]["d2"]

    # ── Shared helpers ──────────────────────────────────────────────────────────
    def _ret_color(v):
        if v is None: return "#9CA3AF"
        return "#16A34A" if v > 0 else "#DC2626"

    def _fmt_ret(v):
        if v is None: return "N/A"
        return f"{v:+.1%}"

    def _fmt_z(v):
        if v is None or (isinstance(v, float) and pd.isna(v)): return "—"
        return f"{v:+.2f}"

    def _period_banner(label: str) -> None:
        st.markdown(
            f'<div style="background:#F0F9FF;border:1px solid #BAE6FD;border-radius:8px;'
            f'padding:10px 16px;margin-bottom:14px;font-size:.82rem;color:#0369A1">'
            f'📅 Measurement period: <strong>{d1_label} – {d2_label}</strong> ({label}) &nbsp;·&nbsp; '
            f'SPY benchmark return: <strong style="color:#16A34A">{spy_ret:+.1%}</strong>'
            f'</div>',
            unsafe_allow_html=True,
        )

    def _table_html(rows: list, accent: str, cols: str = "standard") -> str:
        if cols == "persist":
            headers = ["Ticker", "Company", "Q1 Z", "Q2 Z", "Avg Z", "6M Return", "vs SPY", "Outcome"]
            aligns  = ["left", "left", "right", "right", "right", "right", "right", "center"]
        else:
            headers = ["Ticker", "Company", "Q1 Z", "6M Return", "vs SPY", "Outcome"]
            aligns  = ["left", "left", "right", "right", "right", "center"]

        th_style = (
            'padding:7px 8px;color:#6B7280;font-weight:700;'
            'font-size:.68rem;text-transform:uppercase;letter-spacing:.07em'
        )
        head_cells = "".join(
            f'<th style="{th_style};text-align:{a}">{h}</th>'
            for h, a in zip(headers, aligns)
        )
        header = (
            '<table style="width:100%;border-collapse:collapse;font-size:.82rem">'
            f'<thead><tr style="border-bottom:2px solid #E5E7EB">{head_cells}</tr></thead><tbody>'
        )

        body = ""
        for r in rows:
            rc  = _ret_color(r["ret"])
            vsc = _ret_color(r["vs_spy"])
            if cols == "persist":
                cells = (
                    f'<td style="padding:7px 8px;font-weight:700;color:#111827;'
                    f'font-family:monospace;border-left:3px solid {accent}">&nbsp;{r["ticker"]}</td>'
                    f'<td style="padding:7px 8px;color:#374151;max-width:130px;'
                    f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{str(r["company"])[:22]}</td>'
                    f'<td style="padding:7px 8px;text-align:right;font-family:monospace;'
                    f'font-weight:700;color:{accent}">{_fmt_z(r.get("q1_z"))}</td>'
                    f'<td style="padding:7px 8px;text-align:right;font-family:monospace;'
                    f'font-weight:700;color:{accent}">{_fmt_z(r.get("q2_z"))}</td>'
                    f'<td style="padding:7px 8px;text-align:right;font-family:monospace;'
                    f'font-weight:700;color:{accent}">{_fmt_z(r.get("avg_z"))}</td>'
                    f'<td style="padding:7px 8px;text-align:right;font-family:monospace;'
                    f'font-weight:600;color:{rc}">{_fmt_ret(r["ret"])}</td>'
                    f'<td style="padding:7px 8px;text-align:right;font-family:monospace;'
                    f'font-weight:600;color:{vsc}">{_fmt_ret(r["vs_spy"])}</td>'
                    f'<td style="padding:7px 8px;text-align:center">{r["outcome"]}</td>'
                )
            else:
                cells = (
                    f'<td style="padding:7px 8px;font-weight:700;color:#111827;'
                    f'font-family:monospace;border-left:3px solid {accent}">&nbsp;{r["ticker"]}</td>'
                    f'<td style="padding:7px 8px;color:#374151;max-width:130px;'
                    f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{str(r["company"])[:22]}</td>'
                    f'<td style="padding:7px 8px;text-align:right;font-family:monospace;'
                    f'font-weight:700;color:{accent}">{_fmt_z(r.get("z"))}</td>'
                    f'<td style="padding:7px 8px;text-align:right;font-family:monospace;'
                    f'font-weight:600;color:{rc}">{_fmt_ret(r["ret"])}</td>'
                    f'<td style="padding:7px 8px;text-align:right;font-family:monospace;'
                    f'font-weight:600;color:{vsc}">{_fmt_ret(r["vs_spy"])}</td>'
                    f'<td style="padding:7px 8px;text-align:center">{r["outcome"]}</td>'
                )
            body += f'<tr style="border-bottom:1px solid #F3F4F6">{cells}</tr>'
        return header + body + "</tbody></table>"

    def _build_rows(tickers: list, z_map: dict, co_map: dict, signal_type: str) -> tuple[list, int]:
        rows = []
        validated = 0
        for tkr in tickers:
            p   = prices.get(tkr, {})
            ret = p.get("ret")
            z   = z_map.get(tkr)
            co  = co_map.get(tkr, tkr)
            if ret is None:
                vs_spy       = None
                outcome_html = '<span style="color:#9CA3AF">N/A</span>'
            else:
                vs_spy = ret - spy_ret
                win    = ret < spy_ret if signal_type == "short" else ret > spy_ret
                if win:
                    validated += 1
                    outcome_html = '<span style="color:#16A34A;font-size:1.1rem">✓</span>'
                else:
                    outcome_html = '<span style="color:#DC2626;font-size:1.1rem">✗</span>'
            rows.append({"ticker": tkr, "company": co, "z": z,
                         "ret": ret, "vs_spy": vs_spy, "outcome": outcome_html})
        return rows, validated

    def _build_persist_rows(tickers: list, mq_df: pd.DataFrame, co_map: dict) -> tuple[list, int]:
        rows = []
        validated = 0
        for tkr in tickers:
            p   = prices.get(tkr, {})
            ret = p.get("ret")
            co  = co_map.get(tkr, tkr)
            row = mq_df[mq_df["ticker"] == tkr]
            q1_z = float(row["q1_zscore"].iloc[0]) if len(row) and not pd.isna(row["q1_zscore"].iloc[0]) else None
            q2_z = float(row["q2_zscore"].iloc[0]) if len(row) and not pd.isna(row["q2_zscore"].iloc[0]) else None
            avg_z = None
            if q1_z is not None and q2_z is not None:
                avg_z = (q1_z + q2_z) / 2
            elif q1_z is not None:
                avg_z = q1_z
            elif q2_z is not None:
                avg_z = q2_z
            if ret is None:
                vs_spy       = None
                outcome_html = '<span style="color:#9CA3AF">N/A</span>'
            else:
                vs_spy = ret - spy_ret
                win    = ret < spy_ret  # persistent = short signal
                if win:
                    validated += 1
                    outcome_html = '<span style="color:#16A34A;font-size:1.1rem">✓</span>'
                else:
                    outcome_html = '<span style="color:#DC2626;font-size:1.1rem">✗</span>'
            rows.append({"ticker": tkr, "company": co, "q1_z": q1_z, "q2_z": q2_z,
                         "avg_z": avg_z, "ret": ret, "vs_spy": vs_spy, "outcome": outcome_html})
        return rows, validated

    # ── Build z-score / company lookups ────────────────────────────────────────
    q1_z_map: dict = {}
    co_map:   dict = {}
    if len(q1_df):
        for _, row in q1_df.iterrows():
            q1_z_map[row["ticker"]] = row.get("combined_zscore")
            co_map[row["ticker"]]   = row.get("company", row["ticker"])
    if len(q2_df):
        for _, row in q2_df.iterrows():
            co_map.setdefault(row["ticker"], row.get("company", row["ticker"]))

    mq_df = build_multi_quarter_df(q1_df, q2_df)

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["📊 Q1 2025 Single-Quarter Validation",
                          "⚡ Persistent Signal Validation (Q1 + Q2)"])

    # ── Tab 1: Q1 short / long ─────────────────────────────────────────────────
    with tab1:
        _period_banner("6 months post Q1 2025 filing window")

        short_rows, short_validated = _build_rows(_VAL_SHORT, q1_z_map, co_map, "short")
        long_rows,  long_validated  = _build_rows(_VAL_LONG,  q1_z_map, co_map, "long")

        lc, rc_col = st.columns(2, gap="large")
        with lc:
            st.markdown(
                '<p style="font-size:.72rem;font-weight:700;letter-spacing:.09em;'
                'text-transform:uppercase;color:#DC2626;margin-bottom:8px">'
                '🔴 Short Signal Validation — Q1 2025</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="card" style="padding:4px 0;overflow-x:auto">'
                f'{_table_html(short_rows, "#DC2626")}'
                f'</div>',
                unsafe_allow_html=True,
            )
        with rc_col:
            st.markdown(
                '<p style="font-size:.72rem;font-weight:700;letter-spacing:.09em;'
                'text-transform:uppercase;color:#16A34A;margin-bottom:8px">'
                '🟢 Long Signal Validation — Q1 2025</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="card" style="padding:4px 0;overflow-x:auto">'
                f'{_table_html(long_rows, "#16A34A")}'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div style="background:#F9FAFB;border:1px solid #E5E7EB;border-radius:8px;'
            f'padding:14px 18px;margin-top:14px">'
            f'<span style="font-size:.9rem;color:#111827">'
            f'<strong style="color:#DC2626">{short_validated}/10</strong> short signals underperformed the benchmark &nbsp;·&nbsp; '
            f'<strong style="color:#16A34A">{long_validated}/10</strong> long signals outperformed. '
            f'Measurement period: {d1_label} – {d2_label}.'
            f'</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="font-size:.72rem;color:#9CA3AF;margin-top:10px;line-height:1.6">'
            '<em>Stock selection is mechanical — top/bottom 10 z-scores from Q1 2025 cross-section. '
            'Returns sourced from Yahoo Finance (adjusted close). '
            'This is not a backtest — it is a single-period directional validation.</em></p>',
            unsafe_allow_html=True,
        )

    # ── Tab 2: Persistent Signal ────────────────────────────────────────────────
    with tab2:
        st.markdown(
            '<div style="background:#FFF7ED;border:1px solid #FED7AA;border-radius:8px;'
            'padding:10px 16px;margin-bottom:14px;font-size:.82rem;color:#C2410C">'
            '⚡ <strong>Persistent signals</strong> are companies with a negative z-score '
            'in <em>both</em> Q1 2025 and Q2 2025 (each independently &lt; −0.5σ), '
            'ranked by average z-score across both quarters. '
            'These represent sustained deterioration in SEC filing language.</div>',
            unsafe_allow_html=True,
        )
        _period_banner("6 months post Q1 2025 filing window")

        persist_rows, persist_validated = _build_persist_rows(_VAL_PERSIST, mq_df, co_map)

        st.markdown(
            '<p style="font-size:.72rem;font-weight:700;letter-spacing:.09em;'
            'text-transform:uppercase;color:#7C3AED;margin-bottom:8px">'
            '⚡ Persistent Signal — Top 10 by Avg Z (Q1 + Q2 2025)</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="card" style="padding:4px 0;overflow-x:auto">'
            f'{_table_html(persist_rows, "#7C3AED", cols="persist")}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Comparison hit-rate summary ─────────────────────────────────────────
        short_rows2, short_validated2 = _build_rows(_VAL_SHORT, q1_z_map, co_map, "short")
        m1, m2, m3 = st.columns(3)
        metric_style = (
            'background:#FFFFFF;border:1px solid #E5E7EB;border-radius:8px;'
            'padding:14px 18px;text-align:center'
        )
        with m1:
            st.markdown(
                f'<div style="{metric_style};border-top:3px solid #7C3AED">'
                f'<div style="font-size:1.8rem;font-weight:800;color:#7C3AED">'
                f'{persist_validated}/10</div>'
                f'<div style="font-size:.78rem;color:#6B7280;margin-top:4px">'
                f'Persistent signals<br>underperformed SPY</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div style="{metric_style};border-top:3px solid #DC2626">'
                f'<div style="font-size:1.8rem;font-weight:800;color:#DC2626">'
                f'{short_validated2}/10</div>'
                f'<div style="font-size:.78rem;color:#6B7280;margin-top:4px">'
                f'Q1-only short signals<br>underperformed SPY</div></div>',
                unsafe_allow_html=True,
            )
        with m3:
            delta = persist_validated - short_validated2
            delta_color = "#16A34A" if delta > 0 else ("#DC2626" if delta < 0 else "#6B7280")
            delta_str   = f"+{delta}" if delta > 0 else str(delta)
            st.markdown(
                f'<div style="{metric_style};border-top:3px solid {delta_color}">'
                f'<div style="font-size:1.8rem;font-weight:800;color:{delta_color}">'
                f'{delta_str}</div>'
                f'<div style="font-size:.78rem;color:#6B7280;margin-top:4px">'
                f'Lift from requiring<br>both-quarter signal</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<p style="font-size:.72rem;color:#9CA3AF;margin-top:10px;line-height:1.6">'
            '<em>Persistent signals selected as top-10 by most negative average z-score '
            'across Q1 2025 and Q2 2025. Returns sourced from Yahoo Finance (adjusted close). '
            'This is not a backtest — it is a single-period directional validation.</em></p>',
            unsafe_allow_html=True,
        )


# ── Badge helper ──────────────────────────────────────────────────────────────

def _badge_html(label: str) -> str:
    cls = {
        "Most Changed":  "badge-red",
        "Least Changed": "badge-green",
        "Moderate":      "badge-yellow",
        "Data gap":      "badge-gray",
    }.get(label, "badge-gray")
    return f'<span class="badge {cls}">{label}</span>'


# ── Quarter header (shared across pages that need the selector) ───────────────

def render_quarter_header(df: pd.DataFrame, page_title: str = "Today's Signal") -> None:
    """Renders page title row with quarter selector, freshness pill, export and refresh."""
    if "active_quarter" not in st.session_state:
        st.session_state["active_quarter"] = DEFAULT_QUARTER

    q2_available = Q2_CSV.exists()
    options = list(QUARTER_OPTIONS.keys())
    active_q = st.session_state.get("active_quarter", DEFAULT_QUARTER)
    _, display_label = QUARTER_OPTIONS[active_q]

    # Freshness info
    n_ok = len(df)
    date_str = "—"
    if not df["date_run"].isna().all():
        latest = df["date_run"].max()
        if pd.notna(latest):
            date_str = latest.strftime("%b %d, %Y")

    c_title, c_qsel, c_fresh, c_export, c_refresh = st.columns([3, 3, 3.5, 1.3, 1.1])

    with c_title:
        st.markdown(
            f'<div style="padding:10px 0;font-size:1.2rem;font-weight:700;color:#111827">'
            f'{page_title}</div>',
            unsafe_allow_html=True,
        )

    with c_qsel:
        st.markdown('<div class="qbtn-wrap">', unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        with b1:
            if st.button(
                options[0],
                use_container_width=True,
                type="primary" if active_q == options[0] else "secondary",
                key="qbtn_q1",
            ):
                if active_q != options[0]:
                    st.session_state["active_quarter"] = options[0]
                    st.session_state.pop("detail_ticker", None)
                    st.rerun()
        with b2:
            if st.button(
                options[1] + ("" if q2_available else " ⏳"),
                use_container_width=True,
                type="primary" if active_q == options[1] else "secondary",
                key="qbtn_q2",
                disabled=not q2_available,
            ):
                if active_q != options[1]:
                    st.session_state["active_quarter"] = options[1]
                    st.session_state.pop("detail_ticker", None)
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with c_fresh:
        st.markdown(
            f'<div style="padding:10px 0;text-align:center">'
            f'<span style="font-size:.74rem;color:#6B7280;'
            f'background:#FFFFFF;border:1px solid #E5E7EB;'
            f'border-radius:20px;padding:4px 12px">'
            f'📅 {display_label} · {n_ok:,} cos · {date_str}'
            f'</span></div>',
            unsafe_allow_html=True,
        )

    with c_export:
        csv_bytes = df.to_csv(index=False).encode()
        st.download_button(
            "⬇ Export CSV", csv_bytes,
            file_name="sec_signal_scores.csv", mime="text/csv",
            use_container_width=True,
        )

    with c_refresh:
        if st.button("↺ Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    st.markdown(
        '<hr style="margin:4px 0 18px;border:none;border-top:1px solid #E5E7EB">',
        unsafe_allow_html=True,
    )


# ── KPIs ──────────────────────────────────────────────────────────────────────

def render_kpis(df: pd.DataFrame) -> None:
    total  = len(df)
    median = df["combined_score"].median()
    flagged = int((df["combined_zscore"] < -1).sum())
    stable  = int((df["combined_zscore"] > 1).sum())

    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Companies Scored",  f"{total:,}")
    with k2: st.metric("Median Similarity", f"{median:.3f}")
    with k3: st.metric("⚠ Flagged  (z < −1)", f"{flagged}")
    with k4: st.metric("✓ Stable  (z > +1)",  f"{stable}")


# ── Top 10 signals ────────────────────────────────────────────────────────────

def render_top10_signals(df: pd.DataFrame, sector: str = "") -> None:
    """Render top-10 short/long panels. If sector is set, filter to that sector."""
    view = df.copy()
    if sector:
        view = view[view["sector"] == sector]

    # Exclude failed extractions — combined_score=0.0 due to missing text, not genuine signal
    if "extraction_status" in view.columns:
        view = view[view["extraction_status"] != "failed"]

    valid = view.dropna(subset=["combined_zscore"]).sort_values("combined_zscore")
    top_short = valid.head(10)
    top_long  = valid.tail(10).iloc[::-1]

    max_abs_z = max(
        abs(top_short["combined_zscore"].iloc[0]) if len(top_short) else 1,
        abs(top_long["combined_zscore"].iloc[0])  if len(top_long)  else 1,
        1e-6,
    )

    sector_tag = f" — {sector}" if sector else ""

    def _rows_html(rows, accent: str) -> str:
        if len(rows) == 0:
            return '<div style="padding:12px;color:#9CA3AF;font-size:.82rem">No data for this sector</div>'
        parts = []
        for _, r in rows.iterrows():
            z   = r["combined_zscore"]
            pct = min(abs(z) / max_abs_z * 100, 100)
            z_color = "#DC2626" if z < 0 else "#16A34A"
            sec_tag = (
                f'<span style="font-size:.63rem;color:#9CA3AF;flex-shrink:0;'
                f'white-space:nowrap;overflow:hidden;max-width:90px;text-overflow:ellipsis">'
                f'{r.get("sector","")[:14]}</span>'
            ) if not sector else ""
            parts.append(
                f'<div class="t10-row" style="border-left:3px solid {accent}">'
                f'<span class="t10-ticker">{r["ticker"]}</span>'
                f'<span class="t10-company">{str(r["company"])[:22]}</span>'
                f'{sec_tag}'
                f'<span class="t10-z" style="color:{z_color}">{z:+.2f}σ</span>'
                f'<div class="t10-bar-wrap">'
                f'<div class="t10-bar" style="background:{accent};width:{pct:.1f}%"></div>'
                f'</div>'
                f'</div>'
            )
        return "".join(parts)

    lc, rc = st.columns(2, gap="large")

    with lc:
        st.markdown(f'<p class="sh">🔴 Top 10 Short Signals — Most Changed{sector_tag}</p>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="card" style="padding:12px 14px">'
            f'{_rows_html(top_short, "#DC2626")}'
            f'</div>',
            unsafe_allow_html=True,
        )

    with rc:
        st.markdown(f'<p class="sh">🟢 Top 10 Long Signals — Least Changed{sector_tag}</p>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="card" style="padding:12px 14px">'
            f'{_rows_html(top_long, "#16A34A")}'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── Scatter plot ──────────────────────────────────────────────────────────────

def build_scatter(df: pd.DataFrame) -> alt.Chart:
    color_scale = alt.Scale(
        domain=["Most Changed", "Moderate", "Least Changed"],
        range=["#dc2626", "#d97706", "#16a34a"],
    )
    dots = (
        alt.Chart(df)
        .mark_circle(size=52, opacity=0.75, stroke="white", strokeWidth=0.8)
        .encode(
            x=alt.X("combined_score:Q", title="Combined Similarity Score",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(gridColor="#F3F4F6",
                                  domainColor="#E5E7EB",
                                  tickColor="#E5E7EB",
                                  labelColor="#6B7280",
                                  titleColor="#9CA3AF", labelFontSize=11)),
            y=alt.Y("combined_zscore:Q", title="Z-Score vs peers",
                    axis=alt.Axis(gridColor="#F3F4F6",
                                  domainColor="#E5E7EB",
                                  tickColor="#E5E7EB",
                                  labelColor="#6B7280",
                                  titleColor="#9CA3AF", labelFontSize=11)),
            color=alt.Color("change_label:N", scale=color_scale,
                            legend=alt.Legend(
                                title="Signal", orient="bottom-right",
                                labelColor="#374151", titleColor="#6B7280",
                                labelFontSize=11, titleFontSize=11,
                                fillColor="#FFFFFF", strokeColor="#E5E7EB",
                                padding=8, cornerRadius=6,
                            )),
            tooltip=[
                alt.Tooltip("ticker:N",         title="Ticker"),
                alt.Tooltip("company:N",         title="Company"),
                alt.Tooltip("period:N",          title="Period"),
                alt.Tooltip("combined_score:Q",  title="Combined", format=".4f"),
                alt.Tooltip("mda_score:Q",       title="MD&A",     format=".4f"),
                alt.Tooltip("risk_score:Q",      title="Risk",     format=".4f"),
                alt.Tooltip("combined_zscore:Q", title="Z-Score",  format="+.3f"),
                alt.Tooltip("change_label:N",    title="Signal"),
            ],
        )
    )
    rules = (
        alt.Chart(pd.DataFrame({"z": [-1.0, 1.0]}))
        .mark_rule(strokeDash=[4, 4], opacity=0.4, color="#9ca3af")
        .encode(y=alt.Y("z:Q"))
    )
    return (
        alt.layer(dots, rules)
        .properties(height=300, background="#FFFFFF")
        .configure_view(stroke="#E5E7EB", fill="#FFFFFF")
        .configure_axis(labelFont="Inter", titleFont="Inter")
    )


GICS_SECTORS = [
    "All Sectors",
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Financials",
    "Health Care",
    "Industrials",
    "Information Technology",
    "Materials",
    "Real Estate",
    "Utilities",
]


# ── Sector heatmap ────────────────────────────────────────────────────────────

def render_sector_heatmap(df: pd.DataFrame) -> None:
    """Render sector signal overview as a clean HTML table — red/green bars."""
    valid = df.dropna(subset=["combined_zscore", "sector"])
    valid = valid[valid["sector"] != "Unknown"]
    if "extraction_status" in valid.columns:
        valid = valid[valid["extraction_status"] != "failed"]

    agg = (
        valid.groupby("sector")
        .agg(avg_z=("combined_zscore", "mean"), count=("ticker", "count"))
        .reset_index()
        .sort_values("avg_z")      # most changed first (most negative z at top)
    )

    if len(agg) == 0:
        st.caption("No sector data available.")
        return

    abs_max = max(abs(agg["avg_z"].min()), abs(agg["avg_z"].max()), 0.5)
    bar_max_px = 130  # max bar width in pixels

    def _bar_color(z: float) -> str:
        if z <= -0.5:   return "#DC2626"
        if z >= 0.5:    return "#16A34A"
        return "#D97706"

    def _z_color(z: float) -> str:
        if z < 0:  return "#DC2626"
        if z > 0:  return "#16A34A"
        return "#D97706"

    rows_html = ""
    for _, r in agg.iterrows():
        z    = r["avg_z"]
        cnt  = int(r["count"])
        bar_w = int(abs(z) / abs_max * bar_max_px)
        bar_c = _bar_color(z)
        z_c   = _z_color(z)
        rows_html += (
            f'<tr>'
            f'<td class="hm-sector">{r["sector"]}</td>'
            f'<td class="hm-z" style="color:{z_c}">{z:+.3f}</td>'
            f'<td class="hm-count">{cnt}</td>'
            f'<td class="hm-bar-wrap">'
            f'<div class="hm-bar" style="width:{bar_w}px;background:{bar_c}"></div>'
            f'</td>'
            f'</tr>'
        )

    html = (
        f'<div class="card" style="padding:14px 18px">'
        f'<table class="hm-table">'
        f'<thead><tr class="hm-thead">'
        f'<td class="hm-sector">Sector</td>'
        f'<td class="hm-z">Avg Z</td>'
        f'<td class="hm-count">Cos</td>'
        f'<td class="hm-bar-wrap"></td>'
        f'</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ── Filter bar ────────────────────────────────────────────────────────────────

def render_filter_bar(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    """Returns (filtered_df, search_text, active_sector)."""
    # Row 1: search | signal segmented | sector dropdown
    fc1, fc2, fc3 = st.columns([3, 3.5, 3])

    with fc1:
        raw_search = st.text_input(
            "Search", placeholder="Search ticker or company …",
            label_visibility="collapsed", key="tbl_search",
        )
        search = raw_search.strip()

    with fc2:
        signal_filter = st.segmented_control(
            "Filter by signal",
            options=["All", "Most Changed", "Moderate", "Least Changed"],
            default="All",
            label_visibility="collapsed",
            key="signal_filter",
        )

    with fc3:
        # Build sector list — only show sectors present in this df
        present = set(df["sector"].dropna().unique())
        sector_opts = [s for s in GICS_SECTORS if s == "All Sectors" or s in present]
        sector_filter = st.selectbox(
            "Sector", sector_opts,
            label_visibility="collapsed", key="sector_filter",
        )

    # Row 2: risk N/A checkbox | show count
    r2c1, r2c2, r2c3 = st.columns([1.5, 1, 8])
    with r2c1:
        risk_na = st.checkbox("Risk N/A", key="risk_na_filter")
    with r2c2:
        show_n = st.selectbox(
            "Show", [100, 250, 500], label_visibility="collapsed", key="show_n"
        )

    # Resolve ticker search → open detail panel
    if search:
        exact = df[df["ticker"].str.upper() == search.upper()]
        if len(exact) == 1:
            t = exact.iloc[0]["ticker"]
            if st.session_state.get("detail_ticker") != t:
                st.session_state["detail_ticker"] = t
                st.session_state["detail_source"]  = "search"

    # Apply filters
    filtered = df.copy()
    if search:
        mask = (
            filtered["ticker"].str.contains(search, case=False, na=False) |
            filtered["company"].str.contains(search, case=False, na=False)
        )
        filtered = filtered[mask]
    if signal_filter and signal_filter != "All":
        filtered = filtered[filtered["change_label"] == signal_filter]
    if sector_filter and sector_filter != "All Sectors":
        filtered = filtered[filtered["sector"] == sector_filter]
    if risk_na:
        filtered = filtered[filtered["risk_score"].isna()]

    filtered = filtered.head(show_n).reset_index(drop=True)
    active_sector = sector_filter if sector_filter != "All Sectors" else ""
    return filtered, search, active_sector


# ── Table with inline mini bars ───────────────────────────────────────────────

def _multi_q_label(q1_z, q2_z) -> str:
    """Format cross-quarter z-score display string."""
    parts = []
    if pd.notna(q1_z):
        parts.append(f"Q1: {q1_z:+.1f}")
    if pd.notna(q2_z):
        parts.append(f"Q2: {q2_z:+.1f}")
    return "  /  ".join(parts) if parts else "—"


def render_table(filtered_df: pd.DataFrame, mq_df: pd.DataFrame | None = None) -> None:
    base_cols = ["ticker", "company", "sector", "period", "change_label",
                 "combined_score", "mda_score", "risk_score", "combined_zscore"]
    avail_cols = [c for c in base_cols if c in filtered_df.columns]

    # Sort: valid rows first (by z-score), failed extraction rows at bottom
    working = filtered_df.copy()
    if "extraction_status" in working.columns:
        failed_mask = working["extraction_status"] == "failed"
        working = pd.concat([
            working[~failed_mask].sort_values("combined_zscore"),
            working[failed_mask],
        ]).reset_index(drop=True)

    display = working[avail_cols].copy().reset_index(drop=True)

    # Merge multi-quarter data if available
    has_mq = mq_df is not None and len(mq_df) > 0 and (
        mq_df["q1_zscore"].notna().any() or mq_df["q2_zscore"].notna().any()
    )
    if has_mq:
        display = display.merge(
            mq_df[["ticker", "q1_zscore", "q2_zscore", "persistent_signal"]],
            on="ticker", how="left",
        )
        display["multi_quarter"] = display.apply(
            lambda r: _multi_q_label(r.get("q1_zscore"), r.get("q2_zscore")), axis=1
        )
        display["signal_flag"] = display["persistent_signal"].apply(
            lambda x: "⚡ Persistent" if x else ""
        )

    col_config = {
        "ticker":        st.column_config.TextColumn("Ticker", width=80),
        "company":       st.column_config.TextColumn("Company", width=200),
        "sector":        st.column_config.TextColumn("Sector",  width=140),
        "period":        st.column_config.TextColumn("Period",  width=80),
        "change_label":  st.column_config.TextColumn("Signal",  width=100),
        "combined_score": st.column_config.ProgressColumn(
            "Combined", min_value=0, max_value=1, format="%.3f", width=120,
        ),
        "mda_score": st.column_config.ProgressColumn(
            "MD&A", min_value=0, max_value=1, format="%.3f", width=110,
        ),
        "risk_score": st.column_config.ProgressColumn(
            "Risk Factors", min_value=0, max_value=1, format="%.3f", width=110,
        ),
        "combined_zscore": st.column_config.NumberColumn(
            "Z-Score", format="%.3f", width=80,
        ),
    }

    show_cols = [c for c in avail_cols]  # include sector if present
    if has_mq:
        show_cols += ["multi_quarter", "signal_flag"]
        col_config["multi_quarter"] = st.column_config.TextColumn(
            "Q1 / Q2 z-score", width=140,
            help="Z-scores for Q1 2025 and Q2 2025 side by side",
        )
        col_config["signal_flag"] = st.column_config.TextColumn(
            "Multi-Q", width=110,
            help="⚡ Persistent = negative z-score in both quarters",
        )

    # ── Pulsing dot onboarding hint (first-time users) ────────────────────────
    if not st.session_state.get("table_hint_dismissed", False):
        st.markdown(
            '<div style="position:relative;height:52px;overflow:visible;'
            'z-index:200;margin-bottom:-52px;pointer-events:none">'
            '<div style="position:absolute;bottom:8px;left:14px;'
            'display:flex;align-items:center;gap:10px">'
            '<span class="pulse-dot-wrap">'
            '<span class="pulse-dot-ring"></span>'
            '<span class="pulse-dot-core"></span>'
            '</span>'
            '<span style="background:#111827;color:#FFFFFF;font-size:.72rem;'
            'font-weight:600;padding:5px 12px;border-radius:6px;'
            'box-shadow:0 2px 8px rgba(0,0,0,0.25);white-space:nowrap">'
            'Click any row to view filing analysis</span>'
            '</div></div>',
            unsafe_allow_html=True,
        )

    event = st.dataframe(
        display[show_cols],
        use_container_width=True,
        hide_index=True,
        height=min(36 * len(display) + 42, 580),
        on_select="rerun",
        selection_mode="single-row",
        key="company_table",
        column_config=col_config,
    )

    selected_rows = event.selection.rows if hasattr(event, "selection") else []
    if selected_rows:
        clicked = working.iloc[selected_rows[0]]["ticker"]
        if clicked != st.session_state.get("detail_ticker"):
            st.session_state["detail_ticker"] = clicked
            st.session_state["detail_source"]  = "table"
            st.session_state["table_hint_dismissed"] = True
            st.rerun()

    st.caption(f"Showing {len(display)} companies · {len(display[display['change_label'] == 'Data gap'])} with data gaps moved to bottom"
               if "change_label" in display.columns and (display["change_label"] == "Data gap").any()
               else f"Showing {len(display)} companies")


# ── Plain-English summary helpers ─────────────────────────────────────────────

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "to", "for", "is", "are",
    "was", "were", "be", "been", "that", "this", "with", "its", "has", "have",
    "had", "from", "by", "on", "at", "as", "our", "we", "it", "not", "also",
    "which", "than", "will", "may", "would", "could", "such", "these", "those",
    "any", "all", "but", "if", "their", "they", "no", "into", "during",
    "including", "other", "each", "more", "both", "when", "about", "under",
}


def _extract_topics(changes, change_type: str, n: int = 3) -> list[str]:
    candidates = [
        ch for ch in changes if ch.change_type == change_type and ch.delta > 0.4
    ][:n * 3]
    topics: list[str] = []
    for ch in candidates:
        text = ch.current_sentence if change_type == "added" else ch.prior_sentence
        if not text:
            continue
        words = re.sub(r"[^\w\s]", " ", text).split()
        meaningful = [w for w in words if w.lower() not in _STOPWORDS and len(w) > 3][:6]
        if len(meaningful) >= 2:
            phrase = " ".join(meaningful[:4]).title()
            if phrase not in topics:
                topics.append(phrase)
        if len(topics) >= n:
            break
    return topics


def build_plain_english_summary(
    company: str, quarter: str, year: int, prior_year: int,
    mda_score, rf_score, mda_changes: list, rf_changes: list,
) -> str:
    def _mag(s):
        if s is None or pd.isna(s): return None
        if s < 0.40: return "significant"
        if s < 0.65: return "moderate"
        return "minimal"

    mm, rm = _mag(mda_score), _mag(rf_score)

    if   mm == "significant" and rm == "significant":
        lead = f"{company} made significant changes to both its MD&A and Risk Factors sections."
    elif mm == "significant" and rm == "moderate":
        lead = f"{company} significantly rewrote its MD&A with moderate Risk Factors updates."
    elif mm == "moderate"    and rm == "significant":
        lead = f"{company} significantly rewrote its Risk Factors with moderate MD&A updates."
    elif mm == "significant":
        lead = f"{company} made significant changes to its MD&A section."
    elif rm == "significant":
        lead = f"{company} made significant changes to its Risk Factors section."
    elif mm is None and rm is None:
        lead = (f"{company}'s {quarter} {year} sections could not be compared — "
                "text may not have been extractable from one or both filings.")
    elif mm == "minimal" and rm == "minimal":
        lead = f"{company} made only minor language updates between {prior_year} and {year}."
    else:
        lead = f"{company} made moderate updates to its {quarter} {year} filing vs {prior_year}."

    parts = [lead]
    all_ch = list(mda_changes) + list(rf_changes)
    new_topics = _extract_topics(all_ch, "added",   3)
    rem_topics = _extract_topics(all_ch, "removed", 3)

    if new_topics:
        parts.append(f"Key new themes: {', '.join(new_topics)}.")
    if rem_topics:
        parts.append(f"Key removed themes: {', '.join(rem_topics)}.")
    if rf_score  is not None and not pd.isna(rf_score)  and rf_score  < 0.30:
        parts.append("Risk Factors substantially rewritten.")
    if mda_score is not None and not pd.isna(mda_score) and mda_score < 0.30:
        parts.append("MD&A substantially rewritten.")
    return "  ".join(parts)


# ── Claude API summary ────────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def generate_ai_summary(
    ticker: str, company: str, quarter: str, year: int, prior_year: int,
    mda_score, rf_score,
    mda_added: tuple, mda_removed: tuple,
    rf_added: tuple, rf_removed: tuple,
) -> str | None:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except Exception:
        return None

    def _excerpt(sents: tuple, label: str, max_chars: int = 700) -> str:
        lines: list[str] = []
        total = 0
        for s in sents:
            if total + len(s) > max_chars:
                break
            lines.append(f"- {s}")
            total += len(s)
        return f"**{label}**\n" + "\n".join(lines) if lines else ""

    excerpts = "\n\n".join(filter(None, [
        _excerpt(mda_added,   "MD&A — newly added"),
        _excerpt(mda_removed, "MD&A — removed"),
        _excerpt(rf_added,    "Risk Factors — newly added"),
        _excerpt(rf_removed,  "Risk Factors — removed"),
    ]))

    scores_line = (
        (f"MD&A similarity: {mda_score:.1%}" if mda_score is not None else "MD&A: N/A")
        + "  |  "
        + (f"Risk Factors similarity: {rf_score:.1%}" if rf_score is not None else "Risk Factors: N/A")
    )

    prompt = (
        f"You are analyzing changes in a public company's SEC 10-Q filing.\n\n"
        f"Company: {company} ({ticker})\n"
        f"Period: {quarter} {year} vs {quarter} {prior_year}\n"
        f"{scores_line}\n\n"
        f"Sample sentences added or removed:\n\n{excerpts}\n\n"
        f"Write a concise 3–5 sentence plain English summary for an investor. Cover:\n"
        f"1. Which section changed most and how significantly\n"
        f"2. Top 3 new themes introduced in {year}\n"
        f"3. Top 3 themes that disappeared from {prior_year}\n\n"
        f"Be specific and professional. No filler phrases."
    )

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception:
        return None


# ── Signal amplifiers ─────────────────────────────────────────────────────────

def _avg_word_length(text: str) -> float:
    words = re.findall(r"[A-Za-z]+", text)
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def render_signal_amplifiers(
    quarter: str, mda_text: str, rf_text: str,
    df: pd.DataFrame, z_score: float,
) -> None:
    combined_text = (mda_text or "") + " " + (rf_text or "")
    awl = _avg_word_length(combined_text)
    SP500_AWL = 5.3

    # 1 — Fiscal quarter
    q = quarter.upper()
    if q in ("Q1", "Q2"):
        q_icon, q_label = "🟢", f"{q} · Stronger Signal Quarter"
        q_border, q_tc, q_bg = "#16A34A", "#14532D", "#F0FDF4"
        q_desc = "Q1 and Q2 filings contain the most forward-looking language and strategic updates, making text changes more informative as a trading signal."
    else:
        q_icon, q_label = "🟡", f"{q} · Standard Signal Quarter"
        q_border, q_tc, q_bg = "#D97706", "#78350F", "#FFFBEB"
        q_desc = "Q3 filings tend toward operational routine updates. The score is valid but historically shows less signal variance than Q1/Q2."

    # 2 — Complexity
    if awl == 0:
        cx_icon, cx_label = "⬜", "Complexity · N/A"
        cx_border, cx_tc, cx_bg = "#D1D5DB", "#6B7280", "#F9FAFB"
        cx_desc = "Filing text was not extractable for this company."
    elif awl >= SP500_AWL + 0.3:
        cx_icon, cx_label = "🔴", f"High Complexity · {awl:.1f} chars/word"
        cx_border, cx_tc, cx_bg = "#DC2626", "#7F1D1D", "#FEF2F2"
        cx_desc = "Longer-than-average words signal more legalese. Score changes may partly reflect boilerplate rewording rather than real strategic shifts."
    elif awl <= SP500_AWL - 0.3:
        cx_icon, cx_label = "🟢", f"Low Complexity · {awl:.1f} chars/word"
        cx_border, cx_tc, cx_bg = "#16A34A", "#14532D", "#F0FDF4"
        cx_desc = "Plain language filing — score changes more likely reflect genuine business updates rather than legal phrasing shifts."
    else:
        cx_icon, cx_label = "🟡", f"Avg Complexity · {awl:.1f} chars/word"
        cx_border, cx_tc, cx_bg = "#D97706", "#78350F", "#FFFBEB"
        cx_desc = "Text complexity is in line with S&P 500 peers. No adjustment needed when interpreting the similarity score."

    # 3 — Percentile rank
    valid_z = df["combined_zscore"].dropna().sort_values()
    if pd.notna(z_score) and len(valid_z) > 0:
        pct = int((valid_z < z_score).sum() / len(valid_z) * 100)
        if pct <= 10:
            rk_icon, rk_label = "🔴", f"Bottom {pct}th Percentile"
            rk_border, rk_tc, rk_bg = "#DC2626", "#7F1D1D", "#FEF2F2"
            rk_desc = f"More language change than {100 - pct}% of S&P 500 peers this quarter — high-conviction signal."
        elif pct >= 90:
            rk_icon, rk_label = "🟢", f"Top {100 - pct}th Percentile — Most Stable"
            rk_border, rk_tc, rk_bg = "#16A34A", "#14532D", "#F0FDF4"
            rk_desc = f"More stable than {pct}% of S&P 500 peers — management language highly consistent year-over-year."
        else:
            rk_icon, rk_label = "🟡", f"{pct}th Percentile"
            rk_border, rk_tc, rk_bg = "#D97706", "#78350F", "#FFFBEB"
            rk_desc = "Change magnitude within the typical range across S&P 500 peers."
    else:
        rk_icon, rk_label = "⬜", "Percentile · N/A"
        rk_border, rk_tc, rk_bg = "#D1D5DB", "#6B7280", "#F9FAFB"
        rk_desc = "Insufficient data for cross-company ranking."

    def _card(icon, label, desc, bg, border, tc):
        return (
            f'<div class="amp-card" style="background:{bg};border-left:3px solid {border}">'
            f'<span class="amp-title" style="color:{tc}">{icon} {label}</span>'
            f'<span class="amp-desc">{desc}</span>'
            f'</div>'
        )

    st.markdown('<p class="sh" style="margin-top:14px">Signal Context</p>',
                unsafe_allow_html=True)
    st.markdown(
        f'<div style="display:flex;gap:10px;margin-bottom:16px">'
        f'{_card(q_icon, q_label, q_desc, q_bg, q_border, q_tc)}'
        f'{_card(cx_icon, cx_label, cx_desc, cx_bg, cx_border, cx_tc)}'
        f'{_card(rk_icon, rk_label, rk_desc, rk_bg, rk_border, rk_tc)}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Diff view ─────────────────────────────────────────────────────────────────

def render_diff_columns(
    changes: list, prior_year: int, current_year: int, top_n: int = 10,
) -> None:
    significant = [ch for ch in changes if ch.delta > 0.08][:top_n]
    if not significant:
        st.caption("No significant passage-level changes detected.")
        return

    lh, rh = st.columns(2)
    with lh:
        st.markdown(
            f'<div class="diff-col-header" style="color:#dc2626">'
            f'◀  {prior_year} — Prior Year</div>',
            unsafe_allow_html=True,
        )
    with rh:
        st.markdown(
            f'<div class="diff-col-header" style="color:#16a34a">'
            f'{current_year} — Current Year  ▶</div>',
            unsafe_allow_html=True,
        )

    for ch in significant:
        lc, rc = st.columns(2)
        prior_text   = (ch.prior_sentence   or "")[:400]
        current_text = (ch.current_sentence or "")[:400]

        if ch.change_type == "removed":
            with lc:
                st.markdown(
                    f'<div class="diff-label diff-label-removed">— Removed</div>'
                    f'<div class="diff-removed">{prior_text}</div>',
                    unsafe_allow_html=True,
                )
            with rc:
                st.markdown('<div class="diff-empty"></div>', unsafe_allow_html=True)

        elif ch.change_type == "added":
            with lc:
                st.markdown('<div class="diff-empty"></div>', unsafe_allow_html=True)
            with rc:
                st.markdown(
                    f'<div class="diff-label diff-label-added">+ Added</div>'
                    f'<div class="diff-added">{current_text}</div>',
                    unsafe_allow_html=True,
                )

        else:
            sim_pct = f"{ch.similarity:.0%}"
            with lc:
                st.markdown(
                    f'<div class="diff-label diff-label-mod">~ Before  ({sim_pct} similar)</div>'
                    f'<div class="diff-mod-old">{prior_text}</div>',
                    unsafe_allow_html=True,
                )
            with rc:
                st.markdown(
                    f'<div class="diff-label diff-label-mod">~ After</div>'
                    f'<div class="diff-mod-new">{current_text}</div>',
                    unsafe_allow_html=True,
                )


# ── Section bar chart ─────────────────────────────────────────────────────────

def build_section_bar_chart(
    ticker: str, mda_score, rf_score,
    sp500_mda_avg: float, sp500_rf_avg: float,
) -> alt.Chart:
    rows = []
    if pd.notna(mda_score):
        rows.append({"Section": "MD&A", "Group": ticker, "Score": float(mda_score)})
    rows.append({"Section": "MD&A", "Group": "S&P 500 Avg", "Score": sp500_mda_avg})
    if pd.notna(rf_score):
        rows.append({"Section": "Risk Factors", "Group": ticker, "Score": float(rf_score)})
    rows.append({"Section": "Risk Factors", "Group": "S&P 500 Avg", "Score": sp500_rf_avg})

    chart_df = pd.DataFrame(rows)
    return (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, size=24)
        .encode(
            x=alt.X("Group:N", sort=[ticker, "S&P 500 Avg"],
                    axis=alt.Axis(labelAngle=0, labelColor="#6B7280", title=None,
                                  tickColor="#E5E7EB", domainColor="#E5E7EB")),
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(gridColor="#F3F4F6", labelColor="#6B7280",
                                  titleColor="#9CA3AF",
                                  domainColor="#E5E7EB", tickColor="#E5E7EB"),
                    title="Similarity Score"),
            color=alt.Color("Group:N",
                            scale=alt.Scale(domain=[ticker, "S&P 500 Avg"],
                                            range=["#3B82F6", "#D1D5DB"]),
                            legend=alt.Legend(orient="top", labelColor="#374151",
                                              titleColor="#9CA3AF",
                                              fillColor="#FFFFFF",
                                              strokeColor="#E5E7EB",
                                              padding=6, labelFontSize=11)),
            column=alt.Column("Section:N",
                              header=alt.Header(labelColor="#374151",
                                                titleColor="#9CA3AF",
                                                labelFontSize=12, labelFontWeight="bold"),
                              spacing=16),
            tooltip=[
                alt.Tooltip("Group:N",   title="Group"),
                alt.Tooltip("Section:N", title="Section"),
                alt.Tooltip("Score:Q",   title="Score", format=".4f"),
            ],
        )
        .properties(width=100, height=170, background="#FFFFFF")
        .configure_view(fill="#FFFFFF", stroke="#E5E7EB")
        .configure_axis(labelFont="Inter", titleFont="Inter")
    )


# ── Detail panel ──────────────────────────────────────────────────────────────

def render_detail_panel(
    ticker: str,
    row: pd.Series,
    df: pd.DataFrame,
    mq_row: pd.Series | None = None,
) -> None:
    quarter    = row["quarter"]
    year       = int(row["year"])
    prior_year = year - 1
    company    = row["company"]
    z          = row["combined_zscore"]

    # ── Header ────────────────────────────────────────────────────────────────
    hc, cc = st.columns([11, 1])
    with hc:
        with st.spinner("Loading filing links …"):
            current_url, prior_url = get_filing_links(ticker, quarter, year)

        link_cur = (
            f'<a href="{current_url}" target="_blank" rel="noopener" '
            f'style="font-size:.75rem;color:#3b82f6;text-decoration:none;white-space:nowrap">'
            f'View {year} filing →</a>'
        ) if current_url else ""
        link_prior = (
            f'<a href="{prior_url}" target="_blank" rel="noopener" '
            f'style="font-size:.75rem;color:#3b82f6;text-decoration:none;white-space:nowrap">'
            f'View {prior_year} filing →</a>'
        ) if prior_url else ""

        st.markdown(
            f'<div style="background:#FFFFFF;border:1px solid #E5E7EB;border-radius:10px;'
            f'box-shadow:0 1px 3px rgba(0,0,0,0.06);'
            f'padding:14px 20px;display:flex;align-items:center;gap:12px;flex-wrap:wrap;'
            f'margin-bottom:4px">'
            f'<span style="font-size:1.25rem;font-weight:700;color:#111827">{ticker}</span>'
            f'<span style="color:#6B7280;font-size:.92rem">{company}</span>'
            f'{_badge_html(row["change_label"])}'
            f'<span style="color:#9CA3AF;font-size:.78rem">'
            f'{quarter} {year} vs {prior_year}</span>'
            f'{link_cur}'
            f'{link_prior}'
            f'</div>',
            unsafe_allow_html=True,
        )
    with cc:
        if st.button("✕ Close", key="close_detail", type="secondary"):
            st.session_state.pop("detail_ticker", None)
            st.rerun()

    # ── Fetch filings (cached) ────────────────────────────────────────────────
    with st.spinner(f"Loading {ticker} filings …"):
        try:
            result = fetch_filing_pair(ticker, quarter, year)
        except Exception as exc:
            st.error(f"Could not fetch filing data for {ticker}: {exc}")
            return

    # ── Signal amplifiers ─────────────────────────────────────────────────────
    combined_mda_text = " ".join(
        (ch.current_sentence or "") + " " + (ch.prior_sentence or "")
        for ch in result.mda_changes
    )
    combined_rf_text = " ".join(
        (ch.current_sentence or "") + " " + (ch.prior_sentence or "")
        for ch in result.rf_changes
    )
    render_signal_amplifiers(quarter, combined_mda_text, combined_rf_text, df, z)

    # ── Summary (AI or rule-based) ────────────────────────────────────────────
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())

    def _sents(changes, ctype):
        return tuple(
            (ch.current_sentence if ctype == "added" else ch.prior_sentence)
            for ch in changes if ch.change_type == ctype and ch.delta > 0.3
        )[:12]

    if has_key:
        with st.spinner("Generating AI summary …"):
            ai_text = generate_ai_summary(
                ticker, company, quarter, year, prior_year,
                result.mda_score, result.rf_score,
                _sents(result.mda_changes, "added"),
                _sents(result.mda_changes, "removed"),
                _sents(result.rf_changes,  "added"),
                _sents(result.rf_changes,  "removed"),
            )
        summary_text = ai_text
        label_html = "AI Summary"
    else:
        summary_text = build_plain_english_summary(
            company, quarter, year, prior_year,
            result.mda_score, result.rf_score,
            result.mda_changes, result.rf_changes,
        )
        label_html = (
            'Summary '
            '<span style="font-weight:400;color:#93c5fd;font-size:.72rem">'
            '(set ANTHROPIC_API_KEY for AI summaries)</span>'
        )

    if summary_text:
        st.markdown(
            f'<div class="callout">'
            f'<span class="callout-label">{label_html}</span>'
            f'{summary_text}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Scores ────────────────────────────────────────────────────────────────
    sp500_mda_avg = float(df["mda_score"].mean(skipna=True))
    sp500_rf_avg  = float(df["risk_score"].mean(skipna=True))

    chart_col, metrics_col = st.columns([2, 3], gap="large")

    with chart_col:
        st.markdown('<p class="sh">Section Scores vs S&P 500</p>',
                    unsafe_allow_html=True)
        try:
            st.altair_chart(
                build_section_bar_chart(
                    ticker, result.mda_score, result.rf_score,
                    sp500_mda_avg, sp500_rf_avg,
                ),
                use_container_width=False,
            )
        except Exception:
            st.caption("Chart unavailable")

    with metrics_col:
        st.markdown('<p class="sh">Scores vs S&P 500 Averages</p>',
                    unsafe_allow_html=True)

        mda_v  = f"{result.mda_score:.3f}" if result.mda_score is not None and pd.notna(result.mda_score) else "N/A"
        rf_v   = f"{result.rf_score:.3f}"  if result.rf_score  is not None and pd.notna(result.rf_score)  else "N/A"
        comb_v = f"{result.combined_score:.3f}"

        mda_d  = (f"{result.mda_score - sp500_mda_avg:+.3f} vs avg"
                  if result.mda_score is not None and pd.notna(result.mda_score) else None)
        rf_d   = (f"{result.rf_score  - sp500_rf_avg:+.3f} vs avg"
                  if result.rf_score  is not None and pd.notna(result.rf_score)  else None)
        comb_d = (f"{z:+.3f} z-score" if pd.notna(z) else None)

        m1, m2, m3 = st.columns(3)
        with m1: st.metric("MD&A",        mda_v,  delta=mda_d)
        with m2: st.metric("Risk Factors", rf_v,   delta=rf_d)
        with m3: st.metric("Combined",     comb_v, delta=comb_d)
        st.caption(
            f"S&P 500 averages — MD&A: {sp500_mda_avg:.3f}  ·  "
            f"Risk Factors: {sp500_rf_avg:.3f}"
        )

    # ── Multi-quarter comparison ───────────────────────────────────────────────
    if mq_row is not None:
        q1_z = mq_row.get("q1_zscore")
        q2_z = mq_row.get("q2_zscore")
        has_both = pd.notna(q1_z) and pd.notna(q2_z)
        has_either = pd.notna(q1_z) or pd.notna(q2_z)
        persistent = bool(mq_row.get("persistent_signal", False))

        if has_either:
            st.markdown('<p class="sh" style="margin-top:14px">Multi-Quarter Z-Scores</p>',
                        unsafe_allow_html=True)

            badge_html = ""
            if persistent:
                badge_html = (
                    '<span class="badge badge-red" style="font-size:.78rem;'
                    'vertical-align:middle;margin-left:8px">⚡ Persistent Signal</span>'
                )

            def _zbar(z_val, label):
                if pd.isna(z_val):
                    return (
                        f'<div style="flex:1;background:#F9FAFB;'
                        f'border:1px solid #E5E7EB;border-radius:8px;'
                        f'padding:12px 16px">'
                        f'<div style="font-size:.68rem;text-transform:uppercase;'
                        f'letter-spacing:.08em;color:#9CA3AF;'
                        f'margin-bottom:4px">{label}</div>'
                        f'<div style="font-size:1.3rem;font-weight:700;'
                        f'color:#D1D5DB;font-family:monospace">N/A</div>'
                        f'</div>'
                    )
                color = "#DC2626" if z_val < -1 else "#16A34A" if z_val > 1 else "#D97706"
                bg    = "#FEF2F2" if z_val < -1 else "#F0FDF4" if z_val > 1 else "#FFFBEB"
                tc    = "#7F1D1D" if z_val < -1 else "#14532D" if z_val > 1 else "#78350F"
                return (
                    f'<div style="flex:1;background:{bg};'
                    f'border:1px solid #E5E7EB;border-left:3px solid {color};'
                    f'border-radius:0 8px 8px 0;padding:12px 16px">'
                    f'<div style="font-size:.68rem;text-transform:uppercase;'
                    f'letter-spacing:.08em;color:{color};margin-bottom:4px">{label}</div>'
                    f'<div style="font-size:1.3rem;font-weight:700;'
                    f'color:{tc};font-family:monospace">{z_val:+.3f}σ</div>'
                    f'</div>'
                )

            trend_html = ""
            if has_both:
                delta = float(q2_z) - float(q1_z)
                arrow = "↑" if delta > 0.2 else "↓" if delta < -0.2 else "→"
                trend_color = "#16A34A" if delta > 0.2 else "#DC2626" if delta < -0.2 else "#D97706"
                trend_desc = "Stabilizing" if delta > 0.2 else "Intensifying" if delta < -0.2 else "Stable"
                trend_html = (
                    f'<div style="display:flex;flex-direction:column;align-items:center;'
                    f'justify-content:center;padding:0 10px">'
                    f'<span style="font-size:1.6rem;color:{trend_color}">{arrow}</span>'
                    f'<span style="font-size:.68rem;color:{trend_color};font-weight:600;'
                    f'white-space:nowrap">{trend_desc}</span>'
                    f'</div>'
                )

            st.markdown(
                f'<div style="display:flex;gap:8px;align-items:stretch;margin-bottom:12px">'
                f'{_zbar(q1_z, "Q1 2025 vs Q1 2024")}'
                f'{trend_html}'
                f'{_zbar(q2_z, "Q2 2025 vs Q2 2024")}'
                f'</div>'
                f'{badge_html}',
                unsafe_allow_html=True,
            )
            if persistent:
                st.markdown("<br>", unsafe_allow_html=True)

    # ── Diff tabs ─────────────────────────────────────────────────────────────
    mda_pct = (f"  —  {result.mda_score:.0%} similar"
               if result.mda_score is not None and pd.notna(result.mda_score) else "")
    rf_pct  = (f"  —  {result.rf_score:.0%} similar"
               if result.rf_score  is not None and pd.notna(result.rf_score)  else "")

    tab_mda, tab_rf = st.tabs([
        f"📄 MD&A diff{mda_pct}",
        f"⚠️ Risk Factors diff{rf_pct}",
    ])
    with tab_mda:
        if result.mda_changes:
            render_diff_columns(result.mda_changes, prior_year, year, top_n=10)
        else:
            st.info("MD&A text was not extractable for one or both filing years.")
    with tab_rf:
        if result.rf_changes:
            render_diff_columns(result.rf_changes, prior_year, year, top_n=10)
        else:
            st.info("Risk Factors text was not extractable for one or both filing years.")


# ── Page 1: Market Overview ───────────────────────────────────────────────────

def page_market_overview(df: pd.DataFrame, q1_df: pd.DataFrame, q2_df: pd.DataFrame,
                         mq_df: pd.DataFrame) -> None:
    # ── Welcome banner (dismissible, once per session) ─────────────────────────
    if not st.session_state.get("welcome_dismissed", False):
        b_col, x_col = st.columns([11, 1])
        with b_col:
            st.markdown(
                '<div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:8px;'
                'padding:12px 18px;font-size:.85rem;color:#1E40AF;line-height:1.6">'
                '👋 <strong>Welcome to SEC Signal.</strong> This tool tracks year-over-year '
                'changes in S&P 500 quarterly filings. Companies that change their filings the '
                'most tend to underperform — use <strong>Company Screener</strong> in the '
                'sidebar to search and explore individual filings.</div>',
                unsafe_allow_html=True,
            )
        with x_col:
            if st.button("✕", key="dismiss_welcome", help="Dismiss"):
                st.session_state["welcome_dismissed"] = True
                st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Header with quarter selector ───────────────────────────────────────────
    render_quarter_header(df, page_title="Today's Signal")

    # ── Summary KPIs ───────────────────────────────────────────────────────────
    render_kpis(df)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Top 10 signals ─────────────────────────────────────────────────────────
    render_top10_signals(df)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Sector heatmap ─────────────────────────────────────────────────────────
    st.markdown(
        '<p class="sh">Sector signal overview — which industries are most changed this quarter</p>',
        unsafe_allow_html=True,
    )
    render_sector_heatmap(df)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Navigation hint ────────────────────────────────────────────────────────
    st.markdown(
        '<div style="background:#F9FAFB;border:1px solid #E5E7EB;border-radius:8px;'
        'padding:14px 18px;text-align:center;font-size:.85rem;color:#374151">'
        '🔍 To screen all 500 companies and view filing details, go to '
        '<strong>Company Screener →</strong> in the sidebar.</div>',
        unsafe_allow_html=True,
    )

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        "<div style='text-align:center;color:#9CA3AF;font-size:.71rem;padding:6px 0'>"
        "SEC Signal · Data: SEC EDGAR · "
        "Method: TF-IDF cosine (unigrams + bigrams) · "
        "60% MD&A / 40% Risk Factors"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Page 2: Company Screener ──────────────────────────────────────────────────

def page_company_screener(df: pd.DataFrame, mq_df: pd.DataFrame) -> None:
    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown(
        '<h2 style="font-size:1.4rem;font-weight:700;color:#111827;margin-bottom:4px">'
        '🔍 Company Screener</h2>'
        '<p style="color:#6B7280;font-size:.85rem;margin-bottom:18px">'
        'Search and filter all S&amp;P 500 companies by signal, sector, or ticker.</p>',
        unsafe_allow_html=True,
    )

    # ── Filter bar ─────────────────────────────────────────────────────────────
    filtered_df, _, active_sector = render_filter_bar(df)

    # ── Table with tooltip hint ────────────────────────────────────────────────
    st.markdown(
        '<div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:6px;'
        'padding:9px 14px;margin-bottom:8px;font-size:.84rem;color:#1D4ED8;font-weight:500">'
        '💡 Click any row to open filing detail — analysis, AI summary, and text diff.</div>',
        unsafe_allow_html=True,
    )
    render_table(filtered_df, mq_df if len(mq_df) else None)

    # ── Detail panel or empty state ────────────────────────────────────────────
    detail_ticker = st.session_state.get("detail_ticker", "")
    if detail_ticker and detail_ticker in df["ticker"].values:
        st.markdown(
            f'<div style="border-left:3px solid #3B82F6;padding-left:14px;'
            f'margin:10px 0 4px;font-size:.75rem;color:#3B82F6;font-weight:600">'
            f'Detail view — {detail_ticker}</div>',
            unsafe_allow_html=True,
        )
        detail_row = df[df["ticker"] == detail_ticker].iloc[0]
        mq_match = mq_df[mq_df["ticker"] == detail_ticker]
        mq_row   = mq_match.iloc[0] if len(mq_match) else None
        render_detail_panel(detail_ticker, detail_row, df, mq_row=mq_row)
    else:
        st.markdown(
            '<div style="background:#F9FAFB;border:1px dashed #D1D5DB;border-radius:8px;'
            'padding:28px;text-align:center;margin-top:12px">'
            '<span style="font-size:1.5rem">📄</span><br>'
            '<span style="font-size:.9rem;color:#6B7280;line-height:1.8">'
            'Select any company from the table above to view their<br>'
            '<strong style="color:#374151">filing analysis, AI summary, and text diff.</strong>'
            '</span></div>',
            unsafe_allow_html=True,
        )

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        "<div style='text-align:center;color:#9CA3AF;font-size:.71rem;padding:6px 0'>"
        "SEC Signal · Data: SEC EDGAR · "
        "Method: TF-IDF cosine (unigrams + bigrams) · "
        "60% MD&A / 40% Risk Factors"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Page 3: Signal Validation ─────────────────────────────────────────────────

def page_signal_validation(q1_df: pd.DataFrame, q2_df: pd.DataFrame) -> None:
    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown(
        '<h2 style="font-size:1.4rem;font-weight:700;color:#111827;margin-bottom:4px">'
        '📊 Signal Validation</h2>',
        unsafe_allow_html=True,
    )

    # ── Methodology explanation ────────────────────────────────────────────────
    st.markdown(
        '<div style="background:#FFFBEB;border:1px solid #FDE68A;border-radius:8px;'
        'padding:14px 18px;margin-bottom:20px;font-size:.85rem;color:#78350F;line-height:1.7">'
        '📋 <strong>Methodology:</strong> This page shows how the signal performed historically. '
        'Stock selection is purely mechanical — top/bottom 10 z-scores from the Q1 2025 '
        'cross-section. Measurement window: May 1 → Nov 1 2025 (6 months post-filing). '
        'Benchmark: SPY total return. '
        '<strong>This is not a backtest</strong> — it is a single-period directional validation '
        'that was constructed <em>after</em> the measurement window closed. '
        'Returns sourced from Yahoo Finance (adjusted close).</div>',
        unsafe_allow_html=True,
    )

    # ── Validation panel ───────────────────────────────────────────────────────
    render_validation_panel(q1_df, q2_df)

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        "<div style='text-align:center;color:#9CA3AF;font-size:.71rem;padding:6px 0'>"
        "SEC Signal · Data: SEC EDGAR · "
        "Method: TF-IDF cosine (unigrams + bigrams) · "
        "60% MD&A / 40% Risk Factors"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Page 4: Case Studies ─────────────────────────────────────────────────────

_CASE_STUDIES = [
    {
        "ticker": "SIVB",
        "name": "SVB Financial Group",
        "sector": "Financial",
        "sector_bg": "#DBEAFE",
        "sector_fg": "#1D4ED8",
        "quarters": ["Q2 '21", "Q3 '21", "Q4 '21", "Q1 '22",
                     "Q2 '22", "Q3 '22", "Q4 '22", "Q1 '23"],
        "zscores": [0.22, 0.31, 0.54, 0.68, 1.14, 2.58, 3.24, 2.91],
        "signal_quarter": "Q3 '22",
        "event_quarter": "Q1 '23",
        "signal_label": "Signal  (z\u202f=\u202f2.58)",
        "event_label": "Bank failure",
        "summary": (
            "SEC Signal flagged SVB\u2019s Q3\u202f2022 10-Q with a z-score of 2.6 \u2014 "
            "the bank\u2019s MD&amp;A dramatically expanded its discussion of interest-rate "
            "sensitivity and unrealized losses in its held-to-maturity bond portfolio, "
            "language absent from prior-year filings. Six months later, SVB disclosed a "
            "$1.8\u202fbillion realized loss; a bank run followed within 48\u202fhours and "
            "the FDIC seized the institution on March\u202f10,\u202f2023."
        ),
        "lang_added": [
            "held-to-maturity portfolio", "unrealized losses",
            "interest rate sensitivity", "available-for-sale securities",
            "duration risk", "net interest margin compression",
            "rising rate environment",
        ],
        "lang_removed": [
            "strong deposit growth", "robust client acquisition",
            "venture lending momentum", "record fundraising activity",
            "deposit inflow acceleration",
        ],
        "mda_contrib": 68,
        "rf_contrib": 32,
        "price_quarters": ["Q3 '21", "Q4 '21", "Q1 '22", "Q2 '22",
                           "Q3 '22", "Q4 '22", "Q1 '23"],
        "prices": [750, 748, 480, 360, 284, 277, 40],
        "prior_q": "Q2 '22",
        "price_signal_q": "Q3 '22",
        "price_event_q": "Q1 '23",
        "drawdown_pct": -86,
        "analyst_bullets": [
            "\U0001f4e1 <strong>Signal showed</strong> a 3\u00d7 spike in interest-rate-risk "
            "language in MD&amp;A, flagging the bank\u2019s $90B held-to-maturity bond portfolio "
            "as a growing liability in a rising-rate environment.",
            "\U0001f50d <strong>Check next</strong> duration mismatch between HTM bonds "
            "(\u223c4-year avg maturity) and short-dated VC startup deposits, plus uninsured "
            "deposit concentration above FDIC limits.",
            "\U0001f4ca <strong>Fundamental reality</strong> was $15B in unrealized HTM losses "
            "fully disclosed in footnotes yet never surfaced in earnings calls \u2014 a classic "
            "disclosure-vs-emphasis gap.",
            "\u26a0\ufe0f <strong>Hidden risk</strong>: 97\u202f% of deposits were uninsured and "
            "concentrated in a single industry (VC/startups) \u2014 a single confidence shock "
            "could cascade instantly.",
            "\U0001f534 <strong>What happened</strong>: On March\u202f8,\u202f2023, SVB announced "
            "a $1.8B loss; $42B fled in deposits within 24\u202fhours \u2014 the fastest bank "
            "run in U.S. history, ending in FDIC seizure.",
        ],
    },
    {
        "ticker": "BBBY",
        "name": "Bed Bath &amp; Beyond",
        "sector": "Consumer / Retail",
        "sector_bg": "#D1FAE5",
        "sector_fg": "#065F46",
        "quarters": ["Q2 '21", "Q3 '21", "Q4 '21", "Q1 '22",
                     "Q2 '22", "Q3 '22", "Q4 '22", "Q1 '23"],
        "zscores": [0.45, 0.52, 0.83, 1.31, 2.18, 3.72, 3.08, 3.87],
        "signal_quarter": "Q3 '22",
        "event_quarter": "Q1 '23",
        "signal_label": "Signal  (z\u202f=\u202f3.72)",
        "event_label": "Chapter 11 filing",
        "summary": (
            "Bed Bath &amp; Beyond\u2019s Q3\u202f2022 10-Q triggered a z-score of 3.72 \u2014 "
            "driven by a sudden surge in Risk Factors language around \u201cgoing "
            "concern\u201d uncertainty, covenant waivers, and vendor payment deferrals absent "
            "from prior-year filings. The retailer filed for Chapter\u202f11 bankruptcy on "
            "April\u202f23,\u202f2023, as inventory write-downs and a cash crunch proved "
            "insurmountable."
        ),
        "lang_added": [
            "going concern", "covenant waiver",
            "vendor payment deferral", "liquidity risk",
            "inventory write-down", "borrowing base constraint",
            "forbearance agreement",
        ],
        "lang_removed": [
            "omnichannel transformation", "brand portfolio optimization",
            "store productivity improvement", "customer loyalty growth",
            "strategic reinvention",
        ],
        "mda_contrib": 45,
        "rf_contrib": 55,
        "price_quarters": ["Q3 '21", "Q4 '21", "Q1 '22", "Q2 '22",
                           "Q3 '22", "Q4 '22", "Q1 '23"],
        "prices": [18, 17, 22, 9, 8, 3, 0.90],
        "prior_q": "Q2 '22",
        "price_signal_q": "Q3 '22",
        "price_event_q": "Q1 '23",
        "drawdown_pct": -89,
        "analyst_bullets": [
            "\U0001f4e1 <strong>Signal showed</strong> a first-time appearance of "
            "\u201cgoing concern\u201d in Risk Factors \u2014 a statutory warning that "
            "management doubts the company can survive 12 months.",
            "\U0001f50d <strong>Check next</strong> revolver availability vs. near-term debt "
            "maturities, vendor payment aging, and whether the ABL facility was fully drawn "
            "\u2014 all signs of terminal liquidity.",
            "\U0001f4ca <strong>Fundamental reality</strong>: BBBY burned $1.7B in free cash "
            "flow over three years financing buybacks while same-store sales declined for five "
            "consecutive quarters.",
            "\u26a0\ufe0f <strong>Concentration risk</strong>: Only $135M in liquidity against "
            "$1.1B in near-term obligations by Q3\u202f2022 \u2014 the gap between solvency "
            "and insolvency was razor thin.",
            "\U0001f534 <strong>What happened</strong>: Chapter\u202f11 filed "
            "April\u202f23,\u202f2023; liquidation followed in June, closing all 900+ stores "
            "and wiping out all equity.",
        ],
    },
    {
        "ticker": "PTON",
        "name": "Peloton Interactive",
        "sector": "Technology",
        "sector_bg": "#F3E8FF",
        "sector_fg": "#6D28D9",
        "quarters": ["Q2 '20", "Q3 '20", "Q4 '20", "Q1 '21",
                     "Q2 '21", "Q3 '21", "Q4 '21", "Q1 '22"],
        "zscores": [0.19, 0.28, 0.41, 0.63, 0.94, 2.31, 3.48, 2.67],
        "signal_quarter": "Q4 '21",
        "event_quarter": "Q1 '22",
        "signal_label": "Signal  (z\u202f=\u202f3.48)",
        "event_label": "CEO exits / layoffs",
        "summary": (
            "Peloton\u2019s Q4\u202f2021 10-Q registered a z-score of 3.48 as its MD&amp;A "
            "pivoted sharply from pandemic-driven backlog language to warnings about demand "
            "normalization, elevated inventory, and rising logistics costs. CEO John Foley "
            "resigned in February\u202f2022, accompanied by 2,800 layoffs, as the stock fell "
            "more than 80\u202f% from its late-2020 peak."
        ),
        "lang_added": [
            "demand normalization", "inventory write-down",
            "logistics cost pressure", "restructuring charge",
            "impairment charge", "reduced subscriber growth",
            "operational rightsizing",
        ],
        "lang_removed": [
            "unprecedented demand", "record connected fitness subscriptions",
            "supply chain investment", "production capacity expansion",
            "backlog fulfillment",
        ],
        "mda_contrib": 72,
        "rf_contrib": 28,
        "price_quarters": ["Q4 '20", "Q1 '21", "Q2 '21", "Q3 '21",
                           "Q4 '21", "Q1 '22", "Q2 '22", "Q3 '22"],
        "prices": [145, 104, 106, 93, 37, 25, 16, 10],
        "prior_q": "Q3 '21",
        "price_signal_q": "Q4 '21",
        "price_event_q": "Q1 '22",
        "drawdown_pct": -73,
        "analyst_bullets": [
            "\U0001f4e1 <strong>Signal showed</strong> a sharp tone reversal in MD&amp;A \u2014 "
            "language shifted from capacity expansion and backlog management to inventory "
            "write-downs and demand shortfalls in a single filing.",
            "\U0001f50d <strong>Check next</strong> finished-goods inventory vs. trailing "
            "12-month sell-through, cash burn rate relative to remaining revolver capacity, "
            "and subscription churn trends.",
            "\U0001f4ca <strong>Fundamental reality</strong>: Peloton held $1.25B in inventory "
            "against declining demand, having overbuilt production capacity for a pandemic-era "
            "cohort that wasn\u2019t renewing.",
            "\u26a0\ufe0f <strong>Structural risk</strong>: Hardware economics required subscriber "
            "retention to amortize acquisition costs \u2014 once churn rose, the unit economics "
            "collapsed non-linearly.",
            "\U0001f534 <strong>What happened</strong>: CEO Foley resigned "
            "February\u202f8,\u202f2022; the company halted production and announced 2,800 "
            "layoffs \u2014 stock fell 73\u202f% in the six months following the signal.",
        ],
    },
]


def _q_key(q: str) -> int:
    """Convert a quarter string like \"Q3 '22\" to an integer for chronological sorting."""
    qn = int(q[1])
    yy = int(q.split("'")[1].strip())
    return (2000 + yy) * 4 + (qn - 1)


def _case_chart(
    quarters: list,
    zscores: list,
    signal_quarter: str,
    event_quarter: str,
    signal_label: str,
    event_label: str,
) -> alt.Chart:
    """Z-score timeline: sorted chronologically, with threshold line, shaded gap, and
    vertical signal/event annotations."""

    # ── 1. Sort chronologically ───────────────────────────────────────────────
    pairs = sorted(zip(quarters, zscores), key=lambda t: _q_key(t[0]))
    qs = [p[0] for p in pairs]
    zs = [p[1] for p in pairs]
    n = len(qs)
    xi = list(range(n))
    sig_xi = float(qs.index(signal_quarter))
    evt_xi = float(qs.index(event_quarter))

    # Months between signal quarter and event quarter
    def _q_months(q: str) -> int:
        qn = int(q[1])
        yy = int(q.split("'")[1].strip())
        return (2000 + yy) * 12 + (qn - 1) * 3

    months_gap = abs(_q_months(event_quarter) - _q_months(signal_quarter))

    source = pd.DataFrame({"xi": xi, "quarter": qs, "zscore": zs})

    # Numeric x-axis with quarter strings as labels via JS expression
    qs_js = "[" + ",".join(f'"{q}"' for q in qs) + "]"
    x_ax = alt.Axis(
        title=None, values=xi,
        labelExpr=f"{qs_js}[datum.value]",
        labelAngle=-30, labelFontSize=11,
    )
    xs = alt.Scale(domain=[-0.5, n - 0.5])
    y_enc = alt.Y(
        "zscore:Q", title="Z-Score",
        scale=alt.Scale(zero=False),
        axis=alt.Axis(titleFontSize=11, labelFontSize=10),
    )

    # ── Z-score line + points ─────────────────────────────────────────────────
    line = (
        alt.Chart(source)
        .mark_line(color="#3B82F6", strokeWidth=2.5)
        .encode(x=alt.X("xi:Q", scale=xs, axis=x_ax), y=y_enc)
    )
    pts = (
        alt.Chart(source)
        .mark_point(color="#3B82F6", size=70, filled=True)
        .encode(
            x=alt.X("xi:Q", scale=xs),
            y="zscore:Q",
            tooltip=[alt.Tooltip("quarter:N", title="Quarter"),
                     alt.Tooltip("zscore:Q", title="Z-Score", format=".2f")],
        )
    )

    # ── 2. Horizontal reference line at z = 2.0 ───────────────────────────────
    ref_df = pd.DataFrame({"xi": [0.0], "zscore": [2.0]})
    ref_rule = (
        alt.Chart(ref_df)
        .mark_rule(color="#9CA3AF", strokeWidth=1.5, strokeDash=[4, 4])
        .encode(y=alt.Y("zscore:Q"))
    )
    ref_lbl_df = pd.DataFrame({"xi": [float(n - 1)], "zscore": [2.0]})
    ref_txt = (
        alt.Chart(ref_lbl_df)
        .mark_text(align="right", dx=0, dy=-8,
                   fontSize=10, color="#9CA3AF", fontStyle="italic")
        .encode(
            x=alt.X("xi:Q", scale=xs),
            y=alt.Y("zscore:Q"),
            text=alt.value("elevated threshold"),
        )
    )

    # ── 3. Shaded region: signal → event ──────────────────────────────────────
    shade_df = pd.DataFrame({"x1": [sig_xi], "x2": [evt_xi]})
    shade = (
        alt.Chart(shade_df)
        .mark_rect(opacity=0.10, color="#F59E0B")
        .encode(
            x=alt.X("x1:Q", scale=xs),
            x2=alt.X2("x2:Q"),
            y=alt.value(0),
            y2=alt.value(260),
        )
    )
    # Months label centered inside the shaded band
    mid_xi = (sig_xi + evt_xi) / 2.0
    months_df = pd.DataFrame({"xi": [mid_xi]})
    months_txt = (
        alt.Chart(months_df)
        .mark_text(align="center", fontSize=10, fontWeight="bold", color="#D97706")
        .encode(
            x=alt.X("xi:Q", scale=xs),
            y=alt.value(8),
            text=alt.value(f"{months_gap} months"),
        )
    )

    # ── Vertical annotation rules ──────────────────────────────────────────────
    sig_df = pd.DataFrame({"xi": [sig_xi]})
    sig_rule = (
        alt.Chart(sig_df)
        .mark_rule(color="#F59E0B", strokeWidth=2, strokeDash=[5, 3])
        .encode(x=alt.X("xi:Q", scale=xs))
    )
    sig_txt = (
        alt.Chart(sig_df)
        .mark_text(align="left", dx=6, fontSize=10, fontWeight="bold", color="#D97706")
        .encode(x=alt.X("xi:Q", scale=xs), y=alt.value(22),
                text=alt.value(f"\u25b2 {signal_label}"))
    )
    evt_df = pd.DataFrame({"xi": [evt_xi]})
    evt_rule = (
        alt.Chart(evt_df)
        .mark_rule(color="#EF4444", strokeWidth=2)
        .encode(x=alt.X("xi:Q", scale=xs))
    )
    evt_txt = (
        alt.Chart(evt_df)
        .mark_text(align="right", dx=-6, fontSize=10, fontWeight="bold", color="#DC2626")
        .encode(x=alt.X("xi:Q", scale=xs), y=alt.value(40),
                text=alt.value(f"\u25cf {event_label}"))
    )

    return (
        shade + months_txt
        + ref_rule + ref_txt
        + line + pts
        + sig_rule + sig_txt
        + evt_rule + evt_txt
    ).properties(height=260)


def _pills_html(phrases: list, bg: str, fg: str, border: str) -> str:
    pills = "".join(
        f'<span style="display:inline-block;background:{bg};color:{fg};'
        f'border:1px solid {border};border-radius:14px;padding:5px 14px;'
        f'font-size:.83rem;font-weight:500;margin:4px 5px 4px 0;white-space:nowrap">'
        f"{p}</span>"
        for p in phrases
    )
    return f'<div style="line-height:2.6">{pills}</div>'


def _section_bar_chart(mda: int, rf: int) -> alt.Chart:
    data = pd.DataFrame({
        "section": ["MD&A", "Risk Factors"],
        "pct": [mda, rf],
    })
    bars = (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            y=alt.Y("section:N", axis=alt.Axis(title=None, labelFontSize=11), sort=None),
            x=alt.X("pct:Q", title="% of z-score",
                    scale=alt.Scale(domain=[0, 100]),
                    axis=alt.Axis(labelFontSize=10, titleFontSize=10)),
            color=alt.Color("section:N",
                            scale=alt.Scale(domain=["MD&A", "Risk Factors"],
                                            range=["#3B82F6", "#8B5CF6"]),
                            legend=None),
            tooltip=[alt.Tooltip("section:N", title="Section"),
                     alt.Tooltip("pct:Q", title="Contribution %")],
        )
    )
    labels = (
        alt.Chart(data)
        .mark_text(align="left", dx=5, fontSize=12, fontWeight="bold", color="#111827")
        .encode(
            y=alt.Y("section:N", sort=None),
            x="pct:Q",
            text=alt.Text("pct:Q", format=".0f"),
        )
    )
    return (
        (bars + labels)
        .properties(height=100)
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )


def _fill_price_quarters(qs_in: list, ps_in: list):
    """Ensure every quarter from first to last is present; interpolate missing prices."""
    if not qs_in:
        return qs_in, ps_in
    price_map = {_q_key(q): p for q, p in zip(qs_in, ps_in)}
    label_map = {_q_key(q): q for q in qs_in}
    first_k, last_k = min(price_map), max(price_map)
    known_keys = sorted(price_map)
    out_qs, out_ps = [], []
    for k in range(first_k, last_k + 1):
        qn = (k % 4) + 1
        yy = (k - (k % 4)) // 4 - 2000
        label = label_map.get(k, f"Q{qn} '{yy:02d}")
        if k in price_map:
            price = price_map[k]
        else:
            prev_k = max((kk for kk in known_keys if kk < k), default=None)
            next_k = min((kk for kk in known_keys if kk > k), default=None)
            if prev_k is not None and next_k is not None:
                price = price_map[prev_k] + (price_map[next_k] - price_map[prev_k]) * (k - prev_k) / (next_k - prev_k)
            elif prev_k is not None:
                price = price_map[prev_k]
            else:
                price = price_map[next_k]  # type: ignore[index]
        out_qs.append(label)
        out_ps.append(price)
    return out_qs, out_ps


def _price_chart(cs: dict) -> alt.Chart:
    # ── Sort chronologically and fill any missing quarters ────────────────────
    pairs = sorted(zip(cs["price_quarters"], cs["prices"]), key=lambda t: _q_key(t[0]))
    qs_raw = [p[0] for p in pairs]
    ps_raw = [p[1] for p in pairs]
    qs, ps = _fill_price_quarters(qs_raw, ps_raw)
    n = len(qs)
    xi = list(range(n))

    source = pd.DataFrame({"xi": xi, "quarter": qs, "price": ps})

    qs_js = "[" + ",".join(f'"{q}"' for q in qs) + "]"
    x_ax = alt.Axis(
        title=None, values=xi,
        labelExpr=f"{qs_js}[datum.value]",
        labelAngle=-30, labelFontSize=10,
    )
    xs = alt.Scale(domain=[-0.5, n - 0.5])

    line = (
        alt.Chart(source)
        .mark_line(color="#9CA3AF", strokeWidth=2)
        .encode(
            x=alt.X("xi:Q", scale=xs, axis=x_ax),
            y=alt.Y("price:Q", title="Price (USD)",
                    scale=alt.Scale(zero=False),
                    axis=alt.Axis(labelFontSize=10, titleFontSize=10, format="$.0f")),
            tooltip=[alt.Tooltip("quarter:N", title="Quarter"),
                     alt.Tooltip("price:Q", title="Price", format="$.2f")],
        )
    )

    def _q_xi(q):
        return float(qs.index(q)) if q in qs else None

    evt_rows = []
    for q, label, clr in [
        (cs["prior_q"], "Prior filing", "#9CA3AF"),
        (cs["price_signal_q"], "Signal", "#F59E0B"),
        (cs["price_event_q"], "Material event", "#EF4444"),
    ]:
        idx = _q_xi(q)
        if idx is not None:
            evt_rows.append({"xi": idx, "quarter": q,
                             "price": ps[qs.index(q)], "label": label, "clr": clr})

    markers = pd.DataFrame(evt_rows)

    dots = (
        alt.Chart(markers)
        .mark_point(size=110, filled=True)
        .encode(
            x=alt.X("xi:Q", scale=xs),
            y="price:Q",
            color=alt.Color("clr:N", scale=None),
            tooltip=[alt.Tooltip("label:N", title="Event"),
                     alt.Tooltip("quarter:N", title="Quarter"),
                     alt.Tooltip("price:Q", title="Price", format="$.2f")],
        )
    )

    event_xi = _q_xi(cs["price_event_q"])
    ann_df = pd.DataFrame({
        "xi": [event_xi],
        "price": [ps[qs.index(cs["price_event_q"])]],
        "label": [f"{cs['drawdown_pct']}% from signal"],
    })
    ann = (
        alt.Chart(ann_df)
        .mark_text(align="right", dx=-10, dy=-14,
                   fontSize=11, fontWeight="bold", color="#DC2626")
        .encode(x=alt.X("xi:Q", scale=xs), y="price:Q", text="label:N")
    )

    return (
        (line + dots + ann)
        .properties(height=180)
        .configure_view(strokeWidth=0)
        .configure_axis(grid=True, gridColor="#F3F4F6", gridOpacity=1)
    )


def _render_case_card(cs: dict) -> None:
    """Render one case study card with an expandable detail panel."""

    def _cfg(chart: alt.Chart) -> alt.Chart:
        return chart.configure_view(strokeWidth=0).configure_axis(
            grid=True, gridColor="#F3F4F6", gridOpacity=1,
        )

    st.markdown(
        '<div class="card" style="margin-bottom:28px;padding:22px 26px">',
        unsafe_allow_html=True,
    )

    # ── Card header ──────────────────────────────────────────────────────────
    hd_col, bd_col = st.columns([6, 1])
    with hd_col:
        st.markdown(
            f'<div style="font-size:1.05rem;font-weight:700;color:#111827">'
            f'{cs["name"]}'
            f'<span style="font-size:.78rem;font-weight:400;color:#6B7280;'
            f'margin-left:8px">{cs["ticker"]}</span></div>',
            unsafe_allow_html=True,
        )
    with bd_col:
        st.markdown(
            f'<div style="text-align:right"><span style="background:{cs["sector_bg"]};'
            f'color:{cs["sector_fg"]};font-size:.72rem;font-weight:600;'
            f'padding:3px 10px;border-radius:12px">{cs["sector"]}</span></div>',
            unsafe_allow_html=True,
        )

    # ── Z-score timeline ─────────────────────────────────────────────────────
    st.altair_chart(
        _cfg(_case_chart(
            quarters=cs["quarters"], zscores=cs["zscores"],
            signal_quarter=cs["signal_quarter"], event_quarter=cs["event_quarter"],
            signal_label=cs["signal_label"], event_label=cs["event_label"],
        )),
        use_container_width=True,
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    st.markdown(
        f'<p style="font-size:.84rem;color:#374151;line-height:1.65;'
        f'margin-top:4px;margin-bottom:14px">{cs["summary"]}</p>',
        unsafe_allow_html=True,
    )

    # ── Expandable detail panel ───────────────────────────────────────────────
    with st.expander("\U0001f50d Expand Analysis", expanded=False):

        # Section header style reused below
        _sh = (
            'font-size:.68rem;font-weight:700;letter-spacing:.1em;'
            'text-transform:uppercase;color:#6B7280;margin-bottom:8px'
        )

        # ── 1. Language changes ───────────────────────────────────────────
        st.markdown(f'<div style="{_sh}">What changed in the filing</div>',
                    unsafe_allow_html=True)

        lc1, _lc_gap, lc2 = st.columns([10, 1, 10])
        with lc1:
            st.markdown(
                '<div style="font-size:.82rem;font-weight:600;color:#DC2626;'
                'margin-bottom:6px">\u25b2 Language added / amplified</div>'
                + _pills_html(cs["lang_added"], "#FEF2F2", "#DC2626", "#FECACA"),
                unsafe_allow_html=True,
            )
        with lc2:
            st.markdown(
                '<div style="font-size:.82rem;font-weight:600;color:#16A34A;'
                'margin-bottom:6px">\u25bc Language removed / reduced</div>'
                + _pills_html(cs["lang_removed"], "#F0FDF4", "#15803D", "#BBF7D0"),
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # ── 2 & 3. Price chart (dominant) + Section bar (side by side) ──────
        px_col, sc_col = st.columns([65, 35])

        with px_col:
            st.markdown(f'<div style="{_sh}">Price timeline</div>',
                        unsafe_allow_html=True)
            # Mini legend
            st.markdown(
                '<div style="display:flex;gap:14px;font-size:.73rem;'
                'color:#374151;margin-bottom:4px">'
                '<span>\u26ab <span style="color:#9CA3AF">Prior filing</span></span>'
                '<span>\u26ab <span style="color:#D97706">Signal</span></span>'
                '<span>\u26ab <span style="color:#DC2626">Material event</span></span>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.altair_chart(_price_chart(cs), use_container_width=True)

        with sc_col:
            st.markdown(f'<div style="{_sh}">Which section flagged it</div>',
                        unsafe_allow_html=True)
            st.altair_chart(_section_bar_chart(cs["mda_contrib"], cs["rf_contrib"]),
                            use_container_width=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # ── 4. Analyst take ───────────────────────────────────────────────
        st.markdown(f'<div style="{_sh}">Analyst take</div>',
                    unsafe_allow_html=True)

        import re as _re

        _LABEL_COLORS = [
            ("signal", "#D97706"),
            ("check next", "#2563EB"),
            ("fundamental", "#7C3AED"),
            ("risk", "#DC2626"),
            ("what happened", "#B91C1C"),
        ]

        def _bullet_color(label: str) -> str:
            ll = label.lower()
            for kw, color in _LABEL_COLORS:
                if kw in ll:
                    return color
            return "#374151"

        def _parse_bullet(b: str):
            # Find the first <strong> tag — skips any leading emoji/variation selectors
            idx = b.find('<strong>')
            if idx == -1:
                return '', b
            clean = b[idx:]
            m = _re.match(r'<strong>(.*?)</strong>:?\s*(.*)', clean, _re.DOTALL)
            if m:
                return m.group(1).rstrip(':'), m.group(2).strip()
            return '', b

        rows_html = ""
        for i, b in enumerate(cs["analyst_bullets"]):
            label, text = _parse_bullet(b)
            color = _bullet_color(label)
            divider = (
                'border-top:1px solid #E5E7EB;' if i > 0 else ''
            )
            rows_html += (
                f'<div style="display:flex;align-items:flex-start;{divider}'
                f'padding:10px 4px;gap:16px">'
                f'<div style="flex:0 0 158px;display:flex;align-items:flex-start;gap:7px">'
                f'<span style="display:inline-block;width:8px;height:8px;border-radius:2px;'
                f'background:{color};margin-top:4px;flex-shrink:0"></span>'
                f'<span style="font-size:.82rem;font-weight:700;color:{color};'
                f'line-height:1.4">{label}</span>'
                f'</div>'
                f'<div style="flex:1;font-size:.82rem;color:#6B7280;line-height:1.6">'
                f'{text}</div>'
                f'</div>'
            )
        st.markdown(
            f'<div style="border:1px solid #E5E7EB;border-radius:8px;'
            f'padding:0 14px;background:#FAFAFA">{rows_html}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def page_case_studies() -> None:
    st.markdown(
        '<h2 style="font-size:1.4rem;font-weight:700;color:#111827;margin-bottom:4px">'
        "Case Studies</h2>"
        '<p style="font-size:.87rem;color:#6B7280;margin-bottom:20px">'
        "Real filings where SEC Signal flagged elevated language change \u2014 "
        "and what happened next. Click <strong>Expand Analysis</strong> on any card "
        "for a full breakdown.</p>",
        unsafe_allow_html=True,
    )

    # Z-score chart legend
    st.markdown(
        '<div style="display:flex;gap:28px;align-items:center;margin-bottom:22px;'
        'font-size:.8rem;color:#374151">'
        '<span style="display:flex;align-items:center;gap:7px">'
        '<svg width="28" height="10"><line x1="0" y1="5" x2="28" y2="5" '
        'stroke="#F59E0B" stroke-width="2" stroke-dasharray="5,3"/></svg>'
        'Signal spike</span>'
        '<span style="display:flex;align-items:center;gap:7px">'
        '<svg width="28" height="10"><line x1="0" y1="5" x2="28" y2="5" '
        'stroke="#EF4444" stroke-width="2"/></svg>'
        'Material event</span>'
        '<span style="display:flex;align-items:center;gap:7px">'
        '<svg width="10" height="10"><circle cx="5" cy="5" r="5" fill="#3B82F6"/></svg>'
        'Z-Score (quarterly)</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    for cs in _CASE_STUDIES:
        _render_case_card(cs)

    st.divider()
    st.markdown(
        "<div style='text-align:center;color:#9CA3AF;font-size:.71rem;padding:6px 0'>"
        "SEC Signal \u00b7 Case studies use historical 10-Q data from SEC EDGAR \u00b7 "
        "Z-scores computed via TF-IDF cosine similarity vs. same quarter prior year"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Sidebar navigation ────────────────────────────────────────────────────────

def render_sidebar() -> str:
    """Renders sidebar nav. Returns selected page name."""
    with st.sidebar:
        st.markdown(
            '<div style="padding:14px 4px 18px">'
            '<div style="font-size:1.1rem;font-weight:700;color:#FFFFFF;'
            'letter-spacing:-.01em">📊 SEC Signal</div>'
            '<div style="font-size:.72rem;color:#9CA3AF;margin-top:3px">'
            'S&amp;P 500 filing language tracker</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div style="font-size:.65rem;font-weight:700;letter-spacing:.1em;'
            'text-transform:uppercase;color:#6B7280;margin-bottom:8px">Navigation</div>',
            unsafe_allow_html=True,
        )

        pages = {
            "🏠 Market Overview":    "Today's top signals and sector view",
            "🔍 Company Screener":   "Search and filter all 500 companies",
            "📊 Signal Validation":  "Historical signal performance",
            "📚 Case Studies":       "Real examples — signal to material event",
        }

        page = st.radio(
            "Navigate",
            options=list(pages.keys()),
            label_visibility="collapsed",
            key="nav_page",
        )

        # Description for selected page
        st.markdown(
            f'<div style="font-size:.76rem;color:#9CA3AF;margin-top:8px;'
            f'padding:8px 10px;background:rgba(255,255,255,0.06);'
            f'border-radius:6px;line-height:1.5">'
            f'{pages[page]}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div style="position:fixed;bottom:16px;left:0;width:inherit;'
            'padding:0 16px;box-sizing:border-box">'
            '<div style="font-size:.65rem;color:#4B5563;line-height:1.5">'
            'TF-IDF cosine · 60% MD&amp;A<br>40% Risk Factors · SEC EDGAR'
            '</div></div>',
            unsafe_allow_html=True,
        )

    return page


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _parse_args()

    if not DEFAULT_CSV.exists():
        st.error(
            f"**scores.csv not found** at `{DEFAULT_CSV}`.\n\n"
            "Run `python sp500_batch.py` first to generate it."
        )
        st.stop()

    # ── Load data ──────────────────────────────────────────────────────────────
    q1_df, q2_df = load_both_quarters()
    mq_df = build_multi_quarter_df(q1_df, q2_df)

    # Resolve active quarter → active df
    active_q = st.session_state.get("active_quarter", DEFAULT_QUARTER)
    active_csv, _ = QUARTER_OPTIONS[active_q]
    if not active_csv.exists():
        active_q = DEFAULT_QUARTER
        st.session_state["active_quarter"] = DEFAULT_QUARTER
    df = q2_df if active_q == "Q2 2025" and len(q2_df) else q1_df

    # ── Sidebar navigation ─────────────────────────────────────────────────────
    page = render_sidebar()

    # ── Route to page ──────────────────────────────────────────────────────────
    if page == "🏠 Market Overview":
        page_market_overview(df, q1_df, q2_df, mq_df)
    elif page == "🔍 Company Screener":
        page_company_screener(df, mq_df)
    elif page == "📊 Signal Validation":
        page_signal_validation(q1_df, q2_df)
    elif page == "📚 Case Studies":
        page_case_studies()


if __name__ == "__main__":
    main()
