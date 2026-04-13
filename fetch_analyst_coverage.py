"""
fetch_analyst_coverage.py — Pull analyst coverage counts from yfinance and cache them.

Usage:
    python fetch_analyst_coverage.py          # fetch all tickers from scores*.csv
    python fetch_analyst_coverage.py --refresh  # force re-fetch even if cache exists

Outputs:
    analyst_coverage.json  — {ticker: numberOfAnalystOpinions, ...}
                              Missing / errored tickers are stored as null so we
                              know we tried (and won't re-fetch on next run).
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
COVERAGE_JSON = ROOT / "analyst_coverage.json"
DEFAULT_COVERAGE = 15   # median assumption when yfinance fails


def _collect_tickers() -> list[str]:
    """Union of tickers across scores.csv and scores_q2.csv."""
    try:
        import pandas as pd
    except ImportError:
        sys.exit("pandas not installed — activate your venv first.")

    tickers: set[str] = set()
    for fname in ("scores.csv", "scores_q2.csv"):
        p = ROOT / fname
        if p.exists():
            df = pd.read_csv(p, usecols=["ticker"])
            tickers.update(df["ticker"].dropna().str.strip().unique())
    if not tickers:
        sys.exit("No scores CSV files found in this directory.")
    return sorted(tickers)


def _load_cache() -> dict:
    if COVERAGE_JSON.exists():
        with open(COVERAGE_JSON) as f:
            return json.load(f)
    return {}


def _save_cache(data: dict) -> None:
    with open(COVERAGE_JSON, "w") as f:
        json.dump(data, f, indent=2)


def _fetch_one(ticker: str) -> int | None:
    """Return numberOfAnalystOpinions or None on failure."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        val = info.get("numberOfAnalystOpinions")
        if val is not None:
            return int(val)
        return None
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="Re-fetch all tickers, ignoring existing cache")
    args = ap.parse_args()

    tickers = _collect_tickers()
    cache = {} if args.refresh else _load_cache()

    to_fetch = [t for t in tickers if t not in cache]
    already_cached = len(tickers) - len(to_fetch)

    print(f"\nAnalyst coverage fetch")
    print(f"  Total unique tickers : {len(tickers)}")
    print(f"  Already cached       : {already_cached}")
    print(f"  To fetch             : {len(to_fetch)}")
    print()

    if not to_fetch:
        print("Cache is complete — nothing to fetch.")
        _print_summary(cache, tickers)
        return

    success = 0
    failed  = 0

    for i, ticker in enumerate(to_fetch, 1):
        val = _fetch_one(ticker)
        cache[ticker] = val          # None stored so we don't retry on next run
        status = f"{val:>3d}" if val is not None else "N/A"
        flag   = "✓" if val is not None else "✗"
        print(f"  [{i:>3}/{len(to_fetch)}] {flag} {ticker:<6}  analysts: {status}")
        if val is not None:
            success += 1
        else:
            failed += 1

        # Save incrementally every 25 tickers so progress isn't lost on interrupt
        if i % 25 == 0:
            _save_cache(cache)

        # yfinance is best-effort but let's be gentle
        time.sleep(0.25)

    _save_cache(cache)

    print()
    print(f"─" * 40)
    print(f"  Succeeded : {already_cached + success} / {len(tickers)}")
    print(f"  Failed    : {failed}  (will use default={DEFAULT_COVERAGE})")
    print(f"  Saved to  : {COVERAGE_JSON.name}")
    print()
    _print_summary(cache, tickers)


def _print_summary(cache: dict, tickers: list[str]) -> None:
    vals = [cache[t] for t in tickers if cache.get(t) is not None]
    if not vals:
        print("  No coverage data available yet.")
        return
    print(f"Coverage summary across {len(vals)} tickers:")
    print(f"  Min     : {min(vals)}")
    print(f"  Median  : {sorted(vals)[len(vals)//2]}")
    print(f"  Max     : {max(vals)}")
    under_10 = sum(1 for v in vals if v < 10)
    over_25  = sum(1 for v in vals if v > 25)
    print(f"  < 10    : {under_10}  (confidence boost candidates)")
    print(f"  > 25    : {over_25}  (confidence penalty candidates)")


if __name__ == "__main__":
    main()
