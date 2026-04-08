"""
sp500_batch.py — Batch year-over-year 10-Q similarity scoring for S&P 500 companies.

Usage:
    python sp500_batch.py                        # auto-detect quarter per company
    python sp500_batch.py --quarter Q1 2024      # force a specific quarter
    python sp500_batch.py --workers 3            # parallel workers (default: 3)
    python sp500_batch.py --limit 20             # process first N tickers only (testing)
    python sp500_batch.py --resume               # skip tickers already in scores.csv
    python sp500_batch.py --output my_scores.csv # custom output file

Output: scores.csv with columns:
    ticker, company, quarter, year, prior_year,
    mda_score, risk_score, combined_score,
    mda_zscore, risk_zscore, combined_zscore,
    status, error, date_run
"""

import argparse
import csv
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from bs4 import BeautifulSoup

from edgar_pull import (
    _get_cik,
    _get_submissions,
    _QUARTER_END,
    _QUARTER_WINDOW_DAYS,
    WWW_HEADERS,
)
from similarity_score import compute_similarity, SimilarityResult

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("batch_run.log", mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

OUTPUT_FILE = Path("scores.csv")

CSV_FIELDS = [
    "ticker", "company", "quarter", "year", "prior_year",
    "mda_score", "risk_score", "combined_score",
    "mda_zscore", "risk_zscore", "combined_zscore",
    "status", "error", "date_run", "extraction_status",
]

# SEC rate-limit courtesy: minimum seconds between starting new ticker jobs.
# With 3 workers each making ~10 HTTP calls, this keeps us well under 10 req/s.
_TICKER_START_INTERVAL = 1.0  # seconds


# ── S&P 500 ticker list ───────────────────────────────────────────────────────

def fetch_sp500_tickers() -> list[tuple[str, str]]:
    """
    Scrape the current S&P 500 constituent list from Wikipedia.
    Returns a list of (ticker, company_name) tuples.

    Wikipedia table at:
    https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
    Column 0 = Symbol, Column 1 = Security (company name)

    Note: Wikipedia uses dots in tickers (BRK.B) but SEC uses dashes (BRK-B).
    We normalise dots → dashes for SEC API compatibility.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    log.info("Fetching S&P 500 ticker list from Wikipedia ...")
    r = requests.get(url, headers={"User-Agent": "sec-signal-research/1.0"}, timeout=20)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml")
    table = soup.find("table", {"id": "constituents"})
    if not table:
        # Fallback: find the first wikitable with a "Symbol" header
        for tbl in soup.find_all("table", class_="wikitable"):
            headers = [th.get_text(strip=True) for th in tbl.find_all("th")]
            if "Symbol" in headers:
                table = tbl
                break

    if not table:
        raise RuntimeError("Could not locate S&P 500 table on Wikipedia page")

    tickers = []
    for row in table.find_all("tr")[1:]:  # skip header row
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        symbol = cells[0].get_text(strip=True).replace(".", "-")  # BRK.B → BRK-B
        company = cells[1].get_text(strip=True)
        if symbol:
            tickers.append((symbol, company))

    log.info(f"Found {len(tickers)} tickers.")
    return tickers


# ── Quarter auto-detection ────────────────────────────────────────────────────

def _infer_quarter_year(report_date: date) -> tuple[str, int]:
    """
    Given a 10-Q period-end date, return the calendar (quarter, year).
    Uses the same ±46-day window as edgar_pull._quarter_window().
    Tries the report_date's own year first, then ±1 year.
    """
    for yr in [report_date.year, report_date.year - 1, report_date.year + 1]:
        for q, (m, d) in _QUARTER_END.items():
            try:
                center = date(yr, m, d)
            except ValueError:
                continue
            if abs((report_date - center).days) <= _QUARTER_WINDOW_DAYS:
                return q, yr

    # Hard fallback by month
    if report_date.month <= 4:
        return "Q1", report_date.year
    elif report_date.month <= 7:
        return "Q2", report_date.year
    else:
        return "Q3", report_date.year


def detect_most_recent_quarter(
    submissions: dict, cutoff: Optional[date] = None
) -> tuple[str, int]:
    """
    Find the most recently *filed* 10-Q in the submissions data (not after
    `cutoff`, default today) and return the inferred (quarter, year).
    Prefers original filings (10-Q) over amendments (10-Q/A).
    """
    if cutoff is None:
        cutoff = date.today()

    recent = submissions["filings"]["recent"]
    forms = recent.get("form", [])
    report_dates = recent.get("reportDate", [])
    filing_dates = recent.get("filingDate", [])

    best = None   # (filing_date, is_amendment, report_date)

    for form, rdate, fdate in zip(forms, report_dates, filing_dates):
        if form not in ("10-Q", "10-Q/A"):
            continue
        if not rdate or not fdate:
            continue
        try:
            fd = datetime.strptime(fdate, "%Y-%m-%d").date()
            rd = datetime.strptime(rdate, "%Y-%m-%d").date()
        except ValueError:
            continue
        if fd > cutoff:
            continue
        is_amendment = 1 if form == "10-Q/A" else 0
        if best is None or (fd, -is_amendment) > (best[0], -best[1]):
            best = (fd, is_amendment, rd)

    if best is None:
        raise ValueError("No 10-Q filings found")

    return _infer_quarter_year(best[2])


# ── Per-ticker result ─────────────────────────────────────────────────────────

@dataclass
class TickerResult:
    ticker: str
    company: str
    quarter: str = ""
    year: int = 0
    prior_year: int = 0
    mda_score: Optional[float] = None
    risk_score: Optional[float] = None
    combined_score: Optional[float] = None
    mda_zscore: Optional[float] = None
    risk_zscore: Optional[float] = None
    combined_zscore: Optional[float] = None
    status: str = "pending"
    error: str = ""
    date_run: str = ""
    extraction_status: str = "unknown"


# ── Rate limiter ──────────────────────────────────────────────────────────────

class _RateLimiter:
    """
    Simple token-bucket rate limiter.
    Ensures at least `interval` seconds between calls to acquire().
    Thread-safe.
    """
    def __init__(self, interval: float):
        self._interval = interval
        self._lock = threading.Lock()
        self._last = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._interval - (now - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()


# ── Per-ticker worker ─────────────────────────────────────────────────────────

def _process_ticker(
    ticker: str,
    company: str,
    forced_quarter: Optional[str],
    forced_year: Optional[int],
    rate_limiter: _RateLimiter,
) -> TickerResult:
    """
    Full pipeline for one ticker:
      1. Throttle against SEC rate limit
      2. Fetch submissions to determine the most recent quarter (unless forced)
      3. Run year-over-year similarity
      4. Return a TickerResult

    All exceptions are caught; failures set status="failed" with the error message.
    """
    result = TickerResult(
        ticker=ticker,
        company=company,
        date_run=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    try:
        rate_limiter.acquire()

        if forced_quarter and forced_year:
            quarter = forced_quarter
            year = forced_year
        else:
            # Auto-detect: resolve CIK → fetch submissions → infer latest quarter
            cik = _get_cik(ticker)
            time.sleep(0.12)
            subs = _get_submissions(cik)
            quarter, year = detect_most_recent_quarter(subs)
            time.sleep(0.12)

        result.quarter = quarter
        result.year = year
        result.prior_year = year - 1

        sim: SimilarityResult = compute_similarity(ticker, quarter, year)

        result.mda_score = round(sim.mda_score, 6) if sim.mda_score is not None else None
        result.risk_score = round(sim.rf_score, 6) if sim.rf_score is not None else None
        result.combined_score = round(sim.combined_score, 6)
        result.extraction_status = sim.extraction_status
        result.status = "ok"

    except Exception as exc:
        result.status = "failed"
        result.error = str(exc)[:300]
        log.warning(f"  FAILED {ticker}: {exc}")

    return result


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _load_existing(path: Path) -> set[str]:
    """Return the set of tickers already in the CSV with status='ok'."""
    done: set[str] = set()
    if not path.exists():
        return done
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "ok":
                done.add(row["ticker"])
    return done


def _append_rows(path: Path, rows: list[TickerResult]) -> None:
    """Append TickerResult rows to the CSV (write header if file is new)."""
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow({
                "ticker":         r.ticker,
                "company":        r.company,
                "quarter":        r.quarter,
                "year":           r.year,
                "prior_year":     r.prior_year,
                "mda_score":      "" if r.mda_score is None else r.mda_score,
                "risk_score":     "" if r.risk_score is None else r.risk_score,
                "combined_score": "" if r.combined_score is None else r.combined_score,
                "mda_zscore":     "" if r.mda_zscore is None else r.mda_zscore,
                "risk_zscore":    "" if r.risk_zscore is None else r.risk_zscore,
                "combined_zscore":"" if r.combined_zscore is None else r.combined_zscore,
                "extraction_status": r.extraction_status,
                "status":         r.status,
                "error":          r.error,
                "date_run":       r.date_run,
            })


def _rewrite_with_zscores(path: Path) -> None:
    """
    Read all rows from the CSV, compute z-scores across successful runs,
    write the updated CSV back to disk.

    Z-scores are computed per column across all rows where status='ok'.
    Rows with status!='ok' get empty z-score cells.
    """
    if not path.exists():
        return

    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return

    score_cols = ("mda_score", "risk_score", "combined_score")
    zscore_cols = ("mda_zscore", "risk_zscore", "combined_zscore")

    # Collect values for each score column (only successful rows)
    for score_col, zscore_col in zip(score_cols, zscore_cols):
        values = []
        indices = []
        for i, row in enumerate(rows):
            if row.get("status") == "ok" and row.get(score_col) not in ("", None):
                try:
                    values.append(float(row[score_col]))
                    indices.append(i)
                except ValueError:
                    pass

        if len(values) < 2:
            continue

        arr = np.array(values)
        mu = arr.mean()
        sigma = arr.std(ddof=1)

        for i, v in zip(indices, arr):
            z = (v - mu) / sigma if sigma > 0 else 0.0
            rows[i][zscore_col] = f"{z:.6f}"

    # Rewrite — preserve any extra columns (e.g. sector, extraction_status) not in CSV_FIELDS
    all_fields = list(rows[0].keys()) if rows else CSV_FIELDS
    # Ensure CSV_FIELDS columns come first, extra columns appended
    ordered_fields = CSV_FIELDS + [c for c in all_fields if c not in CSV_FIELDS]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in ordered_fields})

    log.info(f"Z-scores written to {path}")


# ── Progress display ──────────────────────────────────────────────────────────

class _Progress:
    """Thread-safe progress tracker."""

    def __init__(self, total: int):
        self.total = total
        self._done = 0
        self._ok = 0
        self._failed = 0
        self._lock = threading.Lock()
        self._start = time.monotonic()

    def update(self, status: str) -> None:
        with self._lock:
            self._done += 1
            if status == "ok":
                self._ok += 1
            else:
                self._failed += 1

    def log_line(self, ticker: str, status: str, detail: str = "") -> None:
        with self._lock:
            elapsed = time.monotonic() - self._start
            pct = 100 * self._done / self.total
            eta_s = ""
            if self._done > 0:
                rate = self._done / elapsed
                remaining = (self.total - self._done) / rate
                eta_s = f"  ETA {remaining/60:.0f}m"
            icon = "✓" if status == "ok" else "✗"
            log.info(
                f"[{self._done:>3}/{self.total}  {pct:5.1f}%{eta_s}]  "
                f"{icon} {ticker:<10}  {detail}"
            )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quarter",  nargs=2,   metavar=("Q", "YEAR"),
                        help="Force a specific quarter, e.g. --quarter Q1 2024")
    parser.add_argument("--workers",  type=int,  default=3,
                        help="Parallel worker threads (default: 3). "
                             "Higher values risk hitting SEC rate limits.")
    parser.add_argument("--limit",    type=int,  default=None,
                        help="Process only first N tickers (useful for testing)")
    parser.add_argument("--resume",   action="store_true",
                        help="Skip tickers already in scores.csv with status=ok")
    parser.add_argument("--output",   type=Path, default=OUTPUT_FILE,
                        help=f"Output CSV path (default: {OUTPUT_FILE})")
    args = parser.parse_args()

    # ── Parse forced quarter ───────────────────────────────────────────────────
    forced_quarter: Optional[str] = None
    forced_year: Optional[int] = None
    if args.quarter:
        forced_quarter = args.quarter[0].upper()
        try:
            forced_year = int(args.quarter[1])
        except ValueError:
            log.error(f"Invalid year: {args.quarter[1]}")
            sys.exit(1)
        if forced_quarter not in _QUARTER_END:
            log.error(f"Invalid quarter: {forced_quarter}. Use Q1, Q2, or Q3.")
            sys.exit(1)
        log.info(f"Forced quarter: {forced_quarter} {forced_year} (prior: {forced_year - 1})")

    # ── Fetch ticker list ──────────────────────────────────────────────────────
    try:
        all_tickers = fetch_sp500_tickers()
    except Exception as e:
        log.error(f"Failed to fetch S&P 500 list: {e}")
        sys.exit(1)

    if args.limit:
        all_tickers = all_tickers[: args.limit]
        log.info(f"Limited to first {args.limit} tickers.")

    # ── Resume: skip already-completed tickers ─────────────────────────────────
    if args.resume:
        done_set = _load_existing(args.output)
        before = len(all_tickers)
        all_tickers = [(t, c) for t, c in all_tickers if t not in done_set]
        log.info(f"Resume: skipping {before - len(all_tickers)} already-done tickers. "
                 f"{len(all_tickers)} remaining.")

    if not all_tickers:
        log.info("Nothing to process — all tickers already in output file.")
        return

    log.info(
        f"Starting batch: {len(all_tickers)} tickers  |  "
        f"workers={args.workers}  |  output={args.output}"
    )

    # ── Batch processing ───────────────────────────────────────────────────────
    rate_limiter = _RateLimiter(_TICKER_START_INTERVAL)
    progress = _Progress(len(all_tickers))
    flush_every = 10  # write to CSV every N completed tickers
    pending_rows: list[TickerResult] = []
    pending_lock = threading.Lock()

    def _flush(force: bool = False) -> None:
        with pending_lock:
            if force or len(pending_rows) >= flush_every:
                if pending_rows:
                    _append_rows(args.output, pending_rows)
                    pending_rows.clear()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _process_ticker,
                ticker, company, forced_quarter, forced_year, rate_limiter
            ): (ticker, company)
            for ticker, company in all_tickers
        }

        for future in as_completed(futures):
            ticker, company = futures[future]
            try:
                result: TickerResult = future.result()
            except Exception as exc:
                # Shouldn't reach here (worker catches all exceptions), but be safe
                result = TickerResult(
                    ticker=ticker, company=company, status="failed",
                    error=str(exc)[:300],
                    date_run=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                )

            progress.update(result.status)

            detail = (
                f"combined={result.combined_score:.4f}  "
                f"mda={result.mda_score if result.mda_score is not None else 'N/A'}  "
                f"rf={result.risk_score if result.risk_score is not None else 'N/A'}  "
                f"[{result.quarter} {result.year}]"
                if result.status == "ok"
                else f"ERROR: {result.error[:80]}"
            )
            progress.log_line(ticker, result.status, detail)

            with pending_lock:
                pending_rows.append(result)
            _flush()

    _flush(force=True)

    # ── Final summary ──────────────────────────────────────────────────────────
    elapsed = time.monotonic() - progress._start
    log.info(
        f"\nBatch complete: {progress._ok} ok, {progress._failed} failed "
        f"out of {progress.total} tickers in {elapsed/60:.1f}m"
    )

    # ── Compute and write z-scores ─────────────────────────────────────────────
    log.info("Computing cross-company z-scores ...")
    _rewrite_with_zscores(args.output)

    # ── Print top 10 most-changed companies ───────────────────────────────────
    _print_leaderboard(args.output)


def _print_leaderboard(path: Path) -> None:
    """Print the 10 most- and least-changed companies by combined z-score."""
    if not path.exists():
        return

    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "ok" and row.get("combined_zscore"):
                try:
                    rows.append((row["ticker"], row["company"],
                                 float(row["combined_zscore"]),
                                 float(row.get("combined_score", 0))))
                except ValueError:
                    pass

    if not rows:
        return

    rows.sort(key=lambda x: x[2])  # ascending z-score = most changed

    w = 72
    print(f"\n{'═' * w}")
    print(f"  MOST CHANGED  (lowest combined similarity z-score)")
    print(f"{'═' * w}")
    print(f"  {'Ticker':<8}  {'Company':<35}  {'Score':>6}  {'Z-score':>8}")
    print(f"  {'─'*6}  {'─'*35}  {'─'*6}  {'─'*8}")
    for ticker, company, z, score in rows[:10]:
        print(f"  {ticker:<8}  {company[:35]:<35}  {score:6.4f}  {z:+8.3f}")

    print(f"\n{'═' * w}")
    print(f"  LEAST CHANGED  (highest combined similarity z-score)")
    print(f"{'═' * w}")
    print(f"  {'Ticker':<8}  {'Company':<35}  {'Score':>6}  {'Z-score':>8}")
    print(f"  {'─'*6}  {'─'*35}  {'─'*6}  {'─'*8}")
    for ticker, company, z, score in rows[-10:][::-1]:
        print(f"  {ticker:<8}  {company[:35]:<35}  {score:6.4f}  {z:+8.3f}")
    print()


if __name__ == "__main__":
    main()
