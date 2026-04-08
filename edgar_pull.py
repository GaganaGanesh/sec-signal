"""
edgar_pull.py — Pull 10-Q filings from SEC EDGAR and extract MD&A + Risk Factors.

Usage:
    python edgar_pull.py AAPL Q1 2024
    python edgar_pull.py MSFT Q3 2023
"""

import re
import sys
import time
import textwrap
import warnings
from datetime import date, datetime, timedelta
from typing import Optional

import requests
from bs4 import BeautifulSoup, NavigableString, Tag, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# SEC requires a descriptive User-Agent with contact info
HEADERS = {
    "User-Agent": "sec-signal-research/1.0 research@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}
EFTS_HEADERS = {**HEADERS, "Host": "efts.sec.gov"}
WWW_HEADERS = {**HEADERS, "Host": "www.sec.gov"}

SEC_BASE = "https://data.sec.gov"
WWW_BASE = "https://www.sec.gov"

# Quarter → (canonical month, day) for the quarter-end date.
# We match any reportDate within ±45 days of month/day in the target year.
# This handles companies whose fiscal quarters end a few days off the calendar
# quarter boundary (e.g. Apple's Q1 ends April 1 instead of March 31).
_QUARTER_END = {
    "Q1": (3, 31),   # March 31  — window: ~Feb 14 – May 15
    "Q2": (6, 30),   # June 30   — window: ~May 15 – Aug 14
    "Q3": (9, 30),   # September 30 — window: ~Aug 15 – Nov 14
}
_QUARTER_WINDOW_DAYS = 92   # ±92 days → ~3-month tolerance; handles fiscal-year companies
# whose quarters end Jan 31 (April FY) or Dec 31 (March FY) — both >46 days from
# the nearest calendar quarter-end but within 92 days.

# Legacy alias kept for any external callers
QUARTER_MONTHS = {
    "Q1": (1, 2, 3),
    "Q2": (4, 5, 6),
    "Q3": (7, 8, 9),
}


def _quarter_window(quarter: str, year: int) -> tuple[date, date]:
    """Return (window_start, window_end) for the given quarter end in year."""
    m, d = _QUARTER_END[quarter.upper()]
    center = date(year, m, d)
    return center - timedelta(days=_QUARTER_WINDOW_DAYS), center + timedelta(days=_QUARTER_WINDOW_DAYS)

# ── Regex patterns for section headers (case-insensitive) ─────────────────────
# Each tuple: (section_name, list_of_compiled_patterns)
def _item_re(item_label: str, title_words: str) -> re.Pattern:
    r"""
    Build a flexible regex for a section header of the form:
      'Item <label>  <title_words>'
    Handles: dots, colons, em/en-dashes, non-breaking spaces (\xa0),
             and curly apostrophes in the title.
    title_words may contain spaces; each word is matched with \W+ between them.
    """
    sep = r"[\.\s\xa0:—–\-]*"  # separators between label and title
    # Replace spaces in title with flexible whitespace
    words = title_words.split()
    title_pat = r"\W{0,5}".join(re.escape(w) for w in words)
    return re.compile(rf"item\s+{re.escape(item_label)}{sep}{title_pat}", re.I)


SECTION_PATTERNS = {
    "mda": [
        _item_re("2", "Management"),          # Item 2 ... Management (catch-all)
        re.compile(r"item\s+2\W{1,20}md\s*[&and]+\s*a", re.I),
        # Standalone heading (no Item prefix) – lower priority
        re.compile(
            r"(?:^|\n)[ \t]*management\W{0,6}s\W{1,3}discussion\s+and\s+analysis",
            re.I | re.M,
        ),
    ],
    "risk_factors": [
        _item_re("1a", "Risk Factors"),       # Part II Item 1A (standard 10-Q)
        _item_re("2a", "Risk Factors"),       # some filings use Item 2A
        re.compile(r"(?:^|\n)[ \t]*risk\s+factors[ \t]*(?:\n|$)", re.I | re.M),
    ],
}

# Headers that signal the START of the NEXT section (stop sentinels).
# Anchored to line-start (after optional whitespace) to avoid matching
# inline cross-references like "…discussed in Part I, Item 1A of the 10-K…"
NEXT_SECTION_PATTERNS = [
    re.compile(r"(?:^|\n)[ \t\xa0]*item\s+\d+[a-z]?[\.\s\xa0:—–\-]", re.I | re.M),
    re.compile(r"(?:^|\n)[ \t\xa0]*part\s+(?:i{1,3}|iv|v)\b", re.I | re.M),
]


# ── CIK lookup ────────────────────────────────────────────────────────────────

def _get_cik(ticker: str) -> str:
    """Return zero-padded 10-digit CIK for a ticker symbol."""
    url = f"{WWW_BASE}/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=10-Q&dateb=&owner=include&count=1&output=atom"
    r = requests.get(url, headers=WWW_HEADERS, timeout=15)
    r.raise_for_status()
    # The CIK appears in the <company-info><cik> tag or in the company-search URL
    match = re.search(r"/cgi-bin/browse-edgar\?action=getcompany&CIK=(\d+)", r.text)
    if not match:
        # Fall back to company_tickers.json
        tj = requests.get(
            f"{WWW_BASE}/files/company_tickers.json", headers=WWW_HEADERS, timeout=15
        )
        tj.raise_for_status()
        for entry in tj.json().values():
            if entry["ticker"].upper() == ticker.upper():
                return str(entry["cik_str"]).zfill(10)
        raise ValueError(f"Cannot resolve CIK for ticker: {ticker}")
    return match.group(1).zfill(10)


# ── Submissions / filing index ────────────────────────────────────────────────

def _get_submissions(cik: str) -> dict:
    """Fetch the submissions JSON for a CIK (handles pagination)."""
    url = f"{SEC_BASE}/submissions/CIK{cik}.json"
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    data = r.json()

    # Merge paginated 'older' filings if present
    for extra_file in data.get("filings", {}).get("files", []):
        extra_url = f"{SEC_BASE}/submissions/{extra_file['name']}"
        er = requests.get(extra_url, headers=HEADERS, timeout=15)
        if er.ok:
            extra = er.json()
            for key in ("accessionNumber", "filingDate", "reportDate", "form"):
                data["filings"]["recent"][key].extend(extra.get(key, []))
    return data


def _find_10q_accession(
    submissions: dict, quarter: str, year: int
) -> tuple[str, str]:
    """
    Return (accession_number, report_date) for the 10-Q matching the
    requested quarter and year.

    Matching strategy (date-range, robust to fiscal-year companies):
      - Compute a ±46-day window around the canonical quarter-end date
        (e.g. March 31 for Q1).  This catches companies like Apple whose
        fiscal Q1 ends April 1 instead of March 31.
      - Among all 10-Q filings whose reportDate falls inside that window,
        pick the one whose reportDate is closest to the canonical quarter end.
      - Amended filings (10-Q/A) are accepted but ranked after originals.
    """
    if quarter.upper() not in _QUARTER_END:
        raise ValueError(f"Invalid quarter '{quarter}'. Use Q1, Q2, or Q3.")

    win_start, win_end = _quarter_window(quarter, year)
    m_end, d_end = _QUARTER_END[quarter.upper()]
    canonical_end = date(year, m_end, d_end)

    recent = submissions["filings"]["recent"]
    forms = recent.get("form", [])
    acc_nums = recent.get("accessionNumber", [])
    report_dates = recent.get("reportDate", [])

    candidates = []
    for form, acc, rdate in zip(forms, acc_nums, report_dates):
        if form not in ("10-Q", "10-Q/A"):
            continue
        if not rdate:
            continue
        try:
            rd = datetime.strptime(rdate, "%Y-%m-%d").date()
        except ValueError:
            continue
        if win_start <= rd <= win_end:
            distance = abs((rd - canonical_end).days)
            is_amended = 1 if form == "10-Q/A" else 0
            candidates.append((is_amended, distance, acc, rdate, rd))

    if not candidates:
        raise ValueError(
            f"No 10-Q found for {quarter} {year} "
            f"(window {win_start} – {win_end}). "
            "The filing may not yet be available, or the company uses a very "
            "non-standard fiscal year."
        )

    # Sort: originals before amendments, then by proximity to canonical quarter end
    candidates.sort(key=lambda x: (x[0], x[1]))
    _, _, acc, rdate, _ = candidates[0]
    return acc, rdate


def _filing_index_url(cik: str, accession: str) -> str:
    acc_clean = accession.replace("-", "")
    return f"{WWW_BASE}/Archives/edgar/data/{int(cik)}/{acc_clean}/{accession}-index.htm"


def _resolve_doc_href(href: str, base: str) -> str:
    """
    Resolve a document href to an absolute URL.
    Strips the SEC inline XBRL viewer prefix (/ix?doc=...) if present.
    """
    # Strip iXBRL viewer wrapper: /ix?doc=/Archives/...
    m = re.match(r"^/ix\?doc=(/Archives/.+)$", href)
    if m:
        href = m.group(1)

    if href.startswith("http"):
        return href
    if href.startswith("/"):
        return f"{WWW_BASE}{href}"
    return base + href


def _get_primary_doc_url(cik: str, accession: str) -> str:
    """Parse the filing index to find the primary 10-Q HTML/HTM document."""
    index_url = _filing_index_url(cik, accession)
    r = requests.get(index_url, headers=WWW_HEADERS, timeout=20)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml")
    acc_clean = accession.replace("-", "")
    base = f"{WWW_BASE}/Archives/edgar/data/{int(cik)}/{acc_clean}/"

    # Index table columns: [seq, description, document(link), type, size]
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 3:
            continue
        # Type is in cells[3]; document link in cells[2]
        doc_type = cells[3].get_text(strip=True) if len(cells) > 3 else ""
        link_tag = cells[2].find("a") if len(cells) > 2 else None
        if not link_tag:
            continue

        href = link_tag.get("href", "")
        # Get the real filename after stripping any iXBRL wrapper
        clean_href = re.sub(r"^/ix\?doc=", "", href)
        filename = clean_href.split("/")[-1].lower()

        if doc_type in ("10-Q", "10-Q/A") and filename.endswith((".htm", ".html")):
            return _resolve_doc_href(href, base)

    # Fallback: find any .htm that looks like the main body (not an exhibit/R-file)
    for a in soup.find_all("a", href=True):
        href = a["href"]
        clean_href = re.sub(r"^/ix\?doc=", "", href)
        name = clean_href.split("/")[-1].lower()
        if (
            name.endswith((".htm", ".html"))
            and not name.startswith("r")
            and "ex" not in name
            and "exhibit" not in name
        ):
            return _resolve_doc_href(href, base)

    raise ValueError(f"Could not locate primary 10-Q document in index: {index_url}")


# ── HTML → clean text extraction ──────────────────────────────────────────────

def _fetch_and_clean_html(url: str) -> str:
    """Download a filing document and return cleaned plain text."""
    r = requests.get(url, headers=WWW_HEADERS, timeout=60)
    r.raise_for_status()
    # html.parser handles iXBRL namespace tags (ix:nonfraction etc.) better than lxml
    soup = BeautifulSoup(r.content, "html.parser")

    # Remove scripts, styles, and hidden elements
    for tag in soup(["script", "style", "meta", "link", "head"]):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
        tag.decompose()

    # Replace <br> with newlines before get_text
    for br in soup.find_all("br"):
        br.replace_with("\n")

    # Use separator="\n" so block elements produce line breaks
    text = soup.get_text(separator="\n")

    # Collapse excessive whitespace while preserving paragraph breaks
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        lines.append(stripped)
    text = "\n".join(lines)

    # Fix mid-word line breaks common in EDGAR iXBRL → text extraction
    # e.g. "ITEM 1A. RI\nSK FACTORS" → "ITEM 1A. RISK FACTORS"
    # Match a short ALL-CAPS fragment (1-5 chars) at end of line immediately
    # followed by ALL-CAPS continuation on the next line.
    text = re.sub(r"\b([A-Z]{1,5})\n([A-Z]{2,})\b", r"\1\2", text)

    # Collapse runs of 3+ blank lines → 2 blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Section extraction ────────────────────────────────────────────────────────

def _all_section_matches(text: str, patterns: list) -> list[int]:
    """Return sorted list of all start positions matching any of the patterns."""
    positions = set()
    for pat in patterns:
        for m in pat.finditer(text):
            positions.add(m.start())
    return sorted(positions)


def _find_next_section_after(text: str, start: int) -> int:
    """
    After `start`, find the earliest next major section header and return
    its start index (or len(text) if none found).
    We skip the very first character so we don't re-match the opening header.
    """
    search_text = text[start + 1:]
    best = -1
    for pat in NEXT_SECTION_PATTERNS:
        for m in pat.finditer(search_text):
            if best == -1 or m.start() < best:
                best = m.start()
                break
    if best == -1:
        return len(text)
    return start + 1 + best


# Minimum content length to consider a match a real section (not TOC)
_MIN_SECTION_CHARS = 500


def _is_toc_block(block: str) -> bool:
    """
    Heuristic: return True if the block looks like a table-of-contents entry
    rather than real narrative prose.

    Signals of a TOC block:
      - High fraction of lines that are standalone integers (page numbers)
      - Very short average line length (titles + numbers, no prose sentences)
    """
    lines = [l.strip() for l in block.splitlines() if l.strip()]
    if not lines:
        return True
    num_only = sum(1 for l in lines if re.fullmatch(r"\d+", l))
    if num_only / len(lines) > 0.15:          # >15% page-number-only lines
        return True
    avg_len = sum(len(l) for l in lines) / len(lines)
    if avg_len < 30 and len(block) < 5000:    # short avg lines AND compact block
        return True
    return False


def extract_section(text: str, section: str) -> Optional[str]:
    """
    Extract a named section ('mda' or 'risk_factors') from full filing text.

    Strategy:
    1. Find all header matches (sorted by position).
    2. Skip matches whose extracted content looks like a TOC entry or is too
       short (<_MIN_SECTION_CHARS).
    3. If all named matches are TOC-like (e.g. JPM-style large integrated filings
       where the header appears only in the TOC), fall back to using the position
       immediately after the last TOC match as the section start, and locate
       the next Part II sentinel as the end.
    """
    patterns = SECTION_PATTERNS.get(section)
    if not patterns:
        raise ValueError(f"Unknown section '{section}'")

    starts = _all_section_matches(text, patterns)
    if not starts:
        return None

    chosen_start = -1
    chosen_end = -1

    for start in starts:
        end = _find_next_section_after(text, start)
        content = text[start:end].strip()
        if len(content) >= _MIN_SECTION_CHARS and not _is_toc_block(content):
            chosen_start = start
            chosen_end = end
            break

    # Fallback for large integrated filings (e.g. bank 10-Qs where the MD&A
    # header appears only in a nested TOC). Take everything from the character
    # after the last TOC block until the first Part-II sentinel.
    if chosen_start == -1:
        last_toc_end = -1
        for start in starts:
            end = _find_next_section_after(text, start)
            last_toc_end = max(last_toc_end, end)

        if last_toc_end != -1:
            # Advance past any blank lines
            pos = last_toc_end
            while pos < len(text) and text[pos] in ("\n", " ", "\t", "\xa0"):
                pos += 1
            end = _find_next_section_after(text, pos)
            content = text[pos:end].strip()
            if content:
                chosen_start = pos
                chosen_end = end

    if chosen_start == -1:
        return None

    raw = text[chosen_start:chosen_end].strip()

    # Strip the header line(s) from the top of the extracted block
    lines = raw.splitlines()
    content_lines = []
    skipped_header = False
    for line in lines:
        if not skipped_header:
            if any(pat.search(line) for pat in patterns):
                skipped_header = True
                continue
        content_lines.append(line)

    return "\n".join(content_lines).strip()


# ── Pretty printer ────────────────────────────────────────────────────────────

def _print_section(title: str, text: Optional[str], max_chars: int = 3000) -> None:
    width = 78
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)
    if text is None:
        print("  [Section not found — check filing format or section name variants]")
        return
    preview = text[:max_chars]
    if len(text) > max_chars:
        preview += f"\n\n  ... [{len(text) - max_chars:,} additional characters truncated] ..."
    # Wrap long lines for readability
    for para in preview.split("\n"):
        if len(para) > width:
            print(textwrap.fill(para, width=width, subsequent_indent="  "))
        else:
            print(para)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def pull_10q(ticker: str, quarter: str, year: int) -> dict:
    """
    End-to-end: resolve ticker → CIK → find 10-Q for (quarter, year) →
    download → extract MD&A and Risk Factors.

    Returns a dict with keys: ticker, quarter, year, report_date,
    accession, filing_url, mda, risk_factors.
    """
    print(f"\n[1/5] Resolving CIK for {ticker.upper()} ...")
    cik = _get_cik(ticker)
    print(f"      CIK: {cik}")
    time.sleep(0.11)  # SEC rate-limit courtesy pause

    print(f"[2/5] Fetching submissions for CIK {cik} ...")
    submissions = _get_submissions(cik)
    company_name = submissions.get("name", ticker.upper())
    print(f"      Company: {company_name}")
    time.sleep(0.11)

    print(f"[3/5] Locating 10-Q for {quarter.upper()} {year} ...")
    accession, report_date = _find_10q_accession(submissions, quarter, year)
    print(f"      Accession: {accession}  |  Period: {report_date}")
    time.sleep(0.11)

    print(f"[4/5] Fetching filing index ...")
    doc_url = _get_primary_doc_url(cik, accession)
    print(f"      Document URL: {doc_url}")
    time.sleep(0.11)

    print(f"[5/5] Downloading and parsing 10-Q document ...")
    full_text = _fetch_and_clean_html(doc_url)
    print(f"      Full text length: {len(full_text):,} characters")

    mda = extract_section(full_text, "mda")
    rf = extract_section(full_text, "risk_factors")

    return {
        "ticker": ticker.upper(),
        "quarter": quarter.upper(),
        "year": year,
        "company": company_name,
        "report_date": report_date,
        "accession": accession,
        "filing_url": doc_url,
        "mda": mda,
        "risk_factors": rf,
        "_full_text": full_text,  # retained for downstream similarity computation
    }


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        print("Example: python edgar_pull.py AAPL Q1 2024")
        sys.exit(1)

    ticker = sys.argv[1].strip().upper()
    quarter = sys.argv[2].strip().upper()
    try:
        year = int(sys.argv[3].strip())
    except ValueError:
        print(f"Error: year must be an integer, got '{sys.argv[3]}'")
        sys.exit(1)

    if quarter not in QUARTER_MONTHS:
        print(f"Error: quarter must be Q1, Q2, or Q3 (10-Qs are not filed for Q4). Got '{quarter}'")
        sys.exit(1)

    result = pull_10q(ticker, quarter, year)

    print(f"\n{'─'*78}")
    print(f"  {result['company']} ({result['ticker']})  ·  {result['quarter']} {result['year']}")
    print(f"  Period ending: {result['report_date']}")
    print(f"  Accession:     {result['accession']}")
    print(f"  Filing URL:    {result['filing_url']}")

    _print_section(
        f"ITEM 2 — MD&A  ({result['ticker']} {result['quarter']} {result['year']})",
        result["mda"],
    )
    _print_section(
        f"ITEM 1A — RISK FACTORS  ({result['ticker']} {result['quarter']} {result['year']})",
        result["risk_factors"],
    )

    mda_len = len(result["mda"]) if result["mda"] else 0
    rf_len = len(result["risk_factors"]) if result["risk_factors"] else 0
    print(f"\nExtraction summary:")
    print(f"  MD&A:          {mda_len:>8,} chars  {'✓' if result['mda'] else '✗ NOT FOUND'}")
    print(f"  Risk Factors:  {rf_len:>8,} chars  {'✓' if result['risk_factors'] else '✗ NOT FOUND'}")


if __name__ == "__main__":
    main()
