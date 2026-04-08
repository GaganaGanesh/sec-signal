"""
similarity_score.py — Year-over-year cosine similarity for 10-Q MD&A and Risk Factors.

Usage:
    python similarity_score.py AAPL Q1 2024
    python similarity_score.py MSFT Q2 2023

Compares the same quarter from the given year vs the prior year.
Scores range 0–1 (1 = identical, 0 = completely different).
Combined score = 0.60 × MDA + 0.40 × RiskFactors.
"""

import re
import sys
import textwrap
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from edgar_pull import pull_10q

# ── Weights ───────────────────────────────────────────────────────────────────
MDA_WEIGHT = 0.60
RF_WEIGHT = 0.40

# ── Text utilities ────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Light normalisation: collapse whitespace, strip page-number artifacts."""
    text = re.sub(r"\s+", " ", text)
    # Remove bare page numbers (e.g. "Apple Inc. | Q2 2024 Form 10-Q | 19")
    text = re.sub(r"[A-Za-z0-9 \.]+\|\s*Q\d \d{4}[^\|]*\|\s*\d+", " ", text)
    return text.strip()


_SENT_SPLIT = re.compile(
    r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+(?=[A-Z\"])",
)

def _sentences(text: str) -> list[str]:
    """
    Split text into sentences. Uses a simple heuristic regex that handles
    common abbreviations (e.g. "U.S.", "Inc.", "No.") without splitting.
    Returns only sentences with at least 8 words.
    """
    text = _clean(text)
    raw = _SENT_SPLIT.split(text)
    return [s.strip() for s in raw if len(s.split()) >= 8]


# ── Cosine similarity ─────────────────────────────────────────────────────────

def _doc_cosine(text_a: str, text_b: str) -> float:
    """
    Compute TF-IDF cosine similarity between two documents.
    Returns a float in [0, 1]. Returns 0.0 if either document is empty.
    """
    if not text_a or not text_b:
        return 0.0
    vec = TfidfVectorizer(
        strip_accents="unicode",
        analyzer="word",
        ngram_range=(1, 2),   # unigrams + bigrams capture phrasing changes
        min_df=1,
        sublinear_tf=True,    # log-normalised term frequency
    )
    try:
        tfidf = vec.fit_transform([_clean(text_a), _clean(text_b)])
    except ValueError:
        return 0.0
    return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0, 0])


# ── Sentence-level change detection ──────────────────────────────────────────

@dataclass
class SentenceChange:
    """Represents a sentence-level change between the prior and current filing."""
    change_type: str          # "added", "removed", or "modified"
    prior_sentence: str       # "" if added
    current_sentence: str     # "" if removed
    similarity: float         # 1 = unchanged, 0 = completely new/gone
    delta: float              # 1 - similarity (higher = more changed)


def _sentence_changes(prior_text: str, current_text: str) -> list[SentenceChange]:
    """
    Align sentences between two versions of the same section using TF-IDF
    cosine similarity.  Returns a list of SentenceChange objects sorted by
    delta descending (most changed first).

    Algorithm:
      1. Vectorise all sentences from both documents together.
      2. Build a similarity matrix: prior_sents × current_sents.
      3. For each prior sentence, find its best match in the current text.
      4. For each current sentence, find its best match in the prior text.
      5. Mark pairs with sim < 0.30 as added/removed; pairs with 0.30–0.90
         as modified; pairs >= 0.90 as unchanged (excluded from top-5).
    """
    prior_sents = _sentences(prior_text)
    current_sents = _sentences(current_text)

    if not prior_sents and not current_sents:
        return []

    all_sents = prior_sents + current_sents
    n_prior = len(prior_sents)

    if len(all_sents) < 2:
        return []

    vec = TfidfVectorizer(
        strip_accents="unicode",
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    try:
        tfidf = vec.fit_transform(all_sents)
    except ValueError:
        return []

    prior_mat = tfidf[:n_prior]
    current_mat = tfidf[n_prior:]

    # sim_matrix[i, j] = similarity between prior_sents[i] and current_sents[j]
    if prior_mat.shape[0] == 0 or current_mat.shape[0] == 0:
        changes = []
        for s in current_sents:
            changes.append(SentenceChange("added", "", s, 0.0, 1.0))
        for s in prior_sents:
            changes.append(SentenceChange("removed", s, "", 0.0, 1.0))
        return sorted(changes, key=lambda c: c.delta, reverse=True)

    sim_matrix = cosine_similarity(prior_mat, current_mat)  # (n_prior, n_current)

    changes: list[SentenceChange] = []
    matched_current = set()

    for i, prior_sent in enumerate(prior_sents):
        j_best = int(np.argmax(sim_matrix[i]))
        best_sim = float(sim_matrix[i, j_best])

        if best_sim < 0.25:
            # Sentence effectively removed
            changes.append(SentenceChange("removed", prior_sent, "", 0.0, 1.0))
        elif best_sim < 0.90:
            # Sentence modified
            matched_current.add(j_best)
            changes.append(SentenceChange(
                "modified", prior_sent, current_sents[j_best], best_sim, 1.0 - best_sim
            ))
        else:
            # Essentially unchanged — still track with low delta
            matched_current.add(j_best)
            changes.append(SentenceChange(
                "modified", prior_sent, current_sents[j_best], best_sim, 1.0 - best_sim
            ))

    # New sentences in current not matched to any prior sentence
    for j, cur_sent in enumerate(current_sents):
        if j not in matched_current:
            # Check if it's genuinely new
            i_best = int(np.argmax(sim_matrix[:, j]))
            best_sim = float(sim_matrix[i_best, j])
            if best_sim < 0.25:
                changes.append(SentenceChange("added", "", cur_sent, 0.0, 1.0))

    return sorted(changes, key=lambda c: c.delta, reverse=True)


# ── Pretty printer ────────────────────────────────────────────────────────────

WIDTH = 78

def _wrap(text: str, indent: str = "  ") -> str:
    if not text:
        return ""
    return "\n".join(
        textwrap.fill(line, width=WIDTH, initial_indent=indent, subsequent_indent=indent)
        if len(line) > WIDTH else indent + line
        for line in text.splitlines()
        if line.strip()
    )


def _bar(score: float, width: int = 30) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


def _print_scores(
    ticker: str,
    quarter: str,
    prior_year: int,
    current_year: int,
    mda_score: Optional[float],
    rf_score: Optional[float],
    combined: float,
) -> None:
    print(f"\n{'═' * WIDTH}")
    print(f"  SIMILARITY REPORT  —  {ticker}  {quarter} {current_year} vs {quarter} {prior_year}")
    print(f"{'═' * WIDTH}")

    def _fmt(label: str, score: Optional[float], weight: str) -> None:
        if score is None:
            print(f"  {label:<22} N/A  (section missing in one or both filings)")
            return
        bar = _bar(score)
        pct = f"{score:.1%}"
        print(f"  {label:<22} {pct:>6}  {bar}  (weight {weight})")

    _fmt("MD&A similarity",         mda_score, "60%")
    _fmt("Risk Factors similarity",  rf_score,  "40%")
    print(f"  {'─' * (WIDTH - 2)}")
    bar = _bar(combined)
    print(f"  {'Combined score':<22} {combined:.1%}  {bar}")
    print()


def _print_top_changes(
    label: str,
    changes: list[SentenceChange],
    prior_year: int,
    current_year: int,
    top_n: int = 5,
) -> None:
    print(f"\n{'─' * WIDTH}")
    print(f"  TOP {top_n} MOST CHANGED SENTENCES — {label}")
    print(f"{'─' * WIDTH}")

    shown = 0
    for ch in changes:
        if shown >= top_n:
            break
        # Skip trivially changed (e.g. only a year number flipped)
        print(f"\n  [{shown + 1}] Change type: {ch.change_type.upper()}  "
              f"(similarity: {ch.similarity:.1%}  Δ={ch.delta:.1%})")

        if ch.change_type == "removed":
            print(f"\n  {prior_year} text (REMOVED):")
            print(_wrap(ch.prior_sentence, "    "))
            print(f"\n  {current_year} text:")
            print("    (no equivalent sentence found)")
        elif ch.change_type == "added":
            print(f"\n  {prior_year} text:")
            print("    (no equivalent sentence found)")
            print(f"\n  {current_year} text (NEW):")
            print(_wrap(ch.current_sentence, "    "))
        else:
            print(f"\n  {prior_year}:  {_wrap(ch.prior_sentence, '    ').lstrip()}")
            print(f"\n  {current_year}:  {_wrap(ch.current_sentence, '    ').lstrip()}")

        shown += 1

    if shown == 0:
        print("  No significant sentence-level changes detected.")


# ── Main ──────────────────────────────────────────────────────────────────────

@dataclass
class SimilarityResult:
    ticker: str
    quarter: str
    prior_year: int
    current_year: int
    mda_score: Optional[float]
    rf_score: Optional[float]
    combined_score: float
    mda_changes: list[SentenceChange]
    rf_changes: list[SentenceChange]
    extraction_status: str = "complete"  # "complete" | "partial" | "failed"


def compute_similarity(ticker: str, quarter: str, current_year: int) -> SimilarityResult:
    prior_year = current_year - 1

    print(f"\n{'─' * WIDTH}")
    print(f"  Fetching {ticker} {quarter} {current_year}  (current year) ...")
    print(f"{'─' * WIDTH}")
    current = pull_10q(ticker, quarter, current_year)

    print(f"\n{'─' * WIDTH}")
    print(f"  Fetching {ticker} {quarter} {prior_year}  (prior year) ...")
    print(f"{'─' * WIDTH}")
    prior = pull_10q(ticker, quarter, prior_year)

    print(f"\n  Computing similarity scores ...")

    # ── Document-level cosine similarity ──────────────────────────────────────
    mda_score: Optional[float] = None
    if current["mda"] and prior["mda"]:
        mda_score = _doc_cosine(prior["mda"], current["mda"])
    elif not current["mda"] or not prior["mda"]:
        print("  Warning: MD&A missing in one or both filings — score N/A")

    rf_score: Optional[float] = None
    if current["risk_factors"] and prior["risk_factors"]:
        rf_score = _doc_cosine(prior["risk_factors"], current["risk_factors"])
    elif not current["risk_factors"] or not prior["risk_factors"]:
        print("  Warning: Risk Factors missing in one or both filings — score N/A")

    # ── Combined weighted score ───────────────────────────────────────────────
    # If a section is missing, redistribute its weight to the other
    if mda_score is not None and rf_score is not None:
        combined = MDA_WEIGHT * mda_score + RF_WEIGHT * rf_score
        extraction_status = "complete"
    elif mda_score is not None:
        combined = mda_score
        extraction_status = "partial"
    elif rf_score is not None:
        combined = rf_score
        extraction_status = "partial"
    else:
        combined = 0.0
        extraction_status = "failed"

    # ── Sentence-level changes ────────────────────────────────────────────────
    mda_changes = _sentence_changes(
        prior["mda"] or "", current["mda"] or ""
    )
    rf_changes = _sentence_changes(
        prior["risk_factors"] or "", current["risk_factors"] or ""
    )

    return SimilarityResult(
        ticker=ticker.upper(),
        quarter=quarter.upper(),
        prior_year=prior_year,
        current_year=current_year,
        mda_score=mda_score,
        rf_score=rf_score,
        combined_score=combined,
        mda_changes=mda_changes,
        rf_changes=rf_changes,
        extraction_status=extraction_status,
    )


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        print("Example: python similarity_score.py AAPL Q1 2024")
        sys.exit(1)

    ticker = sys.argv[1].strip().upper()
    quarter = sys.argv[2].strip().upper()
    try:
        current_year = int(sys.argv[3].strip())
    except ValueError:
        print(f"Error: year must be an integer, got '{sys.argv[3]}'")
        sys.exit(1)

    result = compute_similarity(ticker, quarter, current_year)

    _print_scores(
        result.ticker,
        result.quarter,
        result.prior_year,
        result.current_year,
        result.mda_score,
        result.rf_score,
        result.combined_score,
    )

    _print_top_changes(
        f"MD&A  ({result.ticker} {result.quarter})",
        result.mda_changes,
        result.prior_year,
        result.current_year,
    )

    _print_top_changes(
        f"Risk Factors  ({result.ticker} {result.quarter})",
        result.rf_changes,
        result.prior_year,
        result.current_year,
    )

    print(f"\n{'═' * WIDTH}")
    print(f"  Done.  Combined score: {result.combined_score:.4f}")
    print(f"{'═' * WIDTH}\n")


if __name__ == "__main__":
    main()
