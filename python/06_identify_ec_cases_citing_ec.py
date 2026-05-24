# Original notebook: 06_identify_ec_cases_citing_ec.ipynb
# Converted to Python script on: 2026-05-24
# Outputs and markdown cells have been removed.
# Code logic has been preserved as closely as possible.

# --- Cell 1: 1. Imports and Configuration ---
# 1. Imports and Configuration
import io
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
import pypdf

# ── Paths ──────────────────────────────────────────────────────────────────────────────
DATA_DIR       = Path("data/processed")
EC_MASTER_PATH = DATA_DIR / "ec_antitrust_master.csv"
OUTPUT_PATH    = DATA_DIR / "ec_ec_case_matches.csv"

# ── HTTP settings ──────────────────────────────────────────────────────────────────
REQUEST_TIMEOUT = 30   # seconds per request
REQUEST_DELAY   = 1.0  # seconds between requests
MAX_RETRIES     = 2

# ── Context window around each match (characters) ─────────────────────────────────
CONTEXT_CHARS = 200

# document_type values that should be treated as PDF
PDF_DOCUMENT_TYPES = {"case", "decision"}

print("Configuration loaded.")

# --- Cell 2: 2. Text Normalization and Extraction Helpers ---
# 2. Text Normalization and Extraction Helpers
def normalize_text(raw: str) -> str:
    """Normalize Unicode, collapse whitespace, unify hyphens/separators."""
    text = unicodedata.normalize("NFKC", raw)
    # Unify various dash/hyphen characters to a plain hyphen
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_html(html_bytes: bytes) -> str:
    """Parse HTML, strip scripts/styles, return visible text."""
    soup = BeautifulSoup(html_bytes, "html.parser")
    for tag in soup(["script", "style", "head", "meta", "link"]):
        tag.decompose()
    raw = soup.get_text(separator=" ")
    return normalize_text(raw)


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using pypdf (no OCR)."""
    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        parts = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            parts.append(page_text)
        raw = "\n".join(parts)
        return normalize_text(raw)
    except Exception:
        return ""


print("Text extraction helpers defined.")

# --- Cell 3: 3. Document Fetching ---
# 3. Document Fetching
HEADERS = {
    "User-Agent": "Mozilla/5.0 (research; masterarbeit) compatible",
}


def _get_with_retry(url: str) -> requests.Response | None:
    """GET a URL with simple retry logic. Returns None on failure."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (404, 403, 410):
                return None
        except requests.RequestException:
            pass
        if attempt < MAX_RETRIES:
            time.sleep(REQUEST_DELAY * 2)
    return None


def fetch_document_text(document_url: str, document_type: str) -> tuple[str, str]:
    """
    Fetch and extract text from a document URL.

    - If document_type is 'case' or 'decision': treat as PDF.
    - Otherwise: treat as HTML.

    Returns (text, processing_status). text is empty string on failure.
    """
    time.sleep(REQUEST_DELAY)

    resp = _get_with_retry(document_url)
    if resp is None:
        return "", "fetch_failed"

    is_pdf = document_type.strip().lower() in PDF_DOCUMENT_TYPES

    if is_pdf:
        text = extract_text_from_pdf(resp.content)
        if not text or len(text) < 100:
            return "", "pdf_no_text"
        return text, "ok_pdf"
    else:
        text = extract_text_from_html(resp.content)
        if not text or len(text) < 100:
            return "", "html_no_text"
        return text, "ok_html"


print("Document fetching functions defined.")

# --- Cell 4: 4. Regex Pattern Builder for EC Cases ---
# 4. Regex Pattern Builder for EC Cases
# Same logic as in `05_identify_cjeu_cases_citing_ec.ipynb`.
def _escape(s: str) -> str:
    """Regex-escape a string."""
    return re.escape(s)


# ── EC case-number prefix families ────────────────────────────────────────────────────────────────────────────
# All three prefixes (AT, IV, COMP) are always treated as interchangeable.
_ALL_PREFIXES = r"(?:AT|IV|COMP)"

_SEP = r"[\s./\-]*"   # flexible separator: space, dot, slash, hyphen (zero or more)


def _split_ec_case_number(ec_case_number: str) -> tuple[str, list[str]]:
    """
    Split an EC case number into (prefix, tokens).

    Examples:
      'AT.32.432'      -> ('AT',   ['32', '432'])
      'IV/34.324'      -> ('IV',   ['34', '324'])
      'COMP/38.456'    -> ('COMP', ['38', '456'])
      'COMP/D2/56.334' -> ('COMP', ['D2', '56', '334'])
      'AT/23455'       -> ('AT',   ['23455'])
    """
    m = re.match(r"^([A-Za-z]+)(.*)", ec_case_number)
    if not m:
        return "", [ec_case_number]
    prefix = m.group(1).upper()
    rest   = m.group(2)
    tokens = re.split(r"[\s./\-]+", rest.strip())
    tokens = [t for t in tokens if t]   # drop empty strings
    return prefix, tokens


def _prefix_pattern(prefix: str) -> str:
    """
    Return a regex fragment that matches all three EC case-number prefixes.
    Regardless of which prefix is stored in the master (AT, IV, or COMP),
    the pattern always matches all three alternatives.
    """
    return _ALL_PREFIXES


def _digit_flexible(token: str) -> str:
    """
    If token consists entirely of digits, return a regex fragment that allows
    an optional separator between EVERY digit. Otherwise return re.escape(token).

    Example: '31900' -> r'3[\s./\-]*1[\s./\-]*9[\s./\-]*0[\s./\-]*0'
    """
    if token.isdigit():
        return _SEP.join(re.escape(ch) for ch in token)
    return re.escape(token)


def _build_flexible_case_pattern(prefix: str, tokens: list[str]) -> str:
    """
    Build a flexible regex for an EC case number.

    - Prefix is matched via _prefix_pattern (always AT|IV|COMP).
    - Tokens are joined with a flexible separator (_SEP).
    - Pure-digit tokens additionally allow optional separators between every digit.
    - Word-boundary lookarounds prevent partial matches inside longer IDs.
    """
    if not tokens:
        return re.escape(prefix)
    token_part = _SEP.join(_digit_flexible(t) for t in tokens)
    pattern = _prefix_pattern(prefix) + _SEP + token_part
    return r"(?<![A-Za-z0-9])" + pattern + r"(?![A-Za-z0-9])"


def build_patterns_for_ec_case(row: pd.Series) -> list[dict]:
    """
    Build a list of regex patterns for a single EC antitrust case.

    Only case_number patterns (AT/IV/COMP prefix) are returned.
    No CELEX patterns, no decision-reference patterns.

    Each entry is a dict with keys:
      pattern_str    - the raw regex string
      pattern_type   - 'case_number'
      match_strength - 'medium'
    """
    patterns = []
    ec_case_number = str(row.get("ec_case_number", "")).strip()

    if ec_case_number and ec_case_number not in ("", "nan"):
        # Split on ';' to handle multiple case numbers in one cell
        sub_numbers = [s.strip() for s in ec_case_number.split(";") if s.strip()]
        for sub_number in sub_numbers:
            prefix, tokens = _split_ec_case_number(sub_number)
            if prefix and tokens:
                flex_pattern = _build_flexible_case_pattern(prefix, tokens)
                patterns.append({
                    "pattern_str":    flex_pattern,
                    "pattern_type":   "case_number",
                    "match_strength": "medium",
                })
            else:
                # Fallback: literal escape if parsing failed
                patterns.append({
                    "pattern_str":    _escape(sub_number),
                    "pattern_type":   "case_number",
                    "match_strength": "medium",
                })

    return patterns


# ── Quick smoke-tests ──────────────────────────────────────────────────────────────────────────────
print("=== _split_ec_case_number ===")
for cn in ["AT.32.432", "IV/34.324", "COMP/38.456", "COMP/D2/56.334", "AT/23455"]:
    print(f"  {cn!r:25} -> {_split_ec_case_number(cn)}")

print()
print("=== Regex match smoke-tests ===")
smoke_tests = [
    ("IV/34.324",   ["IV/34.324", "IV 34 324", "COMP/34.324", "AT.34324"]),
    ("COMP/38.456", ["IV/38.456", "COMP/38456", "AT.38456"]),
    ("AT.35814",    ["AT.35814", "IV/35.814", "COMP/35.814"]),
    ("IV/31900",    ["IV/31900", "IV/31.900", "AT/31900", "COMP/31900"]),
]
for cn, examples in smoke_tests:
    row = pd.Series({"ec_case_number": cn})
    pats = [p["pattern_str"] for p in build_patterns_for_ec_case(row)]
    for ex in examples:
        matched = any(re.search(pat, ex, re.IGNORECASE) for pat in pats)
        status = "OK" if matched else "FAIL"
        print(f"  [{status}] {cn!r} matches {ex!r}")

print()
print("Pattern builder defined.")

# --- Cell 5: 5. Self-Reference Detection ---
# 5. Self-Reference Detection
def get_self_case_numbers(source_row: pd.Series) -> set[str]:
    """
    Return the set of all EC case numbers that belong to the source document itself.
    Handles multiple case numbers separated by ';'.
    """
    raw = str(source_row.get("ec_case_number", "")).strip()
    if not raw or raw == "nan":
        return set()
    return {s.strip() for s in raw.split(";") if s.strip()}


print("Self-reference detection defined.")

# --- Cell 6: 6. Match Extraction Function ---
# 6. Match Extraction Function
def find_ec_matches_in_text(
    text: str,
    ec_master: pd.DataFrame,
    source_row: pd.Series,
    processing_status: str,
) -> list[dict]:
    """
    Search text for all EC case references defined in ec_master.

    - Excludes self-references (source document's own case numbers).
    - Deduplicates: same source + same target + same pattern_type -> keep first match only.

    Returns a list of match records.
    """
    records = []
    seen = set()

    source_case_numbers   = get_self_case_numbers(source_row)
    source_ec_case_number = str(source_row.get("ec_case_number", "")).strip()
    source_case_title     = str(source_row.get("case_title", "")).strip()
    source_date           = str(source_row.get("date", "")).strip()
    source_type           = str(source_row.get("type", "")).strip()
    source_document_type  = str(source_row.get("document_type", "")).strip()
    source_document_url   = str(source_row.get("document_url", "")).strip()

    for _, target_row in ec_master.iterrows():
        target_ec_case_number = str(target_row.get("ec_case_number", "")).strip()
        target_case_title     = str(target_row.get("case_title", "")).strip()
        target_celex_no       = str(target_row.get("celex_no", "")).strip()

        # Skip self-references
        target_sub_numbers = {s.strip() for s in target_ec_case_number.split(";") if s.strip()}
        if target_sub_numbers & source_case_numbers:
            continue

        patterns = build_patterns_for_ec_case(target_row)

        for pat_info in patterns:
            dedup_key = (source_ec_case_number, target_ec_case_number, pat_info["pattern_type"])
            if dedup_key in seen:
                continue

            try:
                compiled = re.compile(pat_info["pattern_str"], re.IGNORECASE)
            except re.error:
                continue

            match = compiled.search(text)
            if match:
                seen.add(dedup_key)
                start   = max(0, match.start() - CONTEXT_CHARS)
                end     = min(len(text), match.end() + CONTEXT_CHARS)
                context = text[start:end].replace("\n", " ")

                records.append({
                    "source_ec_case_number": source_ec_case_number,
                    "source_case_title":     source_case_title,
                    "source_date":           source_date,
                    "source_type":           source_type,
                    "source_document_type":  source_document_type,
                    "source_document_url":   source_document_url,
                    "target_ec_case_number": target_ec_case_number,
                    "target_case_title":     target_case_title,
                    "target_celex_no":       target_celex_no,
                    "matched_pattern":       pat_info["pattern_str"],
                    "matched_text":          match.group(0),
                    "match_strength":        pat_info["match_strength"],
                    "match_context":         context,
                    "processing_status":     processing_status,
                })

    return records


print("Match extraction function defined.")

# --- Cell 7: 7. Load Data ---
# 7. Load Data
ec_master = pd.read_csv(EC_MASTER_PATH, dtype=str).fillna("")

print(f"EC antitrust master: {len(ec_master):,} rows")
print("Columns:", list(ec_master.columns))

# --- Cell 8 ---
# Only process rows with a non-empty document_url
docs_to_process = ec_master[ec_master["document_url"].str.strip() != ""].copy()

# Keep only EC cases with at least a case number for the target patterns
ec_master_with_case_number = ec_master[
    ec_master["ec_case_number"].str.strip() != ""
].copy()

print(f"Documents to process (have document_url): {len(docs_to_process):,}")
print(f"EC cases with case number (targets):      {len(ec_master_with_case_number):,}")

# --- Cell 9: 8. Main Processing Loop ---
# 8. Main Processing Loop
all_matches = []
total = len(docs_to_process)

for i, (idx, source_row) in enumerate(docs_to_process.iterrows(), start=1):
    ec_case_number = str(source_row.get("ec_case_number", "")).strip()
    document_url   = str(source_row.get("document_url", "")).strip()
    document_type  = str(source_row.get("document_type", "")).strip()

    print(f"[{i}/{total}] {ec_case_number} | {document_type} | {document_url[:80]}", end="", flush=True)

    try:
        text, processing_status = fetch_document_text(document_url, document_type)
    except Exception as e:
        print(f" -> fetch_error: {e}")
        continue

    if not text:
        print(f" -> {processing_status}")
        continue

    matches = find_ec_matches_in_text(text, ec_master_with_case_number, source_row, processing_status)
    all_matches.extend(matches)
    print(f" -> {len(matches)} match(es) [{processing_status}]")

print(f"\nDone. Total matches found: {len(all_matches):,}")

# --- Cell 10: 9. Build Results DataFrame and Export ---
# 9. Build Results DataFrame and Export
OUTPUT_COLUMNS = [
    "source_ec_case_number",
    "source_case_title",
    "source_date",
    "source_type",
    "source_document_type",
    "source_document_url",
    "target_ec_case_number",
    "target_case_title",
    "target_celex_no",
    "matched_pattern",
    "matched_text",
    "match_strength",
    "match_context",
    "processing_status",
]

if all_matches:
    results_df = pd.DataFrame(all_matches, columns=OUTPUT_COLUMNS)
else:
    results_df = pd.DataFrame(columns=OUTPUT_COLUMNS)

print(f"Result rows: {len(results_df):,}")
results_df.head()

# --- Cell 11 ---
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
print(f"Saved {len(results_df):,} rows to: {OUTPUT_PATH}")

# --- Cell 12: 10. Summary Statistics ---
# 10. Summary Statistics
if not results_df.empty:
    print("=== Match Strength Distribution ===")
    print(results_df["match_strength"].value_counts().to_string())
    print()
    print("=== Processing Status Distribution ===")
    print(results_df["processing_status"].value_counts().to_string())
    print()
    print("=== Top 10 Most-Cited Target EC Cases ===")
    print(
        results_df.groupby(["target_ec_case_number", "target_case_title"])
        .size()
        .sort_values(ascending=False)
        .head(10)
        .to_string()
    )
    print()
    print("=== Top 10 Source EC Documents with Most Citations ===")
    print(
        results_df.groupby("source_ec_case_number")["target_ec_case_number"]
        .nunique()
        .sort_values(ascending=False)
        .head(10)
        .to_string()
    )
else:
    print("No matches found.")

