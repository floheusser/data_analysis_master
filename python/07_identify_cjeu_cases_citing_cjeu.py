# Original notebook: 07_identify_cjeu_cases_citing_cjeu.ipynb
# Converted to Python script on: 2026-05-24
# Outputs and markdown cells have been removed.
# Code logic has been preserved as closely as possible.

# --- Cell 1: 1. Imports and Configuration ---
# 1. Imports and Configuration
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Paths
DATA_DIR        = Path("data/processed")
CJEU_CASES_PATH = DATA_DIR / "cjeu_cases.csv"
OUTPUT_PATH     = DATA_DIR / "cjeu_cjeu_case_matches.csv"

# HTTP settings
REQUEST_TIMEOUT = 30   # seconds per request
REQUEST_DELAY   = 1.0  # seconds between requests (be polite)
MAX_RETRIES     = 2

# Context window around each match (characters)
CONTEXT_CHARS = 200

print("Configuration loaded.")

# --- Cell 2: 2. Derive `case_number` from `celex_id` ---
# 2. Derive `case_number` from `celex_id`
# CELEX format for CJEU cases: 6{year}{court}{number}
# Examples:
#   62012CJ0348 -> C-348/12
#   62012TJ0234 -> T-234/12
#   61993CJ0310 -> C-310/93

_COURT_MAP = {"CJ": "C", "TJ": "T", "FJ": "F"}

_CELEX_RE = re.compile(
    r"^6"
    r"(?P<year>\d{4})"
    r"(?P<court>CJ|TJ|FJ)"
    r"(?P<number>\d+)"
    r"$",
    re.IGNORECASE,
)


def derive_case_number_from_celex(celex_id: str) -> str:
    """
    Derive a normalised CJEU case number from a CELEX ID.

    Examples:
      '62012CJ0348' -> 'C-348/12'
      '62012TJ0234' -> 'T-234/12'
      '61993CJ0310' -> 'C-310/93'

    Returns an empty string if the CELEX ID does not match the expected pattern.
    """
    m = _CELEX_RE.match(celex_id.strip())
    if not m:
        return ""
    court  = _COURT_MAP.get(m.group("court").upper(), m.group("court").upper())
    number = str(int(m.group("number")))   # strip leading zeros
    year   = m.group("year")[-2:]           # last two digits
    return f"{court}-{number}/{year}"


# Smoke-tests
print("=== derive_case_number_from_celex ===")
for celex, expected in [
    ("62012CJ0348", "C-348/12"),
    ("62012TJ0234", "T-234/12"),
    ("61993CJ0310", "C-310/93"),
    ("62009FJ0012", "F-12/09"),
    ("62023CJ0051", "C-51/23"),
]:
    result = derive_case_number_from_celex(celex)
    status = "OK" if result == expected else f"FAIL (got {result!r})"
    print(f"  [{status}] {celex} -> {result!r}")

# --- Cell 3: 3. URL Helper Functions ---
# 3. URL Helper Functions
def build_celex_html_url(celex_id: str) -> str:
    """Build the official CELEX resource HTML URL."""
    return f"https://publications.europa.eu/resource/celex/{celex_id}"


def build_cellar_branch_url(cellar_id: str) -> str:
    """Build the CELLAR branch notice URL."""
    return f"https://publications.europa.eu/resource/cellar/{cellar_id}?language=eng"


print(build_celex_html_url("62023CJ0051"))
print(build_cellar_branch_url("ba1070bd-c40d-4171-88df-8578a05e9d17"))

# --- Cell 4: 4. Document Fetching Functions ---
# 4. Document Fetching Functions
def _get_with_retry(url: str, headers: dict) -> requests.Response | None:
    """GET a URL with simple retry logic. Returns None on failure."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (404, 403, 410):
                return None
        except requests.RequestException:
            pass
        if attempt < MAX_RETRIES:
            time.sleep(REQUEST_DELAY * 2)
    return None


def normalize_text(raw: str) -> str:
    """Normalize Unicode, unify dashes, collapse whitespace."""
    text = unicodedata.normalize("NFKC", raw)
    text = re.sub(r"[‐-―−]", "-", text)
    text = re.sub(r"[ 	]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_html(html_bytes: bytes) -> str:
    """Parse HTML, strip scripts/styles, return visible text."""
    soup = BeautifulSoup(html_bytes, "html.parser")
    for tag in soup(["script", "style", "head", "meta", "link"]):
        tag.decompose()
    raw = soup.get_text(separator=" ")
    return normalize_text(raw)


def fetch_html_by_celex(celex_id: str, language: str = "eng") -> tuple[str, str]:
    """
    Fetch HTML document via the official CELEX resource endpoint.

    Returns (text, source_url). text is empty string on failure.
    """
    url = build_celex_html_url(celex_id)
    headers = {
        "User-Agent": "Mozilla/5.0 (research; masterarbeit) compatible",
        "Accept": "text/html",
        "Accept-Language": language,
    }
    resp = _get_with_retry(url, headers)
    if resp is not None:
        content_type = resp.headers.get("Content-Type", "")
        if "html" in content_type or "text" in content_type:
            text = extract_text_from_html(resp.content)
            if len(text) > 200:
                return text, url
    return "", url


def fetch_branch_notice_by_cellar(cellar_id: str) -> tuple[str, str]:
    """
    Fetch Branch Notice XML via the CELLAR endpoint.

    Returns (text, source_url). text is empty string on failure.
    """
    url = build_cellar_branch_url(cellar_id)
    headers = {
        "User-Agent": "Mozilla/5.0 (research; masterarbeit) compatible",
        "Accept": "application/xml;notice=branch",
    }
    resp = _get_with_retry(url, headers)
    if resp is not None:
        text = normalize_text(resp.text)
        if len(text) > 200:
            return text, url
    return "", url


def fetch_document_text(celex_id: str, cellar_id: str) -> tuple[str, str, str]:
    """
    Fetch the full text of a CJEU document.

    Priority:
      1. HTML via resource/celex/{celex_id} - English
      2. HTML via resource/celex/{celex_id} - German
      3. Branch Notice via resource/cellar/{cellar_id} (optional fallback)

    Returns (text, source_url, document_format). text is empty string on failure.
    """
    time.sleep(REQUEST_DELAY)

    # 1. HTML English
    text, url = fetch_html_by_celex(celex_id, language="eng")
    if text:
        return text, url, "html_eng"

    # 2. HTML German
    text, url = fetch_html_by_celex(celex_id, language="deu")
    if text:
        return text, url, "html_deu"

    # 3. Branch Notice via CELLAR (optional fallback)
    if cellar_id:
        text, url = fetch_branch_notice_by_cellar(cellar_id)
        if text:
            return text, url, "branch_notice"

    return "", "", "none"


print("Document fetching functions defined.")

# --- Cell 5: 5. Generic Regex for CJEU Case Citations ---
# 5. Generic Regex for CJEU Case Citations
# Generic pattern that matches modern CJEU case numbers in text.
# Handles:
#   - Court prefixes: C, T, F
#   - Various dashes: -, en-dash, and other Unicode hyphens
#   - Optional spaces around the dash
#   - Optional suffixes: P, R, PPU, RX, etc.
#   - Optional leading keywords: "Case", "Joined Cases", "Cases"

_CJEU_CASE_RE = re.compile(
    r"""
    (?:(?:Joined\s+Cases?|Cases?)\s+)?   # optional prefix keyword
    (?P<court>[CTF])                      # court letter
    \s*[-\u2010-\u2015\u2212]\s*          # dash (various forms), optional spaces
    (?P<number>\d+)                       # case number
    /                                     # slash separator
    (?P<year>\d{2,4})                     # year (2 or 4 digits)
    (?:\s+(?P<suffix>[A-Z]{1,4}))?        # optional suffix: P, R, PPU, RX, ...
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Pattern for old CJEU cases without court prefix, e.g. 6/64, 30/78, 85/76.
# Uses word boundaries and requires the number part to be 1-3 digits
# and the year part to be exactly 2 or 4 digits.
# Negative lookbehind avoids matching inside modern prefixed patterns.
# Also excludes matches directly preceded by a court prefix + dash (C-, T-, F-).
_OLD_CJEU_CASE_RE = re.compile(
    r"""
    (?<![A-Za-z\d/-])          # not preceded by letter, digit, slash or dash
    (?P<number>\d{1,3})        # case number (1-3 digits, old cases are small numbers)
    /                          # slash separator
    (?P<year>\d{2}|\d{4})      # year: 2 or 4 digits
    (?![\d/])                  # not followed by digit or slash
    """,
    re.VERBOSE,
)

# Helper to detect a court prefix immediately before an old-style match position.
_PREFIX_BEFORE_RE = re.compile(
    r"[CTF]\s*[\-\u2010-\u2015\u2212]\s*$",
    re.IGNORECASE,
)


def normalize_cjeu_case_number(case_str: str) -> str:
    """
    Normalise a raw modern CJEU case string to a canonical form.

    Examples:
      'C - 348/12'   -> 'C-348/12'
      'T-234/12'     -> 'T-234/12'
      'C-123/04 P'   -> 'C-123/04 P'
      'Case C-51/23' -> 'C-51/23'
    """
    m = _CJEU_CASE_RE.search(case_str)
    if not m:
        return case_str.strip()
    court  = m.group("court").upper()
    number = m.group("number")
    year   = m.group("year")
    suffix = m.group("suffix")
    base   = f"{court}-{number}/{year}"
    return f"{base} {suffix.upper()}" if suffix else base


def normalize_old_cjeu_case_number(case_str: str) -> str:
    """
    Normalise a raw old CJEU case string (without prefix) to canonical form.

    Examples:
      '6/64'  -> '6/64'
      '30/78' -> '30/78'
    """
    m = _OLD_CJEU_CASE_RE.search(case_str)
    if not m:
        return case_str.strip()
    return f"{m.group('number')}/{m.group('year')}"


def extract_cjeu_case_citations(text: str) -> list[dict]:
    """
    Extract all CJEU case citations from a text.

    Detects both modern prefixed cases (C-348/12) and old unprefixed cases (6/64).

    Returns a list of dicts with keys:
      matched_text   - the raw matched string
      normalized     - the normalised case number
      citation_style - 'modern_prefixed' or 'old_unprefixed'
      is_old_case_citation - True/False
      start          - match start position in text
      end            - match end position in text
    """
    results = []
    covered = set()  # track character ranges already matched by modern regex

    # 1. Modern prefixed cases
    for m in _CJEU_CASE_RE.finditer(text):
        raw        = m.group(0)
        normalized = normalize_cjeu_case_number(raw)
        results.append({
            "matched_text":        raw,
            "normalized":          normalized,
            "citation_style":      "modern_prefixed",
            "is_old_case_citation": False,
            "start":               m.start(),
            "end":                 m.end(),
        })
        # Mark the slash+year portion so old regex won't re-match inside
        covered.add((m.start(), m.end()))

    # 2. Old unprefixed cases — skip positions already covered by modern matches
    for m in _OLD_CJEU_CASE_RE.finditer(text):
        # Skip if this span overlaps with any modern match
        if any(s <= m.start() < e or s < m.end() <= e for s, e in covered):
            continue
        # Skip if directly preceded by a court prefix + dash (e.g. C-, T-, F-)
        context_before = text[max(0, m.start() - 5) : m.start()]
        if _PREFIX_BEFORE_RE.search(context_before):
            continue
        raw        = m.group(0)
        normalized = normalize_old_cjeu_case_number(raw)
        results.append({
            "matched_text":        raw,
            "normalized":          normalized,
            "citation_style":      "old_unprefixed",
            "is_old_case_citation": True,
            "start":               m.start(),
            "end":                 m.end(),
        })

    # Sort by position in text
    results.sort(key=lambda x: x["start"])
    return results


# Smoke-tests
print("=== normalize_cjeu_case_number (modern) ===")
for raw, expected in [
    ("C-348/12",     "C-348/12"),
    ("T-234/12",     "T-234/12"),
    ("F-12/09",      "F-12/09"),
    ("C - 348/12",   "C-348/12"),
    ("C-123/04 P",   "C-123/04 P"),
    ("Case C-51/23", "C-51/23"),
]:
    result = normalize_cjeu_case_number(raw)
    status = "OK" if result == expected else f"FAIL (got {result!r})"
    print(f"  [{status}] {raw!r} -> {result!r}")

print()
print("=== normalize_old_cjeu_case_number (old) ===")
for raw, expected in [
    ("6/64",  "6/64"),
    ("30/78", "30/78"),
    ("85/76", "85/76"),
]:
    result = normalize_old_cjeu_case_number(raw)
    status = "OK" if result == expected else f"FAIL (got {result!r})"
    print(f"  [{status}] {raw!r} -> {result!r}")

print()
print("=== extract_cjeu_case_citations ===")
sample = "See Case C-348/12 and Joined Cases T-1/20 and T-2/20, also F-12/09 P. Old cases: 6/64 and 30/78."
for hit in extract_cjeu_case_citations(sample):
    print(f"  {hit['matched_text']!r:30} -> {hit['normalized']!r:15} [{hit['citation_style']}]")

print()
print("=== extract_cjeu_case_citations (false-positive check) ===")
# 348/12 and 234/12 must NOT appear as old_unprefixed when part of C-348/12 / T-234/12
sample2 = "Judgment in C-348/12 and T-234/12. Also old case 6/64."
for hit in extract_cjeu_case_citations(sample2):
    status = "OK" if not (hit["citation_style"] == "old_unprefixed" and hit["normalized"] in ("348/12", "234/12")) else "FAIL"
    print(f"  [{status}] {hit['matched_text']!r:30} -> {hit['normalized']!r:15} [{hit['citation_style']}]")

# --- Cell 6: 6. Load CJEU Cases and Derive `case_number` ---
# 6. Load CJEU Cases and Derive `case_number`
cjeu_cases = pd.read_csv(CJEU_CASES_PATH, dtype=str).fillna("")

print(f"CJEU cases loaded: {len(cjeu_cases):,} rows")
print("Columns:", list(cjeu_cases.columns))

# --- Cell 7 ---
cjeu_cases["case_number"] = cjeu_cases["celex_id"].apply(derive_case_number_from_celex)

derived_count = (cjeu_cases["case_number"] != "").sum()
print(f"case_number derived for {derived_count:,} / {len(cjeu_cases):,} rows")
cjeu_cases[["celex_id", "case_number"]].head(10)

# --- Cell 8 ---
# Build a lookup dict: case_number -> row  (for fast matching)
cjeu_lookup = (
    cjeu_cases[cjeu_cases["case_number"] != ""]
    .set_index("case_number", drop=False)
        .to_dict(orient="index")
)

print(f"CJEU lookup entries: {len(cjeu_lookup):,}")

# --- Cell 9 ---
# Build a secondary lookup for old unprefixed case numbers.
# Old CJEU cases have CELEX IDs like 61964CJ0006 -> case_number 'C-6/64'.
# We derive an old-style key (e.g. '6/64') from the modern case_number
# so that old citations can be matched against the internal list.

def derive_old_case_key(case_number: str) -> str:
    """
    Derive an old-style key from a modern case_number, e.g. 'C-6/64' -> '6/64'.
    Returns empty string if not applicable.
    """
    m = re.match(r'^[CTF]-(\d+)/(\d{2,4})$', case_number)
    if m:
        return f"{int(m.group(1))}/{m.group(2)}"
    return ""

cjeu_old_lookup = {}
for case_number, row in cjeu_lookup.items():
    old_key = derive_old_case_key(case_number)
    if old_key:
        cjeu_old_lookup[old_key] = row

print(f"CJEU old-style lookup entries: {len(cjeu_old_lookup):,}")

# --- Cell 10: 7. Match Extraction Function ---
# 7. Match Extraction Function
def find_cjeu_matches_in_text(
    text: str,
    source_row: pd.Series,
    cjeu_lookup: dict,
    source_url: str,
    doc_format: str,
    cjeu_old_lookup: dict = None,
) -> list[dict]:
    """
    Search text for CJEU case citations and match them against the known CJEU list.

    - Handles both modern prefixed (C-348/12) and old unprefixed (6/64) citations.
    - Skips self-references (source case citing itself).
    - Deduplicates: same source + same normalised citation -> keep first match only.
    - Old citations that cannot be matched are still included with empty target fields.

    Returns a list of match record dicts.
    """
    if cjeu_old_lookup is None:
        cjeu_old_lookup = {}

    source_celex_id    = str(source_row.get("celex_id", "")).strip()
    source_cellar_id   = str(source_row.get("cellar_id", "")).strip()
    source_case_number = str(source_row.get("case_number", "")).strip()
    source_title       = str(source_row.get("title", "")).strip()
    source_date        = str(source_row.get("document_date", "")).strip()

    citations = extract_cjeu_case_citations(text)

    records = []
    seen    = set()  # (source_celex_id, normalized_match)

    for citation in citations:
        normalized     = citation["normalized"]
        citation_style = citation["citation_style"]
        is_old         = citation["is_old_case_citation"]

        # Skip self-references
        if normalized == source_case_number:
            continue
        # Also skip if old citation matches the source's old-style key
        if is_old and source_case_number:
            source_old_key = derive_old_case_key(source_case_number)
            if normalized == source_old_key:
                continue

        dedup_key = (source_celex_id, normalized)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        start   = max(0, citation["start"] - CONTEXT_CHARS)
        end     = min(len(text), citation["end"] + CONTEXT_CHARS)
        context = text[start:end].replace("\n", " ")

        if not is_old:
            # Modern prefixed citation: must match known CJEU case
            if normalized not in cjeu_lookup:
                continue
            target = cjeu_lookup[normalized]
            records.append({
                "source_celex_id":      source_celex_id,
                "source_cellar_id":     source_cellar_id,
                "source_case_number":   source_case_number,
                "source_title":         source_title,
                "source_document_date": source_date,
                "target_celex_id":      target.get("celex_id", ""),
                "target_cellar_id":     target.get("cellar_id", ""),
                "target_case_number":   normalized,
                "target_title":         target.get("title", ""),
                "matched_text":         citation["matched_text"],
                "normalized_match":     normalized,
                "citation_style":       citation_style,
                "is_old_case_citation": is_old,
                "match_context":        context,
                "document_source_url":  source_url,
                "document_format":      doc_format,
                "processing_status":    "matched",
            })
        else:
            # Old unprefixed citation: try to match via old lookup
            target = cjeu_old_lookup.get(normalized)
            records.append({
                "source_celex_id":      source_celex_id,
                "source_cellar_id":     source_cellar_id,
                "source_case_number":   source_case_number,
                "source_title":         source_title,
                "source_document_date": source_date,
                "target_celex_id":      target.get("celex_id", "") if target else "",
                "target_cellar_id":     target.get("cellar_id", "") if target else "",
                "target_case_number":   target.get("case_number", "") if target else "",
                "target_title":         target.get("title", "") if target else "",
                "matched_text":         citation["matched_text"],
                "normalized_match":     normalized,
                "citation_style":       citation_style,
                "is_old_case_citation": is_old,
                "match_context":        context,
                "document_source_url":  source_url,
                "document_format":      doc_format,
                "processing_status":    "matched" if target else "unmatched_old_case",
            })

    return records


print("Match extraction function defined.")

# --- Cell 11: 8. Main Processing Loop ---
# 8. Main Processing Loop
all_matches = []
total = len(cjeu_cases)

for i, (idx, source_row) in enumerate(cjeu_cases.iterrows(), start=1):
    celex_id  = str(source_row.get("celex_id", "")).strip()
    cellar_id = str(source_row.get("cellar_id", "")).strip()

    print(f"[{i}/{total}] {celex_id}", end="", flush=True)

    try:
        text, source_url, doc_format = fetch_document_text(celex_id, cellar_id)
    except Exception as e:
        print(f" -> fetch_error: {e}")
        continue

    if not text:
        print(f" -> fetch_failed [{doc_format}]")
        continue

    matches = find_cjeu_matches_in_text(text, source_row, cjeu_lookup, source_url, doc_format, cjeu_old_lookup)
    all_matches.extend(matches)
    print(f" -> {len(matches)} match(es) [{doc_format}]")

print(f"\nDone. Total matches found: {len(all_matches):,}")

# --- Cell 12: 9. Build Results DataFrame and Export ---
# 9. Build Results DataFrame and Export
OUTPUT_COLUMNS = [
    "source_celex_id",
    "source_cellar_id",
    "source_case_number",
    "source_title",
    "source_document_date",
    "target_celex_id",
    "target_cellar_id",
    "target_case_number",
    "target_title",
    "matched_text",
    "normalized_match",
    "citation_style",
    "is_old_case_citation",
    "match_context",
    "document_source_url",
    "document_format",
    "processing_status",
]

if all_matches:
    results_df = pd.DataFrame(all_matches, columns=OUTPUT_COLUMNS)
else:
    results_df = pd.DataFrame(columns=OUTPUT_COLUMNS)

print(f"Result rows: {len(results_df):,}")
results_df.head()

# --- Cell 13 ---
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
export_df = results_df[results_df["processing_status"] != "unmatched_old_case"].copy()
export_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
print(f"Saved {len(export_df):,} rows to: {OUTPUT_PATH} (excluded {len(results_df) - len(export_df):,} unmatched_old_case rows)")

# --- Cell 14: 10. Summary Statistics ---
# 10. Summary Statistics
if not results_df.empty:
    print("=== Document Format Distribution ===")
    print(results_df["document_format"].value_counts().to_string())
    print()
    print("=== Citation Style Distribution ===")
    print(results_df["citation_style"].value_counts().to_string())
    print()
    print("=== Processing Status Distribution ===")
    print(results_df["processing_status"].value_counts().to_string())
    print()
    print("=== Top 10 Most-Cited CJEU Cases ===")
    print(
        results_df.groupby(["target_case_number", "target_title"])
        .size()
        .sort_values(ascending=False)
        .head(10)
        .to_string()
    )
    print()
    print("=== CJEU Documents with Most CJEU Citations ===")
    print(
        results_df.groupby("source_celex_id")["target_case_number"]
        .nunique()
        .sort_values(ascending=False)
        .head(10)
        .to_string()
    )
else:
    print("No matches found.")

