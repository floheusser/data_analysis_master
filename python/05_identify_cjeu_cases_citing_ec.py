# Original notebook: 05_identify_cjeu_cases_citing_ec.ipynb
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

# ── Paths ──────────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data/processed")
EC_MASTER_PATH  = DATA_DIR / "ec_antitrust_master.csv"
CJEU_CASES_PATH = DATA_DIR / "cjeu_cases.csv"
OUTPUT_PATH     = DATA_DIR / "cjeu_ec_case_matches.csv"

# ── HTTP settings ──────────────────────────────────────────────────────────────────
REQUEST_TIMEOUT  = 30   # seconds per request
REQUEST_DELAY    = 1.0  # seconds between requests (be polite)
MAX_RETRIES      = 2

# ── Context window around each match (characters) ─────────────────────────────────────────
CONTEXT_CHARS = 200

print("Configuration loaded.")

# --- Cell 2: 2. URL Helper Functions ---
# 2. URL Helper Functions
def build_celex_html_url(celex_id: str) -> str:
    """Build the official CELEX resource HTML URL."""
    return f"https://publications.europa.eu/resource/celex/{celex_id}"


def build_cellar_branch_url(cellar_id: str) -> str:
    """Build the CELLAR branch notice URL."""
    return f"https://publications.europa.eu/resource/cellar/{cellar_id}?language=eng"


# Quick smoke-test
print(build_celex_html_url("62023CJ0051"))
print(build_cellar_branch_url("ba1070bd-c40d-4171-88df-8578a05e9d17"))

# --- Cell 3: 3. Document Fetching Functions ---
# 3. Document Fetching Functions
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
    """Normalize Unicode, collapse whitespace."""
    text = unicodedata.normalize("NFKC", raw)
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
      1. HTML via resource/celex/{celex_id} – English
      2. HTML via resource/celex/{celex_id} – German
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

# --- Cell 4: 4. Regex Pattern Builder for EC Cases ---
# 4. Regex Pattern Builder for EC Cases
def _escape(s: str) -> str:
    """Regex-escape a string."""
    return re.escape(s)


# ── EC case-number prefix families ────────────────────────────────────────────
# All three prefixes (AT, IV, COMP) are always treated as interchangeable.
_ALL_PREFIXES = r"(?:AT|IV|COMP)"

_SEP = r"[\s./\-]*"   # flexible separator between tokens: space, dot, slash, hyphen (zero or more)


def _split_ec_case_number(ec_case_number: str) -> tuple[str, list[str]]:
    """
    Split an EC case number into (prefix, tokens).

    The prefix is the leading alphabetic part (e.g. 'AT', 'IV', 'COMP').
    Tokens are the remaining alphanumeric chunks after stripping separators.

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
    If *token* consists entirely of digits, return a regex fragment that allows
    an optional separator (space, dot, slash, hyphen) between EVERY digit.
    Otherwise return re.escape(token) unchanged.

    Example: '31900' -> r'3[\\s./\\-]*1[\\s./\\-]*9[\\s./\\-]*0[\\s./\\-]*0'
    """
    if token.isdigit():
        return _SEP.join(re.escape(ch) for ch in token)
    return re.escape(token)


def _build_flexible_case_pattern(prefix: str, tokens: list[str]) -> str:
    """
    Build a flexible regex for an EC case number.

    - Prefix is matched via _prefix_pattern (always AT|IV|COMP).
    - Tokens are joined with a flexible separator (_SEP).
    - Pure-digit tokens additionally allow optional separators between every
      single digit (e.g. '31900' matches 'IV/31.900', 'IV/3.1900', etc.).
    - Word-boundary lookarounds prevent partial matches inside longer IDs.
    """
    if not tokens:
        return re.escape(prefix)
    token_part = _SEP.join(_digit_flexible(t) for t in tokens)
    pattern = _prefix_pattern(prefix) + _SEP + token_part
    # Negative lookbehind/lookahead: no alphanumeric char directly adjacent
    return r"(?<![A-Za-z0-9])" + pattern + r"(?![A-Za-z0-9])"


def build_patterns_for_ec_case(row: pd.Series) -> list[dict]:
    """
    Build a list of regex patterns for a single EC antitrust case.

    Each entry is a dict with keys:
      pattern_str    - the raw regex string
      pattern_type   - 'celex' | 'case_number' | 'decision_ref'
      match_strength - 'strong' | 'medium' | 'weak'
    """
    patterns = []
    ec_case_number = str(row.get("ec_case_number", "")).strip()
    celex_no       = str(row.get("celex_no", "")).strip()

    # 1. CELEX pattern (strong)
    if celex_no and celex_no not in ("", "nan"):
        patterns.append({
            "pattern_str": _escape(celex_no),
            "pattern_type": "celex",
            "match_strength": "strong",
        })

    # 2. EC case-number patterns (medium)
    if ec_case_number and ec_case_number not in ("", "nan"):
        # Split on ';' to handle multiple case numbers in one cell
        sub_numbers = [s.strip() for s in ec_case_number.split(";") if s.strip()]
        for sub_number in sub_numbers:
            prefix, tokens = _split_ec_case_number(sub_number)
            if prefix and tokens:
                flex_pattern = _build_flexible_case_pattern(prefix, tokens)
                patterns.append({
                    "pattern_str": flex_pattern,
                    "pattern_type": "case_number",
                    "match_strength": "medium",
                })
            else:
                # Fallback: literal escape if parsing failed
                patterns.append({
                    "pattern_str": _escape(sub_number),
                    "pattern_type": "case_number",
                    "match_strength": "medium",
                })

    return patterns


# Quick tests
print("=== _split_ec_case_number ===")
for cn in ["AT.32.432", "IV/34.324", "COMP/38.456", "COMP/D2/56.334", "AT/23455"]:
    print(f"  {cn!r:25} -> {_split_ec_case_number(cn)}")

print()
print("=== build_patterns_for_ec_case ===")
for cn, cx in [
    ("IV/31.906",   "31989D0093"),
    ("COMP/38.456", ""),
    ("AT.32.432",   ""),
    ("AT/23455",    ""),
]:
    test_row = pd.Series({"ec_case_number": cn, "celex_no": cx})
    print(f"\n  ec_case_number={cn!r}")
    for p in build_patterns_for_ec_case(test_row):
        print(f"    {p}")

print()
print("=== Regex match smoke-tests ===")
smoke_tests = [
    # IV in master -> also matches AT and COMP
    ("IV/34.324",   ["IV/34.324", "IV 34 324", "IV/34324", "COMP/34.324", "COMP 34 324", "COMP/34324", "AT.34324", "AT 34 324"]),
    # COMP in master -> also matches AT and IV
    ("COMP/38.456", ["IV/38.456", "IV 38 456", "COMP/38456", "AT.38456", "AT 38 456"]),
    # AT in master -> also matches IV and COMP
    ("AT.35814",    ["AT.35814", "AT 35814", "AT/35814", "IV/35.814", "IV 35 814", "COMP/35.814", "COMP 35 814"]),
    ("AT.32.432",   ["AT.32.432", "AT 32 432", "AT/32.432", "AT-32432", "IV/32.432", "COMP/32.432"]),
    # digit-flexible tests
    ("IV/31900",    ["IV/31900", "IV 31900", "IV-31900", "IV/31.900", "IV 31 900", "IV/3.1900", "IV/319.00", "AT/31900", "COMP/31900"]),
    ("AT/23455",    ["AT/23455", "AT 23.455", "AT-23 455", "AT/2.3455", "AT/234.55", "IV/23455", "COMP/23455"]),
]
for cn, examples in smoke_tests:
    row = pd.Series({"ec_case_number": cn, "celex_no": ""})
    pats = [p["pattern_str"] for p in build_patterns_for_ec_case(row) if p["pattern_type"] == "case_number"]
    for ex in examples:
        matched = any(re.search(pat, ex, re.IGNORECASE) for pat in pats)
        status = "OK" if matched else "FAIL"
        print(f"  [{status}] {cn!r} matches {ex!r}")

# --- Cell 5: 5. Match Extraction Function ---
# 5. Match Extraction Function
def find_matches_in_text(
    text: str,
    ec_master: pd.DataFrame,
    cjeu_row: pd.Series,
    source_url: str,
    doc_format: str,
) -> list[dict]:
    """
    Search text for all EC case references defined in ec_master.

    Returns a list of match records (one per unique pattern_type per EC case).
    Deduplicates: if the same pattern_type matches multiple times for the same
    EC case in the same CJEU document, only the first match is kept.
    """
    records = []
    seen = set()

    cjeu_celex  = str(cjeu_row.get("celex_id", ""))
    cjeu_cellar = str(cjeu_row.get("cellar_id", ""))
    cjeu_title  = str(cjeu_row.get("title", ""))
    cjeu_date   = str(cjeu_row.get("document_date", ""))

    for _, ec_row in ec_master.iterrows():
        ec_case_number = str(ec_row.get("ec_case_number", "")).strip()
        ec_case_title  = str(ec_row.get("case_title", "")).strip()
        ec_celex_no    = str(ec_row.get("celex_no", "")).strip()

        patterns = build_patterns_for_ec_case(ec_row)

        for pat_info in patterns:
            # Only export case_number matches (AT/COMP/IV prefix); skip celex and decision_ref
            if pat_info["pattern_type"] != "case_number":
                continue

            dedup_key = (cjeu_celex, ec_case_number, pat_info["pattern_type"])
            if dedup_key in seen:
                continue

            try:
                compiled = re.compile(pat_info["pattern_str"], re.IGNORECASE)
            except re.error:
                continue

            match = compiled.search(text)
            if match:
                seen.add(dedup_key)
                start = max(0, match.start() - CONTEXT_CHARS)
                end   = min(len(text), match.end() + CONTEXT_CHARS)
                context = text[start:end].replace("\n", " ")

                records.append({
                    "cjeu_celex_id":       cjeu_celex,
                    "cjeu_cellar_id":      cjeu_cellar,
                    "cjeu_title":          cjeu_title,
                    "cjeu_document_date":  cjeu_date,
                    "ec_case_number":      ec_case_number,
                    "ec_case_title":       ec_case_title,
                    "ec_celex_no":         ec_celex_no,
                    "matched_pattern":     pat_info["pattern_str"],
                    "matched_text":        match.group(0),
                    "match_strength":      pat_info["match_strength"],
                    "match_context":       context,
                    "document_source_url": source_url,
                    "document_format":     doc_format,
                    "processing_status":   "matched",
                })

    return records


print("Match extraction function defined.")

# --- Cell 6: 6. Load Data ---
# 6. Load Data
ec_master = pd.read_csv(EC_MASTER_PATH, dtype=str).fillna("")
cjeu_cases = pd.read_csv(CJEU_CASES_PATH, dtype=str).fillna("")

print(f"EC antitrust master: {len(ec_master):,} rows")
print(f"CJEU cases:          {len(cjeu_cases):,} rows")
print()
print("EC master columns:", list(ec_master.columns))
print("CJEU columns:     ", list(cjeu_cases.columns))

# --- Cell 7 ---
# Keep only EC cases that have at least a case number or a CELEX number
ec_master_filtered = ec_master[
    (ec_master["ec_case_number"].str.strip() != "") |
    (ec_master["celex_no"].str.strip() != "")
].copy()

print(f"EC cases with at least one identifier: {len(ec_master_filtered):,}")

# --- Cell 8: 7. Main Processing Loop ---
# 7. Main Processing Loop
all_matches = []
processing_log = []

total = len(cjeu_cases)

for idx, cjeu_row in cjeu_cases.iterrows():
    celex_id  = str(cjeu_row.get("celex_id", "")).strip()
    cellar_id = str(cjeu_row.get("cellar_id", "")).strip()

    print(f"[{idx+1}/{total}] Processing CELEX/ CELLAR: {celex_id}/ {cellar_id}", end="", flush=True)

    try:
        text, source_url, doc_format = fetch_document_text(celex_id, cellar_id)
    except Exception as e:
        print(f"→ fetch error: {e}")
        processing_log.append({
            "cjeu_celex_id": celex_id,
            "cjeu_cellar_id": cellar_id,
            "processing_status": "fetch_error",
        })
        continue

    if not text:
        print("→ fetch failed")
        processing_log.append({
            "cjeu_celex_id": celex_id,
            "cjeu_cellar_id": cellar_id,
            "processing_status": "fetch_failed",
        })
        continue

    matches = find_matches_in_text(text, ec_master_filtered, cjeu_row, source_url, doc_format)
    all_matches.extend(matches)
    print(f"→ {len(matches)} match(es) [{doc_format}]")

    processing_log.append({
        "cjeu_celex_id": celex_id,
        "cjeu_cellar_id": cellar_id,
        "processing_status": "ok",
    })

print(f"\nDone. Total matches found: {len(all_matches):,}")

# --- Cell 9: 8. Build Results DataFrame and Export ---
# 8. Build Results DataFrame and Export
OUTPUT_COLUMNS = [
    "cjeu_celex_id",
    "cjeu_cellar_id",
    "cjeu_title",
    "cjeu_document_date",
    "ec_case_number",
    "ec_case_title",
    "ec_celex_no",
    "matched_pattern",
    "matched_text",
    "match_strength",
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

# --- Cell 10 ---
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
print(f"Saved {len(results_df):,} rows to: {OUTPUT_PATH}")

# --- Cell 11: 9. Summary Statistics ---
# 9. Summary Statistics
if not results_df.empty:
    print("=== Match Strength Distribution ===")
    print(results_df["match_strength"].value_counts().to_string())
    print()
    print("=== Document Format Distribution ===")
    print(results_df["document_format"].value_counts().to_string())
    print()
    print("=== Top 10 Most-Cited EC Cases ===")
    print(
        results_df.groupby(["ec_case_number", "ec_case_title"])
        .size()
        .sort_values(ascending=False)
        .head(10)
        .to_string()
    )
    print()
    print("=== CJEU Documents with Most EC Citations ===")
    print(
        results_df.groupby("cjeu_celex_id")["ec_case_number"]
        .nunique()
        .sort_values(ascending=False)
        .head(10)
        .to_string()
    )
else:
    print("No matches found.")

# --- Cell 12 ---
log_df = pd.DataFrame(processing_log)
if not log_df.empty:
    print("=== Processing Status Summary ===")
    print(log_df["processing_status"].value_counts().to_string())

