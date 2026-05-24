# Original notebook: 02_collect_ec_pre1997_formal_decisions.ipynb
# Converted to Python script on: 2026-05-24
# Outputs and markdown cells have been removed.
# Code logic has been preserved as closely as possible.

# --- Cell 1: 1. Imports and Configuration ---
# 1. Imports and Configuration
import re
import time
import requests
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup

print("Imports OK")

# --- Cell 2: 2. Constants ---
# 2. Constants
# Base URL for legacy year pages
BASE_URL = "https://ec.europa.eu/competition/antitrust/closed/en/"

# Output directories
OUTPUT_DIR = Path("data/processed")
RAW_DIR    = Path("data/raw/ec_legacy_formal_decisions")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Year range of legacy pages
YEAR_START = 1964
YEAR_END   = 1997  # inklusiv

# EUR-Lex language for document_url (variable: "EN", "DE", "FR", ...)
EURLEX_LANG = "EN"

print(f"BASE_URL   : {BASE_URL}")
print(f"OUTPUT_DIR : {OUTPUT_DIR}")
print(f"RAW_DIR    : {RAW_DIR}")
print(f"Year range: {YEAR_START}–{YEAR_END}")
print(f"EURLEX_LANG: {EURLEX_LANG}")

# --- Cell 3: 3. Session / Request Helpers ---
# 3. Session / Request Helpers
def make_session() -> requests.Session:
    """Creates a requests.Session with sensible headers."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; EC-Antitrust-Scraper/2.0; +research)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    })
    return s


def fetch_html(session: requests.Session, url: str, timeout: int = 20):
    """
    Fetches a URL and returns (status_code, text).
    On error: (status_code_or_None, None).
    """
    try:
        r = session.get(url, timeout=timeout)
        if r.status_code == 404:
            return 404, None
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "ISO-8859-1"
        return r.status_code, r.text
    except requests.exceptions.Timeout:
        return None, None
    except requests.exceptions.RequestException:
        return None, None


def save_raw_html(path: Path, text: str) -> None:
    """Saves raw HTML text to a file."""
    path.write_text(text, encoding="utf-8", errors="replace")


print("Helfer-Functions definiert.")

# --- Cell 4: 4. Jahresseiten-Liste erzeugen ---
# 4. Jahresseiten-Liste erzeugen
# URLs are constructed directly and deliberately – no heuristic link detection.
year_pages = []
for year in range(YEAR_START, YEAR_END + 1):
    url = f"{BASE_URL}{year}.html"
    year_pages.append({"year": year, "year_url": url})

df_years = pd.DataFrame(year_pages)
print(f"Geplante Jahresseiten: {len(df_years)}")
df_years.head(10)

# --- Cell 5: 5. CELEX-Helper Functions ---
# 5. CELEX-Helper Functions
def _normalize_celex(s: str) -> str:
    """Removes whitespace, preserves capitalisation."""
    return re.sub(r"\s+", "", s or "")


def _to_celex_complete(celex_no: str) -> str:
    """
    Expands 3+YY+L+NNNN (e.g. 371D0023) to 3+YYYY+L+NNNN (31971D0023).
    Already complete 10-digit CELEX numbers are returned unchanged.
    """
    c = _normalize_celex(celex_no)
    if not c:
        return ""
    if re.fullmatch(r"3\d{4}[A-Z]\d{4}", c):
        return c
    m = re.fullmatch(r"3(\d{2})([A-Z])(\d{4})", c)
    if m:
        yy, letter, num = m.groups()
        yyyy = f"19{yy}" if int(yy) >= 50 else f"20{yy}"
        return f"3{yyyy}{letter}{num}"
    return c


print("CELEX-Helper Functions definiert.")

# --- Cell 6: 6. Date Normalization for legacy data ---
# 6. Date Normalization for legacy data
# Legacy-Pagen enthalten Datumsangaben in verschiedenen Formaten (DD.MM.YYYY, DD/MM/YYYY,
# month name variants etc.). This function normalises all known formats to YYYY-MM-DD.
# Known date formats on the legacy pages
_LEGACY_DATE_FORMATS = [
    "%d.%m.%Y",   # 23.04.1985
    "%d/%m/%Y",   # 23/04/1985
    "%d-%m-%Y",   # 23-04-1985
    "%Y-%m-%d",   # 1985-04-23 (bereits normalisiert)
    "%d %B %Y",   # 23 April 1985
    "%d %b %Y",   # 23 Apr 1985
    "%B %d, %Y",  # April 23, 1985
    "%b %d, %Y",  # Apr 23, 1985
]

# Pattern for DD.MM.YYYY / DD/MM/YYYY / DD-MM-YYYY
DATE_PAT = re.compile(r"\b(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{4})\b")


def normalize_legacy_date(raw: str) -> tuple[str | None, str]:
    """
    Attempts to normalise a raw legacy date string to YYYY-MM-DD.

    Returns: (normalized_date_or_None, source_label)
      source_label is 'legacy_raw' on success, 'missing' if no date is recognised.
    """
    if not raw or not str(raw).strip():
        return None, "missing"

    raw = str(raw).strip()

    # Try directly with pandas (recognises many formats automatically)
    parsed = pd.to_datetime(raw, dayfirst=True, errors="coerce", format="mixed")
    if pd.notna(parsed):
        return parsed.strftime("%Y-%m-%d"), "legacy_raw"

    # Explizite Formate durchprobieren
    for fmt in _LEGACY_DATE_FORMATS:
        try:
            import datetime
            dt = datetime.datetime.strptime(raw, fmt)
            return dt.strftime("%Y-%m-%d"), "legacy_raw"
        except ValueError:
            continue

    # Regex extraction as last resort: DD.MM.YYYY / DD/MM/YYYY / DD-MM-YYYY
    m = DATE_PAT.search(raw)
    if m:
        day, month, year = m.group(1), m.group(2), m.group(3)
        candidate = f"{year}-{int(month):02d}-{int(day):02d}"
        parsed2 = pd.to_datetime(candidate, errors="coerce")
        if pd.notna(parsed2):
            return parsed2.strftime("%Y-%m-%d"), "legacy_raw"

    return None, "missing"


print("Date Normalization definiert.")

# Quick tests
assert normalize_legacy_date("23.04.1985")[0] == "1985-04-23"
assert normalize_legacy_date("23/04/1985")[0] == "1985-04-23"
assert normalize_legacy_date("1985-04-23")[0] == "1985-04-23"
assert normalize_legacy_date("")[1]           == "missing"
assert normalize_legacy_date(None)[1]         == "missing"
print("All date tests passed.")

# --- Cell 7: 7. Enge Extraktionslogik pro Jahresseite ---
# 7. Enge Extraktionslogik pro Jahresseite
# The parser follows the old, working logic directly:
# - Case title from `<a name="...">` anchors
# - Metadata from the next `<font>` block with date
# - No generic collection of all `<li>` or `<p>` elements
def parse_case_entries_from_year_page(soup: BeautifulSoup, year: int, source_url: str, lang: str = EURLEX_LANG) -> list:
    """
    Extracts case entries from a legacy year page.

    Logic (closely tailored to the old page structure):
    - Finds all <a name="..."> anchors as case start points
    - Reads the next <font> block with date as detail block
    - Extracts metadata directly from this block
    """
    entries = []
    anchors = soup.find_all("a", attrs={"name": True})

    for idx, a_tag in enumerate(anchors):
        # Skip navigation anchors ("top" etc.)
        name_attr = a_tag.get("name", "").lower().strip()
        if name_attr in ("top", "", "bottom", "index"):
            continue

        title = a_tag.get_text(" ", strip=True)
        if not title:
            continue

        # Find the next <font> block containing a date
        details_font = None
        candidate = a_tag.find_next("font")
        steps = 0
        while candidate and steps < 40:
            text = " ".join(candidate.stripped_strings)
            if DATE_PAT.search(text):
                details_font = candidate
                break
            candidate = candidate.find_next("font")
            steps += 1

        if not details_font:
            continue

        details_text = " ".join(details_font.stripped_strings)

        # --- Datum (roh) ---
        m_date = DATE_PAT.search(details_text)
        decision_date_raw = m_date.group(0) if m_date else ""

        # --- Official Journal ---
        oj_str = ""
        m_oj = re.search(
            r"Official Journal\s*:\s*(.+?)(?=\s+Celex No\.\s*:|\s+\b[A-Z]{1,3}\s*/\s*\d{1,6}\b|$)",
            details_text
        )
        if m_oj:
            oj_str = m_oj.group(1).strip()

        # --- CELEX (preferred from href numdoc=, fallback from text) ---
        celex_raw = ""
        celex_display = ""
        document_url = ""

        celex_anchor = details_font.find("a", href=re.compile(r"CELEXnumdoc", re.I))
        if celex_anchor and celex_anchor.has_attr("href"):
            m_href = re.search(r"numdoc=([0-9A-Za-z]+)", celex_anchor["href"])
            if m_href:
                celex_raw = _normalize_celex(m_href.group(1))
            celex_display = celex_anchor.get_text(" ", strip=True)

        if not celex_raw:
            m_celex_text = re.search(
                r"Celex No\.\s*:\s*([0-9A-Za-z\s]+?)(?=\s+-\s+|\s+\b[A-Z]{1,3}\s*/\s*\d{1,6}\b|$)",
                details_text
            )
            if m_celex_text:
                celex_display = m_celex_text.group(1).strip()
                celex_raw = _normalize_celex(celex_display)

        celex_complete = _to_celex_complete(celex_raw)

        # Construct EUR-Lex document_url from celex_complete + lang (DE as fallback)
        if celex_complete:
            document_url    = f"https://publications.europa.eu/resource/celex/{celex_complete}"


        # --- Fallnummern (IV/…) ---
        raw_cases = re.findall(r"\b([A-Z]{1,3})\s*/\s*(\d{1,6})\b", details_text)
        seen = set()
        case_numbers_list = []
        for prefix, num in raw_cases:
            normalized = f"{prefix}/{num}"
            if normalized not in seen:
                seen.add(normalized)
                case_numbers_list.append(normalized)
        case_number_raw = "; ".join(case_numbers_list)

        # --- Decision type (after date, before OJ/CELEX) ---
        decision_type = ""
        if decision_date_raw:
            post_date = details_text.split(decision_date_raw, 1)[1].strip()
            cut_points = []
            for marker in [
                r"Official Journal\s*:",
                r"Celex No\.\s*:",
                r"\b[A-Z]{1,3}\s*/\s*\d{1,6}\b"
            ]:
                m = re.search(marker, post_date)
                if m:
                    cut_points.append(m.start())
            end_idx = min(cut_points) if cut_points else len(post_date)
            decision_type = post_date[:end_idx].strip(" -\u00A0").strip()

        entries.append({
            "year_page":             year,
            "source_url":            source_url,
            "case_title":            title,
            "decision_date_raw":     decision_date_raw,
            "decision_type_raw":     decision_type,
            "publication_ref_raw":   oj_str,
            "case_number_raw":       case_number_raw,
            "celex_complete":        celex_complete,
            "document_url":          document_url,
        })

    return entries


print("Parser-Funktion definiert.")

# --- Cell 8: 8. Load and parse year pages ---
# 8. Load and parse year pages
session = make_session()

all_records = []
success_years = []
failed_years  = []

for row in df_years.itertuples():
    year     = row.year
    year_url = row.year_url

    status_code, html = fetch_html(session, year_url)

    if html is None:
        reason = "404" if status_code == 404 else (f"HTTP {status_code}" if status_code else "Timeout/Error")
        print(f"   ⚠️  {year}: {reason} – {year_url}")
        failed_years.append({"year": year, "url": year_url, "reason": reason})
        continue

    # Optional: rohe HTML save
    raw_path = RAW_DIR / f"formal_decision_{year}.html"
    save_raw_html(raw_path, html)

    soup    = BeautifulSoup(html, "html.parser")
    entries = parse_case_entries_from_year_page(soup, year, year_url, lang=EURLEX_LANG)

    all_records.extend(entries)
    success_years.append(year)
    print(f"   ✅ {year}: {len(entries)} cases extracted")

    time.sleep(0.3)  # polite delay

print(f"\nFertig. Erfolgreich: {len(success_years)}, Fehlgeschlagen: {len(failed_years)}")
print(f"Total cases extracted: {len(all_records)}")

# --- Cell 9: 9. Build DataFrame and normalise dates ---
# 9. Build DataFrame and normalise dates
df = pd.DataFrame(all_records)

print(f"DataFrame: {len(df)} rows, {len(df.columns)} columns")
print(f"columns: {list(df.columns)}")
df.head(3)

# --- Cell 10 ---
# Datum normalisieren: decision_date_raw -> date (YYYY-MM-DD) + date_source
if len(df) > 0:
    results = df["decision_date_raw"].apply(normalize_legacy_date)
    df["date"]        = results.apply(lambda x: x[0])
    df["date_source"] = results.apply(lambda x: x[1])

    # Fallback: OJ reference sometimes contains a year – mark as legacy_fallback
    mask_missing = df["date"].isna()
    if mask_missing.any():
        oj_year_pat = re.compile(r"\b(19[6-9]\d|20[0-2]\d)\b")
        for idx in df[mask_missing].index:
            oj_val = str(df.at[idx, "publication_ref_raw"] or "")
            m = oj_year_pat.search(oj_val)
            if m:
                df.at[idx, "date"]        = f"{m.group(1)}-01-01"
                df.at[idx, "date_source"] = "legacy_fallback"

    print(f"Datum normalisiert.")
    print(f"  With date      : {df['date'].notna().sum()}")
    print(f"  Without date   : {df['date'].isna().sum()}")
    print(f"  Quellen:\n{df['date_source'].value_counts(dropna=False).to_string()}")

# --- Cell 11 ---
# Vorschau der ersten rows
if len(df) > 0:
    display(df[["year_page", "case_title", "decision_date_raw", "date", "date_source", "celex_complete", "document_url"]].head(10))

# --- Cell 12: 10. Export ---
# 10. Export
if len(df) > 0:
    csv_path = OUTPUT_DIR / "ec_legacy_formal_decisions.csv"
    df.rename(columns={
        "year_page":          "year",
        "case_title":         "title",
        "decision_date_raw":  "date_raw",
        "decision_type_raw":  "decision_type",
        "publication_ref_raw":"official_journal",
        "celex_complete":     "celex_no",
        "case_number_raw":    "case_numbers",
    }).to_csv(csv_path, index=False, encoding="utf-8")
    print(f"✅ CSV gespeichert → {csv_path}")

else:
    print("⚠️  Keine Daten zum Speichern.")

