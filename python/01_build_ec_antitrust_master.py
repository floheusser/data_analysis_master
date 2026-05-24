# Original notebook: 01_build_ec_antitrust_master.ipynb
# Converted to Python script on: 2026-05-24
# Outputs and markdown cells have been removed.
# Code logic has been preserved as closely as possible.

# --- Cell 1: 1 – Imports and Configuration ---
# 1 – Imports and Configuration
import json
from pathlib import Path

import pandas as pd

# Widen pandas display so columns remain readable
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 80)
pd.set_option("display.width", 120)

print("Imports OK")

# --- Cell 2: 2 – Define Input and Output Paths ---
# 2 – Define Input and Output Paths
# Define where the source file is located and where results will be saved.
# All paths use `pathlib.Path` – works on Windows, Mac and Linux alike.
# Path to the source file (JSON or Excel)
# Source: https://data.europa.eu/data/datasets/18489cb7-bce7-4d44-a138-795b390d2109~~1?locale=en
# --> https://compcases-open-data-portal-files-prod.s3.eu-west-1.amazonaws.com/case-data-AT.json

INPUT_FILE = Path("data/case-data-AT.json")

# Output directory – created automatically if it does not exist
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Output files
OUTPUT_CSV     = OUTPUT_DIR / "ec_antitrust_master.csv"

print(f"Input : {INPUT_FILE.resolve()}")
print(f"Output: {OUTPUT_DIR.resolve()}")
print(f"File exists: {INPUT_FILE.exists()}")

# --- Cell 3: 3 – Load Data ---
# 3 – Load Data
# The source file is a JSON file where each key is an EC case ID (e.g. `AT.39294`).
# The value is an object with `metadata`, `caseAttachments` and `decisions`.
def load_input_file(path: Path):
    """Loads JSON – returns a dict."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        print(f"JSON loaded: {len(data)} entries (top-level keys)")
        return "json", data
    else:
        raise ValueError(f"Unknown file format: {suffix}. Expected: .json")


file_type, raw_data = load_input_file(INPUT_FILE)

# --- Cell 4: 4 – Helper Functions ---
# 4 – Helper Functions
# JSON fields are almost always **lists** (even if only one value is present).
# Some fields like `caseSectors` or `caseLegalBasis` are encoded as **JSON strings within the list**.
# These small helper functions keep the code clean and robust.
def first_value(x):
    """Returns the first entry of a list, or None if empty/no value."""
    if isinstance(x, list) and len(x) > 0:
        return x[0]
    return None


print("Helper Functions definiert.")

# Quick tests
assert first_value(["a", "b"]) == "a"
assert first_value([]) is None
print("All tests passed.")

# --- Cell 5: 5 – Antitrust Filter Logic ---
# 5 – Antitrust Filter Logic
# Why do we filter?
# The EC database contains various case types:
# - **Antitrust** (Art. 101/102 TFEU) – this is what we want
# - **Cartels** – often part of antitrust, but classified separately
# - **Mergers** – merger control, not relevant
# - **State Aid** – state aid law, not relevant
# - **DMA / FSR** – newer instruments, not relevant
# Filter criteria
# We keep a case if **at least one** of these criteria applies:
# 1. `caseCartel` contains `"Antitrust"` oder `"Cartel"` → primary criterion
# 2. `caseInstrument` contains `"Antitrust"` → secondary criterion
# Cases with `caseInstrument` wie `"Merger"`, `"State Aid"`, `"DMA"`, `"FSR"` are **excluded**.
# Keywords indicating antitrust
ANTITRUST_KEYWORDS = {"antitrust", "cartel"}

# Keywords explicitly NOT antitrust
EXCLUDE_KEYWORDS = {"merger", "state aid", "dma", "fsr"}


def is_antitrust_case(metadata: dict) -> bool:
    """
    Determines whether a case is an antitrust case.

    Returns True if:
    - caseCartel or caseInstrument contains an antitrust keyword
    AND
    - no exclusion keyword (Merger, State Aid, etc.) is present
    """
    cartel_val = " ".join(str(v) for v in metadata.get("caseCartel", []) if v).lower()
    instr_val  = " ".join(str(v) for v in metadata.get("caseInstrument", []) if v).lower()
    combined   = cartel_val + " " + instr_val

    has_antitrust = any(kw in combined for kw in ANTITRUST_KEYWORDS)
    has_exclusion = any(kw in combined for kw in EXCLUDE_KEYWORDS)

    return has_antitrust and not has_exclusion


print("Filter logic defined.")

# Quick tests
assert is_antitrust_case({"caseCartel": ["Antitrust"], "caseInstrument": ["Antitrust & Cartels"]}) == True
assert is_antitrust_case({"caseCartel": ["Cartel"],   "caseInstrument": ["Antitrust & Cartels"]}) == True
assert is_antitrust_case({"caseCartel": [],            "caseInstrument": ["Merger"]})              == False
assert is_antitrust_case({"caseCartel": ["Antitrust"], "caseInstrument": ["Merger"]})              == False
print("All filter tests passed.")

# --- Cell 6: 6 – Flattening: Build a flat table from JSON structure ---
# 6 – Flattening: Build a flat table from JSON structure
# Each case in the JSON file has a nested structure.
# We "flatten" this structure into a single row per case.
# The function `extract_case_row()` does exactly that: it takes a case and returns a dictionary,
# that can be used directly as a row in a DataFrame.
# Date fallback chain (modern JSON cases)
# Priority for `date` (first non-empty date wins):
# 1. `caseLastDecisionDate`
# 2. `decisionAdoptionDate` (from decisions[])
# 3. `decisionOfficialJournalPublicationsPublishedDates` (from decisions[])
# 4. `attachmentDocumentDate` (from caseAttachments[])
# 5. `attachmentSentDate` (from caseAttachments[])
# 6. `attachmentPublicationBusinessDate` (from caseAttachments[])
# 7. `caseOfficialJournalPublicationsPublishedDates`
# 8. `caseInitiationDate` (last fallback)
# The field `date_source` documents which source was used.
def get_first_decision_attachment_link(case_obj: dict) -> str | None:
    """Returns the first attachmentLink from decisions[].decisionAttachments[], or None."""
    for decision in case_obj.get("decisions", []):
        for att in decision.get("decisionAttachments", []):
            link = first_value(att.get("metadata", {}).get("attachmentLink", []))
            if link:
                return link
    return None


def get_first_case_attachment_link(case_obj: dict) -> str | None:
    """Returns the first attachmentLink from caseAttachments[], or None."""
    for att in case_obj.get("caseAttachments", []):
        link = first_value(att.get("metadata", {}).get("attachmentLink", []))
        if link:
            return link
    return None


def get_document_type_and_url(case_obj: dict) -> tuple[str, str]:
    """
    Determines document_type and document_url by priority rule:
    A. decisionAttachments -> 'decision'
    B. caseAttachments     -> 'case'
    C. otherwise               -> 'none', ''
    """
    link = get_first_decision_attachment_link(case_obj)
    if link:
        return "decision", link
    link = get_first_case_attachment_link(case_obj)
    if link:
        return "case", link
    return "none", ""


def _nonempty(val: str | None) -> str | None:
    """Returns val if not empty/None, otherwise None."""
    if val and str(val).strip():
        return str(val).strip()
    return None


def resolve_date(case_obj: dict) -> tuple[str | None, str]:
    """
    Determines the final date and date source according to the fallback chain.

    Returns: (date_str_or_None, date_source_label)
    """
    meta = case_obj.get("metadata", {})

    # 1. caseLastDecisionDate
    val = _nonempty(first_value(meta.get("caseLastDecisionDate", [])))
    if val:
        return val, "case_last_decision"

    # 2. decisionAdoptionDate (from decisions[])
    for dec in case_obj.get("decisions", []):
        val = _nonempty(first_value(dec.get("metadata", {}).get("decisionAdoptionDate", [])))
        if val:
            return val, "decision_adoption"

    # 3. decisionOfficialJournalPublicationsPublishedDates (from decisions[])
    for dec in case_obj.get("decisions", []):
        val = _nonempty(first_value(dec.get("metadata", {}).get("decisionOfficialJournalPublicationsPublishedDates", [])))
        if val:
            return val, "decision_oj"

    # 4. attachmentDocumentDate (from caseAttachments[])
    for att in case_obj.get("caseAttachments", []):
        val = _nonempty(first_value(att.get("metadata", {}).get("attachmentDocumentDate", [])))
        if val:
            return val, "attachment_document"

    # 5. attachmentSentDate (from caseAttachments[])
    for att in case_obj.get("caseAttachments", []):
        val = _nonempty(first_value(att.get("metadata", {}).get("attachmentSentDate", [])))
        if val:
            return val, "attachment_sent"

    # 6. attachmentPublicationBusinessDate (from caseAttachments[])
    for att in case_obj.get("caseAttachments", []):
        val = _nonempty(first_value(att.get("metadata", {}).get("attachmentPublicationBusinessDate", [])))
        if val:
            return val, "attachment_publication"

    # 7. caseOfficialJournalPublicationsPublishedDates
    val = _nonempty(first_value(meta.get("caseOfficialJournalPublicationsPublishedDates", [])))
    if val:
        return val, "case_oj"

    # 8. caseInitiationDate (last fallback)
    val = _nonempty(first_value(meta.get("caseInitiationDate", [])))
    if val:
        return val, "initiation"

    return None, "missing"


def extract_case_row(case_id: str, case_obj: dict) -> dict:
    """
    Extracts the relevant fields from a case object.

    Parameters:
        case_id  : the key from the JSON (e.g. 'AT.39294')
        case_obj : the complete case object with metadata, caseAttachments, decisions
    """
    meta = case_obj.get("metadata", {})
    doc_type, doc_url = get_document_type_and_url(case_obj)
    date_val, date_source = resolve_date(case_obj)

    row = {
        "ec_case_number": first_value(meta.get("caseNumber", [])) or case_id,
        "case_title"    : first_value(meta.get("caseTitle", [])),
        "date"          : date_val,
        "date_source"   : date_source,
        "type"          : first_value(meta.get("caseType", [])),
        "document_type" : doc_type,
        "document_url"  : doc_url,
    }
    return row


print("extract_case_row() and document helper functions defined.")

# --- Cell 7: 7 – Main Processing: Iterate, filter and flatten all cases ---
# 7 – Main Processing: Iterate, filter and flatten all cases
# Now we combine everything:
# 1. Read each case from the JSON file
# 2. Check whether it is an antitrust case
# 3. If yes: convert to a flat row
# 4. Merge all rows into a DataFrame
source_file_name = INPUT_FILE.name
rows = []
skipped_non_antitrust = 0
skipped_errors = 0

if file_type == "json":
    for case_id, case_obj in raw_data.items():
        try:
            meta = case_obj.get("metadata", {})

            # Apply antitrust filter
            if not is_antitrust_case(meta):
                skipped_non_antitrust += 1
                continue

            row = extract_case_row(case_id, case_obj)
            rows.append(row)

        except Exception as e:
            print(f"  WARNING: Error for case '{case_id}': {e}")
            skipped_errors += 1


# Create DataFrame
df = pd.DataFrame(rows)

print(f"\nProcessing complete:")
print(f"  Antitrust cases found      : {len(df)}")
print(f"  Non-antitrust skipped  : {skipped_non_antitrust}")
print(f"  Errors during processing       : {skipped_errors}")

# --- Cell 8: 8 – Normalize Date Columns ---
# 8 – Normalize Date Columns
df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
df["date"] = df["date"].where(df["date"].notna(), other=None)
df = df.replace("", None)

print("Date columns normalized.")
print(df[["ec_case_number", "date", "date_source"]].head(5))

# --- Cell 9: 9 – Quality Checks ---
# 9 – Quality Checks
# Before exporting, we check data quality:
# - How many cases do we have?
# - How many case IDs are missing?
# - Are there duplicates?
# - How are case types distributed?
print("=" * 60)
print("QUALITY CHECKS")
print("=" * 60)

total_input = len(raw_data)
print(f"\nNumber of input cases (total in file) : {total_input}")
print(f"Number of antitrust cases (after filter) : {len(df)}")
print(f"Share of antitrust                     : {len(df) / total_input * 100:.1f}%")

missing_ids   = df["ec_case_number"].isna().sum()
dupes         = df["ec_case_number"].duplicated().sum()
missing_title = df["case_title"].isna().sum()
missing_date  = df["date"].isna().sum()

print(f"\nMissing case IDs (ec_case_number)   : {missing_ids}")
print(f"Duplicates (ec_case_number)           : {dupes}")
print(f"Missing case titles                   : {missing_title}")
print(f"Missing date values                 : {missing_date}")

# --- Cell 10 ---
print("\nFrequency distribution: type")
print("-" * 40)
print(df["type"].value_counts(dropna=False).to_string())

# --- Cell 11 ---
print("\nOverview of all columns and data types:")
print(df.dtypes)
print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")

# --- Cell 12: 10 – Export ---
# 10 – Export
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"CSV saved: {OUTPUT_CSV.resolve()}")
print(f"  File size: {OUTPUT_CSV.stat().st_size / 1024:.1f} KB")

