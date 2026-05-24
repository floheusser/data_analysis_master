# Original notebook: 03_merge_pre1997_into_ec_master.ipynb
# Converted to Python script on: 2026-05-24
# Outputs and markdown cells have been removed.
# Code logic has been preserved as closely as possible.

# --- Cell 1: 1. Imports ---
# 1. Imports
import pandas as pd
from pathlib import Path

print("Imports OK")

# --- Cell 2: 2. Paths ---
# 2. Paths
MASTER_PATH = Path("data/processed/ec_antitrust_master.csv")
LEGACY_PATH = Path("data/processed/ec_legacy_formal_decisions.csv")
OUTPUT_PATH = Path("data/processed/ec_antitrust_master.csv")

FINAL_COLUMNS = ["ec_case_number", "case_title", "date", "date_source", "type", "celex_no", "document_type", "document_url"]

print(f"Master : {MASTER_PATH}")
print(f"Legacy : {LEGACY_PATH}")
print(f"Output : {OUTPUT_PATH}")

# --- Cell 3: 3. Load Master ---
# 3. Load Master
df_master = pd.read_csv(MASTER_PATH, dtype=str)
print(f"Master loaded: {len(df_master)} rows, columns: {df_master.columns.tolist()}")
df_master.head(3)

# --- Cell 4: 4. Load Legacy ---
# 4. Load Legacy
df_legacy = pd.read_csv(LEGACY_PATH, dtype=str)
print(f"Legacy loaded: {len(df_legacy)} rows, columns: {df_legacy.columns.tolist()}")
df_legacy.head(3)

# --- Cell 5: 5. Map Columns to Target Schema ---
# 5. Map Columns to Target Schema
# Master: columns already matching (ec_case_number, case_title, date, date_source, type)
# celex_no missing in master -> leave empty
df_master_mapped = df_master.reindex(columns=FINAL_COLUMNS)

print(f"Master mapped: {df_master_mapped.shape}")
df_master_mapped.head(3)

# --- Cell 6 ---
# Legacy: rename columns and align to target schema
# Vorhandene columns in ec_legacy_formal_decisions.csv:
#   year, title, date_raw, date, date_source, decision_type, official_journal,
#   celex_no, case_numbers, source_url, document_url
df_legacy_mapped = df_legacy.rename(columns={
    "case_numbers": "ec_case_number",
    "title":        "case_title",
    "date":         "date",
    "date_source":  "date_source",
    "celex_no":     "celex_no",
})

# type as constant value
df_legacy_mapped["type"] = "formal_decision"

df_legacy_mapped = df_legacy_mapped.reindex(columns=FINAL_COLUMNS)

print(f"Legacy mapped: {df_legacy_mapped.shape}")
df_legacy_mapped.head(3)

# --- Cell 7: 6. Merge ---
# 6. Merge
df_combined = pd.concat([df_master_mapped, df_legacy_mapped], ignore_index=True)
print(f"Combined: {len(df_combined)} rows")

# --- Cell 8: 7. Final Column Order ---
# 7. Final Column Order
df_final = df_combined[FINAL_COLUMNS]
print(f"Final columns: {df_final.columns.tolist()}")
print(f"Final shape  : {df_final.shape}")
df_final.head(5)

# --- Cell 9: 8. Quality Summary ---
# 8. Quality Summary
total        = len(df_final)
filled       = df_final["date"].notna().sum()
missing      = df_final["date"].isna().sum()
by_source    = df_final["date_source"].value_counts(dropna=False)

# Year range (valid dates only)
years = pd.to_datetime(df_final["date"], errors="coerce").dt.year.dropna()
first_year = int(years.min()) if len(years) > 0 else None
last_year  = int(years.max()) if len(years) > 0 else None

print("=" * 50)
print("EC CASES – DATE QUALITY SUMMARY")
print("=" * 50)
print(f"  Total EC cases          : {total}")
print(f"  Cases with date filled  : {filled}  ({filled/total*100:.1f}%)")
print(f"  Cases still missing     : {missing}  ({missing/total*100:.1f}%)")
print(f"  First year              : {first_year}")
print(f"  Last year               : {last_year}")
print()
print("  Counts by date_source:")
for src, cnt in by_source.items():
    print(f"    {str(src):<25} {cnt}")
print("=" * 50)

# --- Cell 10: 9. ec_antitrust_master.csv save ---
# 9. ec_antitrust_master.csv save
df_final.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
print(f"✅ Saved: {OUTPUT_PATH.resolve()}")
print(f"   {len(df_final)} rows, {len(df_final.columns)} columns")

