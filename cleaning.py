"""Layer 1 - data cleaning.

Input : path to the CAPA export (.xlsx or .csv) or a raw DataFrame.
Output: models.CleanedDataset (validated records + log of every change).

Every step appends a CleaningChange, including checks that changed nothing,
so the report's data-quality appendix shows what was checked as well as what
was fixed.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from models import CapaRecord, CleanedDataset, CleaningChange

log = logging.getLogger(__name__)

COLUMNS = {
    "RecordID": "record_id", "CAPA_Number": "capa_number", "Title": "title",
    "SiteName": "site", "ResponsibleDepartment": "department",
    "InitiatedThrough": "initiated_through", "CAPAClassification": "classification",
    "RecordState": "state", "Date Opened": "opened", "Date Closed": "closed",
    "DueDate": "due_date", "CAPAExtensionCounter": "extension_count",
}
# Alternative export layout (TrackWise silver-layer CSV). Either layout is accepted.
COLUMNS_ALT = {
    "record_id": "record_id", "capa_number": "capa_number", "capa_title": "title",
    "site_name": "site", "resp_dept": "department", "origin_qms": "initiated_through",
    "capa_classification": "classification", "record_status": "state",
    "date_opened": "opened", "date_completed": "closed", "due_date": "due_date",
    "ext_counter": "extension_count",
}
HEADER_ALIASES = {
    "DateOpened": "Date Opened", "date_closed": "Date Closed", "CAPA_Type": "capa_type",
}
CAPA_TYPE_MAP = {"Corrective Actions": "Corrective", "Preventive Actions": "Preventive"}
DATE_COLS = ["opened", "closed", "due_date"]
TEXT_COLS = ["capa_number", "title", "site", "department", "initiated_through",
             "classification", "state"]

# Known spelling / naming variants found in the export. Case and whitespace
# variants are merged automatically; these cover genuinely different spellings.
LABEL_FIXES = {
    "department": {
        "Engineering Department": "Engineering",
        "Quality Assurance Validation": "Quality Assurance - Validation Group",
    },
    "classification": {"Venodr/Manufacturer": "Vendor/Manufacturer"},
    "site": {"Onesource Corp": "Onesource Corporate"},
}
MISSING_FILL = {
    "capa_number": "(not recorded)", "title": "(no title)", "department": "Unassigned",
    "initiated_through": "Not recorded", "classification": "Not recorded",
}

# ASSUMPTION: the export has no Corrective/Preventive field (the reference's
# section 5.10 needs one). CAPAs raised in response to an event that already
# happened are treated as Corrective; planned/proactive sources as Preventive.
CORRECTIVE_SOURCES = {
    "Deviation", "OOS", "OOT", "Complaint", "Internal Audit",
    "Customer Inspections", "Regulatory Inspections", "Customer Notification",
}


def load(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def clean(source: str | Path | pd.DataFrame) -> CleanedDataset:
    raw = source if isinstance(source, pd.DataFrame) else load(source)
    name = "<dataframe>" if isinstance(source, pd.DataFrame) else Path(source).name
    changes: list[CleaningChange] = []

    def note(step, column, rows, action, reason):
        changes.append(CleaningChange(step=step, column=column, rows_affected=int(rows),
                                      action=action, reason=reason))
        log.info("%s [%s]: %s rows - %s", step, column, rows, action)

    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for alias, canonical in HEADER_ALIASES.items():
        if alias in df.columns and canonical not in df.columns:
            df = df.rename(columns={alias: canonical})
    layout = COLUMNS if set(COLUMNS) <= set(df.columns) else COLUMNS_ALT
    missing = set(layout) - set(df.columns)
    if missing:
        raise ValueError(f"export is missing required columns: {sorted(missing)}")
    has_type = "capa_type" in df.columns

    # 1. Columns that carry no data at all
    empty = [c for c in df.columns if df[c].isna().all()]
    note("Drop empty columns", ", ".join(empty) or "-", len(empty), "dropped",
         "column is 100% blank in the export and carries no information")

    # 2. Pre-computed period columns: check them, then rely on derived values
    if "Date OpenedQuarter" in df.columns:
        derived_q = pd.to_datetime(df["Date Opened"], errors="coerce").dt.to_period("Q")
        given = df["Date OpenedQuarter"].astype(str).str.replace(" ", "")
        expected = "Q" + derived_q.dt.quarter.astype("string") + "-" + derived_q.dt.year.astype("string")
        bad = (given != expected).sum()
        note("Verify quarter labels", "Date OpenedQuarter", bad, "ignored; quarters re-derived from dates",
             "pre-computed labels can drift from the dates; derive from the source date instead")

    keep = list(layout) + (["capa_type"] if has_type else [])
    df = df[keep].rename(columns=layout)

    # 3. Exact duplicate rows, then duplicate record IDs
    n = len(df)
    df = df.drop_duplicates()
    note("Remove exact duplicates", "all", n - len(df), "dropped", "identical rows double-count CAPAs")
    n = len(df)
    df = df.drop_duplicates(subset="record_id", keep="first")
    note("Remove duplicate IDs", "record_id", n - len(df), "kept first occurrence",
         "RecordID is the TrackWise primary key and must be unique")

    # 4. Types: record id and extension counter
    ids = pd.to_numeric(df["record_id"], errors="coerce")
    note("Coerce types", "record_id", ids.isna().sum(), "rows with non-numeric IDs dropped",
         "a record without a valid ID cannot be traced")
    df = df[ids.notna()].assign(record_id=ids.dropna().astype(int))
    ext = pd.to_numeric(df["extension_count"], errors="coerce")
    note("Fill missing", "extension_count", ext.isna().sum(), "blank -> 0",
         "ASSUMPTION: TrackWise leaves the counter blank when no extension was raised")
    df["extension_count"] = ext.fillna(0).clip(lower=0).astype(int)

    # 5. Whitespace and label consistency
    for col in TEXT_COLS:
        s = df[col].astype("string")
        tidy = s.str.strip().str.replace(r"\s+", " ", regex=True)
        changed = ((s != tidy) & s.notna()).sum()
        if changed:
            note("Trim whitespace", col, changed, "stripped / collapsed spaces", "stray spaces split one label into two")
        df[col] = tidy
    for col in ["site", "department", "initiated_through", "classification", "state"]:
        # merge case variants onto the most frequent spelling
        counts = df[col].value_counts()
        canon = {v: counts[counts.index.str.lower() == v.lower()].idxmax() for v in counts.index}
        canon.update(LABEL_FIXES.get(col, {}))
        mapped = df[col].map(lambda v: canon.get(v, v) if pd.notna(v) else v)
        changed = ((mapped != df[col]) & df[col].notna()).sum()
        if changed or col in LABEL_FIXES:
            fixes = {k: v for k, v in canon.items() if k != v and (df[col] == k).any()}
            note("Standardise labels", col, changed, "; ".join(f"'{k}' -> '{v}'" for k, v in fixes.items()) or "none needed",
                 "the same category was spelled differently and would be split in counts")
        df[col] = mapped

    # 6. Missing categorical values
    for col, fill in MISSING_FILL.items():
        n_missing = df[col].isna().sum()
        note("Fill missing", col, n_missing, f"blank -> '{fill}'", "keep the record in totals while making the gap visible")
        df[col] = df[col].fillna(fill)

    # 7. Dates: coerce, drop rows that cannot be placed in time
    for col in DATE_COLS:
        # utc=True accepts ISO strings with 'Z' offsets; drop the zone so all dates are naive
        parsed = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_localize(None)
        malformed = (parsed.isna() & df[col].notna()).sum()
        note("Parse dates", col, malformed, "unparseable values set to blank",
             "malformed dates cannot be trended")
        df[col] = parsed
    n = len(df)
    df = df[df["opened"].notna() & df["due_date"].notna()]
    note("Drop undatable rows", "opened, due_date", n - len(df), "dropped",
         "a CAPA without an opening or due date cannot be placed in any trend window")

    # 8. State vs closure-date consistency
    is_closed = df["state"].str.startswith("Closed")
    stray = (~is_closed & df["closed"].notna()).sum()
    note("Reconcile state", "closed", stray, "closure date cleared on open records",
         "RecordState is the system of record for whether a CAPA is closed")
    df.loc[~is_closed, "closed"] = pd.NaT
    n = len(df)
    df = df[~(is_closed & df["closed"].isna())]
    note("Reconcile state", "closed", n - len(df), "closed records with no closure date dropped",
         "closure timing is unknown, so the record cannot be placed in the closed trend")
    n = len(df)
    df = df[~(df["closed"] < df["opened"])]
    note("Validate dates", "closed", n - len(df), "rows closed before opening dropped", "impossible chronology")

    # 9. Outliers: flagged, not removed (long-running CAPAs are real and matter)
    days = (df["closed"] - df["opened"]).dt.days.dropna()
    q1, q3 = days.quantile([0.25, 0.75])
    fence = q3 + 3 * (q3 - q1)
    note("Flag outliers", "days to close", (days > fence).sum(),
         f"flagged (> {fence:.0f} days, Q3 + 3xIQR); retained",
         "extreme cycle times are genuine long-running CAPAs; medians are used so they do not distort averages")
    early_due = (df["due_date"] < df["opened"].dt.normalize()).sum()
    note("Flag outliers", "due_date", early_due, "flagged (due date before opening); retained",
         "likely data-entry error; affects only the on-time metric for these rows")

    # 10. Derived field
    derived = df["initiated_through"].map(lambda s: "Corrective" if s in CORRECTIVE_SOURCES else "Preventive")
    if has_type:
        given = df["capa_type"].map(CAPA_TYPE_MAP)
        note("Fill missing", "capa_type", given.isna().sum(), "blank -> derived from initiation source",
             "CAPA type is required for the type trend; source-based rule used only where blank")
        df["capa_type"] = given.fillna(derived)
    else:
        df["capa_type"] = derived
        note("Derive field", "capa_type", len(df), "Corrective/Preventive derived from initiation source",
             "export has no CAPA type field; see assumptions")

    # 11. Row-level validation through the Pydantic model
    records, rejected = [], 0
    for row in df.to_dict("records"):
        row = {k: (None if pd.isna(v) else v) for k, v in row.items()}
        try:
            records.append(CapaRecord.model_validate(row))
        except ValidationError as exc:
            rejected += 1
            log.warning("record %s rejected: %s", row.get("record_id"), exc.errors()[0]["msg"])
    note("Validate records", "all", rejected, "rows failing CapaRecord validation dropped",
         "enforces types and state/date consistency at the layer boundary")

    complete_from = df["closed"].min().date()
    extract_end = max(df["opened"].max(), df["closed"].max()).date()
    assumptions = [
        f"Extract coverage: the earliest closure in the file is {complete_from}, so the export appears to contain "
        f"every CAPA open on or after that date plus any closed after it. Quarters opened before {complete_from} "
        "are incomplete and are not used for trending.",
        ("CAPA type is taken from the export's capa_type field; where blank it is derived from the initiation source "
         if has_type else
         "CAPA type (Corrective/Preventive) is not in the export. It is derived from the initiation source: ")
        + f"{', '.join(sorted(CORRECTIVE_SOURCES))} -> Corrective; all other sources -> Preventive.",
        "Duplicate record IDs (one row per child record in the export) are collapsed to the first row per CAPA.",
        "Closed counts use 'Closed - Done' only; 'Closed - Cancelled' records are reported separately "
        "(as in the reference report). Initiated counts include records later cancelled.",
        "On-time closure compares the closure date with the current DueDate in the export (which already "
        "includes approved extensions).",
        "A blank CAPAExtensionCounter means no extension was raised.",
    ]
    return CleanedDataset(source_file=name, rows_in=len(raw), rows_out=len(records),
                          complete_from=complete_from, extract_end=extract_end,
                          records=records, changes=changes, assumptions=assumptions)
