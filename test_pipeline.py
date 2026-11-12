"""Layer tests on small synthetic data. Run: python -m pytest -q"""
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from cleaning import clean
from insights import analyse
from models import Insight, Paragraph, TimeWindow
from report import build_report

SITE = "Test Site"


def raw_rows() -> pd.DataFrame:
    """One CAPA per week from Jul 2024 to Jun 2025, plus deliberate defects."""
    rows = []
    for i, d in enumerate(pd.date_range("2024-07-01", "2025-06-27", freq="7D")):
        rows.append({
            "RecordID": 1000 + i, "CAPA_Number": f"TST-CAPA-{i}", "Title": f" Action {i} ",
            "SiteName": SITE, "ResponsibleDepartment": ["Production", "QA", "QC"][i % 3],
            "InitiatedThrough": ["Deviation", "Internal Audit", "Improvement Plans"][i % 3],
            "CAPAClassification": "Improvement", "RecordState": "Closed - Done",
            "Date Opened": d, "Date Closed": d + pd.Timedelta(days=30), "DueDate": d + pd.Timedelta(days=60),
            "CAPAExtensionCounter": None,
        })
    df = pd.DataFrame(rows)
    df.loc[1, "ResponsibleDepartment"] = "production"              # case variant
    df["Date Opened"] = df["Date Opened"].astype(object)
    df.loc[2, "Date Opened"] = "not a date"                         # malformed date
    return pd.concat([df, df.iloc[[0]]], ignore_index=True)         # exact duplicate


def test_cleaning_fixes_and_logs():
    ds = clean(raw_rows())
    log = {(c.step, c.column): c.rows_affected for c in ds.changes}
    assert log[("Remove exact duplicates", "all")] == 1
    assert log[("Parse dates", "opened")] == 1
    assert log[("Standardise labels", "department")] == 1
    assert ds.rows_out == ds.rows_in - 2
    assert {r.department for r in ds.records} == {"Production", "QA", "QC"}
    assert all(r.title == r.title.strip() for r in ds.records)


def test_cleaning_accepts_book1_header_variants():
    df = raw_rows().rename(columns={"Date Opened": "DateOpened", "Date Closed": "date_closed"})
    df["CAPA_Type"] = "Corrective Actions"
    df.loc[0, "CAPA_Number"] = None

    ds = clean(df)

    assert ds.records
    assert {r.capa_type for r in ds.records} == {"Corrective"}
    assert next(r for r in ds.records if r.record_id == 1000).capa_number == "(not recorded)"
    assert next(c.rows_affected for c in ds.changes if c.step == "Fill missing" and c.column == "capa_number") == 1


def test_insights_are_validated_and_consistent():
    b = analyse(clean(raw_rows()), SITE, "2025Q2")
    assert b.quarters == ["Q3 - 2024", "Q4 - 2024", "Q1 - 2025", "Q2 - 2025"]
    total = b.insights["initiated_total"].value
    assert total == next(r[1] for r in b.tables["quarterly_initiated"].rows if r[0] == "Q2 - 2025")
    assert b.insights["oos_oot"].evidence == "insufficient"         # no OOS events at all


def test_optional_chart_is_added_to_matching_report_section(tmp_path):
    dataset = clean(raw_rows())
    bundle = analyse(dataset, SITE, "2025Q2")

    report = build_report(bundle, dataset, tmp_path, chart_overrides={"classification": "pie"})

    section = next(section for section in report.sections if section.number == "5.5")
    assert section.figures[0].caption == "CAPA classification by quarter (Pie view)"
    assert Path(section.figures[0].path).is_file()


def test_untraced_number_is_rejected():
    w = TimeWindow(label="Q2 - 2025", start=date(2025, 4, 1), end=date(2025, 6, 30))
    ins = Insight(id="x", title="x", value=98, unit="CAPAs", window=w, interpretation="98 CAPAs in Q2 - 2025.",
                  evidence="strong", method="count")
    Paragraph(text="A total of 98 CAPAs were opened in Apr - 2025 to Jun - 2025.", sources=[ins])
    with pytest.raises(ValidationError):
        Paragraph(text="A total of 99 CAPAs were opened.", sources=[ins])
