"""Pydantic contracts between the three pipeline layers.

    cleaning.clean()     -> CleanedDataset   (layer 1 output / layer 2 input)
    insights.analyse()   -> InsightBundle    (layer 2 output / layer 3 input)
    report.build_report()-> TrendReport      (layer 3, rendered to .docx)

Traceability rule: every number that appears in narrative text (an Insight's
interpretation or a report Paragraph) must equal a figure stored on the Insight
it cites (`value` or one of `facts`). Period labels such as "Q2 - 2025" or
"Apr - 2025" are not counted as numbers. The validators below enforce this, so a
report that quotes an untraceable number cannot be constructed.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Evidence = Literal["strong", "moderate", "weak", "insufficient"]

# Numbers a narrative may use without citing a fact: the significance level and
# the display floor for very small p-values.
STATISTICAL_CONSTANTS = {0.05, 0.001}

_PERIOD_PATTERNS = [
    r"Q[1-4]\s*-\s*\d{4}",                                            # Q2 - 2025
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*-?\s*\d{4}",  # Apr - 2025
    r"\b\d{4}-\d{2}-\d{2}\b",                                         # ISO dates
]
_NUMBER = re.compile(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?")


def untraced_numbers(text: str, allowed: set[float]) -> list[str]:
    """Return numeric tokens in `text` that do not match any allowed figure
    at the precision they are printed with."""
    for pat in _PERIOD_PATTERNS:
        text = re.sub(pat, " ", text)
    allowed = {abs(v) for v in allowed} | STATISTICAL_CONSTANTS
    bad = []
    for tok in _NUMBER.findall(text):
        num = float(tok.replace(",", ""))
        decimals = len(tok.split(".")[1]) if "." in tok else 0
        if not any(round(v, decimals) == num for v in allowed):
            bad.append(tok)
    return bad


# --------------------------------------------------------------------------
# Layer 1: data cleaning output
# --------------------------------------------------------------------------
class CapaRecord(BaseModel):
    """One validated CAPA record."""
    model_config = ConfigDict(frozen=True)

    record_id: int
    capa_number: str
    title: str
    site: str
    department: str
    initiated_through: str
    classification: str
    state: str
    opened: datetime
    closed: datetime | None
    due_date: datetime
    extension_count: int = Field(ge=0)
    capa_type: Literal["Corrective", "Preventive"]

    @property
    def is_closed(self) -> bool:
        return self.state.startswith("Closed")

    @model_validator(mode="after")
    def _consistent(self) -> "CapaRecord":
        if self.is_closed and self.closed is None:
            raise ValueError("closed state without a closure date")
        if not self.is_closed and self.closed is not None:
            raise ValueError("open state with a closure date")
        if self.closed is not None and self.closed < self.opened:
            raise ValueError("closure date precedes opening date")
        return self


class CleaningChange(BaseModel):
    """One logged transformation (or check) applied by the cleaning layer."""
    step: str
    column: str
    rows_affected: int = Field(ge=0)
    action: str
    reason: str


class CleanedDataset(BaseModel):
    source_file: str
    rows_in: int = Field(ge=0)
    rows_out: int = Field(ge=0)
    complete_from: date      # opened-date from which the extract is complete
    extract_end: date        # latest date present in the extract
    records: list[CapaRecord]
    changes: list[CleaningChange]
    assumptions: list[str]

    @model_validator(mode="after")
    def _counts(self) -> "CleanedDataset":
        if self.rows_out != len(self.records):
            raise ValueError("rows_out does not match number of records")
        if len({r.record_id for r in self.records}) != len(self.records):
            raise ValueError("duplicate record_id survived cleaning")
        return self

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([r.model_dump() for r in self.records])


# --------------------------------------------------------------------------
# Layer 2: statistical insight output
# --------------------------------------------------------------------------
class TimeWindow(BaseModel):
    label: str
    start: date
    end: date

    @model_validator(mode="after")
    def _ordered(self) -> "TimeWindow":
        if self.start > self.end:
            raise ValueError("window start after end")
        return self


class Insight(BaseModel):
    """A single statistical finding with its supporting figures."""
    id: str
    title: str
    value: float
    unit: str
    window: TimeWindow
    facts: dict[str, float] = {}
    interpretation: str
    evidence: Evidence
    method: str

    @property
    def figures(self) -> set[float]:
        return {self.value, *self.facts.values()}

    @model_validator(mode="after")
    def _traceable(self) -> "Insight":
        bad = untraced_numbers(self.interpretation, self.figures)
        if bad:
            raise ValueError(f"insight {self.id!r} quotes untraced numbers {bad}")
        return self


class Table(BaseModel):
    title: str
    columns: list[str] = Field(min_length=1)
    rows: list[list[str | int | float]]
    note: str | None = None

    @model_validator(mode="after")
    def _rectangular(self) -> "Table":
        for row in self.rows:
            if len(row) != len(self.columns):
                raise ValueError(f"table {self.title!r}: row width {len(row)} != {len(self.columns)}")
        return self


class MonthlyPoint(BaseModel):
    month: str
    opened: int = Field(ge=0)
    closed: int = Field(ge=0)
    opened_rolling3: float | None = None


class ControlChart(BaseModel):
    """c-chart (Poisson) limits for monthly CAPA initiations."""
    center: float
    ucl: float
    lcl: float = Field(ge=0)
    points: list[MonthlyPoint]
    flagged_months: list[str]


class AgingItem(BaseModel):
    record_id: int
    title: str
    department: str
    opened: date
    due_date: date
    age_days: int = Field(gt=90)
    overdue: bool
    extensions: int
    state: str


class InsightBundle(BaseModel):
    site: str
    site_code: str
    period: TimeWindow
    quarters: list[str] = Field(min_length=4, max_length=4)
    insights: dict[str, Insight]
    tables: dict[str, Table]
    control: ControlChart
    aging: list[AgingItem]
    data_notes: list[str]

    @field_validator("insights")
    @classmethod
    def _keys_match(cls, v: dict[str, Insight]) -> dict[str, Insight]:
        for k, ins in v.items():
            if k != ins.id:
                raise ValueError(f"insight key {k!r} != id {ins.id!r}")
        return v


# --------------------------------------------------------------------------
# Layer 3: report structure
# --------------------------------------------------------------------------
class Paragraph(BaseModel):
    lead: str | None = None          # bold lead-in, as in the reference report
    text: str
    sources: list[Insight] = Field(min_length=1)

    @property
    def evidence(self) -> Evidence:
        """Evidence of the primary (first-cited) insight."""
        return self.sources[0].evidence

    @model_validator(mode="after")
    def _traceable(self) -> "Paragraph":
        allowed = set().union(*(s.figures for s in self.sources))
        bad = untraced_numbers(self.text, allowed)
        if bad:
            raise ValueError(f"paragraph quotes numbers not in cited insights: {bad}: {self.text[:80]}")
        return self


class Figure(BaseModel):
    caption: str
    path: str


class ReportSection(BaseModel):
    number: str
    title: str
    paragraphs: list[Paragraph] = Field(min_length=1)
    tables: list[Table] = []
    figures: list[Figure] = []


REQUIRED_SECTIONS = ["5.1", "5.2", "5.3", "5.4", "5.5", "5.10"]


class TrendReport(BaseModel):
    title: str
    organisation: str
    site: str
    period_label: str
    generated_on: date
    executive_summary: list[Paragraph] = Field(min_length=3)
    sections: list[ReportSection]
    data_quality: Table
    insight_register: Table          # every cited insight: id, value, window, method
    assumptions: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _structure(self) -> "TrendReport":
        numbers = [s.number for s in self.sections]
        if numbers != REQUIRED_SECTIONS:
            raise ValueError(f"sections must be {REQUIRED_SECTIONS}, got {numbers}")
        return self
