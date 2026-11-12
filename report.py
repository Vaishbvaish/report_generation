"""Layer 3 - report generation.

Input : models.InsightBundle (+ the CleanedDataset for the data-quality appendix).
Output: models.TrendReport, rendered to .docx (the reference report's format).

All narrative is built from Insight objects; each Paragraph cites the insights
it uses and the Paragraph validator rejects any number not traceable to them.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from models import (CleanedDataset, Figure, Insight, InsightBundle, Paragraph, ReportSection,
                    Table, TrendReport)

ORGANISATION = "OneSource Specialty Pharma Limited"   # as in the reference report
QUARTER_NAMES = {1: "First", 2: "Second", 3: "Third", 4: "Fourth"}

# Chart styling: validated categorical order (dataviz reference palette), recessive ink/grid.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MUTED_BLUE, INK, INK_2, GRID, CRITICAL = "#86b6ef", "#0b0b0b", "#52514e", "#e5e4e0", "#d03b3b"
REPORT_CHART_SPECS = {
    "quarterly": {"section": "5.1", "figure": 0, "default": "bar", "types": ("bar", "line", "pie", "doughnut")},
    "control": {"section": "5.2", "figure": 0, "default": "line", "types": ("line", "bar")},
    "flow": {"section": "5.2", "figure": 1, "default": "bar", "types": ("bar", "line")},
    "departments": {"section": "5.3", "figure": 0, "default": "bar", "types": ("bar", "line", "pie", "doughnut")},
    "sources": {"section": "5.4", "figure": 0, "default": "bar", "types": ("bar", "line", "pie", "doughnut")},
    "classification": {"section": "5.5", "figure": 0, "default": "bar", "types": ("bar", "line", "pie", "doughnut")},
    "capa_type": {"section": "5.10", "figure": 0, "default": "bar", "types": ("bar", "line", "pie", "doughnut")},
}


# ---------------------------------------------------------------- charts
def _axes(title: str, figsize=(7.5, 3.6)):
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return str(path)


def chart_quarterly(t: Table, current: str, path: Path) -> str:
    rows = [r for r in t.rows if r[0] != "Total"]
    fig, ax = _axes("CAPAs initiated per quarter")
    labels, vals = [r[0] for r in rows], [r[1] for r in rows]
    bars = ax.bar(labels, vals, width=0.55, color=[SERIES[0] if l == current else MUTED_BLUE for l in labels])
    ax.bar_label(bars, padding=3, fontsize=8, color=INK)
    return _save(fig, path)


def chart_control(b: InsightBundle, path: Path) -> str:
    c = b.control
    months = [p.month for p in c.points]
    x = range(len(months))
    fig, ax = _axes("Monthly CAPA initiations - individuals control chart", figsize=(7.5, 3.8))
    ax.plot(x, [p.opened for p in c.points], color=SERIES[0], lw=2, marker="o", ms=4, label="Initiated")
    ax.plot(x, [p.opened_rolling3 for p in c.points], color=SERIES[1], lw=2, ls="--", label="3-month rolling avg")
    ax.plot(x, [p.closed for p in c.points], color=SERIES[2], lw=2, marker="s", ms=4, label="Closed (done)")
    for y, name, ls in ((c.center, "Mean", "-"), (c.ucl, "UCL", ":"), (c.lcl, "LCL", ":")):
        ax.axhline(y, color=INK_2, lw=1, ls=ls)
        ax.annotate(f" {name} {y:.1f}", (len(months) - 0.6, y), xytext=(0, 2), textcoords="offset points", va="bottom", fontsize=7, color=INK_2)
    for i, p in enumerate(c.points):
        if p.month in c.flagged_months:
            ax.plot(i, p.opened, "o", ms=9, mfc="none", mec=CRITICAL, mew=2)
    ax.set_xticks(list(x), [m.replace(" - 20", " ") for m in months], rotation=45, ha="right")
    ax.set_xlim(-0.5, len(months) + 0.8)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    return _save(fig, path)


def chart_grouped(categories: list[str], series: dict[str, list[float]], title: str, path: Path,
                  horizontal=False) -> str:
    fig, ax = _axes(title, figsize=(7.5, 0.45 * len(categories) + 1.2) if horizontal else (7.5, 3.6))
    width = 0.8 / len(series)
    for i, (name, vals) in enumerate(series.items()):
        pos = [k + (i - (len(series) - 1) / 2) * width for k in range(len(categories))]
        (ax.barh if horizontal else ax.bar)(pos, vals, width * 0.92, color=SERIES[i], label=name)
    if horizontal:
        ax.set_yticks(range(len(categories)), categories)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
    else:
        ax.set_xticks(range(len(categories)), categories)
    ax.legend(fontsize=8, frameon=False)
    return _save(fig, path)


def chart_stacked(t: Table, title: str, path: Path) -> str:
    """Stacked bars by quarter from a category x quarter table (Total row/column ignored)."""
    rows = [r for r in t.rows if r[0] != "Total"]
    quarters = t.columns[1:-1]
    fig, ax = _axes(title)
    bottom = [0] * len(quarters)
    for i, r in enumerate(rows[:len(SERIES)]):
        vals = r[1:-1]
        ax.bar(quarters, vals, 0.55, bottom=bottom, color=SERIES[i], label=r[0], edgecolor="white", linewidth=1.5)
        bottom = [a + v for a, v in zip(bottom, vals)]
    ax.legend(fontsize=8, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    return _save(fig, path)


def _custom_chart_data(chart_id: str, b: InsightBundle) -> tuple[list[str], dict[str, list[float]], str | None]:
    current = b.quarters[-1]
    if chart_id == "quarterly":
        rows = [r for r in b.tables["quarterly_initiated"].rows if r[0] != "Total"]
        return [str(r[0]) for r in rows], {"CAPAs initiated": [float(r[1]) for r in rows]}, "CAPAs initiated"
    if chart_id == "control":
        points = b.control.points
        return ([p.month for p in points],
                {"Initiated": [float(p.opened) for p in points],
                 "Closed (done)": [float(p.closed) for p in points]}, None)
    if chart_id == "flow":
        rows = b.tables["quarterly_flow"].rows
        return ([str(r[0]) for r in rows],
                {"Initiated": [float(r[1]) for r in rows],
                 "Closed (done)": [float(r[2]) for r in rows]}, None)
    if chart_id == "departments":
        rows = b.tables["dept_movers"].rows
        return ([str(r[0]) for r in rows],
                {"Current quarter": [float(r[1]) for r in rows],
                 "Prior 3-quarter avg": [float(r[2]) for r in rows]}, "Current quarter")
    if chart_id == "sources":
        table = b.tables["source_group"]
        rows = [r for r in table.rows if r[0] != "Total"]
        categories = table.columns[1:4]
        series = {str(r[0]): [float(r[i + 1]) for i in range(len(categories))] for r in rows}
        return list(categories), series, current

    table_key = "classification" if chart_id == "classification" else "capa_type"
    table = b.tables[table_key]
    rows = [r for r in table.rows if r[0] != "Total"]
    quarters = table.columns[1:-1]
    series = {str(q): [float(r[i + 1]) for r in rows] for i, q in enumerate(quarters)}
    return [str(r[0]) for r in rows], series, current


def chart_custom(categories: list[str], series: dict[str, list[float]], pie_series: str | None,
                 chart_type: str, title: str, path: Path) -> str:
    if chart_type in {"pie", "doughnut"}:
        selected = pie_series or next(iter(series))
        values = [(label, value) for label, value in zip(categories, series[selected]) if value > 0]
        if not values:
            raise ValueError(f"no positive values available for {title}")
        if len(values) > 8:
            values.sort(key=lambda item: item[1], reverse=True)
            values = values[:7] + [("Other", sum(value for _, value in values[7:]))]
        labels, counts = zip(*values)
        fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
        ax.pie(counts, labels=labels, autopct="%1.1f%%", startangle=90,
               colors=SERIES[:len(labels)], textprops={"fontsize": 8},
               wedgeprops={"width": 0.48} if chart_type == "doughnut" else {})
        ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
        ax.axis("equal")
    else:
        fig, ax = _axes(title, figsize=(7.5, max(3.6, 0.32 * len(categories) + 1.2)))
        positions = list(range(len(categories)))
        if chart_type == "line":
            for index, (name, values) in enumerate(series.items()):
                ax.plot(positions, values, marker="o", color=SERIES[index % len(SERIES)], label=name)
        else:
            width = 0.8 / len(series)
            for index, (name, values) in enumerate(series.items()):
                offset = (index - (len(series) - 1) / 2) * width
                bars = ax.bar([position + offset for position in positions], values, width * 0.92,
                              color=SERIES[index % len(SERIES)], label=name)
                ax.bar_label(bars, padding=2, fontsize=7, color=INK)
        ax.set_xticks(positions, categories, rotation=35 if len(categories) > 6 else 0,
                      ha="right" if len(categories) > 6 else "center")
        if len(series) > 1:
            ax.legend(fontsize=8, frameon=False)
    return _save(fig, path)


# ---------------------------------------------------------------- report model
def build_report(b: InsightBundle, ds: CleanedDataset, out_dir: Path,
                 chart_overrides: dict[str, str] | None = None) -> TrendReport:
    I = b.insights
    out_dir.mkdir(parents=True, exist_ok=True)
    cur = b.quarters[-1]
    q_num, year = int(cur[1]), cur[-4:]
    period_label = f"{b.period.start:%b %Y} - {b.period.end:%b %Y}"

    def P(lead: str | None, *ids: str, text: str | None = None) -> Paragraph:
        src = [I[i] for i in ids]
        return Paragraph(lead=lead, text=text or " ".join(s.interpretation for s in src), sources=src)

    def fig(caption: str, path: str) -> Figure:
        return Figure(caption=caption, path=path)

    T = b.tables
    ic = I["initiated_total"]
    up, down = I.get("dept_up"), I.get("dept_down")
    dept_parts, dept_ids = [], []
    if up:
        dept_ids.append("dept_up")
        dept_parts.append(f"The largest departmental increase is {up.title.split(': ')[1]} "
                          f"({fmt_v(up.facts['avg_prior'], 1)} -> {fmt_v(up.facts['current'])} per quarter, "
                          f"{fmt_p(up.facts['p_value'])}).")
    if down:
        dept_ids.append("dept_down")
        dept_parts.append(f"{down.title.split(': ')[1]} fell ({fmt_v(down.facts['avg_prior'], 1)} -> "
                          f"{fmt_v(down.facts['current'])}, {fmt_p(down.facts['p_value'])})"
                          + (", within normal variation." if down.facts["p_value"] >= 0.05 else "."))
    if not dept_parts:
        dept_ids.append("dept_overview")
        dept_parts.append(f"No department reached {fmt_v(I['dept_overview'].facts['sparse'])} CAPAs in the quarter "
                          "or on average, so department-level changes cannot be tested.")

    summary = [
        P("Volume", "initiated_qoq", "initiated_total",
          text=f"{fmt_v(ic.value)} CAPAs were initiated in {cur}. " + I["initiated_qoq"].interpretation),
        P("Throughput", "closure_ratio"),
        P("Speed", "cycle_time", "on_time",
          text=f"Median time to close rose to {fmt_v(I['cycle_time'].value)} days from "
               f"{fmt_v(I['cycle_time'].facts['median_prior'])} days "
               f"({fmt_p(I['cycle_time'].facts['p_value'])}), while on-time closure held at "
               f"{I['on_time'].value:.1f}%. " + ("CAPAs are taking longer but are still closing within their due dates."
                                              if I["on_time"].value >= 95 else "Slower closure is starting to affect due-date compliance.")
          if I["cycle_time"].value > I["cycle_time"].facts["median_prior"] else None),
        P("Departments", *dept_ids, text=" ".join(dept_parts)),
        P("Watch items", "aging", "oos_oot",
          text=f"{fmt_v(I['aging'].value)} CAPAs had been open for more than {fmt_v(I['aging'].facts['threshold'])} "
               f"days at quarter end ({fmt_v(I['aging'].facts['overdue'])} overdue). OOS/OOT-triggered CAPAs "
               f"numbered {fmt_v(I['oos_oot'].value)} this quarter ({fmt_v(I['oos_oot'].facts['total_4q'])} over four quarters)"
               + ("; too few to trend, so review them case by case." if I["oos_oot"].evidence == "insufficient"
                  else "; see the CAPA source trend section.")),
    ]

    ch = out_dir
    flow = T["quarterly_flow"]
    movers = T["dept_movers"]
    sections = [
        ReportSection(number="5.1", title="CAPA Initiation Trend", paragraphs=[
            P("Total CAPAs Initiated", "initiated_total"),
            P("Quarterly Comparison (Past 4 Quarters)", "initiated_vs_avg"),
            P(None, "initiated_qoq"),
            P(None, "initiated_trend"),
            P("Initiation Source Breakdown", "source_mix"),
            *([P("Site Breakdown", "site_mix")] if "site_mix" in I else []),
            P("Closed-Cancelled Records", "initiated_total",
              text=f"Out of {fmt_v(ic.value)} CAPAs initiated in {cur}, {fmt_v(ic.facts['cancelled'])} were "
                   "subsequently closed as cancelled. They remain in the initiation count but are excluded "
                   "from closure trending."),
        ], tables=[T["quarterly_initiated"], *([T["site_initiated"]] if "site_initiated" in T else [])],
            figures=[fig("CAPAs initiated per quarter (current quarter highlighted)",
                         chart_quarterly(T["quarterly_initiated"], cur, ch / "1_quarterly.png"))]),

        ReportSection(number="5.2", title="Month wise trend of Opened and Closed CAPAs", paragraphs=[
            P("Monthly Breakdown (Initiated)", "monthly_initiated"),
            P("Process Stability", "control"),
            P("Closing Statement", "closed_total"),
            P("Closure Ratio and Backlog", "closure_ratio"),
            P("Time to Close", "cycle_time"),
            P("On-time Closure", "on_time"),
            P("CAPAs Open More Than 90 Days", "aging"),
        ], tables=[T["monthly"], flow, T["aging"]],
            figures=[
                fig("Monthly initiations with control limits, 3-month rolling average and closures",
                    chart_control(b, ch / "2_control.png")),
                fig("Quarterly CAPAs initiated vs closed",
                    chart_grouped([r[0] for r in flow.rows],
                                  {"Initiated": [r[1] for r in flow.rows], "Closed (done)": [r[2] for r in flow.rows]},
                                  "CAPAs initiated vs closed per quarter", ch / "3_flow.png")),
            ]),

        ReportSection(number="5.3", title="Department Wise CAPA Open and Closure Trend", paragraphs=[
            P("Department-wise Initiation Overview", "dept_overview"),
            *[P("Individual Department Analysis" if k == 0 else None, f"dept_{k}")
              for k in range(6) if f"dept_{k}" in I],
            *[P("Significant Changes vs Previous Quarters" if i == 0 else None, k)
              for i, k in enumerate(k for k in ("dept_up", "dept_down") if k in I)],
            P("Closure Analysis and Historical Comparison", "dept_closed"),
        ], tables=[T["dept_opened"], T["dept_closed"], movers],
            figures=[fig("Department CAPAs: current quarter vs average of previous three quarters",
                         chart_grouped([r[0] for r in movers.rows],
                                       {cur: [r[1] for r in movers.rows], "Prior 3-quarter avg": [r[2] for r in movers.rows]},
                                       "Department CAPAs: current vs prior average", ch / "4_departments.png",
                                       horizontal=True))]),

        ReportSection(number="5.4", title="CAPA Purpose Trend and Source of CAPA (Category Trend)", paragraphs=[
            P("Overview Statement", "deviation_share"),
            P("Source Mix Over Time", "source_shift"),
            P("Quality System Assessment", "deviation_share", "source_mix",
              text="CAPAs continue to arise both from deviation investigations (reactive) and from standalone routes "
                   "such as improvement plans, audits and risk assessments (proactive). Both routes are active, which "
                   "indicates the quality system is correcting issues and also driving improvement."),
            P("OOS/OOT Specific Statement", "oos_oot"),
        ], tables=[T["source_group"], T["source_detail"]],
            figures=[fig("CAPA source group by quarter",
                         chart_stacked(_group_as_category_table(T["source_group"]),
                                       "CAPA source group by quarter", ch / "5_sources.png"))]),

        ReportSection(number="5.5", title="CAPA Classification Trend", paragraphs=[
            P("Classification Breakdown", "classification"),
            P("Effectiveness Assessment", "classification",
              text="The export contains no effectiveness-check results (those columns are blank), so effectiveness "
                   "cannot be trended from this data; it remains assessed per CAPA in TrackWise."),
        ], tables=[T["classification"]],
            figures=[fig("CAPA classification by quarter",
                         chart_stacked(T["classification"], "CAPA classification by quarter", ch / "6_classification.png"))]),

        ReportSection(number="5.10", title="CAPA Type Trend", paragraphs=[
            P("Overview and Categorization", "capa_type"),
            P("Process Control Statement", "capa_type",
              text="The balance between corrective and preventive CAPAs "
                   + ("has not shifted significantly; monitoring should continue."
                      if I["capa_type"].facts["p_value"] >= 0.05 else
                      "has shifted significantly and the cause should be reviewed.")),
        ], tables=[T["capa_type"]],
            figures=[fig("CAPA type by quarter",
                         chart_stacked(T["capa_type"], "CAPA type by quarter", ch / "7_type.png"))]),
    ]

    for chart_id, chart_type in (chart_overrides or {}).items():
        spec = REPORT_CHART_SPECS.get(chart_id)
        if not spec or chart_type not in spec["types"]:
            raise ValueError(f"unsupported chart override: {chart_id}={chart_type}")
        target = next(section for section in sections if section.number == spec["section"])
        original = target.figures[spec["figure"]]
        categories, series, pie_series = _custom_chart_data(chart_id, b)
        title = original.caption
        chart_path = ch / f"custom_{chart_id}_{chart_type}.png"
        chart_file = chart_custom(categories, series, pie_series, chart_type, title, chart_path)
        target.figures[spec["figure"]] = Figure(caption=f"{title} ({chart_type.title()} view)", path=chart_file)

    dq = Table(title="Data cleaning log", columns=["Step", "Column", "Rows affected", "Action", "Reason"],
               rows=[[c.step, c.column, c.rows_affected, c.action, c.reason] for c in ds.changes],
               note=f"Source: {ds.source_file}; {ds.rows_in} rows read, {ds.rows_out} validated records retained.")
    register = Table(title="Insight register (every narrative figure traces to one of these)",
                columns=["Insight ID", "Finding", "Value", "Window", "Method"],
                     rows=[[i.id, i.title, f"{fmt_v(i.value, 3 if i.unit == 'p-value' else 1)} {i.unit}",
                    i.window.label, i.method] for i in I.values()])
    return TrendReport(
        title=f"CAPA Trend Analysis - {b.site_code.replace('-', ' ')}", organisation=ORGANISATION, site=b.site,
        period_label=f"{period_label} ({QUARTER_NAMES[q_num]} Quarter {year})", generated_on=date.today(),
        executive_summary=summary, sections=sections, data_quality=dq, insight_register=register,
        assumptions=ds.assumptions + b.data_notes + [
            "Statistical tests treat CAPA counts as Poisson events; quarters are assumed to be of equal length.",
        ])


def _group_as_category_table(t: Table) -> Table:
    """Transpose the quarter x group table into the category x quarter shape used by chart_stacked."""
    rows = [r for r in t.rows if r[0] != "Total"]
    groups = t.columns[1:4]
    return Table(title=t.title, columns=["Group", *[r[0] for r in rows], "Total"],
                 rows=[[g, *[r[1 + i] for r in rows], sum(r[1 + i] for r in rows)] for i, g in enumerate(groups)])


def fmt_v(v: float, decimals: int = 1) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:.{decimals}f}"


def fmt_p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


# ---------------------------------------------------------------- docx rendering
def _shade(cell, hex_fill: str) -> None:
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hex_fill)
    cell._tc.get_or_add_tcPr().append(shd)


def _table(doc: Document, t: Table) -> None:
    doc.add_paragraph(t.title).runs[0].bold = True
    tbl = doc.add_table(rows=1, cols=len(t.columns))
    tbl.style = "Table Grid"
    for cell, name in zip(tbl.rows[0].cells, t.columns):
        cell.text = name
        cell.paragraphs[0].runs[0].bold = True
        _shade(cell, "D9E2F3")
    for row in t.rows:
        cells = tbl.add_row().cells
        for cell, v in zip(cells, row):
            cell.text = str(v)
        if row and row[0] == "Total":
            for cell in cells:
                cell.paragraphs[0].runs[0].bold = True
    for r in tbl.rows:
        for cell in r.cells:
            for run in cell.paragraphs[0].runs:
                run.font.size = Pt(8)
    if t.note:
        p = doc.add_paragraph(t.note)
        p.runs[0].italic = True
        p.runs[0].font.size = Pt(8)
    doc.add_paragraph()


def _paragraph(doc: Document, p: Paragraph) -> None:
    para = doc.add_paragraph(style="List Bullet" if p.lead else None)
    if p.lead:
        para.add_run(f"{p.lead}: ").bold = True
    para.add_run(p.text)


def render_docx(r: TrendReport, path: Path) -> Path:
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10)
    for text, size, bold in ((r.title, 20, True), (r.organisation, 12, False), (r.site, 12, False),
                             (r.period_label, 12, True)):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        run.font.size, run.bold = Pt(size), bold
    p = doc.add_paragraph(f"Generated {r.generated_on}")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading("Executive Summary", level=1)
    for para in r.executive_summary:
        _paragraph(doc, para)

    for s in r.sections:
        doc.add_page_break()
        doc.add_heading(f"Section {s.number} : {s.title}", level=1)
        for para in s.paragraphs:
            _paragraph(doc, para)
        for t in s.tables:
            _table(doc, t)
        for f in s.figures:
            doc.add_picture(f.path, width=Inches(6.3))
            cap = doc.add_paragraph(f.caption)
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap.runs[0].italic = True

    doc.add_page_break()
    doc.add_heading("Appendix A : Assumptions and Method Notes", level=1)
    for a in r.assumptions:
        doc.add_paragraph(a, style="List Bullet")
    doc.add_heading("Appendix B : Data Quality", level=1)
    _table(doc, r.data_quality)
    doc.add_heading("Appendix C : Insight Register", level=1)
    _table(doc, r.insight_register)
    doc.save(path)
    return path
