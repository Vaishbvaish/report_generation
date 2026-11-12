"""Minimal web front end for the CAPA trend pipeline.

    python web.py          ->  open http://127.0.0.1:5000

Flow: upload an export (/api/inspect) -> pick site and quarter -> /api/run ->
summary shown in the page, Word/JSON report downloadable. Each upload gets its
own folder under runs/.
"""
import json
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file, send_from_directory
from werkzeug.utils import secure_filename

from cleaning import clean
from insights import default_quarter, qlabel
from main import run_pipeline
from report import REPORT_CHART_SPECS

ROOT = Path(__file__).parent
RUNS = ROOT / "runs"
app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


def _run_dir(run_id: str) -> Path:
    d = RUNS / secure_filename(run_id)
    if not d.is_dir():
        abort(404, "unknown upload")
    return d


def _upload(run_dir: Path) -> Path:
    files = [f for f in run_dir.iterdir() if f.suffix.lower() in {".csv", ".xlsx", ".xls"}]
    if not files:
        abort(404, "upload missing")
    return files[0]


def _chart_metadata(report, run_id: str, overrides: dict[str, str]) -> list[dict]:
    charts = []
    for chart_id, spec in REPORT_CHART_SPECS.items():
        section = next(s for s in report.sections if s.number == spec["section"])
        figure = section.figures[spec["figure"]]
        image_path = Path(figure.path)
        charts.append({
            "id": chart_id,
            "section": section.number,
            "section_title": section.title,
            "caption": figure.caption,
            "type": overrides.get(chart_id, spec["default"]),
            "types": spec["types"],
            "image": f"/api/chart-image/{run_id}/{chart_id}?v={image_path.stat().st_mtime_ns}",
        })
    return charts


@app.errorhandler(Exception)
def _error(exc):
    code = getattr(exc, "code", 400)
    return jsonify(error=getattr(exc, "description", None) or str(exc)), code if isinstance(code, int) else 400


@app.get("/")
def index():
    return send_from_directory(ROOT / "web", "index.html")


@app.post("/api/inspect")
def inspect():
    f = request.files.get("file")
    if not f or Path(f.filename).suffix.lower() not in {".csv", ".xlsx", ".xls"}:
        abort(400, "Please choose a .csv or .xlsx export.")
    run_id = uuid.uuid4().hex[:10]
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True)
    path = run_dir / secure_filename(f.filename)
    f.save(path)

    df = clean(path).to_frame()
    df["oq"] = df["opened"].dt.to_period("Q")
    sites = df["site"].value_counts()
    latest_complete = default_quarter(df)   # completeness is judged on the whole extract

    def quarter_options(sub):
        """Quarters in which this selection has opened CAPAs, newest first, plus its default."""
        oq = sub["oq"].value_counts().sort_index(ascending=False)
        options = [{"value": str(q), "label": qlabel(q), "count": int(v)} for q, v in oq.items()]
        complete = [q for q in oq.index if q <= latest_complete]
        return {"options": options, "default": str(complete[0] if complete else oq.index[0])}

    # "" = all sites; one entry per site so the quarter list follows the chosen site
    quarters = {"": quarter_options(df), **{s: quarter_options(df[df["site"] == s]) for s in sites.index}}
    return jsonify(
        id=run_id, filename=f.filename, records=len(df),
        sites=[{"name": k, "count": int(v)} for k, v in sites.items()],
        quarters=quarters,
    )


@app.post("/api/run")
def run():
    body = request.get_json(force=True)
    run_dir = _run_dir(body.get("id", ""))
    _, bundle, report, docx = run_pipeline(_upload(run_dir), body.get("site") or None,
                                           body.get("quarter") or None, run_dir)
    I = bundle.insights

    def para(p):
        return {"lead": p.lead, "text": p.text}

    flow = next(t for s in report.sections for t in s.tables if t.title.startswith("Quarterly initiation"))
    cur, prev = flow.rows[-1], flow.rows[-2]   # [quarter, initiated, closed, ratio, open, median days, on-time %]
    num = lambda v: float(v) if str(v) not in ("nan", "") else None
    # better: which direction counts as an improvement (None = no judgement, e.g. volume of new CAPAs)
    kpis = [
        {"label": "CAPAs initiated", "value": num(cur[1]), "prev": num(prev[1]), "better": None, "unit": "",
         "note": f"{I['initiated_total'].facts['still_open']:.0f} still open"},
        {"label": "Closed (done)", "value": num(cur[2]), "prev": num(prev[2]), "better": "up", "unit": "",
         "note": f"plus {I['closed_total'].facts['cancelled']:.0f} cancelled"},
        {"label": "Closure ratio", "value": num(cur[3]), "prev": num(prev[3]), "better": "up", "unit": "",
         "note": f"{cur[4]} open at quarter end"},
        {"label": "Median days to close", "value": num(cur[5]), "prev": num(prev[5]), "better": "down", "unit": "",
         "note": "opening to closure"},
        {"label": "On-time closure", "value": num(cur[6]), "prev": num(prev[6]), "better": "up", "unit": "%",
         "note": f"{I['on_time'].facts['late']:.0f} closed late"},
        {"label": "Open over 90 days", "value": I["aging"].value, "prev": None, "better": None, "unit": "",
         "note": f"{I['aging'].facts['overdue']:.0f} past due date"},
    ]
    return jsonify(
        title=report.title, site=report.site, period=report.period_label, quarter=bundle.quarters[-1],
        prev_quarter=bundle.quarters[-2],
        kpis=kpis,
        findings=[para(p) for p in report.executive_summary],
        flow={"columns": flow.columns, "rows": flow.rows},
        sections=[{"number": s.number, "title": s.title, "paragraphs": [para(p) for p in s.paragraphs]}
                  for s in report.sections],
        charts=_chart_metadata(report, run_dir.name, {}),
        cleaning=[r for r in report.data_quality.rows if r[2]],
        cleaning_note=report.data_quality.note,
        assumptions=report.assumptions,
        files={"docx": docx.name, "json": docx.with_suffix(".json").name},
    )


@app.post("/api/chart/<run_id>")
def update_chart(run_id):
    run_dir = _run_dir(run_id)
    body = request.get_json(force=True)
    overrides = body.get("chart_overrides", {})
    if not isinstance(overrides, dict) or any(
        chart_id not in REPORT_CHART_SPECS or chart_type not in REPORT_CHART_SPECS[chart_id]["types"]
        for chart_id, chart_type in overrides.items()
    ):
        abort(400, "Unsupported chart selection.")

    _, _, report, docx = run_pipeline(
        _upload(run_dir), body.get("site") or None, body.get("quarter") or None,
        run_dir, chart_overrides=overrides,
    )
    return jsonify(
        charts=_chart_metadata(report, run_dir.name, overrides),
        files={"docx": docx.name, "json": docx.with_suffix(".json").name},
    )


@app.get("/api/chart-image/<run_id>/<chart_id>")
def chart_image(run_id, chart_id):
    if chart_id not in REPORT_CHART_SPECS:
        abort(404, "unknown chart")
    run_dir = _run_dir(run_id)
    reports = list(run_dir.glob("*.json"))
    if not reports:
        abort(404, "report missing")
    report_data = json.loads(max(reports, key=lambda p: p.stat().st_mtime_ns).read_text(encoding="utf-8"))
    spec = REPORT_CHART_SPECS[chart_id]
    section = next((s for s in report_data["sections"] if s["number"] == spec["section"]), None)
    if not section or len(section["figures"]) <= spec["figure"]:
        abort(404, "chart missing")
    image_path = Path(section["figures"][spec["figure"]]["path"]).resolve()
    if not image_path.is_relative_to(run_dir.resolve()) or not image_path.is_file():
        abort(404, "chart missing")
    return send_file(image_path, mimetype="image/png", conditional=True)


@app.get("/api/download/<run_id>/<name>")
def download(run_id, name):
    return send_from_directory(_run_dir(run_id), secure_filename(name), as_attachment=True)


if __name__ == "__main__":
    app.run(debug=False, port=5000)
