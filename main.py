"""Run the CAPA trend pipeline: clean -> analyse -> report.

    python main.py                                   # all sites, latest quarter
    python main.py --site "Onesource Beta Lactam Division" --quarter 2025Q2   # one site only
"""
import argparse
import json
import logging
from pathlib import Path

from cleaning import clean
from insights import analyse
from report import build_report, render_docx


def run_pipeline(data, site=None, quarter=None, out="output", chart_overrides=None):
    """Run all three layers; returns (dataset, bundle, report, docx_path). Used by main() and web.py."""
    out = Path(out)
    dataset = clean(data)                                        # layer 1 -> CleanedDataset
    bundle = analyse(dataset, site, quarter)                     # layer 2 -> InsightBundle
    report = build_report(bundle, dataset, out / "charts", chart_overrides=chart_overrides)  # layer 3 -> TrendReport
    name = f"CAPA_Trend_Report_{bundle.site_code}_{bundle.quarters[-1].replace(' ', '')}"
    report_data = report.model_dump(mode="json")
    paragraphs = list(report_data["executive_summary"])
    paragraphs.extend(p for section in report_data["sections"] for p in section["paragraphs"])
    for paragraph in paragraphs:
        for source in paragraph["sources"]:
            source.pop("evidence", None)
    (out / f"{name}.json").write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    path = render_docx(report, out / f"{name}.docx")
    return dataset, bundle, report, path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="capa_test_2025_Q2.xlsx", help="CAPA export (.xlsx or .csv)")
    ap.add_argument("--site", default=None, help="SiteName to report on (default: all sites in the file)")
    ap.add_argument("--quarter", default=None, help="report quarter, e.g. 2025Q2 (default: latest complete)")
    ap.add_argument("--out", default="output", help="output folder")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    *_, path = run_pipeline(args.data, args.site, args.quarter, args.out)
    print(f"Report written to {path}")


if __name__ == "__main__":
    main()
