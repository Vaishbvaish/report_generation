"""Layer 2 - statistical insights.

Input : models.CleanedDataset, a site name, and optionally a report quarter.
Output: models.InsightBundle (insights + tables + control chart + aging list).

Conventions
- Analysis window: the report quarter plus the three before it (as in the
  reference report). Monthly control limits use the same 12 months.
- Shares and changes are stored in percent (16.7, not 0.167).
- Evidence grading for a stated change: p < 0.01 strong, p < 0.05 moderate,
  otherwise weak. Direct counts are 'strong'. Fewer than MIN_N events ->
  'insufficient', and the text says the data cannot support a conclusion.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2_contingency, fisher_exact, mannwhitneyu, poisson

from models import (AgingItem, CleanedDataset, ControlChart, Insight, InsightBundle,
                    MonthlyPoint, Table, TimeWindow)

MIN_N = 20          # minimum events before a change is tested
SPARSE = 5          # department/category counts below this are not trended
AGING_DAYS = 90
TOP_DEPTS = 6


# ---------------------------------------------------------------- helpers
def qlabel(p: pd.Period) -> str:
    return f"Q{p.quarter} - {p.year}"


def mlabel(p: pd.Period) -> str:
    return p.strftime("%b - %Y")


def fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:.1f}"


def fmt_p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def grade(p: float, n: int) -> str:
    if n < MIN_N:
        return "insufficient"
    return "strong" if p < 0.01 else "moderate" if p < 0.05 else "weak"


def significance(p: float) -> str:
    if p < 0.05:
        return f"This is statistically significant ({fmt_p(p)}), i.e. unlikely to be normal fluctuation."
    return f"This is not statistically significant ({fmt_p(p)}) and is consistent with normal fluctuation."


def poisson_p(k: int, mu: float) -> float:
    """Two-sided exact Poisson test of an observed count against an expected rate."""
    if mu <= 0:
        return 1.0
    return float(min(1.0, 2 * min(poisson.cdf(k, mu), poisson.sf(k - 1, mu))))


def chi2_p(table: pd.DataFrame) -> tuple[float, float, dict]:
    """Chi-square test of independence.

    Returns (p, share of cells with expected < 5, driver), where driver is the
    cell with the largest standardised residual - the main source of any shift.
    """
    table = table.loc[table.sum(axis=1) > 0, table.sum(axis=0) > 0]
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 1.0, 1.0, {}
    _, p, _, expected = chi2_contingency(table.values)
    resid = (table.values - expected) / np.sqrt(expected)
    i, j = np.unravel_index(np.abs(resid).argmax(), resid.shape)
    driver = {"row": table.index[i], "col": qlabel(table.columns[j]),
              "observed": int(table.values[i, j]), "expected": float(expected[i, j])}
    return float(p), float((expected < 5).mean()), driver


def driver_text(p: float, d: dict) -> str:
    if p >= 0.05 or not d:
        return ""
    return (f" The largest departure is {d['row']} in {d['col']}: {d['observed']} observed versus "
            f"{d['expected']:.1f} expected if the mix were constant.")


def fold_small(s: pd.Series, min_total: int) -> pd.Series:
    """Fold low-volume categories into 'Other' so chi-square expectations hold."""
    return s.where(s.map(s.value_counts()) >= min_total, "Other")


def counts_text(counts: pd.Series) -> str:
    return ", ".join(f"{k} ({fmt(v)})" for k, v in counts.items())


def source_group(src: str) -> str:
    if src == "Deviation":
        return "Deviation"
    return "OOS/OOT" if src in {"OOS", "OOT"} else "Standalone"


def crosstab(df: pd.DataFrame, row: str, col: str, cols: list) -> pd.DataFrame:
    t = pd.crosstab(df[row], df[col]).reindex(columns=cols, fill_value=0)
    return t.loc[t.sum(axis=1).sort_values(ascending=False).index]


def to_table(t: pd.DataFrame, title: str, first_col: str, col_labels: list[str],
             note: str | None = None) -> Table:
    rows = [[str(idx), *map(int, r), int(sum(r))] for idx, r in zip(t.index, t.values)]
    rows.append(["Total", *map(int, t.sum().values), int(t.values.sum())])
    return Table(title=title, columns=[first_col, *col_labels, "Total"], rows=rows, note=note)


# ---------------------------------------------------------------- analysis
def default_quarter(df: pd.DataFrame) -> pd.Period:
    """Latest quarter fully covered by the extract.

    ASSUMPTION: a quarter still in progress at extraction time would under-count,
    so if the latest opened quarter ends after the last date in the data, step back one.
    """
    last_date = max(df["opened"].max(), df["closed"].max())
    cur = df["opened"].max().to_period("Q")
    return cur - 1 if cur.end_time > last_date else cur


def analyse(ds: CleanedDataset, site: str | None = None, quarter: str | None = None) -> InsightBundle:
    """Analyse one site, or every site in the file when `site` is None (the default)."""
    df = ds.to_frame()
    if site:
        df = df[df["site"] == site]
        if df.empty:
            raise ValueError(f"no records for site {site!r}")
    df = df.copy()
    df["oq"] = df["opened"].dt.to_period("Q")
    df["om"] = df["opened"].dt.to_period("M")
    df["cq"] = df["closed"].dt.to_period("Q")
    df["cm"] = df["closed"].dt.to_period("M")
    df["group"] = df["initiated_through"].map(source_group)

    cur = pd.Period(quarter, "Q") if quarter else default_quarter(df)
    quarters = list(pd.period_range(cur - 3, cur, freq="Q"))
    prev = quarters[2]
    qlabels = [qlabel(q) for q in quarters]
    L, PL = qlabel(cur), qlabel(prev)
    months4 = list(pd.period_range(quarters[0].asfreq("M", "start"), cur.asfreq("M", "end"), freq="M"))
    months_cur = months4[-3:]

    def win(a: pd.Period, b: pd.Period, label: str) -> TimeWindow:
        return TimeWindow(label=label, start=a.start_time.date(), end=b.end_time.date())

    wq, wp = win(cur, cur, L), win(prev, prev, PL)
    w4 = win(quarters[0], cur, f"{qlabels[0]} to {L}")
    period_end = cur.end_time

    notes = []
    if quarters[0].start_time.date() < ds.complete_from:
        notes.append(f"The extract is only complete from {ds.complete_from}; quarters before that "
                     "under-count initiations and should not be compared directly.")
    if ds.extract_end > period_end.date():
        notes.append(f"The extract runs to {ds.extract_end}; record states (e.g. cancelled, open) "
                     "reflect that date, while all period counts use the event dates.")

    ins: dict[str, Insight] = {}

    def add(**kw) -> None:
        ins[kw["id"]] = Insight(**kw)

    done = df[df["state"] == "Closed - Done"]
    cancelled = df[df["state"] == "Closed - Cancelled"]
    opened_q = df.groupby("oq").size().reindex(quarters, fill_value=0)
    closed_q = done.groupby("cq").size().reindex(quarters, fill_value=0)
    cancel_q = cancelled.groupby("cq").size().reindex(quarters, fill_value=0)
    cohort = df[df["oq"] == cur]
    n = len(cohort)
    if n == 0:
        raise ValueError(f"No CAPAs were opened in {qlabel(cur)} for this selection. Choose a different quarter.")

    # ---- 5.1 initiation -------------------------------------------------
    n_cancel = int((cohort["state"] == "Closed - Cancelled").sum())
    n_open = int((~cohort["state"].str.startswith("Closed")).sum())
    add(id="initiated_total", title="CAPAs initiated", value=n, unit="CAPAs", window=wq,
        facts={"cancelled": n_cancel, "still_open": n_open, "done": n - n_cancel - n_open},
        interpretation=f"{n} CAPAs were initiated in {L}. By the data extract date {n - n_cancel - n_open} "
                       f"were closed as done, {n_open} remain open and {n_cancel} were cancelled.",
        evidence="strong", method="Count of records with Date Opened in the quarter (all states).")

    p_prev = int(opened_q[prev])
    diff = n - p_prev
    pct = diff / p_prev * 100 if p_prev else 0.0
    p = binomtest(n, n + p_prev, 0.5).pvalue if n + p_prev else 1.0
    direction = "rose" if diff > 0 else "fell" if diff < 0 else "were unchanged"
    add(id="initiated_qoq", title="Quarter-on-quarter change in initiations", value=pct, unit="%", window=w4,
        facts={"current": n, "previous": p_prev, "abs_change": diff, "p_value": p},
        interpretation=(f"Initiations {direction} by {abs(diff)} ({abs(pct):.1f}%) from {p_prev} in {PL} to {n} in {L}. "
                        if diff else f"Initiations were unchanged at {n}. ") + significance(p),
        evidence=grade(p, n + p_prev),
        method="Exact conditional binomial test of two Poisson counts (equal-length quarters assumed).")

    prior = opened_q.iloc[:3]
    mean_prior = float(prior.mean())
    vs_mean = (n - mean_prior) / mean_prior * 100 if mean_prior else 0.0
    rank = int((opened_q > n).sum()) + 1
    p = poisson_p(n, mean_prior)
    hi_q, lo_q = opened_q.idxmax(), opened_q.idxmin()
    add(id="initiated_vs_avg", title="Initiations versus prior-quarter average", value=vs_mean, unit="%", window=w4,
        facts={"current": n, "mean_prior": mean_prior, "rank": rank, "n_quarters": 4,
               "max": int(opened_q.max()), "min": int(opened_q.min()), "p_value": p},
        interpretation=(f"{L} ({n}) is {abs(vs_mean):.1f}% {'above' if vs_mean >= 0 else 'below'} the average of the "
                        f"previous three quarters ({mean_prior:.1f}) and ranks {rank} of 4 quarters by volume "
                        f"(highest {qlabel(hi_q)} with {int(opened_q.max())}, lowest {qlabel(lo_q)} with "
                        f"{int(opened_q.min())}). " + significance(p)),
        evidence=grade(p, n), method="Two-sided exact Poisson test of the quarter count against the prior-3 mean.")

    slope = float(np.polyfit(range(4), opened_q.values.astype(float), 1)[0])
    add(id="initiated_trend", title="Four-quarter initiation trend", value=slope, unit="CAPAs per quarter", window=w4,
        facts={"n_quarters": 4},
        interpretation=(f"A straight-line fit across the 4 quarters gives a trend of {slope:+.1f} CAPAs per quarter. "
                        "Four points are too few to establish a trend statistically, so this is indicative only."),
        evidence="insufficient", method="Least-squares slope over quarterly counts; no significance claimed.")

    src = cohort["initiated_through"].value_counts()
    add(id="source_mix", title="Initiation sources", value=n, unit="CAPAs", window=wq,
        facts={**{k: int(v) for k, v in src.items()}, "top_share": src.iloc[0] / n * 100},
        interpretation=(f"{src.index[0]} was the largest source ({int(src.iloc[0])}, {src.iloc[0] / n * 100:.1f}% of {n}). "
                        f"Full breakdown: {counts_text(src)}."),
        evidence="strong", method="Count by InitiatedThrough for CAPAs opened in the quarter.")

    sites = cohort["site"].value_counts()
    if len(sites) > 1:
        add(id="site_mix", title="Initiations by site", value=len(sites), unit="sites", window=wq,
            facts={**{k: int(v) for k, v in sites.items()}, "total": n, "top_share": sites.iloc[0] / n * 100},
            interpretation=(f"CAPAs were initiated at {len(sites)} sites. {sites.index[0]} contributed the most "
                            f"({int(sites.iloc[0])}, {sites.iloc[0] / n * 100:.1f}% of {n}). "
                            f"By site: {counts_text(sites)}."),
            evidence="strong", method="Count by SiteName for CAPAs opened in the quarter.")

    # ---- 5.2 monthly / closure -------------------------------------------
    opened_m = df.groupby("om").size().reindex(months4, fill_value=0)
    closed_m = done.groupby("cm").size().reindex(months4, fill_value=0)
    cancel_m = cancelled.groupby("cm").size().reindex(months4, fill_value=0)
    rolling = opened_m.rolling(3).mean()
    m_cur = opened_m[months_cur]
    add(id="monthly_initiated", title="Monthly initiations", value=n, unit="CAPAs", window=wq,
        facts={mlabel(m): int(v) for m, v in m_cur.items()} | {"peak": int(m_cur.max())},
        interpretation=(f"The peak month was {mlabel(m_cur.idxmax())} with {int(m_cur.max())} CAPAs; "
                        f"monthly initiations were {counts_text(m_cur.rename(index=mlabel))}."),
        evidence="strong", method="Count by month of Date Opened.")

    c_cur = closed_m[months_cur]
    n_closed, n_cc = int(closed_q[cur]), int(cancel_q[cur])
    add(id="closed_total", title="CAPAs closed", value=n_closed, unit="CAPAs", window=wq,
        facts={mlabel(m): int(v) for m, v in c_cur.items()} | {"cancelled": n_cc, "all_closures": n_closed + n_cc},
        interpretation=(f"{n_closed} CAPAs were closed as done in {L} ({counts_text(c_cur.rename(index=mlabel))}). "
                        f"A further {n_cc} were closed as cancelled and are excluded from closure trending, "
                        f"giving {n_closed + n_cc} closures in total."),
        evidence="strong", method="Records in state 'Closed - Done' by month of Date Closed.")

    # XmR (individuals) chart: robust to the extra-Poisson variation typical of CAPA counts.
    center = float(opened_m.mean())
    mr_bar = float(opened_m.diff().abs().mean())
    sigma = mr_bar / 1.128
    ucl, lcl = center + 3 * sigma, max(0.0, center - 3 * sigma)
    flagged = [m for m, v in opened_m.items() if v > ucl or v < lcl]
    side = np.sign(opened_m.values - center)
    run, longest = 0, 0
    for a, b in zip(side, np.r_[0, side[:-1]]):
        run = run + 1 if a == b and a != 0 else 1
        longest = max(longest, run)
    flag_txt = (f"{len(flagged)} month(s) fell outside the limits: "
                f"{counts_text(opened_m[flagged].rename(index=mlabel))}. These are special-cause signals worth investigating."
                if flagged else "No month fell outside the limits, so month-to-month variation is consistent with a stable process.")
    run_txt = (f" A run of {longest} consecutive months on one side of the average indicates a sustained shift."
               if longest >= 8 else "")
    r_now, r_prev = float(rolling.iloc[-1]), float(rolling.iloc[-4])
    add(id="control", title="Monthly initiation control chart", value=len(flagged), unit="months flagged",
        window=win(quarters[0], cur, f"{mlabel(months4[0])} to {mlabel(months4[-1])}"),
        facts={"center": center, "ucl": ucl, "lcl": lcl, "n_months": len(months4), "window_months": 3,
               "rolling_now": r_now, "rolling_prev": r_prev, "longest_run": longest,
               **{mlabel(m): int(opened_m[m]) for m in flagged}},
        interpretation=(f"Over the {len(months4)} months to {mlabel(months4[-1])}, initiations averaged {center:.1f} per month "
                        f"(control limits {lcl:.1f} to {ucl:.1f}). {flag_txt}{run_txt} The 3-month rolling average "
                        f"was {r_now:.1f} at quarter end versus {r_prev:.1f} at the end of {PL}."),
        evidence="moderate" if len(months4) >= 12 else "insufficient",
        method="Individuals (XmR) chart: centre = mean, limits = mean +/- 2.66 x average moving range; "
               "run rule: 8 consecutive points on one side of the centre.")

    ratio = closed_q / opened_q.replace(0, np.nan)
    ends = [q.end_time for q in quarters]
    backlog = pd.Series([int(((df["opened"] <= e) & (df["closed"].isna() | (df["closed"] > e))).sum()) for e in ends],
                        index=quarters)
    b_now, b_prev = int(backlog[cur]), int(backlog[prev])
    r = float(ratio[cur])
    add(id="closure_ratio", title="Closure ratio and open backlog", value=r, unit="closed per initiated", window=w4,
        facts={"closed": n_closed, "initiated": n, "backlog_now": b_now, "backlog_prev": b_prev,
               "backlog_change": b_now - b_prev, "backlog_start": int(backlog.iloc[0])},
        interpretation=(f"{n_closed} closures against {n} initiations gives a closure ratio of {r:.2f}. "
                        + ("Closures did not keep pace with new CAPAs" if r < 1 else "Closures outpaced new CAPAs")
                        + f", and open CAPAs at quarter end moved from {b_prev} to {b_now} ({b_now - b_prev:+d}); "
                        f"at the end of {qlabels[0]} the open count was {int(backlog.iloc[0])}."),
        evidence="strong", method="Closed-Done / initiated in the quarter; open = opened by and not closed by quarter end.")

    days = (done["closed"] - done["opened"]).dt.days
    d_cur, d_prior = days[done["cq"] == cur], days[done["cq"].isin(quarters[:3])]
    med_cur = float(d_cur.median()) if len(d_cur) else 0.0
    med_prior = float(d_prior.median()) if len(d_prior) else 0.0
    p = float(mannwhitneyu(d_cur, d_prior).pvalue) if len(d_cur) and len(d_prior) else 1.0
    add(id="cycle_time", title="Time to close", value=med_cur, unit="days (median)", window=w4,
        facts={"median_prior": med_prior, "p_value": p, "n_current": len(d_cur)},
        interpretation=(f"The {len(d_cur)} CAPAs closed in {L} took a median of {fmt(med_cur)} days from initiation "
                        f"to closure, versus {fmt(med_prior)} days for closures in the previous three quarters. "
                        + significance(p)),
        evidence=grade(p, len(d_cur)), method="Median days Date Opened -> Date Closed; Mann-Whitney U test.")

    on_time = done["closed"].dt.normalize() <= done["due_date"]
    ot_c, ot_p = on_time[done["cq"] == cur], on_time[done["cq"].isin(quarters[:3])]
    rate_c = ot_c.mean() * 100 if len(ot_c) else 0.0
    rate_p = ot_p.mean() * 100 if len(ot_p) else 0.0
    p = float(fisher_exact([[ot_c.sum(), (~ot_c).sum()], [ot_p.sum(), (~ot_p).sum()]])[1])
    add(id="on_time", title="On-time closure rate", value=rate_c, unit="%", window=w4,
        facts={"rate_prior": rate_p, "late": int((~ot_c).sum()), "p_value": p},
        interpretation=(f"{rate_c:.1f}% of CAPAs closed in {L} were closed on or before their due date "
                        f"({int((~ot_c).sum())} late), versus {rate_p:.1f}% in the previous three quarters. "
                        + significance(p)),
        evidence=grade(p, len(ot_c)), method="Share of Closed-Done with closure date <= current DueDate; Fisher exact test.")

    open_at_end = df[(df["opened"] <= period_end) & (df["closed"].isna() | (df["closed"] > period_end))]
    age = (period_end - open_at_end["opened"]).dt.days
    aged = open_at_end[age > AGING_DAYS].assign(age=age[age > AGING_DAYS]).sort_values("age", ascending=False)
    aging = [AgingItem(record_id=int(r.record_id), title=r.title, department=r.department, opened=r.opened.date(),
                       due_date=r.due_date.date(), age_days=int(r.age), overdue=bool(r.due_date < period_end),
                       extensions=int(r.extension_count), state=r.state) for r in aged.itertuples()]
    n_over = sum(a.overdue for a in aging)
    add(id="aging", title="CAPAs open more than 90 days", value=len(aging), unit="CAPAs", window=wq,
        facts={"threshold": AGING_DAYS, "overdue": n_over, "within_due": len(aging) - n_over,
               "extended": sum(a.extensions > 0 for a in aging), "open_total": len(open_at_end)},
        interpretation=(f"At quarter end {len(open_at_end)} CAPAs were open, of which {len(aging)} had been open for more "
                        f"than {AGING_DAYS} days: {len(aging) - n_over} were still within their due date and "
                        f"{n_over} were past it ({sum(a.extensions > 0 for a in aging)} carry a recorded extension)."),
        evidence="strong", method="Open at quarter end and age > 90 days; overdue = DueDate before quarter end.")

    # ---- 5.3 departments ------------------------------------------------
    dept_o = crosstab(df[df["oq"].isin(quarters)], "department", "oq", quarters)
    dept_c = crosstab(done[done["cq"].isin(quarters)], "department", "cq", quarters)
    d_cur = dept_o[cur][dept_o[cur] > 0].sort_values(ascending=False)
    top3 = d_cur.head(3)
    add(id="dept_overview", title="Departments initiating CAPAs", value=len(d_cur), unit="departments", window=wq,
        facts={**{k: int(v) for k, v in d_cur.items()}, "total": n, "top3_share": top3.sum() / n * 100,
               "n_small": int((d_cur < SPARSE).sum()), "sparse": SPARSE},
        interpretation=(f"{len(d_cur)} departments initiated CAPAs in {L}. {counts_text(top3)} led, together accounting "
                        f"for {top3.sum() / n * 100:.1f}% of the {n}. {int((d_cur < SPARSE).sum())} departments had fewer "
                        f"than {SPARSE} CAPAs each; counts that small are not trended individually."),
        evidence="strong", method="Count by ResponsibleDepartment.")

    for i, dept in enumerate(d_cur.index[:TOP_DEPTS]):
        s = cohort.loc[cohort["department"] == dept, "initiated_through"].value_counts()
        avg = float(dept_o.loc[dept, quarters[:3]].mean())
        add(id=f"dept_{i}", title=f"{dept} breakdown", value=int(s.sum()), unit="CAPAs", window=wq,
            facts={**{k: int(v) for k, v in s.items()}, "avg_prior": avg},
            interpretation=(f"{dept} initiated {int(s.sum())} CAPAs (previous three-quarter average {avg:.1f}): "
                            f"{counts_text(s)}."),
            evidence="strong", method="Count by InitiatedThrough within department.")

    movers = []
    for dept, row in dept_o.iterrows():
        c, avg = int(row[cur]), float(row[quarters[:3]].mean())
        if max(c, avg) >= SPARSE:
            movers.append((dept, c, avg, c - avg, poisson_p(c, avg) if avg > 0 else 0.0))
    movers.sort(key=lambda m: m[3])
    mover_rows = [[d, c, round(a, 1), f"{x:+.1f}", "< 0.001" if p < 0.001 else f"{p:.3f}"]
                  for d, c, a, x, p in sorted(movers, key=lambda m: -abs(m[3]))]
    extremes = []
    if movers and movers[-1][3] > 0:
        extremes.append(("dept_up", movers[-1]))
    if movers and movers[0][3] < 0:
        extremes.append(("dept_down", movers[0]))
    for key, (dept, c, avg, x, p) in extremes:
        add(id=key, title=f"Largest {'increase' if key == 'dept_up' else 'decrease'}: {dept}", value=x, unit="CAPAs",
            window=w4, facts={"current": c, "avg_prior": avg, "p_value": p},
            interpretation=(f"{dept} moved from an average of {avg:.1f} CAPAs per quarter to {c} in {L} ({x:+.1f}). "
                            + significance(p)
                            + " Several departments are compared at once, so a single borderline result should be confirmed next quarter."),
            evidence=grade(p, c + round(avg * 3)),
            method="Two-sided exact Poisson test of the quarter count against the department's prior-3 mean; "
                   f"only departments with >= {SPARSE} CAPAs in the quarter or on average are tested.")

    c_cur = dept_c[cur][dept_c[cur] > 0].sort_values(ascending=False)
    tot_o, tot_c = int(opened_q.sum()), int(closed_q.sum())
    add(id="dept_closed", title="Department closures", value=n_closed, unit="CAPAs", window=w4,
        facts={**{k: int(v) for k, v in c_cur.head(3).items()}, "total_opened_4q": tot_o, "total_closed_4q": tot_c,
               "net_4q": tot_o - tot_c, "cancelled_4q": int(cancel_q.sum())},
        interpretation=(f"Of the {n_closed} closures in {L}, the most came from {counts_text(c_cur.head(3))}. "
                        f"Across the four quarters {tot_o} CAPAs were initiated and {tot_c} closed as done "
                        f"({int(cancel_q.sum())} cancelled), a net addition of {tot_o - tot_c} to the workload."),
        evidence="strong", method="Closed-Done by department and closure quarter.")

    # ---- 5.4 source groups ----------------------------------------------
    win4 = df[df["oq"].isin(quarters)]
    grp = pd.crosstab(win4["oq"], win4["group"]).reindex(index=quarters, columns=["Deviation", "OOS/OOT", "Standalone"],
                                                        fill_value=0)
    dev_c, dev_p = int(grp.loc[cur, "Deviation"]), int(grp.loc[quarters[:3], "Deviation"].sum())
    n_prior = max(int(opened_q.iloc[:3].sum()), 1)   # guard against an empty comparison window
    share_c, share_p = dev_c / n * 100, dev_p / n_prior * 100
    p = float(fisher_exact([[dev_c, n - dev_c], [dev_p, n_prior - dev_p]])[1])
    add(id="deviation_share", title="Deviation-driven share", value=share_c, unit="%", window=w4,
        facts={"deviation": dev_c, "standalone": int(grp.loc[cur, "Standalone"]), "oos_oot": int(grp.loc[cur, "OOS/OOT"]),
               "non_deviation": n - dev_c, "share_prior": share_p, "total": n, "p_value": p},
        interpretation=(f"{dev_c} of {n} CAPAs ({share_c:.1f}%) came from deviation investigations and "
                        f"{n - dev_c} were standalone or OOS/OOT-driven. "
                        f"The deviation share compares with {share_p:.1f}% in the previous three quarters. " + significance(p)),
        evidence=grade(p, n), method="Fisher exact test, current quarter vs pooled prior three quarters.")

    test = pd.crosstab(fold_small(win4["initiated_through"], MIN_N), win4["oq"])
    p, low, drv = chi2_p(test)
    add(id="source_shift", title="Stability of source mix", value=p, unit="p-value", window=w4,
        facts={"low_expected_pct": low * 100, **{k: v for k, v in drv.items() if k in ("observed", "expected")}},
        interpretation=("Across the four quarters the mix of initiation sources "
                        + ("changed significantly" if p < 0.05 else "showed no statistically significant change")
                        + f" ({fmt_p(p)}, chi-square)." + driver_text(p, drv)
                        + (f" {low * 100:.1f}% of cells have low expected counts, so treat this result with caution."
                           if low > 0.2 else "")),
        evidence="weak" if low > 0.2 else grade(p, len(win4)),
        method="Chi-square test of independence, source x quarter; sources under 20 CAPAs folded into 'Other'.")

    oos = grp["OOS/OOT"]
    add(id="oos_oot", title="OOS/OOT-triggered CAPAs", value=int(oos[cur]), unit="CAPAs", window=w4,
        facts={**{qlabel(q): int(v) for q, v in oos.items()}, "total_4q": int(oos.sum()), "min_n": MIN_N},
        interpretation=(f"{int(oos[cur])} CAPAs in {L} were triggered by OOS/OOT results "
                        f"({int(oos.sum())} across the four quarters). "
                        + ("This is too few to support a trend conclusion; each case should be reviewed individually."
                           if oos.sum() < MIN_N else "")),
        evidence="insufficient" if oos.sum() < MIN_N else "strong", method="Count of OOS + OOT sources.")

    # ---- 5.5 classification ---------------------------------------------
    cls = crosstab(win4, "classification", "oq", quarters)
    cls_cur = cls[cur][cls[cur] > 0].sort_values(ascending=False)
    p, low, drv = chi2_p(pd.crosstab(fold_small(win4["classification"], MIN_N), win4["oq"]))
    top_cls = cls_cur.index[0]
    top_share_p = cls.loc[top_cls, quarters[:3]].sum() / n_prior * 100
    add(id="classification", title="CAPA classification", value=cls_cur.iloc[0] / n * 100, unit="%", window=w4,
        facts={**{k: int(v) for k, v in cls_cur.items()}, "share_prior": top_share_p, "p_value": p,
               **{k: v for k, v in drv.items() if k in ("observed", "expected")}},
        interpretation=(f"{top_cls} was the most frequent classification ({int(cls_cur.iloc[0])}, "
                        f"{cls_cur.iloc[0] / n * 100:.1f}% versus {top_share_p:.1f}% in the previous three quarters). "
                        f"Breakdown: {counts_text(cls_cur)}. Across the four quarters the classification mix "
                        + ("changed significantly" if p < 0.05 else "showed no statistically significant change")
                        + f" ({fmt_p(p)})." + driver_text(p, drv)
                        + (" 'Not recorded' means the classification field is still blank in the export (typically "
                           "CAPAs not yet approved); treat this as a data-completeness gap, not a real shift."
                           if "Not recorded" in cls_cur.index else "")),
        evidence="weak" if low > 0.2 else grade(p, len(win4)),
        method="Chi-square test, classification x quarter; classes under 20 CAPAs folded into 'Other'.")

    # ---- 5.10 CAPA type -------------------------------------------------
    typ = pd.crosstab(win4["capa_type"], win4["oq"]).reindex(index=["Corrective", "Preventive"], columns=quarters,
                                                              fill_value=0)
    cor_c, cor_p = int(typ.loc["Corrective", cur]), int(typ.loc["Corrective", quarters[:3]].sum())
    p = float(fisher_exact([[cor_c, n - cor_c], [cor_p, n_prior - cor_p]])[1])
    add(id="capa_type", title="Corrective vs preventive", value=cor_c / n * 100, unit="%", window=w4,
        facts={"corrective": cor_c, "preventive": n - cor_c, "share_prior": cor_p / n_prior * 100, "p_value": p},
        interpretation=(f"{cor_c} CAPAs were corrective and {n - cor_c} preventive; the corrective share was "
                        f"{cor_c / n * 100:.1f}% versus {cor_p / n_prior * 100:.1f}% in the previous three quarters. "
                        + significance(p) + " Where the export has no CAPA type, it is derived from the initiation source (see assumptions)."),
        evidence=grade(p, n), method="Fisher exact test on derived CAPA type, current vs prior three quarters.")

    # ---- tables -----------------------------------------------------------
    T = {}
    dups = sum(c.rows_affected for c in ds.changes if c.step.startswith("Remove duplicate"))
    DEDUP_NOTE = (f"Counts exclude {dups} duplicated record(s) found in the export (see Appendix B), so they can be "
                  f"{dups} lower than a raw count of the file.") if dups else None
    T["quarterly_initiated"] = Table(title="CAPAs initiated per quarter", columns=["Quarter", "CAPAs initiated", "Change vs previous"],
        rows=[[qlabel(q), int(v), "-" if i == 0 else f"{int(v - opened_q.iloc[i - 1]):+d}"]
              for i, (q, v) in enumerate(opened_q.items())] + [["Total", tot_o, ""]],
        note=DEDUP_NOTE)
    T["monthly"] = Table(title=f"Month-wise CAPAs initiated and closed, {L}",
        columns=["Month", "Initiated", "Closed (done)", "Closed (cancelled)"],
        rows=[[mlabel(m), int(opened_m[m]), int(closed_m[m]), int(cancel_m[m])] for m in months_cur]
             + [["Total", n, n_closed, n_cc]],
        note="Cancelled closures are excluded from closure trending, as in the reference report. " + (DEDUP_NOTE or ""))
    T["quarterly_flow"] = Table(title="Quarterly initiation, closure and backlog",
        columns=["Quarter", "Initiated", "Closed (done)", "Closure ratio", "Open at quarter end",
                 "Median days to close", "On-time closure %"],
        rows=[[qlabel(q), int(opened_q[q]), int(closed_q[q]), f"{ratio[q]:.2f}", int(backlog[q]),
               fmt(days[done["cq"] == q].median()), f"{on_time[done['cq'] == q].mean() * 100:.1f}"] for q in quarters])
    T["dept_opened"] = to_table(dept_o, "Department-wise CAPAs initiated", "Responsible department", qlabels)
    T["dept_closed"] = to_table(dept_c, "Department-wise CAPAs closed (done)", "Responsible department", qlabels)
    T["dept_movers"] = Table(title=f"Department change: {L} vs average of previous three quarters",
        columns=["Department", L, "Prior 3-quarter avg", "Change", "p-value"], rows=mover_rows,
        note=f"Only departments with at least {SPARSE} CAPAs in the quarter or on average are tested.")
    T["source_group"] = Table(title="CAPA source group by quarter",
        columns=["Quarter", "Deviation", "OOS/OOT", "Standalone", "Total", "Deviation share %"],
        rows=[[qlabel(q), *map(int, grp.loc[q]), int(grp.loc[q].sum()), f"{grp.loc[q, 'Deviation'] / grp.loc[q].sum() * 100:.1f}"]
              for q in quarters] + [["Total", *map(int, grp.sum()), int(grp.values.sum()), ""]],
        note="Standalone = Complaint, Customer Inspections, Improvement Plans, Internal Audit, PPT/PQR, "
             "Quality Risk Management, Quality System Review, Regulatory Inspections, Validation/Qualification.")
    T["source_detail"] = to_table(crosstab(win4, "initiated_through", "oq", quarters), "CAPAs by initiation source",
                                  "Initiated through", qlabels)
    T["classification"] = to_table(cls, "CAPAs by classification", "CAPA classification", qlabels)
    T["capa_type"] = to_table(typ, "CAPAs by type", "CAPA type", qlabels)
    T["aging"] = Table(title=f"CAPAs open more than {AGING_DAYS} days at end of {L}",
        columns=["Record ID", "Title", "Department", "Opened", "Due date", "Age (days)", "Status at quarter end"],
        rows=[[a.record_id, a.title[:90], a.department, str(a.opened), str(a.due_date), a.age_days,
               ("Overdue" if a.overdue else "Within due date") + (f", {a.extensions} extension(s)" if a.extensions else "")]
              for a in aging])

    if df["site"].nunique() > 1:
        T["site_initiated"] = to_table(crosstab(df[df["oq"].isin(quarters)], "site", "oq", quarters),
                                       "Site-wise CAPAs initiated", "Site", qlabels)
    code = df["capa_number"].str.extract(r"^([A-Z]+)-")[0].mode()
    site_code = code.iloc[0] if site and len(code) else (site or "All-Sites")
    control = ControlChart(center=center, ucl=ucl, lcl=lcl, flagged_months=[mlabel(m) for m in flagged],
                           points=[MonthlyPoint(month=mlabel(m), opened=int(opened_m[m]), closed=int(closed_m[m]),
                                                opened_rolling3=None if math.isnan(rolling[m]) else float(rolling[m]))
                                   for m in months4])
    return InsightBundle(site=site or f"All sites ({df['site'].nunique()})", site_code=site_code, period=wq,
                         quarters=qlabels, insights=ins, tables=T, control=control, aging=aging, data_notes=notes)
