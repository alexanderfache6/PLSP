#!/usr/bin/env python3
"""
qacct_dashboard.py
Interactive dashboard from SGE qacct CSV:
  1. Horizontal stacked bar (Gantt-style): queue wait (qsub->start, yellow)
     + run time (start->end, green), back to back, x-axis = duration,
     with a dot at the end of each bar colored by exit_status
  2. Bar chart: ru_* fields + cpu/mem/io/iow/maxvmem/arid vs jobnumber (dropdown to switch field)
  3. Pie chart: distribution of categorical fields (dropdown to switch field)

Usage:

    module load miniconda
    conda activate LCSC

    python qacct_dashboard.py qacct_fache_7days_20260702.csv
    python qacct_dashboard.py qacct_fache_7days_20260702.csv --out report.html
"""

import argparse
import os
import sys
import re
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

PIE_FIELDS = ["jobname", "owner", "group", "project", "department", "account", "qname", "hostname"]
EXTRA_NUMERIC_FIELDS = ["cpu", "mem", "io", "iow", "maxvmem", "arid"]

EXIT_STATUS_LABELS = {
    0: "0 = success",
    1: "1 = general error",
    2: "2 = misuse of shell command",
    126: "126 = command not executable",
    127: "127 = command not found",
    134: "134 = SIGABRT (aborted)",
    137: "137 = SIGKILL (killed, often OOM)",
    139: "139 = SIGSEGV (segfault)",
    143: "143 = SIGTERM (terminated)",
}

PALETTE = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B","#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC"]

UNIT_MULTIPLIERS = {"K": 1 / (1024 * 1024), "M": 1 / 1024, "G": 1, "T": 1024}


def parse_args():
    p = argparse.ArgumentParser(description="Interactive qacct dashboard")
    p.add_argument("csv_file", help="Path to CSV produced by qacct_to_csv.sh")
    p.add_argument("--out", default=None, help="Output HTML file (default: <csv_file>_dashboard.html)")
    return p.parse_args()


def to_numeric(series):
    return pd.to_numeric(series, errors="coerce").fillna(0)


def to_numeric_with_units(series):
    # handles plain numbers (cpu, mem, io, iow) and values with a K/M/G/T
    # suffix (maxvmem, e.g. "98.578G"); non-numeric junk (e.g. "undefined"
    # in arid) coerces to 0. Unit values are normalized to G.
    def parse_one(val):
        s = str(val).strip()
        m = re.match(r"^([0-9.]+)\s*([KMGT])?$", s, re.IGNORECASE)
        if not m:
            return 0.0
        num = float(m.group(1))
        unit = m.group(2)
        if unit:
            num *= UNIT_MULTIPLIERS.get(unit.upper(), 1)
        return num
    return series.apply(parse_one)


def parse_qacct_time(series):
    # qacct time format: "Thu Jul  2 13:17:07 2026"
    return pd.to_datetime(series, errors="coerce", format="%a %b %d %H:%M:%S %Y")


def exit_status_label(code):
    try:
        code = int(code)
    except (ValueError, TypeError):
        return f"{code} = unrecognized"
    return EXIT_STATUS_LABELS.get(code, f"{code} = other/unrecognized")


def fmt_duration(hours):
    # human readable duration for hover text, e.g. "1h 03m 12s"
    total_seconds = int(round(hours * 3600))
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m}m {s}s"


def main():
    args = parse_args()

    try:
        df = pd.read_csv(args.csv_file)
    except FileNotFoundError:
        sys.exit(f"Error: file not found: {args.csv_file}")

    # derive output filename from the input csv name if --out wasn't given
    if args.out is None:
        stem = os.path.splitext(os.path.basename(args.csv_file))[0]
        args.out = f"{stem}_dashboard.html"

    if "jobnumber" not in df.columns:
        sys.exit("Error: CSV has no 'jobnumber' column")

    df["jobnumber"] = pd.to_numeric(df["jobnumber"], errors="coerce")
    df = df.dropna(subset=["jobnumber"]).sort_values("jobnumber").reset_index(drop=True)
    x_labels = df["jobnumber"].astype(int).astype(str).tolist()

    # ---------- Panel 1: exit_status (used as dots on the Gantt bars) ----------
    have_exit_status = "exit_status" in df.columns
    if have_exit_status:
        df["exit_status"] = pd.to_numeric(df["exit_status"], errors="coerce")
        unique_statuses = sorted(df["exit_status"].dropna().unique().tolist())
    else:
        unique_statuses = []

    # ---------- Panel 2: ru_* fields + cpu/mem/io/iow/maxvmem/arid ----------
    ru_fields = [c for c in df.columns if c.startswith("ru_")]
    if not ru_fields:
        sys.exit("Error: no ru_* columns found in CSV")
    for f in ru_fields:
        df[f] = to_numeric(df[f])

    extra_fields = [f for f in EXTRA_NUMERIC_FIELDS if f in df.columns]
    for f in extra_fields:
        df[f] = to_numeric_with_units(df[f])

    bar_fields = ru_fields + extra_fields
    default_ru = "ru_wallclock" if "ru_wallclock" in bar_fields else bar_fields[0]

    # ---------- Panel 3: categorical fields ----------
    pie_fields = [f for f in PIE_FIELDS if f in df.columns]
    if not pie_fields:
        sys.exit("Error: none of the expected categorical fields found in CSV")
    pie_data = {f: df[f].fillna("(empty)").astype(str).value_counts() for f in pie_fields}
    default_pie = pie_fields[0]

    # ---------- Panel 1: queue wait (qsub->start) + run time (start->end) ----------
    have_times = all(c in df.columns for c in ("qsub_time", "start_time", "end_time"))
    if not have_times:
        sys.exit("Error: CSV must have qsub_time, start_time, end_time columns")

    qsub_t = parse_qacct_time(df["qsub_time"])
    start_t = parse_qacct_time(df["start_time"])
    end_t = parse_qacct_time(df["end_time"])

    wait_hours = (start_t - qsub_t).dt.total_seconds() / 3600.0
    run_hours = (end_t - start_t).dt.total_seconds() / 3600.0
    wait_hours = wait_hours.fillna(0).clip(lower=0)
    run_hours = run_hours.fillna(0).clip(lower=0)
    total_hours = wait_hours + run_hours

    wait_hover = [fmt_duration(h) for h in wait_hours]
    run_hover = [fmt_duration(h) for h in run_hours]

    y_category_order = list(reversed(x_labels))

    # ---------- Build figure: 3 rows ----------
    # row1: combined Gantt bar + exit_status dots (full width)
    # row2: SPACER (reserved space for dropdown menus, no plot)
    # row3: bar | pie
    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{"type": "xy", "colspan": 2}, None],
               [{"type": "xy", "colspan": 2}, None],
               [{"type": "xy"}, {"type": "domain"}]],
        subplot_titles=("Queue wait + run time by jobnumber (dot = exit_status)",
                         "",
                         f"{default_ru} by jobnumber",
                         f"Distribution: {default_pie}"),
        vertical_spacing=0.08,
        row_heights=[0.42, 0.12, 0.42],
    )

    trace_index = 0
    ru_trace_range = []
    pie_trace_range = []

    # --- Panel 1: stacked horizontal bar (queue wait + run time), back to back ---
    fig.add_trace(
        go.Bar(
            x=wait_hours, y=x_labels, orientation="h",
            name="Queue wait (qsub→start)",
            legendgroup="timeline",
            legendgrouptitle_text="timeline segments",
            marker_color="#F2C41A",
            customdata=wait_hover,
            hovertemplate="jobnumber=%{y}<br>queue wait=%{customdata}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=run_hours, y=x_labels, orientation="h",
            name="Run time (start→end)",
            legendgroup="timeline",
            marker_color="#4CAF50",
            customdata=run_hover,
            hovertemplate="jobnumber=%{y}<br>run time=%{customdata}<extra></extra>",
        ),
        row=1, col=1,
    )
    trace_index += 2

    # --- Panel 1 (cont.): dot at the end of each bar, colored by exit_status ---
    if have_exit_status:
        for i, status in enumerate(unique_statuses):
            mask = df["exit_status"] == status
            fig.add_trace(
                go.Scatter(
                    x=total_hours[mask], y=[x_labels[j] for j in mask[mask].index],
                    mode="markers",
                    name=exit_status_label(status),
                    legendgroup="exit_status",
                    legendgrouptitle_text="exit_status",
                    marker=dict(size=10, color=PALETTE[i % len(PALETTE)],
                                line=dict(width=1, color="white")),
                    hovertemplate="jobnumber=%{y}<br>exit_status=" + exit_status_label(status) + "<extra></extra>",
                ),
                row=1, col=1,
            )
            trace_index += 1

    fig.update_yaxes(categoryorder="array", categoryarray=y_category_order, row=1, col=1)
    fig.update_xaxes(title_text="duration (hours)", row=1, col=1)
    fig.update_yaxes(title_text="jobnumber", row=1, col=1)

    # --- Spacer row (row 2): hide its axes entirely, dropdowns live here ---
    fig.update_xaxes(visible=False, row=2, col=1)
    fig.update_yaxes(visible=False, row=2, col=1)

    # --- Panel 2: bar (ru_* + cpu/mem/io/iow/maxvmem/arid) ---
    start = trace_index
    for field in bar_fields:
        fig.add_trace(
            go.Bar(
                x=x_labels, y=df[field], name=field,
                visible=(field == default_ru),
                hovertemplate="jobnumber=%{x}<br>" + field + "=%{y}<extra></extra>",
                marker_color="#4C78A8",
                showlegend=False,
            ),
            row=3, col=1,
        )
        trace_index += 1
    ru_trace_range = list(range(start, trace_index))

    # --- Panel 3: pie ---
    start = trace_index
    for field in pie_fields:
        counts = pie_data[field]
        fig.add_trace(
            go.Pie(
                labels=counts.index.tolist(),
                values=counts.values.tolist(),
                name=field,
                visible=(field == default_pie),
                hovertemplate="%{label}: %{value} jobs (%{percent})<extra></extra>",
                showlegend=False,
            ),
            row=3, col=2,
        )
        trace_index += 1
    pie_trace_range = list(range(start, trace_index))

    fig.update_xaxes(title_text="jobnumber", type="category", row=3, col=1)
    fig.update_yaxes(title_text=default_ru, row=3, col=1)

    # ---------- Dropdown menus ----------
    def build_buttons(field_list, trace_range):
        buttons = []
        for i, field in enumerate(field_list):
            visible = [j == i for j in range(len(field_list))]
            buttons.append(dict(label=field, method="restyle",
                                 args=[{"visible": visible}, trace_range]))
        return buttons

    ru_buttons = build_buttons(bar_fields, ru_trace_range)
    pie_buttons = build_buttons(pie_fields, pie_trace_range)

    fig.update_layout(
        updatemenus=[
            dict(buttons=ru_buttons, direction="down", x=0.0, xanchor="left",
                 y=0.52, yanchor="middle", showactive=True),
            dict(buttons=pie_buttons, direction="down", x=0.62, xanchor="left",
                 y=0.52, yanchor="middle", showactive=True),
        ],
        barmode="stack",
        height=1250,
        template="plotly_white",
        title="qacct Job Dashboard",
        bargap=0.2,
        legend=dict(x=1.02, y=1.0, yanchor="top", groupclick="togglegroup"),
        margin=dict(t=100),
    )

    fig.write_html(args.out)
    print(f"Wrote dashboard to {args.out}")
    if have_exit_status:
        print(f"  Panel 1 (gantt+dots): queue wait + run time per job, {len(unique_statuses)} distinct exit_status values")
    print(f"  Panel 2 (bar):     {len(bar_fields)} fields ({len(ru_fields)} ru_*, {len(extra_fields)} extra), {len(df)} jobs")
    print(f"  Panel 3 (pie):     {len(pie_fields)} categorical fields")

    try:
        fig.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()