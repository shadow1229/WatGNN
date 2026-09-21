
#!/usr/bin/env python3
"""
plot_watgnn_point3_histograms.py

Parse WatGNN histogram text files of the form:

#histogram: <description>
 0.00A - 0.10A : 123 0.001
 ...

and generate publication-ready matplotlib figures.

Usage:
    python plot_watgnn_point3_histograms.py gnn_revision_point3_test.txt

Outputs:
    - one PNG and one PDF per histogram section
    - a CSV summary of section totals
    - an optional assignment-summary figure for sections containing
      "(p-w)", "(w-w, no p-w)", and "(no p-w / w-w)"

Notes:
    * The program uses the supplied "portion" column as the fraction of the
      total crystallographic-water population and plots it as percentage.
    * The final "inf" bin is excluded from distance plots because it has no
      finite bin width, but its count remains in the summary CSV.
    * No custom color palette is imposed; matplotlib defaults are used.
"""

from pathlib import Path
import argparse
import re
import math
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import rc
import numpy as np
from matplotlib.font_manager import FontProperties #unicode

BIN_RE = re.compile(
    r"^\s*([0-9.]+)A\s*-\s*(inf|[0-9.]+A)\s*:\s*([0-9]+)\s+([0-9.]+)\s*$"
)
color  = ['#000000','#FF0000','#FF8800','#00FF00','#0000FF','#000000','#880088','#008888']

def parse_histogram_file(path):
    path = Path(path)
    total_water = None
    sections = []
    current = None

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip("\n")

            if "total water molecules" in line:
                m = re.search(r"([0-9]+)\s*$", line) #finding last number part
                if m:
                    total_water = int(m.group(1))

            if line.startswith("#histogram:"):
                if current is not None:
                    sections.append(current)
                current = {
                    "label": line.split(":", 1)[1].strip(),
                    "rows": []
                }
                continue

            if current is None:
                continue

            m = BIN_RE.match(line)
            if m:
                left = float(m.group(1))
                right_token = m.group(2)
                right = math.inf if right_token == "inf" else float(right_token[:-1])
                count = int(m.group(3))
                portion = float(m.group(4))
                current["rows"].append(
                    {
                        "left": left,
                        "right": right,
                        "count": count,
                        "portion": portion,
                    }
                )

    if current is not None:
        sections.append(current)

    return total_water, sections


def short_name(label):
    ll = label.lower()
    if "ignore h-bond eligibility" in ll:
        return "all_polar"
    if "(p-w)" in ll:
        return "pw_assignable"
    if "(w-w, no p-w)" in ll:
        return "ww_assignable"
    if "(no p-w / w-w)" in ll:
        return "unassigned_carbon"
    return re.sub(r"[^a-z0-9]+", "_", ll).strip("_")[:60]


def display_title(label):
    ll = label.lower()
    if "ignore h-bond eligibility" in ll:
        return r'$\mathrm{Nearest}$ $\mathrm{polar-atom}$ $\mathrm{distance}$ $\mathrm{for}$ $\mathrm{crystallographic}$ $\mathrm{water}$ $\mathrm{molecules}$'
    if "(p-w)" in ll:
        return r'$\mathrm{Protein–water}$ $\mathrm{channel}$ $\mathrm{assignable}$ $\mathrm{crystallographic}$ $\mathrm{water}$ $\mathrm{molecules}$'
    if "(w-w, no p-w)" in ll:
        return r'$\mathrm{Water–water}$ $\mathrm{channel}$ $\mathrm{assignable}$ $\mathrm{crystallographic}$ $\mathrm{water}$ $\mathrm{molecules}$'
    if "(no p-w / w-w)" in ll:
        return r'$\mathrm{Unassignable}$ $\mathrm{crystallographic}$ $\mathrm{water}$ $\mathrm{molecules}$'
    return label


def plot_section(section, out_dir, total_water, xmax=None, ymax=None, normalization="total"):
    df = pd.DataFrame(section["rows"])
    finite = df[df["right"].apply(math.isfinite)].copy()

    centers = (finite["left"] + finite["right"]) / 2.0
    widths = finite["right"] - finite["left"]

    if normalization == "section":
        section_total = df["count"].sum()
        y_pct = 100.0 * finite["count"] / section_total
        ylabel = r'$\mathrm{Fraction}$ $\mathrm{within}$ $\mathrm{this}$ $\mathrm{water}$ $\mathrm{class}$ (%)'
        suffix = "sectionnorm"
    else:
        # Use exact counts rather than the rounded portion column.
        y_pct = 100.0 * finite["count"] / total_water
        ylabel = r'$\mathrm{Fraction}$ $\mathrm{of}$ $\mathrm{water}$ $\mathrm{molecules}$ (%)'
        suffix = "totalnorm"
    plt.rc('mathtext', fontset='cm')
    fig = plt.figure(figsize = (6,4),
                     facecolor = 'white',
                     edgecolor = 'black',
                     dpi  = 300
                    )
    prop = FontProperties(size=12) #unicode
    #ax = fig.add_subplot(111)
    ax = fig.add_axes([0.14,0.15,0.80,0.74])

    if 'ignore h-bond eligibility' in section["label"]:
        ax.set_xlabel(r'$\mathrm{Minimum}$ $\mathrm{polar}$ $\mathrm{atom}$ $\mathrm{distance}$ $\mathrm{( {\AA } )}$',fontproperties=prop)
        color  = '#888888'
    elif "(p-w)" in section["label"]:
        ax.set_xlabel(r'$\mathrm{Assigned}$ $\mathrm{polar}$ $\mathrm{atom}$ $\mathrm{distance}$ $\mathrm{( {\AA } )}$',fontproperties=prop)
        color  = '#00FF00'        

    elif "(w-w, no p-w)" in section["label"]:
        ax.set_xlabel(r'$\mathrm{Assigned}$ $\mathrm{polar}$ $\mathrm{atom}$ $\mathrm{distance}$ $\mathrm{( {\AA } )}$',fontproperties=prop)
        color  = '#FF8800'  
        
    elif "(no p-w / w-w)" in section["label"]:
        ax.set_xlabel(r'$\mathrm{Minimum}$ $\mathrm{carbon}$ $\mathrm{atom}$ $\mathrm{distance}$ $\mathrm{( {\AA } )}$',fontproperties=prop)
        color  = '#FF0000'
        
    else:
        ax.set_xlabel(r'$\mathrm{Minimum}$ $\mathrm{polar}$ $\mathrm{atom}$ $\mathrm{distance}$ $\mathrm{( {\AA } )}$',fontproperties=prop)
        color  = '#0000FF'
        
    ax.bar(
        centers,
        y_pct,
        width=widths * 0.96,
        align="center",
        linewidth=0.35,
        color = color,
        edgecolor="black",
    )


    ax.set_ylabel(ylabel,fontproperties=prop)
    ax.set_title(display_title(section["label"]),fontproperties=prop)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    if xmax is not None:
        ax.set_xlim(0, xmax)
    else:
        # Avoid long visually empty tails if all nonzero finite bins end early.
        nonzero = finite[finite["count"] > 0]
        if len(nonzero):
            data_max = float(nonzero["right"].max())
            ax.set_xlim(0, min(max(5.0, math.ceil(data_max * 2) / 2), 10.1))
    if normalization == "total":
        ax.set_ylim(0, ymax)        
    #fig.tight_layout()

    stem = short_name(section["label"])
    png = out_dir / f"{stem}_{suffix}.png"
    pdf = out_dir / f"{stem}_{suffix}.pdf"

    fig.savefig(png, dpi=600, bbox_inches=None)
    fig.savefig(pdf, bbox_inches=None)
    plt.close(fig)

    return stem, int(df["count"].sum()), float(df["portion"].sum())


def plot_assignment_summary(total_water, sections, out_dir):
    selected = []
    colors = ['#00FF00','#FF8800','#FF0000']
    for section in sections:
        label = section["label"].lower()
        count = sum(row["count"] for row in section["rows"])

        if "(p-w)" in label:
            selected.append((r'$\mathrm{Protein–water}$ $\mathrm{channel}$', count))
        elif "(w-w, no p-w)" in label:
            selected.append((r'$\mathrm{Water–water}$ $\mathrm{channel}$', count))
        elif "(no p-w / w-w)" in label:
            selected.append((r'$\mathrm{Unassignable}$', count))

    if len(selected) != 3 or not total_water:
        return

    names = [x[0] for x in selected]
    percentages = [100.0 * x[1] / total_water for x in selected]
    plt.rc('mathtext', fontset='cm')
    fig = plt.figure(figsize = (6,4),
                     facecolor = 'white',
                     edgecolor = 'black',
                     dpi  = 300
                    )
    prop_ax = FontProperties(size=12) #unicode
    prop = FontProperties(size=12) #unicode
    #ax = fig.add_subplot(111)
    ax = fig.add_axes([0.14,0.15,0.80,0.74])
    ax.set_ylim(0, 100)
    bars = ax.bar(names, percentages, edgecolor="black", color=colors, width=0.5, linewidth=0.5)
    ax.set_ylabel(r'$\mathrm{Fraction}$ $\mathrm{of}$ $\mathrm{water}$ $\mathrm{molecules}$ (%)',fontproperties=prop)
    ax.set_title(r'$\mathrm{Crystallographic}$ $\mathrm{water}$ $\mathrm{assignment}$ $\mathrm{rate}$',fontproperties=prop)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=0)

    for bar, value in zip(bars, percentages):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}%",
            ha="center",
            va="bottom",
        )

    #fig.tight_layout()
    fig.savefig(out_dir / "assignment_summary.png", dpi=600, bbox_inches=None)
    fig.savefig(out_dir / "assignment_summary.pdf", bbox_inches=None)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("--output-dir", default="watgnn_point3_histograms")
    parser.add_argument(
        "--normalization",
        choices=["total", "section", "both"],
        default="total",
        help="Normalize each bin by all crystallographic waters, by the section total, or save both."
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_water, sections = parse_histogram_file(args.input_file)

    summary = []
    for section in sections:
        label = section["label"].lower()
    
        xmax = 6.0
        if 'ignore h-bond eligibility' in label or "(p-w)" in label:
            ymax = 25.0
        elif "(no p-w / w-w)" in label or "(w-w, no p-w)" in label:
            ymax = 2.5
        else:
            ymax = 25.0

        norms = ["total", "section"] if args.normalization == "both" else [args.normalization]
        for norm in norms:
            stem, count, portion_sum = plot_section(
                section, out_dir, total_water, xmax=xmax, ymax=ymax, normalization=norm
            )
        summary.append(
            {
                "section": stem,
                "description": section["label"],
                "count": count,
                "fraction_from_input": portion_sum,
                "percent_of_total_from_count":
                    (100.0 * count / total_water if total_water else None),
            }
        )

    plot_assignment_summary(total_water, sections, out_dir)

    pd.DataFrame(summary).to_csv(out_dir / "histogram_summary.csv", index=False)

    print(f"Total crystallographic waters reported in file: {total_water}")
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"\nFigures written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
