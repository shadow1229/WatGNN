
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

def read_section(section, total_water):
    df = pd.DataFrame(section["rows"])
    finite = df[df["right"].apply(math.isfinite)].copy()
    centers = (finite["left"] + finite["right"]) / 2.0
    y_pct = 100.0 * finite["count"] / total_water
    return centers, y_pct

def main():
    total_water_cryst, sections_cryst = parse_histogram_file('watgnn_test_S2_S8.txt')
    total_water_pred, sections_pred = parse_histogram_file('watgnn_test_S2_S8_pred.txt')
    
    summary = []
    for i in range(len(sections_cryst)):

        label = sections_cryst[i]["label"].lower()
        if 'ignore h-bond eligibility' not in label:
            continue

        section_cryst = sections_cryst[i]   
        section_pred  = sections_pred[i]
        centers_cryst, y_pct_cryst = read_section(section_cryst, total_water_cryst)
        centers_pred , y_pct_pred = read_section(section_pred, total_water_pred)


        plt.rc('mathtext', fontset='cm')
        fig = plt.figure(figsize = (6,4),
                     facecolor = 'white',
                     edgecolor = 'black',
                     dpi  = 300
                        )
        prop = FontProperties(size=12) #unicode
        xmax = 6.0
        ymax = 40.0
        
        ax = fig.add_axes([0.14,0.15,0.80,0.74])
        ax.set_title(r'$\mathrm{Nearest}$ $\mathrm{polar-atom}$ $\mathrm{distance}$ $\mathrm{for}$ $\mathrm{crystallographic}$ $\mathrm{water}$ $\mathrm{molecules}$',fontproperties=prop)
        ax.set_xlabel(r'$\mathrm{Minimum}$ $\mathrm{polar}$ $\mathrm{atom}$ $\mathrm{distance}$ $\mathrm{( {\AA } )}$',fontproperties=prop)
        ax.set_ylabel(r'$\mathrm{Fraction}$ $\mathrm{of}$ $\mathrm{water}$ $\mathrm{molecules}$ (%)',fontproperties=prop)
        
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        
        ax.plot(centers_cryst , y_pct_cryst, color='#FF0000' ,label=r'$\mathrm{Crystallographic}$ $\mathrm{water}$ $\mathrm{positions}$')
        ax.plot(centers_pred  , y_pct_pred,  color='#000000' ,label=r'$\mathrm{WatGNN}$ $\mathrm{predictions}$')
        ax.legend(bbox_to_anchor=(1.00,1.0))

        png = "FigS8.png"
        fig.savefig(png, dpi=600, bbox_inches=None)
        plt.close(fig)



if __name__ == "__main__":
    main()
