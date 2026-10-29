#!/usr/bin/env python3
"""Plot every recorded sample before and after the reproduced affine fit."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--audit-alignment", action="store_true",
                        help="Run the optional audit_panel_alignment helper from PYTHONPATH")
    args = parser.parse_args()
    results = args.results.resolve()
    with (results / "per_sample.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    summary = json.loads((results / "summary.json").read_text())
    if len(rows) != summary["samples"]:
        raise ValueError("Sample count differs from the fit summary")
    groups = {(ch, voltage): [] for ch in "XYZ" for voltage in range(1, 6)}
    for row in rows:
        groups[(row["channel"], int(row["voltage_V"]))].append(
            (int(row["sample_index"]), float(row["pre_uT"]), float(row["post_uT"])))
    if any(not group for group in groups.values()):
        raise ValueError("A required coil/voltage condition is missing")
    max_count = max(len(group) for group in groups.values())
    max_value = max(max(row[1], row[2]) for group in groups.values() for row in group)
    upper = 10 * np.ceil(max_value / 10)

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "pdf.fonttype": 42, "svg.fonttype": "none", "legend.frameon": False,
    })
    fig, axes = plt.subplots(3, 5, figsize=(7.2, 4.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.14, top=0.87,
                        hspace=0.43, wspace=0.22)
    for row_index, channel in enumerate("XYZ"):
        for column, voltage in enumerate(range(1, 6)):
            ax = axes[row_index, column]
            data = np.asarray(groups[(channel, voltage)])
            if not np.array_equal(data[:, 0], np.arange(1, len(data) + 1)):
                raise ValueError("Sample order is not consecutive")
            ax.plot(data[:, 0], data[:, 1], color="#777777", linewidth=0.65,
                    linestyle="--", label="Before affine fit")
            ax.plot(data[:, 0], data[:, 2], color="#0072B2", linewidth=0.8,
                    label="After affine fit")
            ax.set_title(f"{channel}, {voltage} V", loc="left", pad=5)
            ax.set_xlim(0, max_count + 5)
            ax.set_ylim(0, upper)
            ax.set_xticks([0, 100, 200])
            ax.set_yticks(np.arange(0, upper + 1, 20))
            ax.tick_params(length=2.5, pad=2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.54, 0.995),
               ncols=2, handlelength=2.5, columnspacing=2)
    fig.supxlabel("Recorded sample index within each condition", y=0.045, fontsize=8)
    fig.supylabel("Residual inconsistency Δo (μT)", x=0.012, fontsize=8)
    fig.canvas.draw()
    if args.audit_alignment:
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(
            fig, json_out=results / "residual_consistency.alignment.json",
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
        )
    fig.savefig(results / "residual_consistency.pdf")
    fig.savefig(results / "residual_consistency.svg")
    fig.savefig(results / "residual_consistency.png", dpi=600)
    plt.close(fig)
    print(json.dumps({"samples_plotted": len(rows), "conditions": len(groups),
                      "excluded_samples": 0, "matplotlib_version": matplotlib.__version__}))


if __name__ == "__main__":
    main()
