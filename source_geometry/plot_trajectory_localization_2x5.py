#!/usr/bin/env python3
"""Compose height, planar-error, and RViz panels for five frozen source states."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


FIGURE_WIDTH_CM = 19.8
FIGURE_WIDTH_IN = FIGURE_WIDTH_CM / 2.54
FIGURE_ASPECT_RATIO = 0.53

matplotlib.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{textcomp}",
        "font.size": 8,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


ERROR_CMAP = colors.LinearSegmentedColormap.from_list(
    "paper_error_green_to_red",
    ["#1A9850", "#91CF60", "#FEE08B", "#FC8D59", "#D73027"],
    N=256,
)

SOURCE_MARKERS = ("o", "s", "^")
SOURCE_COLOR = "#a5534d"


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--rviz-images", nargs=5, required=True, type=Path)
    parser.add_argument("--output-stem", required=True, type=Path)
    args = parser.parse_args()
    missing = [str(path) for path in args.rviz_images if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing RViz images: " + ", ".join(missing))

    targets = read_rows(args.results / "target_poses.csv")
    errors = read_rows(args.results / "target_error_summary.csv")
    sources = read_rows(args.results / "source_states.csv")
    summaries = read_rows(args.results / "summary_by_state.csv")
    target_xy = np.array([[float(row["x_m"]), float(row["y_m"])] for row in targets])
    target_xz = np.array([[float(row["x_m"]), float(row["z_m"])] for row in targets])
    all_errors = np.array([float(row["position_error_mean_mm"]) for row in errors])
    positive = all_errors[all_errors > 0.0]
    norm = colors.LogNorm(vmin=float(np.percentile(positive, 2)), vmax=float(np.percentile(positive, 98)))
    cmap = ERROR_CMAP

    source_xy_all = np.array([[float(row["x_m"]), float(row["y_m"])] for row in sources])
    combined_xy = np.vstack((target_xy, source_xy_all))
    center = np.mean(combined_xy, axis=0)
    radius = 0.5 * max(np.ptp(combined_xy[:, 0]), np.ptp(combined_xy[:, 1])) + 0.025

    fig = plt.figure(
        figsize=(FIGURE_WIDTH_IN, FIGURE_WIDTH_IN * FIGURE_ASPECT_RATIO),
        facecolor="white",
    )
    grid = fig.add_gridspec(
        3, 5, height_ratios=[1.0, 0.25, 1.0], left=0.085, right=0.915,
        bottom=0.030, top=0.915, hspace=0.12, wspace=0.06,
    )
    for index, summary in enumerate(summaries):
        state_errors = [
            row for row in errors if int(row["state_index"]) == index
        ]
        error_by_target = {
            int(row["target_index"]): float(row["position_error_mean_mm"])
            for row in state_errors
        }
        values = np.array([
            error_by_target[int(row["target_index"])]
            for row in targets
        ])
        height_axis = fig.add_subplot(grid[1, index])
        height_axis.scatter(
            target_xz[:, 0],
            target_xz[:, 1],
            c=values,
            cmap=cmap,
            norm=norm,
            s=7,
            linewidths=0,
            alpha=0.92,
        )
        g_shape = float(summary["g_shape"])
        height_axis.set_xlim(
            center[0] - radius,
            center[0] + radius,
        )
        height_axis.set_ylim(0.70, 0.90)
        height_axis.set_xticks([0.8, 0.9, 1.0, 1.1])
        height_axis.set_xticklabels(["0.8", "0.9", "1.0", "1.1"])
        height_axis.tick_params(axis="x", labelsize=7, length=2, pad=1)
        if index == 2:
            height_axis.set_xlabel(r"$x$ [m]", fontsize=8, labelpad=1)
        if index == 0:
            height_axis.set_yticks([0.70, 0.80, 0.90])
            height_axis.set_yticklabels(["0.70", "0.80", "0.90"])
            height_axis.set_ylabel(r"$z$ [m]", fontsize=8, labelpad=2)
            height_axis.tick_params(axis="y", labelsize=7, length=2, pad=1)
        else:
            height_axis.set_yticks([])
        for spine in height_axis.spines.values():
            spine.set_color("#b7b7b7")

        axis = fig.add_subplot(grid[0, index])
        axis.scatter(
            target_xy[:, 0], target_xy[:, 1], c=values, cmap=cmap, norm=norm,
            s=9, linewidths=0, alpha=0.92, zorder=2,
        )
        state_sources = sorted(
            [row for row in sources if int(row["state_index"]) == index],
            key=lambda row: int(row["source_index"]),
        )
        source_xy = np.array([[float(row["x_m"]), float(row["y_m"])] for row in state_sources])
        closed = np.vstack((source_xy, source_xy[0]))
        axis.plot(closed[:, 0], closed[:, 1], color="#8b3f39", lw=1.1, zorder=3)
        for row, xy in zip(state_sources, source_xy):
            source_index = int(row["source_index"])
            marker = SOURCE_MARKERS[source_index]
            axis.scatter(
                [xy[0]],
                [xy[1]],
                s=42,
                marker=marker,
                facecolors="none",
                edgecolors="white",
                linewidths=2.5,
                zorder=4,
            )
            axis.scatter(
                [xy[0]],
                [xy[1]],
                s=34,
                marker=marker,
                facecolors="white",
                edgecolors=SOURCE_COLOR,
                linewidths=1.2,
                zorder=5,
            )
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_aspect("equal")
        axis.set_title(rf"$g_{{\mathrm{{shape}}}}={g_shape:g}$", fontsize=11, pad=3)
        axis.set_xticks([])
        if index == 0:
            axis.set_yticks([-0.1, 0.0, 0.1])
            axis.set_yticklabels(["-0.1", "0", "0.1"])
            axis.set_ylabel(r"$y$ [m]", fontsize=8, labelpad=2)
        else:
            axis.set_yticks([])
        axis.tick_params(axis="y", labelsize=7, length=2, pad=1)
        for spine in axis.spines.values():
            spine.set_color("#b7b7b7")

        image_axis = fig.add_subplot(grid[2, index])
        image_axis.imshow(plt.imread(str(args.rviz_images[index])))
        image_axis.set_axis_off()

    colorbar_axis = fig.add_axes([0.945, 0.125, 0.013, 0.675])
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=colorbar_axis,
        orientation="vertical",
    )
    colorbar.set_label(
        "Mean position error per workspace sample [mm]",
        fontsize=9,
        rotation=90,
        labelpad=8,
    )
    colorbar.ax.yaxis.set_label_position("left")
    colorbar.ax.tick_params(labelsize=8, pad=1)
    legend_handles = [
        Line2D(
            [],
            [],
            marker=marker,
            markerfacecolor="white",
            markeredgecolor=SOURCE_COLOR,
            markeredgewidth=1.2,
            markersize=5.5,
            linestyle="none",
            label=rf"EM{index + 1}",
        )
        for index, marker in enumerate(SOURCE_MARKERS)
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.924, 0.905),
        frameon=False,
        fontsize=8,
        handletextpad=0.35,
        labelspacing=0.25,
        borderaxespad=0.0,
    )
    fig.text(0.010, 0.735, r"(a) Top view", rotation=90, va="center", ha="center", fontsize=9)
    fig.text(0.010, 0.475, r"(b) Side view", rotation=90, va="center", ha="center", fontsize=9)
    fig.text(0.010, 0.225, r"(c) RViz state", rotation=90, va="center", ha="center", fontsize=9)

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_stem.with_suffix(".png"), dpi=600, facecolor="white")
    # The PDF backend keeps plotted geometry as vectors, but ``imshow`` panels
    # are rasterized at the save DPI. Match the 600 dpi PNG export so that the
    # RViz row remains legible after the five-column figure is typeset.
    fig.savefig(
        args.output_stem.with_suffix(".pdf"),
        dpi=600,
        facecolor="white",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
