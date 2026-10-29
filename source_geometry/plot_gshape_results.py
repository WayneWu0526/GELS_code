#!/usr/bin/env python3
"""Render position and orientation error from a completed g_shape sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


CM_TO_INCH = 1.0 / 2.54
FIGURE_WIDTH_CM = 19.6
FIGURE_HEIGHT_CM = 8.74
CAMERA_FOCAL_POINT_M = np.array([0.91, 0.0, 1.05], dtype=float)
CAMERA_DISTANCE_M = 1.35
CAMERA_YAW_RAD = 1.80
CAMERA_PITCH_RAD = 0.42
CAMERA_VERTICAL_FOV_RAD = 0.75
SCENE_G_SHAPE = 1.0
SOURCE_COLOR = "#F4B942"
POSITION_COLOR = "#2377B4"  # RGB (35, 119, 180)
ORIENTATION_COLOR = "#D62628"  # RGB (214, 38, 40)


def configure_plot_style() -> None:
    """Apply the manuscript's compact, LaTeX-rendered figure style."""
    plt.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{bm}",
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.minor.width": 0.9,
            "ytick.minor.width": 0.9,
            "xtick.major.size": 5.0,
            "ytick.major.size": 5.0,
            "xtick.minor.size": 3.0,
            "ytick.minor.size": 3.0,
            "xtick.major.pad": 3.6,
            "ytick.major.pad": 3.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_summary(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no rows found in {path}")
    keys = (
        "g_shape",
        "position_error_mean_mm",
        "position_error_p10_mm",
        "position_error_p90_mm",
        "orientation_error_mean_deg",
        "orientation_error_p10_deg",
        "orientation_error_p90_deg",
        "success_rate",
    )
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=float)
        for key in keys
    }


def load_source_tcp_points(
    path: Path,
    g_shape: float = SCENE_G_SHAPE,
) -> np.ndarray:
    """Read the three EM40250 TCP positions for one frozen geometry state."""
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    states = document.get("winner", {}).get("states", [])
    matches = [
        state
        for state in states
        if np.isclose(float(state["g_shape"]), g_shape, atol=1.0e-12)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one frozen source state for g={g_shape:g}")
    targets = sorted(
        matches[0]["targets"].values(),
        key=lambda target: int(target["vertex_index"]),
    )
    link_poses = matches[0]["link_poses"]
    return np.asarray(
        [link_poses[target["ee_link"]]["position_m"] for target in targets],
        dtype=float,
    )


def project_rviz_points(points_m: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray:
    """Project world points with the fixed RViz Orbit camera used for capture."""
    camera_from_focal = np.array(
        [
            np.cos(CAMERA_YAW_RAD) * np.cos(CAMERA_PITCH_RAD),
            np.sin(CAMERA_YAW_RAD) * np.cos(CAMERA_PITCH_RAD),
            np.sin(CAMERA_PITCH_RAD),
        ],
        dtype=float,
    )
    camera_right = np.array(
        [-camera_from_focal[1], camera_from_focal[0], 0.0], dtype=float
    )
    camera_right /= np.linalg.norm(camera_right)
    camera_up = np.cross(camera_from_focal, camera_right)
    camera_up /= np.linalg.norm(camera_up)
    camera_position = (
        CAMERA_FOCAL_POINT_M + CAMERA_DISTANCE_M * camera_from_focal
    )
    view_direction = -camera_from_focal

    image_height, image_width = image_shape[:2]
    focal_pixels = 0.5 * image_height / np.tan(0.5 * CAMERA_VERTICAL_FOV_RAD)
    relative = points_m - camera_position
    depth = relative @ view_direction
    if np.any(depth <= 0.0):
        raise ValueError("source point lies behind the fixed RViz camera")
    horizontal = relative @ camera_right
    vertical = relative @ camera_up
    return np.column_stack(
        (
            0.5 * image_width + focal_pixels * horizontal / depth,
            0.5 * image_height - focal_pixels * vertical / depth,
        )
    )


def draw_scene_panel(
    scene_axis: plt.Axes,
    scene_image: Path | None,
    scene_geometry: Path | None,
    scene_g_shape: float = SCENE_G_SHAPE,
    panel_label: str = "(a)",
    show_source_labels: bool = True,
) -> None:
    if panel_label:
        scene_axis.set_title(panel_label, loc="left", pad=2.0)
    if scene_image is None or scene_geometry is None:
        scene_axis.set_xticks([])
        scene_axis.set_yticks([])
        for spine in scene_axis.spines.values():
            spine.set_color("#AFAFAF")
            spine.set_linewidth(1.2)
        return

    image = plt.imread(str(scene_image))
    source_pixels = project_rviz_points(
        load_source_tcp_points(scene_geometry, scene_g_shape), image.shape
    )
    closed = np.vstack((source_pixels, source_pixels[0]))

    scene_axis.imshow(image)
    scene_axis.plot(
        closed[:, 0], closed[:, 1], color=SOURCE_COLOR, linewidth=2.6, zorder=4
    )
    label_offsets = ((24, 0), (-24, 0), (0, 20))
    marker_styles = ("o", "s", "^")
    marker_sizes = (58, 62, 72)
    for source_index, (pixel, offset, marker_style, marker_size) in enumerate(
        zip(source_pixels, label_offsets, marker_styles, marker_sizes), start=1
    ):
        scene_axis.scatter(
            [pixel[0]],
            [pixel[1]],
            s=marker_size,
            marker=marker_style,
            facecolor="white",
            edgecolor=SOURCE_COLOR,
            linewidth=1.8,
            zorder=5,
        )
        if show_source_labels:
            scene_axis.annotate(
                rf"$\mathrm{{EM}}_{source_index}$",
                xy=pixel,
                xytext=offset,
                textcoords="offset points",
                ha="center",
                va="center",
                color=SOURCE_COLOR,
                arrowprops={
                    "arrowstyle": "-",
                    "color": SOURCE_COLOR,
                    "lw": 1.4,
                },
                zorder=6,
            )
    scene_axis.text(
        0.97,
        0.96,
        rf"$g_{{\mathrm{{shape}}}}={scene_g_shape:g}$",
        transform=scene_axis.transAxes,
        ha="right",
        va="top",
        color=SOURCE_COLOR,
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "#303030",
            "edgecolor": "none",
            "alpha": 0.72,
        },
        zorder=7,
    )
    scene_axis.set_xlim(-0.5, image.shape[1] - 0.5)
    scene_axis.set_ylim(image.shape[0] - 0.5, -0.5)
    scene_axis.set_xticks([])
    scene_axis.set_yticks([])
    for spine in scene_axis.spines.values():
        spine.set_color("#AFAFAF")
        spine.set_linewidth(1.2)


def render(
    summary_csv: Path,
    output_stem: Path,
    label: str,
    scene_image: Path | None,
    scene_geometry: Path | None,
) -> None:
    configure_plot_style()
    data = load_summary(summary_csv)
    order = np.argsort(data["g_shape"])
    g_shape = data["g_shape"][order]
    mean_error = data["position_error_mean_mm"][order]
    p10 = data["position_error_p10_mm"][order]
    p90 = data["position_error_p90_mm"][order]
    mean_orientation_error = data["orientation_error_mean_deg"][order]
    orientation_p10 = data["orientation_error_p10_deg"][order]
    orientation_p90 = data["orientation_error_p90_deg"][order]

    figure = plt.figure(
        figsize=(FIGURE_WIDTH_CM * CM_TO_INCH, FIGURE_HEIGHT_CM * CM_TO_INCH)
    )
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(1.0, 1.1),
        left=0.05,
        right=0.97,
        bottom=0.12,
        top=0.94,
        wspace=0.34,
    )

    scene_axis = figure.add_subplot(grid[0, 0])
    draw_scene_panel(scene_axis, scene_image, scene_geometry)

    error_grid = grid[0, 1].subgridspec(2, 1, hspace=0.16)
    position_axis = figure.add_subplot(error_grid[0, 0])
    orientation_axis = figure.add_subplot(
        error_grid[1, 0],
        sharex=position_axis,
    )

    position_axis.set_title(r"(b)", loc="left", pad=2.0)
    position_axis.fill_between(
        g_shape,
        p10,
        p90,
        color=POSITION_COLOR,
        alpha=0.20,
        linewidth=0.0,
    )
    position_axis.plot(
        g_shape,
        mean_error,
        color=POSITION_COLOR,
        marker="o",
        markevery=2,
        markersize=4.8,
        linewidth=2.2,
    )
    position_axis.set_xscale("log")
    position_axis.set_yscale("log")
    position_axis.set_xlim(1.0e-2, 1.0)
    position_axis.set_ylim(5.0, 5.0e2)
    position_axis.set_yticks((1.0e1, 1.0e2))
    position_axis.set_ylabel(r"$e_{\bm{p}}$ [mm]")
    position_axis.tick_params(axis="x", which="both", labelbottom=False)
    position_axis.grid(True, which="both", color="#D9D9D9", linewidth=0.7)

    orientation_axis.fill_between(
        g_shape,
        orientation_p10,
        orientation_p90,
        color=ORIENTATION_COLOR,
        alpha=0.15,
        linewidth=0.0,
    )
    orientation_axis.plot(
        g_shape,
        mean_orientation_error,
        color=ORIENTATION_COLOR,
        marker="s",
        markevery=2,
        markersize=4.4,
        linewidth=2.0,
    )
    orientation_axis.set_xscale("log")
    orientation_axis.set_yscale("log")
    orientation_axis.set_xlim(1.0e-2, 1.0)
    orientation_axis.set_ylim(8.0e-1, 1.0e2)
    orientation_axis.set_yticks((1.0e0, 1.0e1))
    orientation_axis.set_xticks((1.0e-2, 1.0e-1, 1.0))
    orientation_axis.set_xticklabels(
        (r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$")
    )
    orientation_axis.set_xlabel(r"$g_{\mathrm{shape}}=\sigma_2/\sigma_1$")
    orientation_axis.set_ylabel(r"$e_{\bm{R}}$ [deg]")
    orientation_axis.grid(True, which="both", color="#D9D9D9", linewidth=0.7)
    if label:
        latex_label = label.replace("•", r"$\bullet$")
        orientation_axis.text(
            0.02,
            0.02,
            latex_label,
            transform=orientation_axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=16,
            color="#777777",
        )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"))
    figure.savefig(output_stem.with_suffix(".png"), dpi=300)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output-stem", required=True, type=Path)
    parser.add_argument("--label", default="")
    parser.add_argument("--scene-image", type=Path)
    parser.add_argument("--scene-geometry", type=Path)
    arguments = parser.parse_args()
    if (arguments.scene_image is None) != (arguments.scene_geometry is None):
        parser.error("--scene-image and --scene-geometry must be provided together")
    render(
        arguments.summary,
        arguments.output_stem,
        arguments.label,
        arguments.scene_image,
        arguments.scene_geometry,
    )


if __name__ == "__main__":
    main()
