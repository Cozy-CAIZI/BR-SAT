#!/usr/bin/env python3
"""Render verified Figure 3 forest-plot panels from the frozen cohort table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


COHORTS = [
    ("development_618", "Development", "#6F8795"),
    ("internal_160", "Historical internal", "#4F9791"),
    ("multicentre_200", "Multicentre", "#0F6B67"),
    ("prospective_102", "Prospective", "#D49335"),
]

METRICS = {
    "sensitivity": {
        "panel": "A",
        "title": "Sensitivity",
        "percent": True,
        "ci_low": "sensitivity_ci_low",
        "ci_high": "sensitivity_ci_high",
    },
    "specificity": {
        "panel": "B",
        "title": "Specificity",
        "percent": True,
        "ci_low": "specificity_ci_low",
        "ci_high": "specificity_ci_high",
    },
    "balanced_accuracy": {
        "panel": "C",
        "title": "Balanced accuracy",
        "percent": True,
        "ci_low": "balanced_accuracy_ci_low",
        "ci_high": "balanced_accuracy_ci_high",
    },
    "auroc": {
        "panel": "D",
        "title": "AUROC",
        "percent": False,
        "ci_low": "auroc_ci_low",
        "ci_high": "auroc_ci_high",
    },
}


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["cohort"]: row for row in rows}


def render(metric: str, source: Path, output: Path) -> None:
    spec = METRICS[metric]
    rows = read_rows(source)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.linewidth": 0.7,
            "axes.edgecolor": "#46565C",
            "axes.labelcolor": "#111111",
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    # 89 mm single-column module; designed for a later 2×2 Figure 3 assembly.
    fig, ax = plt.subplots(figsize=(3.50, 2.65), dpi=600)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y_positions = [3, 2, 1, 0]
    y_labels: list[str] = []

    for y, (key, label, color) in zip(y_positions, COHORTS):
        row = rows[key]
        estimate = float(row[metric])
        lower = float(row[spec["ci_low"]])
        upper = float(row[spec["ci_high"]])
        if spec["percent"]:
            estimate *= 100
            lower *= 100
            upper *= 100

        ax.errorbar(
            estimate,
            y,
            xerr=[[estimate - lower], [upper - estimate]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.55,
            capsize=3.0,
            capthick=1.25,
            markersize=5.8,
            markeredgecolor="white",
            markeredgewidth=0.55,
            zorder=3,
        )

        if spec["percent"]:
            value_label = f"{estimate:.2f}%"
        else:
            value_label = f"{estimate:.4f}"

        ax.text(
            1.025,
            y,
            value_label,
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=6.8,
            color="#111111",
            clip_on=False,
        )
        y_labels.append(label)

    if spec["percent"]:
        ax.set_xlim(50, 100)
        ax.set_xticks([50, 60, 70, 80, 90, 100])
        ax.set_xlabel("Estimate (%)", fontsize=8)
    else:
        ax.set_xlim(0.50, 1.00)
        ax.set_xticks([0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
        ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
        ax.set_xlabel("Estimate", fontsize=8)

    ax.set_ylim(-0.65, 3.65)
    ax.set_yticks(y_positions, labels=y_labels)
    ax.tick_params(axis="y", length=0, pad=5, labelsize=7.1)
    ax.tick_params(axis="x", length=3.2, width=0.7, labelsize=6.8)
    ax.grid(axis="x", color="#DDE3E6", linewidth=0.55, zorder=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    fig.text(0.035, 0.955, spec["panel"], ha="left", va="top", fontsize=11, fontweight="bold", color="#111111")
    fig.text(0.180, 0.955, spec["title"], ha="left", va="top", fontsize=9.3, fontweight="bold", color="#111111")

    fig.subplots_adjust(left=0.330, right=0.825, top=0.825, bottom=0.195)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, facecolor="white")
    fig.savefig(output.with_suffix(".svg"), facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    fig.savefig(output.with_suffix(".tiff"), dpi=600, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=METRICS, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.metric, args.source, args.output)


if __name__ == "__main__":
    main()
