#!/usr/bin/env python3
"""Generate quantitative BR-SAT figures and manuscript-ready CSV tables."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure3(metrics: list[dict[str, str]], output: Path) -> None:
    specifications = (
        ("sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("auroc", "AUROC"),
    )
    colours = ["#6F8795", "#4F9791", "#0F6B67", "#D49335"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3), dpi=300)
    y = np.arange(len(metrics))[::-1]
    for panel, (axis, (metric, title)) in enumerate(zip(axes.flat, specifications)):
        for index, row in enumerate(metrics):
            estimate = float(row[metric])
            low = float(row[f"{metric}_ci_low"])
            high = float(row[f"{metric}_ci_high"])
            axis.errorbar(
                estimate,
                y[index],
                xerr=[[estimate - low], [high - estimate]],
                fmt="o",
                color=colours[index % len(colours)],
                capsize=3,
            )
        axis.set_yticks(y, [row["cohort"] for row in metrics])
        axis.set_xlim(0.5, 1.0)
        axis.grid(axis="x", color="#DDE3E6", linewidth=0.6)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
        axis.set_title(f"{chr(65 + panel)}  {title}", loc="left", fontweight="bold")
        axis.set_xlabel("Estimate (95% CI)")
    fig.suptitle("Fixed-threshold binary performance across evidence layers", fontweight="bold")
    fig.tight_layout()
    save(fig, output / "Figure_3_quantitative_performance")


def figure4(metrics: list[dict[str, str]], roc: list[dict[str, str]], pr: list[dict[str, str]], output: Path) -> None:
    prospective = next((row for row in metrics if row["cohort"] == "prospective_102"), metrics[-1])
    matrix = np.array([[int(prospective["tn"]), int(prospective["fp"])], [int(prospective["fn"]), int(prospective["tp"])]])
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.8), dpi=300)
    image = axes[0].imshow(matrix, cmap="Blues", vmin=0)
    for (row, col), value in np.ndenumerate(matrix):
        axes[0].text(col, row, str(value), ha="center", va="center", fontweight="bold")
    axes[0].set_xticks([0, 1], ["Non-reactive", "Reactive"], rotation=20)
    axes[0].set_yticks([0, 1], ["Non-reactive", "Reactive"])
    axes[0].set_xlabel("BR-SAT prediction")
    axes[0].set_ylabel("Reference")
    axes[0].set_title("A  Confusion matrix", loc="left", fontweight="bold")
    fig.colorbar(image, ax=axes[0], fraction=0.046)

    axes[1].plot([float(row["fpr"]) for row in roc], [float(row["tpr"]) for row in roc], color="#0F6B67", lw=2)
    axes[1].plot([0, 1], [0, 1], "--", color="#8C989E", lw=1)
    axes[1].set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1))
    axes[1].set_title(f"B  ROC (AUROC {float(prospective['auroc']):.4f})", loc="left", fontweight="bold")

    axes[2].plot([float(row["recall"]) for row in pr], [float(row["precision"]) for row in pr], color="#D49335", lw=2)
    baseline = int(prospective["reactive_n"]) / int(prospective["n"])
    axes[2].axhline(baseline, ls="--", color="#8C989E", lw=1)
    axes[2].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    axes[2].set_title(f"C  PR (AUPRC {float(prospective['auprc']):.4f})", loc="left", fontweight="bold")
    for axis in axes[1:]:
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, output / "Figure_4_prospective_evaluation")


def figure5(site_rows: list[dict[str, str]], output: Path) -> None:
    if not site_rows:
        return
    ordered = sorted(site_rows, key=lambda row: float(row["balanced_accuracy"]))
    y = np.arange(len(ordered))
    estimate = np.array([float(row["balanced_accuracy"]) for row in ordered])
    low = np.array([float(row["balanced_accuracy_ci_low"]) for row in ordered])
    high = np.array([float(row["balanced_accuracy_ci_high"]) for row in ordered])
    fig, axis = plt.subplots(figsize=(5.4, max(3.0, 0.3 * len(ordered) + 1.2)), dpi=300)
    axis.errorbar(estimate, y, xerr=[estimate - low, high - estimate], fmt="o", color="#0F6B67", capsize=3)
    axis.set_yticks(y, [row["site_code"] for row in ordered])
    axis.set_xlim(0, 1)
    axis.set_xlabel("Balanced accuracy (95% CI)")
    axis.set_title("Descriptive site heterogeneity", loc="left", fontweight="bold")
    axis.grid(axis="x", color="#DDE3E6", linewidth=0.6)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    fig.tight_layout()
    save(fig, output / "Figure_5_site_heterogeneity")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    evaluation = args.evaluation_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    configure_style()
    metrics = read_csv(evaluation / "cohort_metrics.csv")
    if not metrics:
        raise FileNotFoundError(evaluation / "cohort_metrics.csv")
    roc = read_csv(evaluation / "prospective_roc.csv")
    pr = read_csv(evaluation / "prospective_pr.csv")
    figure3(metrics, output)
    if roc and pr:
        figure4(metrics, roc, pr, output)
    site_rows = read_csv(evaluation / "site_metrics.csv")
    figure5(site_rows, output)
    for source, target in (
        (evaluation / "cohort_metrics.csv", output / "Table_cohort_metrics.csv"),
        (evaluation / "site_metrics.csv", output / "Table_site_metrics.csv"),
    ):
        if source.is_file():
            shutil.copy2(source, target)
    print(output)


if __name__ == "__main__":
    main()
