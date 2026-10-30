"""
Generate Abaqus-vs-computational error plots grouped by ply count.

Input:
    ABaqusData_with_computational_results.csv

Outputs:
    abaqus_error_l2_by_ply_count.png
    abaqus_component_abs_error_by_ply_count.png
    abaqus_error_by_ply_count_summary.csv
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = SCRIPT_DIR / "ABaqusData_with_computational_results.csv"

L2_PLOT = SCRIPT_DIR / "abaqus_error_l2_by_ply_count.png"
COMPONENT_PLOT = SCRIPT_DIR / "abaqus_component_abs_error_by_ply_count.png"
SUMMARY_CSV = SCRIPT_DIR / "abaqus_error_by_ply_count_summary.csv"

STRAIN_COMPONENTS = ("e0_11", "e0_22", "e0_12")
CURVATURE_COMPONENTS = ("K_11", "K_22", "K_12")
ALL_COMPONENTS = STRAIN_COMPONENTS + CURVATURE_COMPONENTS


def load_rows(input_csv=INPUT_CSV):
    """Load augmented CSV rows and group them by NumberOfPlies."""
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Missing {input_csv}. Run compare_abaqus_to_computational.py first."
        )

    groups = defaultdict(list)
    with input_csv.open(newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            ply_count = int(row["NumberOfPlies"])
            groups[ply_count].append(row)

    return dict(sorted(groups.items()))


def numeric_column(rows, column):
    """Return one numeric CSV column as a NumPy array."""
    return np.array([float(row[column]) for row in rows], dtype=float)


def grouped_column(groups, column, scale=1.0):
    """Return labels and grouped numeric arrays for one CSV column."""
    ply_counts = list(groups)
    values = [numeric_column(groups[ply_count], column) * scale for ply_count in ply_counts]
    return ply_counts, values


def boxplot_grouped(ax, ply_counts, values, title, ylabel):
    """Draw a consistent by-ply-count boxplot."""
    labels = [str(ply_count) for ply_count in ply_counts]
    try:
        ax.boxplot(values, tick_labels=labels, showfliers=True)
    except TypeError:
        ax.boxplot(values, labels=labels, showfliers=True)

    ax.set_title(title)
    ax.set_xlabel("Number of plies")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)


def plot_l2_errors(groups, output_path=L2_PLOT):
    """Plot row-level strain, curvature, and total response L2 errors."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    ply_counts, strain_l2 = grouped_column(groups, "StrainL2Error", scale=1e6)
    boxplot_grouped(
        axes[0],
        ply_counts,
        strain_l2,
        "Strain L2 Error",
        "microstrain",
    )

    ply_counts, curvature_l2 = grouped_column(groups, "CurvatureL2Error")
    boxplot_grouped(
        axes[1],
        ply_counts,
        curvature_l2,
        "Curvature L2 Error",
        "1/m",
    )

    ply_counts, response_l2 = grouped_column(groups, "ResponseL2Error")
    boxplot_grouped(
        axes[2],
        ply_counts,
        response_l2,
        "Total Response L2 Error",
        "mixed response units",
    )

    fig.suptitle("Abaqus vs Computational Model Error by Ply Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_component_abs_errors(groups, output_path=COMPONENT_PLOT):
    """Plot component-wise absolute errors grouped by ply count."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)

    for component_index, component in enumerate(STRAIN_COMPONENTS):
        ply_counts, values = grouped_column(
            groups,
            f"AbsError_{component}",
            scale=1e6,
        )
        boxplot_grouped(
            axes[0, component_index],
            ply_counts,
            values,
            f"{component} Abs Error",
            "microstrain",
        )

    for component_index, component in enumerate(CURVATURE_COMPONENTS):
        ply_counts, values = grouped_column(groups, f"AbsError_{component}")
        boxplot_grouped(
            axes[1, component_index],
            ply_counts,
            values,
            f"{component} Abs Error",
            "1/m",
        )

    fig.suptitle("Component Absolute Error by Ply Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def summarize_by_ply_count(groups, output_csv=SUMMARY_CSV):
    """Write mean/max error summaries by ply count."""
    fieldnames = [
        "NumberOfPlies",
        "Cases",
        "MeanStrainL2Error_microstrain",
        "MaxStrainL2Error_microstrain",
        "MeanCurvatureL2Error",
        "MaxCurvatureL2Error",
        "MeanResponseL2Error",
        "MaxResponseL2Error",
    ]

    for component in ALL_COMPONENTS:
        suffix = "_microstrain" if component in STRAIN_COMPONENTS else ""
        fieldnames.extend(
            [
                f"MeanAbsError_{component}{suffix}",
                f"MaxAbsError_{component}{suffix}",
            ]
        )

    rows = []
    for ply_count, group_rows in groups.items():
        strain_l2 = numeric_column(group_rows, "StrainL2Error") * 1e6
        curvature_l2 = numeric_column(group_rows, "CurvatureL2Error")
        response_l2 = numeric_column(group_rows, "ResponseL2Error")

        summary_row = {
            "NumberOfPlies": ply_count,
            "Cases": len(group_rows),
            "MeanStrainL2Error_microstrain": np.mean(strain_l2),
            "MaxStrainL2Error_microstrain": np.max(strain_l2),
            "MeanCurvatureL2Error": np.mean(curvature_l2),
            "MaxCurvatureL2Error": np.max(curvature_l2),
            "MeanResponseL2Error": np.mean(response_l2),
            "MaxResponseL2Error": np.max(response_l2),
        }

        for component in ALL_COMPONENTS:
            scale = 1e6 if component in STRAIN_COMPONENTS else 1.0
            values = numeric_column(group_rows, f"AbsError_{component}") * scale
            suffix = "_microstrain" if component in STRAIN_COMPONENTS else ""
            summary_row[f"MeanAbsError_{component}{suffix}"] = np.mean(values)
            summary_row[f"MaxAbsError_{component}{suffix}"] = np.max(values)

        rows.append(summary_row)

    with output_csv.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        value
                        if isinstance(value, int)
                        else f"{float(value):.8e}"
                    )
                    for key, value in row.items()
                }
            )

    return output_csv, rows


def generate_plots():
    """Generate all plots and summary data."""
    groups = load_rows()
    l2_plot = plot_l2_errors(groups)
    component_plot = plot_component_abs_errors(groups)
    summary_csv, summary_rows = summarize_by_ply_count(groups)

    print("=" * 80)
    print("ERROR-BY-PLY-COUNT PLOTS GENERATED")
    print("=" * 80)
    print(f"Input: {INPUT_CSV}")
    print(f"L2 plot: {l2_plot}")
    print(f"Component plot: {component_plot}")
    print(f"Summary CSV: {summary_csv}")
    print("\nCases by ply count:")
    for row in summary_rows:
        print(f"  {row['NumberOfPlies']} plies: {row['Cases']} cases")
    print("=" * 80)


if __name__ == "__main__":
    generate_plots()
