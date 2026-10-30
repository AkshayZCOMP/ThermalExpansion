"""CSV, metric, and plot helpers for PIML training and held-out testing."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

# Training and evaluation are normally launched from a terminal or CI runner;
# never require a GUI window just to write a report.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .physics import T_PLY


COMPONENTS = ("kappa_x", "kappa_y", "kappa_xy")


def _safe_symlog(axis) -> None:
    """Use a scale that preserves the sign of energy-difference losses."""
    axis.set_yscale("symlog", linthresh=1.0e-8)


def write_training_history(history: Sequence[Mapping[str, float | int]], output_prefix: Path) -> tuple[Path, Path]:
    """Write loss/residual history to CSV and a compact PNG diagnostics figure."""
    if not history:
        raise ValueError("Cannot write an empty training history.")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_name(output_prefix.name + "_history.csv")
    png_path = output_prefix.with_name(output_prefix.name + "_training.png")
    columns = [
        "step", "total_loss", "physics_loss", "stability_loss", "path_loss", "data_loss",
        "validation_energy_delta_j", "validation_projected_gradient",
        "learning_rate", "continuation_factor", "physics_weight", "fidelity_weight",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)

    steps = np.asarray([float(row["step"]) for row in history])
    total = np.asarray([float(row["total_loss"]) for row in history])
    physics = np.asarray([float(row["physics_loss"]) for row in history])
    stability = np.asarray([float(row.get("stability_loss", 0.0)) for row in history])
    path = np.asarray([float(row.get("path_loss", 0.0)) for row in history])
    data = np.asarray([float(row["data_loss"]) for row in history])
    val_energy = np.asarray([float(row["validation_energy_delta_j"]) for row in history])
    val_gradient = np.asarray([float(row["validation_projected_gradient"]) for row in history])

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(steps, total, label="total", linewidth=2)
    axes[0].plot(steps, physics, label="physics", alpha=0.8)
    if np.any(stability != 0.0):
        axes[0].plot(steps, stability, label="stability", alpha=0.8)
    if np.any(path != 0.0):
        axes[0].plot(steps, path, label="path", alpha=0.8)
    if np.any(data != 0.0):
        axes[0].plot(steps, data, label="auxiliary target", alpha=0.8)
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("optimization step")
    axes[0].set_ylabel("normalized loss")
    _safe_symlog(axes[0])
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    left = axes[1]
    left.plot(steps, val_energy, color="tab:purple", label="validation energy difference")
    left.set_title("Physics validation")
    left.set_xlabel("optimization step")
    left.set_ylabel("mean energy difference [J]", color="tab:purple")
    left.tick_params(axis="y", labelcolor="tab:purple")
    _safe_symlog(left)
    left.grid(True, alpha=0.25)
    right = left.twinx()
    right.plot(steps, np.maximum(val_gradient, np.finfo(float).tiny), color="tab:green", label="projected residual")
    right.set_ylabel("mean ||dU/dr||", color="tab:green")
    right.tick_params(axis="y", labelcolor="tab:green")
    right.set_yscale("log")

    figure.suptitle("PIML Deep-Ritz training diagnostics")
    figure.savefig(png_path, dpi=180)
    plt.close(figure)
    return csv_path, png_path


def _error_metrics(error: np.ndarray) -> dict[str, float]:
    """Return consistently named absolute-error metrics for one error vector."""
    return {
        "mae_1_per_m": float(np.mean(np.abs(error))),
        "rmse_1_per_m": float(np.sqrt(np.mean(np.square(error)))),
        "max_abs_error_1_per_m": float(np.max(np.abs(error))),
        "mean_signed_error_1_per_m": float(np.mean(error)),
    }


def summarize_errors(
    rows: Sequence[Mapping[str, object]],
    *,
    near_zero_curvature_threshold: float = 0.02,
) -> dict[str, object]:
    """Return full and behavior-focused errors for PIML and linear CLT.

    The behavior-focused metrics omit an individual component only when its
    reference magnitude is below ``near_zero_curvature_threshold``. Full
    metrics are retained so the omitted low-signal errors remain auditable.
    """
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation.")
    if near_zero_curvature_threshold < 0.0:
        raise ValueError("near_zero_curvature_threshold cannot be negative.")
    summary: dict[str, object] = {
        "cases": len(rows),
        "near_zero_curvature_threshold_1_per_m": near_zero_curvature_threshold,
        "components": {},
    }
    components: dict[str, dict[str, dict[str, float]]] = {}
    target_matrix: list[np.ndarray] = []
    pinn_error_matrix: list[np.ndarray] = []
    clt_error_matrix: list[np.ndarray] = []
    for component in COMPONENTS:
        target = np.asarray([float(row[f"reference_{component}"]) for row in rows])
        pinn = np.asarray([float(row[f"pinn_{component}"]) for row in rows])
        clt = np.asarray([float(row[f"linear_clt_{component}"]) for row in rows])
        active = np.abs(target) >= near_zero_curvature_threshold
        if not np.any(active):
            raise ValueError(f"No active {component} references at the selected near-zero threshold.")
        component_summary: dict[str, dict[str, float]] = {}
        for label, predicted in (("pinn", pinn), ("linear_clt", clt)):
            error = predicted - target
            component_summary[label] = _error_metrics(error)
            component_summary[label].update(
                {
                    "active_cases": int(np.count_nonzero(active)),
                    "excluded_near_zero_cases": int(np.count_nonzero(~active)),
                    **{f"active_{key}": value for key, value in _error_metrics(error[active]).items()},
                }
            )
        components[component] = component_summary
        target_matrix.append(target)
        pinn_error_matrix.append(pinn - target)
        clt_error_matrix.append(clt - target)
    summary["components"] = components
    target_values = np.column_stack(target_matrix)
    active_values = np.abs(target_values) >= near_zero_curvature_threshold
    pinn_error_values = np.column_stack(pinn_error_matrix)
    clt_error_values = np.column_stack(clt_error_matrix)
    behavior: dict[str, object] = {
        "active_component_values": int(np.count_nonzero(active_values)),
        "excluded_near_zero_component_values": int(np.count_nonzero(~active_values)),
    }
    for label, error_values in (("pinn", pinn_error_values), ("linear_clt", clt_error_values)):
        masked = error_values * active_values
        behavior[label] = {
            "masked_vector_rmse_1_per_m": float(np.sqrt(np.mean(np.sum(np.square(masked), axis=1)))),
            "active_component_rmse_1_per_m": float(np.sqrt(np.mean(np.square(error_values[active_values])))),
        }
    summary["behavior_focused"] = behavior
    return summary


def write_evaluation_report(
    rows: Sequence[Mapping[str, object]],
    output_prefix: Path,
    *,
    reference_label: str = "FREEFREE",
    near_zero_curvature_threshold: float = 0.02,
) -> tuple[Path, Path, Path]:
    """Write per-case predictions, JSON metrics, and a reference-comparison PNG."""
    if not rows:
        raise ValueError("Cannot write an empty evaluation report.")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_name(output_prefix.name + "_predictions.csv")
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.json")
    png_path = output_prefix.with_name(output_prefix.name + "_comparison.png")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            summarize_errors(rows, near_zero_curvature_threshold=near_zero_curvature_threshold),
            handle,
            indent=2,
        )

    figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    flat_axes = axes.flat
    for component, axis in zip(COMPONENTS, flat_axes):
        reference = np.asarray([float(row[f"reference_{component}"]) for row in rows])
        pinn = np.asarray([float(row[f"pinn_{component}"]) for row in rows])
        clt = np.asarray([float(row[f"linear_clt_{component}"]) for row in rows])
        bounds = np.array([reference.min(), reference.max(), pinn.min(), pinn.max(), clt.min(), clt.max()])
        lower, upper = float(bounds.min()), float(bounds.max())
        pad = max((upper - lower) * 0.05, 1.0e-8)
        axis.scatter(reference, clt, label="linear CLT", color="tab:gray", alpha=0.65, marker="x")
        axis.scatter(reference, pinn, label="PIML", color="tab:blue", alpha=0.8)
        axis.plot([lower - pad, upper + pad], [lower - pad, upper + pad], "k--", linewidth=1, label="perfect")
        axis.set_title(component)
        axis.set_xlabel(f"{reference_label} reference [1/m]")
        axis.set_ylabel("prediction [1/m]")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)

    axis = axes[1, 1]
    gradient = np.asarray(
        [
            (float(row["top_face_temperature"]) - float(row["bottom_face_temperature"]))
            / (int(row["nplies"]) * T_PLY)
            for row in rows
        ]
    )
    pinn_norm = np.asarray([float(row["pinn_curvature_error_norm_1_per_m"]) for row in rows])
    clt_norm = np.asarray([float(row["linear_clt_curvature_error_norm_1_per_m"]) for row in rows])
    axis.scatter(gradient, clt_norm, label="linear CLT", color="tab:gray", alpha=0.65, marker="x")
    axis.scatter(gradient, pinn_norm, label="PIML", color="tab:blue", alpha=0.8)
    axis.set_title("Vector curvature error vs thermal gradient")
    axis.set_xlabel("thermal gradient dT/dz [C/m]")
    axis.set_ylabel("||kappa prediction - reference|| [1/m]")
    axis.set_yscale("symlog", linthresh=1.0e-10)
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)

    figure.suptitle(f"{reference_label} curvature evaluation")
    figure.savefig(png_path, dpi=180)
    plt.close(figure)
    return csv_path, summary_path, png_path
