"""Evaluate a checkpoint against held-out FREEFREE or Abaqus test data.

The evaluator auto-detects either CSV produced by
``PIML.scripts.generate_fvk_dataset`` or the existing ``AbaqusSweep/ABaqusData.csv``.
Do not evaluate against labels that were used for optional fitting.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from ..core.physics import LayupBatch, layup_string, make_batch, parse_layup
from .predict import load_model
from ..core.reporting import COMPONENTS, write_evaluation_report
from ..core.runtime import select_device
from ..core.run_paths import add_run_id_arguments, resolve_run_id, timestamped_prefix


def load_test_cases(path: Path, *, include_nonconverged: bool = False) -> tuple[LayupBatch, list[dict[str, object]], str]:
    """Load converged FREEFREE or Abaqus rows, retaining source traceability."""
    rows: list[dict[str, object]] = []
    layups: list[list[int]] = []
    tops: list[float] = []
    bottoms: list[float] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header.")
        is_freefree = "layup" in reader.fieldnames
        is_abaqus = "Layup" in reader.fieldnames
        if not (is_freefree or is_abaqus):
            raise ValueError(f"{path} is not a recognized FREEFREE or Abaqus reference CSV.")
        source = "FREEFREE" if is_freefree else "Abaqus"
        for raw in reader:
            if is_freefree and not include_nonconverged and raw.get("converged", "True").strip().lower() != "true":
                continue
            layup = parse_layup(raw["layup"] if is_freefree else raw["Layup"])
            top = float(raw["top_face_temperature"] if is_freefree else raw["TopTemperature"])
            bottom = float(raw["bottom_face_temperature"] if is_freefree else raw["BottomTemperature"])
            layups.append(layup)
            tops.append(top)
            bottoms.append(bottom)
            if is_freefree:
                # Evaluation reports preserve their trusted reference values
                # under this prefix, allowing an existing frozen holdout to be
                # reused without copying or rewriting its source data.
                curvature_columns = (
                    ("kappa_x", "kappa_y", "kappa_xy")
                    if "kappa_x" in reader.fieldnames
                    else ("reference_kappa_x", "reference_kappa_y", "reference_kappa_xy")
                )
            else:
                curvature_columns = ("K_11", "K_22", "K_12")
            rows.append(
                {
                    "reference_source": source,
                    "layup": layup_string(layup),
                    "nplies": len(layup),
                    "top_face_temperature": top,
                    "bottom_face_temperature": bottom,
                    **{
                        f"reference_{name}": float(raw[column])
                        for name, column in zip(COMPONENTS, curvature_columns)
                    },
                    "reference_solver_iterations": int(raw.get("iterations", 0)) if is_freefree else "",
                    "reference_solver_message": raw.get("message", "") if is_freefree else "Abaqus reference",
                }
            )
    if not rows:
        raise ValueError(f"No converged test rows found in {path}.")
    return make_batch(layups, tops, bottoms), rows, source


def batches(batch: LayupBatch, size: int):
    for start in range(0, len(batch), size):
        index = torch.arange(start, min(start + size, len(batch)), device=batch.nplies.device)
        yield start, batch.index_select(index)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare a PIML checkpoint to held-out nonlinear FREEFREE or Abaqus test cases.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--test-data", required=True, type=Path, help="FREEFREE generator CSV or AbaqusSweep/ABaqusData.csv, not used for fitting.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--include-nonconverged", action="store_true", help="Retain stored FREEFREE rows flagged unconverged.")
    parser.add_argument(
        "--near-zero-curvature-threshold",
        type=float,
        default=0.02,
        help="Report a separate behavior-focused score using only components with |reference kappa| at least this value [1/m].",
    )
    parser.add_argument("--output-prefix", type=Path, default=Path("PIML/artifacts/test"))
    add_run_id_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    args.output_prefix = timestamped_prefix(args.output_prefix, resolve_run_id(args))
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.near_zero_curvature_threshold < 0.0:
        raise ValueError("--near-zero-curvature-threshold cannot be negative.")
    device = select_device(args.device)
    physics, model = load_model(args.checkpoint, device)
    test_batch, rows, source = load_test_cases(args.test_data, include_nonconverged=args.include_nonconverged)
    test_batch = test_batch.to(device)

    model.eval()
    for start, batch in batches(test_batch, args.batch_size):
        with torch.enable_grad():
            q_pinn = model(batch)
            q_clt = physics.linear_initial_q(batch)
            pinn = physics.curvatures(q_pinn)
            clt = physics.curvatures(q_clt)
            energy_delta = physics.energy(q_pinn, batch) - physics.energy(q_clt, batch)
            projected_gradient = torch.linalg.vector_norm(physics.reduced_gradient(q_pinn, batch), dim=1)
        pinn_np = pinn.detach().cpu().numpy()
        clt_np = clt.detach().cpu().numpy()
        energy_np = energy_delta.detach().cpu().numpy()
        residual_np = projected_gradient.detach().cpu().numpy()
        for offset in range(len(batch)):
            row = rows[start + offset]
            pinn_error = []
            clt_error = []
            for component_index, component in enumerate(COMPONENTS):
                reference = float(row[f"reference_{component}"])
                p_value = float(pinn_np[offset, component_index])
                l_value = float(clt_np[offset, component_index])
                row[f"pinn_{component}"] = p_value
                row[f"linear_clt_{component}"] = l_value
                row[f"pinn_error_{component}"] = p_value - reference
                row[f"linear_clt_error_{component}"] = l_value - reference
                row[f"reference_{component}_active"] = abs(reference) >= args.near_zero_curvature_threshold
                pinn_error.append(p_value - reference)
                clt_error.append(l_value - reference)
            row["pinn_curvature_error_norm_1_per_m"] = float(np.linalg.norm(pinn_error))
            row["linear_clt_curvature_error_norm_1_per_m"] = float(np.linalg.norm(clt_error))
            row["pinn_energy_change_from_clt_j"] = float(energy_np[offset])
            row["pinn_projected_gradient_norm"] = float(residual_np[offset])

    csv_path, summary_path, png_path = write_evaluation_report(
        rows,
        args.output_prefix,
        reference_label=source,
        near_zero_curvature_threshold=args.near_zero_curvature_threshold,
    )
    print(f"Evaluated {len(rows)} {source} reference cases.")
    print(f"Predictions: {csv_path}")
    print(f"Metrics:     {summary_path}")
    print(f"Plots:       {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
