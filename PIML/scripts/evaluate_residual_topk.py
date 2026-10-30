"""Rank canonical layups with a direct residual-curvature surrogate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from .train_direct_residual_nn import ResidualCurvatureNN
from ..core.physics import FVKPhysics, all_sweep_layups, layup_string, make_batch, parse_layup


COMPONENTS = ("kappa_x", "kappa_y", "kappa_xy")


def parse_scales(text: str) -> torch.Tensor:
    values = tuple(float(value.strip()) for value in text.split(",") if value.strip())
    if len(values) != 3 or any(value <= 0.0 for value in values):
        raise ValueError("--component-scales must specify three positive values.")
    return torch.tensor(values, dtype=torch.float64)


def d4_average_prediction(
    models: list[ResidualCurvatureNN], physics: FVKPhysics,
    candidates: list[tuple[int, ...]], top: float, bottom: float,
) -> torch.Tensor:
    """Return the exact D4 group-averaged prediction in the original frame.

    Square plates are invariant to a reflection and a 90-degree rotation.  We
    evaluate all four in-plane representations of each candidate, transform
    their predicted curvature tensors back to the original axes, then average.
    This does not create new information; it removes small symmetry violations
    learned from finite data and optimization.
    """
    mirror = {-45: 45, 0: 0, 45: -45, 90: 90}
    rotate = {-45: 45, 0: 90, 45: -45, 90: 0}
    transformed_candidates = (
        candidates,
        [tuple(mirror[angle] for angle in layup) for layup in candidates],
        [tuple(rotate[angle] for angle in layup) for layup in candidates],
        [tuple(rotate[mirror[angle]] for angle in layup) for layup in candidates],
    )
    predictions: list[torch.Tensor] = []
    for transform_index, transformed in enumerate(transformed_candidates):
        batch = make_batch(transformed, [top] * len(transformed), [bottom] * len(transformed))
        linear = physics.curvatures(physics.linear_initial_q(batch))
        abd, thermal = physics.laminate_matrices(batch)
        linear_strain = torch.linalg.solve(abd, thermal.unsqueeze(-1)).squeeze(-1)[:, :3]
        prediction = linear + torch.stack(
            [model(batch, linear, abd, linear_strain) for model in models]
        ).mean(dim=0)
        if transform_index == 1:
            prediction[:, 2] *= -1.0
        elif transform_index == 2:
            prediction = prediction[:, [1, 0, 2]]
            prediction[:, 2] *= -1.0
        elif transform_index == 3:
            prediction = prediction[:, [1, 0, 2]]
        predictions.append(prediction)
    return torch.stack(predictions).mean(dim=0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate top-k exact layup recovery using forward residual ranking.")
    parser.add_argument(
        "--checkpoint", type=Path, action="append", required=True,
        help="Checkpoint to include; repeat to average a seed ensemble.",
    )
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--nplies", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--component-scales", default="0.01,0.01,0.02")
    parser.add_argument(
        "--d4-average", action="store_true",
        help="Average predictions over exact square-plate reflection/rotation symmetries.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--summary-output", type=Path,
        help="Optional JSON file for curvature error metrics at each true candidate.",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.nplies <= 10 or args.top_k < 1:
        raise ValueError("--nplies must be in [1, 10] and --top-k must be positive.")

    models: list[ResidualCurvatureNN] = []
    for checkpoint in args.checkpoint:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        checkpoint_nplies = int(saved.get("nplies", 8))
        if checkpoint_nplies != args.nplies:
            raise ValueError(f"Checkpoint {checkpoint} is {checkpoint_nplies}-ply, not {args.nplies}-ply.")
        if saved.get("residual_components", "all") != "all":
            raise ValueError("Top-k ranking requires a model trained on all three curvature residual components.")
        model = ResidualCurvatureNN(
            int(saved["width"]), int(saved["layers"]), nplies=args.nplies,
            split_offaxis=bool(saved.get("split_offaxis_experts", False)),
            include_abd_features=bool(saved.get("include_abd_features", False)),
            include_linear_strain_features=bool(saved.get("include_linear_strain_features", False)),
        ).double()
        model.load_state_dict(saved["model_state_dict"])
        model.eval()
        models.append(model)

    candidates = [layup for layup in all_sweep_layups() if len(layup) == args.nplies]
    if not candidates:
        raise ValueError(f"The canonical sweep has no {args.nplies}-ply candidates.")
    scales = parse_scales(args.component_scales)
    physics = FVKPhysics().double()
    with args.test_data.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("converged", "True").lower() == "true"]
    if not rows:
        raise ValueError("No converged test records found.")

    output_rows: list[dict[str, object]] = []
    ranks: list[int] = []
    references: list[torch.Tensor] = []
    truth_predictions: list[torch.Tensor] = []
    with torch.no_grad():
        for row in rows:
            truth = tuple(parse_layup(row["layup"]))
            if len(truth) != args.nplies or truth not in candidates:
                raise ValueError(f"Test layup {row['layup']} is not a canonical {args.nplies}-ply candidate.")
            top = float(row["top_face_temperature"])
            bottom = float(row["bottom_face_temperature"])
            if args.d4_average:
                predicted = d4_average_prediction(models, physics, candidates, top, bottom)
            else:
                batch = make_batch(candidates, [top] * len(candidates), [bottom] * len(candidates))
                linear = physics.curvatures(physics.linear_initial_q(batch))
                abd, thermal = physics.laminate_matrices(batch)
                linear_strain = torch.linalg.solve(abd, thermal.unsqueeze(-1)).squeeze(-1)[:, :3]
                predicted = linear + torch.stack([model(batch, linear, abd, linear_strain) for model in models]).mean(dim=0)
            target = torch.tensor([float(row[component]) for component in COMPONENTS], dtype=torch.float64)
            score = ((predicted - target) / scales).square().sum(dim=1)
            truth_index = candidates.index(truth)
            truth_prediction = predicted[truth_index]
            rank = int((score < score[truth_index]).sum()) + 1
            order = torch.argsort(score).tolist()
            ranks.append(rank)
            result: dict[str, object] = {
                "target_layup": row["layup"], "top_face_temperature": top,
                "bottom_face_temperature": bottom, "true_rank": rank,
                "target_in_top_k": rank <= args.top_k,
            }
            for component_index, component in enumerate(COMPONENTS):
                result[f"reference_{component}"] = float(target[component_index])
                result[f"predicted_{component}"] = float(truth_prediction[component_index])
                result[f"error_{component}"] = float(truth_prediction[component_index] - target[component_index])
            result.update({f"rank_{index + 1}_layup": layup_string(candidates[candidate]) for index, candidate in enumerate(order[:args.top_k])})
            output_rows.append(result)
            references.append(target)
            truth_predictions.append(truth_prediction)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    top1 = sum(rank == 1 for rank in ranks) / len(ranks)
    topk = sum(rank <= args.top_k for rank in ranks) / len(ranks)
    reference_tensor = torch.stack(references)
    prediction_tensor = torch.stack(truth_predictions)
    error = prediction_tensor - reference_tensor
    metric_components: dict[str, dict[str, float | int]] = {}
    for component_index, component in enumerate(COMPONENTS):
        component_error = error[:, component_index]
        active = reference_tensor[:, component_index].abs() >= 0.02
        metric_components[component] = {
            "rmse_1_per_m": float(component_error.square().mean().sqrt()),
            "active_cases": int(active.sum()),
            "active_rmse_1_per_m": float(component_error[active].square().mean().sqrt()),
        }
    metric_summary = {
        "cases": len(ranks),
        "d4_average": bool(args.d4_average),
        "components": metric_components,
    }
    print(
        f"Evaluated {len(ranks)} cases against {len(candidates)} canonical {args.nplies}-ply layups "
        f"with {len(models)} model(s). "
        f"Top-1={top1:.3%}; top-{args.top_k}={topk:.3%}; worst rank={max(ranks)}."
    )
    print(
        "Kxy RMSE="
        f"{metric_components['kappa_xy']['rmse_1_per_m']:.6f} 1/m; active Kxy RMSE="
        f"{metric_components['kappa_xy']['active_rmse_1_per_m']:.6f} 1/m."
    )
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(metric_summary, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.summary_output}")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
