"""Monte-Carlo measurement-noise evaluation for forward layup retrieval."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from .evaluate_residual_topk import COMPONENTS, parse_scales
from .train_direct_residual_nn import ResidualCurvatureNN
from ..core.physics import FVKPhysics, all_sweep_layups, make_batch, parse_layup


def load_models(checkpoints: list[Path], nplies: int) -> list[ResidualCurvatureNN]:
    models: list[ResidualCurvatureNN] = []
    for checkpoint in checkpoints:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if int(saved.get("nplies", 8)) != nplies:
            raise ValueError(f"Checkpoint {checkpoint} does not match --nplies {nplies}.")
        if saved.get("residual_components", "all") != "all":
            raise ValueError("Noise ranking requires all three residual-curvature components.")
        model = ResidualCurvatureNN(
            int(saved["width"]), int(saved["layers"]), nplies=nplies,
            split_offaxis=bool(saved.get("split_offaxis_experts", False)),
        ).double()
        model.load_state_dict(saved["model_state_dict"])
        model.eval()
        models.append(model)
    return models


def rank_metrics(scores: torch.Tensor, truth_indices: torch.Tensor) -> tuple[float, float, int]:
    truth_scores = scores[torch.arange(len(scores)), truth_indices]
    ranks = 1 + (scores < truth_scores[:, None]).sum(dim=1)
    return (
        float((ranks == 1).double().mean()),
        float((ranks <= 5).double().mean()),
        int(ranks.max()),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate top-k forward retrieval after Gaussian curvature-measurement noise."
    )
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--nplies", type=int, required=True)
    parser.add_argument("--noise-std", type=float, required=True, help="Per-component Gaussian SD in 1/m.")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--component-scales", default="0.01,0.01,0.02")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.noise_std < 0 or args.trials < 1 or not 1 <= args.nplies <= 10:
        raise ValueError("Require nonnegative --noise-std, positive --trials, and --nplies in [1, 10].")

    models = load_models(args.checkpoint, args.nplies)
    candidates = [layup for layup in all_sweep_layups() if len(layup) == args.nplies]
    if not candidates:
        raise ValueError(f"No canonical {args.nplies}-ply candidates exist.")
    scales = parse_scales(args.component_scales)
    with args.test_data.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("converged", "True").lower() == "true"]
    if not rows:
        raise ValueError("No converged test rows were found.")

    physics = FVKPhysics().double()
    predictions: list[torch.Tensor] = []
    targets: list[list[float]] = []
    truth_indices: list[int] = []
    with torch.no_grad():
        for row in rows:
            truth = tuple(parse_layup(row["layup"]))
            if truth not in candidates:
                raise ValueError(f"Test layup {row['layup']} is not a canonical candidate.")
            top, bottom = float(row["top_face_temperature"]), float(row["bottom_face_temperature"])
            batch = make_batch(candidates, [top] * len(candidates), [bottom] * len(candidates))
            linear = physics.curvatures(physics.linear_initial_q(batch))
            residual = torch.stack([model(batch, linear) for model in models]).mean(dim=0)
            predictions.append(linear + residual)
            targets.append([float(row[component]) for component in COMPONENTS])
            truth_indices.append(candidates.index(truth))

    predicted = torch.stack(predictions)
    target = torch.tensor(targets, dtype=torch.float64)
    truth = torch.tensor(truth_indices, dtype=torch.long)
    baseline_scores = ((predicted - target[:, None, :]) / scales).square().sum(dim=2)
    baseline_top1, baseline_top5, baseline_worst_rank = rank_metrics(baseline_scores, truth)

    generator = torch.Generator().manual_seed(args.seed)
    trial_top1: list[float] = []
    trial_top5: list[float] = []
    trial_worst_ranks: list[int] = []
    for _ in range(args.trials):
        noisy_target = target + torch.randn(target.shape, dtype=torch.float64, generator=generator) * args.noise_std
        scores = ((predicted - noisy_target[:, None, :]) / scales).square().sum(dim=2)
        top1, top5, worst_rank = rank_metrics(scores, truth)
        trial_top1.append(top1)
        trial_top5.append(top5)
        trial_worst_ranks.append(worst_rank)

    def describe(values: list[float]) -> dict[str, float]:
        tensor = torch.tensor(values, dtype=torch.float64)
        return {
            "mean": float(tensor.mean()), "std": float(tensor.std(unbiased=False)),
            "min": float(tensor.min()), "max": float(tensor.max()),
        }

    summary = {
        "test_cases": len(rows),
        "nplies": args.nplies,
        "candidate_count": len(candidates),
        "checkpoints": [str(path) for path in args.checkpoint],
        "perturbation": {
            "distribution": "independent zero-mean Gaussian curvature noise",
            "noise_std_1_per_m_per_component": args.noise_std,
            "temperature_or_layup_perturbed": False,
            "trials": args.trials,
            "seed": args.seed,
        },
        "unperturbed": {
            "top1_fraction": baseline_top1,
            "top5_fraction": baseline_top5,
            "worst_true_rank": baseline_worst_rank,
        },
        "noisy_trials": {
            "top1_fraction": describe(trial_top1),
            "top5_fraction": describe(trial_top5),
            "worst_true_rank": {
                "mean": float(torch.tensor(trial_worst_ranks, dtype=torch.float64).mean()),
                "min": int(min(trial_worst_ranks)), "max": int(max(trial_worst_ranks)),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
