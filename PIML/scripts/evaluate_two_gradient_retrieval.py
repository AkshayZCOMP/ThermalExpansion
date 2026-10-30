"""Evaluate summed candidate scores from paired low/high thermal gradients."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from .train_direct_residual_nn import ResidualCurvatureNN
from ..core.physics import FVKPhysics, all_sweep_layups, make_batch, parse_layup


def load_model(path: Path) -> ResidualCurvatureNN:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if bool(saved.get("include_abd_features", False)):
        raise ValueError("This evaluator currently supports the standard residual model only.")
    model = ResidualCurvatureNN(
        int(saved["width"]), int(saved["layers"]),
        nplies=int(saved.get("nplies", 8)),
        split_offaxis=bool(saved.get("split_offaxis_experts", False)),
    ).double()
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    return model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rank layups from paired low/high-gradient curvature measurements.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args(argv)
    model = load_model(args.checkpoint)
    physics = FVKPhysics().double()
    candidates = [layup for layup in all_sweep_layups() if len(layup) == model.nplies]
    scales = torch.tensor((0.01, 0.01, 0.02), dtype=torch.float64)
    with args.test_data.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("converged", "True").lower() == "true"]
    groups: dict[tuple[str, float, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        top, bottom = float(row["top_face_temperature"]), float(row["bottom_face_temperature"])
        groups[(row["layup"], round(0.5 * (top + bottom), 8), 1 if top > bottom else -1)].append(row)
    result_rows: list[dict[str, object]] = []
    with torch.no_grad():
        for (layup, midplane, sign), group in groups.items():
            if len(group) != 2:
                raise ValueError(f"Expected one low/high gradient pair for {(layup, midplane, sign)}, found {len(group)} rows.")
            total = torch.zeros(len(candidates), dtype=torch.float64)
            for row in sorted(group, key=lambda item: abs(float(item["top_face_temperature"]) - float(item["bottom_face_temperature"]))):
                top, bottom = float(row["top_face_temperature"]), float(row["bottom_face_temperature"])
                batch = make_batch(candidates, [top] * len(candidates), [bottom] * len(candidates))
                linear = physics.curvatures(physics.linear_initial_q(batch))
                prediction = linear + model(batch, linear)
                target = torch.tensor([float(row[key]) for key in ("kappa_x", "kappa_y", "kappa_xy")], dtype=torch.float64)
                total += ((prediction - target) / scales).square().sum(dim=1)
            truth = candidates.index(tuple(parse_layup(layup)))
            rank = int((total < total[truth]).sum()) + 1
            result_rows.append({"layup": layup, "midplane_temperature": midplane, "gradient_sign": sign, "true_rank": rank})
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_name(args.output_prefix.name + "_ranks.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0])); writer.writeheader(); writer.writerows(result_rows)
    ranks = [int(row["true_rank"]) for row in result_rows]
    summary = {"paired_queries": len(ranks), "top1_count": sum(rank == 1 for rank in ranks), "top1_fraction": sum(rank == 1 for rank in ranks) / len(ranks), "top5_count": sum(rank <= 5 for rank in ranks), "top5_fraction": sum(rank <= 5 for rank in ranks) / len(ranks), "worst_rank": max(ranks), "rank_counts": dict(sorted(Counter(ranks).items()))}
    summary_path = args.output_prefix.with_name(args.output_prefix.name + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
