"""Create reproducible layup-level train/holdout CSVs from a FREEFREE dataset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch


def symmetry_orbit_key(layup_text: str) -> str:
    """Canonical key for the square-plate rotation/reflection layup orbit."""
    angles = tuple(int(value) for value in layup_text.strip("[]").split("/"))
    rotate = {-45: 45, 0: 90, 45: -45, 90: 0}
    mirror = {-45: 45, 0: 0, 45: -45, 90: 90}
    reflected = tuple(mirror[value] for value in angles)
    rotated = tuple(rotate[value] for value in angles)
    rotated_reflected = tuple(rotate[value] for value in reflected)
    return min("[" + "/".join(map(str, item)) + "]" for item in (angles, reflected, rotated, rotated_reflected))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Split a FREEFREE CSV into disjoint layup train and holdout datasets.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--holdout-output", type=Path, required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=313)
    parser.add_argument(
        "--group-by", choices=("case", "layup", "symmetry-orbit"), default="layup",
        help="Split individual rows, exact layups, or complete square-plate symmetry orbits.",
    )
    args = parser.parse_args(argv)
    if not 0.0 < args.holdout_fraction < 0.5:
        raise ValueError("--holdout-fraction must lie in (0, 0.5).")
    with args.input.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "layup" not in reader.fieldnames:
            raise ValueError("Input must be a FREEFREE CSV with a layup column.")
        fields, rows = reader.fieldnames, list(reader)
    # A temperature/gradient grid has repeated layups.  Split *layup groups*,
    # never individual rows, so every thermal condition for a held-out layup
    # remains unseen by the curvature-label loss.
    group_for_row = [
        str(index) if args.group_by == "case"
        else row["layup"] if args.group_by == "layup"
        else symmetry_orbit_key(row["layup"])
        for index, row in enumerate(rows)
    ]
    groups = sorted(set(group_for_row))
    if len(groups) < 2:
        raise ValueError("At least two distinct layups are required for a split.")
    holdout_count = max(1, round(len(groups) * args.holdout_fraction))
    permutation = torch.randperm(len(groups), generator=torch.Generator().manual_seed(args.seed)).tolist()
    holdout_groups = {groups[index] for index in permutation[:holdout_count]}
    train_rows = [row for row, group in zip(rows, group_for_row) if group not in holdout_groups]
    holdout_rows = [row for row, group in zip(rows, group_for_row) if group in holdout_groups]
    for path, subset in ((args.train_output, train_rows), (args.holdout_output, holdout_rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(subset)
    print(
        f"Created {len(train_rows)} train and {len(holdout_rows)} held-out records "
        f"from {len(groups) - len(holdout_groups)} train and {len(holdout_groups)} held-out {args.group_by} groups."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
