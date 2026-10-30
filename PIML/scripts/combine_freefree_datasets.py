"""Combine compatible FREEFREE CSV datasets, rejecting duplicate conditions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Combine FREEFREE label CSVs.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-duplicates", action="store_true", help="Keep the first occurrence of repeated layup/field conditions.")
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip malformed rows that do not contain a complete FREEFREE condition.",
    )
    args = parser.parse_args(argv)
    fields: list[str] | None = None
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    skipped_invalid = 0
    for path in args.inputs:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"{path} has no header.")
            if fields is None:
                fields = reader.fieldnames
            elif reader.fieldnames != fields:
                raise ValueError(f"{path} has incompatible columns.")
            for row in reader:
                required = ("layup", "nplies", "top_face_temperature", "bottom_face_temperature", "kappa_x", "kappa_y", "kappa_xy", "converged")
                if any(not row.get(name) for name in required):
                    if args.skip_invalid:
                        skipped_invalid += 1
                        continue
                    raise ValueError(f"Malformed FREEFREE row in {path}: {row}.")
                key = (row["layup"], row["top_face_temperature"], row["bottom_face_temperature"])
                if key in seen:
                    if args.skip_duplicates:
                        continue
                    raise ValueError(f"Duplicate condition {key}.")
                seen.add(key); rows.append(row)
    assert fields is not None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(
        f"Combined {len(rows)} FREEFREE cases: {args.output} "
        f"(skipped {skipped_invalid} invalid row(s))."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
