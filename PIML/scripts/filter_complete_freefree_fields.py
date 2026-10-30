"""Retain only fully populated thermal fields from an incrementally written FREEFREE CSV."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Filter a FREEFREE matrix to fields with every requested layup.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases-per-field", type=int, required=True)
    args = parser.parse_args(argv)
    if args.cases_per_field <= 0:
        raise ValueError("--cases-per-field must be positive.")
    with args.input.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header.")
        fields, rows = reader.fieldnames, list(reader)
    key = lambda row: (row["top_face_temperature"], row["bottom_face_temperature"])
    counts = Counter(key(row) for row in rows)
    complete = {field for field, count in counts.items() if count == args.cases_per_field}
    kept = [row for row in rows if key(row) in complete]
    if not kept:
        raise ValueError("No complete thermal fields found.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(kept)
    print(f"Retained {len(kept)} cases across {len(complete)} complete fields: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
