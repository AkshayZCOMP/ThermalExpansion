"""Stable, timestamped paths for generated PIML artifacts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def make_run_id() -> str:
    """Return a sortable local-time run identifier, including timezone offset."""
    return datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")


def timestamped_file(path: Path, run_id: str | None) -> Path:
    """Insert ``run_id`` before a file extension when one is supplied."""
    if not run_id:
        return path
    return path.with_name(f"{path.stem}_{run_id}{path.suffix}")


def timestamped_prefix(prefix: Path, run_id: str | None) -> Path:
    """Append ``run_id`` to a multi-file output prefix when one is supplied."""
    if not run_id:
        return prefix
    return prefix.with_name(f"{prefix.name}_{run_id}")


def add_run_id_arguments(parser) -> None:
    """Add consistent timestamp controls to an artifact-producing CLI parser."""
    parser.add_argument(
        "--run-id",
        help="Identifier appended to generated output names; defaults to a sortable local timestamp.",
    )
    parser.add_argument(
        "--no-timestamp-output",
        action="store_true",
        help="Keep the exact output name instead of appending a timestamp.",
    )


def resolve_run_id(args) -> str | None:
    """Return an explicit/generated run ID unless exact output naming was requested."""
    if args.no_timestamp_output:
        return None
    return args.run_id or make_run_id()
