"""Regenerate only Abaqus records produced with the unintended 141 GPa E1.

The script classifies each stored reference strain against 130 and 141 GPa
linear-CLT predictions.  It writes replacement Abaqus rows to a separate,
resumable CSV.  With --apply, it atomically replaces the source datasets only
after every flagged case has regenerated, then rebuilds their comparison CSVs.
"""

import argparse
import csv
import importlib.util
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
ABAQUS_EXECUTABLE = "abaqus.bat"
STRAIN_COLUMNS = ("e0_11", "e0_22", "e0_12")

DATASETS = {
    "abaqus": {
        "baseline": ROOT / "AbaqusSweep" / "ABaqusData.csv",
        "comparison": ROOT / "AbaqusSweep" / "ABaqusData_with_computational_results.csv",
        "abaqus_script": ROOT / "AbaqusSweep" / "ProperlyRotatedFREE.py",
        "comparison_script": ROOT / "Computational Real" / "compare_abaqus_to_computational.py",
        "repair_output": ROOT / "AbaqusSweep" / "ABaqusData_130GPa_regenerated.csv",
        "work_dir": ROOT / "AbaqusSweep" / "material_130_repair_work",
        "job_prefix": "Material130Main",
        "midplane_temperature": 50.0,
        "thermal_gradient": -53333.3,
    },
    "layer": {
        "baseline": ROOT / "LayupSweep" / "LayerEffect.csv",
        "comparison": ROOT / "LayupSweep" / "LayerEffect_with_computational_results.csv",
        "abaqus_script": ROOT / "LayupSweep" / "ProperlyRotatedFREELIVE.py",
        "comparison_script": ROOT / "LayupSweep" / "compare_abaqus_to_computational.py",
        "repair_output": ROOT / "LayupSweep" / "LayerEffect_130GPa_regenerated.csv",
        "work_dir": ROOT / "LayupSweep" / "material_130_repair_work",
        "job_prefix": "Material130Layer",
        "midplane_temperature": 50.0,
        "thermal_gradient": -53333.3,
    },
}


def load_freefree():
    path = ROOT / "Computational Real" / "FREEFREE.py"
    spec = importlib.util.spec_from_file_location("material_audit_freefree", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load %s" % path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FREEFREE = load_freefree()


def set_fiber_modulus(e1):
    FREEFREE.E1 = e1
    FREEFREE.nu21 = FREEFREE.nu12 * FREEFREE.E2 / FREEFREE.E1
    denominator = 1.0 - FREEFREE.nu12 * FREEFREE.nu21
    FREEFREE.Q11 = FREEFREE.E1 / denominator
    FREEFREE.Q22 = FREEFREE.E2 / denominator
    FREEFREE.Q12 = FREEFREE.nu12 * FREEFREE.E2 / denominator
    FREEFREE.Q66 = FREEFREE.G12
    FREEFREE.Q = np.array(
        [[FREEFREE.Q11, FREEFREE.Q12, 0.0],
         [FREEFREE.Q12, FREEFREE.Q22, 0.0],
         [0.0, 0.0, FREEFREE.Q66]],
        dtype=float,
    )


def parse_layup(layup_text):
    return [float(angle) for angle in layup_text.strip()[1:-1].split("/")]


def raw_key(row):
    return row["Layup"], int(row["NumberOfPlies"])


def strain_residual(row, e1):
    set_fiber_modulus(e1)
    FREEFREE.configure_case(
        parse_layup(row["Layup"]),
        float(row["TopSinkTemperature"]),
        float(row["BottomSinkTemperature"]),
    )
    clt_strain = np.linalg.solve(
        FREEFREE.ABD, np.concatenate((FREEFREE.N_T, FREEFREE.M_T))
    )[:3]
    abaqus_strain = np.array([float(row[name]) for name in STRAIN_COLUMNS])
    return float(np.linalg.norm(abaqus_strain - clt_strain))


def classify_rows(baseline):
    with baseline.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise RuntimeError("No data rows found in %s" % baseline)

    mismatched = []
    counts = {"130": 0, "141": 0, "insensitive": 0}
    for row in rows:
        error_130 = strain_residual(row, 130e9)
        error_141 = strain_residual(row, 141e9)
        if abs(error_130 - error_141) < 1.0e-10:
            counts["insensitive"] += 1
        elif error_141 < error_130:
            counts["141"] += 1
            mismatched.append(row)
        else:
            counts["130"] += 1
    return rows, mismatched, counts


def read_csv_rows(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def run_one_case(config, row, index):
    """Run one Abaqus case in an isolated folder and return its output row."""
    layup = row["Layup"]
    job_name = "%s_%04d" % (config["job_prefix"], index)
    case_dir = config["work_dir"] / job_name
    case_dir.mkdir(parents=True, exist_ok=True)
    case_output = case_dir / "summary.csv"
    if case_output.exists():
        case_output.unlink()

    environment = os.environ.copy()
    environment["ABAQUS_LAYUP"] = repr(parse_layup(layup))
    environment["ABAQUS_JOB_NAME"] = job_name
    environment["ABAQUS_SUMMARY_CSV"] = str(case_output)
    environment["ABAQUS_MIDPLANE_TEMPERATURE"] = str(config["midplane_temperature"])
    environment["ABAQUS_THERMAL_GRADIENT"] = str(config["thermal_gradient"])
    command = [
        ABAQUS_EXECUTABLE,
        "cae",
        "noGUI=" + str(config["abaqus_script"]),
    ]
    result = subprocess.run(command, cwd=str(case_dir), env=environment)
    if result.returncode != 0:
        raise RuntimeError("Abaqus failed for %s with code %s" % (layup, result.returncode))
    rows = read_csv_rows(case_output)
    if len(rows) != 1 or raw_key(rows[0]) != raw_key(row):
        raise RuntimeError("Unexpected Abaqus output for %s in %s" % (layup, case_output))
    return rows[0]


def append_output_row(path, row):
    """Append from the parent process only, avoiding concurrent CSV writes."""
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_cases(config, targets, workers=1):
    output_rows = read_csv_rows(config["repair_output"])
    completed = {raw_key(row) for row in output_rows}
    target_keys = {raw_key(row) for row in targets}
    unexpected = completed - target_keys
    if unexpected:
        raise RuntimeError(
            "Repair output %s contains unexpected cases: %s"
            % (config["repair_output"], sorted(unexpected)[:3])
        )

    pending = [row for row in targets if raw_key(row) not in completed]
    print("%s: %d corrected rows present; %d cases pending" % (
        config["baseline"].name, len(completed), len(pending)
    ))
    config["work_dir"].mkdir(parents=True, exist_ok=True)

    if workers < 1:
        raise ValueError("workers must be at least 1")
    numbered_pending = list(enumerate(pending, start=len(completed) + 1))
    print("Launching up to %d independent Abaqus workers" % workers, flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_one_case, config, row, index): (index, row)
            for index, row in numbered_pending
        }
        for future in as_completed(futures):
            index, row = futures[future]
            output_row = future.result()
            append_output_row(config["repair_output"], output_row)
            print("[%d/%d complete] %s" % (index, len(targets), row["Layup"]), flush=True)


def write_merged_baseline(config, original_rows, targets):
    repaired_rows = read_csv_rows(config["repair_output"])
    target_keys = {raw_key(row) for row in targets}
    repaired_by_key = {raw_key(row): row for row in repaired_rows}
    if set(repaired_by_key) != target_keys:
        missing = target_keys - set(repaired_by_key)
        unexpected = set(repaired_by_key) - target_keys
        raise RuntimeError(
            "Repair output is incomplete (missing=%d, unexpected=%d)"
            % (len(missing), len(unexpected))
        )

    baseline = config["baseline"]
    backup = baseline.with_name(baseline.stem + "_pre_130GPa_repair" + baseline.suffix)
    if not backup.exists():
        shutil.copy2(baseline, backup)

    fieldnames = list(original_rows[0].keys())
    merged_rows = [repaired_by_key.get(raw_key(row), row) for row in original_rows]
    temporary = baseline.with_name(baseline.name + ".material_130.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)
    os.replace(str(temporary), str(baseline))
    print("Updated %s (backup: %s)" % (baseline, backup))


def rebuild_comparison(config):
    command = [
        sys.executable,
        str(config["comparison_script"]),
        "--input-csv", str(config["baseline"]),
        "--output-csv", str(config["comparison"]),
    ]
    print("Rebuilding %s" % config["comparison"], flush=True)
    result = subprocess.run(command, cwd=str(config["baseline"].parent))
    if result.returncode != 0:
        raise RuntimeError("Comparison rebuild failed for %s" % config["baseline"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("abaqus", "layer", "all"), default="all"
    )
    parser.add_argument("--run", action="store_true", help="Run pending Abaqus repairs.")
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Maximum independent Abaqus cases to run concurrently (default: 1).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Replace repaired baseline rows and rebuild the comparison data.",
    )
    args = parser.parse_args()
    if args.apply and not args.run:
        parser.error("--apply requires --run so completeness can be verified.")

    selected = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}
    for name, config in selected.items():
        rows, targets, counts = classify_rows(config["baseline"])
        print(
            "%s audit: 130 GPa=%d, 141 GPa=%d, insensitive=%d"
            % (name, counts["130"], counts["141"], counts["insensitive"])
        )
        if args.run:
            run_cases(config, targets, workers=args.workers)
        if args.apply:
            write_merged_baseline(config, rows, targets)
            rebuild_comparison(config)
            _, remaining, final_counts = classify_rows(config["baseline"])
            if remaining:
                raise RuntimeError("Material audit still finds %d 141 GPa rows" % len(remaining))
            print("%s verified: %s" % (name, final_counts))


if __name__ == "__main__":
    main()
