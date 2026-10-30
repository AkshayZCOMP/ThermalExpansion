"""Probe symmetric laminates for branch splitting and snap-through candidates.

The test ramps the imposed through-thickness thermal gradient and follows an
unperturbed continuation path plus equal-and-opposite ``w ~ X*Y`` twist seeds.
This directly tests the nonzero-K12 branch seen in the comparison results. It
is a screening diagnostic: paired branches, a sudden continuation jump, or a
negative projected stiffness (when requested) identifies cases that deserve a
more detailed displacement-controlled Abaqus study.
"""

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
FREEFREE_PATH = REPO_ROOT / "Computational Real" / "FREEFREE.py"
DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "bifurcation_snap_results.csv"

# Symmetric cross-ply cases whose computational result developed a nonzero K12
# while Abaqus remained on the zero-twist solution.
DEFAULT_LAYUPS = [
    "[0/90/90/0/0/90/90/0]",
    "[0/90/90/90/90/90/90/0]",
    "[90/0/0/0/0/0/0/90]",
    "[90/0/0/90/90/0/0/90]",
]


def load_freefree_solver():
    """Load the shared nonlinear model without requiring a package install."""
    spec = importlib.util.spec_from_file_location("freefree_bifurcation", FREEFREE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("Could not load FREEFREE.py from %s" % FREEFREE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FREEFREE = load_freefree_solver()


def parse_layup(text):
    """Parse the same bracket-and-slash layup format used by the sweep CSVs."""
    cleaned = text.strip().strip("[]").replace(",", "/")
    if not cleaned:
        raise ValueError("Layup cannot be empty.")
    try:
        return [float(value.strip()) for value in cleaned.split("/") if value.strip()]
    except ValueError as error:
        raise ValueError("Invalid layup %r" % text) from error


def layup_label(layup):
    """Format a layup consistently for console output and CSV rows."""
    return "[" + "/".join("%g" % angle for angle in layup) + "]"


def is_symmetric(layup, tolerance=1.0e-10):
    """Return whether the stacking sequence mirrors exactly about the midplane."""
    return all(
        abs(first - second) <= tolerance
        for first, second in zip(layup, reversed(layup))
    )


def twist_seed_index():
    """Return the quadratic ``X*Y`` transverse coefficient, which controls K12."""
    return FREEFREE.NU + FREEFREE.NV + 2


def solve_branch(layup, load_scale, branch, previous_q, args):
    """Solve a zero, positive-twist, or negative-twist continuation branch."""
    thermal_gradient = load_scale * args.thermal_gradient
    FREEFREE.configure_case(
        layup,
        args.top_sink_temperature,
        args.bottom_sink_temperature,
        transverse_degree=args.ritz_degree,
        midplane_temperature=args.midplane_temperature,
        thermal_gradient=thermal_gradient,
    )
    _, _, nullspace, _, linear_q = FREEFREE.build_projected_initial_state()
    q_start = linear_q.copy() if previous_q is None else previous_q.copy()

    seed_curvature = {
        "zero_twist": 0.0,
        "positive_twist": args.twist_seed_curvature,
        "negative_twist": -args.twist_seed_curvature,
    }[branch]
    # FREEFREE's quadratic coefficient obeys kappa_xy = -c_xy / (L * W).
    q_start[twist_seed_index()] -= seed_curvature * FREEFREE.L * FREEFREE.W
    q_start = nullspace @ (nullspace.T @ q_start)
    solver = FREEFREE.solve_equilibrium_bfgs_constrained(
        q_start,
        nullspace,
        verbose=False,
    )

    response = np.full(6, np.nan)
    center_w = np.nan
    if solver.converged:
        strain, curvature = FREEFREE.area_averaged_generalized_quantities(solver.q)
        response = np.concatenate((strain, curvature))
        center_w = float(FREEFREE.eval_fields(FREEFREE.L / 2.0, FREEFREE.W / 2.0, solver.q)[2])

    symmetry_error = float(
        np.linalg.norm(FREEFREE.B)
        / max(np.linalg.norm(FREEFREE.A) * FREEFREE.h, np.finfo(float).eps)
    )
    return {
        "q": solver.q if solver.converged else None,
        "response": response,
        "center_w": center_w,
        "energy": solver.energy,
        "gradient_norm": solver.gradient_norm,
        "relative_gradient_norm": solver.relative_gradient_norm,
        "solver_converged": solver.converged,
        "solver_iterations": solver.iterations,
        "solver_message": solver.message,
        "solver_method": solver.method,
        "symmetry_error": symmetry_error,
        "thermal_gradient": thermal_gradient,
        "twist_seed_curvature": seed_curvature,
        "nullspace": nullspace,
    }


def minimum_projected_stiffness(q, nullspace, step=1.0e-5):
    """Estimate the smallest reduced Hessian eigenvalue from exact gradients."""
    r = nullspace.T @ q
    hessian = np.empty((len(r), len(r)), dtype=float)

    for index in range(len(r)):
        delta = step * max(1.0, abs(r[index]))
        direction = np.zeros_like(r)
        direction[index] = delta
        _, gradient_plus = FREEFREE.total_internal_energy_and_gradient_q(
            nullspace @ (r + direction)
        )
        _, gradient_minus = FREEFREE.total_internal_energy_and_gradient_q(
            nullspace @ (r - direction)
        )
        hessian[:, index] = (nullspace.T @ (gradient_plus - gradient_minus)) / (2.0 * delta)

    eigenvalues = np.linalg.eigvalsh(0.5 * (hessian + hessian.T))
    return float(eigenvalues[0])


def result_row(layup, load_scale, branch, result, previous_result, args):
    """Build one CSV row, including continuation changes for snap screening."""
    curvature_increment = np.nan
    center_w_increment = np.nan
    if previous_result is not None and result["solver_converged"]:
        center_w_increment = abs(result["center_w"] - previous_result["center_w"])
        curvature_increment = float(
            np.linalg.norm(result["response"][3:] - previous_result["response"][3:])
        )

    min_stiffness = np.nan
    if args.hessian and result["solver_converged"]:
        min_stiffness = minimum_projected_stiffness(result["q"], result["nullspace"])

    snap_candidate = bool(
        np.isfinite(center_w_increment)
        and (
            center_w_increment >= args.snap_displacement_threshold
            or curvature_increment >= args.snap_curvature_threshold
        )
    )
    return {
        "Layup": layup_label(layup),
        "Branch": branch,
        "LoadScale": load_scale,
        "ThermalGradient_C_per_m": result["thermal_gradient"],
        "TwistSeedCurvature_1_per_m": result["twist_seed_curvature"],
        "RitzDegree": FREEFREE.ACTIVE_W_DEGREE,
        "SymmetryError_B_over_Ah": result["symmetry_error"],
        "SolverConverged": result["solver_converged"],
        "SolverIterations": result["solver_iterations"],
        "SolverMethod": result["solver_method"],
        "SolverMessage": result["solver_message"],
        "Energy_J": result["energy"],
        "ProjectedGradientNorm": result["gradient_norm"],
        "RelativeProjectedGradientNorm": result["relative_gradient_norm"],
        "CenterDisplacement_m": result["center_w"],
        "CenterDisplacementIncrement_m": center_w_increment,
        "CurvatureIncrement_1_per_m": curvature_increment,
        "MinimumProjectedStiffness": min_stiffness,
        "SnapCandidate": snap_candidate,
        "e0_11": result["response"][0],
        "e0_22": result["response"][1],
        "e0_12": result["response"][2],
        "K_11": result["response"][3],
        "K_22": result["response"][4],
        "K_12": result["response"][5],
    }


def add_branch_diagnostics(rows, zero_twist, positive_twist, negative_twist, args):
    """Annotate rows with paired-twist branch and zero-path diagnostics."""
    if positive_twist["solver_converged"] and negative_twist["solver_converged"]:
        displacement_separation = abs(
            positive_twist["center_w"] - negative_twist["center_w"]
        )
        curvature_separation = float(
            np.linalg.norm(
                positive_twist["response"][3:] - negative_twist["response"][3:]
            )
        )
        twist_separation = abs(
            positive_twist["response"][5] - negative_twist["response"][5]
        )
        energy_difference = abs(positive_twist["energy"] - negative_twist["energy"])
        relative_energy_difference = energy_difference / max(
            abs(positive_twist["energy"]), abs(negative_twist["energy"]), 1.0
        )
        opposite_twist_signs = bool(
            positive_twist["response"][5] * negative_twist["response"][5] < 0.0
        )
    else:
        displacement_separation = np.nan
        curvature_separation = np.nan
        twist_separation = np.nan
        energy_difference = np.nan
        relative_energy_difference = np.nan
        opposite_twist_signs = False

    if zero_twist["solver_converged"] and positive_twist["solver_converged"]:
        zero_to_twist_separation = abs(
            zero_twist["response"][5] - positive_twist["response"][5]
        )
    else:
        zero_to_twist_separation = np.nan

    branch_split = bool(
        np.isfinite(displacement_separation)
        and (
            displacement_separation >= args.branch_displacement_threshold
            or curvature_separation >= args.branch_curvature_threshold
            or twist_separation >= args.branch_twist_curvature_threshold
        )
    )
    bifurcation_candidate = bool(
        branch_split
        and opposite_twist_signs
        and relative_energy_difference <= args.branch_energy_relative_tolerance
    )
    for row in rows:
        row["BranchSeparationDisplacement_m"] = displacement_separation
        row["BranchSeparationCurvature_1_per_m"] = curvature_separation
        row["TwistBranchSeparation_1_per_m"] = twist_separation
        row["PositiveNegativeEnergyDifference_J"] = energy_difference
        row["PositiveNegativeRelativeEnergyDifference"] = relative_energy_difference
        row["OppositeTwistSigns"] = opposite_twist_signs
        row["ZeroToPositiveTwistSeparation_1_per_m"] = zero_to_twist_separation
        row["MultipleEquilibriaCandidate"] = branch_split
        row["BifurcationCandidate"] = bifurcation_candidate


def write_rows(path, rows):
    """Write the latest diagnostic rows with stable, spreadsheet-friendly headers."""
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Screen symmetric laminates for bifurcation and snap-through "
            "candidates using thermally loaded FREEFREE continuation branches."
        )
    )
    parser.add_argument(
        "--layup",
        action="append",
        help="Symmetric layup such as [0/90/90/0/0/90/90/0]. Repeat as needed.",
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--ritz-degree", type=int, default=3)
    parser.add_argument("--load-steps", type=int, default=11)
    parser.add_argument("--max-load-scale", type=float, default=1.5)
    parser.add_argument("--thermal-gradient", type=float, default=FREEFREE.FIXED_THERMAL_GRADIENT)
    parser.add_argument("--midplane-temperature", type=float, default=FREEFREE.FIXED_MIDPLANE_TEMPERATURE)
    parser.add_argument("--top-sink-temperature", type=float, default=0.0)
    parser.add_argument("--bottom-sink-temperature", type=float, default=100.0)
    parser.add_argument(
        "--twist-seed-curvature",
        type=float,
        default=2.0e-2,
        help="Magnitude of the positive/negative K12 seed in 1/m.",
    )
    parser.add_argument("--branch-displacement-threshold", type=float, default=1.0e-5)
    parser.add_argument("--branch-curvature-threshold", type=float, default=1.0e-2)
    parser.add_argument("--branch-twist-curvature-threshold", type=float, default=1.0e-2)
    parser.add_argument("--branch-energy-relative-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--snap-displacement-threshold", type=float, default=5.0e-4)
    parser.add_argument("--snap-curvature-threshold", type=float, default=5.0e-1)
    parser.add_argument(
        "--hessian",
        action="store_true",
        help="Estimate reduced-Hessian stiffness at each converged point (substantially slower).",
    )
    args = parser.parse_args()
    if args.ritz_degree < 2:
        parser.error("--ritz-degree must be at least 2")
    if args.load_steps < 2:
        parser.error("--load-steps must be at least 2")
    if args.max_load_scale < 0.0:
        parser.error("--max-load-scale must be nonnegative")
    if args.twist_seed_curvature <= 0.0:
        parser.error("--twist-seed-curvature must be positive")
    if args.branch_energy_relative_tolerance < 0.0:
        parser.error("--branch-energy-relative-tolerance cannot be negative")
    return args


def main():
    args = parse_arguments()
    layups = [parse_layup(text) for text in (args.layup or DEFAULT_LAYUPS)]
    for layup in layups:
        if not is_symmetric(layup):
            raise ValueError(
                "%s is not symmetric; use a sequence that mirrors about the midplane."
                % layup_label(layup)
            )

    load_scales = np.linspace(0.0, args.max_load_scale, args.load_steps)
    all_rows = []
    summary = {
        "converged": 0,
        "failed": 0,
        "branch_splits": 0,
        "bifurcations": 0,
        "snaps": 0,
    }
    print("Bifurcation/snap screening")
    print("  Layups: %d | load points: %d | Ritz degree: %d" % (
        len(layups), len(load_scales), args.ritz_degree
    ))
    print("  Output: %s" % args.output_csv)

    for layup in layups:
        print("\nTesting %s" % layup_label(layup))
        branches = ("zero_twist", "positive_twist", "negative_twist")
        previous_q = {branch: None for branch in branches}
        previous_result = {branch: None for branch in branches}
        for load_scale in load_scales:
            results = {}
            rows = []
            for branch in branches:
                result = solve_branch(
                    layup, load_scale, branch, previous_q[branch], args
                )
                row = result_row(
                    layup, load_scale, branch, result, previous_result[branch], args
                )
                results[branch] = result
                rows.append(row)
                if result["solver_converged"]:
                    previous_q[branch] = result["q"]
                    previous_result[branch] = result
                    summary["converged"] += 1
                else:
                    summary["failed"] += 1

            add_branch_diagnostics(
                rows,
                results["zero_twist"],
                results["positive_twist"],
                results["negative_twist"],
                args,
            )
            if rows[0]["MultipleEquilibriaCandidate"]:
                summary["branch_splits"] += 1
            if rows[0]["BifurcationCandidate"]:
                summary["bifurcations"] += 1
            summary["snaps"] += sum(bool(row["SnapCandidate"]) for row in rows)
            all_rows.extend(rows)
            write_rows(args.output_csv, all_rows)
            print(
                "  scale=%5.2f  zero:%s  +twist:%s  -twist:%s  bifurcation:%s" % (
                    load_scale,
                    "ok" if results["zero_twist"]["solver_converged"] else "failed",
                    "ok" if results["positive_twist"]["solver_converged"] else "failed",
                    "ok" if results["negative_twist"]["solver_converged"] else "failed",
                    "yes" if rows[0]["BifurcationCandidate"] else "no",
                )
            )

    print("\nCompleted %d branch solves: %d converged, %d failed." % (
        summary["converged"] + summary["failed"],
        summary["converged"],
        summary["failed"],
    ))
    print("Branch-split candidates: %d | bifurcation candidates: %d | snap candidates: %d" % (
        summary["branch_splits"], summary["bifurcations"], summary["snaps"]
    ))
    print("Results written to %s" % args.output_csv)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
