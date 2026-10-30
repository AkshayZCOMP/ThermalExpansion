"""
Run the FREEFREE nonlinear von Karman Ritz model for every row in LayerEffect.csv.

The script reads Abaqus cases from:

    LayerEffect.csv

and writes a new augmented file:

    LayerEffect_with_computational_results.csv

Each output row keeps the original Abaqus columns, then appends:
    - computational response components
    - signed errors: computational - Abaqus
    - absolute errors
    - percent errors
    - row-level L2 error summaries

Previously solved FREEFREE results are reused on later runs.  Only cases whose
layup, ply count, sink temperatures, or Ritz configuration changed are solved
again.
"""

import argparse
import csv
import importlib.util
import math
import time
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
INPUT_CSV = SCRIPT_DIR / "LayerEffect.csv"
OUTPUT_CSV = SCRIPT_DIR / "LayerEffect_with_computational_results.csv"
FREEFREE_PATH = REPO_ROOT / "Computational Real" / "FREEFREE.py"


# =============================================================================
# MODEL CONSTANTS MATCHING Computational Real/FREEFREE.py
# =============================================================================

t_ply = 0.000125     # Single ply thickness [m]

E1 = 130e9           # Fiber direction modulus [Pa]
E2 = 10e9            # Transverse modulus [Pa]
nu12 = 0.3           # Major Poisson's ratio
G12 = 5e9            # In-plane shear modulus [Pa]

alpha1 = -0.5e-6     # Fiber CTE [/C]
alpha2 = 30e-6       # Transverse CTE [/C]
kz = 0.5             # Transverse thermal conductivity [W/(m*K)]

T_init = 25.0        # Initial stress-free temp [C]
h_bot = 1000.0       # Bottom contact film coefficient [W/(m^2*K)]
h_top = 1000.0       # Top air knife film coefficient [W/(m^2*K)]

# Shared prescribed field used by both the Abaqus sweep and FREEFREE.
FIXED_MIDPLANE_TEMPERATURE = 50.0
FIXED_THERMAL_GRADIENT = -53333.3  # C/m

def build_model_signature():
    degree = (
        "auto"
        if FREEFREE_TRANSVERSE_RITZ_DEGREE is None
        else FREEFREE_TRANSVERSE_RITZ_DEGREE
    )
    return "FREEFREE_FVK_RITZ_v2|degree=%s|tmid=%g|beta=%g|tinit=%g" % (
        degree,
        FIXED_MIDPLANE_TEMPERATURE,
        FIXED_THERMAL_GRADIENT,
        T_init,
    )


ABAQUS_COMPONENT_COLUMNS = ("e0_11", "e0_22", "e0_12", "K_11", "K_22", "K_12")
COMPONENT_NAMES = ("e0_11", "e0_22", "e0_12", "K_11", "K_22", "K_12")


def log_progress(message):
    """Print immediately so long FREEFREE solves do not look stalled."""
    print(message, flush=True)


def load_freefree_solver():
    """Load the reusable nonlinear free-free solver from the sibling module."""
    spec = importlib.util.spec_from_file_location("freefree_solver", FREEFREE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load FREEFREE solver from {FREEFREE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FREEFREE = load_freefree_solver()
# Use FREEFREE's configured high-order degree for every laminate case.
FREEFREE_TRANSVERSE_RITZ_DEGREE = FREEFREE.HIGH_ORDER_W_DEGREE
# FREEFREE.py is reusable for several studies; align its case-level thermal
# constants with the Abaqus sweep before solving any CSV row.
FREEFREE.T_init = T_init
FREEFREE.h_bot = h_bot
FREEFREE.h_top = h_top
FREEFREE.FIXED_MIDPLANE_TEMPERATURE = FIXED_MIDPLANE_TEMPERATURE
FREEFREE.FIXED_THERMAL_GRADIENT = FIXED_THERMAL_GRADIENT
MODEL_SIGNATURE = build_model_signature()


# =============================================================================
# CLT MODEL
# =============================================================================

nu21 = nu12 * (E2 / E1)

Q11 = E1 / (1.0 - nu12 * nu21)
Q22 = E2 / (1.0 - nu12 * nu21)
Q12 = nu12 * E2 / (1.0 - nu12 * nu21)
Q66 = G12

Q = np.array(
    [
        [Q11, Q12, 0.0],
        [Q12, Q22, 0.0],
        [0.0, 0.0, Q66],
    ],
    dtype=float,
)

R_mat = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 2.0],
    ],
    dtype=float,
)


def parse_layup(layup_text):
    """Parse strings like '[0/45/-45/90]' into a list of integer angles."""
    clean = layup_text.strip()
    if not clean.startswith("[") or not clean.endswith("]"):
        raise ValueError(f"Invalid layup string: {layup_text!r}")

    body = clean[1:-1].strip()
    if not body:
        return []

    return [int(float(value.strip())) for value in body.split("/")]


def transformed_ply_matrices(angle):
    """Return Q_bar and alpha_bar for one ply angle."""
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)

    T_mat = np.array(
        [
            [c**2, s**2, 2.0 * c * s],
            [s**2, c**2, -2.0 * c * s],
            [-c * s, c * s, c**2 - s**2],
        ],
        dtype=float,
    )

    q_bar = np.linalg.inv(T_mat) @ Q @ R_mat @ T_mat @ np.linalg.inv(R_mat)
    alpha_bar = np.linalg.inv(T_mat) @ np.array([alpha1, alpha2, 0.0], dtype=float)

    return q_bar, alpha_bar


PLY_LOOKUP = {
    angle: transformed_ply_matrices(angle)
    for angle in (-45, 0, 45, 90)
}


def surface_temperatures(num_plies, top_sink_temp, bottom_sink_temp):
    """Compute top and bottom laminate surface temperatures from sink temps."""
    total_thickness = num_plies * t_ply

    R_bot = 1.0 / h_bot
    R_cond = total_thickness / kz
    R_top = 1.0 / h_top
    R_total = R_bot + R_cond + R_top

    q_flux = (bottom_sink_temp - top_sink_temp) / R_total
    bottom_surface_temp = bottom_sink_temp - (q_flux * R_bot)
    top_surface_temp = top_sink_temp + (q_flux * R_top)

    return top_surface_temp, bottom_surface_temp


def calculate_response(layup, top_sink_temp, bottom_sink_temp):
    """Calculate the Abaqus-comparable response with FREEFREE's FvK solver."""
    FREEFREE.configure_case(
        layup,
        top_sink_temp,
        bottom_sink_temp,
        transverse_degree=FREEFREE_TRANSVERSE_RITZ_DEGREE,
        midplane_temperature=FIXED_MIDPLANE_TEMPERATURE,
        thermal_gradient=FIXED_THERMAL_GRADIENT,
    )
    solution = FREEFREE.solve_default_equilibrium(verbose=False)
    solver = solution["solver"]
    eps_0, curvature = FREEFREE.area_averaged_generalized_quantities(solver.q)
    result = {
        "response": np.concatenate((eps_0, curvature)),
        "top_surface_temperature": FREEFREE.T_top_surf,
        "bottom_surface_temperature": FREEFREE.T_bot_surf,
        "transverse_degree": FREEFREE.ACTIVE_W_DEGREE,
        "solver_iterations": solver.iterations,
        "solver_converged": solver.converged,
        "solver_message": solver.message,
        "solution": solution,
    }
    return (
        result["response"],
        result["top_surface_temperature"],
        result["bottom_surface_temperature"],
        result,
    )


# =============================================================================
# CSV AUGMENTATION
# =============================================================================

def safe_percent_error(error, reference):
    """Return signed percent error, using blank for near-zero references."""
    if abs(reference) < 1e-14:
        return ""
    return 100.0 * error / reference


def format_csv_value(value):
    """Format numeric values consistently while leaving blanks as blanks."""
    if value == "":
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return f"{float(value):.8e}"


def augmented_fieldnames(input_fieldnames):
    """Build output fieldnames with computational and error columns appended."""
    fieldnames = list(input_fieldnames)
    fieldnames.extend(
        [
            "CompTopTemperature",
            "CompBottomTemperature",
            "CompRitzDegree",
            "CompSolverIterations",
            "CompSolverConverged",
            "CompSolverMessage",
            "CompModelSignature",
            "CompResultSource",
        ]
    )

    for name in COMPONENT_NAMES:
        fieldnames.append(f"Comp_{name}")

    for name in COMPONENT_NAMES:
        fieldnames.append(f"Error_{name}")

    for name in COMPONENT_NAMES:
        fieldnames.append(f"AbsError_{name}")

    for name in COMPONENT_NAMES:
        fieldnames.append(f"PercentError_{name}")

    fieldnames.extend(
        [
            "ResponseL2Error",
            "StrainL2Error",
            "CurvatureL2Error",
            "Agreement",
            "CaseMaxNormalizedError",
        ]
    )
    return fieldnames


def case_cache_key(row):
    """Return the inputs that uniquely determine one FREEFREE solve."""
    layup = "/".join(str(angle) for angle in parse_layup(row["Layup"]))
    return (
        layup,
        int(row["NumberOfPlies"]),
        f"{float(row['TopSinkTemperature']):.12g}",
        f"{float(row['BottomSinkTemperature']):.12g}",
        MODEL_SIGNATURE,
    )


def load_computational_cache(output_csv):
    """Load compatible computational results from a prior comparison output."""
    if not output_csv.exists():
        return {}

    cache = {}
    with output_csv.open(newline="", encoding="utf-8-sig") as output_file:
        for row in csv.DictReader(output_file):
            if row.get("CompModelSignature") != MODEL_SIGNATURE:
                continue
            if row.get("CompSolverConverged", "True") != "True":
                continue
            try:
                cache[case_cache_key(row)] = row
            except (KeyError, TypeError, ValueError):
                continue
    return cache


def cached_computational_result(row):
    """Rebuild the computational result from an existing augmented CSV row."""
    response = np.array(
        [float(row[f"Comp_{name}"]) for name in COMPONENT_NAMES], dtype=float
    )
    return (
        response,
        float(row["CompTopTemperature"]),
        float(row["CompBottomTemperature"]),
        {
            "transverse_degree": int(row["CompRitzDegree"]),
            "solver_iterations": int(row["CompSolverIterations"]),
            "solver_converged": row.get("CompSolverConverged", "True") == "True",
            "solver_message": row.get("CompSolverMessage", ""),
        },
    )


def augment_row(row, strain_tolerance, curvature_tolerance, cached_row=None):
    """Run the computational model for one CSV row and append comparison fields."""
    layup = parse_layup(row["Layup"])
    declared_plies = int(row["NumberOfPlies"])
    if len(layup) != declared_plies:
        raise ValueError(
            f"Layup {row['Layup']} has {len(layup)} plies, "
            f"but NumberOfPlies is {declared_plies}."
        )

    top_sink_temp = float(row["TopSinkTemperature"])
    bottom_sink_temp = float(row["BottomSinkTemperature"])
    if cached_row is None:
        comp_response, comp_top_temp, comp_bottom_temp, solver_result = calculate_response(
            layup,
            top_sink_temp,
            bottom_sink_temp,
        )
        result_source = (
            "SOLVED"
            if solver_result.get("solver_converged", True)
            else "SOLVER_NOT_CONVERGED"
        )
    else:
        comp_response, comp_top_temp, comp_bottom_temp, solver_result = (
            cached_computational_result(cached_row)
        )
        result_source = "CACHED"

    abaqus_response = np.array(
        [float(row[column]) for column in ABAQUS_COMPONENT_COLUMNS],
        dtype=float,
    )
    errors = comp_response - abaqus_response
    abs_errors = np.abs(errors)
    percent_errors = [
        safe_percent_error(error, reference)
        for error, reference in zip(errors, abaqus_response)
    ]

    output = dict(row)
    output["CompTopTemperature"] = format_csv_value(comp_top_temp)
    output["CompBottomTemperature"] = format_csv_value(comp_bottom_temp)
    output["CompRitzDegree"] = str(solver_result["transverse_degree"])
    output["CompSolverIterations"] = str(solver_result["solver_iterations"])
    output["CompSolverConverged"] = str(solver_result.get("solver_converged", True))
    output["CompSolverMessage"] = solver_result.get("solver_message", "")
    output["CompModelSignature"] = MODEL_SIGNATURE
    output["CompResultSource"] = result_source

    for name, value in zip(COMPONENT_NAMES, comp_response):
        output[f"Comp_{name}"] = format_csv_value(value)

    for name, value in zip(COMPONENT_NAMES, errors):
        output[f"Error_{name}"] = format_csv_value(value)

    for name, value in zip(COMPONENT_NAMES, abs_errors):
        output[f"AbsError_{name}"] = format_csv_value(value)

    for name, value in zip(COMPONENT_NAMES, percent_errors):
        output[f"PercentError_{name}"] = format_csv_value(value)

    output["ResponseL2Error"] = format_csv_value(float(np.linalg.norm(errors)))
    output["StrainL2Error"] = format_csv_value(float(np.linalg.norm(errors[:3])))
    output["CurvatureL2Error"] = format_csv_value(float(np.linalg.norm(errors[3:])))

    tolerances = np.array(
        [strain_tolerance, strain_tolerance, strain_tolerance,
         curvature_tolerance, curvature_tolerance, curvature_tolerance],
        dtype=float,
    )
    normalized_errors = abs_errors / tolerances
    solver_converged = solver_result.get("solver_converged", True)
    output["Agreement"] = (
        "PASS" if solver_converged and np.all(normalized_errors <= 1.0) else "FAIL"
    )
    output["CaseMaxNormalizedError"] = format_csv_value(
        float(np.max(normalized_errors))
    )

    return output, errors, output["Agreement"], result_source


def run_batch(
    input_csv=INPUT_CSV,
    output_csv=OUTPUT_CSV,
    strain_tolerance=2.0e-4,
    curvature_tolerance=5.0e-1,
    force_recompute=False,
):
    """Write comparisons, solving only cases not already present in the cache."""
    computational_cache = {} if force_recompute else load_computational_cache(output_csv)
    with input_csv.open(newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {input_csv}")

        fieldnames = augmented_fieldnames(reader.fieldnames)
        input_rows = list(reader)
        rows = []
        error_vectors = []
        failed_layups = []
        solved_count = 0
        cached_count = 0

        log_progress("=" * 80)
        log_progress("ABAQUS VS COMPUTATIONAL MODEL BATCH START")
        log_progress("=" * 80)
        log_progress(f"Input:  {input_csv}")
        log_progress(f"Output: {output_csv}")
        log_progress(f"Rows to process: {len(input_rows)}")
        log_progress(f"Cached compatible cases: {len(computational_cache)}")
        log_progress(f"Model signature: {MODEL_SIGNATURE}")
        log_progress("=" * 80)
        batch_start = time.perf_counter()

        for index, row in enumerate(input_rows, start=1):
            row_number = index + 1
            key = case_cache_key(row)
            cached_row = computational_cache.get(key)
            source_label = "cache" if cached_row is not None else "solve"
            case_start = time.perf_counter()
            log_progress(
                f"[{index}/{len(input_rows)}] {source_label.upper()} row "
                f"{row_number}: {row['Layup']}"
            )
            try:
                output_row, errors, agreement, result_source = augment_row(
                    row,
                    strain_tolerance,
                    curvature_tolerance,
                    cached_row,
                )
            except Exception as exc:
                raise RuntimeError(f"Failed on CSV row {row_number}: {row}") from exc
            rows.append(output_row)
            error_vectors.append(errors)
            if result_source == "SOLVED":
                solved_count += 1
            else:
                cached_count += 1
            if agreement == "FAIL":
                failed_layups.append(row["Layup"])
            case_elapsed = time.perf_counter() - case_start
            log_progress(
                f"    -> {agreement} via {result_source} in {case_elapsed:.1f}s; "
                f"iters={output_row['CompSolverIterations']}; "
                f"converged={output_row['CompSolverConverged']}; "
                f"max_norm_error={output_row['CaseMaxNormalizedError']}"
            )

    with output_csv.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    error_matrix = np.array(error_vectors)
    max_abs_errors = np.max(np.abs(error_matrix), axis=0)
    mean_abs_errors = np.mean(np.abs(error_matrix), axis=0)

    log_progress("=" * 80)
    log_progress("ABAQUS VS COMPUTATIONAL MODEL BATCH COMPLETE")
    log_progress("=" * 80)
    log_progress(f"Input:  {input_csv}")
    log_progress(f"Output: {output_csv}")
    log_progress(f"Rows processed: {len(rows)}")
    log_progress(f"Elapsed: {time.perf_counter() - batch_start:.1f}s")
    log_progress(f"FREEFREE solves: {solved_count}; cached results reused: {cached_count}")
    log_progress(
        f"Agreement tolerances: strain <= {strain_tolerance:.3e}, "
        f"curvature <= {curvature_tolerance:.3e}"
    )
    log_progress(f"Cases passing: {len(rows) - len(failed_layups)}/{len(rows)}")
    log_progress("\nMax absolute error by component:")
    for name, value in zip(COMPONENT_NAMES, max_abs_errors):
        log_progress(f"  {name:6s}: {value:.8e}")
    log_progress("\nMean absolute error by component:")
    for name, value in zip(COMPONENT_NAMES, mean_abs_errors):
        log_progress(f"  {name:6s}: {value:.8e}")
    log_progress("=" * 80)

    if failed_layups:
        log_progress("Disagreeing layups: %s" % ", ".join(failed_layups))

    return rows, failed_layups


def raw_case_key(row):
    """Identify one Abaqus case without tying it to a solver implementation."""
    return (
        "/".join(str(angle) for angle in parse_layup(row["Layup"])),
        int(row["NumberOfPlies"]),
        f"{float(row['TopSinkTemperature']):.12g}",
        f"{float(row['BottomSinkTemperature']):.12g}",
    )


def has_cached_computational_result(row):
    """Return True when a row carries a complete reusable solver result."""
    required_fields = [
        "CompTopTemperature",
        "CompBottomTemperature",
        "CompRitzDegree",
        "CompSolverIterations",
    ] + [f"Comp_{name}" for name in COMPONENT_NAMES]
    try:
        return (
            row.get("CompSolverConverged", "True") == "True"
            and all(row[field] != "" for field in required_fields)
        )
    except KeyError:
        return False


def run_single_case(
    input_csv,
    output_csv,
    layup_text,
    strain_tolerance=2.0e-4,
    curvature_tolerance=5.0e-1,
):
    """Compare only one newly finished Abaqus layup and preserve prior rows."""
    with input_csv.open(newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {input_csv}")
        input_fieldnames = reader.fieldnames
        target_row = None
        for row in reader:
            if row["Layup"] == layup_text:
                target_row = row

    if target_row is None:
        raise ValueError(f"Layup {layup_text} was not found in {input_csv}")

    fieldnames = augmented_fieldnames(input_fieldnames)
    existing_rows = []
    cached_row = None
    target_key = raw_case_key(target_row)
    if output_csv.exists():
        with output_csv.open(newline="", encoding="utf-8-sig") as output_file:
            for row in csv.DictReader(output_file):
                existing_rows.append(row)
                if (
                    raw_case_key(row) == target_key
                    and row.get("CompModelSignature") == MODEL_SIGNATURE
                    and has_cached_computational_result(row)
                ):
                    cached_row = row

    cached_label = "cache" if cached_row is not None else "solve"
    log_progress(f"Updating {layup_text} from {cached_label.upper()}...")
    case_start = time.perf_counter()
    output_row, errors, agreement, result_source = augment_row(
        target_row,
        strain_tolerance,
        curvature_tolerance,
        cached_row,
    )
    log_progress(
        f"Finished {layup_text} via {result_source} in "
        f"{time.perf_counter() - case_start:.1f}s"
    )

    updated_rows = []
    replaced = False
    for row in existing_rows:
        if raw_case_key(row) == target_key:
            if not replaced:
                updated_rows.append(output_row)
                replaced = True
            continue
        updated_rows.append({field: row.get(field, "") for field in fieldnames})
    if not replaced:
        updated_rows.append(output_row)

    with output_csv.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    print("=" * 80)
    print("ABAQUS VS COMPUTATIONAL MODEL SINGLE-CASE UPDATE")
    print("=" * 80)
    print(f"Layup: {layup_text}")
    print(f"Computational result: {result_source}")
    print(f"Solver converged: {output_row['CompSolverConverged']}")
    if output_row.get("CompSolverMessage"):
        print(f"Solver message: {output_row['CompSolverMessage']}")
    print(
        "Surface temperatures [C]: Abaqus top/bottom = "
        f"{float(target_row['TopTemperature']):.8e} / "
        f"{float(target_row['BottomTemperature']):.8e}; "
        "computational top/bottom = "
        f"{float(output_row['CompTopTemperature']):.8e} / "
        f"{float(output_row['CompBottomTemperature']):.8e}"
    )
    print(f"Agreement: {agreement}")
    print(f"Output: {output_csv}")
    print("=" * 80)
    return [output_row], [layup_text] if agreement == "FAIL" else []


def main():
    global FREEFREE_TRANSVERSE_RITZ_DEGREE, MODEL_SIGNATURE

    parser = argparse.ArgumentParser(
        description="Compare Abaqus results with the FREEFREE nonlinear FvK Ritz model."
    )
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--strain-tolerance", type=float, default=2.0e-4)
    parser.add_argument("--curvature-tolerance", type=float, default=5.0e-1)
    parser.add_argument(
        "--ritz-degree",
        type=int,
        default=FREEFREE_TRANSVERSE_RITZ_DEGREE,
        help=(
            "Transverse polynomial degree for FREEFREE (default: the shared "
            "high-order degree used for every layup)."
        ),
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore cached computational results and solve every CSV row again.",
    )
    parser.add_argument(
        "--layup",
        help="Update only this completed layup instead of processing the full CSV.",
    )
    parser.add_argument("--fail-on-disagreement", action="store_true")
    args = parser.parse_args()

    if args.ritz_degree is not None and args.ritz_degree < 2:
        parser.error("--ritz-degree must be at least 2")
    FREEFREE_TRANSVERSE_RITZ_DEGREE = args.ritz_degree
    MODEL_SIGNATURE = build_model_signature()
    

    if args.layup is not None:
        _, failed_layups = run_single_case(
            args.input_csv,
            args.output_csv,
            args.layup,
            args.strain_tolerance,
            args.curvature_tolerance,
        )
    else:
        _, failed_layups = run_batch(
            args.input_csv,
            args.output_csv,
            args.strain_tolerance,
            args.curvature_tolerance,
            args.force_recompute,
        )
    return 1 if args.fail_on_disagreement and failed_layups else 0


if __name__ == "__main__":
    raise SystemExit(main())
