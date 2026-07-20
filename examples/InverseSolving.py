"""
INVERSE SOLVING WITH TRANSIENT THERMAL DIFFUSION

This file pulls the useful setup from thermal_diffusion_example.py into
functions so inverse-solving code can call the thermal model repeatedly.
"""

import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path for imports when running from examples/
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.InputProperties import MaterialProperties, LayupSequence, LaminateProperties
from src.ThermalDiffusion import ThermalDiffusionAnalyzer
from src.workflow import LaminateAnalysis


# ============================================================================
# CONFIGURATION - EDIT HERE
# ============================================================================

MATERIAL = MaterialProperties(
    name="Carbon/Epoxy",
    E_11=130e9,
    E_22=10e9,
    v_12=0.3,
    G_12=5e9,
    alpha_1=-0.5e-6,
    alpha_2=30e-6,
)

PLY_THICKNESS = 0.125e-3
LAYUP = [0, -45, 15,15, -45,0]

INITIAL_TEMP = 20
TOP_SURFACE_TEMP = 100
BOTTOM_SURFACE_TEMP = 25
ANALYSIS_TIME = None
N_TIME_STEPS = 100
N_SPATIAL_NODES = 75
THERMAL_DIFFUSIVITY = 1e-6

CANDIDATE_ANGLES = list(range(-90, 91, 15))
SOLVER_METHOD = "beam" #exhasutive, beam, 
BEAM_WIDTH = 50
ASSUME_SYMMETRIC_LAYUP = True
RUN_VALIDATION_CASES = False
VALIDATION_PLY_COUNTS = [2, 3, 4, 5, 6, 7, 8]
VALIDATION_CASES_PER_PLY_COUNT = 60
VALIDATION_RANDOM_SEED = 11
VALIDATION_RESPONSE_ERROR_TOLERANCE = 1e-8
INVERSE_RANDOM_STARTS = 25
INVERSE_MAX_ITERATIONS = 40
INVERSE_TOLERANCE = 1e-12
MIN_RESPONSE_SCALE = 1e-12


# ============================================================================
# REUSABLE THERMAL DIFFUSION FUNCTIONS
# ============================================================================

def build_laminate(material, layup_angles, ply_thickness):
    """Create a LaminateProperties object from the inverse-solver inputs."""
    layup = LayupSequence(layup_angles, name="Inverse Solver Layup")
    return LaminateProperties(
        material=material,
        layup=layup,
        ply_thickness=ply_thickness,
        delta_T=0,
    )


def run_thermal_diffusion_analysis(
    material=MATERIAL,
    layup_angles=LAYUP,
    ply_thickness=PLY_THICKNESS,
    initial_temp=INITIAL_TEMP,
    top_surface_temp=TOP_SURFACE_TEMP,
    bottom_surface_temp=BOTTOM_SURFACE_TEMP,
    analysis_time=ANALYSIS_TIME,
    n_time_steps=N_TIME_STEPS,
    n_spatial_nodes=N_SPATIAL_NODES,
    thermal_diffusivity=THERMAL_DIFFUSIVITY,
):
    """
    Run the same thermal diffusion analysis as thermal_diffusion_example.py.

    Returns
    -------
    tuple
        (analyzer, results, laminate)
    """
    laminate = build_laminate(material, layup_angles, ply_thickness)

    analyzer = ThermalDiffusionAnalyzer(
        laminate_properties=laminate,
        top_surface_temp=top_surface_temp,
        initial_temp=initial_temp,
        bottom_surface_temp=bottom_surface_temp,
    )

    results = analyzer.analyze(
        t_final=analysis_time,
        n_steps=n_time_steps,
        n_nodes=n_spatial_nodes,
        thermal_diffusivity=thermal_diffusivity,
    )

    return analyzer, results, laminate


def get_final_response(results):
    """Return final membrane strains and curvatures from a thermal run."""
    return {
        "strains": results["strains"][-1],
        "curvatures": results["curvatures"][-1],
    }


def strain_error(results, target_strains):
    """Compare final predicted membrane strains against a target strain vector."""
    predicted = results["strains"][-1]
    target = np.asarray(target_strains)
    return float(np.linalg.norm(predicted - target))


def inverse_solve_top_surface_temp(
    target_strains,
    candidate_temperatures=None,
    material=MATERIAL,
    layup_angles=LAYUP,
    ply_thickness=PLY_THICKNESS,
    initial_temp=INITIAL_TEMP,
    bottom_surface_temp=BOTTOM_SURFACE_TEMP,
    analysis_time=ANALYSIS_TIME,
    n_time_steps=N_TIME_STEPS,
    n_spatial_nodes=N_SPATIAL_NODES,
    thermal_diffusivity=THERMAL_DIFFUSIVITY,
):
    """
    Simple inverse solve: find the top-surface temperature that best matches
    measured/target final membrane strains.

    This is a brute-force grid search so it is easy to understand and modify.
    """
    if candidate_temperatures is None:
        candidate_temperatures = np.linspace(initial_temp, 200, 50)

    best = None

    for top_temp in candidate_temperatures:
        analyzer, results, laminate = run_thermal_diffusion_analysis(
            material=material,
            layup_angles=layup_angles,
            ply_thickness=ply_thickness,
            initial_temp=initial_temp,
            top_surface_temp=float(top_temp),
            bottom_surface_temp=bottom_surface_temp,
            analysis_time=analysis_time,
            n_time_steps=n_time_steps,
            n_spatial_nodes=n_spatial_nodes,
            thermal_diffusivity=thermal_diffusivity,
        )
        error = strain_error(results, target_strains)

        if best is None or error < best["error"]:
            best = {
                "top_surface_temp": float(top_temp),
                "error": error,
                "analyzer": analyzer,
                "results": results,
                "laminate": laminate,
            }

    return best


def normalize_response(response):
    """
    Scale each strain/curvature component by the measured target magnitude.

    This makes the solver compare relative error in each response component:
    eps_x, eps_y, gamma_xy, kappa_x, kappa_y, and kappa_xy.
    """
    response = np.asarray(response, dtype=float)
    return np.maximum(np.abs(response), MIN_RESPONSE_SCALE)


def make_symmetric_layup(independent_angles, num_plies):
    """Build a symmetric layup from the bottom half plus optional middle ply."""
    independent_angles = list(independent_angles)
    half_count = num_plies // 2

    if num_plies % 2 == 0:
        bottom_half = independent_angles[:half_count]
        return bottom_half + bottom_half[::-1]

    bottom_half = independent_angles[:half_count]
    middle = [independent_angles[half_count]]
    return bottom_half + middle + bottom_half[::-1]


def symmetrize_layup(layup_angles):
    """Mirror the first half of a layup across the laminate mid-plane."""
    layup_angles = list(layup_angles)
    independent_count = (len(layup_angles) + 1) // 2
    return make_symmetric_layup(layup_angles[:independent_count], len(layup_angles))


def get_search_ply_indices(num_plies):
    """Return the independent ply indices for the configured search space."""
    if ASSUME_SYMMETRIC_LAYUP:
        return list(range((num_plies + 1) // 2))
    return list(range(num_plies))


def set_candidate_angle(layup_angles, ply_index, angle):
    """Set one candidate angle, mirroring it when symmetry is enabled."""
    trial = list(layup_angles)
    trial[ply_index] = angle

    if ASSUME_SYMMETRIC_LAYUP:
        mirror_index = len(trial) - 1 - ply_index
        trial[mirror_index] = angle

    return trial


def build_candidate_analysis(
    layup_angles,
    material=MATERIAL,
    ply_thickness=PLY_THICKNESS,
):
    """Calculate Q, Q_bar, and ABD for one candidate layup."""
    layup = LayupSequence(list(layup_angles), name="Inverse Candidate")
    laminate = LaminateProperties(
        material=material,
        layup=layup,
        ply_thickness=ply_thickness,
        delta_T=0,
    )
    analysis = LaminateAnalysis(laminate)
    analysis.calculate_transformed_matrices()
    analysis.calculate_laminate_stiffness_matrices()
    material_matrices = analysis.calculate_material_matrices()
    return laminate, analysis, material_matrices["Q"]


def response_from_temperature_profile(
    layup_angles,
    z_thermal,
    temperature_profile,
    reference_temp,
    material=MATERIAL,
    ply_thickness=PLY_THICKNESS,
):
    """
    Predict strains/curvatures for one candidate layup under the known
    through-thickness temperature profile.
    """
    laminate, analysis, Q = build_candidate_analysis(
        layup_angles,
        material=material,
        ply_thickness=ply_thickness,
    )
    ply_count = laminate.layup.num_plies
    total_thickness = laminate.total_thickness
    z_bottom = -total_thickness / 2

    N_thermal = np.zeros(3)
    M_thermal = np.zeros(3)

    for i in range(ply_count):
        z_i = z_bottom + i * ply_thickness
        z_ip1 = z_i + ply_thickness
        z_mechanical = np.linspace(z_i, z_ip1, 5)

        z_for_temperature = total_thickness / 2 - z_mechanical
        delta_T = np.interp(z_for_temperature, z_thermal, temperature_profile) - reference_temp

        thermal_stress_per_degree = (
            analysis.Q_bar_list[i] @ analysis.alpha_bar_list[i]
        )
        N_thermal += thermal_stress_per_degree * np.trapezoid(delta_T, z_mechanical)
        M_thermal += thermal_stress_per_degree * np.trapezoid(
            delta_T * z_mechanical,
            z_mechanical,
        )

    response = np.linalg.inv(analysis.ABD_matrix) @ np.concatenate(
        (N_thermal, M_thermal)
    )

    return {
        "layup": list(layup_angles),
        "Q": Q,
        "Q_bar_list": analysis.Q_bar_list,
        "A": analysis.A_matrix,
        "B": analysis.B_matrix,
        "D": analysis.D_matrix,
        "ABD": analysis.ABD_matrix,
        "N_thermal": N_thermal,
        "M_thermal": M_thermal,
        "response": response,
        "strains": response[:3],
        "curvatures": response[3:6],
    }


def score_layup_candidate(
    layup_angles,
    target_response,
    response_scale,
    z_thermal,
    temperature_profile,
    reference_temp,
):
    """Return the normalized response error for one complete layup candidate."""
    result = response_from_temperature_profile(
        layup_angles,
        z_thermal,
        temperature_profile,
        reference_temp,
    )
    residual = (result["response"] - target_response) / response_scale
    result["normalized_residual"] = residual
    result["component_errors"] = np.abs(residual)
    result["strain_error"] = float(np.linalg.norm(residual[:3]))
    result["curvature_error"] = float(np.linalg.norm(residual[3:6]))
    result["error"] = float(np.linalg.norm(residual))
    return result


def inverse_solve_layup_from_gradient(
    target_response,
    z_thermal,
    temperature_profile,
    reference_temp,
    num_plies,
):
    """
    Predict the layup angles from a known gradient response.

    This checks all allowed angles concurrently for each ply position and
    repeats from multiple starting points. It avoids changing material Q;
    instead, candidate angles change Q_bar, ABD, N_T, and M_T.
    """
    rng = np.random.default_rng(7)
    response_scale = normalize_response(target_response)
    starts = [
        [0] * num_plies,
        [90, 0] * (num_plies // 2) + ([90] if num_plies % 2 else []),
    ]
    if ASSUME_SYMMETRIC_LAYUP:
        starts = [symmetrize_layup(start) for start in starts]

    while len(starts) < INVERSE_RANDOM_STARTS:
        if ASSUME_SYMMETRIC_LAYUP:
            independent_count = (num_plies + 1) // 2
            independent = rng.choice(CANDIDATE_ANGLES, size=independent_count)
            starts.append(make_symmetric_layup(independent, num_plies))
        else:
            starts.append(list(rng.choice(CANDIDATE_ANGLES, size=num_plies)))

    best_overall = None

    for start_index, start in enumerate(starts, start=1):
        current_layup = start
        current = score_layup_candidate(
            current_layup,
            target_response,
            response_scale,
            z_thermal,
            temperature_profile,
            reference_temp,
        )

        for iteration in range(1, INVERSE_MAX_ITERATIONS + 1):
            previous_error = current["error"]

            for ply_index in get_search_ply_indices(num_plies):
                trial_layups = []
                for angle in CANDIDATE_ANGLES:
                    trial_layups.append(set_candidate_angle(current_layup, ply_index, angle))

                with ThreadPoolExecutor() as executor:
                    trial_results = list(executor.map(
                        lambda angles: score_layup_candidate(
                            angles,
                            target_response,
                            response_scale,
                            z_thermal,
                            temperature_profile,
                            reference_temp,
                        ),
                        trial_layups,
                    ))

                current = min(trial_results, key=lambda item: item["error"])
                current_layup = current["layup"]

            if previous_error - current["error"] < INVERSE_TOLERANCE:
                break

        current["start_index"] = start_index
        current["iterations"] = iteration

        if best_overall is None or current["error"] < best_overall["error"]:
            best_overall = current

    return best_overall


def inverse_solve_layup_from_gradient_beam(
    target_response,
    z_thermal,
    temperature_profile,
    reference_temp,
    num_plies,
):
    """
    Predict layup angles using beam search over complete candidate layups.

    Unlike coordinate descent, this keeps the best BEAM_WIDTH full layups after
    every ply update, so one bad early angle choice does not control the whole
    solve.
    """
    rng = np.random.default_rng(7)
    response_scale = normalize_response(target_response)

    initial_layups = [
        tuple([0] * num_plies),
        tuple([90, 0] * (num_plies // 2) + ([90] if num_plies % 2 else [])),
    ]
    if ASSUME_SYMMETRIC_LAYUP:
        initial_layups = [tuple(symmetrize_layup(layup)) for layup in initial_layups]

    while len(initial_layups) < BEAM_WIDTH:
        if ASSUME_SYMMETRIC_LAYUP:
            independent_count = (num_plies + 1) // 2
            independent = rng.choice(CANDIDATE_ANGLES, size=independent_count)
            initial_layups.append(tuple(make_symmetric_layup(independent, num_plies)))
        else:
            initial_layups.append(tuple(rng.choice(CANDIDATE_ANGLES, size=num_plies)))

    unique_initial_layups = list(dict.fromkeys(initial_layups))
    with ThreadPoolExecutor() as executor:
        beam = list(executor.map(
            lambda angles: score_layup_candidate(
                angles,
                target_response,
                response_scale,
                z_thermal,
                temperature_profile,
                reference_temp,
            ),
            unique_initial_layups,
        ))

    beam = sorted(beam, key=lambda item: item["error"])[:BEAM_WIDTH]

    for iteration in range(1, INVERSE_MAX_ITERATIONS + 1):
        previous_best_error = beam[0]["error"]

        for ply_index in get_search_ply_indices(num_plies):
            trial_layups = []
            for candidate in beam:
                for angle in CANDIDATE_ANGLES:
                    trial_layups.append(tuple(
                        set_candidate_angle(candidate["layup"], ply_index, angle)
                    ))

            unique_trial_layups = list(dict.fromkeys(trial_layups))

            with ThreadPoolExecutor() as executor:
                trial_results = list(executor.map(
                    lambda angles: score_layup_candidate(
                        angles,
                        target_response,
                        response_scale,
                        z_thermal,
                        temperature_profile,
                        reference_temp,
                    ),
                    unique_trial_layups,
                ))

            beam = sorted(trial_results, key=lambda item: item["error"])[:BEAM_WIDTH]

        improvement = previous_best_error - beam[0]["error"]
        if improvement < INVERSE_TOLERANCE:
            break

    best = beam[0]
    best["iterations"] = iteration
    best["beam_width"] = BEAM_WIDTH
    best["beam"] = beam
    return best


def solve_layup_inverse(
    target_response,
    z_thermal,
    temperature_profile,
    reference_temp,
    num_plies,
):
    """Dispatch to the configured inverse solver."""
    if SOLVER_METHOD == "beam":
        return inverse_solve_layup_from_gradient_beam(
            target_response=target_response,
            z_thermal=z_thermal,
            temperature_profile=temperature_profile,
            reference_temp=reference_temp,
            num_plies=num_plies,
        )

    return inverse_solve_layup_from_gradient(
        target_response=target_response,
        z_thermal=z_thermal,
        temperature_profile=temperature_profile,
        reference_temp=reference_temp,
        num_plies=num_plies,
    )


def random_valid_layup(num_plies, rng):
    """Generate one random layup that respects the configured constraints."""
    if ASSUME_SYMMETRIC_LAYUP:
        independent_count = (num_plies + 1) // 2
        independent = rng.choice(CANDIDATE_ANGLES, size=independent_count)
        return make_symmetric_layup(independent, num_plies)

    return list(rng.choice(CANDIDATE_ANGLES, size=num_plies))


def get_validation_temperature_profile(num_plies):
    """Compute one known temperature profile for a given laminate thickness."""
    dummy_layup = [0] * num_plies
    _, results, _ = run_thermal_diffusion_analysis(
        layup_angles=dummy_layup,
        n_time_steps=N_TIME_STEPS,
        n_spatial_nodes=N_SPATIAL_NODES,
    )
    return results["z"], results["T"][-1]


def validate_inverse_solver(
    ply_counts=VALIDATION_PLY_COUNTS,
    cases_per_ply_count=VALIDATION_CASES_PER_PLY_COUNT,
    random_seed=VALIDATION_RANDOM_SEED,
):
    """
    Run random synthetic inverse checks and report failures per ply count.

    The report separates exact matches, equivalent response matches, and true
    response failures. A different layup can sometimes reproduce the same
    response, which means the inverse signature is non-unique rather than wrong.
    """
    rng = np.random.default_rng(random_seed)
    report = []

    print("\n" + "=" * 80)
    print("INVERSE SOLVER VALIDATION")
    print("=" * 80)
    print(f"Solver method: {SOLVER_METHOD}")
    print(f"Beam width: {BEAM_WIDTH if SOLVER_METHOD == 'beam' else 'n/a'}")
    print(f"Assume symmetric layup: {ASSUME_SYMMETRIC_LAYUP}")
    print(f"Cases per ply count: {cases_per_ply_count}")
    print(f"Candidate angles: {CANDIDATE_ANGLES}")

    for num_plies in ply_counts:
        z_thermal, temperature_profile = get_validation_temperature_profile(num_plies)
        equivalent_matches = []
        response_failures = []

        for case_index in range(1, cases_per_ply_count + 1):
            target_layup = random_valid_layup(num_plies, rng)
            target_result = response_from_temperature_profile(
                target_layup,
                z_thermal,
                temperature_profile,
                INITIAL_TEMP,
            )

            best = solve_layup_inverse(
                target_response=target_result["response"],
                z_thermal=z_thermal,
                temperature_profile=temperature_profile,
                reference_temp=INITIAL_TEMP,
                num_plies=num_plies,
            )

            exact_match = best["layup"] == target_layup
            response_match = best["error"] <= VALIDATION_RESPONSE_ERROR_TOLERANCE
            failure_detail = {
                    "case": case_index,
                    "target": target_layup,
                    "predicted": best["layup"],
                    "error": best["error"],
                    "strain_error": best["strain_error"],
                    "curvature_error": best["curvature_error"],
                }

            if response_match and not exact_match:
                equivalent_matches.append(failure_detail)
            elif not response_match:
                response_failures.append(failure_detail)

        equivalent_count = len(equivalent_matches)
        response_failure_count = len(response_failures)
        exact_pass_count = cases_per_ply_count - equivalent_count - response_failure_count
        response_pass_count = cases_per_ply_count - response_failure_count
        report_row = {
            "num_plies": num_plies,
            "cases": cases_per_ply_count,
            "exact_passes": exact_pass_count,
            "equivalent_matches": equivalent_count,
            "response_passes": response_pass_count,
            "response_failures": response_failure_count,
            "equivalent_match_details": equivalent_matches,
            "response_failure_details": response_failures,
        }
        report.append(report_row)

        print(
            f"\n{num_plies} plies: "
            f"exact_passes={exact_pass_count}, "
            f"equivalent_matches={equivalent_count}, "
            f"response_failures={response_failure_count}"
        )
        for failure in equivalent_matches[:5]:
            print(
                f"  equivalent case {failure['case']:2d}: "
                f"original={failure['target']} "
                f"predicted={failure['predicted']} "
                f"error={failure['error']:.3e}"
            )
        if len(equivalent_matches) > 5:
            print(f"  ... {len(equivalent_matches) - 5} more equivalent matches")

        for failure in response_failures[:5]:
            print(
                f"  failure case {failure['case']:2d}: "
                f"original={failure['target']} "
                f"predicted={failure['predicted']} "
                f"error={failure['error']:.3e}"
            )
        if len(response_failures) > 5:
            print(f"  ... {len(response_failures) - 5} more response failures")

    total_cases = sum(row["cases"] for row in report)
    total_equivalent = sum(row["equivalent_matches"] for row in report)
    total_response_failures = sum(row["response_failures"] for row in report)
    print("\n" + "-" * 80)
    print(
        f"Total: cases={total_cases}, "
        f"exact_passes={total_cases - total_equivalent - total_response_failures}, "
        f"equivalent_matches={total_equivalent}, "
        f"response_failures={total_response_failures}"
    )
    print("=" * 80 + "\n")

    return report


def print_response(label, response):
    """Print membrane strains and curvatures."""
    print(f"\n{label}")
    print(f"  eps_x    = {response[0]:+.6e}")
    print(f"  eps_y    = {response[1]:+.6e}")
    print(f"  gamma_xy = {response[2]:+.6e}")
    print(f"  kappa_x  = {response[3]:+.6e} 1/m")
    print(f"  kappa_y  = {response[4]:+.6e} 1/m")
    print(f"  kappa_xy = {response[5]:+.6e} 1/m")


def print_thermal_diffusion_report(analyzer, results, laminate):
    """Print the detailed report from thermal_diffusion_example.py."""
    print("\n" + "=" * 80)
    print("TRANSIENT THERMAL DIFFUSION ANALYSIS")
    print("=" * 80)

    print("\nLaminate Configuration:")
    print(f"  Material: {laminate.material.name}")
    print(f"  Layup: {laminate.layup.angles}")
    print(f"  Number of plies: {laminate.layup.num_plies}")
    print(f"  Total thickness: {laminate.total_thickness * 1e3:.3f} mm")
    print(f"  Ply thickness: {laminate.ply_thickness * 1e6:.1f} um")

    print("\nThermal Configuration:")
    print(f"  Initial temperature: {analyzer.initial_temp}")
    print(f"  Top surface temperature: {analyzer.top_temp}")
    print(f"  Temperature rise: {analyzer.top_temp - analyzer.initial_temp}")
    print(f"  Analysis duration: {results['t'][-1]:.6f} s")

    analyzer.print_summary()

    t = results["t"]
    T = results["T"]
    strains = results["strains"]
    curvatures = results["curvatures"]

    print("\n--- Temperature Evolution ---")
    time_indices = np.linspace(0, len(t) - 1, 6, dtype=int)
    for idx in time_indices:
        T_avg = np.mean(T[idx])
        T_max = np.max(T[idx])
        T_min = np.min(T[idx])
        strain_x = strains[idx, 0]
        print(
            f"  t = {t[idx]:10.6f} s: "
            f"T_avg = {T_avg:6.2f}, "
            f"T_range = [{T_min:6.2f}, {T_max:6.2f}], "
            f"eps_x = {strain_x:8.3e}"
        )

    print("\n--- Final Response ---")
    print(f"  eps_x    = {strains[-1, 0]:+.6e}")
    print(f"  eps_y    = {strains[-1, 1]:+.6e}")
    print(f"  gamma_xy = {strains[-1, 2]:+.6e}")
    print(f"  kappa_x  = {curvatures[-1, 0]:+.6e} 1/m")
    print(f"  kappa_y  = {curvatures[-1, 1]:+.6e} 1/m")
    print(f"  kappa_xy = {curvatures[-1, 2]:+.6e} 1/m")


def save_temperature_plot(analyzer, output_name="inverse_solving_thermal_diffusion.png"):
    """Save the thermal diffusion plot if matplotlib is installed."""
    plot_result = analyzer.plot_temperature_evolution()
    if plot_result is None:
        print("Matplotlib not installed. Install with: pip install matplotlib")
        return None

    fig, _ = plot_result
    output_path = Path(__file__).parent / output_name
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
    return output_path


# ============================================================================
# RUN AS A SCRIPT
# ============================================================================

if __name__ == "__main__":
    analyzer, results, laminate = run_thermal_diffusion_analysis()
    print_thermal_diffusion_report(analyzer, results, laminate)
    save_temperature_plot(analyzer)

    target_response = np.concatenate((
        results["strains"][-1],
        results["curvatures"][-1],
    ))

    best_layup = solve_layup_inverse(
        target_response=target_response,
        z_thermal=results["z"],
        temperature_profile=results["T"][-1],
        reference_temp=INITIAL_TEMP,
        num_plies=len(LAYUP),
    )

    print("\n" + "=" * 80)
    print("INVERSE SOLVE: PREDICT LAYUP FROM GRADIENT RESPONSE")
    print("=" * 80)
    print(f"Target layup used to create response: {LAYUP}")
    print(f"Assume symmetric layup: {ASSUME_SYMMETRIC_LAYUP}")
    if ASSUME_SYMMETRIC_LAYUP and LAYUP != symmetrize_layup(LAYUP):
        print("Warning: target layup is not symmetric, so exact recovery is not allowed.")
    print(f"Predicted layup: {best_layup['layup']}")
    print(f"Solver method: {SOLVER_METHOD}")
    if SOLVER_METHOD == "beam":
        print(f"Beam width: {best_layup['beam_width']}")
    print(f"Candidate angles: {CANDIDATE_ANGLES}")
    print(f"Converged error: {best_layup['error']:.6e}")
    print(f"Strain-only normalized error: {best_layup['strain_error']:.6e}")
    print(f"Curvature-only normalized error: {best_layup['curvature_error']:.6e}")
    if "start_index" in best_layup:
        print(f"Best start index: {best_layup['start_index']}")
    print(f"Iterations: {best_layup['iterations']}")
    print_response("TARGET RESPONSE", target_response)
    print_response("PREDICTED RESPONSE", best_layup["response"])

    print("\nComponent normalized absolute errors:")
    component_names = ["eps_x", "eps_y", "gamma_xy", "kappa_x", "kappa_y", "kappa_xy"]
    for name, error in zip(component_names, best_layup["component_errors"]):
        print(f"  {name:8s}: {error:.6e}")

    print("\nResponse scales from measured target vector:")
    for name, scale in zip(component_names, normalize_response(target_response)):
        print(f"  {name:8s}: {scale:.6e}")

    if "beam" in best_layup:
        print("\nTop beam candidates:")
        for rank, candidate in enumerate(best_layup["beam"][:10], start=1):
            print(
                f"  {rank:2d}. error={candidate['error']:.6e} "
                f"layup={candidate['layup']}"
            )

    print("\nBase material Q matrix:")
    print(best_layup["Q"])

    print("\nConverged N_thermal:")
    print(best_layup["N_thermal"])

    print("\nConverged M_thermal:")
    print(best_layup["M_thermal"])

    print("\nConverged ABD matrix:")
    print(best_layup["ABD"])
    print("=" * 80 + "\n")

    if RUN_VALIDATION_CASES:
        validate_inverse_solver()
