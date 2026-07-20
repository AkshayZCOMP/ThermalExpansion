"""
COMPOSITE LAMINATE ANALYSIS - MAIN RUNNER

Edit the configuration section below and run this script.

For steady-state thermal analysis: Keep ENABLE_TRANSIENT_THERMAL = False
For transient thermal diffusion: Set ENABLE_TRANSIENT_THERMAL = True
"""

import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.InputProperties import MaterialProperties, LayupSequence, LaminateProperties
from src.workflow import LaminateAnalysis

# ============================================================================
# CONFIGURATION - EDIT HERE
# ============================================================================

# ANALYSIS TYPE
ENABLE_TRANSIENT_THERMAL = True  # Set to True for transient thermal diffusion analysis
                                  # Set to False for steady-state thermal analysis
ENABLE_INVERSE_SOLVING = True     # Uses gradient strains/curvatures to recover a layup

# Material Properties
MATERIAL = MaterialProperties(
    name="Carbon/Epoxy",
    E_11=130e9,      # Young's modulus in fiber direction (Pa)
    E_22=10e9,       # Young's modulus in transverse direction (Pa)
    v_12=0.3,        # Poisson's ratio
    G_12=5e9,        # Shear modulus (Pa)
    alpha_1=-0.5e-6, # CTE in fiber direction (1/K)
    alpha_2=30e-6    # CTE in transverse direction (1/K)
)

# Temperature Change (Kelvin) - for STEADY-STATE analysis
DELTA_T = -50  # Negative = cooling, Positive = heating

# TRANSIENT THERMAL DIFFUSION PARAMETERS (only used if ENABLE_TRANSIENT_THERMAL = True)
INITIAL_TEMP = 25            # Initial temperature throughout laminate (C)
TOP_SURFACE_TEMP = 100       # Temperature imposed at top surface (C)
BOTTOM_SURFACE_TEMP = 25     # Temperature imposed at bottom surface (C)
THERMAL_DIFFUSIVITY = 1e-6   # Heat diffusivity α = k/(ρ*c) in m²/s
TRANSIENT_TIME = None        # None uses about 2*h^2/alpha for short transients
TRANSIENT_STEPS = 100        # Number of time steps

# INVERSE SOLVER PARAMETERS
CANDIDATE_ANGLES = list(range(-90, 91, 15))
INVERSE_RANDOM_STARTS = 20
INVERSE_MAX_ITERATIONS = 30
INVERSE_TOLERANCE = 1e-12
INVERSE_WEIGHT_CURVATURES = 1e-3

# Ply Thickness (meters)
PLY_THICKNESS = 0.125e-3  # 0.125 mm (standard)

# LAYUP SEQUENCE - Change this to modify fiber angles
# Format: [angle1, angle2, angle3, ...]
# Common examples:
#   [0, 45, -45, 90, 90, -45, 45, 0]        # 8-ply QI
#   [0, 45, -45, 90, 0, 45, -45, 90, ...]   # 16-ply QI
#   [0, 90, 0, 90]                          # Cross-ply
#   [0, 0, 0, 0]                            # Unidirectional
LAYUP = [90,0, 90] 

# ============================================================================
# INVERSE SOLVING HELPERS
# ============================================================================

def normalize_response(response):
    """Scale strain/curvature response so the inverse error is dimensionless."""
    response = np.asarray(response, dtype=float)
    scale = np.maximum(np.abs(response), 1e-12)
    scale[3:6] = np.maximum(scale[3:6], INVERSE_WEIGHT_CURVATURES)
    return scale


def build_candidate_analysis(layup_angles):
    """Calculate Q, Q_bar, ABD, and transformed CTEs for one candidate layup."""
    layup = LayupSequence(list(layup_angles), name="Inverse Candidate")
    laminate = LaminateProperties(
        material=MATERIAL,
        layup=layup,
        ply_thickness=PLY_THICKNESS,
        delta_T=0
    )
    analysis = LaminateAnalysis(laminate)
    analysis.calculate_transformed_matrices()
    analysis.calculate_laminate_stiffness_matrices()
    material_matrices = analysis.calculate_material_matrices()
    return laminate, analysis, material_matrices["Q"]


def response_from_temperature_profile(layup_angles, z_thermal, temperature_profile,
                                      reference_temp):
    """
    Calculate strain/curvature response for a candidate layup under a known
    through-thickness temperature profile.
    """
    laminate, analysis, Q = build_candidate_analysis(layup_angles)
    ply_count = laminate.layup.num_plies
    ply_thickness = laminate.ply_thickness
    total_thickness = laminate.total_thickness
    z_bottom = -total_thickness / 2

    N_thermal = np.zeros(3)
    M_thermal = np.zeros(3)

    for i in range(ply_count):
        z_i = z_bottom + i * ply_thickness
        z_ip1 = z_i + ply_thickness
        z_mechanical = np.linspace(z_i, z_ip1, 5)

        # Thermal z starts at the heated top surface; mechanical z is centered.
        z_for_temperature = total_thickness / 2 - z_mechanical
        delta_T = np.interp(z_for_temperature, z_thermal, temperature_profile) - reference_temp

        thermal_stress_per_degree = (
            analysis.Q_bar_list[i] @ analysis.alpha_bar_list[i]
        )
        N_thermal += thermal_stress_per_degree * np.trapezoid(delta_T, z_mechanical)
        M_thermal += thermal_stress_per_degree * np.trapezoid(
            delta_T * z_mechanical,
            z_mechanical
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


def score_candidate(layup_angles, target_response, response_scale,
                    z_thermal, temperature_profile, reference_temp):
    """Return inverse error and full candidate result."""
    result = response_from_temperature_profile(
        layup_angles,
        z_thermal,
        temperature_profile,
        reference_temp
    )
    residual = (result["response"] - target_response) / response_scale
    result["error"] = float(np.linalg.norm(residual))
    return result


def solve_layup_from_gradient_response(target_response, z_thermal, temperature_profile,
                                       reference_temp, num_plies):
    """
    Recover a candidate layup from gradient-induced strains/curvatures.

    Uses concurrent coordinate descent: for each ply, try all allowed angles in
    parallel, keep the best angle, and repeat until the ABD/Q_bar response
    stops improving.
    """
    rng = np.random.default_rng(7)
    response_scale = normalize_response(target_response)
    starts = [list(rng.choice(CANDIDATE_ANGLES, size=num_plies))]
    starts.append([0] * num_plies)
    starts.append([90, 0] * (num_plies // 2) + ([90] if num_plies % 2 else []))

    while len(starts) < INVERSE_RANDOM_STARTS:
        starts.append(list(rng.choice(CANDIDATE_ANGLES, size=num_plies)))

    best_overall = None

    for start_index, start in enumerate(starts, start=1):
        current_layup = start
        current = score_candidate(
            current_layup,
            target_response,
            response_scale,
            z_thermal,
            temperature_profile,
            reference_temp
        )

        for iteration in range(1, INVERSE_MAX_ITERATIONS + 1):
            previous_error = current["error"]

            for ply_index in range(num_plies):
                trial_layups = []
                for angle in CANDIDATE_ANGLES:
                    trial = current_layup.copy()
                    trial[ply_index] = angle
                    trial_layups.append(trial)

                with ThreadPoolExecutor() as executor:
                    trial_results = list(executor.map(
                        lambda angles: score_candidate(
                            angles,
                            target_response,
                            response_scale,
                            z_thermal,
                            temperature_profile,
                            reference_temp
                        ),
                        trial_layups
                    ))

                current = min(trial_results, key=lambda item: item["error"])
                current_layup = current["layup"]

            improvement = previous_error - current["error"]
            if improvement < INVERSE_TOLERANCE:
                break

        current["start_index"] = start_index
        current["iterations"] = iteration

        if best_overall is None or current["error"] < best_overall["error"]:
            best_overall = current

    return best_overall


def print_gradient_response(label, response):
    """Print strains and curvatures in a compact format."""
    print(f"\n{label}")
    print(f"  eps_x    = {response[0]:+.6e}")
    print(f"  eps_y    = {response[1]:+.6e}")
    print(f"  gamma_xy = {response[2]:+.6e}")
    print(f"  kappa_x  = {response[3]:+.6e} 1/m")
    print(f"  kappa_y  = {response[4]:+.6e} 1/m")
    print(f"  kappa_xy = {response[5]:+.6e} 1/m")


# ============================================================================
# RUN ANALYSIS
# ============================================================================

if __name__ == "__main__":
    if ENABLE_TRANSIENT_THERMAL:
        # TRANSIENT THERMAL DIFFUSION ANALYSIS
        from src.ThermalDiffusion import ThermalDiffusionAnalyzer
        
        # Create laminate configuration
        layup = LayupSequence(LAYUP, name="Custom Layup")
        laminate = LaminateProperties(
            material=MATERIAL,
            layup=layup,
            ply_thickness=PLY_THICKNESS,
            delta_T=0  # No net temperature for transient analysis
        )
        
        print("\n" + "="*80)
        print("TRANSIENT THERMAL DIFFUSION ANALYSIS")
        print("="*80)
        print(f"\nLayup: {LAYUP}")
        print(f"Laminate thickness: {laminate.total_thickness*1e3:.3f} mm")
        print(f"Initial temperature: {INITIAL_TEMP} C")
        print(f"Top surface temperature: {TOP_SURFACE_TEMP} C")
        print(f"Bottom surface temperature: {BOTTOM_SURFACE_TEMP} C")
        if TRANSIENT_TIME is None:
            estimated_time = 2.0 * laminate.total_thickness**2 / THERMAL_DIFFUSIVITY
            print(f"Analysis duration: auto ({estimated_time:.3f} s)")
        else:
            print(f"Analysis duration: {TRANSIENT_TIME} s ({TRANSIENT_TIME/60:.1f} min)")
        print(f"Thermal diffusivity: {THERMAL_DIFFUSIVITY:.2e} m²/s")
        
        # Run thermal diffusion analysis
        print(f"\nRunning transient thermal analysis...")
        analyzer = ThermalDiffusionAnalyzer(
            laminate_properties=laminate,
            top_surface_temp=TOP_SURFACE_TEMP,
            initial_temp=INITIAL_TEMP,
            bottom_surface_temp=BOTTOM_SURFACE_TEMP
        )
        
        results = analyzer.analyze(
            t_final=TRANSIENT_TIME,
            n_steps=TRANSIENT_STEPS,
            thermal_diffusivity=THERMAL_DIFFUSIVITY
        )
        
        # Print summary
        analyzer.print_summary()

        target_response = np.concatenate((
            results["strains"][-1],
            results["curvatures"][-1]
        ))
        print_gradient_response(
            "TARGET RESPONSE FROM TEMPERATURE GRADIENT",
            target_response
        )

        if ENABLE_INVERSE_SOLVING:
            print("\n" + "="*80)
            print("INVERSE SOLVE: LAYUP FROM GRADIENT RESPONSE")
            print("="*80)
            print(f"Known number of plies: {len(LAYUP)}")
            print(f"Candidate angles: {CANDIDATE_ANGLES}")
            print("Solving for the candidate Q_bar/ABD response...")

            best = solve_layup_from_gradient_response(
                target_response=target_response,
                z_thermal=results["z"],
                temperature_profile=results["T"][-1],
                reference_temp=INITIAL_TEMP,
                num_plies=len(LAYUP)
            )

            print(f"\nBest layup found: {best['layup']}")
            print(f"Target layup used to generate data: {LAYUP}")
            print(f"Converged error: {best['error']:.6e}")
            print(f"Best random start: {best['start_index']}")
            print(f"Iterations on that start: {best['iterations']}")
            print_gradient_response("PREDICTED RESPONSE FROM BEST LAYUP", best["response"])

            print("\nBase material Q matrix:")
            print(best["Q"])

            print("\nConverged A matrix:")
            print(best["A"])

            print("\nConverged B matrix:")
            print(best["B"])

            print("\nConverged D matrix:")
            print(best["D"])

            print("\nConverged ABD matrix:")
            print(best["ABD"])
        
    else:
        # STEADY-STATE THERMAL ANALYSIS
        # Create laminate configuration
        layup = LayupSequence(LAYUP, name="Custom Layup")
        laminate = LaminateProperties(
            material=MATERIAL,
            layup=layup,
            ply_thickness=PLY_THICKNESS,
            delta_T=DELTA_T
        )
        
        # Run analysis
        analysis = LaminateAnalysis(laminate)
        analysis.run_full_analysis()
        
        # Print clean results
        print("\n" + "="*80)
        print(f"LAMINATE ANALYSIS: {MATERIAL.name}")
        print("="*80)
        
        print(f"\nLayup: {LAYUP}")
        print(f"Number of plies: {len(LAYUP)}")
        print(f"Total thickness: {laminate.total_thickness*1e3:.3f} mm")
        print(f"Temperature change: {DELTA_T} K")
        
        print("\n" + "-"*80)
        print("STIFFNESS MATRICES")
        print("-"*80)
        print(f"\nA Matrix (Extensional Stiffness - N/m):")
        print(analysis.A_matrix)
        
        print(f"\nB Matrix (Coupling - N):")
        print(analysis.B_matrix)
        
        print(f"\nD Matrix (Bending Stiffness - N·m):")
        print(analysis.D_matrix)
        
        print("\n" + "-"*80)
        print("THERMAL EFFECTS")
        print("-"*80)
        print(f"\nThermal Force Vector (N/m): {analysis.N_thermal}")
        print(f"Thermal Moment Vector (N): {analysis.M_thermal}")
        
        print("\n" + "-"*80)
        print("RESULTANT STRAINS & CURVATURES")
        print("-"*80)
        
        eps_x, eps_y, gamma_xy = analysis.resultant_strains[0:3]
        kappa_x, kappa_y, kappa_xy = analysis.resultant_strains[3:6]
        
        print(f"\nMembrane Strains:")
        print(f"  eps_x    = {eps_x:+.6e}")
        print(f"  eps_y    = {eps_y:+.6e}")
        print(f"  gamma_xy = {gamma_xy:+.6e}")
        
        print(f"\nCurvatures (1/m):")
        print(f"  kappa_x  = {kappa_x:+.6e}")
        print(f"  kappa_y  = {kappa_y:+.6e}")
        print(f"  kappa_xy = {kappa_xy:+.6e}")
        
        print("\n" + "-"*80)
        print("PLY STRAINS (Global Coordinates)")
        print("-"*80)
        for i, strain in enumerate(analysis.ply_strains):
            angle = LAYUP[i]
            print(f"Ply {i+1:2d} ({angle:+3.0f}°): eps_x={strain[0]:+.3e}  eps_y={strain[1]:+.3e}  gamma_xy={strain[2]:+.3e}")
        
        print("\n" + "="*80 + "\n")
