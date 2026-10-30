"""
Guess the symmetric layup for the same case used in realisticconfirmation.py.

Known inputs:
    - plate dimensions
    - material properties
    - ply thickness
    - number of plies
    - thermal boundary conditions
    - allowed candidate ply angles

Unknown input:
    - layup sequence

The hidden layup is used only to create a synthetic measured response. The
inverse search then exhaustively checks every symmetric layup with the known
number of plies and ranks candidates by strain/curvature response error.
"""

import itertools

import numpy as np


# =============================================================================
# 1. SAME INPUT PARAMETERS AS realisticconfirmation.py
# =============================================================================

L = 0.3              # Plate length along X [m]
W = 0.3              # Plate width along Y [m]
t_ply = 0.000125     # Single ply thickness [m]

# This is hidden from the inverse solver. It only creates the synthetic target.
HIDDEN_LAYUP = [0,0,0]
n_plies = len(HIDDEN_LAYUP)

# Material Mechanical Properties (Carbon/Epoxy)
E1 = 130e9           # Fiber direction modulus [Pa]
E2 = 10e9            # Transverse modulus [Pa]
nu12 = 0.3           # Major Poisson's ratio
G12 = 5e9            # In-plane shear modulus [Pa]

# Material Thermal Properties
alpha1 = -0.5e-6     # Fiber CTE [/C]
alpha2 = 30e-6       # Transverse CTE [/C]
kz = 0.5             # Transverse thermal conductivity [W/(m*K)]

# Thermal Boundary Conditions
T_init = 25.0        # Initial stress-free temp [C]
T_bot_sink = 100.0   # Hot plate temperature [C]
T_top_sink = 0.0     # Air knife temperature [C]
h_bot = 1000.0       # Bottom contact film coefficient [W/(m^2*K)]
h_top = 400.0        # Top air knife film coefficient [W/(m^2*K)]

# Guessing settings
CANDIDATE_ANGLES = [0, 45, -45, 90]
STRAIN_SCALE_FLOOR = 1e-6
CURVATURE_SCALE_FLOOR = 1e-3


# =============================================================================
# 2. SAME THERMAL GRADIENT & HEAT FLUX AS realisticconfirmation.py
# =============================================================================

h = n_plies * t_ply  # Total laminate thickness [m]

R_bot = 1.0 / h_bot
R_cond = h / kz
R_top = 1.0 / h_top
R_total = R_bot + R_cond + R_top

q_flux = (T_bot_sink - T_top_sink) / R_total

T_bot_surf = T_bot_sink - (q_flux * R_bot)
T_top_surf = T_top_sink + (q_flux * R_top)

T_mid = 0.5 * (T_top_surf + T_bot_surf)
dT_0 = T_mid - T_init
beta = (T_top_surf - T_bot_surf) / h


# =============================================================================
# 3. CLT HELPERS MATCHING realisticconfirmation.py
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


def transformed_ply_matrices(angle):
    """Return Q_bar and alpha_bar using the exact transforms from the base case."""
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
    for angle in sorted(set(CANDIDATE_ANGLES) | set(HIDDEN_LAYUP))
}


def calculate_response_for_layup(layup):
    """
    Return ABD, thermal loads, and [eps0, kappa] for one candidate layup.

    This duplicates the ABD/N_T/M_T integration from realisticconfirmation.py,
    with only the layup variable changed.
    """
    z_coords = np.linspace(-h / 2.0, h / 2.0, n_plies + 1)

    A = np.zeros((3, 3), dtype=float)
    B = np.zeros((3, 3), dtype=float)
    D = np.zeros((3, 3), dtype=float)
    N_T = np.zeros(3, dtype=float)
    M_T = np.zeros(3, dtype=float)

    for i, angle in enumerate(layup):
        q_bar, alpha_bar = PLY_LOOKUP[angle]

        z_k = z_coords[i + 1]
        z_km1 = z_coords[i]

        A += q_bar * (z_k - z_km1)
        B += 0.5 * q_bar * (z_k**2 - z_km1**2)
        D += (1.0 / 3.0) * q_bar * (z_k**3 - z_km1**3)

        dt_k = dT_0 * (z_k - z_km1) + 0.5 * beta * (z_k**2 - z_km1**2)
        dt_m = (
            0.5 * dT_0 * (z_k**2 - z_km1**2)
            + (1.0 / 3.0) * beta * (z_k**3 - z_km1**3)
        )

        N_T += (q_bar @ alpha_bar) * dt_k
        M_T += (q_bar @ alpha_bar) * dt_m

    ABD = np.block(
        [
            [A, B],
            [B, D],
        ]
    )
    response = np.linalg.solve(ABD, np.concatenate([N_T, M_T]))

    return {
        "layup": list(layup),
        "A": A,
        "B": B,
        "D": D,
        "ABD": ABD,
        "N_T": N_T,
        "M_T": M_T,
        "response": response,
        "eps_0": response[:3],
        "kappa": response[3:],
    }


# =============================================================================
# 4. SYMMETRIC LAYUP SEARCH
# =============================================================================

def symmetric_base_length(num_plies):
    """Return the number of independent plies in a symmetric laminate."""
    return (num_plies + 1) // 2


def make_symmetric_layup(independent_angles, num_plies):
    """Mirror independent angles into a complete symmetric layup."""
    independent_angles = list(independent_angles)
    half_count = num_plies // 2

    if num_plies % 2:
        return (
            independent_angles[:half_count]
            + [independent_angles[half_count]]
            + independent_angles[:half_count][::-1]
        )

    return independent_angles[:half_count] + independent_angles[:half_count][::-1]


def iter_symmetric_layup_candidates(num_plies):
    """Yield every symmetric candidate layup with the known ply count."""
    base_count = symmetric_base_length(num_plies)
    for independent_angles in itertools.product(CANDIDATE_ANGLES, repeat=base_count):
        yield make_symmetric_layup(independent_angles, num_plies)


def response_scale(response):
    """Scale strain and curvature components separately for scoring."""
    scale = np.abs(response)
    scale[:3] = np.maximum(scale[:3], STRAIN_SCALE_FLOOR)
    scale[3:] = np.maximum(scale[3:], CURVATURE_SCALE_FLOOR)
    return scale


def score_candidate(candidate_layup, target_response, scale):
    """Return a ranked inverse-search result for one candidate layup."""
    result = calculate_response_for_layup(candidate_layup)
    residual = (result["response"] - target_response) / scale
    result["normalized_residual"] = residual
    result["error"] = float(np.linalg.norm(residual))
    result["strain_error"] = float(np.linalg.norm(residual[:3]))
    result["curvature_error"] = float(np.linalg.norm(residual[3:]))
    result["max_abs_response_delta"] = float(
        np.max(np.abs(result["response"] - target_response))
    )
    return result


def guess_layup_from_known_parameters(target_response):
    """
    Guess the layup using every known parameter except the layup itself.

    The inverse solver is told only:
        - n_plies
        - candidate angle set
        - material properties
        - ply thickness
        - thermal boundary conditions
        - symmetry
    """
    scale = response_scale(target_response)
    ranked = [
        score_candidate(candidate_layup, target_response, scale)
        for candidate_layup in iter_symmetric_layup_candidates(n_plies)
    ]
    return sorted(ranked, key=lambda item: item["error"])


# =============================================================================
# 5. REPORTING
# =============================================================================

def format_response_vector(label, response):
    """Return compact lines for one [strain, curvature] response vector."""
    return "\n".join(
        [
            label,
            f"  eps_x    = {response[0]:+.6e}",
            f"  eps_y    = {response[1]:+.6e}",
            f"  gamma_xy = {response[2]:+.6e}",
            f"  kappa_x  = {response[3]:+.6e} 1/m",
            f"  kappa_y  = {response[4]:+.6e} 1/m",
            f"  kappa_xy = {response[5]:+.6e} 1/m",
        ]
    )


def format_abd_block(abd):
    """Return a readable 6x6 ABD matrix with A/B/D block separators."""
    lines = []
    horizontal_rule = "  " + "-" * 91

    for row_index, row in enumerate(abd):
        if row_index == 3:
            lines.append(horizontal_rule)

        left = " ".join(f"{value: .4e}" for value in row[:3])
        right = " ".join(f"{value: .4e}" for value in row[3:])
        lines.append(f"  {left} | {right}")

    return "\n".join(lines)


def print_report(target_result, ranked_results):
    """Print the synthetic target and inverse-search result."""
    best = ranked_results[0]
    exact_match = best["layup"] == HIDDEN_LAYUP
    search_space_size = len(CANDIDATE_ANGLES) ** symmetric_base_length(n_plies)

    print("\n" + "=" * 80)
    print("REALISTIC CONFIRMATION LAYUP GUESS")
    print("=" * 80)

    print("\nKnown inputs used by the guessing script:")
    print(f"  L={L:.6f} m, W={W:.6f} m")
    print(f"  Number of plies={n_plies}")
    print(f"  Ply thickness={t_ply:.6e} m")
    print(
        f"  E1={E1:.3e} Pa, E2={E2:.3e} Pa, "
        f"nu12={nu12:.3f}, G12={G12:.3e} Pa"
    )
    print(f"  alpha1={alpha1:.3e} 1/C, alpha2={alpha2:.3e} 1/C, kz={kz:.3f}")
    print(
        f"  T_init={T_init:.2f} C, T_bot_sink={T_bot_sink:.2f} C, "
        f"T_top_sink={T_top_sink:.2f} C"
    )
    print(f"  h_bot={h_bot:.3f} W/(m^2*K), h_top={h_top:.3f} W/(m^2*K)")
    print(f"  Computed bottom surface temp={T_bot_surf:.6f} C")
    print(f"  Computed top surface temp={T_top_surf:.6f} C")
    print(f"  dT_0={dT_0:.6f} C, beta={beta:.6e} C/m")
    print(f"  Candidate angle pool={CANDIDATE_ANGLES}")
    print("  Symmetric layup assumed=True")

    print("\nSynthetic target used to test the guess:")
    print(f"  Hidden layup={HIDDEN_LAYUP}")
    print(format_response_vector("  Target response", target_result["response"]))

    print("\nInverse search result:")
    print(f"  Search space size={search_space_size:,}")
    print(f"  Searched candidates={len(ranked_results):,}")
    print(f"  Predicted layup={best['layup']}")
    print(f"  Exact layup match={exact_match}")
    print(f"  Normalized error={best['error']:.6e}")
    print(f"  Strain-only error={best['strain_error']:.6e}")
    print(f"  Curvature-only error={best['curvature_error']:.6e}")
    print(f"  Max abs response delta={best['max_abs_response_delta']:.6e}")
    print(format_response_vector("  Predicted response", best["response"]))

    print("\nPredicted ABD matrix:")
    print("  Columns 1-3 are A/B blocks; columns 4-6 are B/D blocks.")
    print("  Rows 1-3 are A/B blocks; rows 4-6 are B/D blocks.")
    print(format_abd_block(best["ABD"]))

    print("\nTop ranked candidates:")
    for rank, candidate in enumerate(ranked_results[:10], start=1):
        print(
            f"  {rank:2d}. error={candidate['error']:.6e} "
            f"layup={candidate['layup']}"
        )

    print("=" * 80 + "\n")


def run_confirmation():
    """Generate a hidden response, then guess the layup from known inputs."""
    target_result = calculate_response_for_layup(HIDDEN_LAYUP)
    ranked_results = guess_layup_from_known_parameters(target_result["response"])
    print_report(target_result, ranked_results)
    return {
        "target": target_result,
        "ranked_results": ranked_results,
        "best": ranked_results[0],
    }


if __name__ == "__main__":
    run_confirmation()
