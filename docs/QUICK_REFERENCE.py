"""
Quick Reference Guide - Composite Laminate Analysis

One-page cheat sheet for common operations
"""

# ============================================================================
# IMPORTS
# ============================================================================

from InputProperties import (
    create_standard_QI_8ply,
    create_standard_QI_16ply,
    create_standard_QI_with_temperature,
    MaterialProperties,
    LayupSequence,
    LaminateProperties
)
from workflow import LaminateAnalysis, run_standard_QI_analysis


# ============================================================================
# 1. SIMPLEST: Standard QI Analysis
# ============================================================================

# One-liner analysis with auto-print
analysis = run_standard_QI_analysis(create_standard_QI_8ply(), verbose=True)


# ============================================================================
# 2. WITH THERMAL LOADING
# ============================================================================

# Cool by 50K
laminate = create_standard_QI_with_temperature(delta_T=-50)
analysis = run_standard_QI_analysis(laminate, verbose=True)

# Access results directly
strains = analysis.resultant_strains
ply_strains = analysis.ply_strains
a_matrix = analysis.A_matrix


# ============================================================================
# 3. CUSTOM MATERIAL
# ============================================================================

material = MaterialProperties(
    name="My Material",
    E_11=100e9,
    E_22=10e9,
    v_12=0.3,
    G_12=5e9,
    alpha_1=-1e-6,
    alpha_2=40e-6
)


# ============================================================================
# 4. CUSTOM LAYUP
# ============================================================================

# Quasi-isotropic
qi = LayupSequence.create_QI(num_plies_per_set=8)

# Symmetric
sym = LayupSequence.create_symmetric([0, 45, -45])

# Custom angles
custom = LayupSequence([0, 90, 0, 90], name="Cross-ply")


# ============================================================================
# 5. CREATE LAMINATE
# ============================================================================

laminate = LaminateProperties(
    material=material,
    layup=qi,
    ply_thickness=0.125e-3,  # meters
    delta_T=-50  # Kelvin
)


# ============================================================================
# 6. RUN ANALYSIS
# ============================================================================

analysis = LaminateAnalysis(laminate)
analysis.run_full_analysis()  # Calls all calculations


# ============================================================================
# 7. ACCESS RESULTS
# ============================================================================

# Stiffness matrices
A_matrix = analysis.A_matrix       # (3x3) Extensional
B_matrix = analysis.B_matrix       # (3x3) Coupling
D_matrix = analysis.D_matrix       # (3x3) Bending
ABD_matrix = analysis.ABD_matrix   # (6x6) Combined

# Thermal loads
N_thermal = analysis.N_thermal     # (3,) Force vector
M_thermal = analysis.M_thermal     # (3,) Moment vector

# Strains & curvatures
strains_all = analysis.resultant_strains  # (6,)
# [eps_x, eps_y, gamma_xy, kappa_x, kappa_y, kappa_xy]

eps_x = analysis.resultant_strains[0]
eps_y = analysis.resultant_strains[1]
gamma_xy = analysis.resultant_strains[2]
kappa_x = analysis.resultant_strains[3]
kappa_y = analysis.resultant_strains[4]
kappa_xy = analysis.resultant_strains[5]

# Individual ply strains
ply_strains = analysis.ply_strains  # List of (3,) arrays
# Each: [eps_x, eps_y, gamma_xy]

for i, strain in enumerate(ply_strains):
    angle = analysis.layup.angles[i]
    print(f"Ply {i+1} ({angle}deg): eps_x={strain[0]:.3e}")


# ============================================================================
# 8. PRINT RESULTS
# ============================================================================

print(analysis.get_summary())


# ============================================================================
# 9. COMPARE LAYUPS
# ============================================================================

laminates_to_compare = [
    ("8-ply QI", create_standard_QI_8ply()),
    ("16-ply QI", create_standard_QI_16ply()),
    ("8-ply QI -50K", create_standard_QI_with_temperature(-50)),
]

for name, lam in laminates_to_compare:
    ana = LaminateAnalysis(lam)
    ana.run_full_analysis()
    print(f"{name:20s} A11={ana.A_matrix[0,0]:.2e} dT={lam.delta_T}K")


# ============================================================================
# 10. INTERMEDIATE ACCESS (STEP-BY-STEP)
# ============================================================================

analysis = LaminateAnalysis(laminate)

# Calculate just the transformed matrices
analysis.calculate_transformed_matrices()
# Now access: analysis.Q_bar_list, analysis.alpha_bar_list

# Add stiffness matrices
analysis.calculate_laminate_stiffness_matrices()
# Now access: analysis.A_matrix, analysis.B_matrix, analysis.D_matrix

# Add thermal loading
analysis.calculate_thermal_loading()
# Now access: analysis.N_thermal, analysis.M_thermal

# Calculate strains
analysis.calculate_resultant_strains()
# Now access: analysis.resultant_strains

# Calculate ply strains
analysis.calculate_ply_strains()
# Now access: analysis.ply_strains


# ============================================================================
# 11. TYPICAL PARAMETERS
# ============================================================================

# Material: Carbon/Epoxy
E_11_carbon = 130e9  # GPa = 130
E_22_carbon = 10e9
G_12_carbon = 5e9
CTE_fiber_carbon = -0.5e-6  # /K
CTE_matrix_carbon = 30e-6   # /K

# Material: Glass/Epoxy
E_11_glass = 50e9
E_22_glass = 15e9
G_12_glass = 4e9
CTE_fiber_glass = 10e-6
CTE_matrix_glass = 50e-6

# Typical ply thickness: 0.125 mm (0.005 inches)
ply_thickness = 0.125e-3  # meters

# Typical temperature changes
cool_from_cure = -50  # K
typical_service_temp = -30  # K


# ============================================================================
# 12. COMMON LAYUPS
# ============================================================================

# Quasi-Isotropic (approximate)
QI_8 = LayupSequence([0, 45, -45, 90, 90, -45, 45, 0], "QI [8]")
QI_4 = LayupSequence([0, 45, -45, 90], "QI [4]")

# Symmetric laminates
SYM_0_45 = LayupSequence.create_symmetric([0, 45, -45])
SYM_0_90 = LayupSequence.create_symmetric([0, 90])

# Cross-ply
CROSS = LayupSequence([0, 90, 0, 90], "Cross-ply")

# Unidirectional
UNI = LayupSequence([0, 0, 0, 0], "Unidirectional")


# ============================================================================
# 13. INTERPRETATION GUIDE
# ============================================================================

# A_11 >> A_22: Fiber-dominated in x-direction
# A_11 ≈ A_22: Nearly isotropic (QI layups)
# |B_ij| << |A_ij|: Symmetric laminate (good!)
# B_ij ≈ 0: No bending-extension coupling
# D_11 << A_11: Bending is much softer than extension


# ============================================================================
# 14. THERMAL EFFECTS INTERPRETATION
# ============================================================================

# delta_T < 0: Cooling (typical manufacturing to room temperature)
# delta_T > 0: Heating (typical service temperature increase)
# eps_x < 0: Compressive strain in x (common in carbon/epoxy cooling)
# kappa_x != 0: Laminate will warp/curve (if B_matrix >> 0)


# ============================================================================
# 15. RUN ALL TESTS
# ============================================================================

# In terminal:
# python test_cases.py

# Expected: 44 tests, 0 failures, 0 errors


# ============================================================================
# SHORTCUTS FOR MOST COMMON USE CASES
# ============================================================================

# CASE 1: Just give me the strains for standard QI with cooling
laminate = create_standard_QI_with_temperature(delta_T=-50)
ana = LaminateAnalysis(laminate)
ana.run_full_analysis()
print(f"Strains: {ana.resultant_strains[0:3]}")

# CASE 2: Compare A_11 values
for n_plies in [8, 16]:
    lam = LaminateProperties(MaterialProperties(), LayupSequence.create_QI(n_plies))
    ana = LaminateAnalysis(lam)
    ana.calculate_transformed_matrices()
    ana.calculate_laminate_stiffness_matrices()
    print(f"{n_plies}-ply: A_11 = {ana.A_matrix[0,0]:.2e}")

# CASE 3: Full summary for quick review
analysis = run_standard_QI_analysis(create_standard_QI_8ply())
