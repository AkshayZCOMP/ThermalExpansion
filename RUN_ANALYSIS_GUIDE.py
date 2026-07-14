"""
QUICK GUIDE: How to Use run_analysis.py

This is your main front-facing script for running laminate analyses.
"""

# ============================================================================
# TO RUN AN ANALYSIS:
# ============================================================================
# 1. Open run_analysis.py
# 2. Edit the CONFIGURATION section (lines 8-45)
# 3. Save the file
# 4. Run: python run_analysis.py

# ============================================================================
# EASY CONFIGURATION CHANGES
# ============================================================================

# CHANGE MATERIAL:
# Replace the MATERIAL definition with different properties
# Example - Glass/Epoxy:
#   MATERIAL = MaterialProperties(
#       name="Glass/Epoxy",
#       E_11=50e9,
#       E_22=15e9,
#       v_12=0.25,
#       G_12=4e9,
#       alpha_1=10e-6,
#       alpha_2=50e-6
#   )

# CHANGE TEMPERATURE:
#   DELTA_T = -50      # Cool by 50 K
#   DELTA_T = 100      # Heat by 100 K
#   DELTA_T = 0        # Room temperature (no thermal effects)

# CHANGE LAYUP:
# Replace the LAYUP list with your desired angles (in degrees)

# Example 1 - 8-ply QI:
#   LAYUP = [0, 45, -45, 90, 90, -45, 45, 0]

# Example 2 - 16-ply QI:
#   LAYUP = [0, 45, -45, 90, 0, 45, -45, 90, 90, -45, 45, 0, 90, -45, 45, 0]

# Example 3 - Cross-ply:
#   LAYUP = [0, 90, 0, 90]

# Example 4 - Unidirectional:
#   LAYUP = [0, 0, 0, 0]

# Example 5 - Symmetric [0/45/-45]s:
#   LAYUP = [0, 45, -45, -45, 45, 0]

# Example 6 - Angle-ply [+45/-45]s:
#   LAYUP = [45, -45, -45, 45]

# CHANGE PLY THICKNESS:
#   PLY_THICKNESS = 0.125e-3      # 0.125 mm (standard)
#   PLY_THICKNESS = 0.1e-3        # 0.1 mm (thinner)
#   PLY_THICKNESS = 0.15e-3       # 0.15 mm (thicker)

# ============================================================================
# UNDERSTANDING THE OUTPUT
# ============================================================================

# A Matrix (Extensional Stiffness):
#   - Controls in-plane stiffness
#   - A_11 = stiffness in x-direction
#   - A_22 = stiffness in y-direction
#   - For QI laminates: A_11 ≈ A_22 (isotropic)

# B Matrix (Coupling):
#   - Should be ~0 for symmetric laminates
#   - Large values mean bending causes extension

# D Matrix (Bending Stiffness):
#   - Controls out-of-plane (bending) stiffness
#   - Similar to A matrix but for bending

# Thermal Effects:
#   - Negative = compressive (typical for cooling)
#   - Positive = tensile

# Strains:
#   - eps_x, eps_y: membrane strains
#   - gamma_xy: shear strain
#   - kappa_*: curvatures (bending)

# ============================================================================
# COMMON WORKFLOWS
# ============================================================================

# Workflow 1: Test different layup sequences
#   1. Run with [0, 45, -45, 90, 90, -45, 45, 0]  (8-ply QI)
#   2. Compare results
#   3. Run with [0, 90, 0, 90]  (cross-ply)
#   4. Compare results

# Workflow 2: Test thermal effects
#   1. Run with DELTA_T = 0   (room temp)
#   2. Run with DELTA_T = -50 (cooled)
#   3. Run with DELTA_T = 100 (heated)
#   4. Compare strains

# Workflow 3: Compare materials
#   1. Run with Carbon/Epoxy
#   2. Run with Glass/Epoxy
#   3. Compare A matrix values

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

# If you get "ModuleNotFoundError":
#   - Make sure you're running from HeatXferDIC folder
#   - Make sure numpy is installed: pip install numpy

# If results look wrong:
#   - Check that angles are in degrees (not radians)
#   - Check DELTA_T has correct sign
#   - Verify ply thickness is in meters (not mm)

# To get more detailed output:
#   - Use workflow.py instead (has full detailed output)
#   - Or modify run_analysis.py to add more print statements
