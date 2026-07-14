"""
COMPOSITE LAMINATE ANALYSIS - MAIN RUNNER

Edit the configuration section below and run this script.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.InputProperties import MaterialProperties, LayupSequence, LaminateProperties
from src.workflow import LaminateAnalysis

# ============================================================================
# CONFIGURATION - EDIT HERE
# ============================================================================

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

# Temperature Change (Kelvin)
DELTA_T = -50  # Negative = cooling, Positive = heating

# Ply Thickness (meters)
PLY_THICKNESS = 0.125e-3  # 0.125 mm (standard)

# LAYUP SEQUENCE - Change this to modify fiber angles
# Format: [angle1, angle2, angle3, ...]
# Common examples:
#   [0, 45, -45, 90, 90, -45, 45, 0]        # 8-ply QI
#   [0, 45, -45, 90, 0, 45, -45, 90, ...]   # 16-ply QI
#   [0, 90, 0, 90]                          # Cross-ply
#   [0, 0, 0, 0]                            # Unidirectional
LAYUP = [5, 0, 0, 0, 0, 0, 0, 5]

# ============================================================================
# RUN ANALYSIS
# ============================================================================

if __name__ == "__main__":
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
