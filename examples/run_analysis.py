"""
COMPOSITE LAMINATE ANALYSIS - MAIN RUNNER

Edit the configuration section below and run this script.

For steady-state thermal analysis: Keep ENABLE_TRANSIENT_THERMAL = False
For transient thermal diffusion: Set ENABLE_TRANSIENT_THERMAL = True
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

# ANALYSIS TYPE
ENABLE_TRANSIENT_THERMAL = True  # Set to True for transient thermal diffusion analysis
                                  # Set to False for steady-state thermal analysis

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

# Ply Thickness (meters)
PLY_THICKNESS = 0.125e-3  # 0.125 mm (standard)

# LAYUP SEQUENCE - Change this to modify fiber angles
# Format: [angle1, angle2, angle3, ...]
# Common examples:
#   [0, 45, -45, 90, 90, -45, 45, 0]        # 8-ply QI
#   [0, 45, -45, 90, 0, 45, -45, 90, ...]   # 16-ply QI
#   [0, 90, 0, 90]                          # Cross-ply
#   [0, 0, 0, 0]                            # Unidirectional
LAYUP = [90,0] 

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
