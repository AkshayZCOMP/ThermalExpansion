"""
THERMAL DIFFUSION ANALYSIS - TRANSIENT HEAT DIFFUSION THROUGH LAMINATE THICKNESS

This example demonstrates how to:
1. Set up a composite laminate
2. Model heat diffusion from a heated top surface through the thickness
3. Calculate resulting strains and curvatures over time
4. Analyze steady-state thermal response
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.InputProperties import MaterialProperties, LayupSequence, LaminateProperties
from src.ThermalDiffusion import ThermalDiffusionAnalyzer

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

# Laminate Configuration
PLY_THICKNESS = 0.125e-3  # 0.125 mm (standard)
LAYUP = [0, 45, -45, 90, 90, -45, 45, 0]  # 8-ply QI layup

# Thermal Boundary Conditions
INITIAL_TEMP = 20          # Initial uniform temperature throughout (°C or K)
TOP_SURFACE_TEMP = 100     # Temperature imposed at top surface (°C or K)
ANALYSIS_TIME = 10         # Total analysis time (seconds) - SHORT for visible changes!
N_TIME_STEPS = 100         # Number of time steps (0.1 sec per step)
N_SPATIAL_NODES = 50       # Number of nodes through thickness

# Material Thermal Properties
THERMAL_DIFFUSIVITY = 1e-3 # Thermal diffusivity α = k/(ρ*c) in m²/s
                            # INCREASED for visible transient effects!
                            # Real values (use these for realistic):
                            # - Carbon/Epoxy: 1e-6 to 2e-6 m²/s
                            # - Glass/Epoxy: 0.5e-6 to 1.5e-6 m²/s
                            # - We use 1e-3 to see transient in 10 seconds

# ============================================================================
# RUN THERMAL DIFFUSION ANALYSIS
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("TRANSIENT THERMAL DIFFUSION ANALYSIS")
    print("="*80)
    
    # Create laminate configuration
    layup = LayupSequence(LAYUP, name="QI Layup")
    laminate = LaminateProperties(
        material=MATERIAL,
        layup=layup,
        ply_thickness=PLY_THICKNESS,
        delta_T=0  # Start at zero for transient analysis
    )
    
    print(f"\nLaminate Configuration:")
    print(f"  Material: {MATERIAL.name}")
    print(f"  Layup: {LAYUP}")
    print(f"  Number of plies: {len(LAYUP)}")
    print(f"  Total thickness: {laminate.total_thickness*1e3:.3f} mm")
    print(f"  Ply thickness: {PLY_THICKNESS*1e6:.1f} µm")
    
    print(f"\nThermal Configuration:")
    print(f"  Initial temperature: {INITIAL_TEMP} K")
    print(f"  Top surface temperature: {TOP_SURFACE_TEMP} K")
    print(f"  Temperature rise: {TOP_SURFACE_TEMP - INITIAL_TEMP} K")
    print(f"  Analysis duration: {ANALYSIS_TIME} s ({ANALYSIS_TIME/60:.1f} min)")
    print(f"  Thermal diffusivity: {THERMAL_DIFFUSIVITY:.2e} m²/s")
    
    # Run thermal diffusion analysis
    print(f"\nRunning transient thermal analysis...")
    analyzer = ThermalDiffusionAnalyzer(
        laminate_properties=laminate,
        top_surface_temp=TOP_SURFACE_TEMP,
        initial_temp=INITIAL_TEMP
    )
    
    results = analyzer.analyze(
        t_final=ANALYSIS_TIME,
        n_steps=N_TIME_STEPS,
        n_nodes=N_SPATIAL_NODES,
        thermal_diffusivity=THERMAL_DIFFUSIVITY
    )
    
    # Print summary
    analyzer.print_summary()
    
    # Extract results
    t = results['t']
    z = results['z'] * 1e3  # Convert to mm
    T = results['T']
    strains = results['strains']
    curvatures = results['curvatures']
    
    # Print time evolution
    print(f"\n--- Temperature Evolution ---")
    time_indices = np.linspace(0, len(t)-1, 6, dtype=int)
    for idx in time_indices:
        T_avg = np.mean(T[idx])
        T_max = np.max(T[idx])
        T_min = np.min(T[idx])
        strain_x = strains[idx, 0]
        print(f"  t = {t[idx]:8.1f} s: T_avg = {T_avg:6.2f} K, " +
              f"T_range = [{T_min:6.2f}, {T_max:6.2f}] K, " +
              f"εₓ = {strain_x:8.3e}")
    
    # Strain evolution
    print(f"\n--- Membrane Strain Evolution ---")
    print(f"Initial strains:")
    print(f"  εₓ: {strains[0, 0]:.6e}")
    print(f"  εᵧ: {strains[0, 1]:.6e}")
    print(f"  γₓᵧ: {strains[0, 2]:.6e}")
    
    print(f"\nFinal (steady state) strains:")
    print(f"  εₓ: {strains[-1, 0]:.6e}")
    print(f"  εᵧ: {strains[-1, 1]:.6e}")
    print(f"  γₓᵧ: {strains[-1, 2]:.6e}")
    
    print(f"\nStrain changes (initial to steady state):")
    print(f"  Δεₓ: {strains[-1, 0] - strains[0, 0]:.6e}")
    print(f"  Δεᵧ: {strains[-1, 1] - strains[0, 1]:.6e}")
    print(f"  Δγₓᵧ: {strains[-1, 2] - strains[0, 2]:.6e}")
    
    # Curvature evolution
    print(f"\n--- Bending Curvature Evolution ---")
    print(f"Initial curvatures:")
    print(f"  κₓ: {curvatures[0, 0]:.6e} m⁻¹")
    print(f"  κᵧ: {curvatures[0, 1]:.6e} m⁻¹")
    print(f"  κₓᵧ: {curvatures[0, 2]:.6e} m⁻¹")
    
    print(f"\nFinal (steady state) curvatures:")
    print(f"  κₓ: {curvatures[-1, 0]:.6e} m⁻¹")
    print(f"  κᵧ: {curvatures[-1, 1]:.6e} m⁻¹")
    print(f"  κₓᵧ: {curvatures[-1, 2]:.6e} m⁻¹")
    
    # Attempt to plot if matplotlib is available
    try:
        print(f"\nGenerating plots...")
        fig, axes = analyzer.plot_temperature_evolution()
        
        # Save plot
        plot_file = Path(__file__).parent / "thermal_diffusion_analysis.png"
        fig.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to: {plot_file}")
        
        # Try to display
        try:
            import matplotlib.pyplot as plt
            plt.show()
        except:
            pass
        
    except ImportError:
        print("Matplotlib not installed. Install with: pip install matplotlib")
        print("Skipping visualization.")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80 + "\n")
