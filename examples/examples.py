"""
Example Usage of Composite Laminate Analysis

This script demonstrates how to use the InputProperties, workflow, and 
analysis modules to perform composite laminate analysis.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.InputProperties import (
    create_standard_QI_8ply, create_standard_QI_16ply,
    create_standard_QI_with_temperature, MaterialProperties, 
    LayupSequence, LaminateProperties
)
from src.workflow import LaminateAnalysis, run_standard_QI_analysis


def example_1_standard_8ply_qi():
    """Example 1: Analyze a standard 8-ply QI laminate at room temperature."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Standard 8-Ply QI Laminate (No Temperature Change)")
    print("="*80)
    
    laminate = create_standard_QI_8ply()
    analysis = run_standard_QI_analysis(laminate, verbose=True)


def example_2_8ply_qi_with_cooling():
    """Example 2: Analyze an 8-ply QI laminate cooled by 50 K."""
    print("\n" + "="*80)
    print("EXAMPLE 2: 8-Ply QI Laminate with -50 K Temperature Change")
    print("="*80)
    
    laminate = create_standard_QI_with_temperature(delta_T=-50)
    analysis = run_standard_QI_analysis(laminate, verbose=True)


def example_3_16ply_qi():
    """Example 3: Analyze a standard 16-ply QI laminate."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Standard 16-Ply QI Laminate")
    print("="*80)
    
    laminate = create_standard_QI_16ply()
    analysis = run_standard_QI_analysis(laminate, verbose=True)


def example_4_custom_layup():
    """Example 4: Create and analyze a custom cross-ply laminate."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Custom Cross-Ply Laminate [0/90]_2s")
    print("="*80)
    
    # Create a custom symmetric cross-ply laminate
    material = MaterialProperties()
    layup = LayupSequence.create_symmetric([0, 90])
    laminate = LaminateProperties(
        material=material,
        layup=layup,
        ply_thickness=0.125e-3,
        delta_T=-50
    )
    
    analysis = run_standard_QI_analysis(laminate, verbose=True)


def example_5_compare_layups():
    """Example 5: Compare different layup configurations."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Comparing Different Layup Configurations")
    print("="*80)
    
    layups = [
        ("8-Ply QI", create_standard_QI_8ply()),
        ("16-Ply QI", create_standard_QI_16ply()),
        ("8-Ply QI with -50K", create_standard_QI_with_temperature(-50)),
    ]
    
    print("\nComparison of A_11 (Extensional Stiffness in x-direction):\n")
    print(f"{'Layup':<25} {'A_11 (N/m)':<20} {'Total Thickness (mm)':<20}")
    print("-" * 65)
    
    for name, laminate in layups:
        analysis = LaminateAnalysis(laminate)
        analysis.run_full_analysis()
        a11 = analysis.A_matrix[0, 0]
        thickness = laminate.total_thickness * 1e3
        print(f"{name:<25} {a11:<20.2e} {thickness:<20.3f}")
    
    print("\n" + "-" * 65)
    print("\nComparison of Thermal Strains (eps_x at zero external load):\n")
    print(f"{'Layup':<25} {'eps_x':<20} {'eps_y':<20} {'gamma_xy':<20}")
    print("-" * 65)
    
    for name, laminate in layups:
        analysis = LaminateAnalysis(laminate)
        analysis.run_full_analysis()
        eps = analysis.resultant_strains[0:3]
        print(f"{name:<25} {eps[0]:<20.6e} {eps[1]:<20.6e} {eps[2]:<20.6e}")


def example_6_detailed_analysis():
    """Example 6: Perform detailed analysis with custom parameters."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Detailed Analysis - Glass/Polyester Cross-Ply")
    print("="*80)
    
    # Define glass/polyester material properties
    material = MaterialProperties(
        name="Glass/Polyester",
        E_11=50e9,      # Glass fiber modulus
        E_22=15e9,      # Matrix modulus
        v_12=0.25,      # Poisson's ratio
        G_12=4e9,       # Shear modulus
        alpha_1=10e-6,  # CTE in fiber direction
        alpha_2=50e-6   # CTE in transverse direction
    )
    
    # Create a cross-ply layup
    layup = LayupSequence([0, 90, 90, 0], name="Cross-ply [0/90/90/0]")
    
    # Create laminate
    laminate = LaminateProperties(
        material=material,
        layup=layup,
        ply_thickness=0.15e-3,
        delta_T=-40
    )
    
    # Run analysis
    analysis = run_standard_QI_analysis(laminate, verbose=True)
    
    print("\n" + "="*80)
    print("ADDITIONAL ANALYSIS")
    print("="*80)
    
    # Print ply-by-ply strains in material coordinates
    print("\nPly Strains (Global Coordinates):")
    for i, strain in enumerate(analysis.ply_strains):
        angle = analysis.layup.angles[i]
        print(f"  Ply {i+1} ({angle:+5.0f}°): eps_x={strain[0]:+.6e}, eps_y={strain[1]:+.6e}, gamma_xy={strain[2]:+.6e}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("       COMPOSITE LAMINATE ANALYSIS EXAMPLES")
    print("="*80)
    
    # Run all examples
    example_1_standard_8ply_qi()
    example_2_8ply_qi_with_cooling()
    example_3_16ply_qi()
    example_4_custom_layup()
    example_5_compare_layups()
    example_6_detailed_analysis()
    
    print("\n" + "="*80)
    print("All examples completed successfully!")
    print("="*80 + "\n")
