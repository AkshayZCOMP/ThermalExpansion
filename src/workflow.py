"""
Composite Laminate Workflow

This module provides a workflow to analyze composite laminates, calculate
stiffness matrices, thermal effects, and resulting strains.
"""

import numpy as np
from .MatrixFunctions import (
    Q_matrix, CTE_vetor, T_matrix, reuters_matrix, Q_bar, alpha_bar,
    A_matrix, B_matrix, D_matrix, arrange_ABD,
    N_thermal, M_thermal, resultantvector
)
from .InputProperties import LaminateProperties


class LaminateAnalysis:
    """
    Main class for analyzing composite laminates.
    
    Performs calculations including:
    - Stiffness matrix transformation for each ply
    - Laminate stiffness (A, B, D) matrix assembly
    - Thermal loading calculations
    - Resultant strains and curvatures
    - Individual ply strains
    """
    
    def __init__(self, laminate_properties):
        """
        Initialize laminate analysis.
        
        Parameters:
        -----------
        laminate_properties : LaminateProperties
            Object containing material, layup, and geometric properties
        """
        self.laminate = laminate_properties
        
        # Material properties
        self.material = laminate_properties.material
        self.layup = laminate_properties.layup
        self.ply_thickness = laminate_properties.ply_thickness
        self.delta_T = laminate_properties.delta_T
        
        # Initialize storage for intermediate calculations
        self.Q_matrix_list = []
        self.T_matrix_list = []
        self.R_matrix = reuters_matrix()
        self.Q_bar_list = []
        self.alpha_bar_list = []
        
        # Storage for final results
        self.A_matrix = None
        self.B_matrix = None
        self.D_matrix = None
        self.ABD_matrix = None
        
        self.N_thermal = None
        self.M_thermal = None
        self.resultant_strains = None
        self.ply_strains = []
        
    def calculate_material_matrices(self):
        """
        Calculate the Q matrix (stiffness matrix) and CTE vector
        in the material coordinate system (before transformation).
        
        Returns:
        --------
        dict
            Dictionary with 'Q' and 'alpha' matrices
        """
        Q = Q_matrix(
            self.material.E_11,
            self.material.E_22,
            self.material.v_12,
            self.material.G_12
        )
        
        alpha = CTE_vetor(self.material.alpha_1, self.material.alpha_2)
        
        return {"Q": Q, "alpha": alpha}
    
    def calculate_transformed_matrices(self):
        """
        Transform material matrices to global coordinate system for each ply.
        
        This calculates Q_bar and alpha_bar for each ply based on its angle.
        """
        material_matrices = self.calculate_material_matrices()
        Q = material_matrices["Q"]
        alpha = material_matrices["alpha"]
        
        for angle in self.layup.angles:
            # Create transformation matrices
            T = T_matrix(angle)
            self.T_matrix_list.append(T)
            
            # Transform stiffness matrix
            Q_bar_ply = Q_bar(Q, T, self.R_matrix)
            self.Q_bar_list.append(Q_bar_ply)
            
            # Transform CTE vector
            alpha_bar_ply = alpha_bar(alpha, T, self.R_matrix)
            self.alpha_bar_list.append(alpha_bar_ply.flatten())
    
    def calculate_laminate_stiffness_matrices(self):
        """
        Calculate the A, B, and D stiffness matrices for the laminate.
        
        These matrices represent:
        - A: Extensional stiffness
        - B: Coupling between extension and bending
        - D: Bending stiffness
        """
        self.A_matrix = A_matrix(
            self.Q_bar_list,
            self.layup.num_plies,
            self.ply_thickness
        )
        
        self.B_matrix = B_matrix(
            self.Q_bar_list,
            self.layup.num_plies,
            self.ply_thickness
        )
        
        self.D_matrix = D_matrix(
            self.Q_bar_list,
            self.layup.num_plies,
            self.ply_thickness
        )
        
        self.ABD_matrix = arrange_ABD(self.A_matrix, self.B_matrix, self.D_matrix)
    
    def calculate_thermal_loading(self):
        """
        Calculate thermal force and moment vectors due to temperature change.
        """
        self.N_thermal = N_thermal(
            self.alpha_bar_list,
            self.Q_bar_list,
            self.delta_T,
            self.ply_thickness,
            self.layup.num_plies
        )
        
        self.M_thermal = M_thermal(
            self.alpha_bar_list,
            self.Q_bar_list,
            self.delta_T,
            self.ply_thickness,
            self.layup.num_plies
        )
    
    def calculate_resultant_strains(self):
        """
        Calculate resultant membrane strains and curvatures.
        
        Returns:
        --------
        numpy.ndarray
            Array [eps_x, eps_y, gamma_xy, kappa_x, kappa_y, kappa_xy]
            where:
            - eps_x, eps_y, gamma_xy: membrane strains
            - kappa_x, kappa_y, kappa_xy: curvatures
        """
        self.resultant_strains = resultantvector(
            self.N_thermal,
            self.M_thermal,
            self.ABD_matrix
        )
        return self.resultant_strains
    
    def calculate_ply_strains(self):
        """
        Calculate strains in each individual ply in the global coordinate system.
        
        Returns:
        --------
        list
            List of strain arrays for each ply: [eps_x, eps_y, gamma_xy]
        """
        if self.resultant_strains is None:
            self.calculate_resultant_strains()
        
        eps_0 = self.resultant_strains[0:3]  # Membrane strains
        kappa = self.resultant_strains[3:6]   # Curvatures
        
        # Calculate z-coordinates for each ply
        total_thickness = self.layup.num_plies * self.ply_thickness
        z_bottom = -total_thickness / 2
        
        self.ply_strains = []
        
        for i in range(self.layup.num_plies):
            # Mid-plane z-coordinate of ply
            z_mid = z_bottom + (i + 0.5) * self.ply_thickness
            
            # Strain at this z-location: eps = eps_0 + z * kappa
            strain = eps_0 + z_mid * kappa
            self.ply_strains.append(strain)
        
        return self.ply_strains
    
    def run_full_analysis(self):
        """
        Execute the complete laminate analysis workflow.
        
        This method:
        1. Transforms material matrices for each ply
        2. Calculates laminate stiffness matrices
        3. Calculates thermal loading
        4. Calculates resultant strains and curvatures
        5. Calculates individual ply strains
        """
        self.calculate_transformed_matrices()
        self.calculate_laminate_stiffness_matrices()
        self.calculate_thermal_loading()
        self.calculate_resultant_strains()
        self.calculate_ply_strains()
    
    def get_summary(self):
        """
        Generate a summary of the analysis results.
        
        Returns:
        --------
        str
            Formatted summary of key results
        """
        summary = f"""
================================================================================
                        LAMINATE ANALYSIS SUMMARY
================================================================================

MATERIAL PROPERTIES:
  {self.material}

LAYUP SEQUENCE:
  {self.layup}
  Total thickness: {self.laminate.total_thickness*1e3:.3f} mm

TEMPERATURE CHANGE:
  dT = {self.delta_T} K

================================================================================
                         LAMINATE STIFFNESS MATRICES
================================================================================

A Matrix (Extensional Stiffness):
{self.A_matrix}

B Matrix (Coupling):
{self.B_matrix}

D Matrix (Bending Stiffness):
{self.D_matrix}

================================================================================
                         THERMAL LOADING
================================================================================

Thermal Force Vector N_T:
  {self.N_thermal}

Thermal Moment Vector M_T:
  {self.M_thermal}

================================================================================
                      RESULTANT STRAINS & CURVATURES
================================================================================

Membrane Strains (eps_x, eps_y, gamma_xy):
  eps_x   = {self.resultant_strains[0]:.6e}
  eps_y   = {self.resultant_strains[1]:.6e}
  gamma_xy= {self.resultant_strains[2]:.6e}

Curvatures (kappa_x, kappa_y, kappa_xy):
  kappa_x = {self.resultant_strains[3]:.6e} (1/m)
  kappa_y = {self.resultant_strains[4]:.6e} (1/m)
  kappa_xy= {self.resultant_strains[5]:.6e} (1/m)

================================================================================
                         INDIVIDUAL PLY STRAINS
================================================================================

"""
        for i, strain in enumerate(self.ply_strains):
            angle = self.layup.angles[i]
            summary += f"Ply {i+1:2d} ({angle:+5.0f}): eps_x={strain[0]:+.6e}, eps_y={strain[1]:+.6e}, gamma_xy={strain[2]:+.6e}\n"
        
        summary += "=" * 80 + "\n"
        
        return summary


def run_standard_QI_analysis(laminate_properties=None, verbose=True):
    """
    Convenience function to run analysis on a standard QI laminate.
    
    Parameters:
    -----------
    laminate_properties : LaminateProperties, optional
        Laminate to analyze. If None, uses standard 8-ply QI.
    verbose : bool
        If True, prints summary results. Default: True
    
    Returns:
    --------
    LaminateAnalysis
        Completed analysis object
    """
    if laminate_properties is None:
        from InputProperties import create_standard_QI_8ply
        laminate_properties = create_standard_QI_8ply()
    
    analysis = LaminateAnalysis(laminate_properties)
    analysis.run_full_analysis()
    
    if verbose:
        print(analysis.get_summary())
    
    return analysis


if __name__ == "__main__":
    # Example: Run standard 8-ply QI analysis
    print("Running Standard 8-ply QI Laminate Analysis...")
    analysis = run_standard_QI_analysis()
