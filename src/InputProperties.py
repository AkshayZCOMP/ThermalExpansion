"""
Input Properties for Composite Laminates

This module defines standard input properties for various composite laminate configurations,
including material properties, layup sequences, and geometric parameters.
"""

import numpy as np


class MaterialProperties:
    """Class to store material properties for a composite lamina."""
    
    def __init__(self, name="Carbon/Epoxy", E_11=130e9, E_22=10e9, v_12=0.3, 
                 G_12=5e9, alpha_1=-0.5e-6, alpha_2=30e-6):
        """
        Initialize material properties for a composite lamina.
        
        Parameters:
        -----------
        name : str
            Name of the material
        E_11 : float
            Young's modulus in fiber direction (Pa), default: 130 GPa
        E_22 : float
            Young's modulus in transverse direction (Pa), default: 10 GPa
        v_12 : float
            Poisson's ratio, default: 0.3
        G_12 : float
            Shear modulus (Pa), default: 5 GPa
        alpha_1 : float
            CTE in fiber direction (1/K), default: -0.5e-6 /K (typical for carbon)
        alpha_2 : float
            CTE in transverse direction (1/K), default: 30e-6 /K (typical for epoxy)
        """
        self.name = name
        self.E_11 = E_11
        self.E_22 = E_22
        self.v_12 = v_12
        self.G_12 = G_12
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
    
    def __repr__(self):
        return (f"MaterialProperties(name={self.name}, E_11={self.E_11:.2e}, "
                f"E_22={self.E_22:.2e}, G_12={self.G_12:.2e}, v_12={self.v_12})")


class LayupSequence:
    """Class to define and manage laminate layup sequences."""
    
    def __init__(self, angles, name="Custom Layup"):
        """
        Initialize a layup sequence.
        
        Parameters:
        -----------
        angles : list
            List of ply angles in degrees (e.g., [0, 45, -45, 90, -45, 45, 0])
        name : str
            Name of the layup
        """
        self.angles = angles
        self.name = name
        self.num_plies = len(angles)
    
    @staticmethod
    def create_QI(num_plies_per_set=8):
        """
        Create a quasi-isotropic (QI) layup.
        
        Standard QI has equal numbers of 0°, 45°, -45°, and 90° plies,
        typically arranged symmetrically. For 8-ply configuration: [0, 45, -45, 90, 90, -45, 45, 0]
        
        Parameters:
        -----------
        num_plies_per_set : int
            Number of plies per 0/45/-45/90 set (default: 8)
        
        Returns:
        --------
        LayupSequence
            A QI layup sequence
        """
        if num_plies_per_set == 8:
            angles = [0, 45, -45, 90, 90, -45, 45, 0]
        elif num_plies_per_set == 16:
            angles = [0, 45, -45, 90, 0, 45, -45, 90, 90, -45, 45, 0, 90, -45, 45, 0]
        elif num_plies_per_set == 4:
            angles = [0, 45, -45, 90]
        else:
            # General QI approximation
            angles = [0, 45, -45, 90] * (num_plies_per_set // 4)
        
        return LayupSequence(angles, name=f"QI [{num_plies_per_set}]")
    
    @staticmethod
    def create_symmetric(base_angles):
        """
        Create a symmetric layup from a base sequence.
        
        Parameters:
        -----------
        base_angles : list
            Base sequence angles (will be mirrored)
        
        Returns:
        --------
        LayupSequence
            A symmetric layup
        """
        angles = base_angles + base_angles[::-1]
        return LayupSequence(angles, name="Symmetric Layup")
    
    @staticmethod
    def create_balanced(base_angles):
        """
        Create a balanced layup (equal ±45 plies).
        
        Parameters:
        -----------
        base_angles : list
            Base sequence angles
        
        Returns:
        --------
        LayupSequence
            A balanced layup
        """
        return LayupSequence(base_angles, name="Balanced Layup")
    
    def __repr__(self):
        angle_str = ", ".join([f"{a:g}°" for a in self.angles])
        return f"LayupSequence(name={self.name}, angles=[{angle_str}], num_plies={self.num_plies})"


class LaminateProperties:
    """Class to define complete laminate properties."""
    
    def __init__(self, material, layup, ply_thickness=0.125e-3, delta_T=25):
        """
        Initialize laminate properties.
        
        Parameters:
        -----------
        material : MaterialProperties
            Material properties for the lamina
        layup : LayupSequence
            Layup sequence for the laminate
        ply_thickness : float
            Thickness of each ply (m), default: 0.125 mm (0.005 inches)
        delta_T : float
            Temperature change from reference (K), default: 0
        """
        self.material = material
        self.layup = layup
        self.ply_thickness = ply_thickness
        self.delta_T = delta_T
        self.total_thickness = ply_thickness * layup.num_plies
    
    def __repr__(self):
        return (f"LaminateProperties(\n  material: {self.material}\n"
                f"  layup: {self.layup}\n"
                f"  ply_thickness: {self.ply_thickness*1e3:.3f} mm\n"
                f"  total_thickness: {self.total_thickness*1e3:.3f} mm\n"
                f"  delta_T: {self.delta_T} K)")


# ============================================================================
# STANDARD QI LAYUP DEFINITIONS
# ============================================================================

def create_standard_QI_8ply():
    """
    Create a standard 8-ply quasi-isotropic laminate with carbon/epoxy material.
    
    Returns:
    --------
    LaminateProperties
        Standard 8-ply QI laminate
    """
    material = MaterialProperties(
        name="Carbon/Epoxy (Std)",
        E_11=130e9,
        E_22=10e9,
        v_12=0.3,
        G_12=5e9,
        alpha_1=-0.5e-6,
        alpha_2=30e-6
    )
    layup = LayupSequence.create_QI(num_plies_per_set=8)
    return LaminateProperties(material, layup, ply_thickness=0.125e-3, delta_T=0.0)


def create_standard_QI_16ply():
    """
    Create a standard 16-ply quasi-isotropic laminate with carbon/epoxy material.
    
    Returns:
    --------
    LaminateProperties
        Standard 16-ply QI laminate
    """
    material = MaterialProperties(
        name="Carbon/Epoxy (Std)",
        E_11=130e9,
        E_22=10e9,
        v_12=0.3,
        G_12=5e9,
        alpha_1=-0.5e-6,
        alpha_2=30e-6
    )
    layup = LayupSequence.create_QI(num_plies_per_set=16)
    return LaminateProperties(material, layup, ply_thickness=0.125e-3, delta_T=0.0)


def create_standard_QI_with_temperature(delta_T=-50.0):
    """
    Create a standard 8-ply QI laminate with specified temperature change.
    
    Parameters:
    -----------
    delta_T : float
        Temperature change from manufacturing (K), default: -50 K (typical cooldown)
    
    Returns:
    --------
    LaminateProperties
        Standard 8-ply QI laminate with temperature effects
    """
    material = MaterialProperties(
        name="Carbon/Epoxy (Std)",
        E_11=130e9,
        E_22=10e9,
        v_12=0.3,
        G_12=5e9,
        alpha_1=-0.5e-6,
        alpha_2=30e-6
    )
    layup = LayupSequence.create_QI(num_plies_per_set=8)
    return LaminateProperties(material, layup, ply_thickness=0.125e-3, delta_T=delta_T)
