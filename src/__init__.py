"""
Composite Laminate Analysis Toolkit
Core modules for thermal expansion analysis of composite materials
"""

from .InputProperties import (
    MaterialProperties,
    LayupSequence,
    LaminateProperties,
    create_standard_QI_8ply,
    create_standard_QI_16ply,
    create_standard_QI_with_temperature
)
from .workflow import LaminateAnalysis
from .MatrixFunctions import (
    Q_matrix,
    T_matrix,
    Q_bar,
    alpha_bar,
    A_matrix,
    B_matrix,
    D_matrix,
    N_thermal,
    M_thermal
)

__version__ = "1.0.0"
__author__ = "Akshay"

__all__ = [
    "MaterialProperties",
    "LayupSequence",
    "LaminateProperties",
    "LaminateAnalysis",
    "create_standard_QI_8ply",
    "create_standard_QI_16ply",
    "create_standard_QI_with_temperature",
    "Q_matrix",
    "T_matrix",
    "Q_bar",
    "alpha_bar",
    "A_matrix",
    "B_matrix",
    "D_matrix",
    "N_thermal",
    "M_thermal"
]
