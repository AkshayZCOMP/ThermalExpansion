# Composite Laminate Analysis - Complete Documentation

## Overview

This package provides a comprehensive toolkit for analyzing composite laminates, including:

- **InputProperties.py**: Defines material properties, layup sequences, and laminate configuration
- **MatrixFunctions.py**: Core matrix calculations (Q, T, ABD matrices, thermal effects)
- **workflow.py**: High-level analysis workflow and strain calculations
- **test_cases.py**: 44 comprehensive unit tests covering all functionality
- **examples.py**: Practical examples demonstrating usage

## Key Features

✓ **Material Property Definition**: Carbon/epoxy and custom materials  
✓ **Layup Sequences**: QI (quasi-isotropic), symmetric, balanced, and custom layups  
✓ **Stiffness Matrix Calculation**: A, B, D matrix assembly for laminates  
✓ **Thermal Analysis**: Thermal force/moment vectors and resulting strains  
✓ **Individual Ply Strains**: Through-thickness strain variation  
✓ **Comprehensive Testing**: 44 unit tests with 100% pass rate  

---

## Module Guide

### 1. InputProperties.py

Defines material and laminate configuration classes.

#### MaterialProperties Class
```python
from InputProperties import MaterialProperties

# Create carbon/epoxy material
material = MaterialProperties(
    name="Carbon/Epoxy",
    E_11=130e9,     # Young's modulus in fiber direction (Pa)
    E_22=10e9,      # Young's modulus in transverse direction (Pa)
    v_12=0.3,       # Poisson's ratio
    G_12=5e9,       # Shear modulus (Pa)
    alpha_1=-0.5e-6,  # CTE in fiber direction (1/K)
    alpha_2=30e-6   # CTE in transverse direction (1/K)
)
```

#### LayupSequence Class
```python
from InputProperties import LayupSequence

# Create quasi-isotropic (QI) layup
qi_layup = LayupSequence.create_QI(num_plies_per_set=8)
# Creates: [0, 45, -45, 90, 90, -45, 45, 0]

# Create symmetric layup
sym_layup = LayupSequence.create_symmetric([0, 45, -45])
# Creates: [0, 45, -45, -45, 45, 0]

# Custom layup
custom = LayupSequence([0, 90, 0, 90], name="Cross-ply")
```

#### LaminateProperties Class
```python
from InputProperties import LaminateProperties, create_standard_QI_8ply

# Use standard QI laminate
laminate = create_standard_QI_8ply()

# Or create custom
laminate = LaminateProperties(
    material=material,
    layup=qi_layup,
    ply_thickness=0.125e-3,  # meters
    delta_T=-50  # temperature change in Kelvin
)
```

**Available Standard Functions:**
- `create_standard_QI_8ply()` - 8-ply QI at room temperature
- `create_standard_QI_16ply()` - 16-ply QI at room temperature
- `create_standard_QI_with_temperature(delta_T)` - 8-ply QI with temperature

---

### 2. MatrixFunctions.py

Low-level matrix calculations for composite mechanics.

**Core Functions:**
- `Q_matrix(E_11, E_22, v_12, G_12)` - Material stiffness matrix
- `T_matrix(theta)` - Transformation matrix for angle θ
- `Q_bar(Q, T, R)` - Transformed stiffness matrix
- `CTE_vetor(alpha_1, alpha_2)` - CTE vector
- `alpha_bar(alpha, T, R)` - Transformed CTE vector
- `A_matrix(Q_bars, ply_number, thickness)` - Extensional stiffness
- `B_matrix(Q_bars, ply_number, thickness)` - Coupling matrix
- `D_matrix(Q_bars, ply_number, thickness)` - Bending stiffness
- `arrange_ABD(A, B, D)` - Combine into 6×6 ABD matrix
- `N_thermal(alpha_bars, Q_bars, delta_T, thickness, ply_number)` - Thermal forces
- `M_thermal(alpha_bars, Q_bars, delta_T, thickness, ply_number)` - Thermal moments
- `resultantvector(N_thermal, M_thermal, ABD)` - Resultant strains/curvatures

---

### 3. workflow.py

High-level analysis interface.

#### LaminateAnalysis Class
```python
from workflow import LaminateAnalysis
from InputProperties import create_standard_QI_8ply

# Create analysis
laminate = create_standard_QI_8ply()
analysis = LaminateAnalysis(laminate)

# Run complete analysis
analysis.run_full_analysis()

# Access results
print("A Matrix:", analysis.A_matrix)
print("Resultant strains:", analysis.resultant_strains)
print("Individual ply strains:", analysis.ply_strains)

# Get formatted summary
print(analysis.get_summary())
```

**Main Methods:**
- `calculate_material_matrices()` - Q and α in material coordinates
- `calculate_transformed_matrices()` - Q_bar and α_bar for each ply
- `calculate_laminate_stiffness_matrices()` - A, B, D matrices
- `calculate_thermal_loading()` - Thermal forces and moments
- `calculate_resultant_strains()` - Membrane strains and curvatures
- `calculate_ply_strains()` - Individual ply strains
- `run_full_analysis()` - Execute all calculations
- `get_summary()` - Formatted results output

**Convenience Function:**
```python
from workflow import run_standard_QI_analysis

# Quick analysis with automatic output
analysis = run_standard_QI_analysis(laminate, verbose=True)
```

---

## Usage Examples

### Example 1: Basic QI Laminate Analysis
```python
from InputProperties import create_standard_QI_8ply
from workflow import run_standard_QI_analysis

laminate = create_standard_QI_8ply()
analysis = run_standard_QI_analysis(laminate, verbose=True)
```

### Example 2: Thermal Effects
```python
from InputProperties import create_standard_QI_with_temperature
from workflow import LaminateAnalysis

# Cool by 50 K
laminate = create_standard_QI_with_temperature(delta_T=-50)
analysis = LaminateAnalysis(laminate)
analysis.run_full_analysis()

print(f"Thermal strains: {analysis.resultant_strains[0:3]}")
```

### Example 3: Custom Material and Layup
```python
from InputProperties import MaterialProperties, LayupSequence, LaminateProperties
from workflow import LaminateAnalysis

# Define glass/polyester
glass_epoxy = MaterialProperties(
    name="Glass/Epoxy",
    E_11=50e9, E_22=15e9, v_12=0.25, G_12=4e9,
    alpha_1=10e-6, alpha_2=50e-6
)

# Cross-ply layup
layup = LayupSequence([0, 90, 90, 0], name="Cross-ply")

# Create laminate with -40 K temperature change
laminate = LaminateProperties(
    material=glass_epoxy,
    layup=layup,
    ply_thickness=0.15e-3,
    delta_T=-40
)

# Analyze
analysis = LaminateAnalysis(laminate)
analysis.run_full_analysis()

# Print ply strains
for i, strain in enumerate(analysis.ply_strains):
    angle = laminate.layup.angles[i]
    print(f"Ply {i+1} ({angle}deg): eps_x={strain[0]:.3e}, eps_y={strain[1]:.3e}")
```

### Example 4: Comparing Layups
```python
from InputProperties import create_standard_QI_8ply, create_standard_QI_16ply
from workflow import LaminateAnalysis

laminates = [
    ("8-ply QI", create_standard_QI_8ply()),
    ("16-ply QI", create_standard_QI_16ply()),
]

for name, laminate in laminates:
    analysis = LaminateAnalysis(laminate)
    analysis.run_full_analysis()
    print(f"{name}: A11 = {analysis.A_matrix[0,0]:.2e} N/m")
```

---

## Test Suite

The package includes 44 comprehensive unit tests organized into test classes:

**Test Classes:**
- `TestMaterialProperties` - Material property handling
- `TestLayupSequence` - Layup definition and creation
- `TestLaminateProperties` - Laminate configuration
- `TestLaminateAnalysisBasics` - Basic analysis functionality
- `TestTransformedMatrices` - Matrix transformation verification
- `TestLaminateStiffnessMatrices` - A, B, D matrix properties
- `TestSymmetricLaminate` - Symmetry verification
- `TestThermalLoading` - Thermal effects
- `TestResultantStrains` - Strain calculations
- `TestPlyStrains` - Individual ply strains
- `TestFullWorkflow` - Integration tests
- `TestEdgeCases` - Boundary conditions
- `TestNumericalStability` - Numerical robustness

**Run Tests:**
```bash
cd c:\Users\prz\Documents\HeatXferDIC
python test_cases.py
```

**Expected Output:**
```
Ran 44 tests in 0.24s

OK

================================================================================
Tests run: 44
Failures: 0
Errors: 0
Skipped: 0
================================================================================
```

---

## Output Structure

### Resultant Strains Array (length 6)
```
[eps_x, eps_y, gamma_xy, kappa_x, kappa_y, kappa_xy]

Indices 0-2: Membrane strains (dimensionless)
Indices 3-5: Curvatures (1/m)
```

### Individual Ply Strains
Each ply strain is a 3-element array:
```
[eps_x_global, eps_y_global, gamma_xy_global]
```

Strains are in the global coordinate system and vary through the thickness due to curvature.

### Stiffness Matrices

**A Matrix (3×3)**: Extensional stiffness - relates membrane forces to membrane strains  
**B Matrix (3×3)**: Coupling stiffness - couples bending and extension  
**D Matrix (3×3)**: Bending stiffness - relates moments to curvatures  
**ABD Matrix (6×6)**: Combined stiffness matrix used for stress/strain calculations

---

## Physical Interpretation

### Quasi-Isotropic (QI) Laminates
- Equal fiber amounts in 0°, 45°, -45°, and 90° directions
- Approximately isotropic in-plane stiffness
- B matrix ≈ 0 for symmetric arrangements
- Typical layup: [0, 45, -45, 90, 90, -45, 45, 0]

### Thermal Effects
- Temperature changes cause matrix to expand/contract differently than fibers
- Negative CTE mismatch creates residual stresses
- Cooling (ΔT < 0) typically creates compressive strains in fibers
- Thermal effects especially important in carbon/epoxy systems

### Symmetric Laminates
- Mirror image layup about the mid-plane
- B matrix negligibly small (< 10^-6)
- No bending-extension coupling
- Example: [0, 45, -45, 90, 90, -45, 45, 0]

---

## Common Material Systems

### Carbon/Epoxy (Default)
- E_11 = 130 GPa
- E_22 = 10 GPa
- G_12 = 5 GPa
- v_12 = 0.3
- CTE: α_1 = -0.5×10^-6 /K, α_2 = 30×10^-6 /K

### Glass/Epoxy
- E_11 = 50 GPa
- E_22 = 15 GPa
- G_12 = 4 GPa
- v_12 = 0.25
- CTE: α_1 = 10×10^-6 /K, α_2 = 50×10^-6 /K

---

## File Structure

```
HeatXferDIC/
├── InputProperties.py          # Material and layup definitions
├── MatrixFunctions.py          # Core composite mechanics
├── workflow.py                 # High-level analysis interface
├── test_cases.py               # 44 comprehensive unit tests
├── examples.py                 # Practical usage examples
└── README.md                   # This documentation
```

---

## API Quick Reference

### Most Common Usage
```python
# Standard analysis with default settings
from InputProperties import create_standard_QI_8ply
from workflow import run_standard_QI_analysis

analysis = run_standard_QI_analysis(create_standard_QI_8ply())
```

### Thermal Analysis
```python
from InputProperties import create_standard_QI_with_temperature
from workflow import LaminateAnalysis

laminate = create_standard_QI_with_temperature(delta_T=-50)
analysis = LaminateAnalysis(laminate)
analysis.run_full_analysis()
strains = analysis.resultant_strains
```

### Custom Configuration
```python
from InputProperties import MaterialProperties, LayupSequence, LaminateProperties
from workflow import LaminateAnalysis

mat = MaterialProperties(name="Custom", E_11=120e9, E_22=12e9, ...)
layup = LayupSequence([0, 45, -45, 90], "Custom")
laminate = LaminateProperties(mat, layup, ply_thickness=0.1e-3, delta_T=0)

analysis = LaminateAnalysis(laminate)
analysis.run_full_analysis()
```

---

## Troubleshooting

**Q: Why are strains zero?**  
A: With delta_T = 0 and no external loads, strains will be zero. Set delta_T for thermal loading.

**Q: Are results valid?**  
A: Check that the ABD matrix is invertible (determinant ≠ 0) and no NaN values appear.

**Q: How to verify results?**  
A: For symmetric laminates, B matrix should be ~0. Run `python test_cases.py` to verify installation.

**Q: How to compare different layups?**  
A: Create multiple LaminateAnalysis objects and compare A_matrix[0,0], D_matrix, etc.

---

## Validation

All 44 tests pass successfully:
- ✓ Material property initialization
- ✓ Layup sequence creation
- ✓ Laminate configuration
- ✓ Matrix transformations
- ✓ Stiffness matrix assembly
- ✓ Symmetry properties
- ✓ Thermal loading
- ✓ Strain calculations
- ✓ Individual ply strains
- ✓ Integration workflows
- ✓ Edge cases
- ✓ Numerical stability

---

## Version History

**v1.0** (Current)
- 44 comprehensive unit tests (100% pass rate)
- 6 practical examples
- Support for custom materials and layups
- Thermal analysis capabilities
- Complete documentation
