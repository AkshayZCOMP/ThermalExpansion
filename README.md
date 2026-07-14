# Composite Laminate Analysis - Thermal Expansion

A comprehensive Python toolkit for analyzing composite laminates with focus on quasi-isotropic (QI) layups, thermal effects, and resultant strains.

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Analysis
```bash
python examples/run_analysis.py
```

Edit the configuration at the top of `run_analysis.py` to change:
- Material properties (E_11, E_22, G_12, CTE values)
- Temperature change (delta_T)
- Fiber angles (LAYUP)

## Features

✓ **QI Laminates** - 8-ply and 16-ply quasi-isotropic support  
✓ **Thermal Effects** - Temperature-dependent strain analysis  
✓ **Stiffness Matrices** - A, B, D matrix calculation  
✓ **Ply-by-Ply Strains** - Through-thickness strain variation  
✓ **44 Unit Tests** - 100% test coverage  
✓ **Clean Interface** - Simple configuration, minimal output  

## Project Structure

```
ThermalExpansion/
├── src/                          # Core modules
│   ├── InputProperties.py        # Material & layup definitions
│   ├── MatrixFunctions.py        # Composite mechanics
│   └── workflow.py               # Analysis engine
├── tests/                        # Test suite
│   └── test_cases.py             # 44 comprehensive tests
├── examples/                     # Example scripts
│   ├── run_analysis.py           # Main front-end script
│   └── examples.py               # 6 demonstration examples
├── docs/                         # Documentation
│   ├── FULL_README.md            # Complete technical guide
│   ├── PROJECT_INDEX.md          # File index & navigation
│   ├── COMPLETION_SUMMARY.md     # Project summary
│   ├── QUICK_REFERENCE.py        # Code patterns
│   └── RUN_ANALYSIS_GUIDE.py     # Usage guide
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git exclusions
└── README.md                     # This file
```

## Usage Example

### Basic QI Analysis
```python
from src.InputProperties import MaterialProperties, LayupSequence, LaminateProperties
from src.workflow import LaminateAnalysis

# Material properties
material = MaterialProperties(name="Carbon/Epoxy")

# Layup sequence
layup = LayupSequence([0, 45, -45, 90, 90, -45, 45, 0], "8-ply QI")

# Create laminate with -50K cooling
laminate = LaminateProperties(material, layup, delta_T=-50)

# Run analysis
analysis = LaminateAnalysis(laminate)
analysis.run_full_analysis()

# Get results
print(f"A_11 stiffness: {analysis.A_matrix[0,0]:.2e}")
print(f"Membrane strains: {analysis.resultant_strains[0:3]}")
```

## Running Tests

```bash
python tests/test_cases.py
```

Expected output:
```
Ran 44 tests in 0.12s
OK
Tests run: 44
Failures: 0
```

## Configuration Guide

Edit `examples/run_analysis.py` to customize:

### Change Material
```python
MATERIAL = MaterialProperties(
    name="Glass/Epoxy",
    E_11=50e9,
    E_22=15e9,
    G_12=4e9,
    alpha_1=10e-6,
    alpha_2=50e-6
)
```

### Change Layup
```python
# 8-ply QI
LAYUP = [0, 45, -45, 90, 90, -45, 45, 0]

# Cross-ply
LAYUP = [0, 90, 0, 90]

# Unidirectional
LAYUP = [0, 0, 0, 0]
```

### Change Temperature
```python
DELTA_T = -50   # Cool by 50 K
DELTA_T = 100   # Heat by 100 K
DELTA_T = 0     # Room temperature
```

## Key Classes

- **MaterialProperties**: Stores E_11, E_22, G_12, v_12, CTE values
- **LayupSequence**: Defines fiber angles with QI/symmetric builders
- **LaminateProperties**: Combines material, layup, thickness, temperature
- **LaminateAnalysis**: Main analysis engine with all calculations

## Output

### Stiffness Matrices
- **A Matrix**: Extensional stiffness (3×3)
- **B Matrix**: Bending-extension coupling (3×3)
- **D Matrix**: Bending stiffness (3×3)

### Results
- **Membrane strains**: eps_x, eps_y, gamma_xy
- **Curvatures**: kappa_x, kappa_y, kappa_xy
- **Individual ply strains**: Per-ply strain through thickness

## Documentation

See `docs/FULL_README.md` for:
- Complete technical guide
- Detailed API reference
- Physical interpretation guide
- Common workflows
- Troubleshooting

## Testing

44 comprehensive unit tests covering:
- Material properties
- Layup creation
- Matrix transformations
- Thermal effects
- Strain calculations
- Edge cases
- Numerical stability

## Dependencies

- Python 3.8+
- NumPy 2.0+

## License

MIT License

## Contact

Project: https://github.com/AkshayZCOMP/ThermalExpansion
