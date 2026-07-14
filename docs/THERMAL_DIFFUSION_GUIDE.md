# Thermal Diffusion Analysis - Complete Guide

## Overview

The thermal diffusion module enables **transient thermal analysis** of composite laminates. Instead of assuming uniform temperature throughout (steady-state), it models how heat diffuses from a heated surface through the thickness of the laminate over time, allowing you to see how strains and curvatures evolve as the temperature profile changes.

## Key Features

✓ **Transient heat diffusion** - Models 1D heat diffusion through thickness  
✓ **Time-dependent strains** - Track how strains evolve as temperature diffuses  
✓ **Curvature evolution** - See bending effects develop over time  
✓ **Steady-state analysis** - Automatic detection of when equilibrium is reached  
✓ **Customizable boundary conditions** - Set top surface temperature, initial temperature  
✓ **Material thermal properties** - Configurable thermal diffusivity  
✓ **Visualization** - Built-in plotting of temperature profiles and strain evolution  

## Physical Model

The module solves the 1D transient heat diffusion equation:

$$\frac{\partial T}{\partial t} = \alpha \frac{\partial^2 T}{\partial z^2}$$

Where:
- **T** = temperature (K or °C)
- **t** = time (seconds)
- **z** = position through laminate thickness
- **α** = thermal diffusivity (m²/s)

**Boundary Conditions:**
- **Top surface** (z=0): T = T_surface (constant, user-specified)
- **Bottom surface** (z=h): Adiabatic (no heat flux)
- **Initial condition**: T(z,0) = T_initial (uniform throughout)

**Solution Method:**
- Finite difference discretization in space
- ODE integration in time using scipy.integrate.odeint
- 50 spatial nodes (configurable) through thickness
- Automatic time stepping for accuracy

## Quick Start

### 1. Simple Transient Analysis

```python
from src.InputProperties import MaterialProperties, LayupSequence, LaminateProperties
from src.ThermalDiffusion import ThermalDiffusionAnalyzer

# Define material and laminate
material = MaterialProperties(name="Carbon/Epoxy")
layup = LayupSequence([0, 45, -45, 90, 90, -45, 45, 0], name="QI 8-ply")
laminate = LaminateProperties(material, layup, ply_thickness=0.125e-3, delta_T=0)

# Create analyzer
analyzer = ThermalDiffusionAnalyzer(
    laminate_properties=laminate,
    top_surface_temp=100,    # Heat top surface to 100 K
    initial_temp=20          # Start at 20 K throughout
)

# Run analysis for 1 hour (3600 seconds)
results = analyzer.analyze(
    t_final=3600,           # Total time in seconds
    n_steps=100,            # Number of time steps
    thermal_diffusivity=1e-6 # m²/s
)

# Print results
analyzer.print_summary()

# Plot results
fig, axes = analyzer.plot_temperature_evolution()
```

### 2. Accessing Time-Series Results

```python
# Extract detailed results
t = results['t']              # Time array (seconds)
z = results['z']              # Spatial positions (meters)
T = results['T']              # Temperature profiles [n_steps, n_nodes]
strains = results['strains']  # Membrane strains [n_steps, 3]
                              # Column 0: εₓ, Column 1: εᵧ, Column 2: γₓᵧ

# Access specific time step
time_5min = t[25]
T_profile_at_5min = T[25, :]
strains_at_5min = strains[25, :]
```

### 3. Steady State Properties

```python
# Get final equilibrium state
steady = analyzer.thermal_diffusion.get_steady_state_properties()

print(f"Steady state average temp: {steady['avg_temp']:.2f} K")
print(f"Steady state strains: {steady['strains']}")
print(f"Steady state curvatures: {steady['curvatures']}")

# Time to reach steady state (within 1% tolerance)
t_ss = analyzer.thermal_diffusion.time_to_steady_state(tolerance=0.01)
print(f"Time to steady state: {t_ss:.2f} s ({t_ss/60:.2f} min)")
```

## Using run_analysis.py with Thermal Diffusion

Edit `examples/run_analysis.py`:

```python
# At the top, set analysis type:
ENABLE_TRANSIENT_THERMAL = True  # Switch to thermal diffusion

# Configure transient parameters:
INITIAL_TEMP = 20            # Initial temperature (K)
TOP_SURFACE_TEMP = 100       # Surface temperature (K)
THERMAL_DIFFUSIVITY = 1e-6   # m²/s (typical for composites)
TRANSIENT_TIME = 3600        # Analysis duration (seconds)
TRANSIENT_STEPS = 100        # Number of time steps
```

Then run:
```bash
python examples/run_analysis.py
```

## Parameter Guide

### Thermal Diffusivity

Typical values for composite materials (α = k/(ρ*c)):

| Material | α (m²/s) | Notes |
|----------|----------|-------|
| Carbon/Epoxy | 1.0e-6 to 2.0e-6 | Fiber-dominated |
| Glass/Epoxy | 0.5e-6 to 1.5e-6 | Matrix-dominated |
| Aramid/Epoxy | 0.3e-6 to 1.0e-6 | Very low diffusivity |
| Aluminum | 7.0e-5 | For reference (much faster) |

### Temperature Changes

Common scenarios:

```python
# Rapid heating scenario
TOP_SURFACE_TEMP = 120
INITIAL_TEMP = 20
THERMAL_DIFFUSIVITY = 1e-6

# Slow cooling scenario
TOP_SURFACE_TEMP = 0
INITIAL_TEMP = 50
THERMAL_DIFFUSIVITY = 0.5e-6
```

### Analysis Time

Recommended based on laminate thickness and material:

```python
# Thickness 1mm, typical α:
TRANSIENT_TIME = 3600          # 1 hour for complete diffusion
TRANSIENT_STEPS = 100          # 36 seconds per step

# Thickness 0.5mm:
TRANSIENT_TIME = 900           # 15 minutes
TRANSIENT_STEPS = 50           # 18 seconds per step
```

## Understanding Results

### Temperature Profiles

```
The T array has shape [n_steps, n_nodes]:
- Row index: time step
- Column index: spatial position through thickness
- T[0, :] = initial uniform temperature
- T[-1, :] = final steady-state profile
```

### Strain Evolution

```
strains array shape: [n_steps, 3]
- Column 0: εₓ (longitudinal membrane strain)
- Column 1: εᵧ (transverse membrane strain)
- Column 2: γₓᵧ (shear strain)
```

### Curvature Evolution

```
curvatures array shape: [n_steps, 3]
- Column 0: κₓ (curvature about x-axis in m⁻¹)
- Column 1: κᵧ (curvature about y-axis in m⁻¹)
- Column 2: κₓᵧ (twist curvature in m⁻¹)
```

## Physical Interpretation

### Membrane Strains
- Develop immediately when temperature is imposed at surface
- May increase or decrease depending on CTE mismatch
- Reach quasi-steady state quickly (faster than curvatures)

### Curvatures (Bending)
- Develop due to through-thickness temperature gradients
- Maximum when gradient is steepest (early in transient)
- Decrease as temperature profile becomes more uniform
- Approach zero as temperature becomes uniform (steady state)

### Example Interpretation

```
If strains increase with time:
  → Material expanding due to heating
  
If curvatures decrease with time:
  → Through-thickness gradient diminishing
  → Laminate flattening out as it heats uniformly
  
If curvatures reverse direction:
  → Thermal gradient changing direction
  → Complex CTE interactions between plies
```

## Advanced Usage

### Custom Spatial Resolution

```python
# Increase accuracy with more nodes
analyzer.thermal_diffusion = ThermalDiffusion(
    laminate,
    analysis,
    top_surface_temp=100,
    initial_temp=20,
    thermal_diffusivity=1e-6,
    n_nodes=100  # More nodes = higher resolution, slower computation
)
```

### Temperature Gradient Analysis

```python
# After analysis:
dT_dz = analyzer.thermal_diffusion.get_temperature_gradient()
max_gradient = np.max(np.abs(dT_dz))
print(f"Max temperature gradient: {max_gradient:.2f} K/m")
```

### Intermediate Results

```python
# Access intermediate states
for i, t in enumerate(results['t']):
    temp_profile = results['T'][i, :]
    strain_state = results['strains'][i, :]
    
    if i % 10 == 0:  # Every 10 steps
        avg_temp = np.mean(temp_profile)
        print(f"t={t:.1f}s: T_avg={avg_temp:.2f}K, εₓ={strain_state[0]:.3e}")
```

## Common Issues & Solutions

### Issue: No change in temperature profile

**Cause**: Thermal diffusivity too high or time too short
**Solution**: 
- Decrease `THERMAL_DIFFUSIVITY` 
- Increase `TRANSIENT_TIME`

### Issue: Strains don't change much

**Cause**: Temperature not propagating deep enough into laminate
**Solution**:
- Increase analysis time
- Check laminate thickness (thick laminates take longer to diffuse)

### Issue: Memory issues with large arrays

**Cause**: Too many spatial nodes and time steps
**Solution**:
- Reduce `n_nodes` (50-100 is typical)
- Reduce `n_steps` (sample less frequently)

### Issue: Plots don't show

**Cause**: Matplotlib not installed
**Solution**: `pip install matplotlib`

## Comparison: Steady-State vs Transient

| Aspect | Steady-State | Transient |
|--------|-------------|-----------|
| Temperature | Uniform throughout | Gradient from surface |
| Time to compute | Fast | Slower (integration) |
| Curvatures | May be nonzero | Start high, decrease to zero |
| Real-world application | Long-term behavior | Design for thermal shock |
| Use case | Thermal equilibrium | Manufacturing, service |

## Example Applications

### Manufacturing (Cooling from Process)

```python
# Laminate comes out of autoclave at 180°C
# Cooled to room temperature (20°C)
TOP_SURFACE_TEMP = 20
INITIAL_TEMP = 180
TRANSIENT_TIME = 3600  # 1 hour cooling
```

### Thermal Shock Resistance

```python
# Sudden exposure to heat
TOP_SURFACE_TEMP = 150
INITIAL_TEMP = 20
TRANSIENT_TIME = 1800  # Monitor for 30 minutes
```

### Thermal Cycling

```python
# Run multiple sequential analyses
temps = [20, 50, 80, 50, 20]  # Temperature cycle
for i, T_surface in enumerate(temps):
    results = analyzer.analyze(t_final=1800, top_surface_temp=T_surface)
    print(f"Cycle {i}: Max curvature = {np.max(results['curvatures'][:, :]):.3e}")
```

## References

- **Thermal diffusion**: Carslaw & Jaeger, "Conduction of Heat in Solids"
- **Composite mechanics**: Classical Lamination Theory (CLT)
- **Finite differences**: Standard ODE discretization methods
- **Material properties**: Typical values from supplier data sheets

## Module Structure

```
src/ThermalDiffusion.py
├── ThermalDiffusion               # Core solver
│   ├── __init__()
│   ├── _temperature_diffusion_ode()  # ODE system
│   ├── solve_transient()          # Main solver
│   ├── _compute_strains_from_profile()
│   ├── get_steady_state_properties()
│   └── ...
├── ThermalDiffusionAnalyzer       # High-level interface
│   ├── analyze()
│   ├── plot_temperature_evolution()
│   └── print_summary()
```

## See Also

- `examples/thermal_diffusion_example.py` - Complete standalone example
- `examples/run_analysis.py` - Quick-start script with flag to enable
- `docs/FULL_README.md` - Complete technical documentation
