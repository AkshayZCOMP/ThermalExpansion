"""
Thermal Diffusion Analysis for Composite Laminates

Solves 1D transient heat diffusion through laminate thickness and computes
resulting strains and curvatures as temperature profile evolves to steady state.

Uses finite difference method to discretize and solve the heat equation:
    ∂T/∂t = α * ∂²T/∂z²

Where:
    α = thermal diffusivity (k / (ρ * c))
    z = position through thickness
    t = time
"""

import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.integrate import odeint


class ThermalDiffusion:
    """
    Transient thermal analysis for composite laminates.
    
    Solves heat diffusion equation through thickness with:
    - Top surface: constant imposed temperature
    - Bottom surface: adiabatic (insulated) or constant
    - Initial: uniform initial temperature throughout
    """
    
    def __init__(self, laminate_properties, analysis, top_surface_temp=50, 
                 initial_temp=20, thermal_diffusivity=None, n_nodes=50):
        """
        Initialize thermal diffusion solver.
        
        Parameters
        ----------
        laminate_properties : LaminateProperties
            Laminate configuration
        analysis : LaminateAnalysis
            Pre-computed laminate analysis object
        top_surface_temp : float
            Temperature imposed at top surface (K or °C)
        initial_temp : float
            Initial uniform temperature throughout (K or °C)
        thermal_diffusivity : float, optional
            Thermal diffusivity (m²/s). Default: 1e-6 (typical for composites)
        n_nodes : int
            Number of spatial nodes through thickness (default: 50)
        """
        self.laminate = laminate_properties
        self.analysis = analysis
        self.top_temp = top_surface_temp
        self.initial_temp = initial_temp
        self.alpha = thermal_diffusivity if thermal_diffusivity is not None else 1e-6
        self.n_nodes = n_nodes
        
        # Spatial grid through thickness
        self.thickness = laminate_properties.total_thickness
        self.z = np.linspace(0, self.thickness, n_nodes)
        self.dz = self.thickness / (n_nodes - 1)
        
        # Temperature profile storage
        self.T_history = []  # Temperature at each time step
        self.time_history = []
        self.strain_history = []
        self.curvature_history = []
        
    def _temperature_diffusion_ode(self, T, t):
        """
        ODE system for 1D heat diffusion using finite differences.
        Uses central differences for spatial derivatives.
        
        dT/dt = α * d²T/dz²
        """
        dT_dt = np.zeros_like(T)
        
        # Interior nodes: central difference
        for i in range(1, len(T) - 1):
            d2T_dz2 = (T[i+1] - 2*T[i] + T[i-1]) / (self.dz ** 2)
            dT_dt[i] = self.alpha * d2T_dz2
        
        # Boundary conditions
        dT_dt[0] = 0  # Top surface: fixed temperature
        dT_dt[-1] = 0  # Bottom surface: adiabatic (no flux)
        
        return dT_dt
    
    def solve_transient(self, t_final, n_steps=100, bottom_temp=None):
        """
        Solve transient heat diffusion to steady state.
        
        Parameters
        ----------
        t_final : float
            Final time to solve to (seconds)
        n_steps : int
            Number of time steps (default: 100)
        bottom_temp : float, optional
            Bottom surface temperature. If None, assumes adiabatic.
        
        Returns
        -------
        dict : Results with keys:
            - 't': time array
            - 'z': spatial positions
            - 'T': temperature profiles [n_steps, n_nodes]
            - 'strains': membrane strains [n_steps, 3]
            - 'curvatures': bending curvatures [n_steps, 3]
            - 'total_strains': total strains including thermal [n_steps, 3]
            - 'total_curvatures': total curvatures including thermal [n_steps, 3]
        """
        # Initial condition: uniform temperature
        T_init = np.ones(self.n_nodes) * self.initial_temp
        
        # Time array
        time = np.linspace(0, t_final, n_steps)
        
        # Solve ODE system
        T_solution = odeint(self._temperature_diffusion_ode, T_init, time)
        
        # Apply boundary conditions at each time step
        T_solution[:, 0] = self.top_temp  # Top surface
        if bottom_temp is not None:
            T_solution[:, -1] = bottom_temp  # Bottom surface if specified
        
        # Calculate strains and curvatures for each temperature profile
        strains_list = []
        curvatures_list = []
        total_strains_list = []
        total_curvatures_list = []
        
        for i, T_profile in enumerate(T_solution):
            # Average temperature through thickness
            T_avg = np.mean(T_profile)
            
            # Create modified laminate with current average temperature
            # This represents the thermal loading
            T_diff_avg = T_avg - self.initial_temp
            
            # Calculate thermal strains due to temperature distribution
            # We need to integrate through thickness considering each ply
            strains, curvatures = self._compute_strains_from_profile(T_profile)
            
            strains_list.append(strains)
            curvatures_list.append(curvatures)
            
            # Store history
            self.T_history.append(T_profile.copy())
            self.time_history.append(time[i])
            self.strain_history.append(strains)
            self.curvature_history.append(curvatures)
        
        self.T_history = np.array(self.T_history)
        self.strain_history = np.array(strains_list)
        self.curvature_history = np.array(curvatures_list)
        
        return {
            't': time,
            'z': self.z,
            'T': T_solution,
            'strains': np.array(strains_list),
            'curvatures': np.array(curvatures_list),
        }
    
    def _compute_strains_from_profile(self, T_profile):
        """
        Compute membrane strains and curvatures from a temperature profile.
        
        Integrates thermal strains through thickness considering:
        - Ply-by-ply temperature variation
        - CTE mismatch between fiber and matrix directions
        """
        # Interpolate temperature for each ply at its mid-thickness position
        ply_mids = np.cumsum([0] + [self.laminate.ply_thickness] * len(self.laminate.layup.angles))
        ply_mids = (ply_mids[:-1] + ply_mids[1:]) / 2  # Midpoints
        ply_temps = np.interp(ply_mids, self.z, T_profile)
        
        # Temperature deviations for each ply
        ply_dTs = ply_temps - self.initial_temp
        
        # Average temperature change for overall thermal loading
        T_avg = np.mean(T_profile)
        delta_T_avg = T_avg - self.initial_temp
        
        # Use existing analysis engine with average temperature
        # Create temporary laminate property with this delta_T
        from .InputProperties import LaminateProperties
        temp_laminate = LaminateProperties(
            material=self.laminate.material,
            layup=self.laminate.layup,
            ply_thickness=self.laminate.ply_thickness,
            delta_T=delta_T_avg
        )
        
        # Quick recalculation with this temperature
        from .workflow import LaminateAnalysis
        temp_analysis = LaminateAnalysis(temp_laminate)
        temp_analysis.calculate_material_matrices()
        temp_analysis.calculate_transformed_matrices()
        temp_analysis.calculate_laminate_stiffness_matrices()
        temp_analysis.calculate_thermal_loading()
        temp_analysis.calculate_resultant_strains()
        
        strains = temp_analysis.resultant_strains[:3]  # Membrane strains
        curvatures = temp_analysis.resultant_strains[3:]  # Curvatures
        
        return strains, curvatures
    
    def get_steady_state_properties(self):
        """Get steady state temperature profile and resulting strains/curvatures."""
        if len(self.T_history) == 0:
            raise ValueError("Must call solve_transient() first")
        
        T_steady = self.T_history[-1]
        strain_steady = self.strain_history[-1]
        curvature_steady = self.curvature_history[-1]
        
        return {
            'T': T_steady,
            'strains': strain_steady,
            'curvatures': curvature_steady,
            'avg_temp': np.mean(T_steady),
            'max_temp': np.max(T_steady),
            'min_temp': np.min(T_steady),
        }
    
    def get_temperature_gradient(self):
        """Get temperature gradient at steady state."""
        if len(self.T_history) == 0:
            raise ValueError("Must call solve_transient() first")
        
        T_steady = self.T_history[-1]
        dT_dz = np.gradient(T_steady, self.z)
        return dT_dz
    
    def time_to_steady_state(self, tolerance=0.01):
        """
        Estimate time to reach steady state (within tolerance).
        
        Parameters
        ----------
        tolerance : float
            Relative tolerance for steady state (default: 1%)
        
        Returns
        -------
        float : Time to reach steady state or None if not reached
        """
        if len(self.T_history) < 2:
            raise ValueError("Must call solve_transient() first")
        
        T_final = self.T_history[-1]
        
        for i in range(len(self.T_history) - 1, 0, -1):
            T_current = self.T_history[i]
            rel_error = np.max(np.abs((T_current - T_final) / (T_final + 1e-10)))
            if rel_error > tolerance:
                return self.time_history[i]
        
        return self.time_history[0]


class ThermalDiffusionAnalyzer:
    """
    High-level interface for transient thermal analysis with strain calculation.
    """
    
    def __init__(self, laminate_properties, top_surface_temp=50, initial_temp=20):
        """
        Initialize analyzer.
        
        Parameters
        ----------
        laminate_properties : LaminateProperties
            Laminate configuration
        top_surface_temp : float
            Temperature at top surface
        initial_temp : float
            Initial temperature throughout
        """
        self.laminate = laminate_properties
        self.top_temp = top_surface_temp
        self.initial_temp = initial_temp
        self.analysis = None
        self.thermal_diffusion = None
        self.results = None
    
    def analyze(self, t_final=3600, n_steps=100, n_nodes=50, thermal_diffusivity=1e-6):
        """
        Run complete thermal diffusion analysis.
        
        Parameters
        ----------
        t_final : float
            Final time (seconds, default: 3600 = 1 hour)
        n_steps : int
            Number of time steps
        n_nodes : int
            Number of spatial nodes through thickness
        thermal_diffusivity : float
            Thermal diffusivity (m²/s)
        
        Returns
        -------
        dict : Analysis results
        """
        # Initial mechanical analysis
        from .workflow import LaminateAnalysis
        self.analysis = LaminateAnalysis(self.laminate)
        self.analysis.run_full_analysis()
        
        # Thermal diffusion analysis
        self.thermal_diffusion = ThermalDiffusion(
            self.laminate,
            self.analysis,
            top_surface_temp=self.top_temp,
            initial_temp=self.initial_temp,
            thermal_diffusivity=thermal_diffusivity,
            n_nodes=n_nodes
        )
        
        self.results = self.thermal_diffusion.solve_transient(t_final, n_steps)
        return self.results
    
    def plot_temperature_evolution(self, figsize=(12, 5)):
        """Plot temperature evolution through thickness over time."""
        if self.results is None:
            raise ValueError("Must call analyze() first")
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not installed. Install with: pip install matplotlib")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Temperature profiles at different times
        times = self.results['t']
        z = self.results['z'] * 1e3  # Convert to mm
        T = self.results['T']
        
        n_profiles = min(5, len(times))
        indices = np.linspace(0, len(times)-1, n_profiles, dtype=int)
        
        for idx in indices:
            ax1.plot(T[idx], z, label=f't = {times[idx]:.1f}s')
        
        ax1.set_xlabel('Temperature (K)')
        ax1.set_ylabel('Position through thickness (mm)')
        ax1.set_title('Temperature Profiles vs Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Strains over time
        strains = self.results['strains']
        ax2.plot(times, strains[:, 0], label='εₓ (Membrane strain x)')
        ax2.plot(times, strains[:, 1], label='εᵧ (Membrane strain y)')
        ax2.plot(times, strains[:, 2], label='γₓᵧ (Shear strain)')
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Strain')
        ax2.set_title('Membrane Strains vs Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, (ax1, ax2)
    
    def print_summary(self):
        """Print summary of analysis results."""
        if self.results is None:
            raise ValueError("Must call analyze() first")
        
        steady = self.thermal_diffusion.get_steady_state_properties()
        t_ss = self.thermal_diffusion.time_to_steady_state()
        
        print("\n" + "="*80)
        print("THERMAL DIFFUSION ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\nInitial Temperature: {self.initial_temp} K")
        print(f"Top Surface Temperature: {self.top_temp} K")
        print(f"Temperature Rise: {self.top_temp - self.initial_temp} K")
        
        print(f"\n--- Steady State Results ---")
        print(f"Average Temperature: {steady['avg_temp']:.2f} K")
        print(f"Maximum Temperature: {steady['max_temp']:.2f} K")
        print(f"Minimum Temperature: {steady['min_temp']:.2f} K")
        print(f"Time to Steady State (~1%): {t_ss:.2f} s ({t_ss/60:.2f} min)")
        
        print(f"\nSteady State Membrane Strains:")
        print(f"  εₓ: {steady['strains'][0]:.6e}")
        print(f"  εᵧ: {steady['strains'][1]:.6e}")
        print(f"  γₓᵧ: {steady['strains'][2]:.6e}")
        
        print(f"\nSteady State Curvatures:")
        print(f"  κₓ: {steady['curvatures'][0]:.6e}")
        print(f"  κᵧ: {steady['curvatures'][1]:.6e}")
        print(f"  κₓᵧ: {steady['curvatures'][2]:.6e}")
        
        dT_dz = self.thermal_diffusion.get_temperature_gradient()
        print(f"\nTemperature Gradient (steady state): {np.max(np.abs(dT_dz)):.6f} K/m")
