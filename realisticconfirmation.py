import numpy as np

# ==============================================================================
# 1. INPUT PARAMETERS
# ==============================================================================
L = 0.3              # Plate length along X [m]
W = 0.3              # Plate width along Y [m]
t_ply = 0.000125     # Single ply thickness [m]
layup = [-45,0,45,0,45, 0,-45]
 # Layup orientation angles [deg]

# Material Mechanical Properties (Carbon/Epoxy)
E1 = 130e9           # Fiber direction modulus [Pa]
E2 = 10e9            # Transverse modulus [Pa]
nu12 = 0.3           # Major Poisson's ratio
G12 = 5e9            # In-plane shear modulus [Pa]

# Material Thermal Properties
alpha1 = -0.5e-6     # Fiber CTE [/°C]
alpha2 = 30e-6       # Transverse CTE [/°C]
kz = 0.5             # Transverse thermal conductivity [W/(m*K)]

# Thermal Boundary Conditions
T_init = 25.0        # Initial stress-free temp [°C]
T_bot_sink = 100.0   # Hot plate temperature [°C]
T_top_sink = 0.0     # Air knife temperature [°C]
h_bot = 1000.0       # Bottom contact film coefficient [W/(m^2*K)]
h_top = 400.0        # Top air knife film coefficient [W/(m^2*K)]


# ==============================================================================
# 2. THERMAL GRADIENT & HEAT FLUX
# ==============================================================================
n_plies = len(layup)
h = n_plies * t_ply  # Total laminate thickness [m]

R_bot = 1.0 / h_bot
R_cond = h / kz
R_top = 1.0 / h_top
R_total = R_bot + R_cond + R_top

q_flux = (T_bot_sink - T_top_sink) / R_total

T_bot_surf = T_bot_sink - (q_flux * R_bot)
T_top_surf = T_top_sink + (q_flux * R_top)

# Midplane temp change and gradient
T_mid = (T_top_surf + T_bot_surf) / 2.0
dT_0 = T_mid - T_init                   # Uniform temp rise [°C]
beta = (T_top_surf - T_bot_surf) / h    # Gradient dT/dz [°C/m]


# ==============================================================================
# 3. CLT MATRIX FORMULATION (ABD, N_T, M_T)
# ==============================================================================
nu21 = nu12 * (E2 / E1)

# Stiffness in material coordinates
Q11 = E1 / (1 - nu12 * nu21)
Q22 = E2 / (1 - nu12 * nu21)
Q12 = nu12 * E2 / (1 - nu12 * nu21)
Q66 = G12

Q = np.array([
    [Q11, Q12, 0.0],
    [Q12, Q22, 0.0],
    [0.0, 0.0, Q66]
])

z_coords = np.linspace(-h / 2.0, h / 2.0, n_plies + 1)

A = np.zeros((3, 3))
B = np.zeros((3, 3))
D = np.zeros((3, 3))
N_T = np.zeros(3)
M_T = np.zeros(3)

Q_bar_list = []
alpha_bar_list = []

for i, angle in enumerate(layup):
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)

    # Transformation matrices
    T_mat = np.array([
        [c**2, s**2, 2 * c * s],
        [s**2, c**2, -2 * c * s],
        [-c * s, c * s, c**2 - s**2]
    ])
    
    R_mat = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 2.0]
    ])

    Q_bar = np.linalg.inv(T_mat) @ Q @ R_mat @ T_mat @ np.linalg.inv(R_mat)
    alpha_bar = np.linalg.inv(T_mat) @ np.array([alpha1, alpha2, 0.0])

    Q_bar_list.append(Q_bar)
    alpha_bar_list.append(alpha_bar)

    z_k = z_coords[i + 1]
    z_km1 = z_coords[i]

    # ABD integrations
    A += Q_bar * (z_k - z_km1)
    B += 0.5 * Q_bar * (z_k**2 - z_km1**2)
    D += (1.0 / 3.0) * Q_bar * (z_k**3 - z_km1**3)

    # Thermal force and moment integration
    dt_k = dT_0 * (z_k - z_km1) + 0.5 * beta * (z_k**2 - z_km1**2)
    dt_m = 0.5 * dT_0 * (z_k**2 - z_km1**2) + (1.0 / 3.0) * beta * (z_k**3 - z_km1**3)

    N_T += Q_bar @ alpha_bar * dt_k
    M_T += Q_bar @ alpha_bar * dt_m

# Full ABD Matrix
ABD = np.block([
    [A, B],
    [B, D]
])

NM_T = np.concatenate([N_T, M_T])
eps_kappa = np.linalg.inv(ABD) @ NM_T

eps_0 = eps_kappa[:3]   # Midplane strains [exx, eyy, gxy]
kappa = eps_kappa[3:]   # Curvatures [kx, ky, kxy]


# ==============================================================================
# 4. DISPLACEMENT AND STRAIN EVALUATION FUNCTIONS
# ==============================================================================
def calc_displacements(x, y, z):
    """
    Calculates 3D displacements u, v, w strictly enforcing 3-2-1 BCs:
      - Node 1 (0, 0, 0): u = 0, v = 0, w = 0
      - Node 2 (L, 0, 0): v = 0, w = 0
      - Node 3 (0, W, 0): w = 0
    """
    kx, ky, kxy = kappa[0], kappa[1], kappa[2]
    ex0, ey0, gxy0 = eps_0[0], eps_0[1], eps_0[2]

    # 1. Out-of-plane deflection w(x,y)
    w = 0.5 * kx * x * (L - x) + 0.5 * ky * y * (W - y) - 0.5 * kxy * x * y

    # 2. Slopes dw/dx and dw/dy
    dw_dx = 0.5 * kx * (L - 2 * x) - 0.5 * kxy * y
    dw_dy = 0.5 * ky * (W - 2 * y) - 0.5 * kxy * x

    # 3. Midplane in-plane displacements (3-2-1 BCs fix v(L,0) = 0)
    u_0 = ex0 * x + gxy0 * y
    v_0 = ey0 * y

    # 4. Total 3D displacements through thickness z
    u = u_0 - z * dw_dx
    v = v_0 - z * dw_dy

    return np.array([u, v, w])

def calc_strains_and_stresses(z, ply_idx):
    """Calculates Total, Thermal, Mechanical Strains and Stresses at height z."""
    Q_bar = Q_bar_list[ply_idx]
    alpha_bar = alpha_bar_list[ply_idx]

    # Total geometric strain
    eps_total = eps_0 + z * kappa

    # Free thermal strain
    dT_z = dT_0 + beta * z
    eps_thermal = alpha_bar * dT_z

    # Mechanical strain (drives thermal stresses)
    eps_mech = eps_total - eps_thermal

    # Stress in global (x,y) frame
    stress_xy = Q_bar @ eps_mech

    return eps_total, eps_thermal, eps_mech, stress_xy


# ==============================================================================
# 5. PRINT RESULTS & ANALYSIS
# ==============================================================================
print("=" * 70)
print("                   CLT FULL DISPLACEMENT & STRAIN ANALYSIS              ")
print("=" * 70)

print(f"\n1. MIDPLANE STRAINS & CURVATURES:")
print(f"   eps_x^0  = {eps_0[0]*1e6:10.2f} microstrain")
print(f"   eps_y^0  = {eps_0[1]*1e6:10.2f} microstrain")
print(f"   gamma_xy^0 = {eps_0[2]*1e6:10.2f} microstrain")
print(f"   kappa_x  = {kappa[0]:10.5f} m^-1")
print(f"   kappa_y  = {kappa[1]:10.5f} m^-1")

print("\n2. DISPLACEMENTS AT KEY PLATE LOCATIONS [mm]:")
pts = {
    "Center (L/2, W/2, z=0)": (L/2, W/2, 0),
    "Corner Top (L, W, +h/2)": (L, W, h/2),
    "Corner Bot (L, W, -h/2)": (L, W, -h/2),
    "Mid-Edge Top (L/2, W, +h/2)": (L/2, W, h/2)
}
for name, (x, y, z_val) in pts.items():
    u, v, w = calc_displacements(x, y, z_val) * 1000.0  # convert to mm
    print(f"   {name:30s} -> u: {u:7.3f} mm | v: {v:7.3f} mm | w: {w:7.3f} mm")

print("\n3. THROUGH-THICKNESS STRAINS & STRESSES AT PLATE CENTER (L/2, W/2):")
print("-" * 70)
print(f"{'Location':18s} | {'Total exx (ue)':14s} | {'Therm exx (ue)':14s} | {'Mech exx (ue)':14s} | {'Stress_x (MPa)':12s}")
print("-" * 70)

eval_points = [
    ("Top Face (+h/2)", h/2, 3),
    ("Ply 3/4 Interface", h/4, 2),
    ("Midplane (z=0)", 0.0, 1),
    ("Ply 1/2 Interface", -h/4, 1),
    ("Bottom Face (-h/2)", -h/2, 0)
]

for label, z_val, ply_i in eval_points:
    e_tot, e_th, e_mech, sig = calc_strains_and_stresses(z_val, ply_i)
    print(f"{label:18s} | {e_tot[0]*1e6:14.1f} | {e_th[0]*1e6:14.1f} | {e_mech[0]*1e6:14.1f} | {sig[0]/1e6:12.2f}")

print("=" * 70)