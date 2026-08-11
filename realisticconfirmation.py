import numpy as np

# ==============================================================================
# 1. INPUT PARAMETERS
# ==============================================================================
L = 0.3              # Plate length along X [m]
W = 0.3              # Plate width along Y [m]
t_ply = 0.000125     # Single ply thickness [m]
layup = [0, 45, -45, -45, 45, 0]

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

# Numerical / reporting settings
NUM_MAX_GRID = 201

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
T_mid = 0.5 * (T_top_surf + T_bot_surf)
dT_0 = T_mid - T_init
beta = (T_top_surf - T_bot_surf) / h

print("top temp: ", T_top_surf)
print("bottom temp: ", T_bot_surf)

# ==============================================================================
# 3. CLT MATRIX FORMULATION (ABD, N_T, M_T)
# ==============================================================================
nu21 = nu12 * (E2 / E1)

# Stiffness in material coordinates
Q11 = E1 / (1.0 - nu12 * nu21)
Q22 = E2 / (1.0 - nu12 * nu21)
Q12 = nu12 * E2 / (1.0 - nu12 * nu21)
Q66 = G12

Q = np.array([
    [Q11, Q12, 0.0],
    [Q12, Q22, 0.0],
    [0.0, 0.0, Q66]
], dtype=float)

z_coords = np.linspace(-h / 2.0, h / 2.0, n_plies + 1)

A = np.zeros((3, 3), dtype=float)
B = np.zeros((3, 3), dtype=float)
D = np.zeros((3, 3), dtype=float)
N_T = np.zeros(3, dtype=float)
M_T = np.zeros(3, dtype=float)

Q_bar_list = []
alpha_bar_list = []

for i, angle in enumerate(layup):
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)

    # Engineering-shear strain transformation matrix
    T_mat = np.array([
        [c**2, s**2, 2.0 * c * s],
        [s**2, c**2, -2.0 * c * s],
        [-c * s, c * s, c**2 - s**2]
    ], dtype=float)

    R_mat = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 2.0]
    ], dtype=float)

    Q_bar = np.linalg.inv(T_mat) @ Q @ R_mat @ T_mat @ np.linalg.inv(R_mat)
    alpha_bar = np.linalg.inv(T_mat) @ np.array([alpha1, alpha2, 0.0], dtype=float)

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

    N_T += (Q_bar @ alpha_bar) * dt_k
    M_T += (Q_bar @ alpha_bar) * dt_m

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
def plate_w_and_slopes(x, y):
    """
    Quadratic plate shape used for reconstruction.
    """
    kx, ky, kxy = kappa[0], kappa[1], kappa[2]

    # Out-of-plane deflection
    w = 0.5 * kx * x * (L - x) + 0.5 * ky * y * (W - y) - 0.5 * kxy * x * y

    # Slopes
    dw_dx = 0.5 * kx * (L - 2.0 * x) - 0.5 * kxy * y
    dw_dy = 0.5 * ky * (W - 2.0 * y) - 0.5 * kxy * x

    return w, dw_dx, dw_dy


def calc_displacements(x, y, z):
    """
    3D displacements using von Karman-compatible kinematics.
    """
    ex0, ey0, gxy0 = eps_0[0], eps_0[1], eps_0[2]
    w, dw_dx, dw_dy = plate_w_and_slopes(x, y)

    # Symmetric in-plane midplane field
    u_0 = ex0 * x + 0.5 * gxy0 * y
    v_0 = ey0 * y + 0.5 * gxy0 * x

    # Through-thickness displacement
    u = u_0 - z * dw_dx
    v = v_0 - z * dw_dy

    return np.array([u, v, w], dtype=float)


def calc_strains_and_stresses(x, y, z, ply_idx):
    """
    Total, thermal, mechanical strains and stresses using von Karman terms.
    """
    Q_bar = Q_bar_list[ply_idx]
    alpha_bar = alpha_bar_list[ply_idx]

    _, dw_dx, dw_dy = plate_w_and_slopes(x, y)

    # Linear CLT part + von Karman nonlinear part
    eps_linear = eps_0 + z * kappa
    eps_nl = np.array([
        0.5 * dw_dx**2,
        0.5 * dw_dy**2,
        dw_dx * dw_dy
    ], dtype=float)

    eps_total = eps_linear + eps_nl

    # Free thermal strain
    dT_z = dT_0 + beta * z
    eps_thermal = alpha_bar * dT_z

    # Mechanical strain
    eps_mech = eps_total - eps_thermal

    # Stress in global (x,y) frame
    stress_xy = Q_bar @ eps_mech

    return eps_total, eps_thermal, eps_mech, stress_xy


def find_max_displacements(num_xy_points=NUM_MAX_GRID):
    """Find max displacement magnitude and component extrema over plate faces."""
    x_values = np.linspace(0.0, L, num_xy_points)
    y_values = np.linspace(0.0, W, num_xy_points)
    z_values = np.array([-h / 2.0, 0.0, h / 2.0])

    xx, yy, zz = np.meshgrid(x_values, y_values, z_values, indexing="ij")
    displacements = calc_displacements(xx.ravel(), yy.ravel(), zz.ravel()).T
    locations = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

    magnitudes = np.linalg.norm(displacements, axis=1)
    magnitude_idx = int(np.argmax(magnitudes))

    component_results = {}
    for component_idx, component_name in enumerate(("u", "v", "w")):
        max_idx = int(np.argmax(np.abs(displacements[:, component_idx])))
        component_results[component_name] = {
            "value": displacements[max_idx, component_idx],
            "location": locations[max_idx],
        }

    return {
        "magnitude": {
            "value": magnitudes[magnitude_idx],
            "components": displacements[magnitude_idx],
            "location": locations[magnitude_idx],
        },
        "components": component_results,
        "num_xy_points": num_xy_points,
    }


def format_location(location):
    x, y, z = location
    return f"(x={x:.6f} m, y={y:.6f} m, z={z:.6f} m)"


# ==============================================================================
# 5. PRINT RESULTS & ANALYSIS
# ==============================================================================
print("=" * 70)
print("                   CLT FULL DISPLACEMENT & STRAIN ANALYSIS              ")
print("=" * 70)

print("\n1. MIDPLANE STRAINS & CURVATURES:")
print(f"   eps_x^0   = {eps_0[0] * 1e6:10.2f} microstrain")
print(f"   eps_y^0   = {eps_0[1] * 1e6:10.2f} microstrain")
print(f"   gamma_xy^0= {eps_0[2] * 1e6:10.2f} microstrain")
print(f"   kappa_x   = {kappa[0]:10.5f} m^-1")
print(f"   kappa_y   = {kappa[1]:10.5f} m^-1")
print(f"   kappa_xy  = {kappa[2]:10.5f} m^-1")

print("\n2. DISPLACEMENTS AT KEY PLATE LOCATIONS [mm]:")
pts = {
    "Center (L/2, W/2, z=0)": (L / 2, W / 2, 0.0),
    "Corner Top (L, W, +h/2)": (L, W, h / 2),
    "Corner Bot (L, W, -h/2)": (L, W, -h / 2),
    "Mid-Edge Top (L/2, W, +h/2)": (L / 2, W, h / 2),
    "Top far left corner (0, W, +h/2)": (0.0, W, h / 2),
    "Top far right corner (L, W, +h/2)": (L, W, h / 2),
    "Top close left corner (0, 0, +h/2)": (0.0, 0.0, h / 2),
    "Top close right corner (L, 0, +h/2)": (L, 0.0, h / 2),
}

for name, (x, y, z_val) in pts.items():
    u, v, w = calc_displacements(x, y, z_val) * 1000.0
    print(f"   {name:30s} -> u: {u:7.3f} mm | v: {v:7.3f} mm | w: {w:7.3f} mm")

print("\n3. MIDPLANE DISPLACEMENTS AT EACH PLATE CORNER [mm]:")
corner_pts = {
    "Corner 1 (0, 0, z=0)": (0.0, 0.0, 0.0),
    "Corner 2 (L, 0, z=0)": (L, 0.0, 0.0),
    "Corner 3 (0, W, z=0)": (0.0, W, 0.0),
    "Corner 4 (L, W, z=0)": (L, W, 0.0),
}
for name, (x, y, z_val) in corner_pts.items():
    u, v, w = calc_displacements(x, y, z_val) * 1000.0
    print(f"   {name:25s} -> u: {u:7.3f} mm | v: {v:7.3f} mm | w: {w:7.3f} mm")

print("\n4. MAX DISPLACEMENTS OVER PLATE TOP/MID/BOTTOM [mm]:")
max_report = find_max_displacements()
max_mag = max_report["magnitude"]
max_mag_components = max_mag["components"] * 1000.0

print(
    f"   Max |d|: {max_mag['value'] * 1000.0:7.3f} mm at "
    f"{format_location(max_mag['location'])}"
)
print(
    f"      components there -> u: {max_mag_components[0]:7.3f} mm | "
    f"v: {max_mag_components[1]:7.3f} mm | w: {max_mag_components[2]:7.3f} mm"
)
for component_name, result in max_report["components"].items():
    print(
        f"   Max |{component_name}|: {abs(result['value']) * 1000.0:7.3f} mm "
        f"(signed {result['value'] * 1000.0:7.3f} mm) at "
        f"{format_location(result['location'])}"
    )
print(f"   Search grid: {max_report['num_xy_points']} x {max_report['num_xy_points']} x 3")

print("\n5. THROUGH-THICKNESS STRAINS & STRESSES AT PLATE CENTER (L/2, W/2):")
print("-" * 70)
print(f"{'Location':18s} | {'Total exx (ue)':14s} | {'Therm exx (ue)':14s} | {'Mech exx (ue)':14s} | {'Stress_x (MPa)':12s}")
print("-" * 70)

xc = L / 2.0
yc = W / 2.0

eval_points = [
    ("Top Face (+h/2)", h / 2.0, 3),
    ("Ply 3/4 Interface", h / 4.0, 2),
    ("Midplane (z=0)", 0.0, 1),
    ("Ply 1/2 Interface", -h / 4.0, 1),
    ("Bottom Face (-h/2)", -h / 2.0, 0),
]

for label, z_val, ply_i in eval_points:
    e_tot, e_th, e_mech, sig = calc_strains_and_stresses(xc, yc, z_val, ply_i)
    print(f"{label:18s} | {e_tot[0] * 1e6:14.1f} | {e_th[0] * 1e6:14.1f} | {e_mech[0] * 1e6:14.1f} | {sig[0] / 1e6:12.2f}")

print("=" * 70)
