import numpy as np

# ==============================================================================
# 1. INPUT PARAMETERS
# ==============================================================================
L = 0.3              # Plate length along X [m]
W = 0.3              # Plate width along Y [m]
t_ply = 0.000125     # Single ply thickness [m]

# Enter plies in bottom-to-top order.  Toggle this when the source layup is
# written top-to-bottom, such as an Abaqus composite-layup definition.
INPUT_LAYUP = [0,90,0,90,0]
REVERSE_LAYUP_ORDER = False
layup = list(reversed(INPUT_LAYUP)) if REVERSE_LAYUP_ORDER else list(INPUT_LAYUP)

# Plate-solver settings.  SHAPE_FUNCTION_ORDER is the highest total polynomial
# degree in the Ritz displacement basis.  Order 2 is quadratic; use 3 or 4 to
# add cubic or quartic deformation modes.
USE_VON_KARMAN_PLATE_THEORY = True
SHAPE_FUNCTION_ORDER = 5
FVK_MAX_ITERATIONS = 80
FVK_GRADIENT_TOLERANCE = 1.0e-7

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
h_top = 1000        # Top air knife film coefficient [W/(m^2*K)]

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
# 4. VON KARMAN RITZ PLATE SOLVER
# ==============================================================================
def make_polynomial_modes(min_degree, max_degree, exclude=None):
    """Return (x-power, y-power) modes ordered by total polynomial degree."""
    excluded = set(exclude or [])
    modes = []
    for degree in range(min_degree, max_degree + 1):
        for x_power in range(degree + 1):
            y_power = degree - x_power
            if (x_power, y_power) not in excluded:
                modes.append((x_power, y_power))
    return modes


def evaluate_modes(modes, x, y):
    """Evaluate normalized polynomial modes and their first/second derivatives."""
    xi = 2.0 * np.asarray(x) / L - 1.0
    eta = 2.0 * np.asarray(y) / W - 1.0
    scale_x = 2.0 / L
    scale_y = 2.0 / W

    values = []
    dx = []
    dy = []
    dxx = []
    dyy = []
    dxy = []
    for x_power, y_power in modes:
        values.append(xi**x_power * eta**y_power)
        dx.append(
            x_power * xi**(x_power - 1) * eta**y_power * scale_x
            if x_power else np.zeros_like(xi)
        )
        dy.append(
            y_power * xi**x_power * eta**(y_power - 1) * scale_y
            if y_power else np.zeros_like(xi)
        )
        dxx.append(
            x_power * (x_power - 1) * xi**(x_power - 2) * eta**y_power * scale_x**2
            if x_power >= 2 else np.zeros_like(xi)
        )
        dyy.append(
            y_power * (y_power - 1) * xi**x_power * eta**(y_power - 2) * scale_y**2
            if y_power >= 2 else np.zeros_like(xi)
        )
        dxy.append(
            x_power * y_power * xi**(x_power - 1) * eta**(y_power - 1) * scale_x * scale_y
            if x_power and y_power else np.zeros_like(xi)
        )
    return tuple(np.asarray(item, dtype=float).T for item in (values, dx, dy, dxx, dyy, dxy))


def gauss_quadrature(order):
    """Return tensor-product Gauss points and area weights over the plate."""
    points, weights = np.polynomial.legendre.leggauss(max(4, order + 2))
    xi, eta = np.meshgrid(points, points, indexing="ij")
    wx, wy = np.meshgrid(weights, weights, indexing="ij")
    x = 0.5 * L * (xi.ravel() + 1.0)
    y = 0.5 * W * (eta.ravel() + 1.0)
    area_weights = 0.25 * L * W * (wx * wy).ravel()
    return x, y, area_weights


def initial_ritz_coefficients(u_modes, v_modes, w_modes):
    """Seed the nonlinear solve with the constant-curvature CLT response."""
    u_coeff = np.zeros(len(u_modes))
    v_coeff = np.zeros(len(v_modes))
    w_coeff = np.zeros(len(w_modes))

    # The centered-coordinate linear terms represent uniform CLT strains.
    for index, mode in enumerate(u_modes):
        if mode == (1, 0):
            u_coeff[index] = 0.5 * L * eps_0[0]
        elif mode == (0, 1):
            u_coeff[index] = 0.5 * W * eps_0[2]
    for index, mode in enumerate(v_modes):
        if mode == (0, 1):
            v_coeff[index] = 0.5 * W * eps_0[1]

    # kappa = [-w,xx, -w,yy, -2 w,xy] for Kirchhoff-Love kinematics.
    for index, mode in enumerate(w_modes):
        if mode == (2, 0):
            w_coeff[index] = -kappa[0] * L**2 / 8.0
        elif mode == (0, 2):
            w_coeff[index] = -kappa[1] * W**2 / 8.0
        elif mode == (1, 1):
            w_coeff[index] = -kappa[2] * L * W / 8.0
    return np.concatenate((u_coeff, v_coeff, w_coeff))


def solve_von_karman_ritz():
    """Minimize free-laminate FvK potential energy with a Ritz polynomial basis."""
    if SHAPE_FUNCTION_ORDER < 2:
        raise ValueError("SHAPE_FUNCTION_ORDER must be at least 2.")

    u_modes = make_polynomial_modes(1, SHAPE_FUNCTION_ORDER)
    # Excluding v~x fixes the otherwise zero-energy in-plane rigid rotation.
    v_modes = make_polynomial_modes(1, SHAPE_FUNCTION_ORDER, exclude=[(1, 0)])
    w_modes = make_polynomial_modes(2, SHAPE_FUNCTION_ORDER)
    x_q, y_q, area_weights = gauss_quadrature(SHAPE_FUNCTION_ORDER)
    u_data = evaluate_modes(u_modes, x_q, y_q)
    v_data = evaluate_modes(v_modes, x_q, y_q)
    w_data = evaluate_modes(w_modes, x_q, y_q)
    n_u = len(u_modes)
    n_v = len(v_modes)
    initial = initial_ritz_coefficients(u_modes, v_modes, w_modes)

    # Scale unknowns before BFGS so micrometer in-plane motion and millimeter
    # out-of-plane motion are numerically comparable.
    scale = np.maximum(np.abs(initial), 1.0e-5)
    scale[n_u + n_v:] = np.maximum(scale[n_u + n_v:], 1.0e-4)

    def energy(normalized_coefficients):
        coefficients = normalized_coefficients * scale
        u_coeff = coefficients[:n_u]
        v_coeff = coefficients[n_u:n_u + n_v]
        w_coeff = coefficients[n_u + n_v:]

        _, u_x, u_y, _, _, _ = u_data
        _, v_x, v_y, _, _, _ = v_data
        _, w_x, w_y, w_xx, w_yy, w_xy = w_data

        eps_mid = np.column_stack((
            u_x @ u_coeff + 0.5 * (w_x @ w_coeff)**2,
            v_y @ v_coeff + 0.5 * (w_y @ w_coeff)**2,
            u_y @ u_coeff + v_x @ v_coeff + (w_x @ w_coeff) * (w_y @ w_coeff),
        ))
        curvature = np.column_stack((
            -(w_xx @ w_coeff),
            -(w_yy @ w_coeff),
            -2.0 * (w_xy @ w_coeff),
        ))
        density = (
            0.5 * np.einsum("ni,ij,nj->n", eps_mid, A, eps_mid)
            + np.einsum("ni,ij,nj->n", eps_mid, B, curvature)
            + 0.5 * np.einsum("ni,ij,nj->n", curvature, D, curvature)
            - eps_mid @ N_T
            - curvature @ M_T
        )
        return float(area_weights @ density)

    def gradient(q):
        step = 1.0e-5
        grad = np.zeros_like(q)
        for i in range(len(q)):
            delta = np.zeros_like(q)
            delta[i] = step
            grad[i] = (energy(q + delta) - energy(q - delta)) / (2.0 * step)
        return grad

    q = initial / scale
    hessian_inverse = np.eye(len(q))
    grad = gradient(q)
    converged = False
    for iteration in range(1, FVK_MAX_ITERATIONS + 1):
        if np.linalg.norm(grad, ord=np.inf) < FVK_GRADIENT_TOLERANCE:
            converged = True
            break
        direction = -hessian_inverse @ grad
        if np.dot(direction, grad) >= 0.0:
            direction = -grad
            hessian_inverse = np.eye(len(q))

        current_energy = energy(q)
        step_length = 1.0
        while energy(q + step_length * direction) > current_energy + 1.0e-4 * step_length * np.dot(grad, direction):
            step_length *= 0.5
            if step_length < 1.0e-8:
                break
        q_next = q + step_length * direction
        grad_next = gradient(q_next)
        step_vector = q_next - q
        grad_change = grad_next - grad
        curvature_measure = np.dot(grad_change, step_vector)
        if curvature_measure > 1.0e-14:
            identity = np.eye(len(q))
            outer = np.outer(step_vector, grad_change)
            hessian_inverse = (
                (identity - outer / curvature_measure) @ hessian_inverse
                @ (identity - outer.T / curvature_measure)
                + np.outer(step_vector, step_vector) / curvature_measure
            )
        q, grad = q_next, grad_next
    else:
        iteration = FVK_MAX_ITERATIONS

    return {
        "u_modes": u_modes,
        "v_modes": v_modes,
        "w_modes": w_modes,
        "coefficients": q * scale,
        "n_u": n_u,
        "n_v": n_v,
        "iterations": iteration,
        "converged": converged,
        "gradient_norm": float(np.linalg.norm(grad, ord=np.inf)),
    }


FVK_SOLUTION = solve_von_karman_ritz() if USE_VON_KARMAN_PLATE_THEORY else None


def plate_kinematics(x, y):
    """Return u0, v0, w, slopes, FvK midplane strain, and curvature at x/y."""
    if FVK_SOLUTION is None:
        w, w_x, w_y = plate_w_and_slopes(x, y)
        return (
            eps_0[0] * x + 0.5 * eps_0[2] * y,
            eps_0[1] * y + 0.5 * eps_0[2] * x,
            w, w_x, w_y, eps_0, kappa,
        )

    u_values, u_x, u_y, _, _, _ = evaluate_modes(FVK_SOLUTION["u_modes"], x, y)
    v_values, v_x, v_y, _, _, _ = evaluate_modes(FVK_SOLUTION["v_modes"], x, y)
    w_values, w_x, w_y, w_xx, w_yy, w_xy = evaluate_modes(FVK_SOLUTION["w_modes"], x, y)
    coefficients = FVK_SOLUTION["coefficients"]
    n_u = FVK_SOLUTION["n_u"]
    n_v = FVK_SOLUTION["n_v"]
    u_coeff = coefficients[:n_u]
    v_coeff = coefficients[n_u:n_u + n_v]
    w_coeff = coefficients[n_u + n_v:]
    slope_x = w_x @ w_coeff
    slope_y = w_y @ w_coeff
    eps_mid = np.array([
        u_x @ u_coeff + 0.5 * slope_x**2,
        v_y @ v_coeff + 0.5 * slope_y**2,
        u_y @ u_coeff + v_x @ v_coeff + slope_x * slope_y,
    ])
    curvature = np.array([
        -(w_xx @ w_coeff),
        -(w_yy @ w_coeff),
        -2.0 * (w_xy @ w_coeff),
    ])
    return (
        u_values @ u_coeff,
        v_values @ v_coeff,
        w_values @ w_coeff,
        slope_x,
        slope_y,
        eps_mid,
        curvature,
    )


# ==============================================================================
# 5. DISPLACEMENT AND STRAIN EVALUATION FUNCTIONS
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
    u_0, v_0, w, dw_dx, dw_dy, _, _ = plate_kinematics(x, y)

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

    _, _, _, _, _, eps_mid, local_kappa = plate_kinematics(x, y)
    eps_total = eps_mid + z * local_kappa

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


def format_abd_block(abd):
    """Return a readable 6x6 ABD matrix with A/B/D block separators."""
    lines = []
    horizontal_rule = "   " + "-" * 91

    for row_index, row in enumerate(abd):
        if row_index == 3:
            lines.append(horizontal_rule)

        left = " ".join(f"{value: .4e}" for value in row[:3])
        right = " ".join(f"{value: .4e}" for value in row[3:])
        lines.append(f"   {left} | {right}")

    return "\n".join(lines)


# ==============================================================================
# 5. PRINT RESULTS & ANALYSIS
# ==============================================================================
print("=" * 70)
print("                   CLT FULL DISPLACEMENT & STRAIN ANALYSIS              ")
print("=" * 70)
print(f"Layup used (bottom-to-top): {layup}")
print(f"Layup order reversed: {REVERSE_LAYUP_ORDER}")
print(f"Von Karman Ritz solver: {USE_VON_KARMAN_PLATE_THEORY}")
if FVK_SOLUTION is not None:
    print(f"Shape-function order: {SHAPE_FUNCTION_ORDER}")
    print(
        f"Ritz modes: u={FVK_SOLUTION['n_u']}, v={FVK_SOLUTION['n_v']}, "
        f"w={len(FVK_SOLUTION['w_modes'])}; iterations={FVK_SOLUTION['iterations']}; "
        f"converged={FVK_SOLUTION['converged']}"
    )

print("\n1. MIDPLANE STRAINS & CURVATURES:")
print(f"   eps_x^0   = {eps_0[0] * 1e6:10.2f} microstrain")
print(f"   eps_y^0   = {eps_0[1] * 1e6:10.2f} microstrain")
print(f"   gamma_xy^0= {eps_0[2] * 1e6:10.2f} microstrain")
print(f"   kappa_x   = {kappa[0]:10.5f} m^-1")
print(f"   kappa_y   = {kappa[1]:10.5f} m^-1")
print(f"   kappa_xy  = {kappa[2]:10.5f} m^-1")

print("\n1b. ABD MATRIX BLOCK VIEW:")
print("   Columns 1-3 are A/B blocks; columns 4-6 are B/D blocks.")
print("   Rows 1-3 are A/B blocks; rows 4-6 are B/D blocks.")
print(format_abd_block(ABD))

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


def ply_index_at_z(z_val):
    """Return the valid ply index containing z_val for the current layup."""
    ply_i = int(np.searchsorted(z_coords, z_val, side="right") - 1)
    return int(np.clip(ply_i, 0, n_plies - 1))


eval_points = [
    ("Top Face (+h/2)", h / 2.0),
    ("Upper Quarter", h / 4.0),
    ("Midplane (z=0)", 0.0),
    ("Lower Quarter", -h / 4.0),
    ("Bottom Face (-h/2)", -h / 2.0),
]

for label, z_val in eval_points:
    ply_i = ply_index_at_z(z_val)
    e_tot, e_th, e_mech, sig = calc_strains_and_stresses(xc, yc, z_val, ply_i)
    print(f"{label:18s} | {e_tot[0] * 1e6:14.1f} | {e_th[0] * 1e6:14.1f} | {e_mech[0] * 1e6:14.1f} | {sig[0] / 1e6:12.2f}")

print("=" * 70)
