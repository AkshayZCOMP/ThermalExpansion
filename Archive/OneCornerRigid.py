from dataclasses import dataclass

import numpy as np

try:
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover - exercised only without scipy installed
    minimize = None

# ==============================================================================
# 1. INPUT PARAMETERS
# ==============================================================================
L = 0.3              # Plate length [m]
W = 0.3              # Plate width [m]
t_ply = 0.000125     # Single ply thickness [m]
layup = [90,0,0,90]

# Material mechanical properties (carbon/epoxy)
E1 = 130e9
E2 = 10e9
nu12 = 0.3
G12 = 5e9

# Material thermal properties
alpha1 = -0.5e-6
alpha2 = 30e-6
kz = 0.5

# Thermal boundary conditions
T_init = 25.0
T_bot_sink = 100.0
T_top_sink = 0.0
h_bot = 1000.0
h_top = 400.0

# Numerical settings
AREA_QUAD_N = 7
MAX_ITERS = 300
GRAD_TOL = 1e-7
STEP_TOL = 1e-10
FD_STEP = 1e-6
LINESEARCH_C1 = 1e-4
LINESEARCH_SHRINK = 0.5
LINESEARCH_MIN_ALPHA = 1e-14

# Visualization settings. The first surface is true scale; the second is
# amplified only to make small bending shapes visible on a 0.3 m plate.
PLOT_RESULTS = True
PLOT_GRID_N = 61
PLOT_TRUE_SCALE = 1.0
PLOT_AMPLIFIED_SCALE = 10.0
PLOT_OUTPUT = "freefree_resultant_shape.png"
PLOT_SHOW = False

# Hard-constraint location.
# The single physical support is an encastre at the laminate midplane corner.
BC_Z = 0.0

# The corner boundary conditions are built into the Ritz basis. This flag is
# retained only for compatibility with older callers and is no longer used.
CONSTRAIN_ALL_CORNER_W = False

# ==============================================================================
# 2. THERMAL FIELD (STEADY 1D THROUGH-THICKNESS MODEL)
# ==============================================================================
n_plies = len(layup)
h = n_plies * t_ply

R_bot = 1.0 / h_bot
R_cond = h / kz
R_top = 1.0 / h_top
R_total = R_bot + R_cond + R_top

q_flux = (T_bot_sink - T_top_sink) / R_total
T_bot_surf = T_bot_sink - q_flux * R_bot
T_top_surf = T_top_sink + q_flux * R_top

T_mid = 0.5 * (T_top_surf + T_bot_surf)
dT_0 = T_mid - T_init
beta = (T_top_surf - T_bot_surf) / h

REL_GRAD_TOL = 1e-8

# ==============================================================================
# 3. CLT MATERIAL AND LAMINATE MATRICES
# ==============================================================================
nu21 = nu12 * (E2 / E1)

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


def transformed_ply_matrices(angle):
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)

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
    strain_transform = R_mat @ T_mat @ np.linalg.inv(R_mat)
    alpha_bar = np.linalg.inv(strain_transform) @ np.array([alpha1, alpha2, 0.0], dtype=float)

    return Q_bar, alpha_bar

A = np.zeros((3, 3), dtype=float)
B = np.zeros((3, 3), dtype=float)
D = np.zeros((3, 3), dtype=float)
N_T = np.zeros(3, dtype=float)
M_T = np.zeros(3, dtype=float)

Q_bar_list = []
alpha_bar_list = []

for i, angle in enumerate(layup):
    Q_bar, alpha_bar = transformed_ply_matrices(angle)

    Q_bar_list.append(Q_bar)
    alpha_bar_list.append(alpha_bar)

    z_k = z_coords[i + 1]
    z_km1 = z_coords[i]

    A += Q_bar * (z_k - z_km1)
    B += 0.5 * Q_bar * (z_k**2 - z_km1**2)
    D += (1.0 / 3.0) * Q_bar * (z_k**3 - z_km1**3)

    dt_k = dT_0 * (z_k - z_km1) + 0.5 * beta * (z_k**2 - z_km1**2)
    dt_m = 0.5 * dT_0 * (z_k**2 - z_km1**2) + (1.0 / 3.0) * beta * (z_k**3 - z_km1**3)

    N_T += (Q_bar @ alpha_bar) * dt_k
    M_T += (Q_bar @ alpha_bar) * dt_m

ABD = np.block([
    [A, B],
    [B, D]
])

# ==============================================================================
# 4. RITZ BASIS AND ENCASTRE SHAPE FUNCTIONS
# ==============================================================================
# Coordinates are anchored at the encastre corner (0, 0).
# Unknowns:
#   u0(x,y) = sum a_i * basis_u
#   v0(x,y) = sum b_i * basis_v
#   w(x,y)  = sum c_i * basis_w
#
# The basis enforces the corner translations and transverse rotations. A single
# exact linear constraint below removes the remaining in-plane rigid rotation.
# ==============================================================================

def center_xy(x, y):
    return x, y

def basis_u(x, y):
    # [x, y, x^2, xy, y^2, x^3, x^2 y, x y^2, y^3]
    return np.array([
        x,
        y,
        x * x,
        x * y,
        y * y,
        x * x * x,
        x * x * y,
        x * y * y,
        y * y * y
    ], dtype=float)

def dbasis_u_dx(x, y):
    return np.array([
        1.0,
        0.0,
        2.0 * x,
        y,
        0.0,
        3.0 * x * x,
        2.0 * x * y,
        y * y,
        0.0
    ], dtype=float)

def dbasis_u_dy(x, y):
    return np.array([
        0.0,
        1.0,
        0.0,
        x,
        2.0 * y,
        0.0,
        x * x,
        2.0 * x * y,
        3.0 * y * y
    ], dtype=float)

def basis_v(x, y):
    return basis_u(x, y)

def dbasis_v_dx(x, y):
    return dbasis_u_dx(x, y)

def dbasis_v_dy(x, y):
    return dbasis_u_dy(x, y)

def basis_w(x, y):
    # w = 1/2*c1*x^2 + 1/2*c2*y^2 + 1/2*c3*x*y
    # Omitting the constant and linear terms imposes w=0, dw/dx=0 and
    # dw/dy=0 at the encastre corner.
    return 0.5 * np.array([x * x, y * y, x * y], dtype=float)

def dbasis_w_dx(x, y):
    return np.array([x, 0.0, 0.5 * y], dtype=float)

def dbasis_w_dy(x, y):
    return np.array([0.0, y, 0.5 * x], dtype=float)

def d2basis_w_xx(x, y):
    return np.array([1.0, 0.0, 0.0], dtype=float)

def d2basis_w_xy(x, y):
    return np.array([0.0, 0.0, 0.5], dtype=float)

def d2basis_w_yy(x, y):
    return np.array([0.0, 1.0, 0.0], dtype=float)

NU = 9
NV = 9
NW = 3
NQ = NU + NV + NW


@dataclass
class SolverResult:
    q: np.ndarray
    energy: float
    projected_gradient: np.ndarray
    converged: bool
    iterations: int
    message: str
    method: str
    initial_gradient_norm: float

    @property
    def gradient_norm(self):
        return float(np.linalg.norm(self.projected_gradient))

    @property
    def relative_gradient_norm(self):
        return self.gradient_norm / max(self.initial_gradient_norm, 1.0)

def unpack(q):
    a = q[0:NU]
    b = q[NU:NU + NV]
    c = q[NU + NV:]
    return a, b, c

def eval_fields(x, y, q):
    """
    Returns:
      u0, v0, w
      du0/dx, du0/dy
      dv0/dx, dv0/dy
      dw/dx, dw/dy
      d2w/dx2, d2w/dxdy, d2w/dy2
    """
    xc, yc = center_xy(x, y)
    a, b, c = unpack(q)

    bu = basis_u(xc, yc)
    bv = basis_v(xc, yc)
    bw = basis_w(xc, yc)

    dux_b = dbasis_u_dx(xc, yc)
    duy_b = dbasis_u_dy(xc, yc)
    dvx_b = dbasis_v_dx(xc, yc)
    dvy_b = dbasis_v_dy(xc, yc)
    dwx_b = dbasis_w_dx(xc, yc)
    dwy_b = dbasis_w_dy(xc, yc)

    u0 = np.dot(a, bu)
    v0 = np.dot(b, bv)
    w = np.dot(c, bw)

    du0_dx = np.dot(a, dux_b)
    du0_dy = np.dot(a, duy_b)
    dv0_dx = np.dot(b, dvx_b)
    dv0_dy = np.dot(b, dvy_b)

    dw_dx = np.dot(c, dwx_b)
    dw_dy = np.dot(c, dwy_b)

    d2w_xx = np.dot(c, d2basis_w_xx(xc, yc))
    d2w_xy = np.dot(c, d2basis_w_xy(xc, yc))
    d2w_yy = np.dot(c, d2basis_w_yy(xc, yc))

    return (
        u0, v0, w,
        du0_dx, du0_dy,
        dv0_dx, dv0_dy,
        dw_dx, dw_dy,
        d2w_xx, d2w_xy, d2w_yy
    )

def ply_strain_at(x, y, z, q):
    """
    KL + von Karman strains at a point.
    """
    (
        u0, v0, w,
        du0_dx, du0_dy,
        dv0_dx, dv0_dy,
        dw_dx, dw_dy,
        d2w_xx, d2w_xy, d2w_yy
    ) = eval_fields(x, y, q)

    exx = du0_dx - z * d2w_xx + 0.5 * dw_dx**2
    eyy = dv0_dy - z * d2w_yy + 0.5 * dw_dy**2
    gxy = du0_dy + dv0_dx - 2.0 * z * d2w_xy + dw_dx * dw_dy

    return np.array([exx, eyy, gxy], dtype=float)

def thermal_strain_at(z, alpha_bar):
    dT_z = dT_0 + beta * z
    return alpha_bar * dT_z

def strain_energy_density(x, y, z, ply_idx, q):
    eps = ply_strain_at(x, y, z, q)
    eth = thermal_strain_at(z, alpha_bar_list[ply_idx])
    emech = eps - eth
    Q_bar = Q_bar_list[ply_idx]
    return 0.5 * float(emech @ Q_bar @ emech)

# ==============================================================================
# 5. RIGID-BODY MODE REMOVAL
# ==============================================================================
# The w shape functions already enforce the transverse encastre conditions at
# (0, 0): w = dw/dx = dw/dy = 0. The u/v bases omit constants, enforcing
# u(0,0) = v(0,0) = 0. The remaining zero-energy in-plane rotation is removed
# by requiring u_y(0,0) = v_x(0,0), which retains physical engineering shear
# while excluding u=-theta*y, v=theta*x.
# ============================================================================== 

def build_constraint_matrix(z_bc=BC_Z, constrain_all_corner_w=CONSTRAIN_ALL_CORNER_W):
    del z_bc, constrain_all_corner_w
    C = np.zeros((1, NQ), dtype=float)
    C[0, 1] = 1.0                 # coefficient of y in u
    C[0, NU + 0] = -1.0           # coefficient of x in v
    return C, ["remove in-plane rigid rotation"]

def nullspace_basis(C, rtol=1e-12):
    """
    Return an orthonormal basis for the nullspace of C using SVD.
    """
    U, S, Vt = np.linalg.svd(C, full_matrices=True)
    if S.size == 0:
        return np.eye(C.shape[1], dtype=float)

    smax = np.max(S)
    tol = rtol * smax if smax > 0.0 else rtol
    rank = int(np.sum(S > tol))
    N = Vt[rank:].T.copy()

    if N.size == 0:
        raise RuntimeError("Constraint matrix left no free degrees of freedom.")
    return N

def constraint_values(q, z_bc=BC_Z):
    del z_bc
    C, _ = build_constraint_matrix()
    return C @ q


def encastre_residuals(q, z=BC_Z):
    """Return translational and rotational residuals at the supported corner."""
    u, v, w = calc_displacements_from_q(0.0, 0.0, z, q)
    fields = eval_fields(0.0, 0.0, q)
    return np.array([u, v, w, fields[7], fields[8]], dtype=float)

# ==============================================================================
# 6. QUADRATURE
# ==============================================================================
def gauss_legendre_1d(n, a, b):
    xi, wi = np.polynomial.legendre.leggauss(n)
    x = 0.5 * (b - a) * xi + 0.5 * (a + b)
    w = 0.5 * (b - a) * wi
    return x, w

def build_quadrature():
    xg, wx = gauss_legendre_1d(AREA_QUAD_N, 0.0, L)
    yg, wy = gauss_legendre_1d(AREA_QUAD_N, 0.0, W)

    area_pts = []
    area_wts = []
    for i, x in enumerate(xg):
        for j, y in enumerate(yg):
            area_pts.append((x, y))
            area_wts.append(wx[i] * wy[j])

    zeta, wzeta = np.polynomial.legendre.leggauss(2)

    ply_quad = []
    for p in range(n_plies):
        z0 = z_coords[p]
        z1 = z_coords[p + 1]
        zz = 0.5 * (z1 - z0) * zeta + 0.5 * (z1 + z0)
        wz = 0.5 * (z1 - z0) * wzeta
        ply_quad.append((zz, wz))

    return area_pts, area_wts, ply_quad

AREA_PTS, AREA_WTS, PLY_QUAD = build_quadrature()

# ==============================================================================
# 7. TOTAL POTENTIAL ENERGY
# ==============================================================================
def total_internal_energy_q(q):
    total = 0.0
    for (x, y), w_xy in zip(AREA_PTS, AREA_WTS):
        for p in range(n_plies):
            zz, wz = PLY_QUAD[p]
            for z, wz_z in zip(zz, wz):
                total += strain_energy_density(x, y, z, p, q) * w_xy * wz_z
    return float(total)


def total_internal_energy_and_gradient_q(q):
    """Return internal energy and its exact Ritz-coordinate gradient."""
    total = 0.0
    gradient = np.zeros(NQ, dtype=float)

    for (x, y), w_xy in zip(AREA_PTS, AREA_WTS):
        xc, yc = center_xy(x, y)
        bu = basis_u(xc, yc)
        dux = dbasis_u_dx(xc, yc)
        duy = dbasis_u_dy(xc, yc)
        dvx = dbasis_v_dx(xc, yc)
        dvy = dbasis_v_dy(xc, yc)
        bw = basis_w(xc, yc)
        dwx = dbasis_w_dx(xc, yc)
        dwy = dbasis_w_dy(xc, yc)
        dxx = d2basis_w_xx(xc, yc)
        dxy = d2basis_w_xy(xc, yc)
        dyy = d2basis_w_yy(xc, yc)
        a, b, c = unpack(q)
        dw_x = float(c @ dwx)
        dw_y = float(c @ dwy)

        for p in range(n_plies):
            Q_bar = Q_bar_list[p]
            alpha_bar = alpha_bar_list[p]
            zz, wz = PLY_QUAD[p]
            for z, wz_z in zip(zz, wz):
                eps = np.array([
                    a @ dux - z * (c @ dxx) + 0.5 * dw_x**2,
                    b @ dvy - z * (c @ dyy) + 0.5 * dw_y**2,
                    a @ duy + b @ dvx - 2.0 * z * (c @ dxy) + dw_x * dw_y,
                ])
                emech = eps - alpha_bar * (dT_0 + beta * z)
                weight = w_xy * wz_z
                total += 0.5 * float(emech @ Q_bar @ emech) * weight

                d_eps = np.zeros((3, NQ), dtype=float)
                d_eps[0, 0:NU] = dux
                d_eps[1, NU:NU + NV] = dvy
                d_eps[2, 0:NU] = duy
                d_eps[2, NU:NU + NV] = dvx
                d_eps[0, NU + NV:] = dw_x * dwx - z * dxx
                d_eps[1, NU + NV:] = dw_y * dwy - z * dyy
                d_eps[2, NU + NV:] = dw_y * dwx + dw_x * dwy - 2.0 * z * dxy
                gradient += (d_eps.T @ (Q_bar @ emech)) * weight

    return float(total), gradient

# ==============================================================================
# 8. PROJECTED INITIAL GUESS
# ==============================================================================
def linear_clt_initial_guess():
    """
    Use the linear thermal ABD solution as a starting guess.
    """
    NM_T = np.concatenate([N_T, M_T])
    eps_kappa = np.linalg.solve(ABD, NM_T)

    eps_0 = eps_kappa[:3]
    kappa = eps_kappa[3:]

    q0 = np.zeros(NQ, dtype=float)

    # The in-plane linear terms represent eps_x, eps_y and engineering shear.
    q0[0] = eps_0[0]
    q0[1] = 0.5 * eps_0[2]

    # v basis
    q0[NU + 0] = 0.5 * eps_0[2]
    q0[NU + 1] = eps_0[1]

    # w = 1/2*(c1*x^2 + c2*y^2 + c3*x*y), so kappa = -[c1, c2, c3].
    q0[NU + NV + 0] = -kappa[0]
    q0[NU + NV + 1] = -kappa[1]
    q0[NU + NV + 2] = -kappa[2]

    return q0

# ==============================================================================
# 9. BFGS SOLVER IN THE CONSTRAINED SUBSPACE
# ==============================================================================
def projected_gradient(r, q_from_r, energy_r, h=FD_STEP):
    """Central finite-difference gradient in the constrained Ritz subspace."""
    g = np.zeros_like(r, dtype=float)
    for i in range(len(r)):
        step = h * max(1.0, abs(r[i]))
        dr = np.zeros_like(r)
        dr[i] = step
        fp = energy_r(r + dr)
        fm = energy_r(r - dr)
        g[i] = (fp - fm) / (2.0 * step)
    return g


def solve_equilibrium_bfgs_constrained(q_init, N, verbose=True):
    """
    Solve min U(q) subject to C q = 0 by writing q = N r,
    where columns of N span the nullspace of C.
    """
    r = N.T @ q_init
    I = np.eye(len(r), dtype=float)
    Hinv = I.copy()

    def q_from_r(rr):
        return N @ rr

    def energy_r(rr):
        return total_internal_energy_q(q_from_r(rr))

    def grad_r(rr):
        _, gradient_q = total_internal_energy_and_gradient_q(q_from_r(rr))
        return N.T @ gradient_q

    if minimize is not None:
        initial_gradient_norm = float(np.linalg.norm(grad_r(r)))
        result = minimize(
            energy_r,
            r,
            jac=grad_r,
            method="BFGS",
            options={
                "gtol": GRAD_TOL,
                "maxiter": MAX_ITERS,
                "disp": False,
            },
        )
        final_r = result.x
        final_g = grad_r(final_r)
        final_gradient_norm = float(np.linalg.norm(final_g))
        relative_gradient_norm = final_gradient_norm / max(initial_gradient_norm, 1.0)
        converged = bool(
            final_gradient_norm < GRAD_TOL
            or relative_gradient_norm < REL_GRAD_TOL
        )
        message = str(result.message)
        if converged and not result.success:
            message = (
                f"converged by relative projected-gradient criterion "
                f"({relative_gradient_norm:.3e} < {REL_GRAD_TOL:.1e}); "
                f"SciPy note: {message}"
            )
        return SolverResult(
            q=q_from_r(final_r),
            energy=float(result.fun),
            projected_gradient=final_g,
            converged=converged,
            iterations=int(result.nit),
            message=message,
            method="scipy-bfgs",
            initial_gradient_norm=initial_gradient_norm,
        )

    E = energy_r(r)
    g = grad_r(r)
    initial_gradient_norm = float(np.linalg.norm(g))

    if verbose:
        print("\nImplicit nonlinear solve with one-corner encastre")
        print("=" * 78)
        print(f"{'it':>3s} | {'Energy':>14s} | {'||grad||':>12s} | {'alpha':>10s}")
        print("-" * 78)

    for it in range(MAX_ITERS):
        gnorm = np.linalg.norm(g)

        if gnorm < GRAD_TOL:
            if verbose:
                print(f"{it:3d} | {E:14.6e} | {gnorm:12.4e} | {1.0:10.3e}")
            break

        p = -Hinv @ g

        if np.dot(g, p) >= 0.0:
            p = -g
            Hinv = I.copy()

        alpha = 1.0
        gTp = np.dot(g, p)
        while alpha >= LINESEARCH_MIN_ALPHA:
            r_trial = r + alpha * p
            E_trial = energy_r(r_trial)
            if np.isfinite(E_trial) and (E_trial <= E + LINESEARCH_C1 * alpha * gTp):
                break
            alpha *= LINESEARCH_SHRINK

        if alpha < LINESEARCH_MIN_ALPHA:
            if verbose:
                print("Line search failed; stopping.")
            break

        r_new = r + alpha * p
        E_new = energy_r(r_new)
        g_new = grad_r(r_new)

        s = r_new - r
        y = g_new - g
        ys = np.dot(y, s)

        if ys > 1e-14:
            rho = 1.0 / ys
            Hy = Hinv @ y
            Hinv = (
                Hinv
                - rho * np.outer(Hy, s)
                - rho * np.outer(s, Hy)
                + (rho * np.dot(y, Hy) + rho) * np.outer(s, s)
            )
        else:
            Hinv = I.copy()

        if verbose:
            print(f"{it:3d} | {E_new:14.6e} | {np.linalg.norm(g_new):12.4e} | {alpha:10.3e}")

        if np.linalg.norm(s) < STEP_TOL:
            r = r_new
            E = E_new
            g = g_new
            break

        r = r_new
        E = E_new
        g = g_new

    final_gradient_norm = float(np.linalg.norm(g))
    relative_gradient_norm = final_gradient_norm / max(initial_gradient_norm, 1.0)
    converged = bool(
        final_gradient_norm < GRAD_TOL
        or relative_gradient_norm < REL_GRAD_TOL
    )
    message = "converged" if converged else "line search or iteration limit reached"
    return SolverResult(
        q=q_from_r(r),
        energy=E,
        projected_gradient=g,
        converged=converged,
        iterations=it + 1,
        message=message,
        method="fallback-bfgs",
        initial_gradient_norm=initial_gradient_norm,
    )

# ==============================================================================
# 10. POSTPROCESSING
# ==============================================================================
def eval_generalized_quantities_at(x, y, q):
    (
        u0, v0, w,
        du0_dx, du0_dy,
        dv0_dx, dv0_dy,
        dw_dx, dw_dy,
        d2w_xx, d2w_xy, d2w_yy
    ) = eval_fields(x, y, q)

    eps_x0 = du0_dx + 0.5 * dw_dx**2
    eps_y0 = dv0_dy + 0.5 * dw_dy**2
    gamma_xy0 = du0_dy + dv0_dx + dw_dx * dw_dy

    kappa_x = -d2w_xx
    kappa_y = -d2w_yy
    kappa_xy = -2.0 * d2w_xy

    return np.array([eps_x0, eps_y0, gamma_xy0]), np.array([kappa_x, kappa_y, kappa_xy])


def eval_midplane_quantities(q):
    """Return generalized quantities at the geometric plate center."""
    return eval_generalized_quantities_at(L / 2.0, W / 2.0, q)


def area_averaged_generalized_quantities(q):
    """Return area averages in the same form as the Abaqus plate averages."""
    total_area = 0.0
    eps_sum = np.zeros(3, dtype=float)
    kappa_sum = np.zeros(3, dtype=float)

    for (x, y), weight in zip(AREA_PTS, AREA_WTS):
        eps, kappa = eval_generalized_quantities_at(x, y, q)
        eps_sum += weight * eps
        kappa_sum += weight * kappa
        total_area += weight

    return eps_sum / total_area, kappa_sum / total_area

def calc_displacements_from_q(x, y, z, q):
    (
        u0, v0, w,
        du0_dx, du0_dy,
        dv0_dx, dv0_dy,
        dw_dx, dw_dy,
        _, _, _
    ) = eval_fields(x, y, q)

    u = u0 - z * dw_dx
    v = v0 - z * dw_dy
    return np.array([u, v, w], dtype=float)

def find_max_displacements(q, num_xy_points=151):
    x_values = np.linspace(0.0, L, num_xy_points)
    y_values = np.linspace(0.0, W, num_xy_points)
    z_values = np.array([-h / 2.0, 0.0, h / 2.0])

    xx, yy, zz = np.meshgrid(x_values, y_values, z_values, indexing="ij")
    disp = np.array([
        calc_displacements_from_q(x, y, z, q)
        for x, y, z in zip(xx.ravel(), yy.ravel(), zz.ravel())
    ])
    locs = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

    mags = np.linalg.norm(disp, axis=1)
    idx = int(np.argmax(mags))

    results = {
        "magnitude": {
            "value": mags[idx],
            "components": disp[idx],
            "location": locs[idx],
        },
        "components": {}
    }

    for comp_idx, name in enumerate(("u", "v", "w")):
        j = int(np.argmax(np.abs(disp[:, comp_idx])))
        results["components"][name] = {
            "value": disp[j, comp_idx],
            "location": locs[j]
        }

    return results

def fmt_loc(loc):
    return f"(x={loc[0]:.6f}, y={loc[1]:.6f}, z={loc[2]:.6f})"


def plot_resultant_shape(
    q,
    output_path=PLOT_OUTPUT,
    grid_n=PLOT_GRID_N,
    true_scale=PLOT_TRUE_SCALE,
    amplified_scale=PLOT_AMPLIFIED_SCALE,
    show=PLOT_SHOW,
):
    """Plot true and amplified deformed midsurfaces and the u3 field."""
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.colors import Normalize
    except ImportError as exc:  # pragma: no cover - depends on local install
        raise RuntimeError("Visualization requires matplotlib.") from exc

    x_values = np.linspace(0.0, L, grid_n)
    y_values = np.linspace(0.0, W, grid_n)
    xx, yy = np.meshgrid(x_values, y_values, indexing="xy")
    flat_displacements = np.array([
        calc_displacements_from_q(x, y, 0.0, q)
        for x, y in zip(xx.ravel(), yy.ravel())
    ])
    uu = flat_displacements[:, 0].reshape(xx.shape)
    vv = flat_displacements[:, 1].reshape(xx.shape)
    ww = flat_displacements[:, 2].reshape(xx.shape)

    def deformed_coordinates(scale):
        return xx + scale * uu, yy + scale * vv, scale * ww

    xt, yt, zt = deformed_coordinates(true_scale)
    xa, ya, za = deformed_coordinates(amplified_scale)
    w_mm = ww * 1000.0
    color_norm = Normalize(vmin=float(np.min(w_mm)), vmax=float(np.max(w_mm)))

    fig = plt.figure(figsize=(18, 6), constrained_layout=True)
    ax_true = fig.add_subplot(1, 3, 1, projection="3d")
    ax_amp = fig.add_subplot(1, 3, 2, projection="3d")
    ax_contour = fig.add_subplot(1, 3, 3)

    surfaces = (
        (ax_true, xt, yt, zt, "True-scale resultant midsurface"),
        (ax_amp, xa, ya, za, f"Amplified resultant midsurface ({amplified_scale:g}x)"),
    )
    for ax, x_plot, y_plot, z_plot, title in surfaces:
        ax.plot_surface(
            x_plot,
            y_plot,
            z_plot,
            facecolors=cm.viridis(color_norm(w_mm)),
            linewidth=0.0,
            antialiased=True,
            shade=False,
        )
        ax.plot(
            [0.0, L, L, 0.0, 0.0],
            [0.0, 0.0, W, W, 0.0],
            [0.0] * 5,
            color="black",
            linewidth=0.8,
            alpha=0.65,
        )
        ax.scatter([0.0], [0.0], [0.0], color="red", s=35, label="encastre")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(title)
        ax.set_box_aspect((L, W, max(L, W)))
        ax.view_init(elev=28.0, azim=-125.0)

    contour = ax_contour.contourf(xx, yy, w_mm, levels=31, cmap="viridis")
    ax_contour.contour(xx, yy, w_mm, levels=11, colors="white", linewidths=0.35, alpha=0.55)
    ax_contour.scatter([0.0], [0.0], color="red", s=35, label="encastre")
    ax_contour.set_xlabel("x (m)")
    ax_contour.set_ylabel("y (m)")
    ax_contour.set_title("Transverse displacement $u_3=w$ (mm)")
    ax_contour.set_aspect("equal")
    ax_contour.legend(loc="best")
    fig.colorbar(contour, ax=ax_contour, label="w (mm)")
    fig.suptitle("Thermoelastic laminate resultant shape", fontsize=14)

    fig.savefig(output_path, dpi=220)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path

# ==============================================================================
# 11. RUN SOLVER
# ==============================================================================
def build_projected_initial_state():
    C, constraint_labels = build_constraint_matrix(z_bc=BC_Z)
    N = nullspace_basis(C)
    q0_unconstrained = linear_clt_initial_guess()
    q0 = N @ (N.T @ q0_unconstrained)
    return C, constraint_labels, N, q0_unconstrained, q0


def solve_default_equilibrium(verbose=False):
    C, constraint_labels, N, q0_unconstrained, q0 = build_projected_initial_state()
    result = solve_equilibrium_bfgs_constrained(q0, N, verbose=verbose)
    return {
        "constraint_matrix": C,
        "constraint_labels": constraint_labels,
        "nullspace": N,
        "initial_unconstrained": q0_unconstrained,
        "initial_projected": q0,
        "solver": result,
    }


def print_solution_report(solution):
    C = solution["constraint_matrix"]
    constraint_labels = solution["constraint_labels"]
    N = solution["nullspace"]
    q0_unconstrained = solution["initial_unconstrained"]
    q0 = solution["initial_projected"]
    solver = solution["solver"]
    q_sol = solver.q

    print("Top surface temp    :", T_top_surf)
    print("Bottom surface temp :", T_bot_surf)
    print("Midplane dT         :", dT_0)
    print("Through-thickness beta:", beta)

    print("\nConstraint labels:")
    for i, lab in enumerate(constraint_labels):
        print(f"  {i+1}: {lab}")

    print(f"\nConstraint matrix rank : {np.linalg.matrix_rank(C)}")
    print(f"Free subspace dimension: {N.shape[1]}")
    print(
        "Initial constraint residual norm (q0 projection not yet applied): "
        f"{np.linalg.norm(C @ q0_unconstrained):.6e}"
    )
    print(f"Initial constraint residual norm after projection: {np.linalg.norm(C @ q0):.6e}")

    eps0_center, kappa_center = eval_midplane_quantities(q_sol)
    eps0_avg, kappa_avg = area_averaged_generalized_quantities(q_sol)
    status = "CONVERGED" if solver.converged else "NOT CONVERGED"

    print("\n" + "=" * 78)
    print(f"{status} NONLINEAR EQUILIBRIUM SOLUTION")
    print("=" * 78)
    print(f"Solver method       : {solver.method}")
    print(f"Solver message      : {solver.message}")
    print(f"Iterations          : {solver.iterations}")
    print(f"Internal energy     : {solver.energy:.8e}")
    print(f"Final ||grad||      : {solver.gradient_norm:.8e}")
    print(f"Relative ||grad||   : {solver.relative_gradient_norm:.8e}")
    print(f"Rigid-mode constraint residual : {np.linalg.norm(C @ q_sol):.8e}")
    print(f"Encastre residual norm         : {np.linalg.norm(encastre_residuals(q_sol)):.8e}")
    print("")
    print("Midplane generalized strains at center:")
    print(f"  eps_x^0   = {eps0_center[0] * 1e6:12.3f} microstrain")
    print(f"  eps_y^0   = {eps0_center[1] * 1e6:12.3f} microstrain")
    print(f"  gamma_xy0 = {eps0_center[2] * 1e6:12.3f} microstrain")
    print("\nArea-averaged generalized quantities:")
    print(f"  eps_x^0   = {eps0_avg[0] * 1e6:12.3f} microstrain")
    print(f"  eps_y^0   = {eps0_avg[1] * 1e6:12.3f} microstrain")
    print(f"  gamma_xy0 = {eps0_avg[2] * 1e6:12.3f} microstrain")
    print("")
    print("Curvatures at center:")
    print(f"  kappa_x   = {kappa_center[0]:12.6e} 1/m")
    print(f"  kappa_y   = {kappa_center[1]:12.6e} 1/m")
    print(f"  kappa_xy  = {kappa_center[2]:12.6e} 1/m")
    print("Area-averaged curvatures:")
    print(f"  kappa_x   = {kappa_avg[0]:12.6e} 1/m")
    print(f"  kappa_y   = {kappa_avg[1]:12.6e} 1/m")
    print(f"  kappa_xy  = {kappa_avg[2]:12.6e} 1/m")

    print("\nSelected displacements [mm]:")
    pts = {
        "Center (L/2, W/2, z=0)": (L / 2, W / 2, 0.0),
        "Encastre Top (0, 0, +h/2)": (0.0, 0.0, h / 2),
        "Encastre Bot (0, 0, -h/2)": (0.0, 0.0, -h / 2),
        "Opposite Corner (L, W, z=0)": (L, W, 0.0),
        "Mid-Edge Top (L/2, W, +h/2)": (L / 2, W, h / 2),
    }
    for name, (x, y, z) in pts.items():
        u, v, w = calc_displacements_from_q(x, y, z, q_sol) * 1000.0
        print(f"  {name:28s} -> u: {u:9.4f} | v: {v:9.4f} | w: {w:9.4f}")

    max_report = find_max_displacements(q_sol, num_xy_points=101)
    max_mag = max_report["magnitude"]
    print("\nMax displacement over top/mid/bottom:")
    print(
        f"  Max |d| = {max_mag['value'] * 1000.0:.4f} mm at "
        f"{fmt_loc(max_mag['location'])}"
    )
    print(
        f"    components there -> u={max_mag['components'][0] * 1000.0:.4f} mm,"
        f" v={max_mag['components'][1] * 1000.0:.4f} mm,"
        f" w={max_mag['components'][2] * 1000.0:.4f} mm"
    )

    print("\nComponent extrema:")
    for comp, res in max_report["components"].items():
        print(
            f"  Max |{comp}| = {abs(res['value']) * 1000.0:.4f} mm at "
            f"{fmt_loc(res['location'])}"
        )


def run_analysis():
    solution = solve_default_equilibrium(verbose=True)
    print_solution_report(solution)
    if PLOT_RESULTS:
        plot_path = plot_resultant_shape(solution["solver"].q)
        print(f"\nSaved resultant-shape visualization: {plot_path}")
    return solution


if __name__ == "__main__":
    run_analysis()
