import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_FILE = ROOT / "RealisticConfirmatoin_v2.py"

sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("realistic_confirmation_v2", SCRIPT_FILE)
rc2 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rc2
spec.loader.exec_module(rc2)


class TestRealisticConfirmationV2(unittest.TestCase):
    def test_uses_ritz_shape_functions_with_expected_sizes(self):
        x = 0.01
        y = -0.02

        self.assertEqual(rc2.basis_u(x, y).shape, (9,))
        self.assertEqual(rc2.basis_v(x, y).shape, (9,))
        self.assertEqual(rc2.basis_w(x, y).shape, (3,))
        self.assertEqual(rc2.NQ, rc2.NU + rc2.NV + rc2.NW)

    def test_encastre_corner_is_built_into_shape_functions(self):
        q = np.zeros(rc2.NQ)

        np.testing.assert_allclose(rc2.basis_u(0.0, 0.0), np.zeros(rc2.NU))
        np.testing.assert_allclose(rc2.basis_v(0.0, 0.0), np.zeros(rc2.NV))
        np.testing.assert_allclose(rc2.basis_w(0.0, 0.0), np.zeros(rc2.NW))
        np.testing.assert_allclose(rc2.dbasis_w_dx(0.0, 0.0), np.zeros(rc2.NW))
        np.testing.assert_allclose(rc2.dbasis_w_dy(0.0, 0.0), np.zeros(rc2.NW))
        np.testing.assert_allclose(rc2.encastre_residuals(q), np.zeros(5))

    def test_shape_constraint_removes_in_plane_rigid_rotation(self):
        C, labels = rc2.build_constraint_matrix()
        self.assertEqual(labels, ["remove in-plane rigid rotation"])
        self.assertEqual(np.linalg.matrix_rank(C), 1)

        rigid_rotation = np.zeros(rc2.NQ)
        rigid_rotation[1] = -1.0       # u = -y
        rigid_rotation[rc2.NU] = 1.0   # v = x
        self.assertGreater(abs(float((C @ rigid_rotation)[0])), 1e-12)

    def test_plus_minus_45_plies_have_opposite_shear_coupling(self):
        q_plus, alpha_plus = rc2.transformed_ply_matrices(45)
        q_minus, alpha_minus = rc2.transformed_ply_matrices(-45)

        self.assertAlmostEqual(alpha_plus[0], alpha_minus[0], places=14)
        self.assertAlmostEqual(alpha_plus[1], alpha_minus[1], places=14)
        self.assertAlmostEqual(alpha_plus[2], -alpha_minus[2], places=14)

        np.testing.assert_allclose(q_plus[:2, :2], q_minus[:2, :2], rtol=1e-12)
        np.testing.assert_allclose(q_plus[2, :2], -q_minus[2, :2], rtol=1e-12)
        np.testing.assert_allclose(q_plus[:2, 2], -q_minus[:2, 2], rtol=1e-12)
        self.assertAlmostEqual(q_plus[2, 2], q_minus[2, 2], places=6)

    def test_constrained_energy_minimization_converges_and_satisfies_bcs(self):
        C, _, N, q0_unconstrained, q0 = rc2.build_projected_initial_state()
        initial_energy = rc2.total_internal_energy_q(q0)
        solution = rc2.solve_default_equilibrium(verbose=False)
        solver = solution["solver"]

        self.assertLess(np.linalg.norm(C @ q0), 1e-12)
        self.assertTrue(solver.converged, solver.message)
        self.assertLess(solver.relative_gradient_norm, rc2.REL_GRAD_TOL)
        self.assertLess(solver.energy, initial_energy)
        self.assertLess(np.linalg.norm(C @ solver.q), 1e-12)

        reduced_step = N.T @ (solver.q - q0)
        self.assertGreater(np.linalg.norm(reduced_step), 1e-8)

    def test_free_plate_bending_is_not_pinned_at_other_corners(self):
        solution = rc2.solve_default_equilibrium(verbose=False)
        q = solution["solver"].q

        origin = rc2.calc_displacements_from_q(0.0, 0.0, 0.0, q)
        opposite = rc2.calc_displacements_from_q(rc2.L, rc2.W, 0.0, q)

        np.testing.assert_allclose(origin, np.zeros(3), atol=1e-12)
        self.assertGreater(abs(opposite[2]), 1e-4)

    def test_maximum_displacement_is_not_forced_to_plate_center(self):
        solution = rc2.solve_default_equilibrium(verbose=False)
        q = solution["solver"].q
        max_report = rc2.find_max_displacements(q, num_xy_points=51)
        max_w = max_report["components"]["w"]

        self.assertGreater(abs(max_w["value"]), 1e-4)
        self.assertGreater(np.linalg.norm(max_w["location"][:2]), 1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
