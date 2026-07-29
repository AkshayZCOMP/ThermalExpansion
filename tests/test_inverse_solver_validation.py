"""
Validation report for the inverse layup solver.

This is intentionally a reporting test: it runs many synthetic inverse cases
and prints how many exact layup recoveries fail for each ply count.
"""

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_FILE = ROOT / "examples" / "InverseSolving.py"

sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("inverse_solving_example", EXAMPLE_FILE)
inverse_solving = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inverse_solving)


class TestInverseSolverValidationReport(unittest.TestCase):
    """Run randomized synthetic inverse checks and report failure counts."""

    def test_inverse_solver_candidate_pool_excludes_negative_ninety(self):
        self.assertNotIn(-1 * 90, inverse_solving.CANDIDATE_ANGLES)
        self.assertEqual(min(inverse_solving.CANDIDATE_ANGLES), -75)

    def test_solution_found_uses_inverse_success_tolerance(self):
        self.assertTrue(
            inverse_solving.solution_found(
                {"error": inverse_solving.INVERSE_SUCCESS_TOLERANCE}
            )
        )
        self.assertFalse(
            inverse_solving.solution_found(
                {"error": inverse_solving.INVERSE_SUCCESS_TOLERANCE * 10}
            )
        )

    def test_response_scale_uses_strain_and_curvature_floors(self):
        response = inverse_solving.np.zeros(6)
        scale = inverse_solving.normalize_response(response)
        inverse_solving.np.testing.assert_array_equal(
            scale[:3],
            inverse_solving.np.ones(3) * inverse_solving.STRAIN_SCALE_FLOOR,
        )
        inverse_solving.np.testing.assert_array_equal(
            scale[3:6],
            inverse_solving.np.ones(3) * inverse_solving.CURVATURE_SCALE_FLOOR,
        )

    def test_coordinate_descent_method_is_supported(self):
        original_method = inverse_solving.SOLVER_METHOD
        try:
            inverse_solving.SOLVER_METHOD = "coordinate"
            self.assertEqual(inverse_solving.solver_method(), "coordinate")
            self.assertGreater(inverse_solving.INVERSE_RANDOM_STARTS, 0)

            inverse_solving.SOLVER_METHOD = "coordinate_descent"
            self.assertEqual(inverse_solving.solver_method(), "coordinate_descent")
        finally:
            inverse_solving.SOLVER_METHOD = original_method

    def test_multi_start_coordinate_descent_method_is_supported(self):
        original_method = inverse_solving.SOLVER_METHOD
        rng = inverse_solving.np.random.default_rng(7)
        try:
            inverse_solving.SOLVER_METHOD = "multi_start_coordinate"
            self.assertEqual(
                inverse_solving.solver_method(),
                "multi_start_coordinate",
            )
            starts = inverse_solving.coordinate_descent_starts(
                num_plies=4,
                num_starts=inverse_solving.INVERSE_RANDOM_STARTS,
                rng=rng,
            )
            self.assertEqual(len(starts), inverse_solving.INVERSE_RANDOM_STARTS)

            inverse_solving.SOLVER_METHOD = "multi_start_coordinate_descent"
            self.assertEqual(
                inverse_solving.solver_method(),
                "multi_start_coordinate_descent",
            )
        finally:
            inverse_solving.SOLVER_METHOD = original_method

    def test_iterated_local_search_method_is_supported(self):
        original_method = inverse_solving.SOLVER_METHOD
        try:
            inverse_solving.SOLVER_METHOD = "iterated_local_search"
            self.assertEqual(
                inverse_solving.solver_method(),
                "iterated_local_search",
            )
            self.assertGreater(inverse_solving.ILS_RESTARTS, 0)
            self.assertGreater(inverse_solving.ILS_PERTURBATIONS_PER_RESTART, 0)
            self.assertGreater(inverse_solving.ILS_PERTURBATION_SIZE, 0)

            inverse_solving.SOLVER_METHOD = "ils"
            self.assertEqual(inverse_solving.solver_method(), "ils")
        finally:
            inverse_solving.SOLVER_METHOD = original_method

    def test_iterated_local_search_perturbation_preserves_symmetry(self):
        rng = inverse_solving.np.random.default_rng(13)
        perturbed = inverse_solving.perturb_layup([0, 45, 45, 0], 2, rng)
        self.assertEqual(perturbed, perturbed[::-1])

    def test_inverse_solver_reports_failures_by_ply_count(self):
        ply_counts = [ 3, 6, 8, 10]
        cases_per_ply_count = 60

        report = inverse_solving.validate_inverse_solver(
            ply_counts=ply_counts,
            cases_per_ply_count=cases_per_ply_count,
            random_seed=inverse_solving.VALIDATION_RANDOM_SEED,
        )

        self.assertEqual(len(report), len(ply_counts))
        for row, ply_count in zip(report, ply_counts):
            self.assertEqual(row["num_plies"], ply_count)
            self.assertEqual(row["cases"], cases_per_ply_count)
            self.assertEqual(
                row["exact_passes"]
                + row["equivalent_matches"]
                + row["response_failures"],
                cases_per_ply_count,
            )
            self.assertEqual(
                row["response_passes"] + row["response_failures"],
                cases_per_ply_count,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
