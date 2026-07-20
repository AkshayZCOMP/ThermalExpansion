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

    def test_inverse_solver_reports_failures_by_ply_count(self):
        ply_counts = [ 6, 8, 10]
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
