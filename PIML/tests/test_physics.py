"""Small, solver-independent checks for the PIML mechanics layer."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch

from PIML.core.losses import canonicalize_midplane_curvatures
from PIML.core.model import RitzPINN
from PIML.core.physics import (
    FVKPhysics,
    L,
    SIMPLE_CASE_THERMAL_GRADIENT,
    T_PLY,
    all_sweep_layups,
    fixed_gradient_cases,
    gradient_grid_cases,
    temperature_field_grid_cases,
    sample_cases,
    make_batch,
)
from PIML.core.reporting import summarize_errors


class TestPIMLPhysics(unittest.TestCase):
    def setUp(self) -> None:
        self.physics = FVKPhysics(transverse_degree=3, area_quad_n=4)

    def test_uses_the_canonical_sweep_space(self) -> None:
        layups = all_sweep_layups()
        self.assertGreater(len(layups), 0)
        self.assertEqual(len(layups), len(set(layups)))
        self.assertEqual(min(map(len, layups)), 3)
        self.assertEqual(max(map(len, layups)), 10)
        self.assertIn((0, 0, 0), layups)
        self.assertIn((45, 45, 45), layups)

    def test_fixed_gradient_8ply_case_is_exactly_defined(self) -> None:
        batch = fixed_gradient_cases()
        self.assertEqual(len(batch), 72)
        self.assertTrue(torch.all(batch.nplies == 8))
        difference = batch.top_face_temperature - batch.bottom_face_temperature
        self.assertTrue(torch.all(difference != 0.0))
        recovered_gradient = difference / (8 * T_PLY)
        expected = torch.full_like(recovered_gradient, SIMPLE_CASE_THERMAL_GRADIENT)
        self.assertTrue(torch.allclose(recovered_gradient, expected))

    def test_gradient_grid_pairs_the_same_layups_with_each_nonzero_field(self) -> None:
        gradients = (-80000.0, 40000.0)
        batch = gradient_grid_cases(gradients, count_per_gradient=3, seed=19)
        self.assertEqual(len(batch), 6)
        self.assertTrue(torch.all(batch.nplies == 8))
        recovered = (batch.top_face_temperature - batch.bottom_face_temperature) / (8 * T_PLY)
        expected = torch.tensor([-80000.0] * 3 + [40000.0] * 3, dtype=torch.float64)
        self.assertTrue(torch.allclose(recovered, expected))
        self.assertTrue(torch.all(batch.top_face_temperature != batch.bottom_face_temperature))
        self.assertTrue(torch.equal(batch.angle_indices[:3], batch.angle_indices[3:]))

    def test_temperature_field_grid_varies_midplane_and_gradient(self) -> None:
        fields = ((25.0, -80000.0), (75.0, 80000.0))
        batch = temperature_field_grid_cases(fields, count_per_field=2, seed=29)
        self.assertEqual(len(batch), 4)
        self.assertTrue(torch.equal(batch.angle_indices[:2], batch.angle_indices[2:]))
        midpoint = 0.5 * (batch.top_face_temperature + batch.bottom_face_temperature)
        recovered_gradient = (batch.top_face_temperature - batch.bottom_face_temperature) / (8 * T_PLY)
        self.assertTrue(torch.allclose(midpoint, torch.tensor([25.0, 25.0, 75.0, 75.0], dtype=torch.float64)))
        self.assertTrue(torch.allclose(recovered_gradient, torch.tensor([-80000.0, -80000.0, 80000.0, 80000.0], dtype=torch.float64)))
        self.assertTrue(torch.all(batch.top_face_temperature != batch.bottom_face_temperature))

    def test_random_temperature_cases_never_have_equal_faces(self) -> None:
        batch = sample_cases(64, seed=23, minimum_face_temperature_difference=2.0)
        self.assertTrue(torch.all((batch.top_face_temperature - batch.bottom_face_temperature).abs() >= 2.0))

    def test_linear_profile_reaches_both_faces(self) -> None:
        batch = make_batch([[0, 90, 0]], [10.0], [70.0])
        h = 3 * T_PLY
        z = torch.tensor([-h / 2.0, h / 2.0], dtype=torch.float64)
        temperature = self.physics.temperature_at(z, batch)
        self.assertTrue(torch.allclose(temperature[0], torch.tensor([70.0, 10.0], dtype=torch.float64)))

    def test_linear_initial_state_satisfies_abd_equilibrium(self) -> None:
        batch = make_batch([[0, 45, -45, 90]], [0.0], [100.0])
        abd, thermal = self.physics.laminate_matrices(batch)
        generalized = torch.linalg.solve(abd, thermal.unsqueeze(-1)).squeeze(-1)
        residual = abd @ generalized.unsqueeze(-1) - thermal.unsqueeze(-1)
        self.assertLess(float(residual.abs().max()), 1e-9)

        q = self.physics.linear_initial_q(batch)
        self.assertLess(float((self.physics.constraint @ q.T).abs().max().detach()), 1e-13)
        averaged = self.physics.generalized_average(q)
        self.assertTrue(torch.allclose(averaged[:, 3:], generalized[:, 3:], atol=1e-11, rtol=1e-11))

    def test_network_preserves_the_hard_gauge(self) -> None:
        batch = make_batch([[0, 45, -45, 90], [90, 90, 90]], [0.0, 20.0], [100.0, 60.0])
        model = RitzPINN(self.physics, hidden_width=16, hidden_layers=2)
        q = model(batch)
        self.assertLess(float((self.physics.constraint @ q.T).abs().max().detach()), 1e-13)
        self.assertEqual(tuple(model.predict_curvatures(batch).shape), (2, 3))

    def test_axis_aligned_layups_are_hard_projected_to_zero_twist(self) -> None:
        axis_batch = make_batch([[0, 90, 90, 90, 90, 90, 90, 0]], [115.0], [35.0])
        angle_batch = make_batch([[0, 45, -45, 90, 90, -45, 45, 0]], [115.0], [35.0])
        self.assertTrue(bool(self.physics.axis_aligned_mask(axis_batch)[0]))
        self.assertFalse(bool(self.physics.axis_aligned_mask(angle_batch)[0]))
        model = RitzPINN(self.physics, hidden_width=16, hidden_layers=2)
        with torch.no_grad():
            final_layer = model.network[-1]
            assert isinstance(final_layer, torch.nn.Linear)
            final_layer.bias.fill_(0.5)
        q = model(axis_batch)
        self.assertLess(float((self.physics.constraint @ q.T).abs().max().detach()), 1e-13)
        self.assertLess(abs(float(model.predict_curvatures(axis_batch)[0, 2].detach())), 1e-12)

    def test_zero_initialized_gradient_expert_preserves_base_prediction(self) -> None:
        batch = make_batch([[0, 45, -45, 90]], [20.0], [80.0])
        base = RitzPINN(self.physics, hidden_width=16, hidden_layers=2)
        expert = RitzPINN(
            self.physics,
            hidden_width=16,
            hidden_layers=2,
            gradient_residual_expert=True,
            gradient_expert_width=8,
        )
        expert.network.load_state_dict(base.network.state_dict())
        self.assertTrue(torch.allclose(expert.predict_reduced_correction(batch), base.predict_reduced_correction(batch)))
        self.assertLess(float((self.physics.constraint @ expert(batch).T).abs().max().detach()), 1e-13)

    def test_midplane_difference_temperature_features_are_explicit(self) -> None:
        batch = make_batch([[0, 45, -45, 90]], [20.0], [80.0])
        model = RitzPINN(
            self.physics,
            hidden_width=16,
            hidden_layers=2,
            temperature_scale=100.0,
            temperature_feature_mode="midplane_difference",
        )
        temperature_features = model.features(batch)[0, -2:]
        self.assertTrue(torch.allclose(temperature_features, torch.tensor([0.5, -0.6], dtype=torch.float64)))

    def test_canonical_midplane_curvature_targets_reject_an_isolated_branch(self) -> None:
        fields = ((25.0, -40000.0), (50.0, -40000.0), (75.0, -40000.0))
        targets = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0],
             [1.1, 2.1, 3.1], [4.1, 5.1, 6.1],
             [100.0, 200.0, 300.0], [400.0, 500.0, 600.0]],
            dtype=torch.float64,
        )
        canonical = canonicalize_midplane_curvatures(targets, fields)
        expected = torch.tensor(
            [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]] * 3,
            dtype=torch.float64,
        )
        self.assertTrue(torch.equal(canonical, expected))

    def test_behavior_metrics_mask_only_near_zero_reference_components(self) -> None:
        rows = [
            {
                "reference_kappa_x": 0.0, "pinn_kappa_x": 0.1, "linear_clt_kappa_x": 0.2,
                "reference_kappa_y": 0.1, "pinn_kappa_y": 0.11, "linear_clt_kappa_y": 0.15,
                "reference_kappa_xy": -0.1, "pinn_kappa_xy": -0.11, "linear_clt_kappa_xy": -0.15,
            },
            {
                "reference_kappa_x": 0.1, "pinn_kappa_x": 0.11, "linear_clt_kappa_x": 0.15,
                "reference_kappa_y": 0.0, "pinn_kappa_y": 0.1, "linear_clt_kappa_y": 0.2,
                "reference_kappa_xy": 0.0, "pinn_kappa_xy": 0.1, "linear_clt_kappa_xy": 0.2,
            },
        ]
        summary = summarize_errors(rows, near_zero_curvature_threshold=0.02)
        self.assertEqual(summary["components"]["kappa_x"]["pinn"]["active_cases"], 1)
        self.assertEqual(summary["components"]["kappa_y"]["pinn"]["excluded_near_zero_cases"], 1)
        self.assertEqual(summary["behavior_focused"]["active_component_values"], 3)
        self.assertEqual(summary["behavior_focused"]["excluded_near_zero_component_values"], 3)
        self.assertAlmostEqual(summary["components"]["kappa_x"]["pinn"]["active_rmse_1_per_m"], 0.01)

    def test_energy_difference_matches_freefree_integral(self) -> None:
        """The differentiable energy must have the same q-dependent part as FREEFREE."""
        freefree_path = Path(__file__).resolve().parents[2] / "Computational Real" / "FREEFREE.py"
        spec = importlib.util.spec_from_file_location("piml_freefree_energy_test", freefree_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        freefree = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(freefree)
        layup = [90] * 8
        freefree.configure_case(layup, 0.0, 0.0, transverse_degree=5)

        physics = FVKPhysics(transverse_degree=5, area_quad_n=9)
        batch = make_batch([layup], [freefree.T_top_surf], [freefree.T_bot_surf])
        q0 = physics.linear_initial_q(batch)
        q = q0.clone()
        q[0, -1] += 2.0e-5
        piml_delta = float((physics.energy(q, batch) - physics.energy(q0, batch))[0])
        freefree_delta = freefree.total_internal_energy_q(q[0].numpy()) - freefree.total_internal_energy_q(q0[0].numpy())
        self.assertAlmostEqual(piml_delta, freefree_delta, places=8)


if __name__ == "__main__":
    unittest.main()
