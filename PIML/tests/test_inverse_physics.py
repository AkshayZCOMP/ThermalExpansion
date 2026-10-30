"""Tests for the differentiable physics term used by inverse training."""

from __future__ import annotations

import torch

from PIML.core.physics import ANGLE_SET, FVKPhysics, make_batch
from PIML.scripts.train_inverse_nn import soft_linear_clt_curvatures


def test_soft_linear_clt_matches_discrete_clt_for_one_hot_layups() -> None:
    layups = [[0, 45, -45, 90, 0, 45, -45, 90], [90, 0, 0, 90, 45, -45, 45, -45]]
    top = torch.tensor([65.0, 30.0], dtype=torch.float64)
    bottom = torch.tensor([35.0, 70.0], dtype=torch.float64)
    physics = FVKPhysics()
    indices = torch.tensor([[ANGLE_SET.index(angle) for angle in layup] for layup in layups])
    probabilities = torch.nn.functional.one_hot(indices, num_classes=len(ANGLE_SET)).double().requires_grad_()
    raw = torch.cat((
        torch.zeros(2, 3, dtype=torch.float64), top[:, None] / 100.0, bottom[:, None] / 100.0,
    ), dim=1)
    predicted = soft_linear_clt_curvatures(probabilities, raw, physics)
    batch = make_batch(layups, top, bottom)
    expected = physics.curvatures(physics.linear_initial_q(batch))
    assert torch.allclose(predicted, expected, rtol=1.0e-11, atol=1.0e-11)
    predicted.square().sum().backward()
    assert probabilities.grad is not None
    assert torch.isfinite(probabilities.grad).all()
