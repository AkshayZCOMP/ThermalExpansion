"""Physics and branch losses used by PIML training commands."""

from __future__ import annotations

import torch

from .model import RitzPINN
from .physics import FVKPhysics, LayupBatch


ENERGY_SCALE = 0.1  # J; characteristic laminate energy for stable optimization
CURVATURE_SCALE = 1.0  # 1/m; used only for optional curvature calibration


def energy_difference_loss(model: RitzPINN, physics: FVKPhysics, batch: LayupBatch) -> torch.Tensor:
    """Deep-Ritz energy relative to the exact linear-CLT initial state."""
    q = model(batch)
    return (physics.energy(q, batch) - physics.energy(physics.linear_initial_q(batch), batch)).mean() / ENERGY_SCALE


def equilibrium_residual_loss(model: RitzPINN, physics: FVKPhysics, batch: LayupBatch) -> torch.Tensor:
    """Case-normalized weak FvK equilibrium residual loss."""
    q_linear = physics.linear_initial_q(batch).detach().requires_grad_(True)
    with torch.enable_grad():
        linear_residual = physics.reduced_gradient(q_linear, batch)
    scale = torch.linalg.vector_norm(linear_residual, dim=1).detach().clamp_min(1.0)
    residual = physics.reduced_gradient(model(batch), batch, create_graph=True)
    return torch.mean(torch.sum((residual / scale[:, None]).square(), dim=1))


def directional_stability_loss(
    model: RitzPINN,
    physics: FVKPhysics,
    batch: LayupBatch,
    *,
    directions: int,
    margin: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Penalize negative normalized second variations of FvK energy.

    A vanishing first variation alone permits saddles.  This inexpensive
    stochastic check samples directions in the gauge-valid reduced Ritz
    space.  Each curvature is divided by the magnitude at the linear CLT
    state for the same direction, making the margin comparable across layups,
    thicknesses, and thermal fields.  Positive curvature is a necessary
    local-stability condition; increasing ``directions`` makes its sampled
    approximation stricter.
    """
    if directions < 1:
        raise ValueError("directions must be positive.")
    if margin < 0.0:
        raise ValueError("margin cannot be negative.")

    q = model(batch)
    random_reduced = torch.randn(
        (directions, len(batch), physics.reduced_dofs),
        device=q.device,
        dtype=q.dtype,
        generator=generator,
    )
    random_reduced = random_reduced / torch.linalg.vector_norm(random_reduced, dim=-1, keepdim=True).clamp_min(1.0e-12)
    direction_q = torch.einsum(
        "nr,dbr,n->dbn",
        physics.nullspace_dimensionless,
        random_reduced,
        physics.coordinate_scale,
    )

    def second_variations(state: torch.Tensor, *, create_graph: bool) -> torch.Tensor:
        gradient = torch.autograd.grad(
            physics.energy(state, batch).sum(), state, create_graph=True,
        )[0]
        values: list[torch.Tensor] = []
        for direction in direction_q:
            directional_gradient = torch.sum(gradient * direction, dim=1)
            hessian_direction = torch.autograd.grad(
                directional_gradient.sum(), state, create_graph=create_graph,
                retain_graph=True,
            )[0]
            values.append(torch.sum(hessian_direction * direction, dim=1))
        return torch.stack(values, dim=0)

    predicted = second_variations(q, create_graph=True)
    linear_q = physics.linear_initial_q(batch).detach().requires_grad_(True)
    linear = second_variations(linear_q, create_graph=False).detach()
    normalized = predicted / linear.abs().clamp_min(1.0e-12)
    return torch.mean(torch.relu(margin - normalized).square())


def validation_metrics(model: RitzPINN, physics: FVKPhysics, batch: LayupBatch) -> tuple[float, float]:
    """Return mean energy difference and projected stationarity residual."""
    with torch.enable_grad():
        q = model(batch)
        energy_delta = (physics.energy(q, batch) - physics.energy(physics.linear_initial_q(batch), batch)).mean()
        residual = torch.linalg.vector_norm(physics.reduced_gradient(q, batch), dim=1).mean()
    return float(energy_delta.detach().cpu()), float(residual.detach().cpu())


def target_curvatures(physics: FVKPhysics, cases: LayupBatch, target_reduced: torch.Tensor) -> torch.Tensor:
    """Calculate curvatures of physics-derived local energy targets."""
    correction_q = torch.einsum(
        "nr,br,n->bn",
        physics.nullspace_dimensionless,
        target_reduced,
        physics.coordinate_scale,
    )
    return physics.curvatures(physics.linear_initial_q(cases) + correction_q).detach()


def canonicalize_midplane_curvatures(
    curvatures: torch.Tensor,
    temperature_fields: tuple[tuple[float, float], ...],
) -> torch.Tensor:
    """Reject isolated local-branch switches across matched midplane fields."""
    if len(curvatures) % len(temperature_fields) != 0:
        raise ValueError("Curvature target count is not divisible by the number of temperature fields.")
    cases_per_field = len(curvatures) // len(temperature_fields)
    field_indices_by_gradient: dict[float, list[int]] = {}
    for field_index, (_, gradient) in enumerate(temperature_fields):
        field_indices_by_gradient.setdefault(gradient, []).append(field_index)
    canonical = curvatures.clone()
    for field_indices in field_indices_by_gradient.values():
        if len(field_indices) < 2:
            continue
        stack = torch.stack(
            [curvatures[index * cases_per_field : (index + 1) * cases_per_field] for index in field_indices],
            dim=0,
        )
        median = stack.median(dim=0).values
        for index in field_indices:
            canonical[index * cases_per_field : (index + 1) * cases_per_field] = median
    return canonical


def branch_loss(
    model: RitzPINN,
    cases: LayupBatch,
    target_reduced: torch.Tensor,
    target_curvature: torch.Tensor,
    curvature_loss_weight: float,
    curvature_activity_threshold: float = 0.0,
    curvature_component_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Coordinate and behavior-focused curvature loss against a local branch.

    ``curvature_component_weights`` is ordered ``(kappa_x, kappa_y,
    kappa_xy)``.  It makes it possible to emphasize twist curvature without
    changing the target source or allowing near-zero reference components to
    dominate the auxiliary fit.
    """
    if curvature_activity_threshold < 0.0:
        raise ValueError("curvature_activity_threshold cannot be negative.")
    if len(curvature_component_weights) != 3 or any(weight <= 0.0 for weight in curvature_component_weights):
        raise ValueError("curvature_component_weights must contain three positive values.")
    coordinate_loss = torch.mean((model.predict_reduced_correction(cases) - target_reduced).square())
    component_weights = torch.tensor(
        curvature_component_weights,
        dtype=target_curvature.dtype,
        device=target_curvature.device,
    )
    curvature_squared_error = (model.predict_curvatures(cases) - target_curvature).square() * component_weights
    if curvature_activity_threshold == 0.0:
        curvature_loss = torch.mean(curvature_squared_error)
    else:
        active = target_curvature.abs() >= curvature_activity_threshold
        if not bool(torch.any(active)):
            raise ValueError("No curvature targets exceed curvature_activity_threshold.")
        curvature_loss = torch.mean(curvature_squared_error[active])
    return coordinate_loss + curvature_loss_weight * curvature_loss, coordinate_loss, curvature_loss
