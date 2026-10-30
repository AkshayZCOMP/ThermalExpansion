"""Local, physics-only FvK energy solves used to define an equilibrium branch.

The nonlinear thermal potential can have more than one stationary shape.  This
module always starts at the linear-CLT state and uses a local L-BFGS solve, so
the branch definition is reproducible without using FREEFREE response data.
It is intended for one-time surrogate training-target generation, not online
inference.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .physics import FVKPhysics, LayupBatch


@dataclass(frozen=True)
class LocalEnergyTargets:
    """Physics-derived local equilibria in the model's reduced coordinates."""

    reduced_coordinates: torch.Tensor
    energy_change_from_clt: torch.Tensor
    projected_gradient_norm: torch.Tensor
    closure_evaluations: torch.Tensor


def q_from_reduced(
    physics: FVKPhysics,
    linear_state: torch.Tensor,
    reduced_coordinates: torch.Tensor,
) -> torch.Tensor:
    """Embed a scaled, gauge-compatible correction in the full Ritz vector."""
    correction = torch.einsum(
        "nr,br,n->bn",
        physics.nullspace_dimensionless,
        reduced_coordinates,
        physics.coordinate_scale,
    )
    return linear_state + correction


def solve_local_energy_targets(
    physics: FVKPhysics,
    batch: LayupBatch,
    *,
    max_iterations: int = 600,
    learning_rate: float = 0.5,
    branch_method: str = "freefree_bfgs",
    enforce_axis_aligned_reflection_symmetry: bool = True,
    workers: int = 1,
    progress_callback: Callable[[int, int, int, torch.Tensor, torch.Tensor, int], None] | None = None,
) -> LocalEnergyTargets:
    """Minimize each case's FvK energy from its linear-CLT initialization.

    Cases are solved separately.  That prevents one laminate's line search
    from changing another laminate's path through a multi-well energy
    landscape, and makes the generated local branch deterministic for a given
    setup.  ``freefree_bfgs`` uses the same *unscaled Ritz coordinate metric*
    and BFGS branch-selection strategy as the existing FREEFREE solver, but
    evaluates this differentiable physics module directly and never reads a
    FREEFREE result.
    """
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if branch_method not in {"freefree_bfgs", "scaled_lbfgs"}:
        raise ValueError("branch_method must be 'freefree_bfgs' or 'scaled_lbfgs'.")
    if workers <= 0:
        raise ValueError("workers must be positive.")

    def solve_one(case_index: int) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        index = torch.tensor([case_index], device=batch.nplies.device)
        case = batch.index_select(index)
        reduced, energy_delta, gradient_norm, evaluations = _solve_local_case(
            physics, case, max_iterations=max_iterations, learning_rate=learning_rate,
            branch_method=branch_method,
            enforce_axis_aligned_reflection_symmetry=enforce_axis_aligned_reflection_symmetry,
        )
        return case_index, reduced, energy_delta, gradient_norm, evaluations

    results: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None] = [None] * len(batch)
    if workers == 1:
        iterator = (solve_one(index) for index in range(len(batch)))
        for completed, (case_index, reduced, energy_delta, gradient_norm, evaluations) in enumerate(iterator, start=1):
            results[case_index] = (reduced, energy_delta, gradient_norm, evaluations)
            if progress_callback is not None:
                progress_callback(completed, len(batch), case_index, energy_delta, gradient_norm, evaluations)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(solve_one, index) for index in range(len(batch))]
            for completed, future in enumerate(as_completed(futures), start=1):
                case_index, reduced, energy_delta, gradient_norm, evaluations = future.result()
                results[case_index] = (reduced, energy_delta, gradient_norm, evaluations)
                if progress_callback is not None:
                    progress_callback(completed, len(batch), case_index, energy_delta, gradient_norm, evaluations)

    resolved = [result for result in results if result is not None]

    return LocalEnergyTargets(
        reduced_coordinates=torch.stack([result[0] for result in resolved]),
        energy_change_from_clt=torch.stack([result[1] for result in resolved]),
        projected_gradient_norm=torch.stack([result[2] for result in resolved]),
        closure_evaluations=torch.tensor([result[3] for result in resolved], dtype=torch.int64),
    )


def _solve_local_case(
    physics: FVKPhysics,
    case: LayupBatch,
    *,
    max_iterations: int,
    learning_rate: float,
    branch_method: str,
    enforce_axis_aligned_reflection_symmetry: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Solve one independent local branch and return CPU-safe target values."""
    linear_state = physics.linear_initial_q(case).detach()
    if branch_method == "freefree_bfgs":
        q, evaluations = _solve_freefree_branch(
            physics, case, linear_state, max_iterations=max_iterations,
            enforce_axis_aligned_reflection_symmetry=enforce_axis_aligned_reflection_symmetry,
        )
    else:
        q, evaluations = _solve_scaled_lbfgs_branch(
            physics, case, linear_state, max_iterations=max_iterations, learning_rate=learning_rate,
            enforce_axis_aligned_reflection_symmetry=enforce_axis_aligned_reflection_symmetry,
        )
    initial_energy = physics.energy(linear_state, case).detach()
    final_energy = physics.energy(q, case).detach()
    gradient = physics.reduced_gradient(q, case).detach()
    correction = (q - linear_state) / physics.coordinate_scale
    reduced = torch.einsum("nr,bn->br", physics.nullspace_dimensionless, correction)
    return (
        reduced.squeeze(0).cpu(),
        (final_energy - initial_energy).squeeze(0).cpu(),
        torch.linalg.vector_norm(gradient, dim=1).squeeze(0).cpu(),
        evaluations,
    )


def _solve_scaled_lbfgs_branch(
    physics: FVKPhysics,
    case: LayupBatch,
    linear_state: torch.Tensor,
    *,
    max_iterations: int,
    learning_rate: float,
    enforce_axis_aligned_reflection_symmetry: bool,
) -> tuple[torch.Tensor, int]:
    """Solve in the model's scaled coordinate metric (mainly a diagnostic)."""
    reduced = torch.nn.Parameter(
        torch.zeros((1, physics.reduced_dofs), device=linear_state.device, dtype=linear_state.dtype)
    )
    optimizer = torch.optim.LBFGS(
        [reduced],
        lr=learning_rate,
        max_iter=max_iterations,
        max_eval=max_iterations * 2,
        tolerance_grad=1.0e-10,
        tolerance_change=1.0e-13,
        history_size=50,
        line_search_fn="strong_wolfe",
    )
    evaluations = 0

    def embedded_q() -> torch.Tensor:
        q = q_from_reduced(physics, linear_state, reduced)
        if enforce_axis_aligned_reflection_symmetry:
            q = physics.enforce_axis_aligned_reflection_symmetry(q, case)
        return q

    def closure() -> torch.Tensor:
        nonlocal evaluations
        optimizer.zero_grad(set_to_none=True)
        energy = physics.energy(embedded_q(), case).sum()
        if not torch.isfinite(energy):
            raise FloatingPointError("Non-finite energy while solving scaled local branch.")
        energy.backward()
        evaluations += 1
        return energy

    optimizer.step(closure)
    return embedded_q().detach(), evaluations


def _solve_freefree_branch(
    physics: FVKPhysics,
    case: LayupBatch,
    linear_state: torch.Tensor,
    *,
    max_iterations: int,
    enforce_axis_aligned_reflection_symmetry: bool,
) -> tuple[torch.Tensor, int]:
    """BFGS in the FREEFREE-compatible, unscaled constrained Ritz space."""
    try:
        from scipy.optimize import minimize
    except ImportError as exc:  # pragma: no cover - SciPy is a project dependency
        raise RuntimeError("SciPy is required for the FREEFREE-compatible branch selector.") from exc

    use_axis_symmetry = enforce_axis_aligned_reflection_symmetry and bool(physics.axis_aligned_mask(case)[0])
    if use_axis_symmetry:
        indices = torch.nonzero(physics.axis_aligned_symmetry_mask, as_tuple=False).squeeze(1)
        nullspace = torch.eye(physics.nq, device=linear_state.device, dtype=linear_state.dtype)[:, indices]
    else:
        constraint = physics.constraint.detach().cpu().numpy()
        _, singular_values, vh = np.linalg.svd(constraint, full_matrices=True)
        tolerance = 1.0e-12 * (float(singular_values.max()) if singular_values.size else 1.0)
        rank = int(np.sum(singular_values > tolerance))
        physical_nullspace = vh[rank:].T.copy()
        nullspace = torch.as_tensor(physical_nullspace, device=linear_state.device, dtype=linear_state.dtype)
    initial_r = (linear_state @ nullspace).squeeze(0).detach().cpu().numpy()

    def energy_and_gradient(reduced: np.ndarray) -> tuple[float, np.ndarray]:
        reduced_torch = torch.tensor(reduced, device=linear_state.device, dtype=linear_state.dtype, requires_grad=True)
        q = torch.einsum("nr,r->n", nullspace, reduced_torch).unsqueeze(0)
        energy = physics.energy(q, case).sum()
        gradient = torch.autograd.grad(energy, reduced_torch)[0]
        return float(energy.detach().cpu()), gradient.detach().cpu().numpy()

    result = minimize(
        fun=energy_and_gradient,
        x0=initial_r,
        jac=True,
        method="BFGS",
        options={"gtol": 1.0e-7, "maxiter": max_iterations, "disp": False},
    )
    q = torch.as_tensor(result.x, device=linear_state.device, dtype=linear_state.dtype) @ nullspace.T
    return q.unsqueeze(0), int(result.nfev)
