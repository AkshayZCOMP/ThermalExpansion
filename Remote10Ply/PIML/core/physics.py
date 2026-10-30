"""Differentiable laminate mechanics used by the curvature PINN.

The model is deliberately expressed in the same free-free Föppl--von Kármán
Ritz space as ``Computational Real/FREEFREE.py``.  A neural network provides
the Ritz coefficients, while this module evaluates the thermal potential
energy.  Therefore no finite-element/Abaqus labels are required to train the
base model.

Layup ordering follows ``AbaqusSweep/run_layup_sweep.py``: the first ply is
at the bottom (negative z) and the final ply is at the top.  Temperatures are
*laminate face temperatures*, not sink temperatures.  The prescribed field is
``T(z) = (T_top + T_bottom)/2 + (T_top - T_bottom) * z / h``.
"""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


# This module lives in PIML/core, so the repository root is two levels up.
ROOT = Path(__file__).resolve().parents[2]
SWEEP_SCRIPT = ROOT / "AbaqusSweep" / "run_layup_sweep.py"

# These material and geometry constants intentionally match FREEFREE.py.
L = 0.3
W = 0.3
T_PLY = 0.000125
E1 = 130.0e9
E2 = 10.0e9
NU12 = 0.3
G12 = 5.0e9
ALPHA1 = -0.5e-6
ALPHA2 = 30.0e-6
T_STRESS_FREE = 25.0
ANGLE_SET = (-45, 0, 45, 90)
SIMPLE_CASE_NPLIES = 8
SIMPLE_CASE_MIDPLANE_TEMPERATURE = 50.0
SIMPLE_CASE_THERMAL_GRADIENT = -53333.3
GRADIENT_GRID_TRAINING = (-80000.0, -53333.3, -26666.7, 26666.7, 53333.3, 80000.0)
GRADIENT_GRID_VALIDATION = (-66666.7, -40000.0, 40000.0, 66666.7)
MIDPLANE_TEMPERATURES_TRAINING = (25.0, 50.0, 75.0)
MIDPLANE_TEMPERATURES_VALIDATION = (37.5, 62.5)


@dataclass(frozen=True)
class LayupBatch:
    """A padded batch of bottom-to-top layups and direct face temperatures."""

    angle_indices: torch.Tensor  # (batch, max_plies), -1 in padded entries
    nplies: torch.Tensor  # (batch,)
    top_face_temperature: torch.Tensor  # (batch,)
    bottom_face_temperature: torch.Tensor  # (batch,)

    def to(self, device: torch.device | str) -> "LayupBatch":
        return LayupBatch(
            self.angle_indices.to(device),
            self.nplies.to(device),
            self.top_face_temperature.to(device),
            self.bottom_face_temperature.to(device),
        )

    def index_select(self, indices: torch.Tensor) -> "LayupBatch":
        return LayupBatch(
            self.angle_indices.index_select(0, indices),
            self.nplies.index_select(0, indices),
            self.top_face_temperature.index_select(0, indices),
            self.bottom_face_temperature.index_select(0, indices),
        )

    def __len__(self) -> int:
        return int(self.nplies.shape[0])


def parse_layup(text: str) -> list[int]:
    """Parse ``[0/45/-45/90]`` or ``0/45/-45/90`` into ply angles."""
    clean = text.strip().strip("[]")
    if not clean:
        raise ValueError("A layup must contain at least one ply.")
    try:
        layup = [int(float(part.strip())) for part in clean.split("/")]
    except ValueError as exc:
        raise ValueError(f"Invalid layup {text!r}.") from exc
    invalid = sorted(set(layup).difference(ANGLE_SET))
    if invalid:
        raise ValueError(f"Unsupported angles {invalid}; expected {list(ANGLE_SET)}.")
    return layup


def layup_string(layup: Sequence[int]) -> str:
    return "[" + "/".join(str(int(angle)) for angle in layup) + "]"


@lru_cache(maxsize=1)
def all_sweep_layups() -> tuple[tuple[int, ...], ...]:
    """Load the canonical layup design space from the Abaqus sweep script.

    Importing the script is safe: Abaqus is only launched under its main
    guard.  This avoids duplicating the symmetric/balanced rules in PIML.
    """
    if not SWEEP_SCRIPT.exists():
        raise FileNotFoundError(f"Could not find the layup sweep at {SWEEP_SCRIPT}")
    spec = importlib.util.spec_from_file_location("piml_layup_sweep", SWEEP_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {SWEEP_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tuple(tuple(int(angle) for angle in layup) for _, layup in module.all_cases())


@lru_cache(maxsize=1)
def eight_ply_sweep_layups() -> tuple[tuple[int, ...], ...]:
    """Return only the canonical eight-ply layups from the sweep (72 cases)."""
    layups = tuple(layup for layup in all_sweep_layups() if len(layup) == SIMPLE_CASE_NPLIES)
    if len(layups) != 72:
        raise RuntimeError(f"Expected 72 canonical eight-ply layups, found {len(layups)}.")
    return layups


def fixed_gradient_face_temperatures(
    nplies: int,
    *,
    midplane_temperature: float = SIMPLE_CASE_MIDPLANE_TEMPERATURE,
    thermal_gradient: float = SIMPLE_CASE_THERMAL_GRADIENT,
) -> tuple[float, float]:
    """Convert ``T_mid`` and ``dT/dz`` into direct top/bottom face values."""
    if nplies <= 0:
        raise ValueError("nplies must be positive.")
    thickness = nplies * T_PLY
    top = midplane_temperature + thermal_gradient * thickness / 2.0
    bottom = midplane_temperature - thermal_gradient * thickness / 2.0
    if top == bottom:
        raise ValueError("The fixed thermal gradient must give unequal face temperatures.")
    return float(top), float(bottom)


def fixed_gradient_cases(
    nplies: int = SIMPLE_CASE_NPLIES,
    *,
    count: int | None = None,
    seed: int = 7,
    midplane_temperature: float = SIMPLE_CASE_MIDPLANE_TEMPERATURE,
    thermal_gradient: float = SIMPLE_CASE_THERMAL_GRADIENT,
    dtype: torch.dtype = torch.float64,
) -> LayupBatch:
    """Build unique canonical layups under one prescribed linear field.

    When ``count`` is omitted every valid layup is returned.  Otherwise a
    deterministic sample without replacement is used, which is appropriate
    for held-out FREEFREE validation.
    """
    candidates = tuple(layup for layup in all_sweep_layups() if len(layup) == nplies)
    if not candidates:
        raise ValueError(f"The sweep contains no {nplies}-ply layups.")
    if count is None:
        selected = candidates
    else:
        if count <= 0 or count > len(candidates):
            raise ValueError(f"count must be between 1 and {len(candidates)} for {nplies}-ply cases.")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        indices = torch.randperm(len(candidates), generator=generator)[:count]
        selected = tuple(candidates[int(index)] for index in indices)
    top, bottom = fixed_gradient_face_temperatures(
        nplies,
        midplane_temperature=midplane_temperature,
        thermal_gradient=thermal_gradient,
    )
    return make_batch(
        selected,
        [top] * len(selected),
        [bottom] * len(selected),
        dtype=dtype,
    )


def gradient_grid_cases(
    thermal_gradients: Sequence[float],
    *,
    nplies: int = SIMPLE_CASE_NPLIES,
    count_per_gradient: int | None = None,
    seed: int = 7,
    midplane_temperature: float = SIMPLE_CASE_MIDPLANE_TEMPERATURE,
    dtype: torch.dtype = torch.float64,
) -> LayupBatch:
    """Build 8-ply layup/linear-gradient combinations in canonical order.

    Every gradient must be nonzero, so the direct top and bottom laminate
    faces are always different.  When a count is requested, the same
    deterministic layup subset is paired with every gradient; this is useful
    for a compact, matched FREEFREE validation matrix.
    """
    gradients = tuple(float(value) for value in thermal_gradients)
    return temperature_field_grid_cases(
        tuple((midplane_temperature, gradient) for gradient in gradients),
        nplies=nplies,
        count_per_field=count_per_gradient,
        seed=seed,
        dtype=dtype,
    )


def temperature_field_grid_cases(
    temperature_fields: Sequence[tuple[float, float]],
    *,
    nplies: int = SIMPLE_CASE_NPLIES,
    count_per_field: int | None = None,
    layup_indices: Sequence[int] | None = None,
    seed: int = 7,
    dtype: torch.dtype = torch.float64,
) -> LayupBatch:
    """Build canonical layup/direct-temperature-field combinations.

    ``temperature_fields`` contains ``(T_mid, beta)`` pairs for
    ``T(z)=T_mid+beta*z``.  Every beta must be nonzero, which guarantees
    unequal direct top/bottom face temperatures.  A requested sample is
    matched across fields, making temperature generalization comparisons fair.
    """
    fields = tuple((float(midplane), float(gradient)) for midplane, gradient in temperature_fields)
    if not fields:
        raise ValueError("At least one temperature field is required.")
    if any(
        not np.isfinite(midplane) or not np.isfinite(gradient) or gradient == 0.0
        for midplane, gradient in fields
    ):
        raise ValueError("Temperature fields must be finite and have nonzero thermal gradients.")
    if len(set(fields)) != len(fields):
        raise ValueError("Temperature fields must be unique.")
    candidates = tuple(layup for layup in all_sweep_layups() if len(layup) == nplies)
    if not candidates:
        raise ValueError(f"The sweep contains no {nplies}-ply layups.")
    if layup_indices is not None and count_per_field is not None:
        raise ValueError("Specify either layup_indices or count_per_field, not both.")
    if layup_indices is not None:
        indices = tuple(int(index) for index in layup_indices)
        if not indices or len(set(indices)) != len(indices) or any(index < 0 or index >= len(candidates) for index in indices):
            raise ValueError(f"layup_indices must be unique indices in [0, {len(candidates) - 1}].")
        selected = tuple(candidates[index] for index in indices)
    elif count_per_field is None:
        selected = candidates
    else:
        if count_per_field <= 0 or count_per_field > len(candidates):
            raise ValueError(f"count_per_field must be between 1 and {len(candidates)}.")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        indices = torch.randperm(len(candidates), generator=generator)[:count_per_field]
        selected = tuple(candidates[int(index)] for index in indices)
    layups: list[tuple[int, ...]] = []
    tops: list[float] = []
    bottoms: list[float] = []
    for midplane, gradient in fields:
        top, bottom = fixed_gradient_face_temperatures(
            nplies,
            midplane_temperature=midplane,
            thermal_gradient=gradient,
        )
        layups.extend(selected)
        tops.extend([top] * len(selected))
        bottoms.extend([bottom] * len(selected))
    return make_batch(layups, tops, bottoms, dtype=dtype)


def all_ply_temperature_field_grid_cases(
    temperature_fields: Sequence[tuple[float, float]],
    *,
    dtype: torch.dtype = torch.float64,
) -> LayupBatch:
    """Pair every canonical 3--10 ply layup with each temperature field."""
    fields = tuple((float(midplane), float(gradient)) for midplane, gradient in temperature_fields)
    if not fields:
        raise ValueError("At least one temperature field is required.")
    if any(
        not np.isfinite(midplane) or not np.isfinite(gradient) or gradient == 0.0
        for midplane, gradient in fields
    ):
        raise ValueError("Temperature fields must be finite and have nonzero thermal gradients.")
    if len(set(fields)) != len(fields):
        raise ValueError("Temperature fields must be unique.")
    selected = all_sweep_layups()
    layups: list[tuple[int, ...]] = []
    tops: list[float] = []
    bottoms: list[float] = []
    for midplane, gradient in fields:
        for layup in selected:
            top, bottom = fixed_gradient_face_temperatures(
                len(layup),
                midplane_temperature=midplane,
                thermal_gradient=gradient,
            )
            layups.append(layup)
            tops.append(top)
            bottoms.append(bottom)
    return make_batch(layups, tops, bottoms, dtype=dtype)


def make_batch(
    layups: Sequence[Sequence[int]],
    top_face_temperature: Sequence[float] | torch.Tensor,
    bottom_face_temperature: Sequence[float] | torch.Tensor,
    *,
    max_plies: int = 10,
    dtype: torch.dtype = torch.float64,
) -> LayupBatch:
    """Encode layups and direct face temperatures for the network/physics."""
    if not layups:
        raise ValueError("At least one layup is required.")
    if any(len(layup) > max_plies or len(layup) == 0 for layup in layups):
        raise ValueError(f"Layups must contain between 1 and {max_plies} plies.")
    indices = torch.full((len(layups), max_plies), -1, dtype=torch.long)
    nplies = torch.empty(len(layups), dtype=torch.long)
    angle_to_index = {angle: index for index, angle in enumerate(ANGLE_SET)}
    for row, layup in enumerate(layups):
        try:
            values = [angle_to_index[int(angle)] for angle in layup]
        except KeyError as exc:
            raise ValueError(f"Unsupported angle in {layup}; expected {list(ANGLE_SET)}.") from exc
        indices[row, : len(values)] = torch.tensor(values, dtype=torch.long)
        nplies[row] = len(values)

    top = torch.as_tensor(top_face_temperature, dtype=dtype).reshape(-1)
    bottom = torch.as_tensor(bottom_face_temperature, dtype=dtype).reshape(-1)
    if top.numel() != len(layups) or bottom.numel() != len(layups):
        raise ValueError("One top and one bottom temperature are required per layup.")
    return LayupBatch(indices, nplies, top, bottom)


def sample_cases(
    count: int,
    *,
    temperature_min: float = 0.0,
    temperature_max: float = 100.0,
    seed: int = 7,
    max_plies: int = 10,
    include_temperature_corners: bool = True,
    minimum_face_temperature_difference: float = 1.0,
    dtype: torch.dtype = torch.float64,
) -> LayupBatch:
    """Sample canonical sweep layups with independently varied face values.

    Every sampled case has ``abs(T_top - T_bottom) >=
    minimum_face_temperature_difference``.  A small deterministic subset uses
    the two maximum-gradient corners, while all remaining samples use
    independent uniform draws rejected when their faces are too close.  This
    keeps the training domain strictly within the requested nonuniform linear
    through-thickness temperature-field assumption.
    """
    if count <= 0:
        raise ValueError("count must be positive.")
    if temperature_max <= temperature_min:
        raise ValueError("temperature_max must exceed temperature_min.")
    if minimum_face_temperature_difference <= 0.0:
        raise ValueError("minimum_face_temperature_difference must be positive.")
    if minimum_face_temperature_difference > temperature_max - temperature_min:
        raise ValueError("minimum_face_temperature_difference exceeds the temperature range.")
    candidates = all_sweep_layups()
    if not candidates:
        raise RuntimeError("The layup sweep returned no cases.")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    choice = torch.randint(len(candidates), (count,), generator=generator)
    layups = [candidates[int(index)] for index in choice]
    span = temperature_max - temperature_min
    top = temperature_min + span * torch.rand(count, generator=generator, dtype=dtype)
    bottom = temperature_min + span * torch.rand(count, generator=generator, dtype=dtype)
    invalid = (top - bottom).abs() < minimum_face_temperature_difference
    while bool(invalid.any()):
        redraw_count = int(invalid.sum())
        top[invalid] = temperature_min + span * torch.rand(redraw_count, generator=generator, dtype=dtype)
        bottom[invalid] = temperature_min + span * torch.rand(redraw_count, generator=generator, dtype=dtype)
        invalid = (top - bottom).abs() < minimum_face_temperature_difference
    if include_temperature_corners:
        corners = torch.tensor(
            [
                [temperature_min, temperature_max],
                [temperature_max, temperature_min],
            ],
            dtype=dtype,
        )
        corner_count = min(count, max(2, count // 8))
        repetitions = math.ceil(corner_count / len(corners))
        selected = corners.repeat((repetitions, 1))[:corner_count]
        top[:corner_count] = selected[:, 0]
        bottom[:corner_count] = selected[:, 1]
    return make_batch(layups, top, bottom, max_plies=max_plies, dtype=dtype)


def _transformed_ply_matrices(angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the engineering-strain Qbar and CTE transform used by FREEFREE."""
    theta = np.deg2rad(angle)
    c, s = np.cos(theta), np.sin(theta)
    transform = np.array(
        [[c * c, s * s, 2.0 * c * s], [s * s, c * c, -2.0 * c * s], [-c * s, c * s, c * c - s * s]],
        dtype=float,
    )
    reuter = np.diag([1.0, 1.0, 2.0])
    nu21 = NU12 * E2 / E1
    q11 = E1 / (1.0 - NU12 * nu21)
    q22 = E2 / (1.0 - NU12 * nu21)
    q12 = NU12 * E2 / (1.0 - NU12 * nu21)
    q = np.array([[q11, q12, 0.0], [q12, q22, 0.0], [0.0, 0.0, G12]])
    qbar = np.linalg.inv(transform) @ q @ reuter @ transform @ np.linalg.inv(reuter)
    strain_transform = reuter @ transform @ np.linalg.inv(reuter)
    alpha_bar = np.linalg.inv(strain_transform) @ np.array([ALPHA1, ALPHA2, 0.0])
    return qbar, alpha_bar


class FVKPhysics(torch.nn.Module):
    """Differentiable FvK Ritz energy for a batch of laminate cases.

    The returned energy is the same mechanical energy minimized by the
    existing solver, apart from the temperature-only constant.  The omitted
    constant cannot affect an equilibrium or an energy difference.
    """

    def __init__(self, *, transverse_degree: int = 5, area_quad_n: int = 9):
        super().__init__()
        if transverse_degree < 2:
            raise ValueError("transverse_degree must be at least 2.")
        if area_quad_n < 2:
            raise ValueError("area_quad_n must be at least 2.")
        self.transverse_degree = int(transverse_degree)
        self.area_quad_n = int(area_quad_n)
        self.in_plane_degree = max(3, 2 * transverse_degree - 1)
        self.in_plane_monomials = tuple(
            (px, degree - px)
            for degree in range(1, self.in_plane_degree + 1)
            for px in range(degree, -1, -1)
        )
        w_terms: list[tuple[float, int, int]] = [(0.5, 2, 0), (0.5, 0, 2), (0.5, 1, 1)]
        for degree in range(3, self.transverse_degree + 1):
            w_terms.extend((1.0, px, degree - px) for px in range(degree, -1, -1))
        self.w_monomials = tuple(w_terms)
        self.nu = len(self.in_plane_monomials)
        self.nv = self.nu
        self.nw = len(self.w_monomials)
        self.nq = self.nu + self.nv + self.nw

        qbar_lookup, alpha_lookup = zip(*(_transformed_ply_matrices(angle) for angle in ANGLE_SET))
        self.register_buffer("qbar_lookup", torch.tensor(np.stack(qbar_lookup), dtype=torch.float64))
        self.register_buffer("alpha_lookup", torch.tensor(np.stack(alpha_lookup), dtype=torch.float64))

        basis = self._make_area_basis()
        for name, value in basis.items():
            self.register_buffer(name, torch.tensor(value, dtype=torch.float64))

        constraint = np.zeros((1, self.nq), dtype=float)
        _, duy_at_origin = self._in_plane_derivatives(0.0, 0.0, scale=L)
        dvx_at_origin, _ = self._in_plane_derivatives(0.0, 0.0, scale=W)
        constraint[0, : self.nu] = duy_at_origin
        constraint[0, self.nu : self.nu + self.nv] = -dvx_at_origin
        coordinate_scale = np.full(self.nq, 1.0e-3, dtype=float)
        coordinate_scale[self.nu + self.nv :] = 0.10  # transverse Ritz coefficients [m]
        _, _, vt = np.linalg.svd(constraint * coordinate_scale[None, :], full_matrices=True)
        nullspace_dimensionless = vt[1:].T
        self.register_buffer("coordinate_scale", torch.tensor(coordinate_scale, dtype=torch.float64))
        self.register_buffer("nullspace_dimensionless", torch.tensor(nullspace_dimensionless, dtype=torch.float64))
        self.register_buffer("constraint", torch.tensor(constraint, dtype=torch.float64))

        # For an axis-aligned (0/90-only) laminate under the present uniform
        # in-plane geometry and through-thickness thermal field, the intended
        # free solution is invariant under x and y reflection.  Retain only
        # the corresponding Ritz parities: u odd-x/even-y, v even-x/odd-y,
        # and w even-x/even-y.  This removes the numerically selected,
        # symmetry-broken twist pair that disagrees with Abaqus for these
        # cases.  Every retained coordinate already satisfies the rigid gauge.
        symmetric_indices: list[int] = []
        symmetric_indices.extend(
            index
            for index, (px, py) in enumerate(self.in_plane_monomials)
            if px % 2 == 1 and py % 2 == 0
        )
        symmetric_indices.extend(
            self.nu + index
            for index, (px, py) in enumerate(self.in_plane_monomials)
            if px % 2 == 0 and py % 2 == 1
        )
        symmetric_indices.extend(
            self.nu + self.nv + index
            for index, (_, px, py) in enumerate(self.w_monomials)
            if px % 2 == 0 and py % 2 == 0
        )
        symmetry_mask = np.zeros(self.nq, dtype=bool)
        symmetry_mask[symmetric_indices] = True
        self.register_buffer("axis_aligned_symmetry_mask", torch.tensor(symmetry_mask, dtype=torch.bool))

    @property
    def reduced_dofs(self) -> int:
        return self.nq - 1

    def axis_aligned_mask(self, batch: LayupBatch) -> torch.Tensor:
        """Return cases containing only 0° and 90° active plies."""
        zero_index = ANGLE_SET.index(0)
        ninety_index = ANGLE_SET.index(90)
        active = batch.angle_indices >= 0
        axis_aligned = (batch.angle_indices == zero_index) | (batch.angle_indices == ninety_index)
        return torch.all(~active | axis_aligned, dim=1)

    def enforce_axis_aligned_reflection_symmetry(
        self,
        q: torch.Tensor,
        batch: LayupBatch,
    ) -> torch.Tensor:
        """Project 0/90-only cases onto the Abaqus-consistent symmetric branch."""
        if q.ndim != 2 or q.shape[1] != self.nq:
            raise ValueError(f"q must have shape (batch, {self.nq}).")
        apply = self.axis_aligned_mask(batch)
        if not bool(torch.any(apply)):
            return q
        masked = q * self.axis_aligned_symmetry_mask.to(dtype=q.dtype)
        return torch.where(apply[:, None], masked, q)

    def _in_plane_derivatives(self, x: float, y: float, *, scale: float) -> tuple[np.ndarray, np.ndarray]:
        dx = []
        dy = []
        for px, py in self.in_plane_monomials:
            dx.append(
                0.0 if px == 0 else scale * px * (x / L) ** (px - 1) * (y / W) ** py / L
            )
            dy.append(
                0.0 if py == 0 else scale * py * (x / L) ** px * (y / W) ** (py - 1) / W
            )
        return np.array(dx), np.array(dy)

    @staticmethod
    def _term_derivative(term: tuple[float, int, int], dx_order: int, dy_order: int, x: float, y: float) -> float:
        coefficient, px, py = term
        if px < dx_order or py < dy_order:
            return 0.0
        factor = coefficient
        for offset in range(dx_order):
            factor *= px - offset
        for offset in range(dy_order):
            factor *= py - offset
        return factor * (x / L) ** (px - dx_order) * (y / W) ** (py - dy_order) / (L**dx_order * W**dy_order)

    def _make_area_basis(self) -> dict[str, np.ndarray]:
        nodes, weights = np.polynomial.legendre.leggauss(self.area_quad_n)
        x_values = 0.5 * L * nodes + 0.5 * L
        y_values = 0.5 * W * nodes + 0.5 * W
        samples = [(x - L / 2.0, y - W / 2.0, wx * wy) for x, wx in zip(x_values, weights * L / 2.0) for y, wy in zip(y_values, weights * W / 2.0)]
        dux, duy, dvx, dvy, dwx, dwy, dxx, dxy, dyy, area_weights = ([] for _ in range(10))
        for x, y, weight in samples:
            u_x, u_y = self._in_plane_derivatives(x, y, scale=L)
            v_x, v_y = self._in_plane_derivatives(x, y, scale=W)
            dux.append(u_x)
            duy.append(u_y)
            dvx.append(v_x)
            dvy.append(v_y)
            dwx.append([self._term_derivative(term, 1, 0, x, y) for term in self.w_monomials])
            dwy.append([self._term_derivative(term, 0, 1, x, y) for term in self.w_monomials])
            dxx.append([self._term_derivative(term, 2, 0, x, y) for term in self.w_monomials])
            dxy.append([self._term_derivative(term, 1, 1, x, y) for term in self.w_monomials])
            dyy.append([self._term_derivative(term, 0, 2, x, y) for term in self.w_monomials])
            area_weights.append(weight)
        return {
            "dux": np.asarray(dux), "duy": np.asarray(duy), "dvx": np.asarray(dvx), "dvy": np.asarray(dvy),
            "dwx": np.asarray(dwx), "dwy": np.asarray(dwy), "dxx": np.asarray(dxx), "dxy": np.asarray(dxy), "dyy": np.asarray(dyy),
            "area_weights": np.asarray(area_weights),
        }

    def laminate_matrices(self, batch: LayupBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ABD and [N_T, M_T] for direct linear face temperatures."""
        device = batch.top_face_temperature.device
        dtype = batch.top_face_temperature.dtype
        bsz, max_plies = batch.angle_indices.shape
        if max_plies > 10:
            raise ValueError("This model supports at most ten plies, matching the sweep.")
        h = batch.nplies.to(dtype) * T_PLY
        ply = torch.arange(max_plies, device=device, dtype=dtype).expand(bsz, -1)
        active = ply < batch.nplies[:, None]
        z0 = -0.5 * h[:, None] + ply * T_PLY
        z1 = z0 + T_PLY
        dz = z1 - z0
        dz2 = z1.square() - z0.square()
        dz3 = z1.pow(3) - z0.pow(3)

        safe_indices = batch.angle_indices.clamp_min(0)
        qbar = self.qbar_lookup.to(device=device, dtype=dtype)[safe_indices]
        alpha = self.alpha_lookup.to(device=device, dtype=dtype)[safe_indices]
        active_matrix = active[:, :, None, None].to(dtype)
        qbar = qbar * active_matrix
        qa = torch.matmul(qbar, alpha.unsqueeze(-1)).squeeze(-1)

        a = torch.einsum("bpij,bp->bij", qbar, dz)
        b = 0.5 * torch.einsum("bpij,bp->bij", qbar, dz2)
        d = (1.0 / 3.0) * torch.einsum("bpij,bp->bij", qbar, dz3)
        abd = torch.zeros((bsz, 6, 6), device=device, dtype=dtype)
        abd[:, :3, :3] = a
        abd[:, :3, 3:] = b
        abd[:, 3:, :3] = b
        abd[:, 3:, 3:] = d

        dtemp_mid = 0.5 * (batch.top_face_temperature + batch.bottom_face_temperature) - T_STRESS_FREE
        beta = (batch.top_face_temperature - batch.bottom_face_temperature) / h
        integral_temp = dtemp_mid[:, None] * dz + 0.5 * beta[:, None] * dz2
        integral_z_temp = 0.5 * dtemp_mid[:, None] * dz2 + (1.0 / 3.0) * beta[:, None] * dz3
        n_thermal = torch.einsum("bpi,bp->bi", qa, integral_temp)
        m_thermal = torch.einsum("bpi,bp->bi", qa, integral_z_temp)
        thermal_resultant = torch.cat((n_thermal, m_thermal), dim=1)
        return abd, thermal_resultant

    def linear_initial_q(self, batch: LayupBatch) -> torch.Tensor:
        """Embed the exact linear-CLT solution in the FvK Ritz coefficient vector."""
        abd, thermal = self.laminate_matrices(batch)
        generalized = torch.linalg.solve(abd, thermal.unsqueeze(-1)).squeeze(-1)
        bsz = len(batch)
        q = torch.zeros((bsz, self.nq), device=generalized.device, dtype=generalized.dtype)
        x_linear = self.in_plane_monomials.index((1, 0))
        y_linear = self.in_plane_monomials.index((0, 1))
        eps = generalized[:, :3]
        kappa = generalized[:, 3:]
        q[:, x_linear] = eps[:, 0]
        q[:, y_linear] = 0.5 * eps[:, 2] * W / L
        q[:, self.nu + x_linear] = 0.5 * eps[:, 2] * L / W
        q[:, self.nu + y_linear] = eps[:, 1]
        q[:, self.nu + self.nv + 0] = -kappa[:, 0] * L * L
        q[:, self.nu + self.nv + 1] = -kappa[:, 1] * W * W
        q[:, self.nu + self.nv + 2] = -kappa[:, 2] * L * W
        return q

    def generalized_fields(self, q: torch.Tensor) -> torch.Tensor:
        """Return [eps_x, eps_y, gamma_xy, kappa_x, kappa_y, kappa_xy] at quadrature points."""
        if q.ndim != 2 or q.shape[1] != self.nq:
            raise ValueError(f"q must have shape (batch, {self.nq}).")
        a = q[:, : self.nu]
        b = q[:, self.nu : self.nu + self.nv]
        c = q[:, self.nu + self.nv :]
        du0_dx = torch.einsum("bn,an->ba", a, self.dux)
        du0_dy = torch.einsum("bn,an->ba", a, self.duy)
        dv0_dx = torch.einsum("bn,an->ba", b, self.dvx)
        dv0_dy = torch.einsum("bn,an->ba", b, self.dvy)
        dw_dx = torch.einsum("bn,an->ba", c, self.dwx)
        dw_dy = torch.einsum("bn,an->ba", c, self.dwy)
        d2w_xx = torch.einsum("bn,an->ba", c, self.dxx)
        d2w_xy = torch.einsum("bn,an->ba", c, self.dxy)
        d2w_yy = torch.einsum("bn,an->ba", c, self.dyy)
        eps_x = du0_dx + 0.5 * dw_dx.square()
        eps_y = dv0_dy + 0.5 * dw_dy.square()
        gamma_xy = du0_dy + dv0_dx + dw_dx * dw_dy
        return torch.stack((eps_x, eps_y, gamma_xy, -d2w_xx, -d2w_yy, -2.0 * d2w_xy), dim=-1)

    def energy(self, q: torch.Tensor, batch: LayupBatch) -> torch.Tensor:
        """Mechanical potential energy per case, excluding a q-independent term."""
        abd, thermal = self.laminate_matrices(batch)
        fields = self.generalized_fields(q)
        quadratic = 0.5 * torch.einsum("bai,bij,baj->ba", fields, abd, fields)
        loading = torch.einsum("bai,bi->ba", fields, thermal)
        return torch.einsum("a,ba->b", self.area_weights, quadratic - loading)

    def curvatures(self, q: torch.Tensor) -> torch.Tensor:
        """Area-average [kappa_x, kappa_y, kappa_xy] in 1/m."""
        return torch.einsum("a,bai->bi", self.area_weights, self.generalized_fields(q)[..., 3:]) / self.area_weights.sum()

    def generalized_average(self, q: torch.Tensor) -> torch.Tensor:
        """Area-average generalized strains then curvatures."""
        return torch.einsum("a,bai->bi", self.area_weights, self.generalized_fields(q)) / self.area_weights.sum()

    def reduced_gradient(self, q: torch.Tensor, batch: LayupBatch, *, create_graph: bool = False) -> torch.Tensor:
        """Projected FvK stationarity residual dU/dr for diagnostics/training."""
        if not q.requires_grad:
            q = q.detach().requires_grad_(True)
        total_energy = self.energy(q, batch).sum()
        gradient_q = torch.autograd.grad(total_energy, q, create_graph=create_graph)[0]
        scaled_gradient = gradient_q * self.coordinate_scale
        return torch.einsum("nr,bn->br", self.nullspace_dimensionless, scaled_gradient)

    def temperature_at(self, z: torch.Tensor, batch: LayupBatch) -> torch.Tensor:
        """Evaluate the prescribed direct-face linear temperature profile."""
        h = batch.nplies.to(z.dtype) * T_PLY
        mid = 0.5 * (batch.top_face_temperature + batch.bottom_face_temperature)
        beta = (batch.top_face_temperature - batch.bottom_face_temperature) / h
        z_values = z.reshape(1, -1)
        return mid[:, None] + beta[:, None] * z_values
