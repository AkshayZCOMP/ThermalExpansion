"""Neural Ritz ansatz for fast thermally curved laminate inference."""

from __future__ import annotations

import torch
from torch import nn

from .physics import ANGLE_SET, FVKPhysics, L, LayupBatch, T_STRESS_FREE, W


class RitzPINN(nn.Module):
    """Map a discrete layup and two face temperatures to FvK Ritz coefficients.

    The linear-CLT Ritz state is added as a deterministic baseline.  The MLP
    learns only its nonlinear correction in a hard-constrained nullspace, so
    no network output can violate the free-free rigid-rotation gauge.
    """

    def __init__(
        self,
        physics: FVKPhysics,
        *,
        max_plies: int = 10,
        hidden_width: int = 160,
        hidden_layers: int = 4,
        temperature_scale: float = 100.0,
        correction_amplitude: float = 3.0,
        temperature_feature_mode: str = "faces",
        gradient_residual_expert: bool = False,
        gradient_expert_width: int = 96,
        enforce_axis_aligned_reflection_symmetry: bool = True,
        curvature_trust_region: bool = False,
        curvature_relative_bound: float = 0.10,
        curvature_absolute_bound: float = 0.02,
        fixed_linear_curvature_components: tuple[bool, bool, bool] = (False, False, False),
        thermal_odd_symmetry: bool = False,
        layup_d4_equivariance: bool = False,
        split_offaxis_experts: bool = False,
    ):
        super().__init__()
        if max_plies != 10:
            raise ValueError("max_plies must remain 10 to match run_layup_sweep.py.")
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be positive.")
        if temperature_feature_mode not in {"faces", "midplane_difference", "physics_normalized", "gradient_only"}:
            raise ValueError(
                "temperature_feature_mode must be 'faces', 'midplane_difference', 'physics_normalized', or 'gradient_only'."
            )
        if gradient_expert_width < 1:
            raise ValueError("gradient_expert_width must be positive.")
        if curvature_relative_bound < 0.0 or curvature_absolute_bound < 0.0:
            raise ValueError("curvature trust-region bounds cannot be negative.")
        self.physics = physics
        self.max_plies = max_plies
        self.hidden_width = hidden_width
        self.hidden_layers = hidden_layers
        self.temperature_scale = float(temperature_scale)
        self.correction_amplitude = float(correction_amplitude)
        self.temperature_feature_mode = temperature_feature_mode
        self.gradient_residual_expert = bool(gradient_residual_expert)
        self.gradient_expert_width = int(gradient_expert_width)
        self.enforce_axis_aligned_reflection_symmetry = bool(enforce_axis_aligned_reflection_symmetry)
        self.curvature_trust_region = bool(curvature_trust_region)
        self.curvature_relative_bound = float(curvature_relative_bound)
        self.curvature_absolute_bound = float(curvature_absolute_bound)
        if len(fixed_linear_curvature_components) != 3:
            raise ValueError("fixed_linear_curvature_components must contain three booleans.")
        self.fixed_linear_curvature_components = tuple(bool(value) for value in fixed_linear_curvature_components)
        self.thermal_odd_symmetry = bool(thermal_odd_symmetry)
        self.layup_d4_equivariance = bool(layup_d4_equivariance)
        self.split_offaxis_experts = bool(split_offaxis_experts)

        # Four material angle one-hot entries plus a ply-present flag per ply,
        # then normalized ply count, top face, and bottom face temperatures.
        input_size = max_plies * 5 + 3
        def make_network() -> nn.Sequential:
            layers: list[nn.Module] = []
            previous = input_size
            for _ in range(hidden_layers):
                layers.extend((nn.Linear(previous, hidden_width), nn.SiLU()))
                previous = hidden_width
            layers.append(nn.Linear(previous, physics.reduced_dofs))
            return nn.Sequential(*layers).to(dtype=torch.float64)

        self.network = make_network()
        self.axis_network = make_network() if self.split_offaxis_experts else None
        self.offaxis_network = make_network() if self.split_offaxis_experts else None
        # Laminate ABD systems couple terms across several orders of magnitude;
        # keep the learned ansatz in the same float64 precision as mechanics.
        for network in (self.network, self.axis_network, self.offaxis_network):
            if network is not None:
                final_layer = network[-1]
                assert isinstance(final_layer, nn.Linear)
                nn.init.zeros_(final_layer.weight)
                nn.init.zeros_(final_layer.bias)
        if self.gradient_residual_expert:
            # The primary network already receives T_top and T_bottom.  This
            # zero-initialized residual expert makes signed gradient powers
            # explicit, which helps interpolate nonlinear twist response
            # without altering the hard Ritz/gauge construction.
            self.gradient_expert = nn.Sequential(
                nn.Linear(input_size + 4, self.gradient_expert_width),
                nn.SiLU(),
                nn.Linear(self.gradient_expert_width, self.gradient_expert_width),
                nn.SiLU(),
                nn.Linear(self.gradient_expert_width, physics.reduced_dofs),
            ).to(dtype=torch.float64)
            expert_final = self.gradient_expert[-1]
            assert isinstance(expert_final, nn.Linear)
            nn.init.zeros_(expert_final.weight)
            nn.init.zeros_(expert_final.bias)

    def features(self, batch: LayupBatch) -> torch.Tensor:
        safe_indices = batch.angle_indices.clamp_min(0)
        one_hot = torch.nn.functional.one_hot(safe_indices, num_classes=4).to(batch.top_face_temperature.dtype)
        present = (batch.angle_indices >= 0).unsqueeze(-1).to(one_hot.dtype)
        ply_features = torch.cat((one_hot * present, present), dim=-1).flatten(start_dim=1)
        if self.temperature_feature_mode == "faces":
            temperature_features = (
                batch.top_face_temperature / self.temperature_scale,
                batch.bottom_face_temperature / self.temperature_scale,
            )
        elif self.temperature_feature_mode == "midplane_difference":
            temperature_features = (
                0.5 * (batch.top_face_temperature + batch.bottom_face_temperature) / self.temperature_scale,
                (batch.top_face_temperature - batch.bottom_face_temperature) / self.temperature_scale,
            )
        elif self.temperature_feature_mode == "gradient_only":
            temperature_features = (
                torch.zeros_like(batch.top_face_temperature),
                (batch.top_face_temperature - batch.bottom_face_temperature) / self.temperature_scale,
            )
        else:
            # These are the two thermal quantities that enter the laminate
            # resultants: stress-free-referenced mean temperature and the
            # through-thickness temperature change.  The latter equals
            # h*dT/dz, so it stays well scaled when ply count changes.
            temperature_features = (
                (0.5 * (batch.top_face_temperature + batch.bottom_face_temperature) - T_STRESS_FREE)
                / self.temperature_scale,
                (batch.top_face_temperature - batch.bottom_face_temperature) / self.temperature_scale,
            )
        scalar_features = torch.stack(
            (batch.nplies.to(one_hot.dtype) / self.max_plies, *temperature_features),
            dim=1,
        )
        return torch.cat((ply_features, scalar_features), dim=1)

    def predict_reduced_correction(self, batch: LayupBatch) -> torch.Tensor:
        """Return the dimensionless null-space correction before Ritz embedding."""
        raw = self._raw_reduced_correction(batch)
        if not self.thermal_odd_symmetry:
            return raw
        swapped = LayupBatch(
            batch.angle_indices, batch.nplies,
            batch.bottom_face_temperature, batch.top_face_temperature,
        )
        return 0.5 * (raw - self._raw_reduced_correction(swapped))

    @staticmethod
    def offaxis_mask(batch: LayupBatch) -> torch.Tensor:
        """Identify the +/-45 regime used by the hard-gated Ritz experts."""
        return (
            (batch.angle_indices == ANGLE_SET.index(-45)).any(dim=1)
            | (batch.angle_indices == ANGLE_SET.index(45)).any(dim=1)
        )

    def _raw_reduced_correction(self, batch: LayupBatch) -> torch.Tensor:
        features = self.features(batch)
        if not self.split_offaxis_experts:
            output = self.network(features)
        else:
            assert self.axis_network is not None and self.offaxis_network is not None
            offaxis = self.offaxis_mask(batch)
            output = torch.empty(
                (len(batch), self.physics.reduced_dofs), dtype=features.dtype, device=features.device,
            )
            if bool((~offaxis).any()):
                output[~offaxis] = self.axis_network(features[~offaxis])
            if bool(offaxis.any()):
                output[offaxis] = self.offaxis_network(features[offaxis])
        if self.gradient_residual_expert:
            midplane = 0.5 * (batch.top_face_temperature + batch.bottom_face_temperature) / self.temperature_scale
            half_difference = 0.5 * (batch.top_face_temperature - batch.bottom_face_temperature) / self.temperature_scale
            gradient_features = torch.stack(
                (
                    half_difference.square(),
                    half_difference.pow(3),
                    midplane * half_difference,
                    midplane * half_difference.square(),
                ),
                dim=1,
            )
            output = output + self.gradient_expert(torch.cat((features, gradient_features), dim=1))
        return self.correction_amplitude * torch.tanh(output)

    def forward(self, batch: LayupBatch) -> torch.Tensor:
        q = self._base_q(batch)
        if self.layup_d4_equivariance:
            q = self._enforce_d4_curvature_equivariance(q, batch)
        return q

    def _base_q(self, batch: LayupBatch) -> torch.Tensor:
        correction_reduced = self.predict_reduced_correction(batch)
        # q_correction = S N r, with C S N = 0 by construction.
        correction_q = torch.einsum(
            "nr,br,n->bn",
            self.physics.nullspace_dimensionless,
            correction_reduced,
            self.physics.coordinate_scale,
        )
        q = self.physics.linear_initial_q(batch) + correction_q
        if self.enforce_axis_aligned_reflection_symmetry:
            q = self.physics.enforce_axis_aligned_reflection_symmetry(q, batch)
        if self.curvature_trust_region:
            q = self._apply_curvature_trust_region(q, batch)
        if any(self.fixed_linear_curvature_components):
            q = self._fix_linear_curvature_components(q, batch)
        return q

    @staticmethod
    def _transform_layup(batch: LayupBatch, mapping: tuple[int, int, int, int]) -> LayupBatch:
        lookup = torch.tensor(mapping, dtype=batch.angle_indices.dtype, device=batch.angle_indices.device)
        active = batch.angle_indices >= 0
        safe = batch.angle_indices.clamp_min(0)
        transformed = torch.where(active, lookup[safe], batch.angle_indices)
        return LayupBatch(transformed, batch.nplies, batch.top_face_temperature, batch.bottom_face_temperature)

    def _enforce_d4_curvature_equivariance(self, q: torch.Tensor, batch: LayupBatch) -> torch.Tensor:
        """Average predictions transformed back from square-plate D4 equivalents."""
        mirror = self._transform_layup(batch, (2, 1, 0, 3))
        rotate = self._transform_layup(batch, (2, 3, 0, 1))
        rotate_mirror = self._transform_layup(mirror, (2, 3, 0, 1))
        k_identity = self.physics.curvatures(q)
        k_mirror = self.physics.curvatures(self._base_q(mirror)) * torch.tensor(
            (1.0, 1.0, -1.0), dtype=q.dtype, device=q.device,
        )
        k_rotate = self.physics.curvatures(self._base_q(rotate))[:, (1, 0, 2)] * torch.tensor(
            (1.0, 1.0, -1.0), dtype=q.dtype, device=q.device,
        )
        k_rotate_mirror = self.physics.curvatures(self._base_q(rotate_mirror))[:, (1, 0, 2)]
        desired = (k_identity + k_mirror + k_rotate + k_rotate_mirror) / 4.0
        adjustment = desired - k_identity
        projected = q.clone()
        transverse_offset = self.physics.nu + self.physics.nv
        projected[:, transverse_offset] -= adjustment[:, 0] * L * L
        projected[:, transverse_offset + 1] -= adjustment[:, 1] * W * W
        projected[:, transverse_offset + 2] -= adjustment[:, 2] * L * W
        return projected

    def _fix_linear_curvature_components(self, q: torch.Tensor, batch: LayupBatch) -> torch.Tensor:
        """Keep selected mean-curvature components exactly at the CLT state."""
        current = self.physics.curvatures(q)
        linear = self.physics.curvatures(self.physics.linear_initial_q(batch))
        mask = torch.tensor(self.fixed_linear_curvature_components, dtype=torch.bool, device=q.device)
        desired = torch.where(mask[None, :], linear, current)
        adjustment = desired - current
        projected = q.clone()
        transverse_offset = self.physics.nu + self.physics.nv
        projected[:, transverse_offset] -= adjustment[:, 0] * L * L
        projected[:, transverse_offset + 1] -= adjustment[:, 1] * W * W
        projected[:, transverse_offset + 2] -= adjustment[:, 2] * L * W
        return projected

    def _apply_curvature_trust_region(self, q: torch.Tensor, batch: LayupBatch) -> torch.Tensor:
        """Keep the learned branch correction near the exact CLT state.

        Linear CLT supplies the physically signed initial branch.  FvK may
        alter it, but the learned mean curvature is smoothly saturated within
        a relative/absolute neighborhood instead of being permitted to jump
        to a remote stable basin.  Axis-aligned laminates retain their exact
        zero-twist reflection symmetry.
        """
        clt_curvature = self.physics.curvatures(self.physics.linear_initial_q(batch))
        current_curvature = self.physics.curvatures(q)
        bound = torch.maximum(
            self.curvature_relative_bound * clt_curvature.abs(),
            torch.full_like(clt_curvature, self.curvature_absolute_bound),
        )
        # A tanh projection is differentiable and preserves the sign of an
        # active CLT component whenever the bound is smaller than its
        # magnitude, while still permitting a small nonlinear correction.
        desired_curvature = clt_curvature + bound * torch.tanh((current_curvature - clt_curvature) / bound)
        axis_aligned = self.physics.axis_aligned_mask(batch)
        desired_curvature = desired_curvature.clone()
        desired_curvature[axis_aligned, 2] = 0.0
        adjustment = desired_curvature - current_curvature
        projected = q.clone()
        transverse_offset = self.physics.nu + self.physics.nv
        # The first three transverse Ritz terms are 0.5*x^2, 0.5*y^2,
        # 0.5*x*y.  They adjust area-mean curvatures exactly without changing
        # the free-free gauge or higher-order shape modes.
        projected[:, transverse_offset] -= adjustment[:, 0] * L * L
        projected[:, transverse_offset + 1] -= adjustment[:, 1] * W * W
        projected[:, transverse_offset + 2] -= adjustment[:, 2] * L * W
        return projected

    def predict_curvatures(self, batch: LayupBatch) -> torch.Tensor:
        return self.physics.curvatures(self(batch))

    def checkpoint_metadata(self) -> dict[str, float | int | bool]:
        return {
            "max_plies": self.max_plies,
            "hidden_width": self.hidden_width,
            "hidden_layers": self.hidden_layers,
            "temperature_scale": self.temperature_scale,
            "correction_amplitude": self.correction_amplitude,
            "temperature_feature_mode": self.temperature_feature_mode,
            "gradient_residual_expert": self.gradient_residual_expert,
            "gradient_expert_width": self.gradient_expert_width,
            "enforce_axis_aligned_reflection_symmetry": self.enforce_axis_aligned_reflection_symmetry,
            "curvature_trust_region": self.curvature_trust_region,
            "curvature_relative_bound": self.curvature_relative_bound,
            "curvature_absolute_bound": self.curvature_absolute_bound,
            "fixed_linear_curvature_components": self.fixed_linear_curvature_components,
            "thermal_odd_symmetry": self.thermal_odd_symmetry,
            "layup_d4_equivariance": self.layup_d4_equivariance,
            "split_offaxis_experts": self.split_offaxis_experts,
            "transverse_degree": self.physics.transverse_degree,
            "area_quad_n": self.physics.area_quad_n,
        }
