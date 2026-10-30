"""Direct, symmetry-augmented residual-curvature surrogate for canonical laminates."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from .evaluate import load_test_cases
from .train import load_fidelity_data
from ..core.physics import ANGLE_SET, E1, T_PLY, FVKPhysics, LayupBatch
from ..core.reporting import COMPONENTS, write_evaluation_report
from ..core.runtime import seed_everything, select_device


def parse_component_loss_weights(text: str) -> tuple[float, float, float]:
    """Parse positive relative weights for the three normalized residual losses."""
    try:
        values = tuple(float(value.strip()) for value in text.split(","))
    except ValueError as error:
        raise ValueError("--component-loss-weights must be three comma-separated numbers.") from error
    if len(values) != 3 or any(value <= 0.0 for value in values):
        raise ValueError("--component-loss-weights must contain three positive values.")
    return values


class ResidualCurvatureNN(nn.Module):
    """Residual surrogate with an optional hard 0/90 versus off-axis split."""

    def __init__(
        self,
        width: int,
        layers: int,
        outputs: int = 3,
        *,
        nplies: int = 8,
        split_offaxis: bool = False,
        include_abd_features: bool = False,
        include_linear_strain_features: bool = False,
    ) -> None:
        super().__init__()
        if not 1 <= nplies <= 10:
            raise ValueError("nplies must be between 1 and 10.")
        self.nplies = int(nplies)
        self.split_offaxis = split_offaxis
        self.include_abd_features = include_abd_features
        self.include_linear_strain_features = include_linear_strain_features

        self.network = self._make_network(width, layers, outputs)
        self.axis_network = self._make_network(width, layers, outputs) if split_offaxis else None
        self.offaxis_network = self._make_network(width, layers, outputs) if split_offaxis else None

    def _make_network(self, width: int, layers: int, outputs: int) -> nn.Sequential:
        modules: list[nn.Module] = []
        # The nonlinear correction is primarily governed by the *linear
        # curvature state*, not merely the categorical stacking sequence.  Its
        # curvature-per-gradient components are even under thermal reversal,
        # so they can safely enter the factorised odd-in-gradient surrogate.
        inputs = self.nplies * len(ANGLE_SET) + 1 + 3 + (18 if self.include_abd_features else 0) + (3 if self.include_linear_strain_features else 0)
        for _ in range(layers):
            modules += [nn.Linear(inputs, width), nn.SiLU()]
            inputs = width
        modules.append(nn.Linear(inputs, outputs))
        return nn.Sequential(*modules).double()

    def offaxis_mask(self, batch: LayupBatch) -> torch.Tensor:
        """Return true when a laminate contains one or more +/-45-degree plies."""
        indices = batch.angle_indices[:, :self.nplies]
        return (indices == ANGLE_SET.index(-45)).any(dim=1) | (indices == ANGLE_SET.index(45)).any(dim=1)

    def forward(self, batch: LayupBatch, linear: torch.Tensor, abd: torch.Tensor | None = None, linear_strain: torch.Tensor | None = None) -> torch.Tensor:
        if bool((batch.nplies != self.nplies).any()):
            raise ValueError(f"Model was configured for {self.nplies}-ply layups.")
        indices = batch.angle_indices[:, :self.nplies]
        one_hot = torch.nn.functional.one_hot(indices, num_classes=len(ANGLE_SET)).to(torch.float64).flatten(1)
        # All FreeFree labels establish that the nonlinear correction is
        # temperature-offset invariant and odd in direct face difference.
        gradient_feature = (batch.top_face_temperature - batch.bottom_face_temperature) / 100.0
        linear_per_gradient = linear / gradient_feature[:, None]
        features = torch.cat((one_hot, gradient_feature.square()[:, None], linear_per_gradient), dim=1)
        if self.include_abd_features:
            if abd is None or abd.shape != (len(batch), 6, 6):
                raise ValueError("ABD features require one 6x6 laminate matrix per case.")
            # Independent in-plane components of A, B and D, nondimensionalized
            # by the appropriate stiffness moment E1*h^(1,2,3).
            h = batch.nplies.to(dtype=features.dtype) * T_PLY
            entries = torch.tensor(
                ((0, 0), (1, 1), (0, 1), (2, 2), (0, 2), (1, 2),
                 (0, 3), (1, 4), (0, 4), (2, 5), (0, 5), (1, 5),
                 (3, 3), (4, 4), (3, 4), (5, 5), (3, 5), (4, 5)),
                device=features.device,
            )
            rows, cols = entries[:, 0], entries[:, 1]
            values = abd[:, rows, cols]
            a = values[:, :6] / (E1 * h[:, None])
            b = values[:, 6:12] / (E1 * h[:, None].square())
            d = values[:, 12:] / (E1 * h[:, None].pow(3))
            features = torch.cat((features, a, b, d), dim=1)
        if self.include_linear_strain_features:
            if linear_strain is None or linear_strain.shape != (len(batch), 3):
                raise ValueError("Linear-strain features require three generalized strain values per case.")
            # Nonlinear FvK stretching couples to the CLT membrane state; scale
            # the strain-per-gradient to O(1) while retaining reversal parity.
            features = torch.cat((features, 1.0e4 * linear_strain / gradient_feature[:, None]), dim=1)
        if not self.split_offaxis:
            output = self.network(features)
        else:
            assert self.axis_network is not None and self.offaxis_network is not None
            offaxis = self.offaxis_mask(batch)
            output = torch.empty((len(batch), self.network[-1].out_features), dtype=features.dtype, device=features.device)
            if bool((~offaxis).any()):
                output[~offaxis] = self.axis_network(features[~offaxis])
            if bool(offaxis.any()):
                output[offaxis] = self.offaxis_network(features[offaxis])
        return gradient_feature[:, None] * output


def rows_for_report(batch: LayupBatch, reference: torch.Tensor, predicted: torch.Tensor, linear: torch.Tensor) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    lookup = (-45, 0, 45, 90)
    for index in range(len(batch)):
        nplies = int(batch.nplies[index])
        layup = "[" + "/".join(str(lookup[int(v)]) for v in batch.angle_indices[index, :nplies]) + "]"
        row: dict[str, object] = {"reference_source": "FREEFREE", "layup": layup, "nplies": nplies,
            "top_face_temperature": float(batch.top_face_temperature[index]), "bottom_face_temperature": float(batch.bottom_face_temperature[index])}
        for component, ref, pred, clt in zip(COMPONENTS, reference[index], predicted[index], linear[index]):
            row[f"reference_{component}"] = float(ref); row[f"pinn_{component}"] = float(pred); row[f"linear_clt_{component}"] = float(clt)
            row[f"pinn_error_{component}"] = float(pred - ref)
            row[f"linear_clt_error_{component}"] = float(clt - ref)
        row["pinn_curvature_error_norm_1_per_m"] = float(torch.linalg.vector_norm(predicted[index] - reference[index]))
        row["linear_clt_curvature_error_norm_1_per_m"] = float(torch.linalg.vector_norm(linear[index] - reference[index]))
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train direct CLT-residual curvature NN.")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--nplies", type=int, default=8, help="Ply count represented by both datasets (1-10).")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target-noise-std", type=float, default=0.0, help="Gaussian curvature-label augmentation SD [1/m] applied independently each step.")
    parser.add_argument(
        "--component-loss-weights", default="1,1,1",
        help="Relative weights for normalized kappa_x,kappa_y,kappa_xy residual losses.",
    )
    parser.add_argument(
        "--residual-components", choices=("all", "twist_only"), default="all",
        help="Components corrected by the NN; untouched components remain exact linear CLT.",
    )
    parser.add_argument(
        "--split-offaxis-experts",
        action="store_true",
        help="Use separate hard-gated experts for 0/90-only and +/-45-containing layups.",
    )
    parser.add_argument("--include-abd-features", action="store_true", help="Append normalized candidate A/B/D coupling descriptors.")
    parser.add_argument("--include-linear-strain-features", action="store_true", help="Append normalized CLT membrane-strain-per-gradient features.")
    parser.add_argument("--augment-thickness-reversal", action="store_true", help="Augment exact stack-reversal/face-temperature-swap cases.")
    parser.add_argument("--augment-thickness-reversal-offaxis-only", action="store_true", help="Apply thickness-reversal augmentation only to +/-45-containing laminates.")
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.nplies <= 10:
        raise ValueError("--nplies must be between 1 and 10.")
    if args.target_noise_std < 0.0:
        raise ValueError("--target-noise-std must be nonnegative.")
    component_loss_weights = torch.tensor(
        parse_component_loss_weights(args.component_loss_weights), dtype=torch.float64,
    )
    seed_everything(args.seed); device = select_device("auto")
    physics = FVKPhysics().to(device)
    train_batch, train_target = load_fidelity_data(
        args.train_data, augment_symmetries=True,
        augment_thickness_reversal=args.augment_thickness_reversal,
        augment_thickness_reversal_offaxis_only=args.augment_thickness_reversal_offaxis_only,
    )
    test_batch, test_rows, _ = load_test_cases(args.test_data)
    train_batch, train_target = train_batch.to(device), train_target.to(device)
    test_batch = test_batch.to(device)
    with torch.no_grad():
        linear_train = physics.curvatures(physics.linear_initial_q(train_batch))
        abd_train, thermal_train = physics.laminate_matrices(train_batch)
        linear_strain_train = torch.linalg.solve(abd_train, thermal_train.unsqueeze(-1)).squeeze(-1)[:, :3]
        train_residual = train_target - linear_train
        scale = train_residual.std(0).clamp_min(1e-4)
    target_indices = (0, 1, 2) if args.residual_components == "all" else (2,)
    active_loss_weights = component_loss_weights[list(target_indices)].to(device)
    model = ResidualCurvatureNN(
        args.width, args.layers, outputs=len(target_indices), nplies=args.nplies,
        split_offaxis=args.split_offaxis_experts, include_abd_features=args.include_abd_features,
        include_linear_strain_features=args.include_linear_strain_features,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
    target_noise_generator = torch.Generator(device=device).manual_seed(args.seed + 10_000)
    for step in range(1, args.steps + 1):
        predicted = model(train_batch, linear_train, abd_train, linear_strain_train)
        training_residual = train_residual[:, target_indices]
        if args.target_noise_std > 0.0:
            training_residual = training_residual + torch.randn(
                training_residual.shape, dtype=training_residual.dtype,
                device=device, generator=target_noise_generator,
            ) * args.target_noise_std
        normalized_error = ((predicted - training_residual) / scale[list(target_indices)]).square()
        normalized_error = normalized_error * active_loss_weights
        if args.split_offaxis_experts:
            offaxis = model.offaxis_mask(train_batch)
            # Each mechanics regime contributes equally even if their sample
            # counts differ after symmetry augmentation.
            group_losses = [normalized_error[mask].mean() for mask in (offaxis, ~offaxis) if bool(mask.any())]
            loss = torch.stack(group_losses).mean()
        else:
            loss = normalized_error.mean()
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step == 1 or step % 500 == 0 or step == args.steps:
            print(f"step {step:5d} | residual loss {loss.item():.5e}", flush=True)
    model.eval()
    with torch.no_grad():
        linear = physics.curvatures(physics.linear_initial_q(test_batch))
        abd_test, thermal_test = physics.laminate_matrices(test_batch)
        linear_strain_test = torch.linalg.solve(abd_test, thermal_test.unsqueeze(-1)).squeeze(-1)[:, :3]
        correction = torch.zeros_like(linear)
        correction[:, target_indices] = model(test_batch, linear, abd_test, linear_strain_test)
        predicted = linear + correction
    rows = rows_for_report(test_batch.to("cpu"), torch.tensor([[float(r[f"reference_{c}"]) for c in COMPONENTS] for r in test_rows]), predicted.cpu(), linear.cpu())
    _, summary, plot = write_evaluation_report(rows, args.output_prefix, reference_label="FREEFREE")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "width": args.width, "layers": args.layers, "nplies": args.nplies, "residual_components": args.residual_components, "split_offaxis_experts": args.split_offaxis_experts, "include_abd_features": args.include_abd_features, "include_linear_strain_features": args.include_linear_strain_features, "residual_scale": scale.cpu()}, args.output_prefix.with_name(args.output_prefix.name + "_checkpoint.pt"))
    print(f"Metrics: {summary}\nPlot: {plot}")


if __name__ == "__main__":
    raise SystemExit(main())
