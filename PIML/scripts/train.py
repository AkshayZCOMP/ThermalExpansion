"""Train the physics-informed nonlinear curvature surrogate.

Examples
--------
Train from physics alone over the layups defined by the Abaqus sweep::

    python -m PIML.scripts.train --steps 5000 --cases 4096 \
        --temperature-min 0 --temperature-max 100

Optionally calibrate the same model to a small set of converged FREEFREE
solutions created by ``PIML.scripts.generate_fvk_dataset``::

    python -m PIML.scripts.train --fidelity-data PIML/data/fvk_samples.csv \
        --fidelity-weight 10
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from ..core.losses import (
    CURVATURE_SCALE,
    directional_stability_loss,
    energy_difference_loss,
    equilibrium_residual_loss,
    validation_metrics,
)
from ..core.model import RitzPINN
from ..core.physics import FVKPhysics, LayupBatch, layup_string, make_batch, parse_layup, sample_cases, temperature_field_grid_cases
from ..core.reporting import write_training_history
from ..core.runtime import seed_everything, select_device
from ..core.run_paths import add_run_id_arguments, resolve_run_id, timestamped_file, timestamped_prefix


def load_fidelity_data(
    path: Path, *, max_plies: int = 10, augment_symmetries: bool = False,
    augment_thickness_reversal: bool = False,
    augment_thickness_reversal_offaxis_only: bool = False,
    excluded_layups: set[str] | None = None,
) -> tuple[LayupBatch, torch.Tensor]:
    """Load converged FREEFREE or Abaqus curvature labels for calibration.

    The source is detected from the header.  Both sources use direct laminate
    face temperatures and bottom-to-top layup order, so the feature vector is
    identical to the PIML input.  Abaqus labels are useful for dense sweep
    calibration; FREEFREE labels are the preferred same-model calibration.
    """
    layups: list[list[int]] = []
    tops: list[float] = []
    bottoms: list[float] = []
    curvatures: list[list[float]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header.")
        is_freefree = "layup" in reader.fieldnames
        is_abaqus = "Layup" in reader.fieldnames
        if not (is_freefree or is_abaqus):
            raise ValueError(f"{path} is not a recognized FREEFREE or Abaqus result CSV.")
        curvature_columns = ("kappa_x", "kappa_y", "kappa_xy") if is_freefree else ("K_11", "K_22", "K_12")
        for row in reader:
            if is_freefree and row.get("converged", "True").strip().lower() != "true":
                continue
            layup = parse_layup(row["layup"] if is_freefree else row["Layup"])
            if excluded_layups is not None and layup_string(layup) in excluded_layups:
                continue
            top = float(row["top_face_temperature"] if is_freefree else row["TopTemperature"])
            bottom = float(row["bottom_face_temperature"] if is_freefree else row["BottomTemperature"])
            curvature = [float(row[name]) for name in curvature_columns]
            # For the square 0.3 m x 0.3 m plate, these are exact in-plane
            # D4 transformations.  The engineering twist curvature changes
            # sign under a 90-degree coordinate rotation or a reflection.
            transforms = [(layup, curvature)]
            if augment_symmetries:
                mirror = {-45: 45, 0: 0, 45: -45, 90: 90}
                rotate = {-45: 45, 0: 90, 45: -45, 90: 0}
                reflected = [mirror[angle] for angle in layup]
                rotated = [rotate[angle] for angle in layup]
                rotated_reflected = [rotate[angle] for angle in reflected]
                kx, ky, kxy = curvature
                transforms.extend([
                    (reflected, [kx, ky, -kxy]),
                    (rotated, [ky, kx, -kxy]),
                    (rotated_reflected, [ky, kx, kxy]),
                ])
            reverse_this_case = augment_thickness_reversal and (
                not augment_thickness_reversal_offaxis_only
                or any(abs(angle) == 45 for angle in layup)
            )
            if reverse_this_case:
                # Flipping the laminate through its mid-plane and exchanging
                # the prescribed face temperatures reverses the curvature
                # vector.  This is distinct from the in-plane D4 transforms.
                transforms.extend([
                    (list(reversed(transformed_layup)), [-value for value in transformed_curvature])
                    for transformed_layup, transformed_curvature in list(transforms)
                ])
            augmented_transforms = [
                (transformed_layup, transformed_curvature, False)
                for transformed_layup, transformed_curvature in transforms
            ]
            if reverse_this_case:
                original_count = len(transforms) // 2
                augmented_transforms = [
                    (transformed_layup, transformed_curvature, index >= original_count)
                    for index, (transformed_layup, transformed_curvature) in enumerate(transforms)
                ]
            for transformed_layup, transformed_curvature, reversed_thickness in augmented_transforms:
                layups.append(transformed_layup)
                tops.append(bottom if reversed_thickness else top)
                bottoms.append(top if reversed_thickness else bottom)
                curvatures.append(transformed_curvature)
    if not layups:
        raise ValueError(f"No converged rows found in {path}.")
    return (
        make_batch(layups, tops, bottoms, max_plies=max_plies),
        torch.tensor(curvatures, dtype=torch.float64),
    )


def load_excluded_layups(path: Path) -> set[str]:
    """Read held-out layups from a FREEFREE CSV or an evaluation-report CSV."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "layup" not in reader.fieldnames:
            raise ValueError(f"{path} must contain a layup column.")
        return {layup_string(parse_layup(row["layup"])) for row in reader if row.get("layup")}


def parse_values(text: str, *, name: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in text.split(",") if value.strip())
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{name} must contain one or more unique values.")
    return values


def parse_indices(text: str, *, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    except ValueError as error:
        raise ValueError(f"{name} must be comma-separated integer indices.") from error
    if not values or len(set(values)) != len(values) or any(value < 0 for value in values):
        raise ValueError(f"{name} must contain unique nonnegative indices.")
    return values


def parse_fixed_curvature_components(text: str | None) -> tuple[bool, bool, bool]:
    """Parse curvature names that should retain their exact CLT prediction."""
    names = ("kappa_x", "kappa_y", "kappa_xy")
    if text is None:
        return (False, False, False)
    selected = {value.strip() for value in text.split(",") if value.strip()}
    unknown = selected.difference(names)
    if unknown:
        raise ValueError(f"Unknown curvature components: {sorted(unknown)}.")
    return tuple(name in selected for name in names)  # type: ignore[return-value]


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a hard-constrained Deep-Ritz PINN for laminate curvatures."
    )
    parser.add_argument("--cases", type=int, default=4096, help="Physics-only cases sampled from the canonical sweep.")
    parser.add_argument("--validation-cases", type=int, default=512)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--full-batch",
        action="store_true",
        help="Use every prescribed-field case at every Adam step; recommended for small energy-minimization studies.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--lbfgs-steps",
        type=int,
        default=0,
        help="Optional full-batch L-BFGS energy-minimization polish after Adam (energy-only, no auxiliary losses).",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-7)
    parser.add_argument("--hidden-width", type=int, default=160)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--transverse-degree", type=int, default=5, help="Use 5 to match FREEFREE.py.")
    parser.add_argument("--area-quad-n", type=int, default=9, help="Use 9 to match FREEFREE.py.")
    parser.add_argument("--temperature-min", type=float, default=0.0, help="Minimum direct face temperature [C].")
    parser.add_argument("--temperature-max", type=float, default=100.0, help="Maximum direct face temperature [C].")
    parser.add_argument("--temperature-scale", type=float, default=100.0)
    parser.add_argument(
        "--temperature-feature-mode",
        choices=("faces", "midplane_difference", "physics_normalized", "gradient_only"),
        default="faces",
        help="Thermal feature parameterization; physics_normalized centers T at stress-free and uses h*dT/dz.",
    )
    parser.add_argument("--thermal-odd-symmetry", action="store_true", help="Enforce an odd learned correction under top/bottom temperature exchange.")
    parser.add_argument("--layup-d4-equivariance", action="store_true", help="Hard-enforce square-plate 90-degree rotation and reflection curvature equivariance.")
    parser.add_argument(
        "--split-offaxis-experts", action="store_true",
        help="Hard-route 0/90-only and +/-45-containing laminates to separate Ritz experts.",
    )
    parser.add_argument(
        "--minimum-face-temperature-difference",
        type=float,
        default=1.0,
        help="Require |T_top - T_bottom| >= this value [C] for every physics case.",
    )
    parser.add_argument("--nplies", type=int, help="Restrict direct physics training to one canonical ply count.")
    parser.add_argument("--midplane-temperatures", help="Comma-separated T_mid values [C] for a prescribed field grid.")
    parser.add_argument("--thermal-gradients", help="Comma-separated nonzero dT/dz values [C/m] for a prescribed field grid.")
    parser.add_argument("--cases-per-temperature-field", type=int, help="Canonical layups per prescribed field; omit for all layups.")
    parser.add_argument(
        "--layup-indices",
        help="Comma-separated canonical layup indices for prescribed-field training; enables an explicit held-out layup split.",
    )
    parser.add_argument("--correction-amplitude", type=float, default=3.0)
    parser.add_argument(
        "--curvature-trust-region", action="store_true",
        help="Keep predicted mean curvatures in a bounded, CLT-signed branch neighborhood.",
    )
    parser.add_argument("--curvature-relative-bound", type=float, default=0.10)
    parser.add_argument("--curvature-absolute-bound", type=float, default=0.02)
    parser.add_argument(
        "--fix-linear-curvature-components",
        help="Comma-separated curvature components retained exactly at CLT, e.g. kappa_x,kappa_y.",
    )
    parser.add_argument(
        "--physics-loss",
        choices=("residual", "energy", "hybrid"),
        default="energy",
        help="Energy is Deep-Ritz; residual enforces stationarity; hybrid combines both without labels.",
    )
    parser.add_argument("--residual-weight", type=float, default=1.0, help="Weight of the normalized stationarity residual in hybrid mode.")
    parser.add_argument("--energy-weight", type=float, default=1.0, help="Weight of the scaled Deep-Ritz energy in hybrid mode.")
    parser.add_argument("--stability-weight", type=float, default=0.0, help="Weight of the label-free local-stability penalty.")
    parser.add_argument("--stability-directions", type=int, default=2, help="Random gauge-valid directions checked per stability batch.")
    parser.add_argument("--stability-margin", type=float, default=0.05, help="Required second variation relative to linear-state directional stiffness.")
    parser.add_argument(
        "--path-continuation-points", type=int, default=1,
        help="Number of zero-to-load states evaluated together per laminate; 1 disables path training.",
    )
    parser.add_argument(
        "--path-consistency-weight", type=float, default=0.0,
        help="Weight for continuous, CLT-anchored reduced-coordinate paths; requires path-continuation-points > 1.",
    )
    parser.add_argument("--continuation-stages", type=int, default=1, help="Number of thermal-gradient continuation stages.")
    parser.add_argument("--continuation-start", type=float, default=1.0, help="Initial fraction of each field's top-bottom temperature difference.")
    parser.add_argument("--fidelity-data", type=Path, help="Optional FREEFREE or Abaqus CSV used only for calibration.")
    parser.add_argument(
        "--fidelity-exclude-layups-from", type=Path,
        help="CSV whose layup column defines held-out geometries excluded from FREEFREE calibration.",
    )
    parser.add_argument("--fidelity-augment-symmetries", action="store_true", help="Augment calibration labels with exact square-plate 90-degree rotations and reflections.")
    parser.add_argument("--fidelity-residual-targets", action="store_true", help="Fit curvature residuals relative to the exact CLT state, standardized component-wise.")
    parser.add_argument("--fidelity-weight", type=float, default=0.0, help="Weight of curvature MSE against optional nonlinear labels.")
    parser.add_argument("--physics-weight-start", type=float, default=1.0, help="Physics-loss weight at the beginning of curriculum training.")
    parser.add_argument("--physics-weight-end", type=float, default=1.0, help="Physics-loss weight at the end of curriculum training.")
    parser.add_argument("--fidelity-weight-start", type=float, help="Optional initial curvature-label weight; defaults to --fidelity-weight.")
    parser.add_argument("--fidelity-weight-end", type=float, help="Optional final curvature-label weight; defaults to --fidelity-weight.")
    parser.add_argument(
        "--curriculum-ramp-start",
        type=float,
        default=0.0,
        help="Fraction of Adam steps before linearly transitioning loss weights to their final values.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device string.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("PIML/artifacts/ritz_pinn.pt"))
    parser.add_argument(
        "--report-prefix",
        type=Path,
        help="Prefix for <prefix>_history.csv and <prefix>_training.png; defaults beside --output.",
    )
    add_run_id_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    run_id = resolve_run_id(args)
    args.output = timestamped_file(args.output, run_id)
    if args.report_prefix is not None:
        args.report_prefix = timestamped_prefix(args.report_prefix, run_id)
    if args.steps <= 0 or args.cases <= 0 or args.batch_size <= 0 or args.lbfgs_steps < 0:
        raise ValueError("steps, cases, and batch-size must be positive.")
    if args.residual_weight < 0.0 or args.energy_weight < 0.0 or args.stability_weight < 0.0:
        raise ValueError("physics-loss weights cannot be negative.")
    if args.stability_directions <= 0 or args.stability_margin < 0.0:
        raise ValueError("stability-directions must be positive and stability-margin cannot be negative.")
    if args.path_continuation_points <= 0 or args.path_consistency_weight < 0.0:
        raise ValueError("path-continuation-points must be positive and path-consistency-weight cannot be negative.")
    if args.path_consistency_weight > 0.0 and args.path_continuation_points < 2:
        raise ValueError("path-consistency-weight requires at least two path-continuation-points.")
    if args.curvature_relative_bound < 0.0 or args.curvature_absolute_bound < 0.0:
        raise ValueError("curvature trust-region bounds cannot be negative.")
    if args.continuation_stages <= 0 or not 0.0 < args.continuation_start <= 1.0:
        raise ValueError("continuation-stages must be positive and continuation-start must lie in (0, 1].")
    fidelity_weight_start = args.fidelity_weight if args.fidelity_weight_start is None else args.fidelity_weight_start
    fidelity_weight_end = args.fidelity_weight if args.fidelity_weight_end is None else args.fidelity_weight_end
    if (
        args.physics_weight_start < 0.0 or args.physics_weight_end < 0.0
        or fidelity_weight_start < 0.0 or fidelity_weight_end < 0.0
        or not 0.0 <= args.curriculum_ramp_start < 1.0
    ):
        raise ValueError("Curriculum loss weights must be nonnegative and --curriculum-ramp-start must lie in [0, 1).")
    if max(fidelity_weight_start, fidelity_weight_end) > 0.0 and args.fidelity_data is None:
        raise ValueError("A nonzero fidelity curriculum weight requires --fidelity-data.")
    if args.lbfgs_steps and (
        args.physics_loss != "energy"
        or args.stability_weight != 0.0
        or args.path_consistency_weight != 0.0
        or max(fidelity_weight_start, fidelity_weight_end) != 0.0
    ):
        raise ValueError(
            "--lbfgs-steps supports only pure energy minimization; disable stability, path, and fidelity losses."
        )
    field_grid_requested = args.midplane_temperatures is not None or args.thermal_gradients is not None
    if field_grid_requested and (args.midplane_temperatures is None or args.thermal_gradients is None):
        raise ValueError("--midplane-temperatures and --thermal-gradients must be supplied together.")
    if field_grid_requested and args.nplies is None:
        raise ValueError("--nplies is required for prescribed temperature-field training.")
    if args.layup_indices is not None and not field_grid_requested:
        raise ValueError("--layup-indices requires prescribed temperature-field training.")
    seed_everything(args.seed)
    device = select_device(args.device)
    print(f"Using {device}; PyTorch {torch.__version__}")

    physics = FVKPhysics(transverse_degree=args.transverse_degree, area_quad_n=args.area_quad_n).to(device)
    model = RitzPINN(
        physics,
        hidden_width=args.hidden_width,
        hidden_layers=args.hidden_layers,
        temperature_scale=args.temperature_scale,
        correction_amplitude=args.correction_amplitude,
        temperature_feature_mode=args.temperature_feature_mode,
        curvature_trust_region=args.curvature_trust_region,
        curvature_relative_bound=args.curvature_relative_bound,
        curvature_absolute_bound=args.curvature_absolute_bound,
        fixed_linear_curvature_components=parse_fixed_curvature_components(args.fix_linear_curvature_components),
        thermal_odd_symmetry=args.thermal_odd_symmetry,
        layup_d4_equivariance=args.layup_d4_equivariance,
        split_offaxis_experts=args.split_offaxis_experts,
    ).to(device)
    if field_grid_requested:
        midplanes = parse_values(args.midplane_temperatures, name="--midplane-temperatures")
        gradients = parse_values(args.thermal_gradients, name="--thermal-gradients")
        if any(gradient == 0.0 for gradient in gradients):
            raise ValueError("--thermal-gradients cannot contain zero.")
        fields = tuple((midplane, gradient) for midplane in midplanes for gradient in gradients)
        layup_indices = parse_indices(args.layup_indices, name="--layup-indices") if args.layup_indices else None
        train_cases = temperature_field_grid_cases(
            fields, nplies=args.nplies, count_per_field=args.cases_per_temperature_field,
            layup_indices=layup_indices, seed=args.seed,
        ).to(device)
        validation_cases = temperature_field_grid_cases(
            fields, nplies=args.nplies, count_per_field=args.cases_per_temperature_field,
            layup_indices=layup_indices, seed=args.seed + 1,
        ).to(device)
        print(
            f"Direct physics field grid: {len(fields)} fields x "
            f"{len(train_cases) // len(fields)} canonical {args.nplies}-ply layups; no target labels.",
            flush=True,
        )
    else:
        train_cases = sample_cases(
            args.cases, temperature_min=args.temperature_min, temperature_max=args.temperature_max,
            seed=args.seed, minimum_face_temperature_difference=args.minimum_face_temperature_difference,
        ).to(device)
        validation_cases = sample_cases(
            args.validation_cases, temperature_min=args.temperature_min, temperature_max=args.temperature_max,
            seed=args.seed + 1, minimum_face_temperature_difference=args.minimum_face_temperature_difference,
        ).to(device)

    fidelity_cases: LayupBatch | None = None
    fidelity_targets: torch.Tensor | None = None
    if args.fidelity_data is not None and max(fidelity_weight_start, fidelity_weight_end) > 0.0:
        excluded_layups = (
            load_excluded_layups(args.fidelity_exclude_layups_from)
            if args.fidelity_exclude_layups_from is not None else None
        )
        fidelity_cases, fidelity_targets = load_fidelity_data(
            args.fidelity_data,
            augment_symmetries=args.fidelity_augment_symmetries,
            excluded_layups=excluded_layups,
        )
        fidelity_cases = fidelity_cases.to(device)
        fidelity_targets = fidelity_targets.to(device)
        print(
            f"Loaded {len(fidelity_cases)} curvature calibration cases"
            f"{f' after excluding {len(excluded_layups)} held-out layups' if excluded_layups is not None else ''}."
        )
    fidelity_residual_scale: torch.Tensor | None = None
    if args.fidelity_residual_targets and fidelity_cases is not None and fidelity_targets is not None:
        with torch.no_grad():
            linear_targets = physics.curvatures(physics.linear_initial_q(fidelity_cases))
            fidelity_targets = fidelity_targets - linear_targets
            fidelity_residual_scale = fidelity_targets.std(dim=0).clamp_min(1.0e-4)
        print(f"CLT-residual target scales: {fidelity_residual_scale.detach().cpu().tolist()}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.learning_rate * 0.05)
    train_generator = torch.Generator(device=device).manual_seed(args.seed + 2)
    fidelity_generator = torch.Generator(device=device).manual_seed(args.seed + 3)
    stability_generator = torch.Generator(device=device).manual_seed(args.seed + 4)
    history: list[dict[str, float | int]] = []

    def continuation_batch(batch: LayupBatch, factor: float) -> LayupBatch:
        if factor == 1.0:
            return batch
        midplane = 0.5 * (batch.top_face_temperature + batch.bottom_face_temperature)
        half_difference = 0.5 * (batch.top_face_temperature - batch.bottom_face_temperature) * factor
        return LayupBatch(
            batch.angle_indices,
            batch.nplies,
            midplane + half_difference,
            midplane - half_difference,
        )

    def loading_path_batch(batch: LayupBatch) -> tuple[LayupBatch, int]:
        """Return matched zero-to-load states for each sampled laminate.

        The zero-gradient endpoint fixes the nonlinear correction to zero:
        its exact FvK state is the linear CLT state.  Adjacent states are
        deliberately evaluated together so a network cannot select unrelated
        stable basins at neighboring load levels.
        """
        points = args.path_continuation_points
        if points == 1:
            return batch, 1
        factors = torch.linspace(0.0, 1.0, points, dtype=batch.top_face_temperature.dtype, device=device)
        midplane = 0.5 * (batch.top_face_temperature + batch.bottom_face_temperature)
        half_difference = 0.5 * (batch.top_face_temperature - batch.bottom_face_temperature)
        return (
            LayupBatch(
                batch.angle_indices.repeat((points, 1)),
                batch.nplies.repeat(points),
                (midplane[None, :] + factors[:, None] * half_difference[None, :]).reshape(-1),
                (midplane[None, :] - factors[:, None] * half_difference[None, :]).reshape(-1),
            ),
            points,
        )

    model.train()
    for step in range(1, args.steps + 1):
        index = (
            torch.arange(len(train_cases), device=device)
            if args.full_batch
            else torch.randint(len(train_cases), (args.batch_size,), generator=train_generator, device=device)
        )
        batch = train_cases.index_select(index)
        stage = min(args.continuation_stages - 1, (step - 1) * args.continuation_stages // args.steps)
        continuation_factor = (
            args.continuation_start
            + (1.0 - args.continuation_start) * stage / max(args.continuation_stages - 1, 1)
        )
        batch = continuation_batch(batch, continuation_factor)
        path_batch, path_points = loading_path_batch(batch)
        zero = torch.zeros((), dtype=torch.float64, device=device)
        # A zero-weight data-driven baseline must not spend most of its time
        # differentiating the Ritz energy only to multiply it by zero.
        physics_enabled = args.physics_weight_start > 0.0 or args.physics_weight_end > 0.0
        residual_loss = (
            equilibrium_residual_loss(model, physics, path_batch)
            if physics_enabled and args.physics_loss in {"residual", "hybrid"} else zero
        )
        energy_loss = (
            energy_difference_loss(model, physics, path_batch)
            if physics_enabled and args.physics_loss in {"energy", "hybrid"} else zero
        )
        if args.physics_loss == "residual":
            physics_loss = residual_loss
        elif args.physics_loss == "energy":
            physics_loss = energy_loss
        else:
            physics_loss = args.residual_weight * residual_loss + args.energy_weight * energy_loss
        stability_loss = (
            directional_stability_loss(
                model, physics, batch,
                directions=args.stability_directions,
                margin=args.stability_margin,
                generator=stability_generator,
            )
            if args.stability_weight > 0.0 else zero
        )
        path_loss = zero
        if args.path_consistency_weight > 0.0:
            reduced_path = model.predict_reduced_correction(path_batch).reshape(path_points, len(batch), -1)
            # At zero thermal gradient the correction is exactly zero.  The
            # finite-difference term then selects a smooth quasi-static path
            # rather than independent stable basins at each load level.
            anchor = torch.mean(reduced_path[0].square())
            increments = reduced_path[1:] - reduced_path[:-1]
            smoothness = torch.mean(increments.square())
            path_loss = anchor + smoothness
        data_loss = torch.zeros((), dtype=torch.float64, device=device)
        progress = max(0.0, (step / args.steps - args.curriculum_ramp_start) / (1.0 - args.curriculum_ramp_start))
        physics_weight = args.physics_weight_start + progress * (args.physics_weight_end - args.physics_weight_start)
        fidelity_weight = fidelity_weight_start + progress * (fidelity_weight_end - fidelity_weight_start)
        if fidelity_cases is not None and fidelity_targets is not None and fidelity_weight > 0.0:
            # Full label batches are deterministic.  Random sampling with
            # replacement is useful only when the requested batch is smaller.
            data_index = (
                torch.arange(len(fidelity_cases), device=device)
                if args.batch_size >= len(fidelity_cases)
                else torch.randint(len(fidelity_cases), (args.batch_size,), generator=fidelity_generator, device=device)
            )
            data_batch = fidelity_cases.index_select(data_index)
            predicted = model.predict_curvatures(data_batch)
            target = fidelity_targets.index_select(0, data_index)
            if args.fidelity_residual_targets:
                predicted = predicted - physics.curvatures(physics.linear_initial_q(data_batch))
                assert fidelity_residual_scale is not None
                data_loss = torch.mean(((predicted - target) / fidelity_residual_scale).square())
            else:
                data_loss = torch.mean(((predicted - target) / CURVATURE_SCALE).square())
        loss = (
            physics_weight * physics_loss
            + args.stability_weight * stability_loss
            + args.path_consistency_weight * path_loss
            + fidelity_weight * data_loss
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss. Reduce --learning-rate or --correction-amplitude.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            model.eval()
            validation_energy, validation_residual = validation_metrics(model, physics, validation_cases)
            history.append(
                {
                    "step": step,
                    "total_loss": float(loss.detach().cpu()),
                    "physics_loss": float(physics_loss.detach().cpu()),
                    "data_loss": float(data_loss.detach().cpu()),
                    "stability_loss": float(stability_loss.detach().cpu()),
                    "path_loss": float(path_loss.detach().cpu()),
                    "validation_energy_delta_j": validation_energy,
                    "validation_projected_gradient": validation_residual,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "continuation_factor": continuation_factor,
                    "physics_weight": physics_weight,
                    "fidelity_weight": fidelity_weight,
                }
            )
            print(
                f"step {step:6d} | load {continuation_factor:.3f} | loss {loss.item(): .4e} | physics {physics_loss.item(): .4e} (w={physics_weight:.3g}) "
                f"| stable {stability_loss.item(): .4e} | path {path_loss.item(): .4e} | data {data_loss.item(): .4e} (w={fidelity_weight:.3g}) | val dU {validation_energy: .4e} J "
                f"| val ||dU/dr|| {validation_residual: .4e}"
            )
            model.train()

    if args.lbfgs_steps:
        print(f"Starting full-batch L-BFGS energy polish ({args.lbfgs_steps} iterations maximum).", flush=True)
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=0.5,
            max_iter=args.lbfgs_steps,
            max_eval=args.lbfgs_steps * 2,
            tolerance_grad=1.0e-10,
            tolerance_change=1.0e-13,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def energy_closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            loss = energy_difference_loss(model, physics, train_cases)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite energy during L-BFGS polish.")
            loss.backward()
            return loss

        lbfgs.step(energy_closure)
        model.eval()
        validation_energy, validation_residual = validation_metrics(model, physics, validation_cases)
        polished_loss = energy_difference_loss(model, physics, train_cases)
        history.append(
            {
                "step": args.steps + args.lbfgs_steps,
                "total_loss": float(polished_loss.detach().cpu()),
                "physics_loss": float(polished_loss.detach().cpu()),
                "data_loss": 0.0,
                "stability_loss": 0.0,
                "path_loss": 0.0,
                "validation_energy_delta_j": validation_energy,
                "validation_projected_gradient": validation_residual,
                "learning_rate": 0.0,
                "continuation_factor": 1.0,
            }
        )
        print(
            f"L-BFGS | energy {validation_energy:.4e} J | mean ||dU/dr|| {validation_residual:.4e}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    training_metadata = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    training_metadata["run_id"] = run_id
    checkpoint = {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "model_metadata": model.checkpoint_metadata(),
        "training": training_metadata,
        "temperature_definition": "T(z)=(T_top+T_bottom)/2+(T_top-T_bottom)*z/h",
        "layup_order": "bottom_to_top",
    }
    torch.save(checkpoint, args.output)
    print(f"Saved checkpoint: {args.output}")
    report_prefix = args.report_prefix or args.output.with_suffix("")
    history_csv, history_png = write_training_history(history, report_prefix)
    print(f"Training history: {history_csv}")
    print(f"Training plot:    {history_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
