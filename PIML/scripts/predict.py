"""Run the trained PINN, or the exact linear-CLT baseline, for one laminate."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..core.model import RitzPINN
from ..core.physics import FVKPhysics, make_batch, parse_layup


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict free-free laminate curvatures.")
    parser.add_argument("--layup", required=True, help="Bottom-to-top layup, e.g. '[0/45/-45/90]'.")
    parser.add_argument("--top-face-temperature", type=float, required=True, help="Top laminate face temperature [C].")
    parser.add_argument("--bottom-face-temperature", type=float, required=True, help="Bottom laminate face temperature [C].")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint created by python -m PIML.scripts.train.")
    parser.add_argument("--linear-clt", action="store_true", help="Do not load a network; return the exact linear-CLT baseline.")
    parser.add_argument("--device", default="cpu")
    return parser


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[FVKPhysics, RitzPINN]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    metadata = checkpoint["model_metadata"]
    physics = FVKPhysics(
        transverse_degree=int(metadata["transverse_degree"]),
        area_quad_n=int(metadata["area_quad_n"]),
    ).to(device)
    model = RitzPINN(
        physics,
        max_plies=int(metadata["max_plies"]),
        hidden_width=int(metadata["hidden_width"]),
        hidden_layers=int(metadata["hidden_layers"]),
        temperature_scale=float(metadata["temperature_scale"]),
        correction_amplitude=float(metadata["correction_amplitude"]),
        temperature_feature_mode=str(metadata.get("temperature_feature_mode", "faces")),
        gradient_residual_expert=bool(metadata.get("gradient_residual_expert", False)),
        gradient_expert_width=int(metadata.get("gradient_expert_width", 96)),
        enforce_axis_aligned_reflection_symmetry=bool(
            metadata.get("enforce_axis_aligned_reflection_symmetry", True)
        ),
        curvature_trust_region=bool(metadata.get("curvature_trust_region", False)),
        curvature_relative_bound=float(metadata.get("curvature_relative_bound", 0.10)),
        curvature_absolute_bound=float(metadata.get("curvature_absolute_bound", 0.02)),
        fixed_linear_curvature_components=tuple(metadata.get("fixed_linear_curvature_components", (False, False, False))),
        thermal_odd_symmetry=bool(metadata.get("thermal_odd_symmetry", False)),
        layup_d4_equivariance=bool(metadata.get("layup_d4_equivariance", False)),
        split_offaxis_experts=bool(metadata.get("split_offaxis_experts", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return physics, model


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.linear_clt == (args.checkpoint is not None):
        raise ValueError("Specify exactly one of --checkpoint or --linear-clt.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    layup = parse_layup(args.layup)
    batch = make_batch([layup], [args.top_face_temperature], [args.bottom_face_temperature]).to(device)
    if args.linear_clt:
        physics = FVKPhysics().to(device)
        q = physics.linear_initial_q(batch)
        label = "Exact linear-CLT baseline"
    else:
        physics, model = load_model(args.checkpoint, device)
        with torch.no_grad():
            q = model(batch)
        label = f"PINN: {args.checkpoint}"
    with torch.no_grad():
        generalized = physics.generalized_average(q)[0].cpu()
    print(label)
    print(f"Layup (bottom -> top): {layup}")
    print(f"Top / bottom face temperature: {args.top_face_temperature:.6g} / {args.bottom_face_temperature:.6g} C")
    print("Area-averaged response")
    print(f"  eps_x    = {generalized[0]: .8e}")
    print(f"  eps_y    = {generalized[1]: .8e}")
    print(f"  gamma_xy = {generalized[2]: .8e}")
    print(f"  kappa_x  = {generalized[3]: .8e} 1/m")
    print(f"  kappa_y  = {generalized[4]: .8e} 1/m")
    print(f"  kappa_xy = {generalized[5]: .8e} 1/m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
