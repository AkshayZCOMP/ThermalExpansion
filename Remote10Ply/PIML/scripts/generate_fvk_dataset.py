"""Create optional sparse nonlinear labels with the existing FREEFREE solver.

This is intentionally separate from PINN training.  Physics-only training
does not invoke the slow solver at all; a small, diverse label set can be
added later to select the same equilibrium branch as the current numerical
workflow in difficult post-buckling regions.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import torch

from ..core.physics import (
    ROOT,
    LayupBatch,
    T_PLY,
    all_ply_temperature_field_grid_cases,
    fixed_gradient_cases,
    gradient_grid_cases,
    temperature_field_grid_cases,
    layup_string,
    sample_cases,
)
from ..core.run_paths import add_run_id_arguments, resolve_run_id, timestamped_file


FREEFREE_PATH = ROOT / "Computational Real" / "FREEFREE.py"
_WORKER_SOLVER = None


def load_freefree():
    spec = importlib.util.spec_from_file_location("piml_freefree", FREEFREE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {FREEFREE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def batch_layup(batch: LayupBatch, row: int) -> list[int]:
    lookup = (-45, 0, 45, 90)
    nplies = int(batch.nplies[row])
    return [lookup[int(value)] for value in batch.angle_indices[row, :nplies]]


def _worker_initializer() -> None:
    """Give each subprocess an independent FREEFREE module/global state."""
    global _WORKER_SOLVER
    _WORKER_SOLVER = load_freefree()


def _solve_freefree_case(payload: tuple[int, list[int], float, float, int]) -> tuple[int, dict[str, object]]:
    """Solve one direct-face-temperature case; safe to execute in a worker."""
    global _WORKER_SOLVER
    row, layup, top, bottom, transverse_degree = payload
    if _WORKER_SOLVER is None:
        _worker_initializer()
    solver = _WORKER_SOLVER
    assert solver is not None
    return _solve_with_solver(payload, solver)


def _solve_with_solver(
    payload: tuple[int, list[int], float, float, int],
    solver,
) -> tuple[int, dict[str, object]]:
    """Solve one case using the supplied isolated FREEFREE module instance."""
    row, layup, top, bottom, transverse_degree = payload
    thickness = len(layup) * T_PLY
    solver.configure_case(
        layup,
        top_sink_temperature=top,
        bottom_sink_temperature=bottom,
        transverse_degree=transverse_degree,
        midplane_temperature=0.5 * (top + bottom),
        thermal_gradient=(top - bottom) / thickness,
    )
    solution = solver.solve_default_equilibrium(verbose=False)
    result = solution["solver"]
    if not result.converged and "precision loss" in str(result.message).lower():
        polished = _polish_precision_limited_solution(solver, solution, result)
        if polished is not None:
            result = polished
    eps, kappa = solver.area_averaged_generalized_quantities(result.q)
    return row, {
        "layup": layup_string(layup),
        "nplies": len(layup),
        "top_face_temperature": f"{top:.12g}",
        "bottom_face_temperature": f"{bottom:.12g}",
        "eps_x": f"{eps[0]:.12e}",
        "eps_y": f"{eps[1]:.12e}",
        "gamma_xy": f"{eps[2]:.12e}",
        "kappa_x": f"{kappa[0]:.12e}",
        "kappa_y": f"{kappa[1]:.12e}",
        "kappa_xy": f"{kappa[2]:.12e}",
        "converged": result.converged,
        "iterations": result.iterations,
        "message": result.message,
    }


def _polish_precision_limited_solution(solver, solution, initial_result):
    """Refine a BFGS precision-loss endpoint without weakening convergence.

    ``root`` solves the same analytic projected-gradient equations from the
    BFGS endpoint.  The result is accepted only when FREEFREE's original
    absolute/relative gradient tests pass and its energy did not increase.
    """
    try:
        from scipy.optimize import root
    except ImportError:  # pragma: no cover - SciPy is a project dependency
        return None

    nullspace = solution["nullspace"]
    reduced_initial = nullspace.T @ initial_result.q

    def gradient(reduced):
        _, full_gradient = solver.total_internal_energy_and_gradient_q(nullspace @ reduced)
        return nullspace.T @ full_gradient

    polished = root(
        gradient,
        reduced_initial,
        method="hybr",
        options={"xtol": 1.0e-10, "maxfev": 2000},
    )
    reduced = polished.x
    final_gradient = gradient(reduced)
    gradient_norm = float((final_gradient @ final_gradient) ** 0.5)
    relative_gradient_norm = gradient_norm / max(float(initial_result.initial_gradient_norm), 1.0)
    final_energy = float(solver.total_internal_energy_q(nullspace @ reduced))
    energy_tolerance = 1.0e-10 * max(1.0, abs(float(initial_result.energy)))
    converged = (
        math.isfinite(final_energy)
        and math.isfinite(gradient_norm)
        and final_energy <= float(initial_result.energy) + energy_tolerance
        and (
            gradient_norm < float(solver.GRAD_TOL)
            or relative_gradient_norm < float(solver.REL_GRAD_TOL)
        )
    )
    if not converged:
        return None
    return solver.SolverResult(
        q=nullspace @ reduced,
        energy=final_energy,
        projected_gradient=final_gradient,
        converged=True,
        iterations=int(initial_result.iterations) + int(getattr(polished, "nfev", 0)),
        message=(
            "converged after projected-gradient root polish; initial SciPy exit: "
            f"{initial_result.message}"
        ),
        method="scipy-bfgs+root-polish",
        initial_gradient_norm=float(initial_result.initial_gradient_norm),
    )


def _solve_freefree_case_isolated(payload: tuple[int, list[int], float, float, int]) -> tuple[int, dict[str, object]]:
    """Thread-safe fallback: each task imports an isolated FREEFREE module."""
    return _solve_with_solver(payload, load_freefree())


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate sparse nonlinear FREEFREE labels for PIML calibration.")
    parser.add_argument("--cases", type=int, default=64)
    parser.add_argument("--temperature-min", type=float, default=0.0)
    parser.add_argument("--temperature-max", type=float, default=100.0)
    parser.add_argument(
        "--minimum-face-temperature-difference",
        type=float,
        default=1.0,
        help="Require |T_top - T_bottom| to be at least this value [C].",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--transverse-degree", type=int, default=5)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent FREEFREE solver processes (1 preserves serial behavior).",
    )
    parser.add_argument("--nplies", type=int, help="Restrict fixed-gradient cases to one ply count.")
    parser.add_argument("--all-ply", action="store_true", help="Use every canonical layup for each temperature field; combine with --nplies to select one ply count.")
    parser.add_argument("--case-start", type=int, default=0, help="Inclusive flattened canonical-case index for a deterministic chunk.")
    parser.add_argument("--case-stop", type=int, help="Exclusive flattened canonical-case index for a deterministic chunk.")
    parser.add_argument("--midplane-temperature", type=float, help="Use with --thermal-gradient for T(z)=T_mid+beta*z.")
    parser.add_argument(
        "--midplane-temperatures",
        help="Comma-separated T_mid values [C], crossed with --thermal-gradients.",
    )
    parser.add_argument("--thermal-gradient", type=float, help="Use with --midplane-temperature for T(z)=T_mid+beta*z [C/m].")
    parser.add_argument(
        "--thermal-gradients",
        help="Comma-separated nonzero gradients [C/m]; --cases then means layups per gradient.",
    )
    parser.add_argument("--output", type=Path, default=Path("PIML/data/fvk_samples.csv"))
    add_run_id_arguments(parser)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append only deterministic cases missing from an existing output CSV.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    args.output = timestamped_file(args.output, resolve_run_id(args))
    if args.cases <= 0 or args.workers <= 0:
        raise ValueError("--cases and --workers must be positive.")
    if args.case_start < 0:
        raise ValueError("--case-start cannot be negative.")
    gradient_grid_requested = args.thermal_gradients is not None
    midplane_grid_requested = args.midplane_temperatures is not None
    temperature_field_grid_requested = gradient_grid_requested and midplane_grid_requested
    fixed_gradient_requested = (
        args.midplane_temperature is not None
        or args.thermal_gradient is not None
        or gradient_grid_requested
        or midplane_grid_requested
    )
    if midplane_grid_requested and not gradient_grid_requested:
        raise ValueError("--midplane-temperatures requires --thermal-gradients.")
    if fixed_gradient_requested and args.midplane_temperature is None and not midplane_grid_requested:
        raise ValueError("--midplane-temperature is required for a fixed-gradient FREEFREE set.")
    if fixed_gradient_requested and not gradient_grid_requested and args.thermal_gradient is None:
        raise ValueError("--thermal-gradient is required unless --thermal-gradients is supplied.")
    if gradient_grid_requested and args.thermal_gradient is not None:
        raise ValueError("Use either --thermal-gradient or --thermal-gradients, not both.")
    if fixed_gradient_requested and args.nplies is None and not args.all_ply:
        raise ValueError("--nplies is required for a fixed-gradient FREEFREE set unless --all-ply is used.")
    if args.overwrite and args.resume:
        raise ValueError("Use either --overwrite or --resume, not both.")
    if args.output.exists() and not (args.overwrite or args.resume):
        raise FileExistsError(f"{args.output} exists; use --overwrite to replace it or --resume to append missing cases.")
    if fixed_gradient_requested:
        chunk_requested = args.case_start != 0 or args.case_stop is not None
        if temperature_field_grid_requested:
            gradients = tuple(float(value.strip()) for value in args.thermal_gradients.split(",") if value.strip())
            if not gradients or any(value == 0.0 for value in gradients):
                raise ValueError("--thermal-gradients requires one or more nonzero values.")
            midplanes = tuple(float(value.strip()) for value in args.midplane_temperatures.split(",") if value.strip())
            if not midplanes:
                raise ValueError("--midplane-temperatures requires one or more values.")
            fields = tuple((midplane, gradient) for midplane in midplanes for gradient in gradients)
            all_cases = (
                all_ply_temperature_field_grid_cases(fields)
                if args.all_ply else temperature_field_grid_cases(
                    fields, nplies=args.nplies, count_per_field=args.cases, seed=args.seed,
                )
            )
            if args.all_ply and args.nplies is not None:
                indices = torch.nonzero(all_cases.nplies == args.nplies, as_tuple=False).flatten()
                if len(indices) == 0:
                    raise ValueError(f"No canonical {args.nplies}-ply cases exist.")
                all_cases = all_cases.index_select(indices)
            if chunk_requested:
                stop = len(all_cases) if args.case_stop is None else args.case_stop
                if stop <= args.case_start or stop > len(all_cases):
                    raise ValueError(f"Temperature-field case range must lie within [0, {len(all_cases)}].")
                indices = torch.arange(args.case_start, stop, dtype=torch.long)
                cases = all_cases.index_select(indices)
            else:
                cases = all_cases
        elif gradient_grid_requested:
            gradients = tuple(float(value.strip()) for value in args.thermal_gradients.split(",") if value.strip())
            if not gradients or any(value == 0.0 for value in gradients):
                raise ValueError("--thermal-gradients requires one or more nonzero values.")
            cases = gradient_grid_cases(
                gradients,
                nplies=args.nplies,
                count_per_gradient=args.cases,
                seed=args.seed,
                midplane_temperature=args.midplane_temperature,
            )
        elif chunk_requested:
            all_cases = fixed_gradient_cases(
                args.nplies,
                midplane_temperature=args.midplane_temperature,
                thermal_gradient=args.thermal_gradient,
            )
            stop = len(all_cases) if args.case_stop is None else args.case_stop
            if stop <= args.case_start or stop > len(all_cases):
                raise ValueError(f"Fixed-gradient case range must lie within [0, {len(all_cases)}].")
            indices = torch.arange(args.case_start, stop, dtype=torch.long)
            cases = all_cases.index_select(indices)
        else:
            cases = fixed_gradient_cases(
                args.nplies,
                count=args.cases,
                seed=args.seed,
                midplane_temperature=args.midplane_temperature,
                thermal_gradient=args.thermal_gradient,
            )
    else:
        cases = sample_cases(
            args.cases,
            temperature_min=args.temperature_min,
            temperature_max=args.temperature_max,
            seed=args.seed,
            minimum_face_temperature_difference=args.minimum_face_temperature_difference,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "layup", "nplies", "top_face_temperature", "bottom_face_temperature",
        "eps_x", "eps_y", "gamma_xy", "kappa_x", "kappa_y", "kappa_xy",
        "converged", "iterations", "message",
    ]
    payloads = [
        (
            row,
            batch_layup(cases, row),
            float(cases.top_face_temperature[row]),
            float(cases.bottom_face_temperature[row]),
            args.transverse_degree,
        )
        for row in range(len(cases))
    ]
    existing_keys: set[tuple[str, float, float]] = set()
    if args.resume and args.output.exists():
        with args.output.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"layup", "top_face_temperature", "bottom_face_temperature"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"Cannot resume malformed dataset {args.output}.")
            for record in reader:
                existing_keys.add((
                    record["layup"],
                    float(record["top_face_temperature"]),
                    float(record["bottom_face_temperature"]),
                ))
        payloads = [
            payload for payload in payloads
            if (layup_string(payload[1]), payload[2], payload[3]) not in existing_keys
        ]
        print(f"Resuming {args.output}: {len(existing_keys)} existing records; {len(payloads)} cases remain.", flush=True)
    executor = None
    if args.workers == 1:
        solved_cases = map(_solve_freefree_case, payloads)
    else:
        try:
            executor = ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_initializer)
            solved_cases = executor.map(_solve_freefree_case, payloads)
        except PermissionError:
            # Some restricted Windows environments disallow the anonymous pipes
            # required by ProcessPoolExecutor. Use isolated module instances in
            # a thread pool instead; map() still preserves canonical ordering.
            print("Process FREEFREE workers are unavailable here; using isolated threaded workers.", flush=True)
            executor = ThreadPoolExecutor(max_workers=args.workers)
            solved_cases = executor.map(_solve_freefree_case_isolated, payloads)

    try:
        mode = "a" if args.resume and args.output.exists() else "w"
        with args.output.open(mode, newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if mode == "w":
                writer.writeheader()
            for completed, (row, record) in enumerate(solved_cases, start=1):
                writer.writerow(record)
                layup = record["layup"]
                print(
                    f"{completed:4d}/{len(payloads)} {layup:<34} "
                    f"top={float(record['top_face_temperature']):7.2f} "
                    f"bottom={float(record['bottom_face_temperature']):7.2f} "
                    f"{'OK' if record['converged'] else 'NOT CONVERGED'}",
                    flush=True,
                )
                handle.flush()
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
