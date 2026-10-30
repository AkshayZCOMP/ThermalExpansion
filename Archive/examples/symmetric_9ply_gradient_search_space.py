"""
Plot the response search space for symmetric 9-ply layups under a 12 C gradient.

The angle pool is restricted to 0, +/-45, and 90 degrees. A symmetric 9-ply
laminate is defined by five independent angles, so the full search space is
4^5 = 1,024 layups, below the 20,000 sample cap.
"""

import itertools
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Archive.src.InputProperties import LaminateProperties, LayupSequence, MaterialProperties
from Archive.src.ThermalDiffusion import ThermalDiffusion
from Archive.src.workflow import LaminateAnalysis


ANGLE_POOL = (0, 45, -45, 90)
PLY_COUNT = 9
MAX_SAMPLES = 20_000
PLY_THICKNESS = 0.125e-3
INITIAL_TEMP = 25.0
THROUGH_THICKNESS_DELTA_T = 12.0
N_NODES = 50
THERMAL_DIFFUSIVITY = 1e-6

OUTPUT_DIR = ROOT / "examples"
PLOT_PATH = OUTPUT_DIR / "symmetric_9ply_12deg_gradient_search_space.png"
CSV_PATH = OUTPUT_DIR / "symmetric_9ply_12deg_gradient_search_space.csv"

FIELD_COMPONENTS = ("eps_x", "eps_y", "gamma_xy", "kappa_x", "kappa_y", "kappa_xy")


def symmetric_base_length(ply_count):
    return ply_count // 2 + (ply_count % 2)


def make_symmetric_layup(base_angles):
    half_count = PLY_COUNT // 2
    first_half = list(base_angles[:half_count])
    return first_half + [base_angles[-1]] + first_half[::-1]


def iter_symmetric_angle_sets():
    base_count = symmetric_base_length(PLY_COUNT)
    all_bases = itertools.product(ANGLE_POOL, repeat=base_count)
    for sample_index, base_angles in enumerate(all_bases):
        if sample_index >= MAX_SAMPLES:
            break
        yield make_symmetric_layup(base_angles)


def field_components_for_layup(angles, temperature_profile):
    layup = LayupSequence(list(angles), name="Symmetric 9-ply")
    laminate = LaminateProperties(
        MaterialProperties(),
        layup,
        ply_thickness=PLY_THICKNESS,
        delta_T=0.0,
    )
    analysis = LaminateAnalysis(laminate)
    analysis.run_full_analysis()
    diffusion = ThermalDiffusion(
        laminate,
        analysis,
        top_surface_temp=INITIAL_TEMP + THROUGH_THICKNESS_DELTA_T,
        initial_temp=INITIAL_TEMP,
        thermal_diffusivity=THERMAL_DIFFUSIVITY,
        n_nodes=N_NODES,
    )
    strain, curvature = diffusion._compute_strains_from_profile(temperature_profile)
    return np.concatenate((strain * 1e6, curvature))


def collect_search_space():
    temperature_profile = np.linspace(
        INITIAL_TEMP + THROUGH_THICKNESS_DELTA_T,
        INITIAL_TEMP,
        N_NODES,
    )
    angle_sets = []
    components = []

    for angles in iter_symmetric_angle_sets():
        angle_sets.append(tuple(angles))
        components.append(field_components_for_layup(angles, temperature_profile))

    return angle_sets, np.array(components)


def save_csv(angle_sets, components):
    header = [f"ply_{index + 1}" for index in range(PLY_COUNT)]
    header.extend(FIELD_COMPONENTS)

    rows = []
    for angles, values in zip(angle_sets, components):
        rows.append([*angles, *values])

    np.savetxt(
        CSV_PATH,
        np.array(rows, dtype=float),
        delimiter=",",
        header=",".join(header),
        comments="",
        fmt=["%g"] * PLY_COUNT + ["%.12e"] * len(FIELD_COMPONENTS),
    )


def plot_search_space(components):
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))

    for component_index, component_name in enumerate(FIELD_COMPONENTS):
        ax = axes.flat[component_index]
        values = components[:, component_index]
        ax.boxplot([values], widths=0.45, showfliers=False)
        jitter = rng.normal(loc=1.0, scale=0.035, size=len(values))
        ax.scatter(
            jitter,
            values,
            s=11,
            alpha=0.35,
            color="#2f6f9f",
            edgecolors="none",
        )
        ax.set_title(component_name)
        ax.set_xticks([1])
        ax.set_xticklabels([f"n={len(values):,}"])
        ax.grid(axis="y", alpha=0.3)

        if component_index < 3:
            ax.set_ylabel("microstrain")
        else:
            ax.set_ylabel("1/m")

    fig.suptitle(
        "Symmetric 9-ply search space, angle pool {0, +/-45, 90}, "
        "12 C through-thickness gradient"
    )
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=200)
    plt.close(fig)


def main():
    total_count = len(ANGLE_POOL) ** symmetric_base_length(PLY_COUNT)
    sampled_count = min(total_count, MAX_SAMPLES)
    print(
        f"Evaluating {sampled_count:,} of {total_count:,} symmetric "
        f"{PLY_COUNT}-ply layups."
    )
    angle_sets, components = collect_search_space()
    save_csv(angle_sets, components)
    plot_search_space(components)
    print(f"Saved plot: {PLOT_PATH}")
    print(f"Saved data: {CSV_PATH}")


if __name__ == "__main__":
    main()
