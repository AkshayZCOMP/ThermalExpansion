"""
Exhaustive collision tests for symmetric-layup transient field signatures.

The inverse-prediction problem needs a one-to-one map from measured strain and
curvature fields back to a layup. These tests enumerate every symmetric layup
for tractable ply counts and flag distinct layups whose transient thermal
strain/curvature histories are numerically indistinguishable.

By default, the exhaustive unit test covers 3, 4, 5, and 6 plies. A 6-ply
laminate with canonical 15-degree angles from -75 to 90 has 12^3 = 1728
symmetric layups. Larger cases grow quickly:

    7/8 plies:   12^4 = 20,736 layups each
    10 plies:    12^5 = 248,832 layups
    12 plies:    12^6 = 2,985,984 layups
    14 plies:    12^7 = 35,831,808 layups
    16 plies:    12^8 = 429,981,696 layups

To run a larger exhaustive case deliberately:

    set LAYUP_FIELD_PLY_COUNTS=8
    python tests/test_symmetric_layup_field_signatures.py
"""

import itertools
import os
import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.InputProperties import LaminateProperties, LayupSequence, MaterialProperties
from src.ThermalDiffusion import ThermalDiffusion
from src.workflow import LaminateAnalysis


REQUESTED_PLY_COUNTS = (3, 4, 5, 6, 7, 8, 10, 12, 14, 16)
DEFAULT_EXHAUSTIVE_PLY_COUNTS = (3, 4, 5, 6)
ANGLE_STEP = 15
# Use a canonical orientation set. In CLT, +90 and -90 describe the same
# fiber direction, so including both would create false inverse collisions.
ANGLE_POOL = tuple(range(-75, 91, ANGLE_STEP))

TOP_SURFACE_TEMP = 100.0
BOTTOM_SURFACE_TEMP = 25.0
INITIAL_TEMP = 25.0
THERMAL_DIFFUSIVITY = 1e-6
N_NODES = 50
N_STEPS = 9

SIGNATURE_ATOL = 1e-12
SIGNATURE_RTOL = 1e-9
SCALAR_ZERO_TOL = 1e-12
MAX_COLLISIONS_TO_REPORT = 20
DEFAULT_PLOT_PLY_COUNTS = (3, 4, 5, 6)
DEFAULT_PLOT_MAX_LAYUPS = 200000
DEFAULT_SIGNATURE_PLOT_MAX_LAYUPS = 5000
DEFAULT_PAIRWISE_DISTANCE_SAMPLES = 50000
PLOT_FILE = Path(__file__).with_name("layup_field_signature_boxplot.png")
FIELD_SPREAD_PLOT_FILE = Path(__file__).with_name("layup_field_component_spread.png")
FIELD_SPREAD_REPORT_FILE = Path(__file__).with_name("layup_field_component_closest_pairs.txt")
FIELD_COMPONENT_NAMES = ("eps_x", "eps_y", "gamma_xy", "kappa_x", "kappa_y", "kappa_xy")


def selected_ply_counts():
    raw_value = os.getenv("LAYUP_FIELD_PLY_COUNTS")
    if not raw_value:
        return DEFAULT_EXHAUSTIVE_PLY_COUNTS

    selected = tuple(int(value.strip()) for value in raw_value.split(",") if value.strip())
    unsupported = set(selected) - set(REQUESTED_PLY_COUNTS)
    if unsupported:
        raise ValueError(f"Unsupported ply counts requested: {sorted(unsupported)}")
    return selected


def symmetric_base_length(ply_count):
    return ply_count // 2 + (ply_count % 2)


def symmetric_layup_count(ply_count):
    return len(ANGLE_POOL) ** symmetric_base_length(ply_count)


def capped_layup_count(total_count, max_layups):
    if max_layups is None:
        return total_count
    return min(total_count, max_layups)


def make_symmetric_layup(base_angles, total_plies):
    """Return a symmetric layup with exactly total_plies plies."""
    half_count = total_plies // 2
    expected_base_length = symmetric_base_length(total_plies)
    if len(base_angles) != expected_base_length:
        raise ValueError("base angle count does not match requested ply count")

    first_half = list(base_angles[:half_count])
    if total_plies % 2:
        angles = first_half + [base_angles[-1]] + first_half[::-1]
    else:
        angles = first_half + first_half[::-1]

    return LayupSequence(angles, name=f"Symmetric {total_plies}-ply")


def iter_all_symmetric_layups(ply_count):
    """Yield every symmetric layup sequence for the requested ply count."""
    for base_angles in itertools.product(ANGLE_POOL, repeat=symmetric_base_length(ply_count)):
        yield make_symmetric_layup(base_angles, ply_count)


def analysis_for_layup(layup):
    laminate = LaminateProperties(
        MaterialProperties(),
        layup,
        ply_thickness=0.125e-3,
        delta_T=0.0,
    )
    analysis = LaminateAnalysis(laminate)
    analysis.run_full_analysis()
    return laminate, analysis


def transient_temperature_profiles(ply_count):
    """Build the deterministic temperature history used for every layup."""
    layup = make_symmetric_layup([0] * symmetric_base_length(ply_count), ply_count)
    laminate, analysis = analysis_for_layup(layup)
    diffusion = ThermalDiffusion(
        laminate,
        analysis,
        top_surface_temp=TOP_SURFACE_TEMP,
        initial_temp=INITIAL_TEMP,
        thermal_diffusivity=THERMAL_DIFFUSIVITY,
        n_nodes=N_NODES,
    )

    t_final = 2.0 * diffusion.estimate_diffusion_time_scale()
    return diffusion.solve_transient(
        t_final=t_final,
        n_steps=N_STEPS,
        bottom_temp=BOTTOM_SURFACE_TEMP,
    )["T"]


def transient_field_signature(layup, temperature_profiles):
    """
    Return strain, curvature, and ply strain histories for a transient test.

    The signature includes:
    - membrane strain at every sampled time
    - curvature at every sampled time
    - strain at each ply midpoint at every sampled time
    """
    laminate, analysis = analysis_for_layup(layup)
    diffusion = ThermalDiffusion(
        laminate,
        analysis,
        top_surface_temp=TOP_SURFACE_TEMP,
        initial_temp=INITIAL_TEMP,
        thermal_diffusivity=THERMAL_DIFFUSIVITY,
        n_nodes=N_NODES,
    )

    z_midpoints = np.linspace(
        -laminate.total_thickness / 2 + laminate.ply_thickness / 2,
        laminate.total_thickness / 2 - laminate.ply_thickness / 2,
        layup.num_plies,
    )

    signature_parts = []
    for temperature_profile in temperature_profiles:
        strain, curvature = diffusion._compute_strains_from_profile(temperature_profile)
        ply_strain_field = np.array([strain + z * curvature for z in z_midpoints])
        signature_parts.extend((strain, curvature, ply_strain_field.ravel()))

    return np.concatenate(signature_parts)


def signature_bucket(signature):
    """
    Quantize a signature so exact duplicate candidates can be found cheaply.

    Candidate groups are still checked with np.allclose before being reported.
    """
    return tuple(np.round(signature / SIGNATURE_ATOL).astype(np.int64))


def find_signature_collisions(ply_count):
    """
    Exhaustively search one ply count for indistinguishable transient fields.

    Returns a list of collision dictionaries. Each collision has the previous
    layup, current layup, and maximum absolute signature difference.
    """
    temperature_profiles = transient_temperature_profiles(ply_count)
    signatures_by_bucket = {}
    collisions = []

    for index, layup in enumerate(iter_all_symmetric_layups(ply_count), start=1):
        signature = transient_field_signature(layup, temperature_profiles)
        bucket = signature_bucket(signature)

        for previous_angles, previous_signature in signatures_by_bucket.get(bucket, []):
            if np.allclose(
                signature,
                previous_signature,
                rtol=SIGNATURE_RTOL,
                atol=SIGNATURE_ATOL,
            ):
                collisions.append(
                    {
                        "previous_angles": previous_angles,
                        "current_angles": tuple(layup.angles),
                        "max_abs_difference": float(
                            np.max(np.abs(signature - previous_signature))
                        ),
                    }
                )
                if len(collisions) >= MAX_COLLISIONS_TO_REPORT:
                    return collisions

        signatures_by_bucket.setdefault(bucket, []).append((tuple(layup.angles), signature))

    return collisions


def collect_signatures(ply_count, max_layups=None):
    """Collect transient field signatures for plotting or post-processing."""
    temperature_profiles = transient_temperature_profiles(ply_count)
    signatures = []
    angles = []

    for index, layup in enumerate(iter_all_symmetric_layups(ply_count), start=1):
        if max_layups is not None and index > max_layups:
            break

        signatures.append(transient_field_signature(layup, temperature_profiles))
        angles.append(tuple(layup.angles))

    return angles, np.array(signatures)


def collect_field_components(ply_count, max_layups=None):
    """
    Collect final-time membrane strains and curvatures for every searched layup.

    This is used for physical result-space plots. Unlike the full signature,
    these six values are easy to interpret directly:
    [eps_x, eps_y, gamma_xy] and [kappa_x, kappa_y, kappa_xy].
    """
    temperature_profiles = transient_temperature_profiles(ply_count)
    final_temperature_profile = temperature_profiles[-1]
    strains = []
    curvatures = []
    angles = []

    for index, layup in enumerate(iter_all_symmetric_layups(ply_count), start=1):
        if max_layups is not None and index > max_layups:
            break

        laminate, analysis = analysis_for_layup(layup)
        diffusion = ThermalDiffusion(
            laminate,
            analysis,
            top_surface_temp=TOP_SURFACE_TEMP,
            initial_temp=INITIAL_TEMP,
            thermal_diffusivity=THERMAL_DIFFUSIVITY,
            n_nodes=N_NODES,
        )
        strain, curvature = diffusion._compute_strains_from_profile(
            final_temperature_profile
        )

        angles.append(tuple(layup.angles))
        strains.append(strain)
        curvatures.append(curvature)

    return angles, np.array(strains), np.array(curvatures)


def normalized_pairwise_distance_samples(signatures, max_samples):
    """
    Return deterministic samples from the pairwise normalized distance space.

    For small sets this computes all pairwise distances. For larger sets it
    samples index pairs deterministically so the plot remains reproducible.
    """
    sample_count = len(signatures)
    if sample_count < 2:
        return np.array([])

    total_pairs = sample_count * (sample_count - 1) // 2
    distances = []

    if total_pairs <= max_samples:
        pair_iter = itertools.combinations(range(sample_count), 2)
    else:
        rng = np.random.default_rng(0)
        pair_iter = (
            tuple(rng.choice(sample_count, size=2, replace=False))
            for _ in range(max_samples)
        )

    for first_index, second_index in pair_iter:
        first = signatures[first_index]
        second = signatures[second_index]
        scale = max(np.linalg.norm(first), np.linalg.norm(second), 1.0)
        distances.append(np.linalg.norm(first - second) / scale)

    return np.array(distances)


def plot_result_spaces(
    ply_counts=DEFAULT_PLOT_PLY_COUNTS,
    max_layups=DEFAULT_SIGNATURE_PLOT_MAX_LAYUPS,
    pairwise_samples=DEFAULT_PAIRWISE_DISTANCE_SAMPLES,
    output_path=PLOT_FILE,
):
    """
    Save box plots of the result-space distribution for each ply count.

    The plotted quantity is normalized pairwise distance between transient
    field signatures. Larger values mean layups are more separated in the
    strain/curvature result space. Values near zero indicate likely inverse
    ambiguity.
    """
    import matplotlib.pyplot as plt

    labels = []
    distance_sets = []

    for ply_count in ply_counts:
        total_count = symmetric_layup_count(ply_count)
        limit = capped_layup_count(total_count, max_layups)
        _, signatures = collect_signatures(ply_count, max_layups=limit)
        distances = normalized_pairwise_distance_samples(signatures, pairwise_samples)

        labels.append(f"{ply_count} plies\nn={len(signatures):,}")
        distance_sets.append(distances)

    fig, ax = plt.subplots(figsize=(10, 5))
    try:
        ax.boxplot(distance_sets, tick_labels=labels, showfliers=False)
    except TypeError:
        ax.boxplot(distance_sets, labels=labels, showfliers=False)
    ax.set_title("Symmetric layup transient-field result-space separation")
    ax.set_xlabel("Ply count")
    ax.set_ylabel("Normalized pairwise signature distance")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def closest_component_pairs(angles, strains, curvatures):
    """
    Return the closest layup pair for each final-time scalar output.

    Strain components are reported in microstrain; curvature components are
    reported in 1/m. The closest pair for a scalar component may have zero
    distance even when full six-component signatures remain distinguishable.
    """
    component_values = np.column_stack((strains * 1e6, curvatures))
    closest_pairs = []

    for component_index, component_name in enumerate(FIELD_COMPONENT_NAMES):
        values = component_values[:, component_index]
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        differences = np.abs(np.diff(sorted_values))

        if len(differences) == 0:
            continue

        closest_difference_index = int(np.argmin(differences))
        first_index = int(sorted_indices[closest_difference_index])
        second_index = int(sorted_indices[closest_difference_index + 1])
        nonzero_difference_indices = np.where(differences > SCALAR_ZERO_TOL)[0]

        closest_nonzero = None
        if len(nonzero_difference_indices):
            nonzero_local_index = int(
                nonzero_difference_indices[
                    np.argmin(differences[nonzero_difference_indices])
                ]
            )
            nonzero_first_index = int(sorted_indices[nonzero_local_index])
            nonzero_second_index = int(sorted_indices[nonzero_local_index + 1])
            closest_nonzero = {
                "distance": float(differences[nonzero_local_index]),
                "first_angles": angles[nonzero_first_index],
                "second_angles": angles[nonzero_second_index],
                "first_value": float(values[nonzero_first_index]),
                "second_value": float(values[nonzero_second_index]),
            }

        closest_pairs.append(
            {
                "component": component_name,
                "distance": float(differences[closest_difference_index]),
                "first_angles": angles[first_index],
                "second_angles": angles[second_index],
                "first_value": float(values[first_index]),
                "second_value": float(values[second_index]),
                "closest_nonzero": closest_nonzero,
            }
        )

    return closest_pairs


def format_closest_pair_report(report_rows):
    lines = [
        "Closest final-time scalar component pairs by ply count",
        "",
        "Strain components are in microstrain. Curvature components are in 1/m.",
        "A zero scalar distance does not necessarily mean the full field signature collides.",
    ]

    for row in report_rows:
        lines.append("")
        lines.append(
            f"{row['ply_count']}-ply searched {row['searched_count']:,} "
            f"of {row['total_count']:,} layups"
        )
        for pair in row["closest_pairs"]:
            lines.append(
                "  "
                f"{pair['component']}: distance={pair['distance']:.12e}, "
                f"values=({pair['first_value']:.12e}, {pair['second_value']:.12e})"
            )
            lines.append(f"    {pair['first_angles']}")
            lines.append(f"    {pair['second_angles']}")
            if pair["closest_nonzero"] is not None:
                nonzero = pair["closest_nonzero"]
                lines.append(
                    "    "
                    f"closest nonzero distance={nonzero['distance']:.12e}, "
                    f"values=({nonzero['first_value']:.12e}, "
                    f"{nonzero['second_value']:.12e})"
                )
                lines.append(f"      {nonzero['first_angles']}")
                lines.append(f"      {nonzero['second_angles']}")

    return "\n".join(lines) + "\n"


def plot_field_component_spread(
    ply_counts=DEFAULT_PLOT_PLY_COUNTS,
    max_layups=DEFAULT_PLOT_MAX_LAYUPS,
    output_path=FIELD_SPREAD_PLOT_FILE,
):
    """
    Save box plots of actual final-time strain and curvature result ranges.

    Strains are plotted in microstrain. Curvatures are plotted in 1/m. This
    shows the physical size of the search space for each ply count.
    """
    import matplotlib.pyplot as plt

    strain_component_names = ("eps_x", "eps_y", "gamma_xy")
    curvature_component_names = ("kappa_x", "kappa_y", "kappa_xy")
    labels = []
    strain_by_ply_count = []
    curvature_by_ply_count = []
    report_rows = []

    for ply_count in ply_counts:
        total_count = symmetric_layup_count(ply_count)
        limit = capped_layup_count(total_count, max_layups)
        angles, strains, curvatures = collect_field_components(ply_count, max_layups=limit)

        labels.append(f"{ply_count} plies\nn={len(strains):,}")
        strain_by_ply_count.append(strains * 1e6)
        curvature_by_ply_count.append(curvatures)
        report_rows.append(
            {
                "ply_count": ply_count,
                "searched_count": len(strains),
                "total_count": total_count,
                "closest_pairs": closest_component_pairs(angles, strains, curvatures),
            }
        )

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)

    for component_index, component_name in enumerate(strain_component_names):
        data = [values[:, component_index] for values in strain_by_ply_count]
        ax = axes[0, component_index]
        try:
            ax.boxplot(data, tick_labels=labels, showfliers=False)
        except TypeError:
            ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(component_name)
        ax.set_ylabel("microstrain")
        ax.grid(axis="y", alpha=0.3)

    for component_index, component_name in enumerate(curvature_component_names):
        data = [values[:, component_index] for values in curvature_by_ply_count]
        ax = axes[1, component_index]
        try:
            ax.boxplot(data, tick_labels=labels, showfliers=False)
        except TypeError:
            ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(component_name)
        ax.set_ylabel("1/m")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Symmetric layup final-time strain and curvature search space")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    report_path = FIELD_SPREAD_REPORT_FILE
    report_path.write_text(format_closest_pair_report(report_rows), encoding="utf-8")

    return output_path, report_path


class TestSymmetricLayupFieldSignatures(unittest.TestCase):
    """Exhaustive tests for one-to-one symmetric layup field signatures."""

    def test_requested_ply_counts_are_known(self):
        self.assertEqual(REQUESTED_PLY_COUNTS, (3, 4, 5, 6, 7, 8, 10, 12, 14, 16))

    def test_all_possible_symmetric_layups_have_unique_transient_fields(self):
        failures = []

        for ply_count in selected_ply_counts():
            with self.subTest(ply_count=ply_count):
                collisions = find_signature_collisions(ply_count)
                if collisions:
                    failures.append((ply_count, symmetric_layup_count(ply_count), collisions))

        if failures:
            self.fail(format_collision_report(failures))


def format_collision_report(failures):
    lines = [
        "Found symmetric layups with indistinguishable transient strain/curvature fields.",
        f"Criterion: np.allclose(signature_a, signature_b, "
        f"rtol={SIGNATURE_RTOL:g}, atol={SIGNATURE_ATOL:g})",
    ]

    for ply_count, total_count, collisions in failures:
        lines.append("")
        lines.append(f"{ply_count}-ply exhaustive search over {total_count:,} layups:")
        for collision in collisions:
            lines.append(
                "  "
                f"{collision['previous_angles']} == {collision['current_angles']} "
                f"(max abs diff {collision['max_abs_difference']:.3e})"
            )

    return "\n".join(lines)


def parse_ply_count_list(raw_value):
    return tuple(int(value.strip()) for value in raw_value.split(",") if value.strip())


def parse_layup_cap(raw_value, default_value):
    if raw_value is None:
        return default_value

    cleaned_value = raw_value.strip().lower()
    if cleaned_value in ("", "default"):
        return default_value
    if cleaned_value in ("all", "none", "uncapped", "0"):
        return None

    parsed_value = int(cleaned_value)
    if parsed_value < 0:
        raise ValueError("Layup cap must be positive, 0, or ALL")
    return parsed_value


if __name__ == "__main__":
    if "--plot" in sys.argv:
        sys.argv.remove("--plot")
        raw_plot_counts = os.getenv("LAYUP_FIELD_PLOT_PLY_COUNTS")
        plot_counts = (
            parse_ply_count_list(raw_plot_counts)
            if raw_plot_counts
            else DEFAULT_PLOT_PLY_COUNTS
        )
        component_plot_max_layups = parse_layup_cap(
            os.getenv("LAYUP_FIELD_PLOT_MAX_LAYUPS"),
            DEFAULT_PLOT_MAX_LAYUPS,
        )
        signature_plot_max_layups = parse_layup_cap(
            os.getenv("LAYUP_FIELD_SIGNATURE_PLOT_MAX_LAYUPS"),
            DEFAULT_SIGNATURE_PLOT_MAX_LAYUPS,
        )
        plot_path = plot_result_spaces(
            ply_counts=plot_counts,
            max_layups=signature_plot_max_layups,
            pairwise_samples=int(
                os.getenv(
                    "LAYUP_FIELD_PAIRWISE_DISTANCE_SAMPLES",
                    DEFAULT_PAIRWISE_DISTANCE_SAMPLES,
                )
            ),
        )
        spread_plot_path, closest_pair_report_path = plot_field_component_spread(
            ply_counts=plot_counts,
            max_layups=component_plot_max_layups,
        )
        print(f"Saved result-space box plot to: {plot_path}")
        print(f"Saved field-component spread plot to: {spread_plot_path}")
        print(f"Saved closest-component-pair report to: {closest_pair_report_path}")
    else:
        unittest.main(verbosity=2)
