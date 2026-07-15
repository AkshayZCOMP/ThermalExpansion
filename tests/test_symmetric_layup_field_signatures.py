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

Edit PLY_COUNTS_TO_TEST and MAX_LAYUPS_TO_SEARCH below to control the search.
PLY_COUNTS_TO_TEST is the only place to edit ply counts.
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


# ============================================================================
# USER SETTINGS - edit these two lines for the layup search
# ============================================================================

# Add/remove ply counts here.
PLY_COUNTS_TO_TEST = (3, 4, 5, 6, 10, 12)

# Maximum layups to search per ply count. Use None for uncapped exhaustive runs.
MAX_LAYUPS_TO_SEARCH = 10000

# ============================================================================

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
DEFAULT_PAIRWISE_DISTANCE_SAMPLES = 10000
PLOT_FILE = Path(__file__).with_name("layup_field_signature_boxplot.png")
FIELD_SPREAD_PLOT_FILE = Path(__file__).with_name("layup_field_component_spread.png")
FIELD_SPREAD_REPORT_FILE = Path(__file__).with_name("layup_field_component_closest_pairs.txt")
FIELD_COMPARISON_PLOT_FILE = Path(__file__).with_name("layup_field_pair_comparison.png")
FIELD_COMPONENT_NAMES = ("eps_x", "eps_y", "gamma_xy", "kappa_x", "kappa_y", "kappa_xy")
VERSIONED_OUTPUT_STEMS = (
    "layup_field_signature_boxplot",
    "layup_field_component_spread",
    "layup_field_component_closest_pairs",
    "layup_field_pair_comparison",
)


def selected_ply_counts():
    return PLY_COUNTS_TO_TEST


def validate_user_settings():
    if not PLY_COUNTS_TO_TEST:
        raise ValueError("PLY_COUNTS_TO_TEST must contain at least one ply count.")

    invalid_ply_counts = [
        ply_count
        for ply_count in PLY_COUNTS_TO_TEST
        if not isinstance(ply_count, int) or ply_count < 2
    ]
    if invalid_ply_counts:
        raise ValueError(
            "PLY_COUNTS_TO_TEST values must be integers greater than or equal to 2. "
            f"Invalid values: {invalid_ply_counts}"
        )

    if MAX_LAYUPS_TO_SEARCH is not None:
        if (
            not isinstance(MAX_LAYUPS_TO_SEARCH, int)
            or MAX_LAYUPS_TO_SEARCH < 1
        ):
            raise ValueError(
                "MAX_LAYUPS_TO_SEARCH must be a positive integer or None."
            )


def symmetric_base_length(ply_count):
    return ply_count // 2 + (ply_count % 2)


def symmetric_layup_count(ply_count):
    return len(ANGLE_POOL) ** symmetric_base_length(ply_count)


def capped_layup_count(total_count, max_layups):
    if max_layups is None:
        return total_count
    return min(total_count, max_layups)


def next_output_version():
    """Return the next shared v-number for plot/report artifacts."""
    output_dir = Path(__file__).parent
    used_versions = set()

    for stem in VERSIONED_OUTPUT_STEMS:
        for path in output_dir.glob(f"{stem}_v*.*"):
            version_text = path.stem.rsplit("_v", 1)[-1]
            if version_text.isdigit():
                used_versions.add(int(version_text))

    version = 1
    while version in used_versions:
        version += 1
    return version


def versioned_output_paths(version):
    """Return plot/report paths sharing one version identifier."""
    output_dir = Path(__file__).parent
    return {
        "signature_plot": output_dir / f"layup_field_signature_boxplot_v{version}.png",
        "component_plot": output_dir / f"layup_field_component_spread_v{version}.png",
        "closest_report": output_dir / f"layup_field_component_closest_pairs_v{version}.txt",
        "comparison_plot": output_dir / f"layup_field_pair_comparison_v{version}.png",
    }


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


def find_signature_collisions(ply_count, max_layups=MAX_LAYUPS_TO_SEARCH):
    """
    Exhaustively search one ply count for indistinguishable transient fields.

    Returns a list of collision dictionaries. Each collision has the previous
    layup, current layup, and maximum absolute signature difference.
    """
    temperature_profiles = transient_temperature_profiles(ply_count)
    signatures_by_bucket = {}
    collisions = []
    total_count = symmetric_layup_count(ply_count)
    limit = capped_layup_count(total_count, max_layups)

    print(
        f"Searching {ply_count}-ply layups: {limit:,} of {total_count:,}",
        flush=True,
    )

    for index, layup in enumerate(iter_all_symmetric_layups(ply_count), start=1):
        if max_layups is not None and index > max_layups:
            break

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


def final_field_components_for_angles(angles):
    """Return final-time [microstrain, curvature] values for one explicit layup."""
    layup = LayupSequence(list(angles), name=f"Comparison layup {angles}")
    laminate, analysis = analysis_for_layup(layup)
    temperature_profiles = transient_temperature_profiles(layup.num_plies)
    final_temperature_profile = temperature_profiles[-1]
    diffusion = ThermalDiffusion(
        laminate,
        analysis,
        top_surface_temp=TOP_SURFACE_TEMP,
        initial_temp=INITIAL_TEMP,
        thermal_diffusivity=THERMAL_DIFFUSIVITY,
        n_nodes=N_NODES,
    )
    strain, curvature = diffusion._compute_strains_from_profile(final_temperature_profile)
    return np.concatenate((strain * 1e6, curvature))


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
    ply_counts=PLY_COUNTS_TO_TEST,
    max_layups=MAX_LAYUPS_TO_SEARCH,
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
    ply_counts=PLY_COUNTS_TO_TEST,
    max_layups=MAX_LAYUPS_TO_SEARCH,
    output_path=FIELD_SPREAD_PLOT_FILE,
    report_path=FIELD_SPREAD_REPORT_FILE,
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

    report_path.write_text(format_closest_pair_report(report_rows), encoding="utf-8")

    return output_path, report_path


def plot_layup_pair_comparison(
    ply_count=4,
    close_component="eps_x",
    max_layups=MAX_LAYUPS_TO_SEARCH,
    output_path=FIELD_COMPARISON_PLOT_FILE,
):
    """
    Compare two layup pairs across all six scalar field components.

    Pair 1 is chosen because it is closest in one requested scalar component.
    Pair 2 is a reference contrast between all-0 and all-90 degree laminates.
    Component differences are normalized by the searched result-space range
    for that ply count.
    """
    import matplotlib.pyplot as plt

    if close_component not in FIELD_COMPONENT_NAMES:
        raise ValueError(f"Unknown component: {close_component}")

    total_count = symmetric_layup_count(ply_count)
    limit = capped_layup_count(total_count, max_layups)
    angles, strains, curvatures = collect_field_components(ply_count, max_layups=limit)
    component_values = np.column_stack((strains * 1e6, curvatures))
    component_min = np.min(component_values, axis=0)
    component_max = np.max(component_values, axis=0)
    component_range = np.maximum(component_max - component_min, SCALAR_ZERO_TOL)

    closest_pairs = closest_component_pairs(angles, strains, curvatures)
    close_pair = next(
        pair for pair in closest_pairs if pair["component"] == close_component
    )
    close_angles_a = close_pair["first_angles"]
    close_angles_b = close_pair["second_angles"]

    reference_angles_a = tuple([0] * ply_count)
    reference_angles_b = tuple([90] * ply_count)
    pair_specs = (
        (
            f"Closest in {close_component}",
            close_angles_a,
            close_angles_b,
        ),
        (
            "Reference: all 0 vs all 90",
            reference_angles_a,
            reference_angles_b,
        ),
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex="col")
    x_positions = np.arange(len(FIELD_COMPONENT_NAMES))
    width = 0.38

    for row_index, (title, first_angles, second_angles) in enumerate(pair_specs):
        first_values = final_field_components_for_angles(first_angles)
        second_values = final_field_components_for_angles(second_angles)
        normalized_first = (first_values - component_min) / component_range
        normalized_second = (second_values - component_min) / component_range
        normalized_difference = np.abs(first_values - second_values) / component_range

        value_ax = axes[row_index, 0]
        difference_ax = axes[row_index, 1]

        value_ax.bar(
            x_positions - width / 2,
            normalized_first,
            width=width,
            label=str(first_angles),
        )
        value_ax.bar(
            x_positions + width / 2,
            normalized_second,
            width=width,
            label=str(second_angles),
        )
        value_ax.set_title(title)
        value_ax.set_ylabel("Normalized component value")
        value_ax.set_ylim(-0.05, 1.05)
        value_ax.grid(axis="y", alpha=0.3)
        value_ax.legend(fontsize=7, loc="best")

        difference_ax.bar(x_positions, normalized_difference, color="#4c78a8")
        difference_ax.set_title("Absolute difference by component")
        difference_ax.set_ylabel("Difference / searched range")
        difference_ax.grid(axis="y", alpha=0.3)

        for component_index, difference in enumerate(normalized_difference):
            difference_ax.text(
                component_index,
                difference,
                f"{difference:.2g}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    for ax in axes[-1, :]:
        ax.set_xticks(x_positions)
        ax.set_xticklabels(FIELD_COMPONENT_NAMES, rotation=30, ha="right")

    fig.suptitle(
        f"{ply_count}-ply layup field comparison "
        f"(searched {len(angles):,} of {total_count:,} layups)"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


class TestSymmetricLayupFieldSignatures(unittest.TestCase):
    """Exhaustive tests for one-to-one symmetric layup field signatures."""

    def test_user_settings_are_valid(self):
        validate_user_settings()

    def test_all_possible_symmetric_layups_have_unique_transient_fields(self):
        validate_user_settings()
        failures = []

        for ply_count in selected_ply_counts():
            with self.subTest(ply_count=ply_count):
                collisions = find_signature_collisions(
                    ply_count,
                    max_layups=MAX_LAYUPS_TO_SEARCH,
                )
                if collisions:
                    failures.append((ply_count, symmetric_layup_count(ply_count), collisions))

        if failures:
            self.fail(format_collision_report(failures))

    def test_reversed_two_ply_laminates_have_same_ad_different_transient_response(self):
        """
        Reversing [0, 90] to [90, 0] preserves A and D but flips B.

        This is the simplest laminate pair for showing why transient thermal
        response depends on ply order, not only angle inventory. The full ABD
        matrices are not identical because these two-ply laminates are
        unsymmetric, but their extensional and bending stiffness blocks match.
        """
        layup_a = LayupSequence([0, 90], name="[0/90]")
        layup_b = LayupSequence([90, 0], name="[90/0]")

        laminate_a, analysis_a = analysis_for_layup(layup_a)
        laminate_b, analysis_b = analysis_for_layup(layup_b)

        np.testing.assert_allclose(
            analysis_a.A_matrix,
            analysis_b.A_matrix,
            rtol=1e-12,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            analysis_a.D_matrix,
            analysis_b.D_matrix,
            rtol=1e-12,
            atol=1e-16,
        )
        np.testing.assert_allclose(
            analysis_a.B_matrix,
            -analysis_b.B_matrix,
            rtol=1e-12,
            atol=1e-12,
        )

        diffusion_a = ThermalDiffusion(
            laminate_a,
            analysis_a,
            top_surface_temp=TOP_SURFACE_TEMP,
            initial_temp=INITIAL_TEMP,
            thermal_diffusivity=THERMAL_DIFFUSIVITY,
            n_nodes=N_NODES,
        )
        diffusion_b = ThermalDiffusion(
            laminate_b,
            analysis_b,
            top_surface_temp=TOP_SURFACE_TEMP,
            initial_temp=INITIAL_TEMP,
            thermal_diffusivity=THERMAL_DIFFUSIVITY,
            n_nodes=N_NODES,
        )

        t_final = 2.0 * diffusion_a.estimate_diffusion_time_scale()
        temperatures = diffusion_a.solve_transient(
            t_final=t_final,
            n_steps=N_STEPS,
            bottom_temp=BOTTOM_SURFACE_TEMP,
        )["T"]

        response_a = []
        response_b = []
        for temperature_profile in temperatures:
            strain_a, curvature_a = diffusion_a._compute_strains_from_profile(
                temperature_profile
            )
            strain_b, curvature_b = diffusion_b._compute_strains_from_profile(
                temperature_profile
            )
            response_a.append(np.concatenate((strain_a, curvature_a)))
            response_b.append(np.concatenate((strain_b, curvature_b)))

        response_a = np.array(response_a)
        response_b = np.array(response_b)

        self.assertGreater(
            np.max(np.abs(response_a - response_b)),
            1e-8,
            msg="Reversing [0/90] should change the transient thermal response.",
        )

    def test_symmetric_qi_order_change_preserves_a_but_changes_d_and_transient_response(self):
        """
        Compare [0/45/-45/90]s and [90/-45/45/0]s.

        These laminates have the same angle inventory and both are symmetric,
        so their extensional stiffness A is the same and B is approximately
        zero. They do not have the same D matrix because bending stiffness is
        weighted by distance from the midplane; moving 0-degree plies from the
        outside to the inside changes the bending response.
        """
        layup_1 = LayupSequence(
            [0, 45, -45, 90, 90, -45, 45, 0],
            name="[0/45/-45/90]s",
        )
        layup_2 = LayupSequence(
            [90, -45, 45, 0, 0, 45, -45, 90],
            name="[90/-45/45/0]s",
        )

        laminate_1, analysis_1 = analysis_for_layup(layup_1)
        laminate_2, analysis_2 = analysis_for_layup(layup_2)

        np.testing.assert_allclose(
            analysis_1.A_matrix,
            analysis_2.A_matrix,
            rtol=1e-12,
            atol=1e-8,
        )
        self.assertLess(np.max(np.abs(analysis_1.B_matrix)), 1e-6)
        self.assertLess(np.max(np.abs(analysis_2.B_matrix)), 1e-6)
        self.assertGreater(
            np.max(np.abs(analysis_1.D_matrix - analysis_2.D_matrix)),
            1.0,
            msg="Changing symmetric ply order should change bending stiffness D.",
        )

        diffusion_1 = ThermalDiffusion(
            laminate_1,
            analysis_1,
            top_surface_temp=TOP_SURFACE_TEMP,
            initial_temp=INITIAL_TEMP,
            thermal_diffusivity=THERMAL_DIFFUSIVITY,
            n_nodes=N_NODES,
        )
        diffusion_2 = ThermalDiffusion(
            laminate_2,
            analysis_2,
            top_surface_temp=TOP_SURFACE_TEMP,
            initial_temp=INITIAL_TEMP,
            thermal_diffusivity=THERMAL_DIFFUSIVITY,
            n_nodes=N_NODES,
        )

        temperatures = transient_temperature_profiles(layup_1.num_plies)
        response_differences = []
        for temperature_profile in temperatures:
            strain_1, curvature_1 = diffusion_1._compute_strains_from_profile(
                temperature_profile
            )
            strain_2, curvature_2 = diffusion_2._compute_strains_from_profile(
                temperature_profile
            )
            response_differences.append(
                np.max(
                    np.abs(
                        np.concatenate((strain_1, curvature_1))
                        - np.concatenate((strain_2, curvature_2))
                    )
                )
            )

        self.assertGreater(
            max(response_differences),
            1e-8,
            msg="Different symmetric ply order should change transient response.",
        )


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


def generate_versioned_outputs():
    """Generate all plots/reports for the configured ply counts."""
    version = next_output_version()
    paths = versioned_output_paths(version)
    plot_counts = selected_ply_counts()

    print(
        f"Generating v{version} plots/reports for ply counts {plot_counts} "
        f"with max layups {MAX_LAYUPS_TO_SEARCH}...",
        flush=True,
    )

    plot_path = plot_result_spaces(
        ply_counts=plot_counts,
        max_layups=MAX_LAYUPS_TO_SEARCH,
        pairwise_samples=int(
            os.getenv(
                "LAYUP_FIELD_PAIRWISE_DISTANCE_SAMPLES",
                DEFAULT_PAIRWISE_DISTANCE_SAMPLES,
            )
        ),
        output_path=paths["signature_plot"],
    )
    spread_plot_path, closest_pair_report_path = plot_field_component_spread(
        ply_counts=plot_counts,
        max_layups=MAX_LAYUPS_TO_SEARCH,
        output_path=paths["component_plot"],
        report_path=paths["closest_report"],
    )
    comparison_plot_path = plot_layup_pair_comparison(
        ply_count=int(os.getenv("LAYUP_FIELD_COMPARISON_PLY_COUNT", "4")),
        close_component=os.getenv("LAYUP_FIELD_COMPARISON_COMPONENT", "eps_x"),
        max_layups=MAX_LAYUPS_TO_SEARCH,
        output_path=paths["comparison_plot"],
    )

    print(f"Saved v{version} result-space box plot to: {plot_path}")
    print(f"Saved v{version} field-component spread plot to: {spread_plot_path}")
    print(f"Saved v{version} closest-component-pair report to: {closest_pair_report_path}")
    print(f"Saved v{version} layup-pair comparison plot to: {comparison_plot_path}")


if __name__ == "__main__":
    validate_user_settings()
    if "--plot" in sys.argv:
        sys.argv.remove("--plot")
        generate_versioned_outputs()
    else:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(
            TestSymmetricLayupFieldSignatures
        )
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        generate_versioned_outputs()
        sys.exit(0 if result.wasSuccessful() else 1)
