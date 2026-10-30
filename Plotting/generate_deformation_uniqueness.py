"""Summarize layup-specific deformation uniqueness in the main sweep results.

The six computational response components are used because they all come from
the 130 GPa FREEFREE solver.  Each component is normalized by its spread among
the layups with the same ply count, so strains and curvatures contribute
comparably to the distance calculation.
"""

from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "AbaqusSweep" / "ABaqusData_with_computational_results.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
CSV_OUTPUT = OUTPUT_DIR / "ply_count_deformation_uniqueness.csv"
MARKDOWN_OUTPUT = OUTPUT_DIR / "ply_count_deformation_uniqueness.md"
PLOT_OUTPUT = OUTPUT_DIR / "ply_count_deformation_uniqueness.png"
TABLE_OUTPUT = OUTPUT_DIR / "ply_count_deformation_uniqueness_table.png"
CLOSEST_PAIRS_OUTPUT = OUTPUT_DIR / "closest_layup_deformation_components.csv"
CURVATURE_SEPARATION_OUTPUT = OUTPUT_DIR / "curvature_space_pairwise_separation_summary.csv"

STRAIN_COLUMNS = ["Comp_e0_11", "Comp_e0_22", "Comp_e0_12"]
CURVATURE_COLUMNS = ["Comp_K_11", "Comp_K_22", "Comp_K_12"]
RESPONSE_COLUMNS = STRAIN_COLUMNS + CURVATURE_COLUMNS
UNIQUENESS_TOLERANCE = 1.0e-8


def parse_layup(layup):
    """Parse the project's ``[0/45/-45]`` layup representation."""
    return tuple(float(angle) for angle in layup.strip()[1:-1].split("/"))


def format_layup(angles):
    return "[" + "/".join("%g" % angle for angle in angles) + "]"


def rotate_angle_90(angle):
    """Apply a 90-degree in-plane rotation, using the conventional -45/90 form."""
    rotated = (angle + 90.0) % 180.0
    if np.isclose(rotated, 135.0):
        return -45.0
    return rotated


def canonical_layup(layup):
    """Return a layup's rotation-invariant key and rotation to that key.

    A 180-degree fibre-direction change is identical for a lamina; therefore
    the only distinct in-plane orientation partner in this sweep is +90 deg.
    """
    original = parse_layup(layup)
    rotated = tuple(rotate_angle_90(angle) for angle in original)
    if rotated < original:
        return format_layup(rotated), 90
    return format_layup(original), 0


def rotate_response_90(response):
    """Rotate [e11, e22, gamma12, K11, K22, K12] by +90 degrees."""
    rotated = response.copy()
    rotated[0], rotated[1], rotated[2] = response[1], response[0], -response[2]
    rotated[3], rotated[4], rotated[5] = response[4], response[3], -response[5]
    return rotated


def rotation_canonical_group(raw_group):
    """Deduplicate 90-degree-equivalent layups in a common response frame."""
    exact_group = raw_group.drop_duplicates("Layup", keep="first").copy()
    keys_and_rotations = exact_group["Layup"].map(canonical_layup)
    exact_group["Canonical layup"] = [item[0] for item in keys_and_rotations]
    exact_group["Rotation to canonical frame (deg)"] = [item[1] for item in keys_and_rotations]

    response = exact_group[RESPONSE_COLUMNS].to_numpy(dtype=float, copy=True)
    rotate_rows = exact_group["Rotation to canonical frame (deg)"].to_numpy() == 90
    for index in np.flatnonzero(rotate_rows):
        response[index] = rotate_response_90(response[index])
    exact_group.loc[:, RESPONSE_COLUMNS] = response

    group = exact_group.drop_duplicates("Canonical layup", keep="first").reset_index(drop=True)
    group["Layup"] = group["Canonical layup"]
    return group, len(exact_group) - len(group), int(raw_group.duplicated("Layup", keep=False).sum())


def pairwise_distances(values):
    """Return every pairwise Euclidean distance and its row indices."""
    return [
        (float(np.linalg.norm(values[left] - values[right])), left, right)
        for left, right in combinations(range(len(values)), 2)
    ]


def minimum_distance(values):
    return min(distance for distance, _, _ in pairwise_distances(values))


def curvature_space_separation_summary(raw_group):
    """Summarize all unique layup-pair distances in raw curvature 3-D space."""
    group, rotational_duplicates, _ = rotation_canonical_group(raw_group)
    distances = np.asarray(
        [distance for distance, _, _ in pairwise_distances(
            group[CURVATURE_COLUMNS].to_numpy(dtype=float)
        )],
        dtype=float,
    )
    return {
        "Ply count": int(group["NumberOfPlies"].iloc[0]),
        "Rotation-distinct layups": len(group),
        "Rotational equivalents removed": rotational_duplicates,
        "Layup pairs": len(distances),
        "Minimum curvature-space separation (m^-1)": float(distances.min()),
        "Median curvature-space separation (m^-1)": float(np.median(distances)),
        "Mean curvature-space separation (m^-1)": float(distances.mean()),
    }


def summarize_group(raw_group):
    group, rotational_duplicates, duplicate_row_count = rotation_canonical_group(raw_group)

    response = group[RESPONSE_COLUMNS].to_numpy(dtype=float, copy=True)
    # Report strain in microstrain before range-normalization for better
    # conditioning; this does not alter the final normalized distances.
    response[:, :3] *= 1.0e6
    component_spans = np.ptp(response, axis=0)
    component_spans[component_spans == 0.0] = 1.0
    normalized = response / component_spans

    full_pairs = sorted(pairwise_distances(normalized))
    nearest_neighbor = np.full(len(group), np.inf)
    for distance, left, right in full_pairs:
        nearest_neighbor[left] = min(nearest_neighbor[left], distance)
        nearest_neighbor[right] = min(nearest_neighbor[right], distance)

    minimum_full, left, right = full_pairs[0]
    unique_deformations = len(group) - sum(
        distance <= UNIQUENESS_TOLERANCE for distance, _, _ in full_pairs
    )
    return {
        "Ply count": int(group["NumberOfPlies"].iloc[0]),
        "Dataset rows": len(raw_group),
        "Rotation-distinct layups": len(group),
        "Repeated input rows": duplicate_row_count,
        "Rotational equivalents removed": rotational_duplicates,
        "Unique 6D deformations": unique_deformations,
        "Minimum 6D separation": minimum_full,
        "Median nearest-neighbor separation": float(np.median(nearest_neighbor)),
        "Minimum strain-only separation": minimum_distance(normalized[:, :3]),
        "Minimum curvature-only separation": minimum_distance(normalized[:, 3:]),
        "Closest rotation-distinct layup A": group.loc[left, "Layup"],
        "Closest rotation-distinct layup B": group.loc[right, "Layup"],
    }


def closest_pair_components(raw_group):
    """Return full responses for the closest pair in raw curvature 3-D space.

    Pair selection uses the same coordinates and units as the curvature-space
    scatter: ``Comp_K_11``, ``Comp_K_22``, and engineering ``Comp_K_12``, all
    in m^-1. Strains are included as supplementary values in microstrain.
    """
    group, _, _ = rotation_canonical_group(raw_group)
    response = group[RESPONSE_COLUMNS].to_numpy(dtype=float)
    curvature = response[:, 3:]
    curvature_distance, left, right = min(pairwise_distances(curvature))

    rows = []
    for label, index in (("A", left), ("B", right)):
        rows.append({
            "Ply count": int(group.loc[index, "NumberOfPlies"]),
            "Pair member": label,
            "Curvature-space distance (m^-1)": curvature_distance,
            "Layup": group.loc[index, "Layup"],
            "e0_11 (microstrain)": response[index, 0] * 1.0e6,
            "e0_22 (microstrain)": response[index, 1] * 1.0e6,
            "e0_12 (microstrain)": response[index, 2] * 1.0e6,
            "K_11 (m^-1)": response[index, 3],
            "K_22 (m^-1)": response[index, 4],
            "K_12 (m^-1, engineering)": response[index, 5],
        })
    return rows


def write_markdown(summary):
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            "| {ply} | {layups} | {unique} | {minimum:.3e} | {median:.3e} | {strain:.3e} | {curvature:.3e} |".format(
                ply=row["Ply count"], layups=row["Rotation-distinct layups"],
                unique=row["Unique 6D deformations"], minimum=row["Minimum 6D separation"],
                median=row["Median nearest-neighbor separation"],
                strain=row["Minimum strain-only separation"],
                curvature=row["Minimum curvature-only separation"],
            )
        )
    MARKDOWN_OUTPUT.write_text(
        "# Deformation uniqueness by ply count\n\n"
        "Source: 130 GPa computational response columns (`Comp_e0_*`, `Comp_K_*`) "
        "in the main comparison dataset. Distances are Euclidean distances after "
        "normalizing each of the six response components by its within-ply-count span. "
        "Layups related by a 90-degree in-plane rotation are first collapsed into "
        "one rotation-distinct design and their responses are transformed into a "
        "common coordinate frame. "
        "A six-component distance above `1e-8` is treated as numerically distinct.\n\n"
        "| Plies | Rotation-distinct layups | Unique 6D responses | Minimum 6D separation | "
        "Median nearest neighbor | Minimum strain-only separation | Minimum curvature-only separation |\n"
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        + "\n".join(rows)
        + "\n\nStrain-only separation can be nearly zero for layups with the same "
        "in-plane angle totals. Their curvature components still yield a distinct "
        "six-component deformation vector.\n",
        encoding="utf-8",
    )


def make_plot(summary):
    plies = summary["Ply count"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")

    axes[0].plot(
        plies, summary["Minimum 6D separation"], marker="o", linewidth=2,
        label="Minimum pairwise separation",
    )
    axes[0].plot(
        plies, summary["Median nearest-neighbor separation"], marker="s",
        linewidth=2, label="Median nearest neighbor",
    )
    axes[0].axhline(
        UNIQUENESS_TOLERANCE, color="black", linestyle="--", linewidth=1,
        label="Numerical uniqueness threshold",
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Number of plies")
    axes[0].set_ylabel("Range-normalized 6D deformation distance")
    axes[0].set_title("(a) Full strain + curvature response")
    axes[0].set_xticks(plies)
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=8)

    width = 0.25
    axes[1].bar(
        plies - width, summary["Minimum 6D separation"], width,
        label="Full 6D response",
    )
    axes[1].bar(
        plies, summary["Minimum strain-only separation"], width,
        label="Strain only",
    )
    axes[1].bar(
        plies + width, summary["Minimum curvature-only separation"], width,
        label="Curvature only",
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Number of plies")
    axes[1].set_ylabel("Minimum range-normalized separation")
    axes[1].set_title("(b) What resolves the closest layups")
    axes[1].set_xticks(plies)
    axes[1].grid(True, axis="y", which="both", alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.suptitle("Every rotation-distinct layup has a numerically unique deformation response")
    fig.savefig(PLOT_OUTPUT, dpi=220)
    plt.close(fig)


def make_table_image(summary):
    display_columns = [
        "Ply count", "Rotation-distinct layups", "Unique 6D deformations",
        "Minimum 6D separation", "Median nearest-neighbor separation",
    ]
    table = summary[display_columns].copy()
    table = table.rename(columns={
        "Minimum 6D separation": "Min. 6D separation",
        "Median nearest-neighbor separation": "Median NN separation",
    })
    for column in ("Min. 6D separation", "Median NN separation"):
        table[column] = table[column].map("{:.3e}".format)
    figure, axis = plt.subplots(figsize=(11.5, 0.43 * len(table) + 1.2))
    axis.axis("off")
    rendered = axis.table(
        cellText=table.values,
        colLabels=table.columns,
        cellLoc="center",
        loc="center",
    )
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(8)
    rendered.scale(1, 1.35)
    axis.set_title("Numerical uniqueness of computational deformation responses", pad=12)
    figure.savefig(TABLE_OUTPUT, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main():
    data = pd.read_csv(INPUT)
    missing = [column for column in RESPONSE_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError("Missing computational response columns: %s" % missing)
    summary = pd.DataFrame(
        summarize_group(group)
        for _, group in data.groupby("NumberOfPlies", sort=True)
    )
    summary.to_csv(CSV_OUTPUT, index=False)
    closest_pairs = pd.DataFrame(
        row
        for _, group in data.groupby("NumberOfPlies", sort=True)
        for row in closest_pair_components(group)
    )
    closest_pairs.to_csv(CLOSEST_PAIRS_OUTPUT, index=False)
    curvature_separations = pd.DataFrame(
        curvature_space_separation_summary(group)
        for _, group in data.groupby("NumberOfPlies", sort=True)
    )
    curvature_separations.to_csv(CURVATURE_SEPARATION_OUTPUT, index=False)
    write_markdown(summary)
    make_plot(summary)
    make_table_image(summary)
    print(summary.to_string(index=False))
    print("Wrote:", CSV_OUTPUT)
    print("Wrote:", MARKDOWN_OUTPUT)
    print("Wrote:", PLOT_OUTPUT)
    print("Wrote:", TABLE_OUTPUT)
    print("Wrote:", CLOSEST_PAIRS_OUTPUT)
    print("Wrote:", CURVATURE_SEPARATION_OUTPUT)


if __name__ == "__main__":
    main()
