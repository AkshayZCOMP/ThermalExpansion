"""Export the interactive curvature and midplane-strain point clouds to VTK.

The output is legacy ASCII VTK PolyData, readable by ParaView and VTK without
additional Python packages.  Geometry is the same as the two Plotly scenes in
``Plotting cleaned.ipynb``.  Each point also retains its ply count and source
CSV row number for coloring or selection in ParaView.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "AbaqusSweep" / "ABaqusData_with_computational_results.csv"
OUTPUT_DIRECTORY = Path(__file__).resolve().parent

CURVATURE_COLUMNS = ("K_11", "K_22", "K_12")
STRAIN_COLUMNS = ("e0_11", "e0_22", "e0_12")


def read_points(source: Path) -> list[dict[str, object]]:
    """Read the complete finite response rows used by both interactive plots."""
    required = ("NumberOfPlies", *CURVATURE_COLUMNS, *STRAIN_COLUMNS)
    points: list[dict[str, object]] = []
    with source.open(newline="", encoding="utf-8") as handle:
        for csv_row, row in enumerate(csv.DictReader(handle), start=2):
            try:
                values = {column: float(row[column]) for column in required}
            except (KeyError, TypeError, ValueError):
                continue
            points.append({"source_row": csv_row, **values})
    if not points:
        raise ValueError(f"No complete numeric response rows found in {source}")
    return points


def write_point_cloud(path: Path, title: str, points: list[dict[str, object]], columns: tuple[str, str, str], *, scale: float = 1.0) -> None:
    """Write a VTK PolyData vertex cloud with numeric point-data arrays."""
    count = len(points)
    with path.open("w", encoding="utf-8", newline="\n") as vtk:
        vtk.write("# vtk DataFile Version 3.0\n")
        vtk.write(f"{title}; source: {INPUT_CSV.name}\n")
        vtk.write("ASCII\nDATASET POLYDATA\n")
        vtk.write(f"POINTS {count} double\n")
        for point in points:
            x, y, z = (float(point[column]) * scale for column in columns)
            vtk.write(f"{x:.16e} {y:.16e} {z:.16e}\n")

        vtk.write(f"VERTICES {count} {count * 2}\n")
        for index in range(count):
            vtk.write(f"1 {index}\n")

        vtk.write(f"POINT_DATA {count}\n")
        for name, values in (
            ("ply_count", (int(point["NumberOfPlies"]) for point in points)),
            ("source_csv_row", (int(point["source_row"]) for point in points)),
            (columns[0], (float(point[columns[0]]) for point in points)),
            (columns[1], (float(point[columns[1]]) for point in points)),
            (columns[2], (float(point[columns[2]]) for point in points)),
        ):
            vtk.write(f"SCALARS {name} double 1\nLOOKUP_TABLE default\n")
            for value in values:
                vtk.write(f"{value:.16e}\n")


def main() -> None:
    points = read_points(INPUT_CSV)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    outputs = {
        "curvature_space_by_ply_count.vtk": (
            "Curvature space by laminate ply count (units: 1/m)",
            CURVATURE_COLUMNS,
            1.0,
        ),
        "midplane_strain_space_by_ply_count.vtk": (
            "Midplane strain space by laminate ply count (geometry units: microstrain)",
            STRAIN_COLUMNS,
            1.0e6,
        ),
    }
    for filename, (title, columns, scale) in outputs.items():
        path = OUTPUT_DIRECTORY / filename
        write_point_cloud(path, title, points, columns, scale=scale)
        print(f"Wrote {path} ({len(points)} points)")


if __name__ == "__main__":
    main()
