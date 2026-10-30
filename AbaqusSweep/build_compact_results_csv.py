"""Create a compact, analysis-ready copy of the Abaqus comparison results."""

import csv
from pathlib import Path


SOURCE = Path(__file__).with_name("ABaqusData_with_computational_results.csv")
OUTPUT = Path(__file__).with_name("ABaqusData_compact_errors.csv")

RESPONSE_COMPONENTS = ("e0_11", "e0_22", "e0_12", "K_11", "K_22", "K_12")
CASE_COLUMNS = (
    "Layup",
    "NumberOfPlies",
    "TopSinkTemperature",
    "BottomSinkTemperature",
    "TopTemperature",
    "BottomTemperature",
)
RESPONSE_COLUMNS = tuple(
    column
    for component in RESPONSE_COMPONENTS
    for column in (component, "Comp_" + component, "AbsError_" + component, "PercentError_" + component)
)
FIELDNAMES = CASE_COLUMNS + RESPONSE_COLUMNS


def main():
    with SOURCE.open(newline="", encoding="utf-8-sig") as source_file:
        reader = csv.DictReader(source_file)
        missing = set(FIELDNAMES) - set(reader.fieldnames or ())
        if missing:
            raise ValueError("Source CSV is missing columns: %s" % ", ".join(sorted(missing)))

        with OUTPUT.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows({column: row[column] for column in FIELDNAMES} for row in reader)

    print("Wrote %s" % OUTPUT)


if __name__ == "__main__":
    main()
