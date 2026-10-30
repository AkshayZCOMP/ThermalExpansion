"""Build a short CLT-versus-learned-correction holdout report for 10-ply data."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

from .generate_10ply_hybrid_report import (
    COLORS,
    ROOT,
    REPORT_DIR,
    apply_axis_style,
    set_cell_shading,
    set_document_defaults,
    set_repeat_table_header,
)


MODEL_DIR = ROOT / "PIML/artifacts/experiments/10ply_full_dataset_20261029/orbit_dense_256x4"
REPORT_STEM = "10ply_clt_and_correction_brief_20261030"
FIGURE_NAME = "10ply_clt_vs_correction_identity_20261030.png"
COMPONENTS = ("kappa_x", "kappa_y", "kappa_xy")
LABELS = {
    "kappa_x": r"$\kappa_x$",
    "kappa_y": r"$\kappa_y$",
    "kappa_xy": r"$\kappa_{xy}$",
}


def load_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return reference, CLT baseline, and learned-correction predictions."""
    path = MODEL_DIR / "model_predictions.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    reference = np.array(
        [[float(row[f"reference_{component}"]) for component in COMPONENTS] for row in rows],
        dtype=float,
    )
    clt = np.array(
        [[float(row[f"linear_clt_{component}"]) for component in COMPONENTS] for row in rows],
        dtype=float,
    )
    corrected = np.array(
        [[float(row[f"pinn_{component}"]) for component in COMPONENTS] for row in rows],
        dtype=float,
    )
    return reference, clt, corrected


def metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, np.ndarray]:
    error = prediction - reference
    ss_res = np.sum(error ** 2, axis=0)
    ss_tot = np.sum((reference - reference.mean(axis=0)) ** 2, axis=0)
    return {
        "r2": 1.0 - ss_res / ss_tot,
        "rmse": np.sqrt(np.mean(error ** 2, axis=0)),
        "mae": np.mean(np.abs(error), axis=0),
        "bias": np.mean(error, axis=0),
    }


def identity_figure(
    reference: np.ndarray, clt: np.ndarray, corrected: np.ndarray,
    clt_metrics: dict[str, np.ndarray], corrected_metrics: dict[str, np.ndarray], output: Path,
) -> None:
    """Six compact identity panels with matched physical scales by component."""
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.7), layout="constrained")
    methods = (
        ("Pure CLT", clt, clt_metrics, COLORS["red"]),
        ("CLT + correction", corrected, corrected_metrics, COLORS["blue"]),
    )
    for component_index, component in enumerate(COMPONENTS):
        values = np.concatenate((reference[:, component_index], clt[:, component_index], corrected[:, component_index]))
        extent = max(float(np.abs(values).max()), 0.02) * 1.06
        limits = (-extent, extent)
        for row_index, (method, prediction, summary, color) in enumerate(methods):
            ax = axes[row_index, component_index]
            panel_letter = chr(ord("a") + row_index * len(COMPONENTS) + component_index)
            ax.scatter(
                reference[:, component_index], prediction[:, component_index],
                s=9, color=color, alpha=0.62, linewidths=0, rasterized=True, zorder=2,
            )
            ax.plot(limits, limits, color=COLORS["black"], linestyle="--", linewidth=0.85, zorder=3)
            ax.set(xlim=limits, ylim=limits, aspect="equal")
            ax.set_title(f"({panel_letter}) {method}\n{LABELS[component]}", loc="left", pad=5)
            ax.text(
                0.04, 0.96,
                f"$R^2$ = {summary['r2'][component_index]:.4f}\n"
                f"RMSE = {summary['rmse'][component_index]:.4f} m$^{{-1}}$",
                transform=ax.transAxes, va="top", ha="left", fontsize=6.4,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.2},
            )
            if component_index == 0:
                ax.set_ylabel(r"Predicted curvature [m$^{-1}$]")
            if row_index == 1:
                ax.set_xlabel(r"True curvature [m$^{-1}$]")
            apply_axis_style(ax)
    fig.savefig(output)
    plt.close(fig)


def markdown_table(clt_metrics: dict[str, np.ndarray], corrected_metrics: dict[str, np.ndarray]) -> str:
    rows = []
    for index, component in enumerate(COMPONENTS):
        rows.append(
            f"| `{component}` | {clt_metrics['r2'][index]:.4f} | {corrected_metrics['r2'][index]:.4f} | "
            f"{clt_metrics['rmse'][index]:.5f} | {corrected_metrics['rmse'][index]:.5f} | "
            f"{clt_metrics['mae'][index]:.5f} | {corrected_metrics['mae'][index]:.5f} |"
        )
    return "\n".join(rows)


def write_markdown(clt_metrics: dict[str, np.ndarray], corrected_metrics: dict[str, np.ndarray], output: Path) -> None:
    output.write_text(
        f"""# Classical lamination theory versus learned nonlinear correction

## Scope

This brief compares pure classical lamination theory (CLT) with the selected residual-curvature model on the frozen, orbit-disjoint 10-ply FREEFREE holdout (832 cases). The candidate-library, thermal fields, and test cases are identical for both methods. CLT is the physics baseline. The corrected prediction is

`kappa_hat = kappa_CLT + g f_theta(z), with g = (T_top - T_bottom)/100,`

where the neural network learns only the nonlinear FREEFREE-minus-CLT curvature residual. The identity plots use the true candidate for each holdout query; they evaluate forward curvature prediction, not layup retrieval rank.

## Results

R-squared is the conventional held-out coefficient of determination, `1 - SSE/SST`, computed separately for each curvature component. RMSE and MAE are in m^-1.

| Component | CLT R^2 | Corrected R^2 | CLT RMSE | Corrected RMSE | CLT MAE | Corrected MAE |
|---|---:|---:|---:|---:|---:|---:|
{markdown_table(clt_metrics, corrected_metrics)}

The learned correction improves all three components. The largest absolute residual remains `kappa_xy`, consistent with the greater nonlinear sensitivity of off-axis twist curvature. The comparison is made on layup symmetry orbits excluded from training, so the identity plots do not reflect interpolation to the same layup at a different thermal field.

**Figure 1. Identity plots for pure CLT and CLT plus the learned nonlinear correction.** Columns correspond to `kappa_x`, `kappa_y`, and `kappa_xy`; each column uses matched physical axes across the two methods. Dashed lines denote one-to-one agreement. Insets report held-out R-squared and RMSE.
""",
        encoding="utf-8",
    )


def write_results_table(document: Document, clt_metrics: dict[str, np.ndarray], corrected_metrics: dict[str, np.ndarray]) -> None:
    table = document.add_table(rows=1, cols=7)
    table.style = "Table Grid"
    headers = ("Component", "CLT R²", "Corrected R²", "CLT RMSE", "Corrected RMSE", "CLT MAE", "Corrected MAE")
    for cell, text in zip(table.rows[0].cells, headers):
        cell.text = text
        set_cell_shading(cell, "D9EAF7")
        for run in cell.paragraphs[0].runs:
            run.bold = True
            run.font.size = Pt(7)
    set_repeat_table_header(table.rows[0])
    for index, component in enumerate(COMPONENTS):
        cells = table.add_row().cells
        values = (
            LABELS[component],
            f"{clt_metrics['r2'][index]:.4f}",
            f"{corrected_metrics['r2'][index]:.4f}",
            f"{clt_metrics['rmse'][index]:.5f}",
            f"{corrected_metrics['rmse'][index]:.5f}",
            f"{clt_metrics['mae'][index]:.5f}",
            f"{corrected_metrics['mae'][index]:.5f}",
        )
        for cell, text in zip(cells, values):
            cell.text = text
            for run in cell.paragraphs[0].runs:
                run.font.size = Pt(7)


def write_word(clt_metrics: dict[str, np.ndarray], corrected_metrics: dict[str, np.ndarray], figure: Path, output: Path) -> None:
    document = Document()
    set_document_defaults(document)
    document.core_properties.title = "Classical lamination theory versus learned nonlinear correction"
    document.add_heading("Classical lamination theory versus learned nonlinear correction", 0).alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle = document.add_paragraph("10-ply orbit-disjoint FREEFREE holdout | short forward-prediction brief")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.runs[0].italic = True

    document.add_heading("Comparison", level=1)
    document.add_paragraph("This short report compares pure classical lamination theory (CLT) with the selected CLT-residual neural model on the frozen 832-case, orbit-disjoint 10-ply holdout. Both methods use the same known layup and face-temperature inputs. The neural model adds a learned nonlinear correction to the CLT curvature rather than replacing CLT: kappa_hat = kappa_CLT + g f_theta(z), with g = (Ttop − Tbottom)/100. Thus, the plots isolate the benefit of learning the nonlinear FREEFREE-minus-CLT residual.")
    document.add_paragraph("The holdout comprises complete square-plate symmetry orbits excluded from training. The identity plots evaluate forward curvature prediction for the true candidate, not the inverse candidate-ranking score. R² is the conventional held-out coefficient of determination, 1 − SSE/SST, and RMSE/MAE use units of m^-1.")

    document.add_heading("Results", level=1)
    document.add_paragraph("The learned correction improves agreement for each curvature component. The largest residual error remains the engineering twist curvature, kappa_xy, consistent with the stronger nonlinear behavior of off-axis laminates. The table reports all-case metrics; no near-zero filtering is used for the displayed identity data.")
    write_results_table(document, clt_metrics, corrected_metrics)
    document.add_paragraph()
    document.add_picture(str(figure), width=Inches(6.8))
    document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption = document.add_paragraph("Figure 1. Identity plots for pure CLT (top row) and CLT plus the learned nonlinear correction (bottom row). Columns show kappa_x, kappa_y, and kappa_xy. Each column has matched physical axes across methods; dashed lines indicate one-to-one agreement. Insets give held-out R² and RMSE.")
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in caption.runs:
        run.italic = True
        run.font.size = Pt(8)
    document.add_paragraph("Interpretation: CLT captures the primary thermal-curvature trend, while the residual network corrects the nonlinear FvK departure. The correction is most consequential for twist curvature, where pure CLT has the largest scatter about the identity line. This is a forward-prediction comparison only; layup-retrieval and nonlinear re-ranking results are reported separately in the hybrid methods-and-results document.")
    document.save(output)


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    reference, clt, corrected = load_predictions()
    clt_metrics = metrics(reference, clt)
    corrected_metrics = metrics(reference, corrected)
    figure = REPORT_DIR / FIGURE_NAME
    identity_figure(reference, clt, corrected, clt_metrics, corrected_metrics, figure)
    markdown = REPORT_DIR / f"{REPORT_STEM}.md"
    word = REPORT_DIR / f"{REPORT_STEM}.docx"
    write_markdown(clt_metrics, corrected_metrics, markdown)
    write_word(clt_metrics, corrected_metrics, figure, word)
    print(f"Wrote {figure}")
    print(f"Wrote {markdown}")
    print(f"Wrote {word}")
    for index, component in enumerate(COMPONENTS):
        print(
            f"{component}: CLT R2={clt_metrics['r2'][index]:.6f}, "
            f"corrected R2={corrected_metrics['r2'][index]:.6f}; "
            f"CLT RMSE={clt_metrics['rmse'][index]:.6f}, "
            f"corrected RMSE={corrected_metrics['rmse'][index]:.6f} 1/m"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
