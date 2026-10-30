"""Regenerate the publication-style 10-ply neural--FvK hybrid report.

The script is deliberately data-driven: it reads the frozen orbit-holdout
predictions and the recorded Monte-Carlo hybrid summary, makes journal-format
figures, and writes the paired Markdown and Word report.  The FvK stage is
explicitly described as a solver-consistent stored-response replay.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "PIML/artifacts/experiments/10ply_full_dataset_20261029"
MODEL_DIR = EXPERIMENT / "orbit_dense_256x4"
HYBRID_DIR = EXPERIMENT / "hybrid_fvk_top5_oracle"
REPORT_DIR = ROOT / "PIML/artifacts/reports"
REPORT_STEM = "10ply_neural_fvk_hybrid_methods_results_20261029"
COMPONENTS = ("kappa_x", "kappa_y", "kappa_xy")
COMPONENT_LABELS = {
    "kappa_x": r"$\kappa_x$",
    "kappa_y": r"$\kappa_y$",
    "kappa_xy": r"$\kappa_{xy}$",
}


# Matplotlib settings intentionally match Plotting/Plotting cleaned.ipynb.
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "black": "#202020",
    "gray": "#6E6E6E",
    "light_gray": "#D9D9D9",
}
mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "font.family": "Arial",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "axes.linewidth": 0.8,
    "axes.labelpad": 4,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def apply_axis_style(ax: plt.Axes, *, grid_axis: str = "both") -> None:
    """Use the clean notebook's compact, low-ink axis treatment."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis=grid_axis, color=COLORS["light_gray"], linewidth=0.55, zorder=0)
    ax.tick_params(which="both", top=False, right=False)


def load_prediction_data() -> tuple[np.ndarray, np.ndarray]:
    path = MODEL_DIR / "model_predictions.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    reference = np.array(
        [[float(row[f"reference_{component}"]) for component in COMPONENTS] for row in rows],
        dtype=float,
    )
    prediction = np.array(
        [[float(row[f"pinn_{component}"]) for component in COMPONENTS] for row in rows],
        dtype=float,
    )
    return reference, prediction


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def parity_figure(reference: np.ndarray, prediction: np.ndarray, output: Path) -> None:
    """Component-resolved surrogate parity plot plus FvK oracle parity panel."""
    component_colors = [COLORS["blue"], COLORS["orange"], COLORS["purple"]]
    component_markers = ["o", "s", "^"]
    all_values = np.concatenate((reference.ravel(), prediction.ravel()))
    extent = max(float(np.abs(all_values).max()), 0.05) * 1.06
    limits = (-extent, extent)
    rmse = np.sqrt(np.mean((prediction - reference) ** 2, axis=0))

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.15), layout="constrained")
    ax = axes[0]
    for component_index, (component, color, marker) in enumerate(
        zip(COMPONENTS, component_colors, component_markers)
    ):
        ax.scatter(
            reference[:, component_index], prediction[:, component_index],
            s=10, alpha=0.64, marker=marker, color=color, linewidths=0,
            rasterized=True, label=COMPONENT_LABELS[component], zorder=2,
        )
    ax.plot(limits, limits, color=COLORS["black"], linestyle="--", linewidth=0.9,
            label="1:1", zorder=3)
    ax.set(xlim=limits, ylim=limits, aspect="equal",
           xlabel=r"FREEFREE reference curvature [m$^{-1}$]",
           ylabel=r"Neural prediction [m$^{-1}$]")
    ax.set_title("(a) Neural forward surrogate", loc="left")
    ax.text(
        0.03, 0.97,
        "RMSE [m$^{-1}$]\n"
        + "\n".join(f"{COMPONENT_LABELS[component]}: {value:.4f}" for component, value in zip(COMPONENTS, rmse)),
        transform=ax.transAxes, va="top", ha="left", fontsize=6.7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.8},
    )
    ax.legend(loc="lower right", ncol=2, columnspacing=0.9)
    apply_axis_style(ax)

    # Stored FREEFREE values are exactly the solver-consistent response used
    # for the candidate re-ranking oracle; show this explicitly rather than
    # implying a separately learned FvK surrogate.
    ax = axes[1]
    for component_index, (component, color, marker) in enumerate(
        zip(COMPONENTS, component_colors, component_markers)
    ):
        ax.scatter(
            reference[:, component_index], reference[:, component_index],
            s=10, alpha=0.64, marker=marker, color=color, linewidths=0,
            rasterized=True, label=COMPONENT_LABELS[component], zorder=2,
        )
    ax.plot(limits, limits, color=COLORS["black"], linestyle="--", linewidth=0.9,
            label="1:1", zorder=3)
    ax.set(xlim=limits, ylim=limits, aspect="equal",
           xlabel=r"FREEFREE reference curvature [m$^{-1}$]",
           ylabel=r"Stored FvK response [m$^{-1}$]")
    ax.set_title("(b) Solver-consistent re-ranking response", loc="left")
    ax.text(
        0.03, 0.97, "Stored nonlinear solver output\nfor the queried thermal field",
        transform=ax.transAxes, va="top", ha="left", fontsize=6.7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.8},
    )
    apply_axis_style(ax)
    fig.savefig(output)
    plt.close(fig)


def retrieval_figure(hybrid: dict[str, object], output: Path) -> None:
    """Clean rank distribution and clean/noisy top-one recovery figure."""
    clean = hybrid["clean"]
    noise = hybrid["noise_trials"]
    assert isinstance(clean, dict) and isinstance(noise, dict)
    rank_counts = clean["neural_rank_counts"]
    assert isinstance(rank_counts, dict)
    neural_counts = np.array([int(rank_counts[str(rank)]) for rank in range(1, 5)])
    hybrid_counts = np.array([832, 0, 0, 0])

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), layout="constrained")
    ax = axes[0]
    rank = np.arange(1, 5)
    width = 0.34
    left = ax.bar(rank - width / 2, neural_counts, width, color=COLORS["blue"],
                  label="Neural screen", zorder=2)
    right = ax.bar(rank + width / 2, hybrid_counts, width, color=COLORS["green"],
                   label="Neural + FvK", zorder=2)
    for bars in (left, right):
        for bar in bars:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                        f"{int(bar.get_height())}", ha="center", va="bottom", fontsize=6.8)
    ax.set(xticks=rank, xlabel="True-candidate rank", ylabel="Holdout queries",
           ylim=(0, 900))
    ax.set_title("(a) Clean retrieval-rank distribution", loc="left")
    ax.legend(loc="upper right")
    apply_axis_style(ax, grid_axis="y")

    neural_noise = noise["neural_top1"]
    hybrid_noise = noise["hybrid_top1"]
    assert isinstance(neural_noise, dict) and isinstance(hybrid_noise, dict)
    condition = np.array([0, 1])
    neural_mean = np.array([float(clean["neural_top1"]), float(neural_noise["mean"])]) * 100
    neural_std = np.array([0.0, float(neural_noise["std"]) * 100])
    fvk_mean = np.array([float(clean["hybrid_top1"]), float(hybrid_noise["mean"])]) * 100
    fvk_std = np.array([0.0, float(hybrid_noise["std"]) * 100])
    ax = axes[1]
    ax.errorbar(condition, neural_mean, yerr=neural_std, color=COLORS["blue"], marker="o",
                markersize=4.5, linewidth=1.2, capsize=2.5, label="Neural screen", zorder=3)
    ax.errorbar(condition, fvk_mean, yerr=fvk_std, color=COLORS["green"], marker="s",
                markersize=4.5, linewidth=1.2, capsize=2.5, label="Neural + FvK", zorder=3)
    for x, value in zip(condition, neural_mean):
        ax.annotate(f"{value:.1f}%", (x, value), xytext=(0, -13), textcoords="offset points",
                    ha="center", va="top", color=COLORS["blue"], fontsize=6.8)
    for x, value in zip(condition, fvk_mean):
        ax.annotate(f"{value:.1f}%", (x, value), xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", color=COLORS["green"], fontsize=6.8)
    ax.set(xticks=condition, xticklabels=["Clean", "Gaussian noise\n$\\sigma=0.001$ m$^{-1}$"],
           ylabel="Top-1 recovery [%]", ylim=(78, 101.5), xlim=(-0.18, 1.18))
    ax.set_title("(b) Top-1 recovery", loc="left")
    # Place the legend between the two response bands, away from both the
    # 82% neural points and the ~100% nonlinear points.
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.50))
    apply_axis_style(ax, grid_axis="y")
    fig.savefig(output)
    plt.close(fig)


def runtime_figure(output: Path) -> None:
    """Use point/interval marks on a log axis, not bars whose zero is undefined."""
    fig, ax = plt.subplots(figsize=(7.15, 2.2), layout="constrained")
    neural_time = 3.68e-3
    fvk_interval = (15.0, 30.0)
    ax.scatter(neural_time, 0, s=34, marker="o", color=COLORS["blue"], zorder=3)
    ax.hlines(1, *fvk_interval, color=COLORS["green"], linewidth=3.0, zorder=3)
    ax.scatter(np.mean(fvk_interval), 1, s=30, marker="s", color=COLORS["green"], zorder=4)
    ax.vlines(fvk_interval, 0.91, 1.09, color=COLORS["green"], linewidth=1.0, zorder=4)
    ax.annotate("3.68 ms mean\nmeasured", (neural_time, 0), xytext=(7, 7),
                textcoords="offset points", ha="left", va="bottom", fontsize=7)
    ax.annotate("15–30 s\nthroughput estimate", (np.mean(fvk_interval), 1), xytext=(0, 8),
                textcoords="offset points", ha="center", va="bottom", fontsize=7)
    ax.set_xscale("log")
    ax.set(xlim=(1e-3, 60), ylim=(-0.45, 1.45),
           yticks=[0, 1], yticklabels=["Neural screen\n254 candidates", "FvK re-rank\n5 candidates"],
           xlabel="Query-stage wall time [s; logarithmic scale]")
    ax.set_title("Runtime scope", loc="left")
    ax.text(0.99, 0.04, "Measured neural timing excludes loading, I/O, and DIC extraction.\n"
                        "FvK interval is not a dedicated online benchmark.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.4,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.4})
    apply_axis_style(ax, grid_axis="x")
    fig.savefig(output)
    plt.close(fig)


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def report_markdown(summary: dict[str, object], hybrid: dict[str, object]) -> str:
    components = summary["components"]
    behavior = summary["behavior_focused"]
    assert isinstance(components, dict) and isinstance(behavior, dict)
    neural_behavior = behavior["pinn"]
    clt_behavior = behavior["linear_clt"]
    clean = hybrid["clean"]
    noise = hybrid["noise_trials"]
    assert all(isinstance(item, dict) for item in (neural_behavior, clt_behavior, clean, noise))
    kx = components["kappa_x"]["pinn"]
    ky = components["kappa_y"]["pinn"]
    kxy = components["kappa_xy"]["pinn"]
    neural_noise = noise["neural_top1"]
    hybrid_noise = noise["hybrid_top1"]
    assert all(isinstance(item, dict) for item in (kx, ky, kxy, neural_noise, hybrid_noise))

    return f"""# Neural screening and nonlinear FvK re-ranking for 10-ply laminate retrieval

## Methods

### Task, data, and split

The task was to recover an unknown 10-ply laminate from its free-curvature vector `(kappa_x, kappa_y, kappa_xy)` under known top- and bottom-face temperatures. The candidate library contained 254 legal bottom-to-top stacking sequences drawn from `{{-45 degrees, 0 degrees, +45 degrees, 90 degrees}}`. The data comprised 4,063 converged nonlinear FREEFREE Foppl--von Karman (FvK) responses across 16 direct face-temperature fields (face temperatures 3.75--96.25 degrees). Each label contained the full stacking sequence, the two prescribed face temperatures, and the three equilibrium curvature components. The test set contained 832 records from 14 complete square-plate symmetry orbits (52 unique layups); the 3,231-record training partition comprised the remaining 58 orbits (202 unique layups). Thus, neither an exact layup nor a 90-degree rotation/reflection equivalent was shared between training and test partitions.

### Neural forward surrogate

The selected neural screen was the original three-output, direct residual-curvature model (`orbit_dense_256x4`), chosen before the later exploratory variants. Classical lamination theory (CLT) provided the baseline, and the network predicted the nonlinear correction

`kappa_hat = kappa_CLT + g f_theta(z),  where  g = (T_top - T_bottom)/100.`

The 44-element input `z` comprised a 40-element position-preserving one-hot encoding of the ten ply angles (four categories per ply, ordered `[-45, 0, +45, 90]`), `g^2`, and the three CLT curvature components divided by `g`. Factoring the correction by `g` imposes odd reversal parity with respect to the applied face-temperature difference, while allowing nonlinear dependence on its magnitude through `g^2`. The model was a 209,667-parameter multilayer perceptron with four 256-unit SiLU hidden layers and a three-component output. It used double-precision arithmetic; it did not use a separate off-axis expert, ABD descriptors, membrane-strain features, target-noise augmentation, or thickness-reversal augmentation.

The residual target was `r = kappa_FREEFREE - kappa_CLT`. Each component was standardized by its training-set residual standard deviation and optimized with equally weighted mean-square error. The network was trained for 3,000 full-batch AdamW iterations (learning rate `7 x 10^-4`, weight decay `1 x 10^-6`). Training-only reflection and 90-degree-rotation augmentation expanded the 3,231 physical training records to 12,924 optimization examples. Checkpoint, prediction table, split CSVs, and the model-evaluation script are retained with the experiment artifacts to reproduce all reported scores.

### Candidate ranking and nonlinear confirmation

For each query, the neural forward model was evaluated over all 254 candidate layups and sorted by the component-scaled mismatch

`s_i = sum_j ((kappa_hat_i,j - kappa_meas,j) / d_j)^2,  with  d = (0.01, 0.01, 0.02) m^-1.`

The five lowest-scoring candidates formed the nonlinear shortlist. The intended confirmation stage solves the nonlinear FREEFREE/FvK problem only for those candidates and re-ranks them with the same curvature score. The reported FvK accuracy is a solver-consistent replay: stored FREEFREE responses at the queried temperature field were used in place of a fresh online solve. It therefore measures the achievable re-ranking logic under the same solver, not independent experimental accuracy or live-solver latency.

### Evaluation, noise, and runtime

Forward error is reported against FREEFREE curvatures, with active-component metrics excluding reference magnitudes below 0.02 m^-1. Retrieval is summarized by top-1, top-5, and worst true rank. Independent zero-mean Gaussian noise (SD 0.001 m^-1 per curvature component) was applied to queries in 100 seeded Monte-Carlo trials; temperatures and candidate labels were fixed. Neural timing was measured for 50 loaded-model rankings of all 254 candidates and includes CLT evaluation, neural inference, scoring, and sorting. It excludes loading, I/O, DIC curvature extraction, and nonlinear solution time.

## Results

### Forward prediction and neural screening

The neural residual surrogate reduced the active-component RMSE from {float(clt_behavior['active_component_rmse_1_per_m']):.5f} m^-1 for CLT to {float(neural_behavior['active_component_rmse_1_per_m']):.5f} m^-1. On the 832-case orbit-disjoint holdout, RMSE was {float(kx['rmse_1_per_m']):.5f} m^-1 for `kappa_x`, {float(ky['rmse_1_per_m']):.5f} m^-1 for `kappa_y`, and {float(kxy['rmse_1_per_m']):.5f} m^-1 for `kappa_xy`. The larger twist error ({float(kxy['active_rmse_1_per_m']):.5f} m^-1 over active cases) remained concentrated in off-axis laminates. Figure 1 separates this learned forward-model agreement from the identity of the stored solver response used by the re-ranking replay.

The neural screen recovered the correct layup at rank one for 688/832 queries (82.69%) and within its top five for all 832/832 queries (100%); its worst true rank was four. Therefore, every clean holdout truth was retained by the five-candidate nonlinear shortlist. With Gaussian curvature noise, neural top-1 recovery was {100 * float(neural_noise['mean']):.2f}% (SD {100 * float(neural_noise['std']):.2f} percentage points; range {100 * float(neural_noise['min']):.2f}–{100 * float(neural_noise['max']):.2f}%), while neural top-five recovery was 100% in all trials.

| Holdout metric | CLT / neural forward model | Neural screen | Neural screen + stored FvK re-rank |
|---|---:|---:|---:|
| Active-component RMSE [m^-1] | {float(clt_behavior['active_component_rmse_1_per_m']):.5f} / {float(neural_behavior['active_component_rmse_1_per_m']):.5f} | — | — |
| Clean top-1 | — | 688/832 (82.69%) | 832/832 (100.00%) |
| Clean top-5 | — | 832/832 (100.00%) | 832/832 (100.00%) |
| Noisy top-1, mean (SD) | — | {100 * float(neural_noise['mean']):.2f}% ({100 * float(neural_noise['std']):.2f} pp) | {100 * float(hybrid_noise['mean']):.2f}% ({100 * float(hybrid_noise['std']):.2f} pp) |
| Worst rank under noise | — | not above five | 2 |

**Figure 1. Forward-prediction identity plots.** (a) The neural residual surrogate is evaluated for the true candidate in each held-out query. (b) The FvK panel is an identity by construction because stored nonlinear solver responses are used for the solver-consistent re-ranking replay; it is shown to make that evaluation boundary explicit.

### Nonlinear re-ranking after neural screening

Because the correct laminate was retained in every clean neural top-five shortlist, stored-response FvK re-ranking produced 832/832 clean top-one recoveries. With Gaussian noise, the truth also remained in the neural shortlist in every trial. The solver-consistent re-ranker achieved mean top-one recovery of {100 * float(hybrid_noise['mean']):.2f}% (SD {100 * float(hybrid_noise['std']):.2f} percentage points; range {100 * float(hybrid_noise['min']):.2f}–{100 * float(hybrid_noise['max']):.2f}%), 100% mean top-five recovery, and a maximum observed true rank of two. Figure 2 shows both the clean rank distribution and the separate clean/noisy top-one comparison.

**Figure 2. Retrieval behavior with and without nonlinear confirmation.** (a) The neural screen leaves 144 clean queries below rank one but retains every truth in ranks 1–4; stored-response FvK re-ranking resolves all clean cases at rank one. (b) Points show clean recovery or Monte-Carlo means; error bars denote trial SD for the noise condition.

### Runtime and scope

The loaded neural screen ranked all 254 candidates in 3.68 ms on average (median 4.37 ms). Observed local data-generation throughput suggests approximately 15–30 s wall time to re-rank five FvK candidates in parallel, compared with approximately one hour for serial nonlinear evaluation of all 254 candidates. This is an operational estimate rather than a dedicated persistent-worker benchmark. The two-stage design therefore places the nonlinear effort only after a millisecond-scale full-library screen (Figure 3).

**Figure 3. Runtime scope.** The neural point is a measured loaded-model mean. The FvK interval is an observed-throughput estimate for five parallel candidates and should not be used as a deployment latency claim until it is measured with persistent workers.

### Interpretation

For this canonical candidate library and solver-defined test domain, the selected neural model is a complete top-five screen under both clean and specified noisy curvature queries. Nonlinear confirmation is useful precisely because its cost is bounded to five candidates and it corrects the residual off-axis twist-ordering errors. The solver-consistent re-ranking result does not establish agreement with finite-element or experimental measurements; those independent data, together with a persistent-worker FvK timing benchmark, are required before deployment or experimental-accuracy claims.
"""


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    tc_pr.append(shading)


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)


def add_caption(document: Document, text: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(text)
    run.italic = True
    run.font.name = "Arial"
    run.font.size = Pt(8)


def add_results_table(document: Document, summary: dict[str, object], hybrid: dict[str, object]) -> None:
    behavior = summary["behavior_focused"]
    noise = hybrid["noise_trials"]
    assert isinstance(behavior, dict) and isinstance(noise, dict)
    pinn = behavior["pinn"]
    clt = behavior["linear_clt"]
    neural_noise = noise["neural_top1"]
    hybrid_noise = noise["hybrid_top1"]
    assert all(isinstance(item, dict) for item in (pinn, clt, neural_noise, hybrid_noise))
    table = document.add_table(rows=1, cols=4)
    table.style = "Table Grid"
    headers = ["Metric", "CLT / neural forward", "Neural screen", "Neural + stored FvK"]
    for cell, value in zip(table.rows[0].cells, headers):
        cell.text = value
        set_cell_shading(cell, "D9EAF7")
        for run in cell.paragraphs[0].runs:
            run.bold = True
    set_repeat_table_header(table.rows[0])
    rows = [
        ("Active-component RMSE [m^-1]", f"{float(clt['active_component_rmse_1_per_m']):.5f} / {float(pinn['active_component_rmse_1_per_m']):.5f}", "—", "—"),
        ("Clean top-1", "—", "688/832 (82.69%)", "832/832 (100.00%)"),
        ("Clean top-5", "—", "832/832 (100.00%)", "832/832 (100.00%)"),
        ("Noisy top-1, mean (SD)", "—", f"{100 * float(neural_noise['mean']):.2f}% ({100 * float(neural_noise['std']):.2f} pp)", f"{100 * float(hybrid_noise['mean']):.2f}% ({100 * float(hybrid_noise['std']):.2f} pp)"),
        ("Worst rank", "—", "4 clean; <=5 noisy", "1 clean; <=2 noisy"),
    ]
    for values in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, values):
            cell.text = value
    document.add_paragraph()


def add_model_table(document: Document) -> None:
    table = document.add_table(rows=1, cols=2)
    table.style = "Table Grid"
    for cell, text in zip(table.rows[0].cells, ("Model element", "Selected configuration")):
        cell.text = text
        set_cell_shading(cell, "D9EAF7")
        for run in cell.paragraphs[0].runs:
            run.bold = True
    set_repeat_table_header(table.rows[0])
    rows = [
        ("Candidate domain", "254 legal 10-ply layups; bottom-to-top ply order"),
        ("Inputs", "40 one-hot ply indicators + g^2 + 3 CLT curvature-per-gradient features (44 total)"),
        ("Output", "Three residual-curvature components, multiplied by g = (Ttop − Tbottom)/100"),
        ("Neural architecture", "44 → 256 → 256 → 256 → 256 → 3; SiLU activations; 209,667 parameters"),
        ("Training", "3,000 full-batch AdamW steps; learning rate 7 × 10^-4; weight decay 1 × 10^-6"),
        ("Augmentation", "Training-only reflection and 90-degree rotation; 3,231 physical records expanded to 12,924 examples"),
        ("Ranking", "All candidates scored with d = (0.01, 0.01, 0.02) m^-1; retain top five"),
    ]
    for left, right in rows:
        cells = table.add_row().cells
        cells[0].text = left
        cells[1].text = right
    document.add_paragraph()


def set_document_defaults(document: Document) -> None:
    section = document.sections[0]
    section.top_margin = Inches(0.70)
    section.bottom_margin = Inches(0.70)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)
    normal = document.styles["Normal"]
    normal.font.name = "Arial"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
    normal.font.size = Pt(9)
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.08
    for style_name, size, color in (("Title", 17, "1F4E79"), ("Heading 1", 13, "1F4E79"), ("Heading 2", 11, "1F4E79")):
        style = document.styles[style_name]
        style.font.name = "Arial"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string(color)
    if "Caption" not in document.styles:
        caption = document.styles.add_style("Caption", WD_STYLE_TYPE.PARAGRAPH)
        caption.font.name = "Arial"
        caption.font.size = Pt(8)


def add_body_paragraph(document: Document, text: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.add_run(text)


def generate_word_report(summary: dict[str, object], hybrid: dict[str, object], figures: list[Path], output: Path) -> None:
    document = Document()
    set_document_defaults(document)
    document.core_properties.title = "Neural screening and nonlinear FvK re-ranking for 10-ply laminate retrieval"
    document.core_properties.subject = "Methods and results"
    document.core_properties.author = "HeatXferDIC"
    title = document.add_heading("Neural screening and nonlinear FvK re-ranking for 10-ply laminate retrieval", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle = document.add_paragraph("Methods and results | orbit-disjoint FREEFREE holdout")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.runs[0].italic = True

    document.add_heading("Methods", level=1)
    document.add_heading("Problem definition and nonlinear response data", level=2)
    add_body_paragraph(document, "The objective was discrete inverse retrieval: given the measured free-curvature vector (kappa_x, kappa_y, kappa_xy) of a thermally loaded plate and its prescribed top- and bottom-face temperatures, identify the most likely ten-ply stacking sequence. The search domain was fixed before testing and contained 254 legal canonical bottom-to-top layups assembled from −45°, 0°, +45°, and 90° plies. Every query was therefore ranked against the complete 254-candidate library rather than against only seen training layups.")
    add_body_paragraph(document, "The forward labels were nonlinear FREEFREE Foppl–von Kármán equilibrium solutions. Each record stores a layup, direct face temperatures, and three curvatures in m^-1. The complete analysis set contained 4,063 converged responses across 16 through-thickness thermal fields, with face temperatures spanning 3.75–96.25°. The response used for learning and ranking was the curvature vector only; temperatures were supplied as known loading conditions, not inferred targets.")

    document.add_heading("Split design and leakage control", level=2)
    add_body_paragraph(document, "The study used a layup-orbit holdout rather than a random row split. The training partition comprised 3,231 responses from 58 square-plate symmetry orbits (202 unique layups); the holdout comprised 832 responses from 14 disjoint orbits (52 unique layups). An orbit includes exact 90° rotation and reflection equivalents. Consequently, no test layup and no in-plane symmetry equivalent of a test layup occurred in training. All 16 thermal fields associated with a selected layup orbit were assigned to one partition, preventing the model from seeing a held-out layup at a different temperature field during fitting.")

    document.add_heading("CLT-residual neural forward model", level=2)
    add_body_paragraph(document, "Classical lamination theory (CLT) first supplied a physics-based curvature baseline for each candidate and prescribed temperature field. Rather than regress the total curvature directly, the network learned the nonlinear correction r = kappa_FREEFREE − kappa_CLT and returned kappa_hat = kappa_CLT + g f_theta(z), where g = (Ttop − Tbottom)/100. This residual construction retains the inexpensive CLT trend and focuses the learned function on the nonlinear FvK departure.")
    add_body_paragraph(document, "The input vector z had 44 elements. The first 40 values are a position-preserving one-hot representation of the ten-ply sequence: each ply contributes four indicators in the fixed angle order [−45°, 0°, +45°, 90°]. The remaining four features are g^2 and the three CLT curvature components divided by g. Multiplication of the network output by g enforces odd response reversal when the signed face-temperature difference is reversed, while g^2 permits nonlinear dependence on thermal-gradient magnitude. The model was evaluated in double precision.")
    add_body_paragraph(document, "The selected architecture was a fully connected multilayer perceptron, 44 → 256 → 256 → 256 → 256 → 3, with SiLU activations after every hidden layer and 209,667 trainable parameters. It jointly predicts the x-, y-, and engineering twist-curvature residuals. This is the original dense three-output model; it contains no hard axis/off-axis gate, no ABD coupling descriptors, no membrane-strain features, and no target-noise or thickness-reversal augmentation. Those later exploratory variants are not part of the reported hybrid result.")
    add_model_table(document)

    document.add_heading("Training, normalization, and augmentation", level=2)
    add_body_paragraph(document, "For each curvature component, the residual target was standardized using the training-set residual standard deviation. Training minimized the equally weighted mean-square error of these standardized residuals. Optimization used AdamW for 3,000 full-batch iterations with learning rate 7 × 10^-4 and weight decay 1 × 10^-6. The fitting set therefore contained every augmented training record in each optimization step; no validation examples were used for gradient updates.")
    add_body_paragraph(document, "Only exact in-plane square-plate symmetries were used for augmentation. Each training response was supplemented with its reflected, 90°-rotated, and rotated-reflected representations, together with the corresponding curvature-component transformations. This expanded 3,231 physical training records to 12,924 optimization examples. No augmentation was applied to the holdout. The split CSVs, model checkpoint, per-case predictions, model summary, and the report-generation script are retained in the experiment directory so that the reported result can be regenerated from stored artifacts.")

    document.add_heading("Candidate scoring and FvK confirmation", level=2)
    add_body_paragraph(document, "At inference, the neural forward model was run once for each of the 254 candidates at the query temperatures. Candidates were assigned s_i = sum_j[((kappa_hat_i,j − kappa_meas,j)/d_j)^2], with d = (0.01, 0.01, 0.02) m^-1 for (kappa_x, kappa_y, kappa_xy), and sorted in ascending order. The component scales prevent the larger dynamic range of twist curvature from dominating the retrieval score. The first stage returns a complete ranking as well as a five-candidate shortlist.")
    add_body_paragraph(document, "The intended second stage solves the nonlinear FREEFREE/FvK equilibrium problem only for the neural top five, then applies the same mismatch score to re-rank them. For the present evaluation, fresh solves were replaced by stored FREEFREE responses at the queried field. The reported FvK result is therefore a solver-consistent stored-response replay: it evaluates the two-stage decision logic under the same nonlinear solver that generated the labels, but is not independent experimental validation and is not an online FvK timing measurement.")

    document.add_heading("Metrics, noise experiment, and runtime scope", level=2)
    add_body_paragraph(document, "Forward error was calculated against FREEFREE reference curvatures. In addition to all-case component RMSE, active-component metrics exclude components with |reference curvature| below 0.02 m^-1; this avoids interpreting error relative to a physically near-zero response as meaningful. Retrieval was summarized by exact top-1 recovery, top-5 recovery, and worst true rank. All clean results use the 832-case orbit-disjoint holdout without re-tuning the selected architecture.")
    add_body_paragraph(document, "Noise robustness was measured by adding independent, zero-mean Gaussian noise with standard deviation 0.001 m^-1 to every curvature component of each query. Temperatures, candidate labels, and stored nonlinear responses were fixed. One hundred seeded Monte-Carlo trials were analyzed, and the report gives trial mean, standard deviation, and range. Neural timing is the mean of 50 loaded-model rankings of all 254 candidates and includes CLT evaluation, neural inference, scoring, and sorting. It excludes model loading, disk I/O, DIC curvature extraction, and nonlinear solution time. The five-candidate FvK wall time is an observed local throughput estimate, not a persistent-worker benchmark.")

    document.add_heading("Results", level=1)
    document.add_heading("Forward-model fidelity", level=2)
    add_body_paragraph(document, "On the orbit-disjoint holdout, the learned residual reduced active-component RMSE from 0.02577 m^-1 for CLT to 0.01412 m^-1. All-case neural RMSE was 0.00483 m^-1 for kappa_x, 0.00551 m^-1 for kappa_y, and 0.02174 m^-1 for kappa_xy. The corresponding active kappa_xy RMSE was 0.02263 m^-1, making twist curvature the dominant residual error and the main source of ambiguity in off-axis laminate ranking. Figure 1a shows the component-resolved parity relationship for the true candidate in every holdout query.")
    add_body_paragraph(document, "Figure 1b is intentionally not presented as a learned parity result. It displays the stored nonlinear FREEFREE response used in the FvK re-ranking replay, and therefore lies on the identity line by construction. This panel makes the boundary of the hybrid evidence explicit: it illustrates solver-consistent re-ranking rather than independently predicted FvK response.")

    document.add_heading("Clean retrieval and Gaussian-noise robustness", level=2)
    add_body_paragraph(document, "Neural screening returned the exact layup at rank one for 688 of 832 holdout queries (82.69%). Crucially, it retained the true layup in the top five for all 832 cases (100%), and its largest clean true rank was four. The stored-response FvK stage therefore had access to the truth for every clean query and re-ranked all 832 cases to rank one. The clean rank counts—688, 104, 32, and 8 at neural ranks one through four—are shown in Figure 2a.")
    add_body_paragraph(document, "Under the specified Gaussian curvature noise, mean neural top-one recovery was 82.42% (SD 0.32 percentage points; range 81.73–83.29%), while neural top-five recovery remained 100% in every trial. The stored-response FvK re-ranker achieved 99.91% mean top-one recovery (SD 0.11 percentage points; range 99.64–100.00%), 100% mean top-five recovery, and a maximum observed true rank of two. Figure 2b separates clean values from Monte-Carlo means and shows trial-standard-deviation error bars only where noise was sampled.")
    add_results_table(document, summary, hybrid)

    captions = [
        "Figure 1. Forward-prediction identity plots. The FvK panel is explicitly a stored-response identity used for solver-consistent re-ranking, not a separately learned model.",
        "Figure 2. Retrieval behavior. Counts summarize clean holdout ranks; noise points show 100-trial means with standard-deviation error bars.",
        "Figure 3. Runtime scope. The neural mark is measured; the FvK interval is an observed-throughput estimate for five parallel candidates.",
    ]
    for figure, caption in zip(figures, captions):
        document.add_picture(str(figure), width=Inches(6.8))
        document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_caption(document, caption)

    document.add_heading("Runtime, interpretation, and limitations", level=2)
    add_body_paragraph(document, "The loaded neural screen ranked all 254 candidates in 3.68 ms on average (median 4.37 ms). Observed local solver-generation throughput suggests that parallel re-ranking of five candidates would require approximately 15–30 s wall time, whereas serial nonlinear evaluation of all 254 candidates would require roughly one hour. Figure 3 uses a point for the measured neural value and an interval for the FvK estimate so that these distinct evidentiary levels are not conflated.")
    add_body_paragraph(document, "Within the tested canonical candidate library and solver-defined thermal domain, the selected neural model is a complete top-five screen under both clean and specified noisy curvature queries. The nonlinear stage is useful because it limits expensive confirmation to five candidates and resolves the remaining twist-driven ranking errors. The stored-response FvK score is solver-consistent, not independent finite-element or experimental validation. A persistent-worker FvK timing measurement and external curvature data are required before deployment, physical-accuracy, or experimental-generalization claims.")
    document.save(output)


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    reference, prediction = load_prediction_data()
    summary = load_json(MODEL_DIR / "model_summary.json")
    hybrid = load_json(HYBRID_DIR / "hybrid_noise_summary.json")
    figures = [
        REPORT_DIR / "10ply_hybrid_identity_20261029.png",
        REPORT_DIR / "10ply_hybrid_noise_top1_20261029.png",
        REPORT_DIR / "10ply_hybrid_runtime_20261029.png",
    ]
    parity_figure(reference, prediction, figures[0])
    retrieval_figure(hybrid, figures[1])
    runtime_figure(figures[2])
    markdown_path = REPORT_DIR / f"{REPORT_STEM}.md"
    markdown_path.write_text(report_markdown(summary, hybrid), encoding="utf-8")
    generate_word_report(summary, hybrid, figures, REPORT_DIR / f"{REPORT_STEM}.docx")
    print(f"Wrote {markdown_path}")
    print(f"Wrote {REPORT_DIR / f'{REPORT_STEM}.docx'}")
    for figure in figures:
        print(f"Wrote {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
