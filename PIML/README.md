# Physics-structured layup retrieval: 8 and 10 plies

This directory contains the retained forward neural-retrieval workflow for
canonical 8- and 10-ply composite laminates. Given a measured curvature vector
and a prescribed through-thickness temperature field, it evaluates every
canonical candidate layup with a physics-structured residual-curvature
surrogate and returns the lowest-mismatch candidates.

The forward model is deliberately factorised:

1. Classical laminate theory (CLT) supplies the linear curvature state.
2. A neural network predicts only the nonlinear free-free correction, using
   the one-hot stacking sequence, squared temperature difference, and
   curvature-per-gradient state.
3. The correction is multiplied by the signed temperature difference, which
   preserves thermal-reversal parity. Candidates are ranked from the
   component-scaled curvature mismatch.

## Retained data and results

- `artifacts/datasets/` holds the compact 8- and 10-ply FREEFREE datasets.
- `artifacts/experiments/8ply_direct_residual_sweep_20261026/` holds the
  selected 8-ply forward-model sweep.
- `artifacts/experiments/10ply_direct_residual_top5_20261028/` holds the
  orbit-disjoint 10-ply architecture study and diagnostic final holdout.
- `artifacts/experiments/10ply_random_case_80_20_20261028/` holds the
  selected 10-ply case-level interpolation model, for which top-5 retrieval
  is 205/205 on the held-out cases.
- `artifacts/reports/` contains the paper-ready Word report and retrieval
  figure. The source text is `../NN_METHODS_AND_EVALUATION.md`.

The random case-level result is an interpolation result: each test layup is
also present in training at other thermal fields. The orbit-disjoint study is
the appropriate evidence for unseen-layup behavior.

## Reproduce the retained workflow

Generate a compact FvK/FREEFREE dataset (resumable and duplicate-safe):

```powershell
python -m PIML.scripts.generate_fvk_dataset --help
```

Train a direct residual model (the script supports both 8 and 10 plies):

```powershell
python -m PIML.scripts.train_direct_residual_nn --help
```

Rank all canonical candidates with a saved model (repeat `--checkpoint` to
form a mean ensemble):

```powershell
python -m PIML.scripts.evaluate_residual_topk --help
```

Run mechanics checks with:

```powershell
python -m unittest PIML.tests.test_physics
```
