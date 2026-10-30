# HeatXferDIC Project Status

Last updated: 2026-10-30

## Purpose

This repository studies thermally induced deformation of composite laminates
and the inverse problem of recovering an 8-ply layup from its response. It
combines classical laminate theory (CLT), a nonlinear Föppl--von Kármán (FvK)
Ritz/free-free solver, Abaqus reference sweeps, comparison tooling, and
physics-informed machine-learning (PIML) surrogates.

## Repository map

- `Computational Real/`: nonlinear FREEFREE/FvK solver and comparison and
  confirmation scripts.
- `AbaqusSweep/`: Abaqus generation/sweep workflow for 3--10 ply symmetric,
  balanced and unidirectional cases; compact Abaqus result CSVs and the sweep
  checkpoint are retained here.
- `LayupSweep/`: layer-effect/bifurcation work and regenerated compact
  comparison tables.
- `PIML/`: 8-ply forward and inverse neural surrogates, physics utilities,
  datasets, tests, checkpoints, validation, and experiment artifacts.
- `Plotting/`: curvature/deformation uniqueness analysis and export helpers.
- `Archive/`: the original CLT/thermal-diffusion implementation, tests, and
  historical documentation.

## Work completed

- [x] Implemented the baseline composite-laminate CLT workflow, including
  material/layup definitions, transformed stiffness, ABD matrices, thermal
  loads, resultant strains/curvatures, and ply-level strains.
- [x] Added thermal-diffusion support and historical unit/integration tests;
  the original project documentation records 44 passing tests.
- [x] Built the nonlinear FREEFREE/FvK Ritz workflow used for finite,
  free-free laminate response and Abaqus comparison.
- [x] Ran and compacted the Abaqus layup sweep. The retained CSVs and
  `layup_sweep_checkpoint.json` preserve numerical results and run progress
  without raw solver output databases.
- [x] Regenerated the 130 GPa material data and layer-effect comparison
  tables, including computational-result columns.
- [x] Produced PIML datasets and a selected physics-only 8-ply forward model.
  The selected no-gradient-expert checkpoint was trained on 2,448 local-energy
  targets over 34 temperature fields; its corrected FREEFREE validation covers
  16 converged cases and has active-component RMSEs of 0.002815 (`kappa_x`),
  0.001437 (`kappa_y`), and 0.000591 1/m (`kappa_xy`).
- [x] Implemented direct residual-curvature and inverse layup neural-network
  training/evaluation workflows, including symmetry/twist-expert variants.
- [x] Completed the 2026-10-26 direct-residual 8-ply sweep. The best recorded
  configuration is `b_split_256x4`: on 192 validation cases its active
  component RMSE is 0.03529 1/m, versus 0.06401 1/m for linear CLT.
- [x] Cleared disposable Abaqus solver artifacts on 2026-10-26: raw `.odb`,
  `.inp`, `.dat`, `.msg`, `.sta`, `.com`, `.prt`, logs, replay, and environment
  outputs were removed; compact results, source, checkpoints, and PIML model
  artifacts remain.
- [x] Audited neural-network performance and split integrity; the detailed
  results and paper-ready methods text are in `NN_METHODS_AND_EVALUATION.md`.
  The PIML mechanics/inverse-physics suite passed 14/14 checks.
- [x] Added resumable, duplicate-safe generation support to
  `PIML/scripts/generate_fvk_dataset.py`. It keys records by layup and direct
  face temperatures, allowing an interrupted deterministic grid to append
  only missing cases.
- [x] Completed the 10-ply FREEFREE data collection on 2026-10-28: 1,024
  unique, converged records covering 64 sampled canonical layups across 16
  direct face-temperature fields. No duplicate layup/field keys were found.
- [x] Generalized `PIML/scripts/train_direct_residual_nn.py` from its fixed
  8-ply encoding to a validated 1--10-ply encoding while retaining its
  8-ply default, and added `evaluate_residual_topk.py` to measure exact
  candidate-retrieval rank over the canonical design space.
- [x] Completed the nested 10-ply architecture sweep. The selected 256x4
  residual-curvature model retrieved the correct layup within the top five
  for all 160 orbit-disjoint validation cases when ranked against all 254
  canonical 10-ply candidates (worst true rank: 5).
- [x] Audited the 10-ply final-holdout split and retrieval logic. The 50/14
  layup split has zero exact-layup or D4 symmetry-orbit overlap, and symmetry
  augmentation introduces no final-test layup into training. The candidate
  enumerator correctly ranks all 254 canonical 10-ply layups.
- [x] Completed the requested 10-ply random case-level 80/20 interpolation
  experiment. The 819/205 split places every test layup in training at other
  thermal fields; the forward 256x4 residual model achieved 204/205 exact
  top-1 and 205/205 top-5 recovery over all 254 canonical candidates, with
  active-component RMSE 0.001476 1/m versus 0.040236 1/m for CLT. The results
  are documented as interpolation, not unseen-layup generalization.
- [x] Updated `NN_METHODS_AND_EVALUATION.md` and generated the Word report
  `PIML/artifacts/reports/10ply_nn_methods_and_results_20261028.docx`, with
  journal-style methods, retrieval and prediction-comparison figures, split
  definitions, computational timing, and deployment caveats.
- [x] Revised the methods/results write-up to report only the requested random
  80/20 case-level interpolation experiment, including its 204/205 top-one,
  205/205 top-five, curvature-error, and loaded-query timing results. The
  companion Word report is
  `PIML/artifacts/reports/10ply_random_case_methods_and_results_20261028.docx`.
- [x] Assessed small curvature-measurement noise on the random 10-ply test
  partition using independent Gaussian noise (SD 0.001 1/m per component).
  Across 100 trials (20,500 queries), mean top-one recovery was 99.46% and
  top-five recovery was 100% in every trial; the worst true rank was two.
  The reproducible summary is
  `PIML/artifacts/experiments/10ply_random_case_80_20_20261028/forward_dense_256x4/noise_sigma_0p001_100trials.json`.
- [x] Consolidated unperturbed and noisy random-case results in the methods
  narrative and summary table, while retaining only the unperturbed
  prediction-comparison figure. The corresponding Word report is
  `PIML/artifacts/reports/10ply_random_case_clean_and_noise_results_20261028.docx`.
- [x] Removed 30 retired scripts from `PIML/scripts/`, including superseded
  inverse, energy-target, specialist-branch, repair, tuning, and obsolete
  validation workflows. The retained scripts cover FREEFREE generation and
  merging, completeness checking and splitting, direct-residual training,
  prediction/ranking, and curvature-noise evaluation; no data, checkpoints,
  plots, or reports were deleted.
- [x] Validated and trained on the delivered full 10-ply remote CSV on
  2026-10-29. It contains all 4,064 canonical layup/thermal-field conditions
  (254 layups x 16 fields), but the received file also had one pasted solver
  error line, 32 exact restart duplicates, and one condition flagged
  `converged=False`. The duplicate-safe cleaned copy is
  `PIML/artifacts/datasets/freefree_10ply_254layups_16fields_clean.csv`; the
  retained loaders correctly fit on its 4,063 converged labels. A 256x4
  residual model trained on a 58/14 symmetry-orbit split achieved 82.69%
  top-1 and 100.00% top-5 recovery over 832 held-out cases (worst rank 4),
  with active-component RMSE 0.01412 1/m versus 0.02577 1/m for linear CLT.
  A final 3,000-step deployment fit on all trusted labels is at
  `PIML/artifacts/experiments/10ply_full_dataset_20261029/full_data_dense_256x4/model_checkpoint.pt`.
  Its 100% top-1/top-5 score on the former holdout is a fit smoke test only,
  because those labels were included in that final fit.
- [x] Added a journal-style, evidence-bounded Methods and Results draft for
  the completed full 10-ply study at
  `PIML/artifacts/reports/10ply_full_dataset_methods_and_results_20261029.md`
  and the formatted Word companion
  `PIML/artifacts/reports/10ply_full_dataset_methods_and_results_20261029.docx`.
  It distinguishes the 832-case orbit-disjoint result from final-fit
  diagnostics and specifies the data-audit, model, split, and ranking
  protocols needed for reproducibility. Three publication-oriented,
  orbit-holdout figures (parity, absolute-error distribution, and retrieval
  ranks) were added to the report folder and embedded in the Word document.
  The Methods section was then condensed to the essential data provenance,
  residual-surrogate specification, split, and retrieval protocol.
- [x] Diagnosed the dominant `kappa_xy` error in the full 10-ply
  orbit-disjoint result. The error is confined to off-axis layups: its active
  RMSE is 0.02263 1/m for the 768 off-axis cases but 0.00077 1/m for the 64
  0/90-only cases. It is largest at the eight high-gradient fields (RMSE
  0.02867 1/m) and concentrated in two unseen layup families, rather than in
  temperature-offset dependence. A targeted off-axis/twist model should be
  selected only using a training-derived nested validation split, leaving the
  final 832-case orbit holdout untouched.
- [x] Completed the first nested 10-ply twist-improvement screen on
  2026-10-29. On 672 inner orbit-disjoint validation cases, the baseline
  dense model had the best active `kappa_xy` RMSE (0.02229 1/m); 3x twist
  weighting gave 0.02281 1/m and the 3x-weighted off-axis expert gave
  0.02542 1/m. Neither candidate improved the target twist metric, so neither
  was evaluated on the frozen 832-case final holdout. The expert did improve
  inner top-1 retrieval to 95.24% (baseline 79.76%), while retaining 100%
  top-5; this is a separate retrieval finding, not evidence of improved
  `kappa_xy` prediction.
- [x] Completed a second nested twist-only screen on 2026-10-29. A dedicated
  512x5 `kappa_xy` residual network improved active twist RMSE on the same
  672 inner orbit-disjoint cases from 0.02229 1/m (three-output baseline) to
  0.01989 1/m, an 10.8% reduction; MAE decreased from 0.01083 to 0.00856
  1/m. The off-axis-split twist-only counterpart was weaker (0.02080 1/m).
  The selected dense twist-only model predicts only `kappa_xy`, so a final
  retrieval model must combine it with the baseline model's `kappa_x` and
  `kappa_y` predictions before its one-time frozen-holdout evaluation.
- [x] Compared the dense twist-only expert with and without hard 0/90 gating
  on the 672 inner validation cases. Applying the expert to every layup gave
  active `kappa_xy` RMSE 0.019887 1/m and 77.38% top-1/100% top-5 retrieval;
  hard-gating it to +/-45-containing candidates retained the same retrieval
  and active twist RMSE, while reducing the 0/90-only twist RMSE from 0.000962
  to exactly zero and lowering overall twist MAE from 0.008562 to 0.008452
  1/m. The hard-gated hybrid is therefore the preselected final configuration.
- [x] Completed the one-time frozen outer-holdout evaluation of the
  preselected hard-gated hybrid on 2026-10-29. The twist expert did not
  transfer its nested-validation gain: active `kappa_xy` RMSE was 0.02558
  1/m, versus 0.02263 1/m for the original outer model. It is not used for
  the primary result. The primary 256x4 residual model remains the strongest
  frozen-holdout model (82.69% top-1, 100.00% top-5, active RMSE 0.01412
  1/m); all 144 non-top-one queries contain +/-45-degree plies. Its loaded
  254-candidate ranking time was 3.68 ms mean over 50 CPU repetitions.
- [x] Added a concise submission-style report, with primary frozen-holdout
  methods, curvature-error results, runtime, rank trends, and three figures,
  at `PIML/artifacts/reports/10ply_orbit_holdout_submission_style_20261029.md`
  and `.docx`.
- [x] Added normalized candidate-specific A/B/D coupling descriptors (six
  independent terms each from A, B, and D) as an optional direct-residual NN
  input, with checkpoint and candidate-ranking support. The 256x4
  ABD-augmented model improved inner orbit-validation active `kappa_xy` RMSE
  from 0.02229 to 0.02088 1/m, but did not transfer to the frozen outer
  holdout: active `kappa_xy` RMSE was 0.02516 1/m and top-1 78.85%, versus
  0.02263 1/m and 82.69% for the primary model. Top-5 remained 100%; retain
  the original primary model and treat ABD augmentation as a negative
  generalization result.
- [x] Evaluated paired low/high thermal-gradient retrieval without retraining.
  The inner orbit-validation partition gave 336/336 top-1 paired queries, but
  the frozen outer holdout gave 328/416 top-1 (78.85%), 416/416 top-5, and
  worst rank four. This did not improve on single-field outer retrieval
  (82.69% top-1, 100% top-5); the off-axis prediction error is correlated
  across the two sampled gradient magnitudes. Added the reproducible
  `evaluate_two_gradient_retrieval.py` evaluator.
- [x] Evaluated and documented a two-stage neural--FvK retrieval concept.
  The frozen holdout neural screen placed the truth in its top five for all
  832 clean cases; solver-consistent re-ranking with stored FREEFREE responses
  gave 100% clean top-1. Under independent Gaussian curvature noise (SD
  0.001 1/m; 100 trials), neural top-1 was 82.42% mean (top-five 100% in all
  trials), and the solver-consistent hybrid top-1 was 99.91% mean (range
  99.64--100%). The Word and Markdown report
  `PIML/artifacts/reports/10ply_neural_fvk_hybrid_methods_results_20261029.*`
  includes methods, identity plots, clean/noisy results, and timing scope.
  The nonlinear re-rank is an oracle replay of stored solver responses; a
  persistent-worker online FvK benchmark and independent validation remain
  required before deployment claims.
- [x] Completed three additional nested-screen attempts for off-axis twist
  behavior: D4 group averaging, CLT membrane-strain-per-gradient features,
  and their combination. On 672 inner orbit-disjoint cases, the combination
  reduced `kappa_xy` RMSE from 0.02071 to 0.01687 1/m and improved top-1 from
  79.76% to 83.33%, with top-5 remaining 100%. On the repeatedly inspected
  outer partition, the strain+D4 model improved retrieval to 704/832 top-1
  (84.62%), 100% top-5, and worst rank three, but its `kappa_xy` RMSE (0.02220
  1/m) did not improve on the original (0.02174 1/m). Treat the outer result
  as exploratory because that partition has already informed several model
  iterations. Added optional CLT membrane-strain features to the direct
  residual trainer and candidate evaluator.
- [x] Screened Gaussian target-noise augmentation and a three-seed
  membrane-strain ensemble on the inner orbit-validation partition. Gaussian
  target noise (SD 0.001 1/m) gave active `kappa_xy` RMSE 0.01851 1/m versus
  0.01849 for the corresponding clean single model. The three-seed ensemble
  gave 0.01831 1/m without D4 averaging and 0.01825 with it, both worse than
  the selected single strain+D4 model (0.01687). Do not add label-noise
  augmentation or this ensemble to the primary configuration.
- [x] Added a persistent final-report requirement to `AGENTS.md`: report both
  unperturbed and documented Gaussian curvature-noise results, and include a
  query-time comparison with its timing scope stated explicitly.
- [x] Added a concise final-report Methods template to `AGENTS.md`, covering
  input/output data, architecture, training, and testing; it also requires
  preprocessing, split/leakage controls, baselines, metrics, uncertainty,
  runtime scope, and reproducibility information where relevant.

- [x] Tested a D4-symmetrized linear fusion of the original and
  membrane-strain models on nested validation. Increasing the strain-model
  weight monotonically improved twist RMSE and top-1; the best result was the
  strain+D4 model alone (0.01687 1/m `kappa_xy` RMSE, 83.33% top-1). The
  original model supplied no complementary inner-validation signal, so no
  fused configuration was carried to the outer partition.
- [x] Tested exact thickness-reversal augmentation restricted to laminates
  containing a +/-45-degree ply. Reversing the ply order while exchanging the
  two face temperatures negates the curvature labels (audit RMSE
  7.16e-09 1/m; maximum error 1.41e-07 1/m); no reversed training layup
  leaked into the validation layups. With the existing membrane-strain
  features and exact D4 inference averaging, the nested 672-case validation
  result improved to 604/672 top-1 (89.88%), 100% top-5, and `kappa_xy` RMSE
  0.016828 1/m (active: 0.018177 1/m). The corresponding repeatedly inspected
  832-case outer result was 736/832 top-1 (88.46%; +32 cases versus the
  prior strain+D4 model), 100% top-5, and worst rank three. It did **not**
  improve twist prediction: D4 `kappa_xy` RMSE was 0.022233 1/m (active:
  0.023141 1/m), marginally above the prior strain+D4 values of 0.022203 and
  0.023110 1/m. Retain it as a retrieval-oriented experimental variant, not
  the selected `kappa_xy` predictor. The direct trainer now exposes the
  off-axis-only reversal flag, and `evaluate_residual_topk.py` records
  explicit D4-averaged prediction metrics.
- [x] Rebuilt the neural--FvK hybrid Methods and Results report on
  2026-10-30 from the frozen `orbit_dense_256x4` holdout artifacts. The
  updated Markdown and Word versions at
  `PIML/artifacts/reports/10ply_neural_fvk_hybrid_methods_results_20261029.*`
  now specify the response-data layout, 58/14-orbit (202/52-layup) split and
  leakage controls, 44-input/209,667-parameter residual MLP, parity
  factorization, normalization, loss, optimizer, full-batch schedule,
  augmentation, ranking score, solver-consistent FvK boundary, clean/noisy
  retrieval, metrics, limitations, and timing scope. Its three figures
  were regenerated in the compact journal visual style of `Plotting/Plotting
  cleaned.ipynb`: component-aware parity/oracle-boundary panels, clean rank
  counts with noisy top-one uncertainty, and a measured-versus-estimated
  log-time interval. `generate_10ply_hybrid_report.py` provides the
  reproducible report/figure build; source data and numerical claims were
  unchanged.
- [x] Added a compact forward-prediction comparison brief at
  `PIML/artifacts/reports/10ply_clt_and_correction_brief_20261030.*`. It uses
  the same frozen 832-case orbit-disjoint holdout to compare pure CLT against
  CLT plus the selected learned residual, with a six-panel matched-axis
  identity figure (x-axis: true curvature) and componentwise held-out
  R-squared/RMSE/MAE table. The
  learned correction improved R-squared from 0.9962 to 0.9996 (`kappa_x`),
  0.9962 to 0.9995 (`kappa_y`), and 0.9751 to 0.9918 (`kappa_xy`). The
  reproducible generator is `PIML/scripts/generate_10ply_clt_correction_brief.py`.

## Findings and caveats

- The selected forward PIML model is physics-only: Abaqus and FREEFREE data
  were not used to fit or choose its parameters. FREEFREE is used for the
  corrected validation reported above.
- For 0/90-only laminates, the workflow enforces the reflection-symmetric free
  branch and therefore `kappa_xy = 0`; layups containing +/-45 degree plies
  retain nonlinear twist.
- The newest direct-residual sweep improves substantially on linear CLT, but
  it is not yet the same selected checkpoint documented in `PIML/README.md`.
  Selection and independent validation remain necessary.
- The 2026-10-26 inverse top-5 experiments produced candidate rankings, but
  their orbit-disjoint validation split had no exact top-1 or top-5 recovery.
  Treat the direct inverse NN as experimental rather than validated. A separate
  forward-model-assisted discrete retrieval did recover all 224 cases in its
  frozen orbit-disjoint holdout, but that is not direct-NN performance.
- Older 64-layup direct-residual and random-case inverse scores are not valid
  independent-test claims because symmetry-orbit and exact-layup overlap,
  respectively, was found during the audit.
- The working tree already contains extensive user changes/untracked research
  material and removals predating this status file. Do not assume the root
  `README.md` accurately reflects the current layout; consult this file and
  the subproject documentation first.

## Next work

- [ ] Re-run the one 10-ply condition flagged nonconverged in the delivered
  dataset (`[-45/45/-45/0/45/45/0/-45/45/-45]`, T_top=83.75,
  T_bottom=46.25) if a fully converged 4,064-label matrix is required.
- [ ] Improve off-axis `kappa_xy` generalization with a nested-validation
  comparison of a symmetry-equivariant representation or targeted additional
  off-axis labels. Neither the twist-only specialist nor normalized A/B/D
  descriptors transferred their nested-validation gains to the frozen outer
  holdout; retain the original primary model until a newly preselected model
  surpasses it on an independently reserved evaluation set.
- [ ] If the targeted thickness-reversal model is to be adopted for final
  retrieval, repeat its one-time comparison on a newly reserved, never
  inspected layup-orbit holdout. The current outer partition has already
  informed prior exploratory model iterations; the observed top-1 gain is not
  a fresh deployment claim.
- [ ] Complete the full canonical 10-ply 16-field dataset (4,064 cases). On
  2026-10-28, deterministic multi-field chunking was added to
  `generate_fvk_dataset.py`, including `--all-ply --nplies 10` support. The
  local half (indices 2,032--4,063) was launched with `--resume`, leaving
  1,520 cases after reuse of the 1,024 existing records; the remote half
  (indices 0--2,031) should be generated into a separate CSV and merged with
  duplicate protection after transfer. `Remote10Ply.zip` packages the solver,
  canonical-layup enumerator, required PIML modules, dependencies, and an
  executable remote-half command. A duplicate unlogged local launcher was
  stopped after it wrote 31 repeated conditions; the logged local launcher
  was paused at the user's request after preserving 1,070 unique local
  conditions. The final duplicate-safe merge will remove repeated rows before
  any resumed collection or model retraining.
- [ ] Select a forward direct-residual checkpoint using a documented criterion,
  then validate it on a held-out temperature/layup domain and compare it with
  the established no-gradient-expert model.
- [ ] Freeze a forward checkpoint and reproduce its candidate-retrieval result
  with a command/manifest before using it in a paper.
- [x] Resumed and completed the interrupted 10-ply FREEFREE generation. The
  2026-10-27 resume preserved its 641 existing records and generated the 383
  deterministic remainder; on 2026-10-28 the resulting 1,024-case sample was
  validated for completeness, unique keys, and convergence.
- [ ] Improve the 10-ply residual surrogate before any success claim. Its
  final 14-layup orbit holdout has 208/224 top-5 recovery (92.86%; top-1
  75.00%), not the requested 100%. All 16 misses are two off-axis layups
  under the eight negative-gradient fields, with true ranks 7 or 8. Diagnose
  that off-axis generalization gap using additional labels or a targeted model
  change, then repeat selection without using this final holdout for tuning.
  The initial check confirmed that the signed-gradient factorization is
  physically exact for this dataset (relative reversal error below 3e-8); two
  additional seed models were tested as a validation-selected forward ensemble
  rather than weakening that constraint. The three-seed ensemble retained
  160/160 top-5 recovery on nested validation (top-1 65.0%; worst rank 5),
  and its two additional full-training seed members are now running for a
  diagnostic reassessment of the already-observed final holdout.
- [x] Evaluated a direct 10-ply inverse NN baseline. The inverse
  training/evaluation scripts were generalized from 8 to 10 plies with
  8-ply defaults preserved. On the 160-case orbit-disjoint nested validation
  set, both dense and twist-gated variants had 0% exact top-1 and 10% exact
  top-5 recovery over 254 candidates despite 100% training-ply accuracy.
  The direct inverse is therefore a memorizing baseline, not a replacement
  for forward candidate ranking.
- [ ] Optionally run the direct inverse baseline on the random case-level
  split. It is not needed for the forward-ranker result and must likewise be
  labelled interpolation-only.
- [ ] Diagnose inverse-model ambiguities and improve exact top-1/top-k recovery
  on an orbit-disjoint test set; define acceptance criteria before claiming
  inverse design performance.
- [ ] Re-run the relevant PIML physics and inverse test suites after any model
  or data-pipeline changes.
- [ ] Decide whether the baseline CLT source should be restored from `Archive/`
  into the active root layout, or whether the root README should be rewritten
  to explicitly point to the new research structure.
- [ ] Keep raw Abaqus runs outside the repository (or delete them after compact
  summaries are verified); retain only reproducible inputs and compact outputs.
- [ ] Confirm the proposed PIML-only cleanup before irreversible removal. The
  refreshed `PIML/README.md` now documents the retained 8/10-ply direct
  residual workflow; the proposed deletion set consists of superseded
  physics-only, inverse, and expert-model experiment families, their
  supporting scripts, and generator logs. No cleanup deletion has been made
  yet.

## How to resume

Start with `PIML/README.md` for the selected 8-ply forward model and validation
commands. Use `PIML/artifacts/experiments/8ply_direct_residual_sweep_20261026/`
for the newest forward-sweep reports, `AbaqusSweep/ABaqusData*.csv` for compact
Abaqus results, and `LayupSweep/LayerEffect*.csv` for layer-effect results.

When work changes the project state, update this file in the same change:
move finished tasks to the completed list, preserve the rationale/results, and
add the next actionable tasks.
