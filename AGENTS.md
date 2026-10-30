# Repository Working Agreement

## Project status is mandatory

Before completing any substantive work in this repository, update
`PROJECT_STATUS.md`. Record what changed, relevant results or validation, and
the next concrete work items. Mark completed items with `- [x]`, add newly
identified work as `- [ ]`, and keep the status date current. Do not erase
historical context merely because a task is complete.

## Abaqus run artifacts

Do not commit Abaqus-generated solver outputs or replay files. Preserve
compact CSV summaries, scripts, checkpoints, and other deliberate research
artifacts; treat raw solver files as disposable unless the user explicitly
asks to retain them.

## Final PIML report requirements

For any final PIML methods/results report, include retrieval and prediction
results both without added noise and with a documented Gaussian curvature-noise
perturbation. Include a query-time comparison that states what is timed and
whether model loading, I/O, and upstream curvature extraction are excluded.

Keep Methods concise and organize them in this order:

1. **Input data and output:** candidate/design domain, response labels or
   measurements, thermal/loading conditions, preprocessing, and the model
   output/decision task.
2. **Model architecture:** physics baseline or constraints, features,
   network structure, and candidate-ranking rule.
3. **Training:** loss function, optimizer, hyperparameters, augmentation, and
   any model-selection procedure.
4. **Testing:** held-out split/grouping and leakage controls, baselines,
   prediction and retrieval metrics, Gaussian-noise protocol, runtime scope,
   and random seeds/software/hardware needed for reproducibility.

State the data scope and experimental/computational boundary plainly. Do not
repeat implementation detail that does not affect reproducibility or the
interpretation of results.
