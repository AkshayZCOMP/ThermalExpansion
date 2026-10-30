# Run from the Remote10Ply folder after installing requirements.txt.
# This computes the first 2,032 cases of the deterministic 4,064-case grid.
python -m PIML.scripts.generate_fvk_dataset `
  --all-ply `
  --nplies 10 `
  --midplane-temperatures 35,45,55,65 `
  --thermal-gradients=-50000,-30000,30000,50000 `
  --case-start 0 `
  --case-stop 2032 `
  --workers 16 `
  --output PIML\artifacts\datasets\freefree_10ply_remote_half_0_2031.csv `
  --no-timestamp-output
