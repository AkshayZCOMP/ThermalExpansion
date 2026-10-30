# Remote 10-ply data-generation bundle

This bundle computes the first half (indices 0--2,031) of the complete
10-ply, 16-field FREEFREE dataset. The local computer is computing indices
2,032--4,063.

## Setup

Install Python 3.10 or newer, then run:

```powershell
python -m pip install -r requirements.txt
```

The current solver runs on the CPU; a CUDA build of PyTorch is not required.

## Run

From this directory, run:

```powershell
.\run_remote_10ply_half.ps1
```

The output is written to:

`PIML\artifacts\datasets\freefree_10ply_remote_half_0_2031.csv`

When it finishes, transfer only that CSV back to the primary computer. It
will be merged with the local half while rejecting duplicate layup/temperature
conditions.
