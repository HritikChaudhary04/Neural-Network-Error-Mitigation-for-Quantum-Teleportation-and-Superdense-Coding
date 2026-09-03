# Code: Neural-Network Error Mitigation for Quantum Teleportation and Superdense Coding

This repository reproduces the experiments in the manuscript. Each experiment
exists both as a plain Python script (`.py`) and as a Jupyter notebook (`.ipynb`)
with the same content.

## Files

- `superdense_finite_shot.py` / `.ipynb`
  Superdense coding under a biased Pauli channel + asymmetric readout confusion
  matrix. Compares majority vote, a channel-aware maximum-likelihood (ML)
  decoder, and a learned decoder, versus number of shots. Produces Table 1 and
  Figure 3, with 95% confidence intervals over seeds.

- `teleportation_finite_shot.py` / `.ipynb`
  Teleportation under a fixed systematic rotation (R_fix) + per-qubit jitter,
  with finite-shot tomography. Compares the uncorrected state, a least-squares
  (LS) estimate of the fixed rotation, and a learned corrector. Produces Table 2
  and Figure 4, plus the R_fix recovery check and the generalisation test
  (Sec. 4.3), with 95% confidence intervals.

## Install

    pip install -r requirements.txt

## Run

    python superdense_finite_shot.py
    python teleportation_finite_shot.py

Or open the notebooks and run all cells.

## Runtime

Each script trains a small network once per (shot setting x seed), so a full
run takes a few minutes on CPU. To reproduce the manuscript exactly, use the
seed lists as provided. For tighter confidence intervals, increase the number
of seeds (edit `SEEDS` near the bottom of each script); the means are stable and
only the intervals tighten. To shorten a quick check, reduce `SEEDS`, the shot
list, `N`, or `epochs`.

## Key results

- Superdense: the learned decoder beats majority vote but only *matches* the ML
  decoder (they coincide within confidence intervals). The network approximates
  the optimal decoder rather than exceeding it.
- Teleportation: the LS baseline already recovers most of the fidelity and
  recovers R_fix^{-1} to Frobenius distance ~0.02; the network improves on LS
  only modestly (~1-2 points), from nonlinear jitter and tomographic noise.
- Generalisation: a corrector trained on R_fix does not transfer to a different
  fixed rotation, confirming it learned the specific calibration.

## Reproducibility note

Small numerical differences from the manuscript tables are possible depending on
library versions and the number of seeds used, but the qualitative conclusions
(ML == learned for superdense; LS ~ learned for teleportation) are robust.
