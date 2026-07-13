# Antiphon Status — 2026-07-12 (end of session 1)

## The goal (cleared for tonight, resume tomorrow)

Build the full physics-to-ML pipeline end-to-end: FDTD solver, FxLMS
controller, synthetic data generator, foundation model v1, and close the
loop (model-predicted secondary paths driving FxLMS at >= 80% of the dB
reduction that measured paths achieve). Everything tested, deterministic,
CPU-only, honest results in docs/results.md.

## Done and verified (43 tests passing: `uv run pytest`)

1. **Coordinate fix.** Street centerline is y=0 everywhere (the reference
   script had speakers outside the domain and reflections off wrong walls).
   Figures in docs/figures/ regenerated.
2. **FDTD solver** (`simulation/fdtd.py`). Yee staggered grid, split-field
   PML, rigid + impedance walls, regularized band-limited impulse-response
   extraction. Validated against exact 2D Hankel solutions: open-field max
   amplitude error **0.13 dB**, single-wall interference within **1 dB**,
   phase within 0.2 rad. Throughput ~1300 steps/s on the 400x300 grid
   (float32), real-time factor 0.24x.
3. **Materials + broadband sources** (`materials.py`, `sources.py`).
   Absorption -> impedance conversion; traffic/HVAC/construction generators,
   octave-band metrics. FDTD confirms glass absorbs more than concrete at
   125 Hz.
4. **Multi-channel FxLMS** (`anc/fxlms.py`). NLMS-normalized filtered-x,
   M refs / J speakers / K error mics, plus Eriksson online secondary-path
   identification (converges to <0.1 misalignment). **40+ dB** tone
   reduction on FDTD-measured street-canyon paths (criterion was 10 dB).
5. **Data generator** (`model/dataset.py`, `scripts/generate_training_data.py`).
   Randomized canyons (width 8-30 m, facade absorptions, source position),
   64 log-spaced freqs 30-430 Hz, 24 receivers/scene, HDF5, seeded and
   deterministic, ~2.1 s/scene/core.
6. **Model v1** (`model/architecture.py`): CNN geometry encoder +
   Fourier-feature queries + cross-attention, 10.4M params. Training
   pipeline (`model/train.py`) with scene-level splits and two baselines
   (train-mean H; distance-calibrated free-field Hankel).
   **Key design decision:** targets are delay-compensated (H * e^{+ikr})
   so phase is smooth across the sparse frequency grid; delay reapplied at
   inference. MSE comparisons unaffected (unit-modulus rotation).
7. **Sparse-H -> FIR chain** (`model/inference.py`) validated model-free:
   64 sparse samples reconstruct the measured IR with >0.95 correlation,
   <0.3 rad phase and <1 dB magnitude error at 150/250 Hz. If the model
   predicts H well, the FIR conversion won't be the weak link.
8. **Closed-loop harness** (`scripts/evaluate_closed_loop.py`) written and
   its measured-path arm verified on a held-out scene.

## In flight / interrupted

- **Training data generation was killed ~100/4000 scenes in** (machine was
  needed). Nothing partial saved; fully restartable:
  `nice -n 19 uv run python scripts/generate_training_data.py --scenes 4000
   --out data/synthetic/train.h5 --workers 6 --seed 0`
  (~25 min on 6 idle cores; scales linearly with workers on a cloud box).
  `data/synthetic/pilot.h5` (20 scenes) exists for smoke-testing the
  training loop.

## Findings / tuning notes for tomorrow

- FxLMS step size on the 4-speaker/4-mic held-out config: **mu=0.005
  diverges; mu=0.001 is stable** (23/18/8/13 dB at 150 Hz on scene seed
  100000). `scripts/evaluate_closed_loop.py` still has MU=0.005 — change to
  0.001, and consider longer runs (5 s) since mic 3 sat at 8.4 dB after 3 s.
  Leak made no difference.
- Classical phase-inversion "ANC" in the legacy analytical demo never
  reduces average pedestrian pressure (the old sweep figure clamps
  negatives to zero). The honest baseline is FxLMS.
- The handoff's 10x real-time FDTD target is not met in pure numpy (0.24x);
  numba/C would be needed. Not blocking anything.

## Remaining plan (tasks #5-8)

1. Regenerate 4000 scenes (task #5, command above).
2. Train: `uv run python scripts/train_model.py --data data/synthetic/train.h5
   --out data/runs/v1 --epochs 30 --threads <n>` — smoke-test 2 epochs on
   pilot.h5 first. Must beat both baselines on val (printed each epoch).
3. Closed loop: `uv run python scripts/evaluate_closed_loop.py` (fix MU
   first). Target mean ratio >= 0.8 vs measured paths.
4. Write docs/results.md with every number above, plus failure modes;
   regenerate figures (add FDTD validation + closed-loop comparison plots).

## Machine etiquette (important)

Cap parallel workers at ~6 of Roman's 16 cores, `nice -n 19`, cap torch
threads, and tell him before starting anything heavy. Tomorrow this moves
to a cloud machine where full parallelism is fine.
