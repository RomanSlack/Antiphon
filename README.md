# Antiphon

Urban active noise cancellation and an acoustics foundation model: an ML model
that predicts sound propagation in urban geometries in real time. The ANC
corridor installation is the proving ground; the foundation model API is the
product.

Third Axis AI Consulting / 316 Group.

## What works today

- **2D FDTD acoustic solver** (Yee staggered grid, split-field PML, rigid and
  impedance building walls). Validated against exact 2D analytical solutions:
  max 0.13 dB open-field amplitude error, single-wall interference within
  1 dB.
- **Facade materials** (concrete/glass/brick absorption to impedance) and
  **broadband noise sources** (traffic, HVAC, construction) with octave-band
  metrics.
- **Multi-channel FxLMS controller** with Eriksson online secondary-path
  identification: 40+ dB tone reduction on FDTD-measured street-canyon paths.
- **Synthetic data generator**: randomized street canyons to complex transfer
  functions H(f) at 64 frequencies (30-430 Hz), HDF5, fully seeded.
- **Foundation model** (9.4M-param PyTorch: CNN geometry encoder +
  cross-attention queries) trained on 12,000 synthetic scenes; beats
  physics and statistical baselines by 2.4x on unseen geometries.
- **Closed loop proven**: FxLMS driven by model-predicted secondary paths
  achieves 90% of measured-path cancellation (capped metric) across 10
  held-out scenes, beating measured paths outright in several. Full
  numbers: `docs/results.md`.
- **Fast analytical solver** (Green's functions + image sources) for
  interactive demos, refactored from the original prototype.

**Current state:** v1 pipeline complete; all success criteria met. See
`docs/results.md` for the full evaluation and `docs/STATUS.md` for history.

## Layout

```
src/antiphon/
├── simulation/     # Acoustic solvers and physics
│   ├── geometry.py     # UrbanGeometry, constants (y=0 = street centerline)
│   ├── fdtd.py         # FDTD solver (ground truth)
│   ├── analytical.py   # Green's function solver (fast demos)
│   ├── materials.py    # Absorption -> impedance
│   ├── sources.py      # NoiseSource, SpeakerArray, broadband generators
│   └── metrics.py      # Quiet zone, octave-band levels
├── anc/
│   └── fxlms.py        # Multi-channel FxLMS + online secondary-path ID
├── model/
│   ├── dataset.py      # Scene randomization + HDF5 dataset
│   ├── architecture.py # AcousticsModelV1 (10.4M params)
│   ├── train.py        # Training loop, baselines, scene-level splits
│   └── inference.py    # H prediction, sparse-H -> FIR filters
└── viz/                # Field plots and performance charts
scripts/            # CLI entry points (simulation, data gen, training, eval)
tests/              # 43 tests incl. FDTD-vs-analytical validation
refs/               # Original handoff material (do not modify)
docs/               # Proposal, status, figures
```

## Quickstart

```bash
uv sync
uv run pytest                                     # full test suite
uv run python scripts/run_simulation.py           # 3-panel ANC comparison
uv run python scripts/run_simulation.py --sweep   # frequency sweep

# Full pipeline (compute-heavy; use a machine you can saturate)
uv run python scripts/generate_training_data.py --scenes 4000 --workers 6
uv run python scripts/train_model.py --data data/synthetic/train.h5
uv run python scripts/evaluate_closed_loop.py --ckpt data/runs/v1/best.pt
```

Figures are written to `docs/figures/`.

## Roadmap

See `refs/URBAN_ANC_HANDOFF.md` for the full engineering plan and
`docs/STATUS.md` for current progress:

1. ~~Restructure reference script into this package~~ (done, parity-tested)
2. ~~2D FDTD wave solver (Yee grid, PML boundaries)~~ (done, validated)
3. ~~Material absorption + broadband noise sources~~ (done)
4. ~~Multi-channel FxLMS controller~~ (done)
5. ~~Synthetic training data generation~~ (done: 12k scenes / 288k samples)
6. ~~Foundation model training~~ (done: v2, val MSE 0.755 vs 1.85 baseline)
7. ~~Closed-loop evaluation~~ (done: 90% of measured-path performance)
8. ~~Results report + figures~~ (done: docs/results.md)
9. Next: broadband closed loop, 3D, web demo / investor materials
