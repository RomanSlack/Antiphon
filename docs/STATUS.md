# Antiphon Status

## 2026-07-13 — v1 pipeline COMPLETE

All success criteria met. See **`docs/results.md`** for the full evaluation
(FDTD validation, FxLMS performance, model metrics, closed-loop results,
compute ledger). Highlights:

- FDTD validated to 0.13 dB against exact solutions
- FxLMS: 40+ dB tone reduction on measured paths
- Model v2 (9.4M params, 12k scenes): test MSE 0.768 vs 1.85/2.74 baselines
- **Closed loop: model-predicted secondary paths achieve 90% (capped) /
  109% (raw) of measured-path cancellation across 10 held-out scenes**
- Total cloud cost: ~$1.50 on a Hetzner CPX62

Artifacts: checkpoints and eval JSONs in `data/runs/` (local only,
gitignored; regenerate with the commands in README). Training data is
reproducible from seeds (scene i = seed i; held-out eval seeds 100000+).

## Suggested next steps

1. Broadband (not tonal) closed-loop evaluation
2. Frequency-weighted / phase-targeted loss for the residual 250 Hz cases
3. Data scaling beyond 12k scenes (curve had not flattened)
4. Moving-source scenarios (Task 1.4 in the handoff)
5. Web demo (handoff Task 4.1) using the analytical solver + trained model
6. GPU/numba FDTD kernel if 10x real-time interactive simulation is needed

## 2026-07-12 — session 1 (historical)

Built the package structure, FDTD solver, materials/sources, FxLMS,
data generator, model architecture and training code. Fixed the reference
prototype's coordinate-system bug. 43 tests passing at end of session.
