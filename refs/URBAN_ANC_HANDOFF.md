# Urban ANC & Acoustics Foundation Model — Claude Code Handoff

## Project Overview

We're building a staged urban active noise cancellation system that serves as the proving ground for a broader **acoustics foundation model** — an ML model that predicts sound propagation in arbitrary 3D geometries in real time. The ANC corridor installation is the wedge product; the foundation model API is the platform play.

**Entity:** Third Axis AI Consulting (under 316 Group)
**Status:** Concept validated via simulation, ready for engineering buildout

---

## What Exists Already

### Generated Assets (attached)

| File | Description |
|------|-------------|
| `urban_anc_proposal.md` | Full business + technical proposal covering physics constraints, system architecture, foundation model design, deployment roadmap, revenue model, competitive landscape |
| `urban_anc_simulation.py` | 2D acoustic field simulator — analytical Green's functions with building reflections, classical ANC (FxLMS-style phase inversion), ML-optimized speaker weights via L-BFGS-B gradient descent. CLI flags: `--freq`, `--speakers`, `--mode`, `--sweep`, `--animate` |
| `anc_simulation_results.png` | 3-panel comparison: No ANC vs Classical vs ML-Optimized at 200 Hz, 6 speakers/side |
| `anc_frequency_sweep.png` | Performance curves across 50-2000 Hz showing quiet zone fraction and dB reduction |

### Key Physics Constraints (baked into simulation, must inform all engineering)

- Quiet zone radius ≈ λ/10 where λ = 343/f. At 200 Hz → 17cm zone. At 1 kHz → 3.4cm zone.
- Low-frequency suppression (sub-300 Hz) is tractable with sparse arrays. Broadband is not.
- Multipath from building reflections is the dominant challenge — simulation includes first-order image sources but real deployment needs higher-order modeling.
- Latency budget: <2ms mic-to-speaker for frequencies up to 500 Hz.
- Moving pedestrians cross one wavelength every λ/v seconds (at 500 Hz: 0.49s).

---

## Engineering Tasks

### Phase 1: Simulation Engine Upgrade

The current simulation uses analytical Green's functions (fast, good for steady-state visualization). For proper engineering validation, upgrade to FDTD.

**Task 1.1 — FDTD Wave Solver**
- Implement 2D Finite-Difference Time-Domain solver on the existing grid
- Staggered grid (Yee scheme) for pressure and velocity fields
- PML (Perfectly Matched Layer) absorbing boundary conditions at open edges
- Rigid boundary conditions at building walls (zero normal velocity)
- Courant condition: dt ≤ dx / (c * sqrt(2)) for 2D stability
- Target: real-time factor of at least 10x (simulate 1s of audio in <100ms on CPU)

**Task 1.2 — Material Properties**
- Add frequency-dependent absorption coefficients for building facades
  - Concrete: α ≈ 0.02 (125 Hz) to 0.05 (4 kHz)
  - Glass: α ≈ 0.18 (125 Hz) to 0.02 (4 kHz)
  - Brick: α ≈ 0.03 (125 Hz) to 0.07 (4 kHz)
- Implement impedance boundary conditions (not just rigid)
- Ground reflection with frequency-dependent ground impedance

**Task 1.3 — Broadband Source Modeling**
- Replace single-frequency sinusoid with realistic noise spectra
- Traffic noise: pink noise shaped by ISO 11819 road noise spectrum
- HVAC: tonal at fan blade-pass frequency + broadband
- Construction: impulsive + broadband
- Use octave-band analysis for metrics (not just single-freq RMS)

**Task 1.4 — Multi-Source Scenarios**
- Support N noise sources at arbitrary positions
- Moving source trajectory (vehicle passing at constant speed)
- Coherent vs incoherent source superposition

### Phase 2: ANC Controller

**Task 2.1 — Multi-Channel FxLMS**
- Implement filtered-reference LMS (FxLMS) algorithm
- J speakers, K error microphones, M reference microphones
- Secondary path estimation via online system identification
- Convergence rate vs. stability tradeoff (step size selection)
- Reference: S. Elliott & P. Nelson, "Active Noise Control," IEEE Signal Processing Magazine, 1993

**Task 2.2 — Feedforward Reference Microphone Placement**
- Upstream reference mics 20-50m ahead of cancellation zone
- Compute coherence between reference and error signals
- Optimize placement for maximum causal coherence at target frequencies

**Task 2.3 — Real-Time Audio Pipeline**
- Design the signal flow: mic input → ADC → DSP → DAC → speaker output
- Target latency budget breakdown:
  - ADC: ~0.5ms (at 48 kHz, 24-sample buffer)
  - DSP (FxLMS update): ~0.2ms
  - DAC: ~0.5ms
  - Acoustic propagation (speaker to error mic): variable, geometry-dependent
- Total electronic latency must be < acoustic propagation delay from noise source to cancellation point
- Simulate this pipeline in Python with realistic latency modeling

### Phase 3: Foundation Model Prototype

**Task 3.1 — Synthetic Training Data Generator**
- Wrap the FDTD solver in a data generation pipeline
- Randomize: street width (8-30m), building heights (3-50m), facade materials, source positions, frequencies
- Output: (geometry_voxels, source_pos, receiver_pos, weather_state) → complex transfer_function H(f)
- Generate transfer functions at 64 log-spaced frequencies from 20 Hz to 4 kHz
- Target: 100K scenes for v1, 10M for production
- Store as HDF5 or Zarr for efficient I/O
- Parallelize across CPU cores (embarrassingly parallel)

**Task 3.2 — Model Architecture (PyTorch)**
- Input encoder:
  - 2D occupancy grid → CNN or ViT backbone (start with ResNet-18 for speed)
  - Source/receiver positions → learned positional embeddings
  - Weather state (temp, humidity, wind) → MLP encoder
- Cross-attention between geometry features and source/receiver embeddings
- Output head: predict complex H(f) at each frequency bin
  - Separate magnitude and phase heads (phase wrapping is tricky — use sin/cos representation)
- Loss: complex MSE with A-weighting (emphasize 500-4000 Hz perceptual band)
- Start small: 2D geometry, 10M params, prove the concept before scaling

**Task 3.3 — Training Pipeline**
- PyTorch Lightning or plain PyTorch training loop
- Mixed precision (fp16) for speed
- Cosine annealing LR schedule
- Validation on held-out geometries (not just held-out source positions within same geometry)
- Metrics: transfer function MSE, phase error, and downstream ANC performance (dB reduction when using predicted H vs. measured H)

**Task 3.4 — Inference Integration**
- Export trained model to ONNX for edge deployment
- Benchmark inference latency on CPU (target <50ms per scene)
- Wire model output into the FxLMS controller as secondary path estimate
- A/B test: model-predicted secondary path vs. online system identification

### Phase 4: Visualization & Demo

**Task 4.1 — Interactive Web Demo (React)**
- 3D street scene (Three.js or R3F) with draggable noise sources and speakers
- Real-time 2D pressure field overlay on the ground plane
- Controls: frequency, speaker count, ANC mode, weather conditions
- Metrics dashboard: dB reduction, quiet zone fraction, latency
- Use the existing simulation engine compiled to WASM, or call a Python backend via WebSocket

**Task 4.2 — Investor Deck Generator**
- Script to auto-generate updated simulation figures for pitch materials
- Before/after comparisons at key frequencies
- Cost-per-dB-reduction analysis
- Competitive positioning chart

---

## Architecture Decisions (locked in)

These decisions are made. Don't revisit unless a hard blocker is found.

1. **2D first, 3D later.** The street canyon cross-section is quasi-2D for the dominant propagation modes. Full 3D adds 100x compute cost for marginal accuracy improvement at this stage.

2. **Analytical baseline + FDTD validation.** Keep the fast analytical solver for interactive demos and parameter sweeps. Use FDTD as ground truth for validation and training data.

3. **Foundation model replaces secondary path estimation, not the entire controller.** The FxLMS real-time loop stays classical. The model provides the transfer function matrix that FxLMS needs — this is where the computational bottleneck is in multi-channel ANC.

4. **Low-frequency first (sub-500 Hz).** This is where ANC works in open air. Don't try to solve broadband cancellation — it's a physics impossibility at scale. The product pitch is "we eliminate the low-frequency rumble that makes outdoor spaces unpleasant."

5. **Python for simulation/training, edge deployment via ONNX/C++.** Don't prematurely optimize the simulation — get the physics right first.

---

## Directory Structure (target)

```
urban-anc/
├── README.md
├── pyproject.toml
├── docs/
│   ├── proposal.md              # Business + technical proposal
│   ├── physics_constraints.md   # Detailed physics reference
│   └── figures/                 # Generated plots
├── src/
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── geometry.py          # UrbanGeometry class
│   │   ├── sources.py           # NoiseSource, SpeakerArray
│   │   ├── fdtd.py              # FDTD wave solver
│   │   ├── analytical.py        # Green's function solver (fast)
│   │   ├── materials.py         # Absorption coefficients
│   │   └── metrics.py           # SPL, quiet zone, A-weighting
│   ├── anc/
│   │   ├── __init__.py
│   │   ├── fxlms.py             # Multi-channel FxLMS controller
│   │   ├── secondary_path.py    # Online system ID
│   │   └── pipeline.py          # Real-time signal flow simulation
│   ├── model/
│   │   ├── __init__.py
│   │   ├── dataset.py           # Synthetic data generator + loader
│   │   ├── architecture.py      # Foundation model (PyTorch)
│   │   ├── train.py             # Training loop
│   │   └── export.py            # ONNX export + benchmarking
│   └── viz/
│       ├── __init__.py
│       ├── field_plot.py         # Pressure field visualization
│       ├── metrics_plot.py       # Performance charts
│       └── web_demo/             # React interactive demo
├── scripts/
│   ├── generate_training_data.py
│   ├── run_simulation.py
│   ├── frequency_sweep.py
│   └── benchmark_inference.py
├── tests/
│   ├── test_fdtd.py             # Validate against analytical solutions
│   ├── test_fxlms.py            # Controller convergence tests
│   └── test_model.py            # Model forward pass, loss computation
└── data/
    ├── raw/                     # Real-world recordings (future)
    └── synthetic/               # Generated training data
```

---

## Dependencies

```
# Core simulation
numpy>=1.24
scipy>=1.10
matplotlib>=3.7

# ML / Foundation model
torch>=2.0
pytorch-lightning>=2.0
h5py>=3.8
onnx>=1.14
onnxruntime>=1.15

# Audio processing
soundfile>=0.12
librosa>=0.10

# Visualization / Web
plotly>=5.15  # for interactive plots

# Dev
pytest>=7.3
black>=23.0
ruff>=0.1
```

---

## Priority Order

1. **Restructure existing code** into the directory structure above. The monolithic `urban_anc_simulation.py` splits into `simulation/`, keeping all functionality working.
2. **FDTD solver** (Task 1.1) — this is the foundation everything else builds on.
3. **Material properties + broadband sources** (Tasks 1.2-1.3) — makes simulations realistic.
4. **FxLMS controller** (Task 2.1) — proves the ANC pipeline works end-to-end in simulation.
5. **Training data generator** (Task 3.1) — most time-consuming to run, start early.
6. **Foundation model v1** (Tasks 3.2-3.3) — the core IP.
7. **Inference integration** (Task 3.4) — close the loop: model → controller → measured improvement.
8. **Web demo** (Task 4.1) — for investor conversations.

---

## Notes for Claude Code

- The existing `urban_anc_simulation.py` is functional and tested. Use it as reference implementation — don't break what works, just refactor into modules.
- When implementing FDTD, validate against the analytical solver on simple geometries (open field, single wall reflection) before adding complexity.
- The optimization in the existing code uses L-BFGS-B on speaker weights. This is a good sanity check for the foundation model — if the model can't match L-BFGS-B performance on seen geometries, something is wrong.
- Keep simulation outputs deterministic (seed RNG) for reproducible testing.
- All audio-domain code should use 48 kHz sample rate (professional audio standard).
- Pressure field visualizations should use the RdYlBu_r colormap (red = high pressure, blue = low) with consistent vmin/vmax across comparison plots.
