# Urban Active Noise Cancellation & the Acoustics Foundation Model

**Proposal — Third Axis AI Consulting / 316 Group**
**Date: July 2026**

---

## Executive Summary

We propose a staged approach to deploying active noise cancellation (ANC) in urban pedestrian corridors, beginning with fixed-point low-frequency suppression and culminating in an ML-driven "acoustics foundation model" that predicts and optimizes sound fields across complex urban geometries in real time. The foundation model is the real product — the ANC deployment is the proving ground.

---

## The Physics: What Works and What Doesn't

### Why Headphones Work

Noise-cancelling headphones exploit a near-field advantage: the anti-noise speaker is ~1cm from the eardrum, creating a cancellation zone smaller than a marble. The system only needs to solve a 1D problem (one ear canal, one listener, one microphone). Latency budget is generous because the path length is millimeters.

### Why "AirPods for a City Block" Is Hard

Scaling ANC to open air introduces three compounding constraints:

**1. Wavelength vs. zone size.** The effective quiet zone radius scales as approximately λ/10, where λ = c/f (speed of sound / frequency). At 100 Hz, λ = 3.43m, so you can create a ~34cm quiet zone — roughly head-sized. At 1 kHz, λ = 34cm, and the quiet zone shrinks to ~3.4cm. Broadband cancellation over walking-scale areas (2m+) is physically intractable without speaker densities exceeding one per square meter.

**2. Moving targets.** Pedestrians walk at ~1.4 m/s. The ANC system must continuously re-solve the sound field for every moving body. At 500 Hz, a pedestrian crosses one full wavelength every 0.49 seconds — the controller must update faster than this or the cancellation inverts and becomes amplification.

**3. Multipath propagation.** Urban canyons create reflections off buildings, diffraction around corners, ground reflections, and wind-induced refraction. The direct-path anti-noise signal arrives at the target point, but so do dozens of reflected copies at varying delays. Classical ANC (FxLMS, FxNLMS) struggles with secondary path estimation in such environments.

### What Actually Works Today

Despite these constraints, several regimes are tractable:

- **Low-frequency hum suppression (sub-300 Hz):** Traffic rumble, HVAC drone, transformer hum. Long wavelengths mean quiet zones of 1-3m radius are achievable with sparse speaker arrays. This covers the dominant annoyance band in most urban noise profiles.
- **Fixed-point quiet zones:** Park benches, outdoor dining, transit stops. The listener position is known and approximately stationary, reducing the problem to classical multi-channel ANC.
- **Corridor-shaped cancellation:** Using linear speaker arrays along building facades to create a "quiet curtain" along a sidewalk. The geometry constrains the problem to quasi-2D, making it tractable for real-time optimization.

---

## System Architecture

### Hardware Layer

**Microphone array (sensing):** Distributed MEMS microphone arrays along building facades, spaced at λ_min/2 for the target frequency band. For 50-500 Hz suppression: spacing of ~34cm, yielding ~30 microphones per 10m facade segment.

**Speaker array (actuation):** Compact full-range drivers (4-6") mounted at building facade height (~2.5m) and ground level, creating a vertical aperture for 3D sound field control. Weatherized IP67 enclosures. Spacing follows the same λ_min/2 criterion.

**Edge compute:** Each 50m street segment gets a local compute node (NVIDIA Jetson-class or FPGA) running the real-time controller. Latency budget: <2ms from microphone capture to speaker output for frequencies up to 500 Hz.

**Reference sensors:** Upstream microphones placed 20-50m ahead of the cancellation zone, providing advance warning of approaching noise sources (vehicles, construction). This buys processing time proportional to the source velocity.

### Software Layer

**Classical ANC controller (baseline):** Multi-channel filtered-x LMS (FxLMS) running on the edge compute, handling the real-time loop. This is proven technology — the innovation is in the secondary path modeling.

**Acoustics foundation model (the unlock):** A transformer-based model trained on urban acoustic propagation data that replaces the traditional secondary path estimation. Instead of measuring impulse responses between every speaker-microphone pair (which change with weather, pedestrian density, and parked vehicles), the model predicts the full transfer function matrix from:

- 3D geometry (LiDAR scans of the street, updated periodically)
- Weather state (temperature, humidity, wind speed/direction — all affect sound speed and refraction)
- Pedestrian density (from cameras or radar, as acoustic absorbers/scatterers)
- Time of day (traffic patterns, HVAC schedules)

The model outputs optimal speaker array weights at ~100 Hz update rate, which the FxLMS controller tracks in real time.

---

## The Acoustics Foundation Model

### Why This Is the Real Product

The ANC deployment is one application. An acoustics foundation model that can predict sound propagation in arbitrary 3D geometries has applications across:

- **Architectural acoustics:** Predict room acoustics from floor plans before construction
- **Urban planning:** Model noise impact of new developments, highway routing, transit stations
- **Concert/event sound design:** Optimize speaker placement for coverage and minimal bleed
- **Industrial noise control:** Design enclosures and barriers with ML-optimized geometries
- **Defense/security:** Acoustic surveillance, gunshot localization, vehicle classification
- **Automotive:** In-cabin ANC with model-predictive control replacing fixed FIR filters
- **Real estate valuation:** Quantify noise exposure for any address, any time of day

### Training Data Pipeline

**Synthetic data (primary):** Physics-based acoustic simulation using the Boundary Element Method (BEM) and Finite-Difference Time-Domain (FDTD) solvers. Generate millions of source-geometry-receiver configurations:

- Randomized urban canyon geometries (building heights, street widths, facade materials)
- Varied source types (broadband traffic, tonal HVAC, impulsive construction)
- Weather perturbations (wind gradients, temperature inversions)
- Frequency range: 20 Hz - 4 kHz

**Real-world validation:** Deploy instrumented test corridors in 3-5 cities with diverse urban morphologies. Continuous recording from distributed microphone arrays provides ground truth for model fine-tuning.

**Data augmentation:** Acoustic reciprocity (swap source and receiver, same transfer function) doubles the effective dataset. Time-reversal symmetry provides additional augmentation.

### Model Architecture

**Input representation:** Voxelized 3D geometry (occupancy grid at 10cm resolution for the target frequency range) + source positions + receiver positions + environmental state vector.

**Backbone:** 3D vision transformer operating on the voxel grid, with cross-attention to source/receiver position embeddings. The model predicts complex-valued transfer functions (magnitude + phase) at discrete frequencies, which are interpolated to arbitrary frequencies via learned basis functions.

**Output:** Transfer function matrix H(f) between all source-receiver pairs in the scene. For ANC: this directly gives the optimal anti-noise filter coefficients via Wiener-Hopf solution.

**Training objective:** Complex-valued MSE on transfer functions, with a perceptual weighting that emphasizes the 100-2000 Hz band (human annoyance-weighted). Auxiliary loss on spatial coherence to regularize phase predictions.

### Estimated Scale

- **Parameters:** 500M-2B (comparable to mid-size vision models)
- **Training data:** ~100M synthetic scenes, ~10K hours real-world recordings
- **Training compute:** ~1000 A100-hours for the base model
- **Inference:** <50ms per scene on edge hardware (after quantization + distillation)

---

## Deployment Roadmap

### Phase 1 — Fixed-Point Proof of Concept (Months 1-6)

Deploy a single 20m corridor with 12 speakers and 16 microphones. Target: 10-15 dB reduction in the 50-300 Hz band at 4 fixed listening points (benches). Use classical FxLMS only. Instrument heavily for training data collection.

**Deliverables:** Measured dB reduction, latency metrics, weather sensitivity data, 500+ hours of multi-channel recordings.

**Cost estimate:** $40-60K hardware, $20K installation, $30K compute/engineering.

### Phase 2 — ML-Enhanced Controller (Months 6-12)

Train v1 of the acoustics model on Phase 1 data + synthetic augmentation. Replace secondary path estimation with model predictions. Extend cancellation from fixed points to a 2m-wide walking corridor.

**Deliverables:** Model v1, A/B comparison vs. classical ANC, real-time inference demo on edge hardware.

### Phase 3 — Multi-Site Scaling (Months 12-24)

Deploy to 3-5 diverse urban sites (narrow alley, wide boulevard, transit stop, park edge). Collect diverse training data. Train foundation model v2 with cross-site generalization.

**Deliverables:** Foundation model v2, generalization benchmarks, API for third-party acoustic prediction queries.

### Phase 4 — Platform Launch (Months 24-36)

Productize the acoustics foundation model as an API service. Target customers: architectural firms, urban planners, AV installers, automotive OEMs. The ANC deployments become reference installations and ongoing training data sources.

---

## Revenue Model

### Near-Term (Phase 1-2): Installation Services

Sell ANC corridor installations to municipalities, commercial real estate developers, and hospitality venues. Price per 20m corridor: $80-150K installed, with $12-20K/year maintenance SaaS.

Target customers: outdoor dining districts (noise complaints kill permits), hospital campuses (ambulance noise in patient areas), luxury residential developments (traffic noise reduction as amenity).

### Long-Term (Phase 3-4): Foundation Model API

Usage-based pricing for acoustic prediction queries:
- **Architectural:** $0.50-2.00 per room simulation
- **Urban planning:** $5-20 per site analysis
- **Real-time ANC optimization:** $500-2000/month per installation
- **Automotive OEM licensing:** $1-5 per vehicle

TAM for acoustic simulation alone is ~$2B (architectural acoustics software + consulting). The foundation model approach collapses what currently requires expert consultants + expensive measurement campaigns into an API call.

---

## Competitive Landscape

**Silentium (Israel):** Automotive and HVAC ANC. No urban/open-air capability. Acquired by Harman.

**Bose:** Consumer headphones and automotive. Research-stage open-air work, no commercial deployment.

**University research (NTU Singapore, ISVR Southampton):** Open-window ANC prototypes achieving 10 dB reduction. Lab-scale, no productization path.

**No one is building the foundation model.** Existing acoustic simulation tools (COMSOL, Odeon, CATT-Acoustic) are physics solvers, not learned models. They're accurate but slow (minutes to hours per scene) and require expert setup. A learned model that runs in milliseconds and generalizes across geometries is a different product category entirely.

---

## Why Us / Why Now

Three converging trends make this tractable in 2026 when it wasn't in 2020:

1. **Compute cost collapse.** Edge inference hardware (Jetson Orin, Hailo-8) can run 500M-parameter models at <50ms latency for <$500/unit. This was $5000+ in 2020.

2. **Foundation model architectures.** Vision transformers and 3D scene understanding models (NeRF, 3D Gaussian Splatting) have proven that complex physical scenes can be learned from data. Acoustic propagation is simpler than light transport (scalar field vs. vector field, lower frequencies, fewer bounces).

3. **Urban noise regulation.** WHO guidelines (2018) and EU Environmental Noise Directive are driving municipalities to invest in noise mitigation. The market is being created by regulation.

---

## Technical Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Phase prediction accuracy insufficient for cancellation | High | Fall back to amplitude-only reduction (still useful); hybrid approach using model for amplitude + classical for phase |
| Latency exceeds budget on edge hardware | Medium | Model distillation; FPGA acceleration for inference; reduce target frequency range |
| Weather sensitivity undermines generalization | Medium | Include weather as explicit model input; seasonal fine-tuning; conservative operating envelope |
| Speaker/mic hardware degradation in outdoor environment | Medium | IP67+ enclosures; redundant elements; self-calibration routines |
| Regulatory barriers to outdoor speaker installation | Low-Medium | Partner with municipalities; frame as noise mitigation (net positive); comply with local noise ordinances (anti-noise is still sound) |

---

## Appendix: Simulation Code

See accompanying `urban_anc_simulation.py` for a 2D FDTD acoustic simulation demonstrating quiet zone formation with speaker arrays, including ML-optimized weight computation.
