# Antiphon: The Science, What We Proved, and What We Have

*Written 2026-07-13, after the v1 pipeline run. Companion to
`results.md` (the numbers) and `proposal.md` (the business case).*

## The physics of quiet

Active noise cancellation works by emitting "anti-noise": a sound wave with
the same shape as the noise but inverted, so the two cancel. Headphones do
this easily because the speaker sits 1 cm from your eardrum. Open air is
brutally harder, for three reasons:

1. **Quiet zones scale with wavelength.** Cancellation only holds within
   roughly a tenth of a wavelength of the point you optimized. At 1 kHz
   that is a 3 cm bubble, useless outdoors. At 100 Hz it is a head-sized
   34 cm, and at 50 Hz over half a meter. This is why Antiphon targets
   low-frequency noise only (traffic rumble, HVAC drone, sub-500 Hz):
   it is the band where open-air ANC is physically possible, and also the
   band that walls, windows, and earplugs block worst.

2. **Cities are hall-of-mirrors acoustics.** Sound between buildings
   reflects off every facade. The anti-noise arrives at a listener along
   dozens of paths with different delays. To cancel correctly, the system
   must know the full *transfer function*: how sound transforms (per
   frequency, in amplitude and phase) between every speaker and every
   listening point, reflections included.

3. **Phase is everything.** Anti-noise with the right loudness but the
   wrong timing makes things *louder*. Get the phase wrong by half a cycle
   and cancellation becomes amplification. Any useful prediction of urban
   acoustics must get phase right, which is much harder than predicting
   loudness.

## The classical solution and its bottleneck

The standard controller, FxLMS (filtered-x least mean squares), has been
known since the 1980s: reference microphones hear the noise coming,
adaptive filters compute speaker signals, error microphones report residual
noise, and the filters continuously adapt. It works, and it is not the hard
part.

The hard part is that FxLMS needs the **secondary paths**: the transfer
functions from each speaker to each error microphone. Classically you
measure them by playing test signals through every speaker and recording
every microphone, and you re-measure whenever conditions change (weather,
parked trucks, crowds). For a city-scale deployment this measurement
campaign is the dominant cost and fragility.

**Antiphon's bet:** a neural network can *predict* those transfer functions
directly from the street's geometry, so the measurement campaign becomes a
model inference. The ANC corridor is the proving ground; the general-purpose
acoustics prediction model is the product.

## What we built

A complete, tested, end-to-end pipeline, all 2D, sub-500 Hz, pure CPU:

- **A physics engine (FDTD).** Simulates sound waves propagating through
  street geometry from first principles (the wave equation on a grid),
  with absorbing open boundaries and materials (concrete, glass, brick).
  Validated against exact mathematical solutions to within 0.13 dB.
- **A controller (multichannel FxLMS)** with online secondary-path
  identification, achieving 40+ dB tone cancellation in simulated street
  canyons when given correct paths.
- **A data factory.** 12,000 randomized street canyons (width, facade
  absorption, source position), each solved by the physics engine, giving
  288,000 samples of (geometry, source, receiver) -> complex transfer
  function at 64 frequencies. Fully deterministic from seeds. ~35 minutes
  on a $0.25/hour cloud box.
- **The model (9.4M parameters).** A CNN encodes the street's occupancy
  grid; cross-attention answers queries of the form "source here, receiver
  there, absorptions such-and-such"; output is the complex transfer
  function. One key trick: the model predicts *delay-compensated* transfer
  functions (bulk travel-time phase removed, then restored at inference),
  which turns a wildly oscillating target into a smooth, learnable one.

## What we proved

1. **The model learns real acoustics, not statistics.** On street
   geometries it never saw, it predicts transfer functions 2.4x better
   than the best naive baseline, and 3.6x better than a distance-aware
   free-field physics formula. Notably, that physics formula is *worse*
   than just guessing the average: reflections dominate urban sound, and
   capturing them is exactly what the model adds.

2. **Predictions are good enough to control with.** The decisive test: run
   the same controller against the true physics twice, once with measured
   secondary paths (the expensive classical way) and once with the model's
   predicted paths. Across 10 unseen streets and two tones, predicted
   paths delivered **90% of the measured-path cancellation** (capping both
   at a practical 20 dB ceiling), reached full >= 20 dB cancellation in 15
   of 20 cases, and outright beat measured paths in several. Phase, the
   thing that had to be right, was right.

3. **Failure is predictable and buyable-down with data.** Wherever the
   model's phase error exceeded about 1 radian, the controller stalled;
   under 0.5 radian it performed like measured paths. Tripling the
   training data (4k -> 12k scenes) cut model error 31% and fixed both
   hard failure cases, and the scaling curve had not flattened. The
   residual weakness sits at higher frequency (250 Hz), consistent with
   physics: shorter wavelengths punish the same absolute error more.

In short: **the measurement campaign that makes urban ANC expensive can be
replaced by a learned model, at least in 2D simulation, and we can state
precisely where that replacement holds and what improves it.**

## What we now have

- `src/antiphon/`: validated FDTD solver, analytical solver, materials,
  broadband sources, FxLMS controller, model, training and inference code.
  55+ tests including physics validation against exact solutions.
- 12,000-scene dataset recipe (reproducible from seeds) and trained
  checkpoints (v1, v1-long, v2) with full metrics.
- A closed-loop evaluation harness with honest metrics (step-size ladders,
  convergence-fair timing, capped and raw ratios, phase diagnostics).
- Figures: FDTD validation, closed-loop comparison, ANC field maps.
- Total cloud cost of the entire proof: about $1.50.

## Honest limits (read before quoting numbers)

- Everything is **2D** (a horizontal slice of a street canyon) and
  **simulation-only**; no microphone has heard any of this yet.
- Closed loop tested on **tones**, not broadband traffic noise.
- Evaluation geometries come from the same *family* as training
  (parametric canyons), so this proves generalization across geometries,
  not across geometry families.
- The 20 dB cap in the headline metric is a judgment call (raw ratios are
  reported alongside and are higher).
- Real deployments add weather, moving sources, and hardware latency,
  all modeled in the roadmap, none in this result.
