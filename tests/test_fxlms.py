"""FxLMS controller tests: synthetic plants first, then FDTD-measured paths.

Success criterion: 10+ dB reduction at the error mics for sub-300 Hz tones.
"""

import numpy as np
import pytest

from antiphon.anc import (
    MultichannelFxLMS,
    OnlineSecondaryPathLMS,
    db_reduction,
    simulate_anc,
)
from antiphon.simulation.fdtd import FDTDSolver

FS = 4000.0


def synthetic_plant(K=2, J=2, Ls=32, seed=1):
    """Random stable FIR paths with a propagation delay."""
    rng = np.random.default_rng(seed)
    S = np.zeros((K, J, Ls))
    for k in range(K):
        for j in range(J):
            delay = 3 + rng.integers(0, 5)
            taps = rng.standard_normal(Ls - delay) * \
                np.exp(-np.arange(Ls - delay) / 6.0)
            S[k, j, delay:] = taps
    return S


def tone(freq, duration, fs=FS):
    t = np.arange(int(duration * fs)) / fs
    return np.sin(2 * np.pi * freq * t)


def primary_through(P, x):
    """Filter source x through primary paths P (K, Lp)."""
    return np.stack([np.convolve(x, P[k])[:len(x)] for k in range(P.shape[0])])


def test_fxlms_tone_cancellation_known_secondary():
    """With a perfect secondary-path model, a tone should cancel deeply."""
    K = J = 2
    S = synthetic_plant(K, J, seed=1)
    P = synthetic_plant(K, 1, seed=2)[:, 0, :]

    x = tone(150.0, 4.0)
    d = primary_through(P, x)

    ctl = MultichannelFxLMS(1, J, K, filter_len=64,
                            secondary_estimate=S, mu=0.02)
    e = simulate_anc(x, d, S, ctl)
    red = db_reduction(d, e)
    assert np.all(red > 20), f'reductions: {red}'


def test_fxlms_needs_reasonable_secondary_estimate():
    """A sign-flipped secondary estimate must not converge (sanity check
    that the filtered-x structure actually matters)."""
    K = J = 2
    S = synthetic_plant(K, J, seed=1)
    P = synthetic_plant(K, 1, seed=2)[:, 0, :]
    x = tone(150.0, 2.0)
    d = primary_through(P, x)

    ctl = MultichannelFxLMS(1, J, K, filter_len=64,
                            secondary_estimate=-S, mu=0.1, leak=1e-4)
    e = simulate_anc(x, d, S, ctl)
    red = db_reduction(d, e)
    # Diverged (NaN) or no meaningful reduction both count as not converging
    assert not np.all(red > 10)


def test_online_secondary_path_identification():
    """Auxiliary-noise LMS should identify the true secondary paths."""
    K = J = 2
    S = synthetic_plant(K, J, Ls=32, seed=3)

    ctl = MultichannelFxLMS(1, J, K, filter_len=8, shat_len=32, mu=0.0)
    ident = OnlineSecondaryPathLMS(ctl, level=1.0, mu=0.5, seed=0)

    # No reference, no primary noise: pure identification phase
    N = 20000
    x = np.zeros((1, N))
    d = np.zeros((K, N))
    simulate_anc(x, d, S, ctl, online_id=ident, adapt=False)

    misalignment = np.linalg.norm(ctl.shat - S) / np.linalg.norm(S)
    assert misalignment < 0.1, f'misalignment {misalignment:.3f}'


def test_fxlms_with_identified_secondary_achieves_10db():
    """End-to-end: identify Shat online, then cancel a 150 Hz tone."""
    K = J = 2
    S = synthetic_plant(K, J, Ls=32, seed=4)
    P = synthetic_plant(K, 1, Ls=32, seed=5)[:, 0, :]

    ctl = MultichannelFxLMS(1, J, K, filter_len=64, shat_len=32, mu=0.05)
    ident = OnlineSecondaryPathLMS(ctl, level=1.0, mu=0.5, seed=0)

    # Phase 1: identification with aux noise only
    N_id = 20000
    simulate_anc(np.zeros((1, N_id)), np.zeros((K, N_id)), S, ctl,
                 online_id=ident, adapt=False)

    # Phase 2: cancel the tone using the identified model
    x = tone(150.0, 4.0)
    d = primary_through(P, x)
    e = simulate_anc(x, d, S, ctl)
    red = db_reduction(d, e)
    assert np.all(red > 10), f'reductions: {red}'


@pytest.mark.slow
def test_fxlms_on_fdtd_street_canyon_10db():
    """The headline criterion: 10+ dB at the error mics for a sub-300 Hz
    tone, with paths measured by FDTD in a street canyon."""
    dx = 0.1
    # 20m x 14m street canyon slice, walls top and bottom
    nx, ny = int(20.0 / dx), int(14.0 / dx)
    mask = np.zeros((nx, ny))
    wall = int(2.0 / dx)
    mask[:, :wall] = 1.0
    mask[:, -wall:] = 1.0

    solver = FDTDSolver(mask, dx, pml_cells=15)
    fs = 1.0 / solver.dt

    noise_src = (int(3.0 / dx), ny // 2)
    speakers = [(int(9.0 / dx), int(4.0 / dx)),
                (int(9.0 / dx), int(10.0 / dx))]
    error_mics = [(int(12.0 / dx), int(6.0 / dx)),
                  (int(12.0 / dx), int(8.0 / dx))]

    ir_len = 600
    P, _ = solver.impulse_response(noise_src, error_mics, ir_len=ir_len)
    S = np.stack([
        solver.impulse_response(spk, error_mics, ir_len=ir_len)[0]
        for spk in speakers
    ], axis=1)  # (K, J, Ls)

    x = tone(200.0, 3.0, fs=fs)
    d = primary_through(P, x)

    # Step size: the ~50-sample acoustic delay to the error mics tightens
    # the FxLMS stability bound; mu=0.005 converges to 40+ dB here.
    ctl = MultichannelFxLMS(1, len(speakers), len(error_mics),
                            filter_len=128, secondary_estimate=S, mu=0.005)
    e = simulate_anc(x, d, S, ctl)
    red = db_reduction(d, e)
    assert np.all(red > 10), f'reductions: {red}'
