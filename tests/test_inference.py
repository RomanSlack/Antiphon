"""Tests for the sparse-H -> FIR reconstruction chain (model-independent)."""

import numpy as np

from antiphon.model.dataset import FREQS
from antiphon.model.inference import h_to_fir
from antiphon.model.train import delay_compensate
from antiphon.simulation.fdtd import FDTDSolver


def test_delay_compensate_roundtrip():
    rng = np.random.default_rng(0)
    H = rng.standard_normal((3, 5, len(FREQS))) + \
        1j * rng.standard_normal((3, 5, len(FREQS)))
    dist = rng.uniform(1, 20, (3, 5))
    H2 = delay_compensate(
        delay_compensate(H, FREQS, dist), FREQS, dist, inverse=True)
    np.testing.assert_allclose(H2, H, rtol=1e-12)


def test_h_to_fir_matches_measured_ir():
    """Reconstructing a FIR from 64 sparse H samples must reproduce the
    FDTD-measured band-limited impulse response."""
    dx = 0.08
    n = int(16.0 / dx)
    mask = np.zeros((n, n))
    wall = int(2.0 / dx)
    mask[:, :wall] = 1.0  # one reflecting wall for structure

    solver = FDTDSolver(mask, dx, pml_cells=20)
    fs = 1.0 / solver.dt
    src = (n // 2, int(9.0 / dx))
    rcv_ix = (int(11.0 / dx), int(6.0 / dx))
    dist = np.hypot((rcv_ix[0] - src[0]) * dx, (rcv_ix[1] - src[1]) * dx)

    ir_len = 1024
    ir_meas, _ = solver.impulse_response(
        src, [rcv_ix], f_lo=FREQS[0], f_hi=FREQS[-1], ir_len=ir_len)
    ir_meas = ir_meas[0]

    H_sparse = solver.transfer_function(
        src, [rcv_ix], FREQS, f_max=1.2 * FREQS[-1], duration=0.5)[0]
    ir_rec = h_to_fir(H_sparse, FREQS, dist, fs, ir_len)

    # Compare via normalized correlation (band-edge windows differ slightly)
    num = np.dot(ir_meas, ir_rec)
    den = np.linalg.norm(ir_meas) * np.linalg.norm(ir_rec)
    corr = num / den
    assert corr > 0.95, f'IR correlation {corr:.3f}'

    # And the tone-frequency response must agree in phase where it matters:
    # compare narrowband response at 150 and 250 Hz
    for f0 in (150.0, 250.0):
        w = np.exp(-2j * np.pi * f0 * np.arange(ir_len) / fs)
        g_meas = np.dot(ir_meas, w)
        g_rec = np.dot(ir_rec, w)
        dphi = np.angle(g_rec / g_meas)
        assert abs(dphi) < 0.3, f'{f0} Hz phase error {dphi:.2f} rad'
        dmag = abs(20 * np.log10(abs(g_rec) / abs(g_meas)))
        assert dmag < 1.0, f'{f0} Hz magnitude error {dmag:.2f} dB'
