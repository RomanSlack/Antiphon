"""FDTD validation against exact 2D analytical solutions.

Strategy: measure complex transfer functions between the source and pairs of
receivers, then compare receiver-to-receiver RATIOS against the exact 2D
line-source solution (Hankel functions). Ratios cancel the arbitrary source
injection scaling, so the comparison is parameter-free.
"""

import numpy as np
import pytest
from scipy.special import hankel1

from antiphon.simulation.fdtd import FDTDSolver
from antiphon.simulation.geometry import C_SOUND

DX = 0.05
FREQS = [100.0, 200.0, 300.0, 400.0]


def exact_free_field(k, r):
    """Exact 2D harmonic line-source field (up to a constant): H0(kr)."""
    return hankel1(0, k * r)


def db_error(h_meas, h_exact):
    """|dB difference| between measured and exact receiver ratios."""
    return abs(20 * np.log10(np.abs(h_meas)) - 20 * np.log10(np.abs(h_exact)))


@pytest.fixture(scope='module')
def free_field():
    """20m x 20m open domain, source at center, receivers along +x."""
    n = int(20.0 / DX)
    mask = np.zeros((n, n))
    solver = FDTDSolver(mask, DX, pml_cells=30)
    src = (n // 2, n // 2)
    radii = [2.0, 4.0, 6.0]
    receivers = [(src[0] + int(r / DX), src[1]) for r in radii]
    H = solver.transfer_function(src, receivers, FREQS, duration=0.25)
    return radii, H


def test_open_field_amplitude_decay_within_1db(free_field):
    radii, H = free_field
    for k_idx, f in enumerate(FREQS):
        k = 2 * np.pi * f / C_SOUND
        for i in range(1, len(radii)):
            ratio_meas = H[i, k_idx] / H[0, k_idx]
            ratio_exact = exact_free_field(k, radii[i]) / exact_free_field(k, radii[0])
            err = db_error(ratio_meas, ratio_exact)
            assert err < 1.0, (
                f'{f} Hz, r={radii[i]}m: amplitude error {err:.2f} dB')


def test_open_field_phase_matches(free_field):
    radii, H = free_field
    for k_idx, f in enumerate(FREQS):
        k = 2 * np.pi * f / C_SOUND
        for i in range(1, len(radii)):
            ratio_meas = H[i, k_idx] / H[0, k_idx]
            ratio_exact = exact_free_field(k, radii[i]) / exact_free_field(k, radii[0])
            # FDTD uses exp(-iwt) vs hankel1's exp(+iwt) convention or vice
            # versa; compare phase magnitude against both conjugations.
            dphi = np.angle(ratio_meas / ratio_exact)
            dphi_conj = np.angle(ratio_meas / np.conj(ratio_exact))
            err = min(abs(dphi), abs(dphi_conj))
            assert err < 0.2, f'{f} Hz, r={radii[i]}m: phase error {err:.2f} rad'


def test_single_wall_interference_within_1db():
    """Source near a rigid wall: field = direct + image source."""
    n = int(20.0 / DX)
    mask = np.zeros((n, n))
    wall_col = int(4.0 / DX)
    mask[:, :wall_col] = 1.0  # rigid wall filling y < 4m

    solver = FDTDSolver(mask, DX, pml_cells=30)

    # Pressure cells are at centers (i+0.5)*dx; the rigid face sits exactly
    # at wall_col*dx. Use the true discrete positions in the exact solution.
    def center(idx):
        return (idx + 0.5) * DX

    # Source ~2m from the wall face, centered in x
    src = (n // 2, int(6.0 / DX))
    wall_y = wall_col * DX

    # Receivers chosen away from direct/image interference nulls at every
    # test frequency (path difference stays >0.15 wavelengths from any
    # half-integer multiple); near a null, dB comparisons are ill-conditioned.
    rcv_idx = [(int(x / DX), int(y / DX))
               for (x, y) in [(8.0, 8.0), (12.5, 9.5), (14.0, 10.0), (13.0, 8.0)]]
    rcv_pts = [(center(i), center(j)) for (i, j) in rcv_idx]
    H = solver.transfer_function(src, rcv_idx, FREQS, duration=0.25)

    sx, sy = center(src[0]), center(src[1])
    sy_img = 2 * wall_y - sy  # image source mirrored across the wall face

    for k_idx, f in enumerate(FREQS):
        k = 2 * np.pi * f / C_SOUND

        def exact(pt):
            r_d = np.hypot(pt[0] - sx, pt[1] - sy)
            r_i = np.hypot(pt[0] - sx, pt[1] - sy_img)
            return hankel1(0, k * r_d) + hankel1(0, k * r_i)

        for i in range(1, len(rcv_pts)):
            ratio_meas = H[i, k_idx] / H[0, k_idx]
            ratio_exact = exact(rcv_pts[i]) / exact(rcv_pts[0])
            err = db_error(ratio_meas, ratio_exact)
            assert err < 1.0, (
                f'{f} Hz, receiver {rcv_pts[i]}: error {err:.2f} dB')


def test_impedance_wall_absorbs():
    """An absorbing wall must reflect less than a rigid wall."""
    n = int(16.0 / DX)
    wall_col = int(3.0 / DX)

    def reflected_energy(admittance):
        mask = np.zeros((n, n))
        mask[:, :wall_col] = 1.0
        solver = FDTDSolver(mask, DX, admittance=admittance, pml_cells=30)
        src = (n // 2, int(8.0 / DX))
        rcv = [(n // 2, int(5.0 / DX))]
        signal = solver.gaussian_pulse(f_max=400.0)
        n_steps = int(0.10 / solver.dt)
        trace = solver.run(src, signal, rcv, n_steps)[0]
        # Direct pulse arrives ~3m/c = 8.7ms; reflection ~11m/c = 32ms.
        split = int(0.020 / solver.dt)
        return float(np.sum(trace[split:] ** 2))

    e_rigid = reflected_energy(0.0)
    # Y = 1/(rho*c) is the perfectly matched (fully absorbing) admittance
    e_soft = reflected_energy(1.0 / (1.225 * 343.0))
    assert e_soft < 0.15 * e_rigid


def test_energy_decays_with_pml():
    """Total field energy must decay after the source stops (PML works)."""
    n = int(10.0 / DX)
    mask = np.zeros((n, n))
    solver = FDTDSolver(mask, DX, pml_cells=25)
    signal = solver.gaussian_pulse(f_max=400.0)
    src = (n // 2, n // 2)

    solver.reset()
    for k in range(len(signal)):
        solver.step(src, signal[k])
    e_after_source = np.sum(solver.p ** 2)

    # 10m domain: waves cross in ~30ms
    for _ in range(int(0.06 / solver.dt)):
        solver.step()
    e_late = np.sum(solver.p ** 2)
    assert e_late < 1e-4 * e_after_source


def test_deterministic():
    n = 100
    mask = np.zeros((n, n))
    results = []
    for _ in range(2):
        solver = FDTDSolver(mask, DX, pml_cells=10)
        signal = solver.gaussian_pulse(f_max=300.0)
        traces = solver.run((50, 50), signal, [(70, 50)], 400)
        results.append(traces)
    np.testing.assert_array_equal(results[0], results[1])
