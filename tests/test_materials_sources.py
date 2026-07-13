"""Tests for material properties, broadband sources, and band metrics."""

import numpy as np

from antiphon.simulation import (
    absorption_at,
    band_admittance,
    construction_noise,
    hvac_noise,
    octave_band_levels,
    traffic_noise,
)
from antiphon.simulation.fdtd import FDTDSolver
from antiphon.simulation.materials import admittance_from_alpha
from antiphon.simulation.geometry import C_SOUND, RHO_AIR


def test_absorption_interpolation():
    assert absorption_at('concrete', 125.0) == 0.02
    assert absorption_at('concrete', 4000.0) == 0.05
    assert absorption_at('glass', 125.0) == 0.18
    # Below 125 Hz clamps to the lowest band
    assert absorption_at('brick', 60.0) == 0.03
    # Interpolated values sit between band endpoints
    a = absorption_at('glass', 180.0)
    assert 0.10 < a < 0.18


def test_admittance_limits():
    assert admittance_from_alpha(0.0) == 0.0          # rigid
    y_matched = admittance_from_alpha(1.0)
    np.testing.assert_allclose(y_matched, 1.0 / (RHO_AIR * C_SOUND))
    # More absorption -> larger admittance
    assert admittance_from_alpha(0.2) > admittance_from_alpha(0.05) > 0


def test_glass_absorbs_more_than_concrete_at_125hz():
    """FDTD: a glass wall (alpha=0.18 @ 125 Hz) reflects less energy than
    concrete (alpha=0.02)."""
    dx = 0.05
    n = int(16.0 / dx)
    wall_col = int(3.0 / dx)

    def reflected_energy(material):
        mask = np.zeros((n, n))
        mask[:, :wall_col] = 1.0
        Y = band_admittance(material, 125.0)
        solver = FDTDSolver(mask, dx, admittance=Y, pml_cells=30)
        src = (n // 2, int(8.0 / dx))
        rcv = [(n // 2, int(5.0 / dx))]
        signal = solver.gaussian_pulse(f_max=200.0)
        n_steps = int(0.12 / solver.dt)
        trace = solver.run(src, signal, rcv, n_steps)[0]
        split = int(0.035 / solver.dt)  # after direct pulse passes
        return float(np.sum(trace[split:] ** 2))

    e_concrete = reflected_energy('concrete')
    e_glass = reflected_energy('glass')
    assert e_glass < e_concrete


def test_broadband_sources_deterministic():
    for gen in (traffic_noise, hvac_noise, construction_noise):
        a = gen(0.5, seed=7)
        b = gen(0.5, seed=7)
        np.testing.assert_array_equal(a, b)
        c = gen(0.5, seed=8)
        assert not np.array_equal(a, c)


def test_traffic_noise_is_low_frequency_dominated():
    sig = traffic_noise(2.0, sample_rate=48000, seed=0)
    levels = octave_band_levels(sig, 48000, centers=[63.0, 125.0, 2000.0])
    assert levels[63.0] > levels[2000.0] + 10
    assert levels[125.0] > levels[2000.0] + 10


def test_hvac_noise_has_blade_pass_tone():
    fs = 48000
    sig = hvac_noise(2.0, sample_rate=fs, blade_pass_freq=90.0, seed=0)
    spectrum = np.abs(np.fft.rfft(sig))
    f = np.fft.rfftfreq(len(sig), 1.0 / fs)
    tone_idx = np.argmin(np.abs(f - 90.0))
    tone = spectrum[tone_idx - 2:tone_idx + 3].max()
    # The tone should dominate the nearby broadband floor
    floor = np.median(spectrum[(f > 60) & (f < 120)])
    assert tone > 10 * floor


def test_octave_band_levels_localizes_sine():
    fs = 48000
    t = np.arange(fs) / fs
    sig = np.sin(2 * np.pi * 125.0 * t)
    levels = octave_band_levels(sig, fs, centers=[63.0, 125.0, 250.0])
    assert levels[125.0] > levels[63.0] + 30
    assert levels[125.0] > levels[250.0] + 30
    # A unit sine has RMS 1/sqrt(2) -> 91 dB re 20 uPa
    np.testing.assert_allclose(levels[125.0], 90.97, atol=0.1)
