"""Behavioral tests for the simulation package."""

import numpy as np
import pytest

from antiphon.simulation import (
    AcousticField,
    NoiseSource,
    SpeakerArray,
    UrbanGeometry,
    compute_metrics,
    optimize_speaker_weights,
)


@pytest.fixture(scope='module')
def small_setup():
    """Small geometry to keep optimization tests fast."""
    geo = UrbanGeometry(street_width=10.0, street_length=10.0)
    noise = NoiseSource(x=geo.street_length / 2, y=0.0, frequency=200)
    field = AcousticField(geo)
    arrays = [SpeakerArray(2, side, geo) for side in ('left', 'right')]
    return geo, noise, field, arrays


def test_to_grid():
    geo = UrbanGeometry()
    # y_min maps to row 0; the centerline maps to the middle row
    assert geo.to_grid(0.0, geo.y_min) == (0, 0)
    assert geo.to_grid(1.0, 0.0) == (10, geo.ny // 2)


def test_pressure_zero_inside_buildings():
    geo = UrbanGeometry()
    noise = NoiseSource(x=geo.street_length / 2, y=0.0, frequency=200)
    field = AcousticField(geo)
    p = field.compute_pressure(noise, [], t=1e-3, mode='off')
    assert np.all(p[geo.mask == 1] == 0)


def test_no_speakers_nonoff_mode_matches_noise_only():
    """Regression: reference script crashed with zero speaker arrays."""
    geo = UrbanGeometry()
    noise = NoiseSource(x=geo.street_length / 2, y=0.0, frequency=200)
    field = AcousticField(geo)
    p_off = field.compute_pressure(noise, [], t=1e-3, mode='off')
    p_cl = field.compute_pressure(noise, [], t=1e-3, mode='classical')
    np.testing.assert_array_equal(p_off, p_cl)


def test_classical_anc_changes_pedestrian_pressure():
    # Note: the naive phase-inversion heuristic does NOT reduce average
    # pedestrian pressure in this model (it targets the speaker positions,
    # not the pedestrian zone). Only the optimized mode reduces it, which
    # is why the sweep plot clamps negative reductions to zero.
    geo = UrbanGeometry()
    noise = NoiseSource(x=geo.street_length / 2, y=0.0, frequency=200)
    field = AcousticField(geo)
    arrays = [SpeakerArray(6, side, geo) for side in ('left', 'right')]
    for arr in arrays:
        arr.set_classical_weights(noise)

    p_off = field.compute_rms_pressure(noise, arrays, 'off', n_samples=8)
    p_cl = field.compute_rms_pressure(noise, arrays, 'classical', n_samples=8)

    m_off = compute_metrics(p_off, geo)
    m_cl = compute_metrics(p_cl, geo)
    assert m_cl['avg_pressure'] != m_off['avg_pressure']


def test_compute_spl_reference_value():
    field = AcousticField(UrbanGeometry())
    spl = field.compute_spl(np.array([2e-3]))
    np.testing.assert_allclose(spl, [40.0])


def test_optimizer_improves_objective(small_setup):
    geo, noise, field, arrays = small_setup
    for arr in arrays:
        arr.set_classical_weights(noise)

    def ped_rms(mode):
        p = field.compute_rms_pressure(noise, arrays, mode, n_samples=8)
        return compute_metrics(p, geo)['avg_pressure']

    before = ped_rms('optimal')
    result = optimize_speaker_weights(noise, arrays, field, geo, n_iterations=5)
    after = ped_rms('optimal')
    assert after <= before
    assert result.fun >= 0


def test_optimizer_deterministic(small_setup):
    geo, noise, field, arrays = small_setup

    weights = []
    for _ in range(2):
        for arr in arrays:
            arr.set_classical_weights(noise)
        optimize_speaker_weights(noise, arrays, field, geo,
                                 n_iterations=3, seed=42)
        weights.append(np.concatenate([arr.weights.copy() for arr in arrays]))

    np.testing.assert_array_equal(weights[0], weights[1])
