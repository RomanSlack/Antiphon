"""ANC performance metrics."""

import numpy as np

OCTAVE_BANDS = [63.0, 125.0, 250.0, 500.0]


def octave_band_levels(signal, sample_rate, centers=OCTAVE_BANDS, p_ref=2e-5):
    """RMS level (dB) of a time signal in octave bands.

    Bands span [fc/sqrt(2), fc*sqrt(2)] and are computed by FFT masking.
    Returns {center_frequency: dB}.
    """
    n = len(signal)
    spectrum = np.fft.rfft(signal)
    f = np.fft.rfftfreq(n, 1.0 / sample_rate)

    levels = {}
    for fc in centers:
        band = (f >= fc / np.sqrt(2)) & (f < fc * np.sqrt(2))
        # Parseval: band RMS^2 = 2*sum(|X_k|^2)/n^2 (one-sided)
        power = 2.0 * np.sum(np.abs(spectrum[band]) ** 2) / n ** 2
        rms = np.sqrt(power)
        levels[fc] = 20 * np.log10(max(rms, 1e-12) / p_ref)
    return levels


def compute_metrics(p_rms, geometry):
    """Compute ANC performance metrics."""
    ped_mask = geometry.ped_zone > 0

    # Average pressure in pedestrian zone
    avg_pressure_ped = np.mean(p_rms[ped_mask])

    # Quiet zone fraction (cells below threshold)
    threshold = 0.1 * np.max(p_rms)  # 10% of peak
    quiet_fraction = np.sum(p_rms[ped_mask] < threshold) / np.sum(ped_mask)

    # Max pressure in pedestrian zone
    max_pressure_ped = np.max(p_rms[ped_mask])

    return {
        'avg_pressure': avg_pressure_ped,
        'quiet_fraction': quiet_fraction,
        'max_pressure': max_pressure_ped,
    }
