"""Performance charts across frequencies."""

import matplotlib.pyplot as plt

from ..simulation.geometry import C_SOUND
from ..simulation.sources import NoiseSource
from ..simulation.metrics import compute_metrics

import numpy as np


def frequency_sweep(speaker_arrays, geometry, field,
                    freqs=None, save_path='anc_frequency_sweep.png'):
    """
    Sweep across frequencies to show where ANC is effective.
    Demonstrates the λ/10 quiet zone scaling law.
    """
    if freqs is None:
        freqs = [50, 100, 200, 500, 1000, 2000]

    results = {'freq': [], 'wavelength': [], 'quiet_frac_off': [],
               'quiet_frac_classical': [], 'quiet_frac_optimal': [],
               'reduction_classical': [], 'reduction_optimal': []}

    for f in freqs:
        print(f"\nFrequency: {f} Hz (λ = {C_SOUND/f:.2f} m)")
        noise = NoiseSource(x=geometry.street_length / 2,
                            y=0.0, frequency=f, amplitude=1.0)

        # No ANC
        p_off = field.compute_rms_pressure(noise, speaker_arrays, 'off')
        m_off = compute_metrics(p_off, geometry)

        # Classical ANC
        for arr in speaker_arrays:
            arr.set_classical_weights(noise)
        p_cl = field.compute_rms_pressure(noise, speaker_arrays, 'classical')
        m_cl = compute_metrics(p_cl, geometry)

        # Compute dB reduction
        reduction_cl = 20 * np.log10(
            max(m_off['avg_pressure'], 1e-10) /
            max(m_cl['avg_pressure'], 1e-10)
        )

        results['freq'].append(f)
        results['wavelength'].append(C_SOUND / f)
        results['quiet_frac_off'].append(m_off['quiet_fraction'])
        results['quiet_frac_classical'].append(m_cl['quiet_fraction'])
        results['reduction_classical'].append(max(reduction_cl, 0))

        print(f"  No ANC quiet fraction: {m_off['quiet_fraction']:.1%}")
        print(f"  Classical quiet fraction: {m_cl['quiet_fraction']:.1%}")
        print(f"  Reduction: {reduction_cl:.1f} dB")

    # Plot frequency sweep
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=120)

    ax1.semilogx(results['freq'], results['quiet_frac_classical'],
                 'o-', color='#1d9e75', linewidth=2, markersize=8,
                 label='Classical ANC')
    ax1.semilogx(results['freq'], results['quiet_frac_off'],
                 's--', color='#888', linewidth=1.5, markersize=6,
                 label='No ANC')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Quiet zone fraction')
    ax1.set_title('Quiet zone coverage vs. frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    ax2.semilogx(results['freq'], results['reduction_classical'],
                 'o-', color='#534ab7', linewidth=2, markersize=8)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('dB reduction in pedestrian zone')
    ax2.set_title('Noise reduction vs. frequency')
    ax2.grid(True, alpha=0.3)

    # Add wavelength annotations
    for f, wl in zip(results['freq'], results['wavelength']):
        ax2.annotate(f'λ={wl:.1f}m', (f, 0), fontsize=8,
                     rotation=45, ha='left', va='bottom', alpha=0.6)

    plt.suptitle(f'ANC Performance Across Frequencies — '
                 f'{speaker_arrays[0].n_speakers} speakers/side',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"\nSaved frequency sweep: {save_path}")
    plt.close()

    return results
