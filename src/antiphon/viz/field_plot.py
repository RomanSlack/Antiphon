"""Pressure field visualization."""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from ..simulation.analytical import optimize_speaker_weights
from ..simulation.metrics import compute_metrics


def plot_results(field, noise_source, speaker_arrays, geometry,
                 save_path='anc_simulation_results.png'):
    """Generate comparison plots: noise only vs. ANC modes."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=120)

    modes = ['off', 'classical', 'optimal']
    titles = ['No ANC', 'Classical ANC', 'ML-Optimized ANC']

    for ax, mode, title in zip(axes, modes, titles):
        if mode == 'classical':
            for arr in speaker_arrays:
                arr.set_classical_weights(noise_source)

        if mode == 'optimal':
            # Reset to classical first, then optimize
            for arr in speaker_arrays:
                arr.set_classical_weights(noise_source)
            optimize_speaker_weights(noise_source, speaker_arrays, field, geometry)

        # Compute RMS pressure
        p_rms = field.compute_rms_pressure(noise_source, speaker_arrays, mode)
        spl = field.compute_spl(p_rms)

        # Compute metrics
        metrics = compute_metrics(p_rms, geometry)

        # Plot
        extent = [0, geometry.street_length, geometry.y_min, geometry.y_max]

        im = ax.imshow(spl.T, origin='lower', extent=extent,
                       cmap='RdYlBu_r', vmin=40, vmax=100, aspect='auto')

        # Draw buildings
        bld_y_bottom = geometry.y_min
        bld_y_top = geometry.street_width / 2
        ax.add_patch(Rectangle((0, bld_y_bottom), geometry.street_length,
                               geometry.building_depth,
                               facecolor='#555', alpha=0.8, edgecolor='#333'))
        ax.add_patch(Rectangle((0, bld_y_top), geometry.street_length,
                               geometry.building_depth,
                               facecolor='#555', alpha=0.8, edgecolor='#333'))

        # Draw sidewalk zones
        sw_y1 = -geometry.street_width / 2
        sw_y2 = geometry.street_width / 2 - geometry.sidewalk_width
        ax.add_patch(Rectangle((0, sw_y1), geometry.street_length,
                               geometry.sidewalk_width,
                               facecolor='none', edgecolor='white',
                               linewidth=1, linestyle='--', alpha=0.6))
        ax.add_patch(Rectangle((0, sw_y2), geometry.street_length,
                               geometry.sidewalk_width,
                               facecolor='none', edgecolor='white',
                               linewidth=1, linestyle='--', alpha=0.6))

        # Draw noise source
        ax.plot(noise_source.x, noise_source.y, 'r*', markersize=15,
                markeredgecolor='white', markeredgewidth=0.5, zorder=5)

        # Draw speakers
        if mode != 'off':
            for arr in speaker_arrays:
                for (sx, sy) in arr.positions:
                    ax.plot(sx, sy, 'g^', markersize=8,
                            markeredgecolor='white', markeredgewidth=0.5, zorder=5)

        ax.set_title(f'{title}\n'
                     f'Quiet: {metrics["quiet_fraction"]:.0%} | '
                     f'Avg: {metrics["avg_pressure"]:.2f} Pa',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Along street (m)')
        if ax == axes[0]:
            ax.set_ylabel('Across street (m)')

        plt.colorbar(im, ax=ax, label='SPL (dB)', shrink=0.8)

    # Annotations
    fig.suptitle(f'Urban ANC Simulation — {noise_source.frequency} Hz, '
                 f'{speaker_arrays[0].n_speakers} speakers/side, '
                 f'λ = {noise_source.wavelength:.2f} m',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Saved: {save_path}")

    plt.close()
    return fig


def render_animation_frames(field, noise_source, speaker_arrays, geometry,
                            output_dir, n_frames=30):
    """Render one cycle of the classical-ANC pressure field as PNG frames."""
    os.makedirs(output_dir, exist_ok=True)
    T = 1.0 / noise_source.frequency

    for arr in speaker_arrays:
        arr.set_classical_weights(noise_source)

    for i, t in enumerate(np.linspace(0, T, n_frames)):
        p = field.compute_pressure(noise_source, speaker_arrays, t, mode='classical')
        fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
        ax.imshow(p.T, origin='lower', cmap='RdBu',
                  vmin=-1, vmax=1, aspect='auto',
                  extent=[0, geometry.street_length,
                          geometry.y_min, geometry.y_max])
        ax.set_title(f'Frame {i+1}/{n_frames} — t = {t*1000:.1f}ms')
        fig.savefig(os.path.join(output_dir, f'frame_{i:03d}.png'),
                    bbox_inches='tight')
        plt.close()
        print(f"  Frame {i+1}/{n_frames}")

    print(f"Frames saved to {output_dir}/")
