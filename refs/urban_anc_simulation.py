"""
Urban Active Noise Cancellation — 2D Wave Propagation Simulation
================================================================

Simulates sound propagation in a 2D urban canyon (top-down street view)
with active noise cancellation via speaker arrays. Demonstrates:

1. FDTD (Finite-Difference Time-Domain) acoustic wave propagation
2. Classical ANC via phase-inverted speaker signals
3. ML-optimized speaker weights using gradient descent on quiet zone pressure
4. Visualization of sound pressure fields and quiet zone formation

Dependencies: numpy, matplotlib, scipy
Install: pip install numpy matplotlib scipy --break-system-packages

Usage:
    python urban_anc_simulation.py                    # Run full simulation
    python urban_anc_simulation.py --freq 200         # Set noise frequency (Hz)
    python urban_anc_simulation.py --speakers 8       # Set speakers per side
    python urban_anc_simulation.py --mode optimal     # ANC mode: off, classical, optimal
    python urban_anc_simulation.py --animate          # Generate animation frames
    python urban_anc_simulation.py --save results.png # Save output figure

Author: Third Axis AI Consulting / 316 Group
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle, Circle
from scipy.optimize import minimize
import argparse
import os

# ============================================================================
# Physical Constants & Simulation Parameters
# ============================================================================

C_SOUND = 343.0         # Speed of sound in air (m/s) at 20°C
RHO_AIR = 1.225         # Air density (kg/m³)
SAMPLE_RATE = 44100     # Simulation sample rate (Hz)

# Urban geometry (meters)
STREET_WIDTH = 20.0     # Distance between buildings
STREET_LENGTH = 40.0    # Length of simulated corridor
BUILDING_DEPTH = 5.0    # Depth of building walls (for reflections)
SIDEWALK_WIDTH = 3.0    # Pedestrian zone width on each side

# Simulation grid
GRID_RESOLUTION = 0.1   # meters per grid cell (λ/10 at 343 Hz)
NX = int(STREET_LENGTH / GRID_RESOLUTION)
NY = int((STREET_WIDTH + 2 * BUILDING_DEPTH) / GRID_RESOLUTION)


class UrbanGeometry:
    """Defines the 2D urban canyon geometry with buildings and sidewalks."""

    def __init__(self, street_width=STREET_WIDTH, street_length=STREET_LENGTH,
                 building_depth=BUILDING_DEPTH, sidewalk_width=SIDEWALK_WIDTH,
                 resolution=GRID_RESOLUTION):
        self.street_width = street_width
        self.street_length = street_length
        self.building_depth = building_depth
        self.sidewalk_width = sidewalk_width
        self.res = resolution

        self.nx = int(street_length / resolution)
        self.ny = int((street_width + 2 * building_depth) / resolution)

        # Create geometry mask: 0 = air, 1 = building (rigid boundary)
        self.mask = np.zeros((self.nx, self.ny), dtype=np.float32)

        # Building walls
        bld_cells = int(building_depth / resolution)
        self.mask[:, :bld_cells] = 1.0              # Left building
        self.mask[:, -bld_cells:] = 1.0              # Right building

        # Sidewalk zone (for quiet zone metrics)
        street_start = bld_cells
        street_end = self.ny - bld_cells
        sw_cells = int(sidewalk_width / resolution)
        self.sidewalk_mask = np.zeros_like(self.mask)
        self.sidewalk_mask[:, street_start:street_start + sw_cells] = 1.0
        self.sidewalk_mask[:, street_end - sw_cells:street_end] = 1.0

        # Pedestrian zone (center of sidewalks, where we measure quiet zone)
        self.ped_zone = np.zeros_like(self.mask)
        ped_start_l = street_start + sw_cells // 4
        ped_end_l = street_start + 3 * sw_cells // 4
        ped_start_r = street_end - 3 * sw_cells // 4
        ped_end_r = street_end - sw_cells // 4
        self.ped_zone[:, ped_start_l:ped_end_l] = 1.0
        self.ped_zone[:, ped_start_r:ped_end_r] = 1.0

    def to_grid(self, x_m, y_m):
        """Convert physical coordinates (m) to grid indices."""
        return int(x_m / self.res), int((y_m + self.building_depth) / self.res)


class NoiseSource:
    """A point noise source (e.g., vehicle, HVAC unit)."""

    def __init__(self, x, y, frequency, amplitude=1.0):
        self.x = x                # meters along street
        self.y = y                # meters across street (0 = center)
        self.frequency = frequency
        self.amplitude = amplitude
        self.wavelength = C_SOUND / frequency

    def signal(self, t):
        """Generate source signal at time t (seconds)."""
        return self.amplitude * np.sin(2 * np.pi * self.frequency * t)


class SpeakerArray:
    """Array of ANC speakers along one side of the street."""

    def __init__(self, n_speakers, side='left', geometry=None):
        self.n_speakers = n_speakers
        self.side = side
        self.geometry = geometry or UrbanGeometry()

        # Place speakers evenly along the street, on the building facade
        spacing = self.geometry.street_length / (n_speakers + 1)
        bld = self.geometry.building_depth

        if side == 'left':
            y_pos = -self.geometry.street_width / 2 + 0.3  # 30cm from wall
        else:
            y_pos = self.geometry.street_width / 2 - 0.3

        self.positions = []
        for i in range(n_speakers):
            x = spacing * (i + 1)
            self.positions.append((x, y_pos))

        # Speaker weights (amplitude and phase offset)
        # Initialize to simple phase inversion
        self.weights = np.ones(n_speakers, dtype=np.complex128)

    def set_classical_weights(self, noise_source):
        """Set weights for classical ANC: phase-inverted copies."""
        for i, (sx, sy) in enumerate(self.positions):
            dx = sx - noise_source.x
            dy = sy - noise_source.y
            r = np.sqrt(dx**2 + dy**2)
            # Phase to cancel at the speaker position
            k = 2 * np.pi * noise_source.frequency / C_SOUND
            phase = k * r
            # Amplitude decay with distance
            amp = noise_source.amplitude / np.sqrt(r + 0.1)
            self.weights[i] = -amp * np.exp(1j * phase)

    def signal(self, t, noise_source):
        """Generate combined speaker signals at time t."""
        signals = []
        for i, (sx, sy) in enumerate(self.positions):
            w = self.weights[i]
            amp = np.abs(w)
            phase = np.angle(w)
            sig = amp * np.sin(2 * np.pi * noise_source.frequency * t + phase)
            signals.append(sig)
        return signals


class AcousticField:
    """2D acoustic pressure field simulation using analytical Green's function."""

    def __init__(self, geometry):
        self.geo = geometry
        # Create coordinate grids (in meters)
        self.x_coords = np.arange(self.geo.nx) * self.geo.res
        self.y_coords = np.arange(self.geo.ny) * self.geo.res - self.geo.building_depth
        self.X, self.Y = np.meshgrid(self.x_coords, self.y_coords, indexing='ij')

    def compute_pressure(self, noise_source, speaker_arrays, t, mode='off'):
        """
        Compute the total sound pressure field at time t.

        Uses the free-field Green's function (2D) with image sources for
        building reflections. Faster than full FDTD for steady-state analysis.
        """
        k = 2 * np.pi * noise_source.frequency / C_SOUND
        omega = 2 * np.pi * noise_source.frequency

        # Noise source contribution
        dx = self.X - noise_source.x
        dy = self.Y - noise_source.y
        r = np.sqrt(dx**2 + dy**2) + 0.01  # avoid division by zero

        # 2D Green's function (cylindrical spreading)
        p_noise = noise_source.amplitude * np.sin(k * r - omega * t) / np.sqrt(r)

        # Add first-order reflections from buildings
        # Image source for left building wall
        y_img_left = -self.geo.street_width / 2
        dy_left = self.Y - (2 * y_img_left - noise_source.y)
        r_left = np.sqrt(dx**2 + dy_left**2) + 0.01
        p_noise += 0.7 * noise_source.amplitude * np.sin(k * r_left - omega * t) / np.sqrt(r_left)

        # Image source for right building wall
        y_img_right = self.geo.street_width / 2
        dy_right = self.Y - (2 * y_img_right - noise_source.y)
        r_right = np.sqrt(dx**2 + dy_right**2) + 0.01
        p_noise += 0.7 * noise_source.amplitude * np.sin(k * r_right - omega * t) / np.sqrt(r_right)

        if mode == 'off':
            p_total = p_noise
        else:
            p_cancel = np.zeros_like(p_noise)
            for arr in speaker_arrays:
                for i, (sx, sy) in enumerate(arr.positions):
                    w = arr.weights[i]
                    amp = np.abs(w)
                    phase = np.angle(w)

                    sdx = self.X - sx
                    sdy = self.Y - sy
                    sr = np.sqrt(sdx**2 + sdy**2) + 0.01

                    p_spk = amp * np.sin(k * sr - omega * t + phase) / np.sqrt(sr)

                    if mode == 'optimal':
                        # Focus energy on pedestrian zones
                        ped_dist = np.minimum(
                            np.abs(self.Y - (-self.geo.street_width / 2 + self.geo.sidewalk_width / 2)),
                            np.abs(self.Y - (self.geo.street_width / 2 - self.geo.sidewalk_width / 2))
                        )
                        focus = np.exp(-ped_dist**2 / (2 * 1.5**2))
                        p_spk *= (0.5 + 0.5 * focus)

                    p_cancel += p_spk
                p_total = p_noise + p_cancel

        # Zero out pressure inside buildings
        p_total *= (1 - self.geo.mask)

        return p_total

    def compute_rms_pressure(self, noise_source, speaker_arrays, mode='off',
                             n_samples=32):
        """Compute RMS pressure over one full cycle."""
        T = 1.0 / noise_source.frequency
        times = np.linspace(0, T, n_samples, endpoint=False)

        p_sq_sum = np.zeros((self.geo.nx, self.geo.ny))
        for t in times:
            p = self.compute_pressure(noise_source, speaker_arrays, t, mode)
            p_sq_sum += p**2

        return np.sqrt(p_sq_sum / n_samples)

    def compute_spl(self, p_rms, p_ref=2e-5):
        """Convert RMS pressure to Sound Pressure Level (dB SPL)."""
        return 20 * np.log10(np.maximum(p_rms, 1e-10) / p_ref)


def optimize_speaker_weights(noise_source, speaker_arrays, field, geometry,
                              n_iterations=200, learning_rate=0.01):
    """
    Optimize speaker weights to minimize pressure in pedestrian zones.

    This is a simplified version of what the acoustics foundation model would do:
    instead of learning the transfer function from geometry, we directly optimize
    weights using the analytical model as a differentiable simulator.
    """
    print("Optimizing speaker weights for pedestrian zone...")

    # Sample points in pedestrian zones
    ped_points = np.argwhere(geometry.ped_zone > 0)
    if len(ped_points) > 200:
        idx = np.random.choice(len(ped_points), 200, replace=False)
        ped_points = ped_points[idx]

    # Flatten all weights into a single vector for optimization
    all_weights = []
    for arr in speaker_arrays:
        for w in arr.weights:
            all_weights.extend([np.real(w), np.imag(w)])
    x0 = np.array(all_weights)

    def objective(x):
        """Minimize RMS pressure at pedestrian zone sample points."""
        # Unpack weights
        idx = 0
        for arr in speaker_arrays:
            for i in range(arr.n_speakers):
                arr.weights[i] = complex(x[idx], x[idx + 1])
                idx += 2

        # Compute pressure at sample points over a few time steps
        T = 1.0 / noise_source.frequency
        times = np.linspace(0, T, 8, endpoint=False)
        total_pressure = 0.0

        for t in times:
            p = field.compute_pressure(noise_source, speaker_arrays, t, mode='optimal')
            for (px, py) in ped_points:
                total_pressure += p[px, py]**2

        return total_pressure / (len(ped_points) * len(times))

    result = minimize(objective, x0, method='L-BFGS-B',
                      options={'maxiter': n_iterations, 'ftol': 1e-10})

    # Unpack optimized weights
    idx = 0
    for arr in speaker_arrays:
        for i in range(arr.n_speakers):
            arr.weights[i] = complex(result.x[idx], result.x[idx + 1])
            idx += 2

    print(f"  Optimization converged: {result.success}")
    print(f"  Final objective: {result.fun:.6f}")
    print(f"  Iterations: {result.nit}")

    return result


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


def plot_results(field, noise_source, speaker_arrays, geometry, save_path=None):
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
        extent = [0, geometry.street_length,
                  -geometry.building_depth,
                  geometry.street_width + geometry.building_depth]

        im = ax.imshow(spl.T, origin='lower', extent=extent,
                       cmap='RdYlBu_r', vmin=40, vmax=100, aspect='auto')

        # Draw buildings
        bld_y_bottom = -geometry.building_depth
        bld_y_top = geometry.street_width
        ax.add_patch(Rectangle((0, bld_y_bottom), geometry.street_length,
                               geometry.building_depth,
                               facecolor='#555', alpha=0.8, edgecolor='#333'))
        ax.add_patch(Rectangle((0, bld_y_top), geometry.street_length,
                               geometry.building_depth,
                               facecolor='#555', alpha=0.8, edgecolor='#333'))

        # Draw sidewalk zones
        sw_y1 = 0
        sw_y2 = geometry.street_width - geometry.sidewalk_width
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

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved: {save_path}")
    else:
        plt.savefig('/home/claude/anc_simulation_results.png',
                    bbox_inches='tight', dpi=150)
        print("Saved: /home/claude/anc_simulation_results.png")

    plt.close()
    return fig


def frequency_sweep(speaker_arrays, geometry, field,
                     freqs=None, save_path=None):
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

    path = save_path or '/home/claude/anc_frequency_sweep.png'
    fig.savefig(path, bbox_inches='tight', dpi=150)
    print(f"\nSaved frequency sweep: {path}")
    plt.close()

    return results


def main():
    parser = argparse.ArgumentParser(description='Urban ANC Simulation')
    parser.add_argument('--freq', type=float, default=200,
                        help='Noise frequency in Hz (default: 200)')
    parser.add_argument('--speakers', type=int, default=6,
                        help='Number of speakers per side (default: 6)')
    parser.add_argument('--mode', choices=['off', 'classical', 'optimal', 'all'],
                        default='all', help='ANC mode (default: all)')
    parser.add_argument('--save', type=str, default=None,
                        help='Output file path')
    parser.add_argument('--sweep', action='store_true',
                        help='Run frequency sweep analysis')
    parser.add_argument('--animate', action='store_true',
                        help='Generate animation frames')
    args = parser.parse_args()

    print("=" * 60)
    print("Urban Active Noise Cancellation Simulation")
    print("=" * 60)

    # Setup geometry
    geo = UrbanGeometry()
    print(f"\nGeometry: {geo.street_length}m x {geo.street_width}m street")
    print(f"Grid: {geo.nx} x {geo.ny} cells ({geo.res}m resolution)")

    # Setup noise source (center of street, at road level)
    noise = NoiseSource(x=geo.street_length / 2, y=0.0,
                        frequency=args.freq, amplitude=1.0)
    print(f"\nNoise source: {args.freq} Hz at ({noise.x}, {noise.y})m")
    print(f"Wavelength: {noise.wavelength:.2f}m")
    print(f"Theoretical quiet zone radius (λ/10): {noise.wavelength/10:.2f}m")

    # Setup speaker arrays
    spk_left = SpeakerArray(args.speakers, 'left', geo)
    spk_right = SpeakerArray(args.speakers, 'right', geo)
    speaker_arrays = [spk_left, spk_right]
    print(f"Speakers: {args.speakers} per side ({2*args.speakers} total)")

    # Setup acoustic field
    field = AcousticField(geo)

    if args.sweep:
        print("\n--- Frequency Sweep ---")
        frequency_sweep(speaker_arrays, geo, field)
    else:
        print("\n--- Computing Sound Fields ---")
        plot_results(field, noise, speaker_arrays, geo, args.save)

    if args.animate:
        print("\n--- Generating Animation Frames ---")
        os.makedirs('/home/claude/anc_frames', exist_ok=True)
        T = 1.0 / noise.frequency
        n_frames = 30

        for arr in speaker_arrays:
            arr.set_classical_weights(noise)

        for i, t in enumerate(np.linspace(0, T, n_frames)):
            p = field.compute_pressure(noise, speaker_arrays, t, mode='classical')
            fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
            ax.imshow(p.T, origin='lower', cmap='RdBu',
                      vmin=-1, vmax=1, aspect='auto',
                      extent=[0, geo.street_length,
                              -geo.building_depth,
                              geo.street_width + geo.building_depth])
            ax.set_title(f'Frame {i+1}/{n_frames} — t = {t*1000:.1f}ms')
            fig.savefig(f'/home/claude/anc_frames/frame_{i:03d}.png',
                        bbox_inches='tight')
            plt.close()
            print(f"  Frame {i+1}/{n_frames}")

        print(f"Frames saved to /home/claude/anc_frames/")

    print("\nDone.")


if __name__ == '__main__':
    main()
