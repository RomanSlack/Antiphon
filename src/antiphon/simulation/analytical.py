"""2D analytical acoustic field solver (Green's functions with image sources).

Fast steady-state solver used for interactive demos and parameter sweeps.
The FDTD solver (fdtd.py, planned) is the ground-truth reference.
"""

import numpy as np
from scipy.optimize import minimize

from .geometry import C_SOUND


class AcousticField:
    """2D acoustic pressure field simulation using analytical Green's function."""

    def __init__(self, geometry):
        self.geo = geometry
        # Create coordinate grids (in meters)
        self.x_coords = np.arange(self.geo.nx) * self.geo.res
        self.y_coords = np.arange(self.geo.ny) * self.geo.res + self.geo.y_min
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
                             n_iterations=200, seed=0):
    """
    Optimize speaker weights to minimize pressure in pedestrian zones.

    This is a simplified version of what the acoustics foundation model would do:
    instead of learning the transfer function from geometry, we directly optimize
    weights using the analytical model as a differentiable simulator.
    """
    print("Optimizing speaker weights for pedestrian zone...")

    # Sample points in pedestrian zones
    rng = np.random.default_rng(seed)
    ped_points = np.argwhere(geometry.ped_zone > 0)
    if len(ped_points) > 200:
        idx = rng.choice(len(ped_points), 200, replace=False)
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
