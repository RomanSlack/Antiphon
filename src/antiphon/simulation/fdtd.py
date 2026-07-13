"""2D FDTD acoustic wave solver.

Staggered (Yee) grid: pressure at cell centers, velocity components on cell
faces. Split-field PML absorbing layers on all four domain edges. Building
cells support rigid walls (zero normal velocity) or locally-reacting
impedance boundaries (v_n = p / Z at the surface).

This solver is the ground-truth engine: the analytical Green's-function
solver (analytical.py) is the fast approximation, FDTD is what training data
and validation are built on.
"""

import numpy as np

from .geometry import C_SOUND, RHO_AIR


class FDTDSolver:
    """2D acoustic FDTD on a rectangular grid with building mask.

    Parameters
    ----------
    mask : 2D array (nx, ny)
        1 where building (solid), 0 where air.
    dx : float
        Grid spacing in meters. Accurate up to roughly f_max = c / (10 * dx).
    admittance : float or 2D array
        Specific acoustic admittance (1/Z, in 1/rayl) of building surfaces.
        0 = rigid. If an array, gives per-cell material admittance and the
        value of the building cell adjacent to each face is used.
    courant : float
        Fraction of the 2D stability limit dt = dx / (c * sqrt(2)).
    pml_cells : int
        Thickness of the absorbing layer on each edge, in cells.
    """

    def __init__(self, mask, dx, admittance=0.0, c=C_SOUND, rho=RHO_AIR,
                 courant=0.9, pml_cells=20):
        self.mask = np.asarray(mask) > 0.5
        self.nx, self.ny = self.mask.shape
        self.dx = dx
        self.c = c
        self.rho = rho
        self.dt = courant * dx / (c * np.sqrt(2.0))
        self.pml_cells = pml_cells

        if np.isscalar(admittance):
            adm = np.full((self.nx, self.ny), float(admittance))
        else:
            adm = np.asarray(admittance, dtype=float)

        self._build_pml()
        self._build_boundaries(adm)
        self.reset()

    @classmethod
    def from_geometry(cls, geometry, **kwargs):
        """Build a solver from an UrbanGeometry (uses its mask and resolution)."""
        return cls(geometry.mask, geometry.res, **kwargs)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _sigma_profile(self, n, n_pml, staggered):
        """PML conductivity along one axis (cubic grading)."""
        # sigma_max for target reflection R0 with polynomial order m=3
        R0 = 1e-6
        m = 3
        L = n_pml * self.dx
        sigma_max = -(m + 1) * self.c * np.log(R0) / (2 * L)

        coords = np.arange(n, dtype=float)
        if staggered:
            coords -= 0.5  # face positions sit half a cell before centers
        sigma = np.zeros(n)
        if n_pml > 0:
            d_lo = (n_pml - coords) / n_pml          # depth into low-side PML
            d_hi = (coords - (n - 1 - n_pml)) / n_pml
            sigma = sigma_max * (np.clip(d_lo, 0, 1) ** m
                                 + np.clip(d_hi, 0, 1) ** m)
        return sigma

    def _build_pml(self):
        n = self.pml_cells
        # At pressure cells
        sx_p = self._sigma_profile(self.nx, n, staggered=False)
        sy_p = self._sigma_profile(self.ny, n, staggered=False)
        # At velocity faces (vx has nx+1 face columns, vy has ny+1 face rows)
        sx_v = self._sigma_profile(self.nx + 1, n, staggered=True)
        sy_v = self._sigma_profile(self.ny + 1, n, staggered=True)

        dt2 = self.dt / 2
        f32 = np.float32
        # Pressure update coefficients: (1 - s*dt/2)/(1 + s*dt/2) etc.
        ax = (sx_p * dt2)[:, None] * np.ones((1, self.ny))
        ay = (sy_p * dt2)[None, :] * np.ones((self.nx, 1))
        kp = self.rho * self.c ** 2 * self.dt / self.dx
        self._px_c1 = ((1 - ax) / (1 + ax)).astype(f32)
        self._px_c2 = (kp / (1 + ax)).astype(f32)
        self._py_c1 = ((1 - ay) / (1 + ay)).astype(f32)
        self._py_c2 = (kp / (1 + ay)).astype(f32)

        # Velocity update coefficients (interior faces only)
        kv = self.dt / (self.rho * self.dx)
        bx = (sx_v[1:-1] * dt2)[:, None] * np.ones((1, self.ny))
        by = (sy_v[1:-1] * dt2)[None, :] * np.ones((self.nx, 1))
        self._vx_c1 = ((1 - bx) / (1 + bx)).astype(f32)
        self._vx_c2 = (kv / (1 + bx)).astype(f32)
        self._vy_c1 = ((1 - by) / (1 + by)).astype(f32)
        self._vy_c2 = (kv / (1 + by)).astype(f32)

    def _build_boundaries(self, adm):
        """Precompute boundary-face masks and per-face admittances."""
        wall = self.mask
        air = ~wall

        # x-faces between cell columns i-1 and i -> shape (nx-1, ny)
        self._fx_aw = air[:-1, :] & wall[1:, :]   # air left, wall right
        self._fx_wa = wall[:-1, :] & air[1:, :]   # wall left, air right
        self._fx_ww = wall[:-1, :] & wall[1:, :]
        self._fx_aw_Y = adm[1:, :][self._fx_aw].astype(np.float32)
        self._fx_wa_Y = adm[:-1, :][self._fx_wa].astype(np.float32)

        # y-faces between cell rows j-1 and j -> shape (nx, ny-1)
        self._fy_aw = air[:, :-1] & wall[:, 1:]
        self._fy_wa = wall[:, :-1] & air[:, 1:]
        self._fy_ww = wall[:, :-1] & wall[:, 1:]
        self._fy_aw_Y = adm[:, 1:][self._fy_aw].astype(np.float32)
        self._fy_wa_Y = adm[:, :-1][self._fy_wa].astype(np.float32)

        self._air = air

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------

    def reset(self):
        f32 = np.float32
        self.px = np.zeros((self.nx, self.ny), dtype=f32)
        self.py = np.zeros((self.nx, self.ny), dtype=f32)
        self.vx = np.zeros((self.nx + 1, self.ny), dtype=f32)
        self.vy = np.zeros((self.nx, self.ny + 1), dtype=f32)
        self._p_buf = np.zeros((self.nx, self.ny), dtype=f32)

    @property
    def p(self):
        return self.px + self.py

    def step(self, source_ix=None, source_value=0.0):
        """Advance one time step. Optionally inject a soft pressure source."""
        p = np.add(self.px, self.py, out=self._p_buf)

        # Velocity updates (interior faces; outer domain faces stay 0,
        # the PML absorbs anything before it reaches them)
        vxi = self.vx[1:-1, :]
        vxi *= self._vx_c1
        vxi -= self._vx_c2 * (p[1:, :] - p[:-1, :])

        vyi = self.vy[:, 1:-1]
        vyi *= self._vy_c1
        vyi -= self._vy_c2 * (p[:, 1:] - p[:, :-1])

        # Boundary conditions at building faces.
        # Impedance: v_n = p_air / Z with outward normal into the wall;
        # rigid is the Y=0 special case.
        vxi[self._fx_aw] = self._fx_aw_Y * p[:-1, :][self._fx_aw]
        vxi[self._fx_wa] = -self._fx_wa_Y * p[1:, :][self._fx_wa]
        vxi[self._fx_ww] = 0.0
        vyi[self._fy_aw] = self._fy_aw_Y * p[:, :-1][self._fy_aw]
        vyi[self._fy_wa] = -self._fy_wa_Y * p[:, 1:][self._fy_wa]
        vyi[self._fy_ww] = 0.0

        # Pressure updates (split field)
        self.px *= self._px_c1
        self.px -= self._px_c2 * (self.vx[1:, :] - self.vx[:-1, :])
        self.py *= self._py_c1
        self.py -= self._py_c2 * (self.vy[:, 1:] - self.vy[:, :-1])

        # Keep solid cells at zero pressure
        self.px[self.mask] = 0.0
        self.py[self.mask] = 0.0

        if source_ix is not None:
            i, j = source_ix
            self.px[i, j] += 0.5 * source_value
            self.py[i, j] += 0.5 * source_value

    def run(self, source_ix, source_signal, receiver_ix, n_steps=None):
        """Run the simulation, recording pressure at receiver cells.

        Parameters
        ----------
        source_ix : (i, j) grid indices of the source cell
        source_signal : 1D array of source samples (one per step)
        receiver_ix : list of (i, j) receiver cells
        n_steps : total steps (defaults to len(source_signal))

        Returns
        -------
        traces : array (n_receivers, n_steps)
        """
        self.reset()
        n_steps = n_steps or len(source_signal)
        rx = np.array([r[0] for r in receiver_ix])
        ry = np.array([r[1] for r in receiver_ix])
        traces = np.zeros((len(receiver_ix), n_steps))

        for n in range(n_steps):
            s = source_signal[n] if n < len(source_signal) else 0.0
            self.step(source_ix, s)
            traces[:, n] = self.px[rx, ry] + self.py[rx, ry]

        return traces

    # ------------------------------------------------------------------
    # Impulse response / transfer functions
    # ------------------------------------------------------------------

    def gaussian_pulse(self, f_max):
        """Gaussian pulse with usable energy up to f_max (about -20 dB there)."""
        tau = 1.517 / (np.pi * f_max)
        t0 = 4 * tau
        n = int(np.ceil(2 * t0 / self.dt))
        t = np.arange(n) * self.dt
        return np.exp(-(((t - t0) / tau) ** 2))

    def transfer_function(self, source_ix, receiver_ix, freqs,
                          f_max=600.0, duration=0.6):
        """Measure complex transfer functions H(f) source -> receivers.

        Runs a Gaussian pulse through the domain and deconvolves:
        H(f) = P_receiver(f) / S(f), evaluated at the requested frequencies.
        """
        signal = self.gaussian_pulse(f_max)
        n_steps = int(np.ceil(duration / self.dt))
        traces = self.run(source_ix, signal, receiver_ix, n_steps)

        src = np.zeros(n_steps)
        src[:len(signal)] = signal[:n_steps]
        f_axis = np.fft.rfftfreq(n_steps, self.dt)
        S = np.fft.rfft(src)
        P = np.fft.rfft(traces, axis=1)

        H = np.empty((len(receiver_ix), len(freqs)), dtype=complex)
        for k, f in enumerate(freqs):
            idx = np.argmin(np.abs(f_axis - f))
            H[:, k] = P[:, idx] / S[idx]
        return H

    def impulse_response(self, source_ix, receiver_ix, f_lo=20.0, f_hi=500.0,
                         duration=0.5, ir_len=None):
        """Band-limited impulse responses source -> receivers.

        Deconvolves the Gaussian pulse (regularized) and applies a raised-
        cosine band-limit window over [f_lo, f_hi], so the returned FIR is
        independent of the probe pulse within the band.

        Returns
        -------
        irs : array (n_receivers, ir_len) at sample rate 1/dt
        sample_rate : float
        """
        signal = self.gaussian_pulse(f_max=1.2 * f_hi)
        n_steps = int(np.ceil(duration / self.dt))
        traces = self.run(source_ix, signal, receiver_ix, n_steps)

        src = np.zeros(n_steps)
        src[:len(signal)] = signal[:n_steps]
        f_axis = np.fft.rfftfreq(n_steps, self.dt)
        S = np.fft.rfft(src)
        P = np.fft.rfft(traces, axis=1)

        # Raised-cosine band window with 20% transition edges
        w = np.zeros_like(f_axis)
        lo_t, hi_t = 0.2 * f_lo, 0.2 * f_hi
        rising = (f_axis >= f_lo - lo_t) & (f_axis < f_lo + lo_t)
        w[rising] = 0.5 * (1 - np.cos(
            np.pi * (f_axis[rising] - (f_lo - lo_t)) / (2 * lo_t)))
        w[(f_axis >= f_lo + lo_t) & (f_axis <= f_hi - hi_t)] = 1.0
        falling = (f_axis > f_hi - hi_t) & (f_axis <= f_hi + hi_t)
        w[falling] = 0.5 * (1 + np.cos(
            np.pi * (f_axis[falling] - (f_hi - hi_t)) / (2 * hi_t)))

        eps = 1e-3 * np.max(np.abs(S))
        H = P * w / (S + eps)
        irs = np.fft.irfft(H, n_steps, axis=1)
        if ir_len is not None:
            irs = irs[:, :ir_len]
        return irs, 1.0 / self.dt
