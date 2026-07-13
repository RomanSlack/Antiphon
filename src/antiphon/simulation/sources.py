"""Noise sources, broadband signal generators, and ANC speaker arrays."""

import numpy as np

from .geometry import C_SOUND, SAMPLE_RATE, UrbanGeometry


def _shaped_noise(n, sample_rate, envelope_fn, rng):
    """White noise shaped in the frequency domain by envelope_fn(f)."""
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1.0 / sample_rate)
    spectrum *= envelope_fn(f)
    sig = np.fft.irfft(spectrum, n)
    return sig / (np.std(sig) + 1e-12)


def traffic_noise(duration, sample_rate=SAMPLE_RATE, seed=0):
    """Road-traffic-like noise: pink noise shaped by an approximate road
    noise spectrum (low-frequency rumble emphasis, rolling off above 1 kHz,
    after the general shape of ISO 11819 measurements)."""
    rng = np.random.default_rng(seed)
    n = int(duration * sample_rate)

    def envelope(f):
        fs = np.maximum(f, 1.0)
        pink = 1.0 / np.sqrt(fs)                     # -3 dB/octave
        rumble = 1.0 + 2.0 * np.exp(-((np.log2(fs / 80.0)) ** 2) / 2.0)
        rolloff = 1.0 / (1.0 + (fs / 1000.0) ** 2)
        env = pink * rumble * rolloff
        env[f < 20.0] = 0.0
        return env

    return _shaped_noise(n, sample_rate, envelope, rng)


def hvac_noise(duration, sample_rate=SAMPLE_RATE, blade_pass_freq=90.0,
               n_harmonics=3, tonal_ratio=0.7, seed=0):
    """HVAC-like noise: tonal components at the fan blade-pass frequency and
    harmonics, plus a broadband floor."""
    rng = np.random.default_rng(seed)
    n = int(duration * sample_rate)
    t = np.arange(n) / sample_rate

    tonal = np.zeros(n)
    for h in range(1, n_harmonics + 1):
        phase = rng.uniform(0, 2 * np.pi)
        tonal += np.sin(2 * np.pi * blade_pass_freq * h * t + phase) / h
    tonal /= np.std(tonal) + 1e-12

    def envelope(f):
        fs = np.maximum(f, 1.0)
        env = 1.0 / np.sqrt(fs)
        env[f < 20.0] = 0.0
        return env

    broadband = _shaped_noise(n, sample_rate, envelope, rng)
    return tonal_ratio * tonal + (1 - tonal_ratio) * broadband


def construction_noise(duration, sample_rate=SAMPLE_RATE, impact_rate=2.0,
                       seed=0):
    """Construction-like noise: repeated impulsive impacts over a broadband
    floor."""
    rng = np.random.default_rng(seed)
    n = int(duration * sample_rate)
    t = np.arange(n) / sample_rate

    sig = np.zeros(n)
    n_impacts = max(1, int(duration * impact_rate))
    tau = 0.01  # impact decay time constant (s)
    for _ in range(n_impacts):
        t0 = rng.uniform(0, duration)
        after = t >= t0
        sig[after] += np.exp(-(t[after] - t0) / tau) * \
            np.sin(2 * np.pi * 150.0 * (t[after] - t0))

    def envelope(f):
        env = np.ones_like(f)
        env[f < 20.0] = 0.0
        return env

    floor = _shaped_noise(n, sample_rate, envelope, rng)
    sig = sig / (np.std(sig) + 1e-12)
    return 0.8 * sig + 0.2 * floor


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
