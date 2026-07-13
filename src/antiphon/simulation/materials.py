"""Facade material properties: absorption coefficients and FDTD admittance.

Octave-band absorption coefficients for common facade materials. Endpoint
values (125 Hz, 4 kHz) follow the project spec; intermediate octaves are
interpolated from typical published ranges.

The time-domain solver takes a frequency-independent admittance per run, so
`band_admittance` picks the value at the frequency band being simulated.
Frequency-dependent boundary filters are a known future upgrade.
"""

import numpy as np

from .geometry import C_SOUND, RHO_AIR

OCTAVE_CENTERS = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])

# material -> absorption coefficient per octave band (125 Hz ... 4 kHz)
ABSORPTION = {
    'concrete': np.array([0.02, 0.02, 0.03, 0.03, 0.04, 0.05]),
    'glass':    np.array([0.18, 0.10, 0.06, 0.04, 0.03, 0.02]),
    'brick':    np.array([0.03, 0.03, 0.03, 0.04, 0.05, 0.07]),
}


def absorption_at(material, frequency):
    """Absorption coefficient alpha, log-frequency interpolated (clamped
    to the 125 Hz value below 125 Hz)."""
    alphas = ABSORPTION[material]
    return float(np.interp(np.log10(frequency),
                           np.log10(OCTAVE_CENTERS), alphas))


def admittance_from_alpha(alpha, rho=RHO_AIR, c=C_SOUND):
    """Specific admittance Y = 1/Z for a normal-incidence absorption
    coefficient alpha, assuming a real, locally-reacting impedance:
    R = sqrt(1 - alpha), Z = rho*c*(1+R)/(1-R)."""
    if alpha <= 0:
        return 0.0
    R = np.sqrt(1.0 - min(alpha, 1.0))
    if R >= 1.0:
        return 0.0
    Z = rho * c * (1 + R) / (1 - R)
    return 1.0 / Z


def band_admittance(material, frequency, rho=RHO_AIR, c=C_SOUND):
    """FDTD wall admittance for a material at a given frequency band."""
    return admittance_from_alpha(absorption_at(material, frequency), rho, c)
