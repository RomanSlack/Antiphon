"""Synthetic training data: randomized street canyons -> transfer functions.

Each scene is a randomized urban canyon (street width, facade absorption,
source position) simulated on a fixed physical domain. One FDTD run per
scene measures complex transfer functions H(f) from the source to many
receivers. A training sample is (occupancy grid, source, receiver,
absorptions) -> H(f).

All randomness is seeded per scene, so generation is deterministic
regardless of worker scheduling.
"""

import numpy as np

from ..simulation.fdtd import FDTDSolver
from ..simulation.materials import admittance_from_alpha

# Fixed physical domain (meters) and grid
DOMAIN_X = 24.0
DOMAIN_Y = 36.0
DX = 0.08
NX = int(DOMAIN_X / DX)      # 300
NY = int(DOMAIN_Y / DX)      # 450

# Frequency grid: log-spaced, sub-500 Hz band (grid supports ~430 Hz at
# 10 points per wavelength)
N_FREQS = 64
F_LO, F_HI = 30.0, 430.0
FREQS = np.logspace(np.log10(F_LO), np.log10(F_HI), N_FREQS)

# Occupancy grid fed to the model (downsampled by 5x)
OCC_DOWNSAMPLE = 5
OCC_SHAPE = (NX // OCC_DOWNSAMPLE, NY // OCC_DOWNSAMPLE)  # (60, 90)

N_RECEIVERS = 24


def make_canyon_mask(street_width):
    """Fixed-size domain: street |y| < W/2 is air, the rest is building."""
    mask = np.zeros((NX, NY), dtype=np.float32)
    y = (np.arange(NY) + 0.5) * DX - DOMAIN_Y / 2  # cell centers, centerline 0
    wall = np.abs(y) >= street_width / 2
    mask[:, wall] = 1.0
    return mask, y


def scene_params(seed):
    """Draw the random parameters of one scene."""
    rng = np.random.default_rng(seed)
    width = rng.uniform(8.0, 30.0)
    alpha_left = rng.uniform(0.01, 0.25)
    alpha_right = rng.uniform(0.01, 0.25)
    src_x = rng.uniform(2.0, DOMAIN_X - 2.0)
    src_y = rng.uniform(-width / 2 + 0.25, width / 2 - 0.25)

    rcv = []
    while len(rcv) < N_RECEIVERS:
        rx = rng.uniform(1.5, DOMAIN_X - 1.5)
        ry = rng.uniform(-width / 2 + 0.3, width / 2 - 0.3)
        if np.hypot(rx - src_x, ry - src_y) >= 1.0:
            rcv.append((rx, ry))

    return {
        'width': width,
        'alpha_left': alpha_left,
        'alpha_right': alpha_right,
        'src': (src_x, src_y),
        'rcv': np.array(rcv),
    }


def to_ix(x, y):
    """Physical coordinates (centerline convention) to grid indices."""
    return int(x / DX), int((y + DOMAIN_Y / 2) / DX)


def downsample_occupancy(mask):
    """Full-resolution mask -> model-input occupancy grid."""
    ds = OCC_DOWNSAMPLE
    occ = mask[:NX - NX % ds, :NY - NY % ds]
    occ = occ.reshape(OCC_SHAPE[0], ds, OCC_SHAPE[1], ds).mean(axis=(1, 3))
    return occ.astype(np.float32)


def generate_scene(seed, duration=0.4):
    """Simulate one scene. Returns dict with occupancy, params, and H."""
    params = scene_params(seed)
    mask, y_centers = make_canyon_mask(params['width'])

    adm = np.zeros((NX, NY))
    adm[:, y_centers < 0] = admittance_from_alpha(params['alpha_left'])
    adm[:, y_centers > 0] = admittance_from_alpha(params['alpha_right'])

    solver = FDTDSolver(mask, DX, admittance=adm, pml_cells=20)
    src_ix = to_ix(*params['src'])
    rcv_ix = [to_ix(x, y) for (x, y) in params['rcv']]

    H = solver.transfer_function(src_ix, rcv_ix, FREQS,
                                 f_max=1.2 * F_HI, duration=duration)

    return {
        'seed': seed,
        'occupancy': downsample_occupancy(mask),
        'width': params['width'],
        'alpha': np.array([params['alpha_left'], params['alpha_right']],
                          dtype=np.float32),
        'src': np.array(params['src'], dtype=np.float32),
        'rcv': params['rcv'].astype(np.float32),
        'H': H.astype(np.complex64),
    }


def write_dataset(path, scenes):
    """Write a list of scene dicts to HDF5."""
    import h5py

    n_scenes = len(scenes)
    with h5py.File(path, 'w') as f:
        f.attrs['freqs'] = FREQS
        f.attrs['domain'] = (DOMAIN_X, DOMAIN_Y)
        f.attrs['dx'] = DX
        f.create_dataset('seed', data=[s['seed'] for s in scenes])
        f.create_dataset('occupancy',
                         data=np.stack([s['occupancy'] for s in scenes]))
        f.create_dataset('width', data=[s['width'] for s in scenes])
        f.create_dataset('alpha', data=np.stack([s['alpha'] for s in scenes]))
        f.create_dataset('src', data=np.stack([s['src'] for s in scenes]))
        f.create_dataset('rcv', data=np.stack([s['rcv'] for s in scenes]))
        H = np.stack([s['H'] for s in scenes])  # (S, R, F) complex
        f.create_dataset('H_real', data=H.real)
        f.create_dataset('H_imag', data=H.imag)
    return n_scenes
