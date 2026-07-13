"""Inference utilities: predict transfer functions and build FIR filters.

The model outputs delay-compensated H at N_FREQS log-spaced frequencies.
To use predictions as FxLMS secondary-path filters we:
1. reapply the physical propagation delay phase,
2. interpolate the (smooth) compensated prediction onto a dense FFT grid,
3. apply the band-limit window and inverse-FFT to a real FIR.
"""

import numpy as np
import torch

from ..simulation.geometry import C_SOUND
from .architecture import AcousticsModelV1
from .dataset import DOMAIN_X, DOMAIN_Y


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    freqs = np.asarray(ckpt['freqs'])
    model = AcousticsModelV1(n_freqs=len(freqs))
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, freqs, float(ckpt['h_scale'])


def norm_pos(pos):
    out = np.array(pos, dtype=np.float32)
    out[..., 0] = out[..., 0] / DOMAIN_X
    out[..., 1] = (out[..., 1] + DOMAIN_Y / 2) / DOMAIN_Y
    return out


def predict_H(model, freqs, h_scale, occupancy, alpha, src, rcv):
    """Predict physical (delay-restored) H for query pairs in one scene.

    src, rcv : arrays (Q, 2) of physical positions (centerline convention)
    Returns complex array (Q, F).
    """
    src = np.atleast_2d(src).astype(np.float32)
    rcv = np.atleast_2d(rcv).astype(np.float32)
    occ_t = torch.from_numpy(occupancy[None].astype(np.float32))
    alpha_t = torch.from_numpy(np.asarray(alpha, dtype=np.float32)[None])
    src_t = torch.from_numpy(norm_pos(src)[None])
    rcv_t = torch.from_numpy(norm_pos(rcv)[None])

    with torch.no_grad():
        pred = model(occ_t, src_t, rcv_t, alpha_t)[0].numpy()  # compensated
    pred = pred * h_scale

    dist = np.linalg.norm(rcv - src, axis=-1)
    k = 2 * np.pi * freqs / C_SOUND
    return pred * np.exp(-1j * dist[:, None] * k)


def h_to_fir(H, freqs, dist, fs, ir_len):
    """Convert sparse H(f) samples into a real FIR filter.

    Interpolation happens in delay-compensated space (smooth), then the
    propagation delay is reapplied on the dense grid before the IFFT.

    H : (F,) complex at `freqs`; dist : source-receiver distance (m)
    """
    n = ir_len * 2
    f_dense = np.fft.rfftfreq(n, 1.0 / fs)
    k_sparse = 2 * np.pi * np.asarray(freqs) / C_SOUND
    k_dense = 2 * np.pi * f_dense / C_SOUND

    H_comp = H * np.exp(1j * dist * k_sparse)
    re = np.interp(f_dense, freqs, H_comp.real, left=0.0, right=0.0)
    im = np.interp(f_dense, freqs, H_comp.imag, left=0.0, right=0.0)
    H_dense = (re + 1j * im) * np.exp(-1j * dist * k_dense)

    # Raised-cosine taper at the band edges (matches the measured IRs' band)
    f_lo, f_hi = freqs[0], freqs[-1]
    w = np.ones_like(f_dense)
    w[f_dense < f_lo] = 0.0
    w[f_dense > f_hi] = 0.0
    lo_t, hi_t = 0.2 * f_lo, 0.1 * (f_hi - f_lo)
    rise = (f_dense >= f_lo) & (f_dense < f_lo + lo_t)
    w[rise] = 0.5 * (1 - np.cos(np.pi * (f_dense[rise] - f_lo) / lo_t))
    fall = (f_dense > f_hi - hi_t) & (f_dense <= f_hi)
    w[fall] = 0.5 * (1 + np.cos(np.pi * (f_dense[fall] - (f_hi - hi_t)) / hi_t))

    ir = np.fft.irfft(H_dense * w, n)
    return ir[:ir_len]
