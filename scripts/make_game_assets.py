"""Bake assets for the playable 2D demo (web/game).

For every walkable grid point we bake two combined impulse responses w.r.t.
the looped noise reference x:
    h_off = IR_noise                      (what you hear, ANC off)
    h_on  = IR_noise + sum_j IR_j * w_j   (speaker filters folded in)
so the browser only needs two convolvers and one noise loop.

Requires data/demo/irs cache stage to have run (uses the Wiener FIRs from
data/demo/paths.npz and the scene constants from make_walk_demo).

Outputs to web/game/assets/:
    meta.json, irs.bin (float16), noise_low.bin (float16, 12 kHz),
    highs.ogg, field_off.png, field_on.png
"""

import json
import os
import subprocess
import sys

import numpy as np
from scipy.signal import fftconvolve, resample_poly

sys.path.insert(0, os.path.dirname(__file__))
from make_walk_demo import (  # noqa: E402
    ALPHA, BENCH, CTL_DECIM, DEMO, DOMAIN_X, DOMAIN_Y, DX, ERROR_MICS,
    F_HI, F_LO, FS_AUDIO, FS_SIM, IR_LEN, NOISE_POS, SPEAKERS, WIDTH,
    load_bands, solver, to_ix,
)

ASSETS = os.path.join(os.path.dirname(__file__), '..', 'web', 'game',
                      'assets')

# Walkable listener grid (0.5 m)
GRID_DX = 0.5
GX = np.arange(2.5, 21.6, GRID_DX)
GY = np.arange(-6.0, 6.01, GRID_DX)

# Everything ships at 12 kHz (WebAudio AudioBuffer needs >= 8 kHz)
FS_SHIP = 12000
UP, DOWN = 7, 4  # 6857.14 * 7/4 = 12000 exactly


def ship_signal(x_sim):
    """Resample a SIGNAL from fs_sim to 12 kHz (values preserved)."""
    return resample_poly(x_sim, up=UP, down=DOWN)


def ship_filter(h_sim):
    """Resample a FILTER from fs_sim to 12 kHz (transfer preserved)."""
    return resample_poly(h_sim, up=UP, down=DOWN, axis=-1) * (DOWN / UP)


def bake_irs():
    """9 FDTD runs -> h_off/h_on at every grid point."""
    pts = [(x, y) for x in GX for y in GY]
    rcv_ix = [to_ix(x, y) for (x, y) in pts]
    n_pts = len(pts)

    print(f'measuring IRs to {n_pts} grid points...', flush=True)
    s = solver()
    IR_noise, _ = s.impulse_response(to_ix(*NOISE_POS), rcv_ix, f_lo=F_LO,
                                     f_hi=F_HI, ir_len=IR_LEN, duration=1.0)
    IR_spk = []
    for j, spk in enumerate(SPEAKERS):
        print(f'  speaker {j+1}/{len(SPEAKERS)}', flush=True)
        IR_spk.append(s.impulse_response(to_ix(*spk), rcv_ix, f_lo=F_LO,
                                         f_hi=F_HI, ir_len=IR_LEN,
                                         duration=1.0)[0])

    # Wiener FIRs (control rate) -> simulation rate, transfer-preserving
    w_ctl = np.load(os.path.join(DEMO, 'paths.npz'))['w']  # (J, LW)
    w_sim = resample_poly(w_ctl, up=CTL_DECIM, down=1, axis=-1) / CTL_DECIM

    print('folding speaker filters into h_on...', flush=True)
    h_off = IR_noise                                 # (n_pts, IR_LEN)
    h_on = h_off.copy()
    h_on_len = IR_LEN + w_sim.shape[1] - 1
    h_on = np.zeros((n_pts, h_on_len))
    h_on[:, :IR_LEN] = h_off
    for j in range(len(SPEAKERS)):
        h_on += fftconvolve(IR_spk[j], w_sim[j][None, :], axes=1)

    # Ship both at the same length, 12 kHz, float16
    h_off_p = np.zeros((n_pts, h_on_len))
    h_off_p[:, :IR_LEN] = h_off
    h_off_12 = ship_filter(h_off_p).astype(np.float16)
    h_on_12 = ship_filter(h_on).astype(np.float16)
    return pts, h_off_12, h_on_12


def bake_audio():
    """Noise loop (low band, 12 kHz) and highs loop (ogg), seam-faded."""
    _, low, highs, low_sim = load_bands()

    def loopify(x, fs, fade=0.5):
        n = int(fade * fs)
        y = x[:-n].copy()
        ramp = np.linspace(0, 1, n)
        y[:n] = y[:n] * ramp + x[-n:] * (1 - ramp)
        return y

    x12 = ship_signal(loopify(low_sim, FS_SIM))
    highs_l = loopify(highs, FS_AUDIO)

    import wave
    tmp = os.path.join(ASSETS, '_highs.wav')
    with wave.open(tmp, 'w') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(FS_AUDIO)
        h = highs_l / (np.max(np.abs(highs_l)) + 1e-9) * 0.9
        w.writeframes((h * 32767).astype(np.int16).tobytes())
    subprocess.run(['ffmpeg', '-y', '-v', 'quiet', '-i', tmp,
                    '-c:a', 'libvorbis', '-q:a', '4',
                    os.path.join(ASSETS, 'highs.ogg')], check=True)
    os.remove(tmp)

    highs_peak_scale = float(np.max(np.abs(highs_l)) + 1e-9) / 0.9
    return x12, float(np.std(low)), float(np.std(low_sim)), highs_peak_scale


def bake_field_maps():
    """Average the playback SPL frames into off/on background images."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = np.load(os.path.join(DEMO, 'playback.npz'))
    frames = data['frames'].astype(np.float32)
    fps = 30
    off = frames[int(5 * fps):int(11 * fps)].mean(axis=0)
    on_raw = frames[int(16 * fps):int(21 * fps)].mean(axis=0)

    # The two averaging windows see different traffic levels; the per-pixel
    # power ratio cancels that, so on = off * (what ANC changes)
    ratio = on_raw / np.maximum(off, 1e-12)
    ratio /= np.median(ratio)  # anchor the far field at unity
    on = off * ratio

    spl_off = 10 * np.log10(np.maximum(off, 1e-12))
    spl_on = 10 * np.log10(np.maximum(on, 1e-12))
    vmax = float(np.percentile(spl_off, 99.5))
    vmin = vmax - 20

    from make_walk_demo import build_scene
    mask, _ = build_scene()
    bmask = mask[::2, ::2] > 0.5
    # Paint the PML absorption strips as if they were walls
    pml = 10  # 20 cells / 2 (frames are 2x-decimated)
    bmask[:pml, :] = True
    bmask[-pml:, :] = True

    for name, spl in [('field_off', spl_off), ('field_on', spl_on)]:
        img = np.ma.masked_where(bmask, spl)
        cmap = plt.cm.magma.copy()
        cmap.set_bad('#21262d')
        fig = plt.figure(figsize=(spl.shape[0] / 30, spl.shape[1] / 30),
                         dpi=60)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img.T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                  interpolation='bilinear')
        ax.axis('off')
        fig.savefig(os.path.join(ASSETS, f'{name}.png'))
        plt.close()


def main():
    os.makedirs(ASSETS, exist_ok=True)

    x12, low_std, low_sim_std, highs_scale = bake_audio()
    pts, h_off, h_on = bake_irs()

    # Interleave per point: [h_off, h_on] so lookup is one contiguous read
    n_pts, L = h_off.shape
    irs = np.stack([h_off, h_on], axis=1)  # (n_pts, 2, L)
    irs.tofile(os.path.join(ASSETS, 'irs.bin'))
    x12.astype(np.float16).tofile(os.path.join(ASSETS, 'noise_low.bin'))

    bake_field_maps()

    # Calibration mirrors stage_audio: lows scaled so ANC-off matches the
    # recording's low band, +4 dB demo boost; highs distance-attenuated
    meta = {
        'fs': FS_SHIP,
        'ir_len': int(L),
        'grid': {'x': GX.tolist(), 'y': GY.tolist()},
        'domain': [DOMAIN_X, DOMAIN_Y],
        'street_halfwidth': WIDTH / 2,
        'noise_pos': list(NOISE_POS),
        'speakers': [list(p) for p in SPEAKERS],
        'mics': [list(p) for p in ERROR_MICS],
        'bench': list(BENCH),
        'low_calibration': low_std / low_sim_std,
        'low_boost': 1.6,
        'highs_scale': highs_scale,
        'walk_bounds': [2.6, 21.4, -6.2, 6.2],
        'field_extent': [0, DOMAIN_X, -DOMAIN_Y / 2, DOMAIN_Y / 2],
        'credit': 'Traffic audio: "Highway from Bridge, Center" by stephan, '
                  'public domain, Wikimedia Commons',
    }
    with open(os.path.join(ASSETS, 'meta.json'), 'w') as f:
        json.dump(meta, f)

    total = sum(os.path.getsize(os.path.join(ASSETS, f))
                for f in os.listdir(ASSETS))
    print(f'assets: {total/1e6:.1f} MB in {ASSETS}')

    # Sanity: bench-point reduction from the baked IRs
    from antiphon.simulation.metrics import octave_band_levels
    bi = min(range(len(pts)),
             key=lambda i: (pts[i][0] - BENCH[0]) ** 2
             + (pts[i][1] - BENCH[1]) ** 2)
    x = x12.astype(np.float64)
    p_off = fftconvolve(x, h_off[bi].astype(np.float64))[:len(x)]
    p_on = fftconvolve(x, h_on[bi].astype(np.float64))[:len(x)]
    for fc in (63.0, 125.0, 250.0):
        lo = octave_band_levels(p_off, FS_SHIP, centers=[fc])[fc]
        ln = octave_band_levels(p_on, FS_SHIP, centers=[fc])[fc]
        print(f'  baked bench check {fc:.0f} Hz: {lo - ln:+.1f} dB reduction')


if __name__ == '__main__':
    main()
