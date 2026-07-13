"""Build the README banner: physics -> control -> learned model.

Panel 1: FDTD pressure snapshot, a pulse mid-flight in a street canyon.
Panel 2: error-mic signal while FxLMS converges (noise dies out).
Panel 3: measured vs model-predicted transfer function on an unseen street
         (uses the trained v2 checkpoint).

Usage:
    uv run python scripts/make_banner.py [--ckpt data/runs/v2/best.pt]
"""

import argparse
import importlib.util
import os

import matplotlib.pyplot as plt
import numpy as np

from antiphon.anc import MultichannelFxLMS, simulate_anc
from antiphon.model.dataset import DX, FREQS, to_ix
from antiphon.model.inference import load_model, predict_H
from antiphon.simulation.fdtd import FDTDSolver

FIGDIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'figures')

BG = '#0d1117'
FG = '#e6edf3'
DIM = '#8b949e'
CYAN = '#3ddbd9'
ORANGE = '#ff9e64'


def style_axis(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color('#30363d')
    ax.tick_params(colors=DIM, labelsize=7)
    ax.xaxis.label.set_color(DIM)
    ax.yaxis.label.set_color(DIM)


def panel_wave(ax):
    """Pulse propagating in a canyon, reflections visible."""
    dx = 0.05
    nx, ny = int(36.0 / dx), int(20.0 / dx)
    mask = np.zeros((nx, ny))
    wall = int(3.0 / dx)
    mask[:, :wall] = 1.0
    mask[:, -wall:] = 1.0

    solver = FDTDSolver(mask, dx, pml_cells=25)
    sig = solver.gaussian_pulse(f_max=350.0)
    src = (int(10.0 / dx), ny // 2)
    solver.reset()
    n_steps = int(0.055 / solver.dt)  # pulse has bounced off both walls
    for k in range(n_steps):
        solver.step(src, sig[k] if k < len(sig) else 0.0)

    p = solver.p.copy()
    vmax = np.percentile(np.abs(p), 99.5)
    img = np.ma.masked_where(mask.T > 0.5, p.T)
    # 'berlin' is a dark-centered diverging map: silence stays dark
    cmap = plt.cm.berlin.copy()
    cmap.set_bad('#21262d')
    ax.imshow(img, origin='lower', cmap=cmap, vmin=-vmax, vmax=vmax,
              aspect='auto', extent=[0, 36, 0, 20], interpolation='bilinear')
    ax.plot(*[[10.0], [10.0]], marker='*', color='#ffd166', markersize=10,
            markeredgecolor='black', markeredgewidth=0.4)
    ax.axhline(3.0, color='#30363d', lw=1)
    ax.axhline(17.0, color='#30363d', lw=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Wave physics, first principles\nFDTD sound field in a street canyon',
                 fontsize=9, color=FG, pad=8)


def panel_control(ax):
    """Error-mic signal: noise dies out as FxLMS converges."""
    rng = np.random.default_rng(1)
    fs = 4000.0
    K = J = 2
    Ls = 32

    def plant(seed, cols):
        r = np.random.default_rng(seed)
        S = np.zeros((K, cols, Ls))
        for k in range(K):
            for j in range(cols):
                d = 3 + r.integers(0, 5)
                S[k, j, d:] = r.standard_normal(Ls - d) * \
                    np.exp(-np.arange(Ls - d) / 6.0)
        return S

    S = plant(1, J)
    P = plant(2, 1)[:, 0, :]
    T_off, T_on = 0.35, 0.85
    t = np.arange(int((T_off + T_on) * fs)) / fs
    x = np.sin(2 * np.pi * 150.0 * t)
    d = np.stack([np.convolve(x, P[k])[:len(x)] for k in range(K)])

    n_off = int(T_off * fs)
    ctl = MultichannelFxLMS(1, J, K, filter_len=64,
                            secondary_estimate=S, mu=0.004)
    e_on = simulate_anc(x[n_off:], d[:, n_off:], S, ctl)
    e = np.concatenate([d[0, :n_off], e_on[0]])

    from scipy.signal import hilbert
    env = np.abs(hilbert(e))
    env_d = np.abs(hilbert(d[0]))
    ax.fill_between(t, -env_d, env_d, color=DIM, alpha=0.18, lw=0)
    ax.plot(t, e, color=CYAN, lw=0.55)
    ax.plot(t, env, color=CYAN, lw=1.1, alpha=0.9)
    ax.plot(t, -env, color=CYAN, lw=1.1, alpha=0.9)

    ymax = 1.25 * np.max(env_d)
    ax.axvline(T_off, color=FG, lw=0.8, ls=':', alpha=0.7)
    ax.text(T_off + 0.02, 0.88 * ymax, 'ANC on', fontsize=8, color=FG)
    ax.text(0.02, 0.88 * ymax, 'noise', fontsize=8, color=DIM)
    ax.set_xlim(0, T_off + T_on)
    ax.set_ylim(-ymax, ymax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Adaptive cancellation\nerror microphone as the controller converges',
                 fontsize=9, color=FG, pad=8)


def panel_model(ax_mag, ax_ph, ckpt):
    """Measured vs predicted H on a held-out scene."""
    spec = importlib.util.spec_from_file_location(
        'ecl', os.path.join(os.path.dirname(__file__),
                            'evaluate_closed_loop.py'))
    ecl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ecl)
    from antiphon.model.dataset import downsample_occupancy
    from antiphon.simulation.geometry import C_SOUND

    params, mask, adm, speakers, mics = ecl.build_scene(100004)
    solver = FDTDSolver(mask, DX, admittance=adm, pml_cells=20)
    spk, mic = speakers[0], mics[0]
    H_meas = solver.transfer_function(
        to_ix(*spk), [to_ix(*mic)], FREQS,
        f_max=1.2 * FREQS[-1], duration=0.5)[0]

    model, freqs, h_scale = load_model(ckpt)
    occ = downsample_occupancy(mask)
    alpha = np.array(
        [params['alpha_left'], params['alpha_right']], dtype=np.float32)
    H_pred = predict_H(model, freqs, h_scale, occ, alpha,
                       np.array([spk]), np.array([mic]))[0]

    dist = np.hypot(mic[0] - spk[0], mic[1] - spk[1])
    k = 2 * np.pi * np.asarray(FREQS) / C_SOUND
    comp = np.exp(1j * dist * k)

    ax_mag.semilogx(FREQS, 20 * np.log10(np.abs(H_meas)), color=DIM,
                    lw=1.4, label='FDTD (measured)')
    ax_mag.semilogx(FREQS, 20 * np.log10(np.abs(H_pred)), color=ORANGE,
                    lw=1.2, ls='--', label='Model (predicted)')
    ax_mag.set_ylabel('|H| dB', fontsize=7)
    ax_mag.set_xticks([])
    ax_mag.set_xticks([], minor=True)
    ax_mag.legend(loc='lower left', fontsize=6.5, framealpha=0,
                  labelcolor=FG)
    ax_mag.set_title('Learned acoustics\nspeaker-to-mic transfer function, unseen street',
                     fontsize=9, color=FG, pad=8)

    ax_ph.semilogx(FREQS, np.angle(H_meas * comp), color=DIM, lw=1.4)
    ax_ph.semilogx(FREQS, np.angle(H_pred * comp), color=ORANGE,
                   lw=1.2, ls='--')
    ax_ph.set_ylabel('phase', fontsize=7)
    ax_ph.set_xlabel('frequency (Hz)', fontsize=7)
    ax_ph.set_ylim(-np.pi, np.pi)
    ax_ph.set_yticks([])
    ax_ph.set_xticks([50, 100, 200, 400])
    ax_ph.set_xticks([], minor=True)
    ax_ph.set_xticklabels(['50', '100', '200', '400'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='data/runs/v2/best.pt')
    args = parser.parse_args()

    fig = plt.figure(figsize=(15, 4.2), dpi=150, facecolor=BG)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1, 1],
                          height_ratios=[2, 1], hspace=0.12, wspace=0.14,
                          left=0.03, right=0.985, top=0.78, bottom=0.08)

    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[:, 1])
    ax3a = fig.add_subplot(gs[0, 2])
    ax3b = fig.add_subplot(gs[1, 2])
    for ax in (ax1, ax2, ax3a, ax3b):
        style_axis(ax)

    panel_wave(ax1)
    panel_control(ax2)
    panel_model(ax3a, ax3b, args.ckpt)

    fig.text(0.03, 0.955, 'ANTIPHON', fontsize=17, fontweight='bold',
             color=FG, family='monospace')
    fig.text(0.135, 0.958, 'urban noise cancellation via a learned acoustics model',
             fontsize=9.5, color=DIM, style='italic')

    path = os.path.join(FIGDIR, 'banner.png')
    fig.savefig(path, facecolor=BG, dpi=150)
    plt.close()
    print(f'Saved {path}')


if __name__ == '__main__':
    main()
