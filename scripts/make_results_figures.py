"""Generate the results figures for docs/results.md.

1. fdtd_validation.png — FDTD vs exact 2D analytical solution (open field
   amplitude decay + single-wall interference).
2. closed_loop.png — per-scene dB reduction, measured vs model-predicted
   secondary paths (from a closed_loop.json produced by
   evaluate_closed_loop.py).

Usage:
    uv run python scripts/make_results_figures.py \
        --closed-loop data/runs/v1/closed_loop.json
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import hankel1

from antiphon.simulation.fdtd import FDTDSolver
from antiphon.simulation.geometry import C_SOUND

FIGDIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'figures')


def fdtd_validation_figure():
    dx = 0.05
    n = int(20.0 / dx)
    freqs = [100.0, 200.0, 300.0, 400.0]

    # Open field: amplitude vs distance
    mask = np.zeros((n, n))
    solver = FDTDSolver(mask, dx, pml_cells=30)
    src = (n // 2, n // 2)
    radii = np.arange(1.5, 7.5, 0.5)
    rcv = [(src[0] + int(r / dx), src[1]) for r in radii]
    H = solver.transfer_function(src, rcv, freqs, duration=0.25)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=120)
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(freqs)))

    for i, (f, c) in enumerate(zip(freqs, colors)):
        k = 2 * np.pi * f / C_SOUND
        meas = 20 * np.log10(np.abs(H[:, i] / H[0, i]))
        exact = 20 * np.log10(np.abs(hankel1(0, k * radii)
                                     / hankel1(0, k * radii[0])))
        axes[0].plot(radii, exact, '-', color=c, alpha=0.5)
        axes[0].plot(radii, meas, 'o', color=c, markersize=5,
                     label=f'{f:.0f} Hz')
    axes[0].set_xlabel('Distance (m)')
    axes[0].set_ylabel('Relative level (dB)')
    axes[0].set_title('Open field: FDTD (dots) vs exact Hankel (lines)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Single wall: interference pattern along a line
    mask = np.zeros((n, n))
    wall_col = int(4.0 / dx)
    mask[:, :wall_col] = 1.0
    solver = FDTDSolver(mask, dx, pml_cells=30)
    src = (n // 2, int(6.0 / dx))
    ys = np.arange(4.6, 14.0, 0.2)
    rcv = [(int(13.0 / dx), int(y / dx)) for y in ys]
    f0 = 200.0
    H = solver.transfer_function(src, rcv, [f0], duration=0.25)[:, 0]

    def center(idx):
        return (idx + 0.5) * dx
    sx, sy = center(src[0]), center(src[1])
    wall_y = wall_col * dx
    k = 2 * np.pi * f0 / C_SOUND
    exact = []
    for (ix, iy) in rcv:
        px, py = center(ix), center(iy)
        r_d = np.hypot(px - sx, py - sy)
        r_i = np.hypot(px - sx, py - (2 * wall_y - sy))
        exact.append(hankel1(0, k * r_d) + hankel1(0, k * r_i))
    exact = np.array(exact)

    ref = len(ys) // 2
    axes[1].plot(ys, 20 * np.log10(np.abs(exact / exact[ref])), '-',
                 color='#444', alpha=0.6, label='Exact (image source)')
    axes[1].plot(ys, 20 * np.log10(np.abs(H / H[ref])), 'o',
                 color='#c0392b', markersize=4, label='FDTD')
    axes[1].axvline(wall_y, color='k', linewidth=3, alpha=0.7)
    axes[1].text(wall_y + 0.1, axes[1].get_ylim()[0] + 1, 'rigid wall',
                 fontsize=9, alpha=0.7)
    axes[1].set_xlabel('y position (m)')
    axes[1].set_ylabel('Relative level (dB)')
    axes[1].set_title(f'Single wall at {f0:.0f} Hz: interference pattern')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle('FDTD solver validation against exact 2D solutions',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(FIGDIR, 'fdtd_validation.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'Saved {path}')


def closed_loop_figure(json_path):
    with open(json_path) as f:
        data = json.load(f)

    scenes = data['scenes']
    tones = sorted({t['tone_hz'] for s in scenes for t in s['tones']})
    fig, axes = plt.subplots(1, len(tones), figsize=(6.5 * len(tones), 5),
                             dpi=120, sharey=True)
    if len(tones) == 1:
        axes = [axes]

    for ax, f0 in zip(axes, tones):
        meas, pred, labels = [], [], []
        for s in scenes:
            for t in s['tones']:
                if t['tone_hz'] == f0:
                    meas.append(t['mean_measured_db'])
                    pred.append(t['mean_predicted_db'])
                    labels.append(f"{s['seed'] % 1000}\nW={s['width']:.0f}m")
        xs = np.arange(len(meas))
        ax.bar(xs - 0.18, meas, 0.36, label='Measured paths', color='#2c6e91')
        ax.bar(xs + 0.18, pred, 0.36, label='Model-predicted paths',
               color='#c0392b')
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f'{f0:.0f} Hz tone')
        ax.set_xlabel('Held-out scene')
        ax.grid(axis='y', alpha=0.3)
    axes[0].set_ylabel('dB reduction at error mics')
    axes[0].legend()

    mr = data.get('mean_ratio')
    plt.suptitle('Closed loop: FxLMS with measured vs model-predicted '
                 f'secondary paths (mean ratio {mr:.2f})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(FIGDIR, 'closed_loop.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'Saved {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--closed-loop', type=str, default=None,
                        help='closed_loop.json path (skip figure if omitted)')
    parser.add_argument('--skip-validation', action='store_true')
    args = parser.parse_args()

    os.makedirs(FIGDIR, exist_ok=True)
    if not args.skip_validation:
        fdtd_validation_figure()
    if args.closed_loop:
        closed_loop_figure(args.closed_loop)
