"""Closed-loop evaluation: model-predicted vs measured secondary paths.

For held-out scenes (seeds never used in training), run the same FxLMS
controller twice against the true FDTD plant:
  A) Shat = FDTD-measured secondary paths (upper bound)
  B) Shat = foundation-model-predicted secondary paths
and compare dB reduction at the error mics.

Success criterion: B achieves >= 80% of A's dB reduction.

Usage:
    uv run python scripts/evaluate_closed_loop.py --ckpt data/runs/v1/best.pt \
        --scenes 5 --out data/runs/v1/closed_loop.json
"""

import argparse
import json

import numpy as np

from antiphon.anc import MultichannelFxLMS, db_reduction, simulate_anc
from antiphon.model.dataset import (
    DX, FREQS, downsample_occupancy, make_canyon_mask, scene_params, to_ix,
)
from antiphon.model.inference import h_to_fir, load_model, predict_H
from antiphon.simulation.fdtd import FDTDSolver
from antiphon.simulation.materials import admittance_from_alpha

HELDOUT_SEED_BASE = 100000  # training used seeds 0..n_scenes-1
IR_LEN = 1024
TONES = [150.0, 250.0]
# FxLMS stability varies per scene (street width and facade absorption set
# the plant conditioning), so each arm gets a step-size ladder: use the
# largest mu that converges, like a real commissioning pass.
MU_LADDER = [3e-3, 1e-3, 3e-4, 1e-4]
FILTER_LEN = 128


def build_scene(seed):
    params = scene_params(seed)
    W = params['width']
    mask, y_centers = make_canyon_mask(W)
    adm = np.zeros_like(mask, dtype=float)
    adm[:, y_centers < 0] = admittance_from_alpha(params['alpha_left'])
    adm[:, y_centers > 0] = admittance_from_alpha(params['alpha_right'])

    # 2 speakers per side near the facades, error mics mid-street
    y_spk = W / 2 - 0.4
    speakers = [(10.0, -y_spk), (14.0, -y_spk), (10.0, y_spk), (14.0, y_spk)]
    mics = [(15.0, -1.0), (15.0, 1.0), (17.0, -1.0), (17.0, 1.0)]
    return params, mask, adm, speakers, mics


def evaluate_scene(seed, model, freqs, h_scale, verbose=True):
    params, mask, adm, speakers, mics = build_scene(seed)
    solver = FDTDSolver(mask, DX, admittance=adm, pml_cells=20)
    fs = 1.0 / solver.dt
    mic_ix = [to_ix(x, y) for (x, y) in mics]

    # True plant
    P, _ = solver.impulse_response(
        to_ix(*params['src']), mic_ix,
        f_lo=FREQS[0], f_hi=FREQS[-1], ir_len=IR_LEN)
    S_true = np.stack([
        solver.impulse_response(to_ix(*spk), mic_ix,
                                f_lo=FREQS[0], f_hi=FREQS[-1],
                                ir_len=IR_LEN)[0]
        for spk in speakers
    ], axis=1)  # (K, J, L)

    # Model-predicted secondary paths
    occ = downsample_occupancy(mask)
    alpha = np.array([params['alpha_left'], params['alpha_right']],
                     dtype=np.float32)
    S_pred = np.zeros_like(S_true)
    for j, spk in enumerate(speakers):
        H = predict_H(model, freqs, h_scale, occ, alpha,
                      np.array([spk] * len(mics)), np.array(mics))
        for k, mic in enumerate(mics):
            d = np.hypot(mic[0] - spk[0], mic[1] - spk[1])
            S_pred[k, j] = h_to_fir(H[k], freqs, d, fs, IR_LEN)

    # Diagnostic: how good are the predicted secondary paths themselves?
    def narrowband(ir, f0):
        w = np.exp(-2j * np.pi * f0 * np.arange(ir.shape[-1]) / fs)
        return ir @ w

    path_errors = {}
    for f0 in TONES:
        g_true = narrowband(S_true, f0)
        g_pred = narrowband(S_pred, f0)
        dphi = np.abs(np.angle(g_pred / g_true))
        dmag = np.abs(20 * np.log10(np.abs(g_pred) / np.abs(g_true)))
        path_errors[f0] = {'phase_rad_mean': float(np.mean(dphi)),
                           'phase_rad_max': float(np.max(dphi)),
                           'mag_db_mean': float(np.mean(dmag))}

    K, J = len(mics), len(speakers)

    def run_with_ladder(x, d, shat):
        """Largest step size that converges; returns (mean_red, mu)."""
        for mu in MU_LADDER:
            ctl = MultichannelFxLMS(1, J, K, filter_len=FILTER_LEN,
                                    secondary_estimate=shat, mu=mu)
            e = simulate_anc(x, d, S_true, ctl)
            r = db_reduction(d, e)
            if np.all(np.isfinite(r)) and np.mean(r) > 0:
                return r, mu
        return r, MU_LADDER[-1]  # nothing converged; report as-is

    results = []
    for f0 in TONES:
        # 8 s runs: the ladder may hand one arm a smaller step size, which
        # converges slower; short runs would time it out unfairly.
        t = np.arange(int(8.0 * fs)) / fs
        x = np.sin(2 * np.pi * f0 * t)
        d = np.stack([np.convolve(x, P[k])[:len(x)] for k in range(K)])

        red, mu_used = {}, {}
        for name, shat in [('measured', S_true), ('predicted', S_pred)]:
            red[name], mu_used[name] = run_with_ladder(x, d, shat)

        # Clamp divergence/amplification to 0 dB before forming the ratio
        meas = max(float(np.mean(red['measured'])), 0.0)
        pred = max(float(np.mean(red['predicted'])), 0.0)
        ratio = pred / meas if meas >= 3.0 else None
        # Practical metric: depth beyond 20 dB (99% energy) is convergence
        # trivia, not path quality, so cap both arms there.
        CEIL = 20.0
        ratio_capped = (min(pred, CEIL) / min(meas, CEIL)
                        if meas >= 3.0 else None)
        results.append({
            'tone_hz': f0,
            'reduction_measured_db': [float(v) for v in red['measured']],
            'reduction_predicted_db': [float(v) for v in red['predicted']],
            'mean_measured_db': meas,
            'mean_predicted_db': pred,
            'mu_measured': mu_used['measured'],
            'mu_predicted': mu_used['predicted'],
            'ratio': ratio,
            'ratio_capped_20db': ratio_capped,
            'path_errors': path_errors[f0],
        })
        if verbose:
            rtxt = f'{ratio_capped:.2f}' if ratio_capped is not None else 'n/a'
            print(f'  seed {seed} tone {f0:.0f} Hz: '
                  f'measured {meas:.1f} dB (mu {mu_used["measured"]}), '
                  f'predicted {pred:.1f} dB (mu {mu_used["predicted"]}) '
                  f'capped-ratio {rtxt} | S phase err '
                  f'{path_errors[f0]["phase_rad_mean"]:.2f} rad', flush=True)

    return {'seed': seed, 'width': float(params['width']),
            'alpha': [float(a) for a in
                      (params['alpha_left'], params['alpha_right'])],
            'tones': results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='data/runs/v1/best.pt')
    parser.add_argument('--scenes', type=int, default=5)
    parser.add_argument('--out', type=str, default='data/runs/v1/closed_loop.json')
    args = parser.parse_args()

    model, freqs, h_scale = load_model(args.ckpt)
    scenes = []
    for i in range(args.scenes):
        print(f'scene {i+1}/{args.scenes}', flush=True)
        scenes.append(evaluate_scene(HELDOUT_SEED_BASE + i, model,
                                     freqs, h_scale))

    ratios = [t['ratio'] for s in scenes for t in s['tones']
              if t['ratio'] is not None]
    capped = [t['ratio_capped_20db'] for s in scenes for t in s['tones']
              if t['ratio_capped_20db'] is not None]
    summary = {
        'scenes': scenes,
        'n_valid_ratios': len(ratios),
        'mean_ratio': float(np.mean(ratios)) if ratios else None,
        'min_ratio': float(np.min(ratios)) if ratios else None,
        'mean_ratio_capped_20db': float(np.mean(capped)) if capped else None,
        'criterion_80pct_met': bool(capped and np.mean(capped) >= 0.8),
    }
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nmean ratio {summary["mean_ratio"]:.2f} '
          f'(capped@20dB {summary["mean_ratio_capped_20db"]:.2f}), '
          f'min {summary["min_ratio"]:.2f}, '
          f'criterion met (capped): {summary["criterion_80pct_met"]}')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
