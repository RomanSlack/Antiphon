"""Build the "walk down an Antiphon street" video demo.

Real public-domain traffic audio (Wikimedia Commons, 'Highway from Bridge,
Center') is band-split; the sub-400 Hz rumble is injected as a noise source
in an FDTD street canyon. A multichannel FxLMS controller cancels it at an
array of error mics. A listener walks through the scene; their audio is the
simulated pressure at their moving position (plus the untouched highs),
and the video shows the SPL field with ANC toggling on mid-walk.

Stages (cached in data/demo/):
    --stage paths     measure IRs, run broadband FxLMS, report reduction
    --stage playback  full FDTD playback run, listener audio, A/B wav
    --stage video     render frames + mux with ffmpeg
"""

import argparse
import os

import numpy as np
from scipy.signal import resample_poly, sosfiltfilt, butter

from antiphon.anc import MultichannelFxLMS, simulate_anc
from antiphon.simulation.fdtd import FDTDSolver
from antiphon.simulation.materials import admittance_from_alpha
from antiphon.simulation.metrics import octave_band_levels

DEMO = os.path.join(os.path.dirname(__file__), '..', 'data', 'demo')

# --- scene ---------------------------------------------------------------
DX = 0.08
DOMAIN_X, DOMAIN_Y = 24.0, 26.0
WIDTH = 14.0
ALPHA = (0.25, 0.30)   # acoustically treated installation corridor
NX, NY = int(DOMAIN_X / DX), int(DOMAIN_Y / DX)

NOISE_POS = (5.0, 2.5)            # vehicle lane, upstream
BENCH = (16.0, -3.5)              # the protected listening point
SPEAKERS = [(x, s * (WIDTH / 2 - 0.4))
            for x in (12.0, 15.0, 18.0, 21.0) for s in (-1, 1)]
# Ring of error mics enclosing the bench: interior cancellation holds when
# mic spacing is below half a wavelength (headrest-ANC style). Sparse mics
# along a walking path do NOT work: nulls at the mics, spill in between.
# Ring radius 0.30 m keeps the first interior cavity resonance (2.405*c /
# (2*pi*R) = 437 Hz) above the control band; at 0.45 m it sat at 292 Hz,
# inside the 250 Hz octave, and the interior blew up there.
ERROR_MICS = [(BENCH[0] + r * np.cos(a), BENCH[1] + r * np.sin(a))
              for r in (0.30,)
              for a in np.arange(8) * (2 * np.pi / 8)]

# --- timing --------------------------------------------------------------
FS_AUDIO = 48000
DECIM = 7                          # sim rate = 48000/7 = 6857.14 Hz
FS_SIM = FS_AUDIO / DECIM
DT = 1.0 / FS_SIM
COURANT = DT * 343.0 * np.sqrt(2.0) / DX   # 0.885, inside stability
DUR = 28.0
T_ANC_ON = 12.0
SEG_START = 96.0                   # segment of the source recording (s)
F_LO, F_HI = 25.0, 400.0
IR_LEN = 4096                      # 0.6 s: covers the reverb tail
# The controller runs at a decimated rate (like a real ANC DSP): the band
# is < 480 Hz, so fs_sim/6 = 1143 Hz suffices and filters get 36x cheaper
CTL_DECIM = 6
FS_CTL = FS_SIM / CTL_DECIM

# Walk waypoints: (time s, x, y) — stroll in, sit at the bench through the
# ANC toggle, walk away at the end (noise returns)
WAYPOINTS = [(0.0, 4.0, 1.0), (10.5, BENCH[0], BENCH[1]),
             (22.0, BENCH[0], BENCH[1]), (28.0, 22.5, 0.5)]


def build_scene():
    mask = np.zeros((NX, NY), dtype=np.float32)
    y = (np.arange(NY) + 0.5) * DX - DOMAIN_Y / 2
    mask[:, np.abs(y) >= WIDTH / 2] = 1.0
    adm = np.zeros((NX, NY))
    adm[:, y < 0] = admittance_from_alpha(ALPHA[0])
    adm[:, y > 0] = admittance_from_alpha(ALPHA[1])
    return mask, adm


def to_ix(x, y):
    return int(x / DX), int((y + DOMAIN_Y / 2) / DX)


def solver():
    mask, adm = build_scene()
    return FDTDSolver(mask, DX, admittance=adm, courant=COURANT,
                      pml_cells=20)


def load_bands():
    """Load the recording segment, split into sim band and pass-through."""
    import wave
    w = wave.open(os.path.join(DEMO, 'highway.wav'))
    fs = w.getframerate()
    assert fs == FS_AUDIO
    x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16) / 32768.0
    seg = x[int(SEG_START * fs):int((SEG_START + DUR) * fs)]

    sos_lo = butter(6, F_HI, 'lowpass', fs=fs, output='sos')
    sos_hp = butter(2, F_LO, 'highpass', fs=fs, output='sos')
    low = sosfiltfilt(sos_lo, seg)
    low = sosfiltfilt(sos_hp, low)
    highs = seg - low
    low_sim = resample_poly(low, up=1, down=DECIM)
    low_sim = low_sim / (np.std(low_sim) + 1e-12)
    return seg, low, highs, low_sim


def stage_paths():
    """Measure IRs and run broadband FxLMS; report octave-band reduction."""
    s = solver()
    mic_ix = [to_ix(*m) for m in ERROR_MICS]
    print('measuring primary paths...', flush=True)
    P, _ = s.impulse_response(to_ix(*NOISE_POS), mic_ix, f_lo=F_LO,
                              f_hi=F_HI, ir_len=IR_LEN, duration=1.0)
    S = []
    for j, spk in enumerate(SPEAKERS):
        print(f'measuring secondary paths {j+1}/{len(SPEAKERS)}...', flush=True)
        S.append(s.impulse_response(to_ix(*spk), mic_ix, f_lo=F_LO,
                                    f_hi=F_HI, ir_len=IR_LEN,
                                    duration=1.0)[0])
    S = np.stack(S, axis=1)  # (K, J, L)
    tail = float(np.sum(S[..., IR_LEN // 4:] ** 2) / np.sum(S ** 2))
    print(f'IR energy beyond 0.15s: {100 * tail:.1f}%')

    np.savez_compressed(os.path.join(DEMO, 'irs.npz'), P=P, S=S)
    print('saved irs.npz')
    stage_wiener()


def band_reduction(d, e, n0, fs=FS_SIM):
    out = {}
    for fc in (63.0, 125.0, 250.0):
        r = []
        for k in range(d.shape[0]):
            ld = octave_band_levels(d[k, n0:], fs, centers=[fc])[fc]
            le = octave_band_levels(e[k, n0:], fs, centers=[fc])[fc]
            r.append(ld - le)
        out[fc] = r
    return out


def stage_wiener():
    """Compute the causal Wiener-optimal MIMO FIR controller directly.

    LMS adaptation crawls on heavily colored references (traffic noise), so
    for the demo we solve the least-squares problem FxLMS converges toward:
    minimize sum_k ||d_k + sum_j S_kj * (w_j * x)||^2 + lam ||w||^2
    over causal FIR weights w (J filters, LW taps), at the control rate.
    """
    from scipy.signal import fftconvolve
    from scipy.linalg import toeplitz, cho_factor, cho_solve

    irs = np.load(os.path.join(DEMO, 'irs.npz'))
    D = CTL_DECIM
    P = D * resample_poly(irs['P'], up=1, down=D, axis=-1)
    S = D * resample_poly(irs['S'], up=1, down=D, axis=-1)
    _, _, _, low_sim = load_bands()
    x = resample_poly(low_sim, up=1, down=D)
    K, J = len(ERROR_MICS), len(SPEAKERS)
    N = len(x)
    d = np.stack([np.convolve(x, P[k])[:N] for k in range(K)])

    LW = 512
    print('building filtered references...', flush=True)
    U = np.stack([[fftconvolve(x, S[k, j])[:N] for j in range(J)]
                  for k in range(K)])  # (K, J, N)

    # Frequency-weight the objective toward the rumble band, but keep a
    # floor above it: with ~zero weight up there the controller happily
    # pumps 250+ Hz energy as a side effect of canceling the lows
    sos_w = butter(4, 215.0, 'lowpass', fs=FS_CTL, output='sos')

    def weight(sig):
        return 0.25 * sig + 0.75 * sosfiltfilt(sos_w, sig, axis=-1)

    d_w = weight(d)
    U_w = weight(U)

    print('assembling normal equations...', flush=True)
    # Cross-correlations via FFT, summed over mics
    nfft = 1 << int(np.ceil(np.log2(2 * N)))
    Uf = np.fft.rfft(U_w, nfft)  # (K, J, F)
    R = np.empty((J * LW, J * LW))
    for j in range(J):
        for j2 in range(J):
            c = np.fft.irfft((Uf[:, j] * np.conj(Uf[:, j2])).sum(axis=0),
                             nfft)
            # c[tau] = sum_k sum_n u_kj(n+tau) u_kj2(n)
            col = c[:LW]                    # tau = 0..LW-1
            row = np.concatenate([[c[0]], c[-1:-LW:-1]])  # tau = 0..-(LW-1)
            R[j * LW:(j + 1) * LW, j2 * LW:(j2 + 1) * LW] = \
                toeplitz(row, col)
    Df = np.fft.rfft(d_w, nfft)
    b = np.empty(J * LW)
    for j in range(J):
        c = np.fft.irfft((np.conj(Uf[:, j]) * Df).sum(axis=0), nfft)
        b[j * LW:(j + 1) * LW] = -c[:LW]

    lam = 1e-3 * np.trace(R) / (J * LW)
    R[np.diag_indices_from(R)] += lam
    print('solving...', flush=True)
    w = cho_solve(cho_factor(R), b).reshape(J, LW)

    y_ctl = np.stack([np.convolve(w[j], x)[:N] for j in range(J)])
    e = d + np.stack([
        sum(fftconvolve(y_ctl[j], S[k, j])[:N] for j in range(J))
        for k in range(K)])

    n0 = N // 2
    print('\noctave-band reduction at error mics (Wiener, final half):')
    for fc, r in band_reduction(d, e, n0, FS_CTL).items():
        print(f'  {fc:5.0f} Hz: mean {np.mean(r):5.1f} dB  '
              f'(per mic: {[round(float(v),1) for v in r]})')
    print(f'speaker RMS / reference RMS: '
          f'{np.std(y_ctl) / np.std(x):.2f}')

    y_hist = resample_poly(y_ctl, up=D, down=1, axis=-1)[:, :len(low_sim)]
    if y_hist.shape[1] < len(low_sim):
        y_hist = np.pad(y_hist,
                        ((0, 0), (0, len(low_sim) - y_hist.shape[1])))
    np.savez_compressed(os.path.join(DEMO, 'paths.npz'),
                        y_hist=y_hist, e=e, d=d, w=w)
    print('saved paths.npz')


def stage_fxlms():
    """Run broadband FxLMS on cached IRs, probing a step-size ladder.

    Runs at the decimated control rate so the filters can span the full
    reverb tail (truncated plant models made the controller confidently
    wrong: great cancellation in its own FIR world, amplification in the
    real FDTD playback).
    """
    irs = np.load(os.path.join(DEMO, 'irs.npz'))
    D = CTL_DECIM
    # Decimating an in-band FIR scales its transfer function by 1/D
    P = D * resample_poly(irs['P'], up=1, down=D, axis=-1)
    S = D * resample_poly(irs['S'], up=1, down=D, axis=-1)
    _, _, _, low_sim = load_bands()
    low_ctl = resample_poly(low_sim, up=1, down=D)
    K, J = len(ERROR_MICS), len(SPEAKERS)
    d = np.stack([np.convolve(low_ctl, P[k])[:len(low_ctl)]
                  for k in range(K)])

    FILTER_LEN = 700
    FS_RUN = FS_CTL
    low_run = low_ctl
    # Leakage suppresses the slow weight-drift instability at the
    # ill-conditioned low band edge (without it, 63 Hz blows up over ~100 s)
    LEAK = 2e-4

    # Probe on the first 8 s, keep the largest converging mu
    n_probe = int(8.0 * FS_RUN)
    best_mu = None
    for mu in (3e-3, 1e-3, 3e-4, 1e-4):
        ctl = MultichannelFxLMS(1, J, K, filter_len=FILTER_LEN,
                                secondary_estimate=S, mu=mu, leak=LEAK)
        e, _ = simulate_anc_capture(low_run[:n_probe], d[:, :n_probe], S, ctl)
        tail = e[:, -int(2 * FS_RUN):]
        ref = d[:, n_probe - int(2 * FS_RUN):n_probe]
        ok = np.all(np.isfinite(tail)) and \
            np.sqrt(np.mean(tail ** 2)) < np.sqrt(np.mean(ref ** 2))
        print(f'  probe mu={mu}: {"converges" if ok else "diverges"}',
              flush=True)
        if ok:
            best_mu = mu
            break
    if best_mu is None:
        raise SystemExit('no stable step size found')

    # Pre-train: loop the segment so the controller fully converges,
    # then a final capture pass (the demo shows the converged system)
    print(f'pre-training FxLMS (mu={best_mu}, {FILTER_LEN} taps '
          f'at {FS_RUN:.0f} Hz)...', flush=True)
    ctl = MultichannelFxLMS(1, J, K, filter_len=FILTER_LEN,
                            secondary_estimate=S, mu=best_mu, leak=LEAK)
    for it in range(4):
        e, _ = simulate_anc_capture(low_run, d, S, ctl)
        n0 = len(low_run) // 2
        red = band_reduction(d, e, n0, FS_RUN)
        print(f'  pass {it+1}: ' + '  '.join(
            f'{fc:.0f}Hz {np.mean(r):+.1f}dB' for fc, r in red.items()),
            flush=True)

    print('capture pass...', flush=True)
    e, y_ctl = simulate_anc_capture(low_run, d, S, ctl)

    n0 = len(low_run) // 2
    print('\noctave-band reduction at error mics (final half):')
    for fc, r in band_reduction(d, e, n0, FS_RUN).items():
        print(f'  {fc:5.0f} Hz: mean {np.mean(r):5.1f} dB  '
              f'(per mic: {[round(float(v),1) for v in r]})')

    # Upsample speaker signals back to the simulation rate for playback
    y_hist = resample_poly(y_ctl, up=D, down=1, axis=-1)[:, :len(low_sim)]
    if y_hist.shape[1] < len(low_sim):
        pad = len(low_sim) - y_hist.shape[1]
        y_hist = np.pad(y_hist, ((0, 0), (0, pad)))

    np.savez_compressed(os.path.join(DEMO, 'paths.npz'),
                        y_hist=y_hist, e=e, d=d, mu=best_mu)
    print('saved paths.npz')


def simulate_anc_capture(reference, primary_d, secondary_ir, controller):
    """simulate_anc, but also capture the speaker output history."""
    reference = np.atleast_2d(reference)
    K, J, Ls = secondary_ir.shape
    N = reference.shape[1]
    ybuf = np.zeros((J, Ls))
    e = np.zeros((K, N))
    y_hist = np.zeros((J, N))
    for n in range(N):
        y = controller.compute_output(reference[:, n])
        y_hist[:, n] = y
        ybuf[:, 1:] = ybuf[:, :-1]
        ybuf[:, 0] = y
        e_n = primary_d[:, n] + np.einsum('kjl,jl->k', secondary_ir, ybuf)
        e[:, n] = e_n
        controller.adapt(e_n)
    return e, y_hist


def walker_pos(t):
    for (t0, x0, y0), (t1, x1, y1) in zip(WAYPOINTS[:-1], WAYPOINTS[1:]):
        if t <= t1:
            f = np.clip((t - t0) / max(t1 - t0, 1e-9), 0, 1)
            return x0 + f * (x1 - x0), y0 + f * (y1 - y0)
    return WAYPOINTS[-1][1], WAYPOINTS[-1][2]


def stage_playback():
    """Full FDTD playback with gated anti-noise; record field + listener."""
    data = np.load(os.path.join(DEMO, 'paths.npz'))
    y_hist = data['y_hist']
    _, low, highs, low_sim = load_bands()

    n_steps = len(low_sim)
    t = np.arange(n_steps) * DT
    gate = np.clip((t - (T_ANC_ON - 0.25)) / 0.5, 0, 1)
    gate = 0.5 - 0.5 * np.cos(np.pi * gate)
    y_gated = y_hist * gate[None, :]

    s = solver()
    s.reset()
    src_ix = to_ix(*NOISE_POS)
    spk_ix = [to_ix(*p) for p in SPEAKERS]

    fps = 30
    frame_steps = np.round(np.arange(1, int(DUR * fps) + 1)
                           * FS_SIM / fps).astype(int)
    ema = np.zeros((NX, NY), dtype=np.float32)
    frames = np.zeros((len(frame_steps),) + ema[::2, ::2].shape,
                      dtype=np.float16)
    a_ema = 4 * DT / 0.08

    earL = np.zeros(n_steps)
    earR = np.zeros(n_steps)
    mic_ix = [to_ix(*m) for m in ERROR_MICS]
    mx = np.array([m[0] for m in mic_ix])
    my = np.array([m[1] for m in mic_ix])
    mic_tr = np.zeros((len(mic_ix), n_steps))
    fi = 0
    for n in range(n_steps):
        s.step()
        s.px[src_ix] += 0.5 * low_sim[n]
        s.py[src_ix] += 0.5 * low_sim[n]
        for j, (i, jj) in enumerate(spk_ix):
            s.px[i, jj] += 0.5 * y_gated[j, n]
            s.py[i, jj] += 0.5 * y_gated[j, n]

        wx, wy = walker_pos(t[n])
        iL = to_ix(wx, wy - 0.12)
        iR = to_ix(wx, wy + 0.12)
        p = s.px + s.py
        earL[n] = p[iL]
        earR[n] = p[iR]
        mic_tr[:, n] = p[mx, my]

        if n % 4 == 0:
            ema += a_ema * (p.astype(np.float32) ** 2 - ema)
        if fi < len(frame_steps) and n == frame_steps[fi] - 1:
            frames[fi] = ema[::2, ::2].astype(np.float16)
            fi += 1
        if n % 20000 == 0:
            print(f'  playback {n}/{n_steps}', flush=True)

    np.savez_compressed(os.path.join(DEMO, 'playback.npz'),
                        frames=frames, earL=earL, earR=earR, gate=gate)

    # Diagnostic: does the REAL plant show the reduction the FxLMS
    # simulation claimed at the error mics?
    n_b = (int(8.0 * FS_SIM), int(11.5 * FS_SIM))
    n_a = (int(13.0 * FS_SIM), int(19.0 * FS_SIM))
    print('error mics in playback (before -> after ANC):')
    for fc in (63.0, 125.0, 250.0):
        pre = np.mean([octave_band_levels(mic_tr[k, n_b[0]:n_b[1]],
                                          FS_SIM, centers=[fc])[fc]
                       for k in range(len(mic_ix))])
        post = np.mean([octave_band_levels(mic_tr[k, n_a[0]:n_a[1]],
                                           FS_SIM, centers=[fc])[fc]
                        for k in range(len(mic_ix))])
        print(f'  {fc:5.0f} Hz: {pre:6.1f} -> {post:6.1f} dB '
              f'({post - pre:+.1f})')

    stage_audio()


def stage_audio():
    """Mix the listener audio from cached ear signals (no physics re-run).

    The pass-through highs get the same cylindrical distance attenuation
    the simulation gives the lows (otherwise constant tire hiss masks the
    rumble drop), and the lows get a +4 dB demo emphasis (disclosed).
    """
    data = np.load(os.path.join(DEMO, 'playback.npz'))
    earL, earR = data['earL'], data['earR']
    _, low, highs, _ = load_bands()

    def up(x):
        out = resample_poly(x, up=DECIM, down=1)
        return out[:len(low)]

    lowL, lowR = up(earL), up(earR)
    # Calibrate so the ANC-off portion matches the real recording's low band
    n_cal = int(8.0 * FS_AUDIO)
    scale = np.std(low[:n_cal]) / (np.std(lowL[:n_cal]) + 1e-12)
    LOW_BOOST = 1.6
    lowL, lowR = lowL * scale * LOW_BOOST, lowR * scale * LOW_BOOST

    # Distance-attenuate the highs along the walk (cylindrical spreading)
    t_a = np.arange(len(low)) / FS_AUDIO
    r = np.array([np.hypot(*(np.array(walker_pos(t)) - NOISE_POS))
                  for t in t_a[::480]])
    r = np.repeat(r, 480)[:len(low)]
    g = np.sqrt(r[0] / np.maximum(r, 0.5))
    highs_a = highs * g

    stereo = np.stack([lowL + highs_a, lowR + highs_a], axis=1)
    stereo /= np.max(np.abs(stereo)) / 0.85

    import wave
    with wave.open(os.path.join(DEMO, 'demo_audio.wav'), 'w') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(FS_AUDIO)
        w.writeframes((stereo * 32767).astype(np.int16).tobytes())
    np.savez_compressed(os.path.join(DEMO, 'mix.npz'),
                        mix=stereo.mean(axis=1).astype(np.float32))

    n_on = int(T_ANC_ON * FS_AUDIO)
    for name, seg_lo in [('before', lowL[n_on - 3 * FS_AUDIO:n_on]),
                         ('after', lowL[n_on + 1 * FS_AUDIO:
                                        n_on + 7 * FS_AUDIO])]:
        lv = octave_band_levels(seg_lo, FS_AUDIO, centers=[63.0, 125.0, 250.0])
        print(f'listener {name}: '
              + '  '.join(f'{k:.0f}Hz {v:.1f}dB' for k, v in lv.items()))
    print('saved demo_audio.wav + mix.npz')


_V = {}


def _render_chunk(chunk):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

    V = _V
    BG, FG, DIM_C, CYAN = '#0d1117', '#e6edf3', '#8b949e', '#3ddbd9'
    STROKE = [pe.withStroke(linewidth=2.5, foreground='#0d1117')]
    fps = 30

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
    ax = fig.add_axes([0.05, 0.13, 0.78, 0.74])
    axm = fig.add_axes([0.87, 0.13, 0.10, 0.74])

    for f in chunk:
        tt = (f + 1) / fps
        ax.clear()
        axm.clear()
        ax.set_facecolor(BG)
        img = np.ma.masked_where(V['bmask'], V['spl'][f].T)
        cmap = plt.cm.magma.copy()
        cmap.set_bad('#21262d')
        ax.imshow(img, origin='lower', cmap=cmap, vmin=V['vmin'],
                  vmax=V['vmax'], aspect='auto',
                  extent=[V['x_lo'], V['x_hi'], -DOMAIN_Y / 2, DOMAIN_Y / 2],
                  interpolation='bilinear')
        ax.set_xlim(V['x_lo'], V['x_hi'])
        ax.plot(*NOISE_POS, marker='*', color='#ff5d5d', markersize=20,
                markeredgecolor='white', markeredgewidth=0.8, zorder=6)
        ax.annotate('real traffic noise', NOISE_POS,
                    xytext=(NOISE_POS[0] - 0.5, NOISE_POS[1] + 1.4),
                    fontsize=10, color=FG, path_effects=STROKE)
        anc_on = V['gate'][min(int(tt * FS_SIM), len(V['gate']) - 1)] > 0.5
        for (sx, sy) in SPEAKERS:
            ax.plot(sx, sy, marker='^', color=CYAN if anc_on else '#444c56',
                    markersize=9, markeredgecolor='black', zorder=6)
        wx, wy = walker_pos(tt)
        trail = np.array([walker_pos(t2)
                          for t2 in np.arange(0, tt + 1e-6, 0.25)])
        ax.plot(trail[:, 0], trail[:, 1], color=FG, lw=1, alpha=0.3,
                zorder=5)
        for (mxp, myp) in ERROR_MICS:
            ax.plot(mxp, myp, marker='.', color='#a5d6ff', markersize=4,
                    zorder=6)
        ax.plot(*BENCH, marker='s', color='#a5d6ff', markersize=8,
                markeredgecolor='black', zorder=6)
        ax.annotate('quiet-zone bench', BENCH,
                    xytext=(BENCH[0] - 2.2, BENCH[1] - 1.9),
                    fontsize=10, color=FG, path_effects=STROKE)
        ax.plot(wx, wy, marker='o', color='white', markersize=9,
                markeredgecolor='black', zorder=7)
        ax.annotate('you', (wx, wy), xytext=(wx - 0.3, wy + 1.1),
                    fontsize=10, color=FG, path_effects=STROKE)
        badge = 'ANC ON' if anc_on else 'ANC OFF'
        ax.text(0.985, 0.965, badge, transform=ax.transAxes, fontsize=13,
                fontweight='bold', ha='right', va='top', color=BG,
                bbox=dict(boxstyle='round,pad=0.35',
                          fc=CYAN if anc_on else '#f85149', ec='none'))
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color('#30363d')
        ax.set_title('Antiphon: walking through a simulated street '
                     '(sound pressure level, sub-400 Hz)',
                     color=FG, fontsize=12, pad=10)

        li = min(np.searchsorted(V['lvl_t'], tt), len(V['lvl']) - 1)
        frac = (V['lvl'][li] - V['lvl_min']) / (V['lvl_max'] - V['lvl_min'])
        frac = float(np.clip(frac, 0, 1))
        axm.set_facecolor('#161b22')
        axm.bar([0], [frac], width=0.6,
                color=CYAN if anc_on else '#f85149')
        axm.set_ylim(0, 1)
        axm.set_xlim(-0.5, 0.5)
        axm.set_xticks([])
        axm.set_yticks([])
        for sp in axm.spines.values():
            sp.set_color('#30363d')
        axm.set_title('what you\nhear (dB)', color=DIM_C, fontsize=9)

        fig.text(0.05, 0.05,
                 'FDTD wave simulation | FxLMS control | '
                 'audio = simulated pressure at the walker',
                 fontsize=9, color=DIM_C)
        fig.savefig(os.path.join(V['outdir'], f'{f:04d}.png'), facecolor=BG)
    plt.close(fig)
    return len(chunk)


def stage_video():
    """Render frames (parallel across cores) and mux with ffmpeg."""
    import subprocess
    from multiprocessing import Pool

    data = np.load(os.path.join(DEMO, 'playback.npz'))
    frames, gate = data['frames'], data['gate']
    mask, _ = build_scene()
    bmask = mask[::2, ::2].T > 0.5

    fps = 30
    spl = 10 * np.log10(np.maximum(frames.astype(np.float32), 1e-12))
    crop = int(1.6 / DX / 2)
    spl = spl[:, crop:-crop, :]
    x_lo, x_hi = 1.6, DOMAIN_X - 1.6
    pre = spl[:int(T_ANC_ON * fps) - 30]
    vmax = float(np.percentile(pre[len(pre) // 2:], 99.5))
    vmin = vmax - 20

    # Sliding listener level for the meter: the FULL mix (what you hear)
    mix = np.load(os.path.join(DEMO, 'mix.npz'))['mix'].astype(np.float64)
    win = int(0.3 * FS_AUDIO)
    csum = np.cumsum(mix ** 2)
    rms = np.sqrt((csum[win:] - csum[:-win]) / win)
    lvl = 20 * np.log10(np.maximum(rms, 1e-9))
    lvl_t = (np.arange(len(lvl)) + win) / FS_AUDIO

    outdir = os.path.join(DEMO, 'frames')
    os.makedirs(outdir, exist_ok=True)

    _V.update(dict(spl=spl, gate=gate, bmask=bmask[:, crop:-crop],
                   vmin=vmin, vmax=vmax, x_lo=x_lo, x_hi=x_hi,
                   lvl=lvl, lvl_t=lvl_t,
                   lvl_min=float(np.percentile(lvl, 1)),
                   lvl_max=float(np.percentile(lvl, 99.5)),
                   outdir=outdir))

    n_frames = len(spl)
    n_workers = max(1, os.cpu_count() - 2)
    chunks = [list(range(i, n_frames, n_workers)) for i in range(n_workers)]
    print(f'rendering {n_frames} frames on {n_workers} workers...',
          flush=True)
    with Pool(n_workers) as pool:
        pool.map(_render_chunk, chunks)

    out = os.path.join(DEMO, 'antiphon_walk_demo.mp4')
    subprocess.run([
        'ffmpeg', '-y', '-v', 'warning',
        '-framerate', str(fps), '-i', os.path.join(outdir, '%04d.png'),
        '-i', os.path.join(DEMO, 'demo_audio.wav'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20',
        '-c:a', 'aac', '-b:a', '160k', '-shortest', out], check=True)
    print(f'saved {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',
                        choices=['paths', 'fxlms', 'wiener', 'playback',
                                 'audio', 'video'],
                        required=True)
    args = parser.parse_args()
    os.makedirs(DEMO, exist_ok=True)
    if args.stage == 'paths':
        stage_paths()
    elif args.stage == 'fxlms':
        stage_fxlms()
    elif args.stage == 'wiener':
        stage_wiener()
    elif args.stage == 'playback':
        stage_playback()
    elif args.stage == 'audio':
        stage_audio()
    else:
        stage_video()
