"""Multi-channel filtered-x LMS (FxLMS) active noise control.

Reference: S. Elliott & P. Nelson, "Active Noise Control", IEEE Signal
Processing Magazine, 1993.

Conventions:
- M reference signals, J speakers, K error microphones.
- The plant is FIR: primary paths d_k(n) (noise already propagated to each
  error mic) and secondary paths S (K, J, Ls) from speakers to error mics.
- The controller adapts W (J, M, Lw) using references filtered through the
  secondary-path estimate Shat (K, J, Lsh), with normalized step size.
- Online secondary-path identification (Eriksson): low-level auxiliary white
  noise is injected at the speakers and Shat is adapted by LMS on it.
"""

import numpy as np


class MultichannelFxLMS:
    """Adaptive feedforward controller (the DSP block only; the acoustic
    plant lives in `simulate_anc`)."""

    def __init__(self, n_ref, n_speakers, n_error, filter_len,
                 secondary_estimate=None, shat_len=None, mu=0.05,
                 leak=0.0):
        self.M = n_ref
        self.J = n_speakers
        self.K = n_error
        self.Lw = filter_len
        self.mu = mu
        self.leak = leak

        if secondary_estimate is not None:
            self.shat = np.array(secondary_estimate, dtype=float)
            self.Lsh = self.shat.shape[2]
        else:
            if shat_len is None:
                raise ValueError('need secondary_estimate or shat_len')
            self.Lsh = shat_len
            self.shat = np.zeros((self.K, self.J, self.Lsh))

        self.W = np.zeros((self.J, self.M, self.Lw))
        # Signal buffers, newest sample first
        self.xbuf = np.zeros((self.M, max(self.Lw, self.Lsh)))
        self.xprime = np.zeros((self.K, self.J, self.M, self.Lw))

    def compute_output(self, x_n):
        """Push new reference samples, return speaker outputs y (J,)."""
        self.xbuf[:, 1:] = self.xbuf[:, :-1]
        self.xbuf[:, 0] = x_n
        y = np.einsum('jml,ml->j', self.W, self.xbuf[:, :self.Lw])

        # Filter the reference through Shat and push into the x' history
        xp_n = np.einsum('kjl,ml->kjm', self.shat, self.xbuf[:, :self.Lsh])
        self.xprime[..., 1:] = self.xprime[..., :-1]
        self.xprime[..., 0] = xp_n
        return y

    def adapt(self, e_n):
        """Normalized LMS weight update from error samples e (K,)."""
        grad = np.einsum('k,kjml->jml', e_n, self.xprime)
        # NLMS: normalize by the filtered-reference buffer energy
        pxp = np.mean(np.sum(self.xprime ** 2, axis=-1))
        mu_n = self.mu / (pxp + 1e-8)
        if self.leak:
            self.W *= (1.0 - self.leak)
        self.W -= mu_n * grad


class OnlineSecondaryPathLMS:
    """Eriksson-style online secondary path identification.

    White auxiliary noise v_j is injected at each speaker; Shat adapts to
    predict the error-mic response to v, using the residual after removing
    the prediction.
    """

    def __init__(self, controller, level=0.05, mu=0.01, seed=0):
        self.ctl = controller
        self.level = level
        self.mu = mu
        self.rng = np.random.default_rng(seed)
        self.vbuf = np.zeros((controller.J, controller.Lsh))

    def inject(self):
        """Draw new auxiliary noise samples v (J,) and push the buffer."""
        v = self.level * self.rng.standard_normal(self.ctl.J)
        self.vbuf[:, 1:] = self.vbuf[:, :-1]
        self.vbuf[:, 0] = v
        return v

    def adapt(self, e_n):
        """Update Shat from error samples (which contain the response to v)."""
        pred = np.einsum('kjl,jl->k', self.ctl.shat, self.vbuf)
        resid = e_n - pred
        pv = np.mean(np.sum(self.vbuf ** 2, axis=-1))
        mu_n = self.mu / (pv + 1e-8)
        self.ctl.shat += mu_n * np.einsum('k,jl->kjl', resid, self.vbuf)


def simulate_anc(reference, primary_d, secondary_ir, controller,
                 online_id=None, adapt=True):
    """Run the full ANC loop through an FIR acoustic plant.

    Parameters
    ----------
    reference : (M, N) reference signals
    primary_d : (K, N) noise at the error mics with ANC off
    secondary_ir : (K, J, Ls) true speaker-to-error-mic impulse responses
    controller : MultichannelFxLMS
    online_id : OnlineSecondaryPathLMS or None
    adapt : whether to adapt W (False = measure with frozen weights)

    Returns
    -------
    e : (K, N) error-mic signals with ANC on
    """
    reference = np.atleast_2d(reference)
    K, J, Ls = secondary_ir.shape
    N = reference.shape[1]
    ybuf = np.zeros((J, Ls))
    e = np.zeros((K, N))

    for n in range(N):
        y = controller.compute_output(reference[:, n])
        if online_id is not None:
            y = y + online_id.inject()

        ybuf[:, 1:] = ybuf[:, :-1]
        ybuf[:, 0] = y

        e_n = primary_d[:, n] + np.einsum('kjl,jl->k', secondary_ir, ybuf)
        e[:, n] = e_n

        if adapt:
            controller.adapt(e_n)
        if online_id is not None:
            online_id.adapt(e_n)

    return e


def db_reduction(d, e, tail_fraction=0.25):
    """Noise reduction in dB per error mic, measured over the final
    tail_fraction of the run (after convergence)."""
    n0 = int(d.shape[1] * (1 - tail_fraction))
    rms_d = np.sqrt(np.mean(d[:, n0:] ** 2, axis=1))
    rms_e = np.sqrt(np.mean(e[:, n0:] ** 2, axis=1))
    return 20 * np.log10(rms_d / np.maximum(rms_e, 1e-12))
