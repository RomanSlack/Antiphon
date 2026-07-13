"""Training pipeline for the acoustics foundation model v1.

- Splits scenes (not samples) into train/val/test, so validation measures
  generalization to unseen geometries.
- Loss: complex MSE on standardized H (equal weight on Re/Im).
- Baselines the model must beat on held-out scenes:
  1. mean-H: predict the training-set mean H(f).
  2. free-field: c(f) * H0(k*r) with the per-frequency complex calibration
     c(f) least-squares fitted on the training set (a strong physics
     baseline that knows the source-receiver distance but no geometry).
"""

import json
import os
import time

import numpy as np
import torch
from scipy.special import hankel1

from ..simulation.geometry import C_SOUND
from .architecture import AcousticsModelV1
from .dataset import DOMAIN_X, DOMAIN_Y, FREQS


def delay_compensate(H, freqs, dist, inverse=False):
    """Multiply H by e^{+i k r} (remove bulk propagation delay) so the
    phase is slowly varying across the frequency grid. inverse=True
    restores the physical H. MSE is invariant under this per-sample
    unit-modulus rotation, so losses/baselines stay comparable."""
    k = 2 * np.pi * np.asarray(freqs) / C_SOUND
    phase = np.exp((1j if not inverse else -1j) * dist[..., None] * k)
    return H * phase


class SceneData:
    """In-memory dataset with scene-level splits.

    Targets are delay-compensated: the model learns H * e^{+ikr}, which is
    smooth in frequency; the known propagation delay is reapplied at
    inference time.
    """

    def __init__(self, path, val_frac=0.05, test_frac=0.05, seed=0):
        import h5py
        with h5py.File(path, 'r') as f:
            self.occ = f['occupancy'][:]          # (S, H, W)
            self.alpha = f['alpha'][:]            # (S, 2)
            self.src = f['src'][:]                # (S, 2)
            self.rcv = f['rcv'][:]                # (S, R, 2)
            self.H = f['H_real'][:] + 1j * f['H_imag'][:]  # (S, R, F)
            self.freqs = f.attrs['freqs']

        self.dist = np.linalg.norm(
            self.rcv - self.src[:, None, :], axis=-1)  # (S, R)
        self.H = delay_compensate(self.H, self.freqs, self.dist)

        S = self.occ.shape[0]
        rng = np.random.default_rng(seed)
        order = rng.permutation(S)
        n_test = max(1, int(S * test_frac))
        n_val = max(1, int(S * val_frac))
        self.test_ix = order[:n_test]
        self.val_ix = order[n_test:n_test + n_val]
        self.train_ix = order[n_test + n_val:]

        # Standardize H by the train-set std (single global scale)
        self.h_scale = float(np.std(
            np.concatenate([self.H[self.train_ix].real.ravel(),
                            self.H[self.train_ix].imag.ravel()])))

    def norm_pos(self, pos):
        """Physical (x, y) -> [0,1]^2 over the fixed domain."""
        out = np.empty_like(pos)
        out[..., 0] = pos[..., 0] / DOMAIN_X
        out[..., 1] = (pos[..., 1] + DOMAIN_Y / 2) / DOMAIN_Y
        return out

    def batch(self, scene_ids, device='cpu'):
        """Assemble one batch: all receivers of the given scenes."""
        occ = torch.from_numpy(self.occ[scene_ids]).to(device)
        alpha = torch.from_numpy(self.alpha[scene_ids]).to(device)
        R = self.rcv.shape[1]
        src = np.repeat(self.src[scene_ids][:, None, :], R, axis=1)
        src_t = torch.from_numpy(self.norm_pos(src).astype(np.float32))
        rcv_t = torch.from_numpy(
            self.norm_pos(self.rcv[scene_ids]).astype(np.float32))
        target = torch.from_numpy(self.H[scene_ids] / self.h_scale)
        return occ, src_t.to(device), rcv_t.to(device), alpha, target.to(device)


def complex_mse(pred, target):
    diff = pred - target
    return torch.mean(diff.real ** 2 + diff.imag ** 2)


def eval_split(model, data, ids, batch_scenes=16):
    model.eval()
    losses, n = [], 0
    with torch.no_grad():
        for i in range(0, len(ids), batch_scenes):
            occ, src, rcv, alpha, target = data.batch(ids[i:i + batch_scenes])
            pred = model(occ, src, rcv, alpha)
            losses.append(complex_mse(pred, target).item() * occ.shape[0])
            n += occ.shape[0]
    return float(np.sum(losses) / n)


def baseline_metrics(data):
    """Complex MSE (standardized units) of the two baselines per split."""
    Htr = data.H[data.train_ix] / data.h_scale     # (S, R, F) compensated
    mean_H = Htr.mean(axis=(0, 1))                 # (F,)

    # Free-field baseline, delay-compensated like the targets
    k = 2 * np.pi * np.asarray(data.freqs) / C_SOUND

    def g_comp(ids):
        d = data.dist[ids][..., None]
        return hankel1(0, d * k) * np.exp(1j * d * k)

    # Fit per-frequency complex calibration on the training set
    g_tr = g_comp(data.train_ix)
    c = (np.sum(np.conj(g_tr) * Htr, axis=(0, 1))
         / np.sum(np.abs(g_tr) ** 2, axis=(0, 1)))         # (F,)

    out = {}
    for name, ids in [('val', data.val_ix), ('test', data.test_ix)]:
        Hs = data.H[ids] / data.h_scale
        mse_mean = float(np.mean(np.abs(Hs - mean_H) ** 2))
        mse_ff = float(np.mean(np.abs(Hs - c * g_comp(ids)) ** 2))
        out[name] = {'mean_H': mse_mean, 'free_field': mse_ff}
    return out


def train(dataset_path, out_dir, epochs=30, batch_scenes=8, lr=3e-4,
          seed=0, threads=None, log_every=20):
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(out_dir, exist_ok=True)
    data = SceneData(dataset_path, seed=seed)
    print(f'scenes: {len(data.train_ix)} train / {len(data.val_ix)} val / '
          f'{len(data.test_ix)} test; h_scale={data.h_scale:.2e}')

    baselines = baseline_metrics(data)
    print(f'baselines (val): {baselines["val"]}')

    model = AcousticsModelV1(n_freqs=len(data.freqs))
    print(f'model params: {model.count_parameters()/1e6:.1f}M')
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_epoch = int(np.ceil(len(data.train_ix) / batch_scenes))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs * steps_per_epoch)

    rng = np.random.default_rng(seed)
    best_val = np.inf
    history = []
    t0 = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(data.train_ix)
        ep_losses = []
        for i in range(0, len(order), batch_scenes):
            occ, src, rcv, alpha, target = data.batch(order[i:i + batch_scenes])
            pred = model(occ, src, rcv, alpha)
            loss = complex_mse(pred, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ep_losses.append(loss.item())
            step = i // batch_scenes
            if step % log_every == 0:
                el = time.perf_counter() - t0
                print(f'epoch {epoch} step {step}/{steps_per_epoch} '
                      f'loss {loss.item():.4f} ({el/60:.1f} min)', flush=True)

        val = eval_split(model, data, data.val_ix)
        history.append({'epoch': epoch,
                        'train_loss': float(np.mean(ep_losses)),
                        'val_loss': val})
        print(f'== epoch {epoch}: train {np.mean(ep_losses):.4f} '
              f'val {val:.4f} (baseline ff {baselines["val"]["free_field"]:.4f})',
              flush=True)
        if val < best_val:
            best_val = val
            torch.save({'model': model.state_dict(),
                        'h_scale': data.h_scale,
                        'freqs': np.asarray(data.freqs)},
                       os.path.join(out_dir, 'best.pt'))

    test = eval_split(model, data, data.test_ix)
    metrics = {
        'best_val_loss': best_val,
        'test_loss': test,
        'baselines': baselines,
        'history': history,
        'h_scale': data.h_scale,
        'params': model.count_parameters(),
        'train_minutes': (time.perf_counter() - t0) / 60,
        'test_scene_ids': data.test_ix.tolist(),
    }
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'done: best val {best_val:.4f}, test {test:.4f}')
    return metrics
