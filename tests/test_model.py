"""Model architecture and training-loop tests (small configs for speed)."""

import numpy as np
import pytest
import torch

from antiphon.model.architecture import AcousticsModelV1
from antiphon.model.train import complex_mse


@pytest.fixture(scope='module')
def small_model():
    torch.manual_seed(0)
    return AcousticsModelV1(n_freqs=16, dim=64, n_blocks=2, n_heads=4)


def test_forward_shapes(small_model):
    occ = torch.rand(3, 60, 90)
    src = torch.rand(3, 7, 2)
    rcv = torch.rand(3, 7, 2)
    alpha = torch.rand(3, 2)
    out = small_model(occ, src, rcv, alpha)
    assert out.shape == (3, 7, 16)
    assert out.dtype == torch.complex64


def test_default_model_size():
    m = AcousticsModelV1()
    n = m.count_parameters()
    assert 8e6 < n < 15e6, f'{n/1e6:.1f}M params, expected ~10M'


def test_complex_mse():
    a = torch.complex(torch.ones(4), torch.zeros(4))
    b = torch.complex(torch.zeros(4), torch.ones(4))
    assert complex_mse(a, b).item() == pytest.approx(2.0)
    assert complex_mse(a, a).item() == 0.0


def test_model_can_overfit_tiny_batch(small_model):
    """The training signal flows: loss should drop sharply on one batch."""
    torch.manual_seed(0)
    occ = torch.rand(2, 60, 90)
    src = torch.rand(2, 4, 2)
    rcv = torch.rand(2, 4, 2)
    alpha = torch.rand(2, 2)
    target = torch.complex(torch.randn(2, 4, 16), torch.randn(2, 4, 16))

    opt = torch.optim.Adam(small_model.parameters(), lr=1e-3)
    first = None
    for i in range(150):
        pred = small_model(occ, src, rcv, alpha)
        loss = complex_mse(pred, target)
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1 * first, f'{first:.3f} -> {loss.item():.3f}'


def test_deterministic_forward():
    outs = []
    for _ in range(2):
        torch.manual_seed(42)
        m = AcousticsModelV1(n_freqs=8, dim=32, n_blocks=1, n_heads=2)
        occ = torch.ones(1, 60, 90) * 0.5
        out = m(occ, torch.full((1, 1, 2), 0.5), torch.full((1, 1, 2), 0.25),
                torch.full((1, 2), 0.1))
        outs.append(out.detach().numpy())
    np.testing.assert_array_equal(outs[0], outs[1])
