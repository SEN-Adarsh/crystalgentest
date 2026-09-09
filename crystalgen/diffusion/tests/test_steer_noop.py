"""w=0 must be a strict no-op: _steer returns the score object unchanged.

This is the code-level parity guarantee (handover section 6, item 4): with
guidance disabled, sampling performs no guidance computation whatsoever, so
stock sampling behaviour is preserved. Full bit-level reproducibility of
two GPU runs is a separate property of the stack itself (torch-cluster /
scatter atomics have no deterministic kernels) and holds equally with or
without this code path.
"""

import torch

from crystalgen.diffusion.data.batched_data import SimpleBatchedData


def _make_sampler(**kwargs):
    from crystalgen.diffusion.corruption.multi_corruption import MultiCorruption
    from crystalgen.diffusion.corruption.sde_lib import VESDE
    from crystalgen.diffusion.sampling.pc_sampler import PredictorCorrector
    from crystalgen.diffusion.sampling.predictors_correctors import LangevinCorrector

    module = SimpleBatchedData(data={}, batch_idx={})
    module.corruption = MultiCorruption(sdes={"pos": VESDE()})
    return PredictorCorrector(
        diffusion_module=module,
        predictor_partials={},
        corrector_partials={
            "pos": lambda corruption, n_steps, score_fn: LangevinCorrector(
                corruption, score_fn=score_fn, n_steps=n_steps
            )
        },
        device=torch.device("cpu"),
        n_steps_corrector=0,
        N=1000,
        **kwargs,
    )


def _score():
    return SimpleBatchedData(
        data={"pos": torch.randn(4, 3)}, batch_idx={"pos": torch.tensor([0, 0, 1, 1])}
    )


def test_steer_w0_is_identity_noop():
    sampler = _make_sampler(polyhedral_guidance_weight=0.0)
    score = _score()
    t = torch.tensor([0.5, 0.5])

    out = sampler._steer(score, score, t)
    assert out is score, "w=0 must return the identical score object"
    assert sampler.steer_trace == [], "w=0 must not record any guidance calls"


def test_steer_w0_beats_every_guard(monkeypatch):
    """Even with a callable grad path and valid batch fields, w=0 returns
    before any guidance computation (first guard in _steer)."""
    from crystalgen.diffusion.sampling import pc_sampler

    sampler = _make_sampler(polyhedral_guidance_weight=0.0)
    score = _score()

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("polyhedral_guidance_grad must not be called at w=0")

    monkeypatch.setattr(pc_sampler, "polyhedral_guidance_grad", _must_not_be_called)
    t = torch.tensor([0.5, 0.5])
    out = sampler._steer(score, score, t)
    assert out is score
