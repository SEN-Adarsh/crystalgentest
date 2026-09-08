# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import json
from pathlib import Path
from typing import Generic, Mapping, Tuple, TypeVar

import torch
from tqdm.auto import tqdm

from crystalgen.diffusion.corruption.multi_corruption import MultiCorruption, apply
from crystalgen.diffusion.data.batched_data import BatchedData
from crystalgen.diffusion.diffusion_module import DiffusionModule
from crystalgen.diffusion.lightning_module import DiffusionLightningModule
from crystalgen.diffusion.polyhedra import (
    PolyhedralConnectivityLoss,
    PolyhedralGeometryLoss,
    polyhedral_guidance_grad,
)
from crystalgen.diffusion.sampling.pc_partials import CorrectorPartial, PredictorPartial

Diffusable = TypeVar(
    "Diffusable", bound=BatchedData
)  # Don't use 'T' because it clashes with the 'T' for time
SampleAndMean = Tuple[Diffusable, Diffusable]
SampleAndMeanAndMaybeRecords = Tuple[Diffusable, Diffusable, list[Diffusable] | None]
SampleAndMeanAndRecords = Tuple[Diffusable, Diffusable, list[Diffusable]]


class PredictorCorrector(Generic[Diffusable]):
    """Generates samples using predictor-corrector sampling."""

    def __init__(
        self,
        *,
        diffusion_module: DiffusionModule,
        predictor_partials: dict[str, PredictorPartial] | None = None,
        corrector_partials: dict[str, CorrectorPartial] | None = None,
        device: torch.device,
        n_steps_corrector: int,
        N: int,
        eps_t: float = 1e-3,
        max_t: float | None = None,
        polyhedral_guidance_weight: float = 0.0,
        polyhedral_geometry_weight: float = 1.0,
        polyhedral_connectivity_weight: float = 1.0,
        polyhedral_annealing: bool = True,
        max_guidance_rel: float | None = None,
        steer_trace_path: str | None = None,
    ):
        """
        Args:
            diffusion_module: diffusion module
            predictor_partials: partials for constructing predictors. Keys are the names of the corruptions.
            corrector_partials: partials for constructing correctors. Keys are the names of the corruptions.
            device: device to run on
            n_steps_corrector: number of corrector steps
            N: number of noise levels
            eps_t: diffusion time to stop denoising at
            max_t: diffusion time to start denoising at. If None, defaults to the maximum diffusion time. You may want to start at T-0.01, say, for numerical stability.
            polyhedral_guidance_weight: strength of the polyhedral geometry/connectivity
                guidance added to the coordinate score at every step. 0 disables it,
                which reproduces stock CrystalGen sampling exactly.
            polyhedral_geometry_weight: relative weight of the shape-regularity term
                within the guidance.
            polyhedral_connectivity_weight: relative weight of the corner-sharing term
                within the guidance.
            polyhedral_annealing: fade the guidance out as (1 - t) at high noise.
                Off = full-strength guidance at every step (Run 3 ablation arm).
            max_guidance_rel: if set, clamp |w * grad| to at most this multiple of
                |score|. Live diagnostics show rare spikes of 90-1000x |score| at
                individual steps; None reproduces the original unclamped behaviour.
            steer_trace_path: if set, write the per-step guidance diagnostics to
                this JSON file at the end of each denoising run.
        """
        self._diffusion_module = diffusion_module
        self.N = N

        if max_t is None:
            max_t = self._multi_corruption.T
        assert max_t <= self._multi_corruption.T, "Denoising cannot start from beyond T"

        self._max_t = max_t
        assert (
            corrector_partials or predictor_partials
        ), "Must specify at least one predictor or corrector"
        corrector_partials = corrector_partials or {}
        predictor_partials = predictor_partials or {}
        if self._multi_corruption.discrete_corruptions:
            # These all have property 'N' because they are D3PM type
            assert set(c.N for c in self._multi_corruption.discrete_corruptions.values()) == {N}  # type: ignore

        self._predictors = {
            k: v(corruption=self._multi_corruption.corruptions[k], score_fn=None)
            for k, v in predictor_partials.items()
        }

        self._correctors = {
            k: v(
                corruption=self._multi_corruption.corruptions[k],
                n_steps=n_steps_corrector,
                score_fn=None,
            )
            for k, v in corrector_partials.items()
        }
        self._eps_t = eps_t
        self._n_steps_corrector = n_steps_corrector
        self._device = device

        self._polyhedral_guidance_weight = polyhedral_guidance_weight
        self._polyhedral_geometry_weight = polyhedral_geometry_weight
        self._polyhedral_connectivity_weight = polyhedral_connectivity_weight
        self._polyhedral_annealing = polyhedral_annealing
        self._max_guidance_rel = max_guidance_rel
        self._steer_trace_path = steer_trace_path
        self._polyhedral_geometry_loss = PolyhedralGeometryLoss()
        self._polyhedral_connectivity_loss = PolyhedralConnectivityLoss()

        # Sampling-time guidance diagnostics: one entry per _steer call, summarized
        # at the end of each _denoise run. ponytail: in-memory only; persist
        # `steer_trace` from the caller if a paper figure needs per-step data.
        self.steer_trace: list[dict] = []

    @property
    def diffusion_module(self) -> DiffusionModule:
        return self._diffusion_module

    @property
    def _multi_corruption(self) -> MultiCorruption:
        return self._diffusion_module.corruption

    def _score_fn(self, x: Diffusable, t: torch.Tensor) -> Diffusable:
        return self._diffusion_module.score_fn(x, t)

    def _steer(self, score: Diffusable, batch: Diffusable, t: torch.Tensor) -> Diffusable:
        """Add polyhedral guidance to the coordinate score.

        The score points towards higher probability density; subtracting the
        gradient of a penalty steers the trajectory away from that penalty, so
        the update is `score - w * grad(loss)`. This shapes samples at inference
        time and needs no retraining, which is what makes it usable with an
        already fine-tuned checkpoint.

        Applied after the model call rather than inside `_score_fn` so that it
        runs once on the real batch, not on the doubled batch that
        classifier-free guidance constructs.
        """
        if self._polyhedral_guidance_weight == 0.0:
            return score
        if getattr(score, "pos", None) is None or getattr(batch, "cell", None) is None:
            return score

        batch_idx = batch.get_batch_idx("pos")
        # Guidance is evaluated on the current noisy coordinates, so fade it in as
        # the geometry becomes meaningful. At t near 1 the structure is still
        # essentially random and its polyhedra carry no signal.
        if self._polyhedral_annealing:
            node_weight = (1.0 - t).clamp(min=0.0, max=1.0)[batch_idx]
        else:
            node_weight = torch.ones_like(t)[batch_idx]

        grad = polyhedral_guidance_grad(
            frac_coords=batch.pos,
            cell=batch.cell,
            atomic_numbers=batch.atomic_numbers,
            num_atoms=batch.num_atoms,
            batch_idx=batch_idx,
            num_graphs=batch.get_batch_size(),
            node_weight=node_weight,
            geometry_loss=self._polyhedral_geometry_loss,
            connectivity_loss=self._polyhedral_connectivity_loss,
            geometry_weight=self._polyhedral_geometry_weight,
            connectivity_weight=self._polyhedral_connectivity_weight,
        )
        if grad is None:
            self.steer_trace.append({"t": t[0].item(), "active": False})
            return score

        grad_norm = grad.norm().item()
        score_norm = score.pos.norm().item()
        scaled_norm = self._polyhedral_guidance_weight * grad_norm
        clamped = False
        if (
            self._max_guidance_rel is not None
            and scaled_norm > self._max_guidance_rel * score_norm
            and scaled_norm > 0.0
        ):
            grad = grad * (self._max_guidance_rel * score_norm / scaled_norm)
            scaled_norm = self._max_guidance_rel * score_norm
            clamped = True
        self.steer_trace.append(
            {
                "t": t[0].item(),
                "active": True,
                "grad_norm": grad_norm,
                "score_norm": score_norm,
                "rel": scaled_norm / max(score_norm, 1e-12),
                "clamped": clamped,
            }
        )
        return score.replace(pos=score.pos - self._polyhedral_guidance_weight * grad)

    def _log_steer_summary(self) -> None:
        """Print guidance diagnostics accumulated during the last _denoise run,
        and persist the full per-step trace if a path was configured."""
        if self._polyhedral_guidance_weight == 0.0 or not self.steer_trace:
            return
        entries = self.steer_trace
        active = [e for e in entries if e["active"]]
        n = len(entries)
        clamped = sum(1 for e in active if e.get("clamped"))
        print(
            f"[_steer] calls={n} active={len(active)} "
            f"({100.0 * len(active) / max(n, 1):.1f}%) no-op={n - len(active)}"
        )
        if active:
            rel = sorted(e["rel"] for e in active)
            grads = sorted(e["grad_norm"] for e in active)
            ts = [e["t"] for e in active]
            print(
                f"[_steer] |w*grad|/|score| median={rel[len(rel) // 2]:.3g} max={rel[-1]:.3g} | "
                f"|grad| median={grads[len(grads) // 2]:.3g} max={grads[-1]:.3g} | "
                f"active t range=[{min(ts):.3f}, {max(ts):.3f}]"
            )
            if clamped:
                print(f"[_steer] clamped steps: {clamped}/{len(active)}")
        if self._steer_trace_path is not None:
            out = Path(self._steer_trace_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            # A multi-batch generate() call runs _denoise once per batch; number
            # the trace files so batches do not overwrite each other.
            idx = 0
            numbered = out.with_name(f"{out.stem}_{idx:03d}{out.suffix}")
            while numbered.exists():
                idx += 1
                numbered = out.with_name(f"{out.stem}_{idx:03d}{out.suffix}")
            numbered.write_text(json.dumps(entries))
            print(f"[_steer] trace written to {numbered}")

    @classmethod
    def from_pl_module(cls, pl_module: DiffusionLightningModule, **kwargs) -> PredictorCorrector:
        return cls(diffusion_module=pl_module.diffusion_module, device=pl_module.device, **kwargs)

    @torch.no_grad()
    def sample(
        self, conditioning_data: BatchedData, mask: Mapping[str, torch.Tensor] | None = None
    ) -> SampleAndMean:
        """Create one sample for each of a batch of conditions.
        Args:
            conditioning_data: batched conditioning data. Even if you think you don't want conditioning, you still need to pass a batch of conditions
               because the sampler uses these to determine the shapes of things to generate.
            mask: for inpainting. Keys should be a subset of the keys in `data`. 1 indicates data that should be fixed, 0 indicates data that should be replaced with sampled values.
                Shapes of values in `mask` must match the shapes of values in `conditioning_data`.
        Returns:
           (batch, mean_batch). The difference between these is that `mean_batch` has no noise added at the final denoising step.

        """
        return self._sample_maybe_record(conditioning_data, mask=mask, record=False)[:2]

    @torch.no_grad()
    def sample_with_record(
        self, conditioning_data: BatchedData, mask: Mapping[str, torch.Tensor] | None = None
    ) -> SampleAndMeanAndRecords:
        """Create one sample for each of a batch of conditions.
        Args:
            conditioning_data: batched conditioning data. Even if you think you don't want conditioning, you still need to pass a batch of conditions
               because the sampler uses these to determine the shapes of things to generate.
            mask: for inpainting. Keys should be a subset of the keys in `data`. 1 indicates data that should be fixed, 0 indicates data that should be replaced with sampled values.
                Shapes of values in `mask` must match the shapes of values in `conditioning_data`.
        Returns:
           (batch, mean_batch). The difference between these is that `mean_batch` has no noise added at the final denoising step.

        """
        return self._sample_maybe_record(conditioning_data, mask=mask, record=True)

    @torch.no_grad()
    def _sample_maybe_record(
        self,
        conditioning_data: BatchedData,
        mask: Mapping[str, torch.Tensor] | None = None,
        record: bool = False,
    ) -> SampleAndMeanAndMaybeRecords:
        """Create one sample for each of a batch of conditions.
        Args:
            conditioning_data: batched conditioning data. Even if you think you don't want conditioning, you still need to pass a batch of conditions
               because the sampler uses these to determine the shapes of things to generate.
            mask: for inpainting. Keys should be a subset of the keys in `data`. 1 indicates data that should be fixed, 0 indicates data that should be replaced with sampled values.
                Shapes of values in `mask` must match the shapes of values in `conditioning_data`.
        Returns:
           (batch, mean_batch, recorded_samples, recorded_predictions).
           The difference between the former two is that `mean_batch` has no noise added at the final denoising step.
           The latter two are only returned if `record` is True, and contain the samples and predictions from each step of the diffusion process.

        """
        if isinstance(self._diffusion_module, torch.nn.Module):
            self._diffusion_module.eval()
        mask = mask or {}
        conditioning_data = conditioning_data.to(self._device)
        mask = {k: v.to(self._device) for k, v in mask.items()}
        batch = _sample_prior(self._multi_corruption, conditioning_data, mask=mask)
        return self._denoise(batch=batch, mask=mask, record=record)

    @torch.no_grad()
    def _denoise(
        self,
        batch: Diffusable,
        mask: dict[str, torch.Tensor],
        record: bool = False,
    ) -> SampleAndMeanAndMaybeRecords:
        """Denoise from a prior sample to a t=eps_t sample."""
        recorded_samples = None
        if record:
            recorded_samples = []
        for k in self._predictors:
            mask.setdefault(k, None)
        for k in self._correctors:
            mask.setdefault(k, None)
        mean_batch = batch.clone()
        self.steer_trace = []

        # Decreasing timesteps from T to eps_t
        timesteps = torch.linspace(self._max_t, self._eps_t, self.N, device=self._device)
        dt = -torch.tensor((self._max_t - self._eps_t) / (self.N - 1)).to(self._device)

        for i in tqdm(range(self.N), miniters=50, mininterval=5):
            # Set the timestep
            t = torch.full((batch.get_batch_size(),), timesteps[i], device=self._device)

            # Corrector updates.
            if self._correctors:
                for _ in range(self._n_steps_corrector):
                    score = self._steer(self._score_fn(batch, t), batch, t)
                    fns = {
                        k: corrector.step_given_score for k, corrector in self._correctors.items()
                    }
                    samples_means: dict[str, Tuple[torch.Tensor, torch.Tensor]] = apply(
                        fns=fns,
                        broadcast={"t": t, "dt": dt},
                        x=batch,
                        score=score,
                        batch_idx=self._multi_corruption._get_batch_indices(batch),
                    )
                    if record:
                        recorded_samples.append(batch.clone().to("cpu"))
                    batch, mean_batch = _mask_replace(
                        samples_means=samples_means, batch=batch, mean_batch=mean_batch, mask=mask
                    )

            # Predictor updates
            score = self._steer(self._score_fn(batch, t), batch, t)
            predictor_fns = {
                k: predictor.update_given_score for k, predictor in self._predictors.items()
            }
            samples_means = apply(
                fns=predictor_fns,
                x=batch,
                score=score,
                broadcast=dict(t=t, batch=batch, dt=dt),
                batch_idx=self._multi_corruption._get_batch_indices(batch),
            )
            if record:
                recorded_samples.append(batch.clone().to("cpu"))
            batch, mean_batch = _mask_replace(
                samples_means=samples_means, batch=batch, mean_batch=mean_batch, mask=mask
            )

        self._log_steer_summary()
        return batch, mean_batch, recorded_samples


def _mask_replace(
    samples_means: dict[str, Tuple[torch.Tensor, torch.Tensor]],
    batch: BatchedData,
    mean_batch: BatchedData,
    mask: dict[str, torch.Tensor | None],
) -> SampleAndMean:
    # Apply masks
    samples_means = apply(
        fns={k: _mask_both for k in samples_means},
        broadcast={},
        sample_and_mean=samples_means,
        mask=mask,
        old_x=batch,
    )

    # Put the updated values in `batch` and `mean_batch`
    batch = batch.replace(**{k: v[0] for k, v in samples_means.items()})
    mean_batch = mean_batch.replace(**{k: v[1] for k, v in samples_means.items()})
    return batch, mean_batch


def _mask_both(
    *, sample_and_mean: Tuple[torch.Tensor, torch.Tensor], old_x: torch.Tensor, mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    return tuple(_mask(old_x=old_x, new_x=x, mask=mask) for x in sample_and_mean)  # type: ignore


def _mask(*, old_x: torch.Tensor, new_x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Replace new_x with old_x where mask is 1."""
    if mask is None:
        return new_x
    else:
        return new_x.lerp(old_x, mask)


def _sample_prior(
    multi_corruption: MultiCorruption,
    conditioning_data: BatchedData,
    mask: Mapping[str, torch.Tensor] | None,
) -> BatchedData:
    samples = {
        k: multi_corruption.corruptions[k]
        .prior_sampling(
            shape=conditioning_data[k].shape,
            conditioning_data=conditioning_data,
            batch_idx=conditioning_data.get_batch_idx(field_name=k),
        )
        .to(conditioning_data[k].device)
        for k in multi_corruption.corruptions
    }
    mask = mask or {}
    for k, msk in mask.items():
        if k in multi_corruption.corrupted_fields:
            samples[k].lerp_(conditioning_data[k], msk)
    return conditioning_data.replace(**samples)
