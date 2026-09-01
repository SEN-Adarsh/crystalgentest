# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from functools import partial
from typing import Dict, Literal, Optional, Protocol, Tuple, TypeVar

import torch

from mattergen.diffusion.corruption.multi_corruption import MultiCorruption, apply
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.model_target import ModelTargets
from mattergen.diffusion.physics import build_physics_graph
from mattergen.diffusion.polyhedra import PolyhedralConnectivityLoss, PolyhedralGeometryLoss
from mattergen.diffusion.redox import DifferentiableRedoxLoss
from mattergen.diffusion.training.field_loss import FieldLoss, denoising_score_matching

T = TypeVar("T", bound=BatchedData)


class Loss(Protocol[T]):
    """Loss function for training a score model on multi-field data."""

    def __call__(
        self,
        *,
        multi_corruption: MultiCorruption[T],
        batch: T,
        noisy_batch: T,
        score_model_output: T,
        t: torch.Tensor,
        node_is_unmasked: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        pass

    """model_targets tells us what this loss function trains the score model to predict.
    We need this information in order to convert the model output to a score during sampling.
    """
    model_targets: ModelTargets


class SummedFieldLoss(Loss[T]):
    """(Weighted) sum of different loss functions applied on each field, plus
    optional physics-informed auxiliary losses on the denoised structure."""

    def __init__(
        self,
        loss_fns: Dict[str, FieldLoss],
        model_targets: ModelTargets,
        weights: Optional[Dict[str, float]] = None,
        redox_weight: float = 0.01,
        poly_weight: float = 0.05,
        connectivity_weight: float = 0.05,
    ) -> None:
        self.model_targets = model_targets
        self.loss_fns = loss_fns
        self.redox_weight = redox_weight
        self.poly_weight = poly_weight
        self.connectivity_weight = connectivity_weight

        # Domain-specific physics losses, evaluated on the denoiser's own
        # prediction of the clean structure (see _physics_losses).
        self.redox_loss_fn = DifferentiableRedoxLoss()
        self.poly_loss_fn = PolyhedralGeometryLoss()
        self.connectivity_loss_fn = PolyhedralConnectivityLoss()

        # weights are optional, if not provided, all fields are weighted equally with weight 1.
        if weights is None:
            self.loss_weights = {k: 1.0 for k in self.loss_fns.keys()}
        else:
            assert set(weights.keys()) == set(
                self.loss_fns.keys()
            ), f"weight keys {set(weights.keys())} do not match loss_fns keys {set(self.loss_fns.keys())}"
            self.loss_weights = weights

    def __call__(
        self,
        *,
        multi_corruption: MultiCorruption[T],
        batch: T,
        noisy_batch: T,
        score_model_output: T,
        t: torch.Tensor,
        node_is_unmasked: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch_idx = {k: batch.get_batch_idx(k) for k in self.loss_fns.keys()}
        node_is_unmasked_dict = {k: node_is_unmasked for k in self.loss_fns.keys()}

        # Dict[str, torch.Tensor]
        # Keys are field names and values are loss per sample, with shape (batch_size,).
        loss_per_sample_per_field = apply(
            fns=self.loss_fns,
            corruption=multi_corruption.corruptions,
            x=batch,
            noisy_x=noisy_batch,
            score_model_output=score_model_output,
            batch_idx=batch_idx,
            broadcast=dict(t=t, batch_size=batch.get_batch_size(), batch=batch),
            node_is_unmasked=node_is_unmasked_dict,
        )
        assert set([v.shape for v in loss_per_sample_per_field.values()]) == {
            (batch.get_batch_size(),)
        }, "All losses should have shape (batch_size,)."

        # Aggregate standard diffusion losses per field over samples.
        metrics_dict: Dict[str, float] = {
            k: v.mean().item() for k, v in loss_per_sample_per_field.items()
        }

        # Baseline weighted denoising score matching loss, averaged over samples.
        agg_loss = (
            torch.stack(
                [self.loss_weights[k] * v for k, v in loss_per_sample_per_field.items()], dim=0
            )
            .sum(0)
            .mean()
        )

        # Physics-informed auxiliary losses, evaluated on the denoised structure.
        physics_losses = self._physics_losses(
            multi_corruption=multi_corruption,
            batch=batch,
            noisy_batch=noisy_batch,
            score_model_output=score_model_output,
            t=t,
        )
        for name, (weight, loss) in physics_losses.items():
            agg_loss = agg_loss + weight * loss
            metrics_dict[name] = loss.detach().item()

        return agg_loss, metrics_dict

    def _physics_losses(
        self,
        *,
        multi_corruption: MultiCorruption[T],
        batch: T,
        noisy_batch: T,
        score_model_output: T,
        t: torch.Tensor,
    ) -> Dict[str, Tuple[float, torch.Tensor]]:
        """Evaluate the physics-informed losses on the denoiser's predicted clean structure.

        The score model is trained to predict `-eps` (score times std). For a
        variance-exploding SDE the marginal mean is the clean sample itself, so
        the clean fractional coordinates can be recovered directly:

            x_noisy = x_0 + eps * std   =>   x_0 = x_noisy + std * model_output

        Gradients therefore flow from these losses into the denoiser, which is
        the entire point of the guidance.

        Returns:
            Mapping from metric name to (weight, scalar loss). Empty when
            guidance is disabled or the batch cannot be scored.
        """
        if self.redox_weight == 0.0 and self.poly_weight == 0.0 and self.connectivity_weight == 0.0:
            return {}

        pos_sde = multi_corruption.sdes.get("pos")
        predicted_pos = getattr(score_model_output, "pos", None)
        clean_pos = getattr(batch, "pos", None)
        noisy_pos = getattr(noisy_batch, "pos", None)
        atomic_numbers = getattr(batch, "atomic_numbers", None)
        cell = getattr(batch, "cell", None)
        num_atoms = getattr(batch, "num_atoms", None)

        if any(x is None for x in (pos_sde, predicted_pos, clean_pos, noisy_pos, cell, num_atoms)):
            return {}
        if atomic_numbers is None:
            return {}

        batch_idx = batch.get_batch_idx("pos")
        mean, std = pos_sde.marginal_prob(x=clean_pos, t=t, batch_idx=batch_idx, batch=batch)

        # The reconstruction above assumes a variance-exploding marginal. Bail out
        # rather than silently applying a wrong denoising formula.
        if not torch.allclose(mean.detach(), clean_pos.detach(), atol=1e-5):
            return {}

        predicted_clean_pos = (noisy_pos + std * predicted_pos) % 1.0

        # Fade the guidance out at high noise. At t near 1 the denoised geometry is
        # still essentially random, so bond-valence sums and coordination angles
        # carry no signal and would otherwise swamp the score-matching loss.
        node_weight = (1.0 - t).clamp(min=0.0, max=1.0)[batch_idx]

        # ponytail: the lattice is diffused too, but reconstructing it from the
        # LatticeVPSDE marginal needs the mean coefficient. Both auxiliary losses
        # are distance-based, so using the clean cell still gives a correct
        # gradient path through the coordinates. Revisit if lattice guidance is
        # ever needed.
        graph = build_physics_graph(
            frac_coords=predicted_clean_pos,
            cell=cell,
            atomic_numbers=atomic_numbers,
            num_atoms=num_atoms,
            batch_idx=batch_idx,
            num_graphs=batch.get_batch_size(),
            node_weight=node_weight,
        )
        if graph is None:
            return {}

        losses: Dict[str, Tuple[float, torch.Tensor]] = {}

        for name, weight, loss_fn in (
            ("loss_redox", self.redox_weight, self.redox_loss_fn),
            ("loss_poly", self.poly_weight, self.poly_loss_fn),
            ("loss_connectivity", self.connectivity_weight, self.connectivity_loss_fn),
        ):
            if weight == 0.0:
                continue
            loss = loss_fn(graph)
            if loss is not None:
                losses[name] = (weight, loss)

        return losses


class DenoisingScoreMatchingLoss(SummedFieldLoss):
    def __init__(
        self,
        model_targets: ModelTargets,
        reduce: Literal["sum", "mean"] = "mean",
        weights: Optional[Dict[str, float]] = None,
        field_center_zero: Optional[Dict[str, bool]] = None,  # Whether to zero center each field.
        redox_weight: float = 0.01,
        poly_weight: float = 0.05,
        connectivity_weight: float = 0.05,
    ):
        if field_center_zero is not None:
            assert set(field_center_zero.keys()) == set(model_targets.keys())

        super().__init__(
            loss_fns={
                k: partial(
                    denoising_score_matching,
                    reduce=reduce,
                    model_target=v,
                )
                for k, v in model_targets.items()
            },
            model_targets=model_targets,
            weights=weights,
            redox_weight=redox_weight,
            poly_weight=poly_weight,
            connectivity_weight=connectivity_weight,
        )