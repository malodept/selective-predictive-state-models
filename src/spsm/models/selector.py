from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SelectorCosts:
    cheap_compute: float = 1.0
    optional_compute: float = 3.0
    utility_scale: float = 1.0
    compute_penalty: float = 0.15
    hard_refinement_factor: float = 0.25
    easy_refinement_factor: float = 0.95


class AdaptiveComputeSelector:
    """Threshold selector for optional compute.

    If surprise probability exceeds a threshold, execute an optional expensive module.
    This is intentionally simple: it creates the first utility-vs-compute frontier.
    """

    def __init__(self, threshold: float = 0.5, costs: SelectorCosts | None = None) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        self.threshold = threshold
        self.costs = costs or SelectorCosts()

    def select(self, surprise_prob: torch.Tensor) -> torch.Tensor:
        return surprise_prob >= self.threshold

    def compute_cost(self, surprise_prob: torch.Tensor) -> float:
        selected = self.select(surprise_prob).float()
        return float(self.costs.cheap_compute + self.costs.optional_compute * selected.mean().item())


def utility_for_policy(
    prediction_error: torch.Tensor,
    surprise_prob: torch.Tensor,
    threshold: float,
    costs: SelectorCosts | None = None,
    benefit_label: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute a simple utility proxy for adaptive execution.

    The cheap path receives the actual prediction error. The optional path is a
    controlled proxy for an expensive refinement/expert. In v0.3 the optional
    module is only truly useful on samples with high expected difficulty
    (``benefit_label=1``). Selecting it on easy samples mostly wastes compute.

    This is closer to the real project: selective compute is interesting only if
    optional computation has uneven value across samples.
    """

    costs = costs or SelectorCosts()
    selector = AdaptiveComputeSelector(threshold=threshold, costs=costs)
    selected = selector.select(surprise_prob)
    cheap_error = prediction_error
    if benefit_label is None:
        refinement_factor = torch.full_like(prediction_error, costs.hard_refinement_factor)
    else:
        benefit = benefit_label.to(prediction_error.device).float()
        refinement_factor = torch.where(
            benefit > 0.5,
            torch.full_like(prediction_error, costs.hard_refinement_factor),
            torch.full_like(prediction_error, costs.easy_refinement_factor),
        )
    refined_error = refinement_factor * prediction_error
    effective_error = torch.where(selected, refined_error, cheap_error)
    mean_error = float(effective_error.mean().item())
    mean_compute = float(costs.cheap_compute + costs.optional_compute * selected.float().mean().item())
    utility = float(costs.utility_scale * (-mean_error) - costs.compute_penalty * mean_compute)
    return {
        "threshold": float(threshold),
        "mean_error": mean_error,
        "mean_compute": mean_compute,
        "utility": utility,
        "selected_fraction": float(selected.float().mean().item()),
    }
