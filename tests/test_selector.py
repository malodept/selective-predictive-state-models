import torch

from spsm.models.selector import AdaptiveComputeSelector, SelectorCosts, utility_for_policy


def test_selector_threshold():
    selector = AdaptiveComputeSelector(threshold=0.5)
    probs = torch.tensor([0.1, 0.5, 0.9])
    selected = selector.select(probs)
    assert selected.tolist() == [False, True, True]


def test_utility_for_policy_keys():
    errors = torch.tensor([1.0, 2.0, 3.0])
    probs = torch.tensor([0.1, 0.8, 0.9])
    row = utility_for_policy(errors, probs, threshold=0.5, costs=SelectorCosts())
    assert {"threshold", "mean_error", "mean_compute", "utility", "selected_fraction"} <= set(row)
    assert 0 <= row["selected_fraction"] <= 1
