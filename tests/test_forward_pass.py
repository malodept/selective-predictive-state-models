import torch

from spsm.models.predictor import PredictiveStateModel


def test_forward_pass_shapes():
    model = PredictiveStateModel(latent_dim=16, action_dim=4, hidden_dim=32)
    z = torch.randn(8, 16)
    a = torch.randn(8, 4)
    out = model(z, a)
    assert out["z_pred"].shape == (8, 16)
    assert out["reliability_logit"].shape == (8,)
