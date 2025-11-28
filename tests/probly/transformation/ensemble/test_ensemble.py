from __future__ import annotations

import pytest
import torch
from torch import nn

from probly.predictor import Predictor
from probly.transformation.ensemble import ensemble


@pytest.fixture
def dummy_model() -> nn.Module:
    class DummyNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(5, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

    return DummyNet()


@pytest.fixture
def sample_data() -> torch.Tensor:
    return torch.randn(3, 5)


def test_ensemble_with_zero_members_returns_empty_list(dummy_model: nn.Module) -> None:
    invalid_size = 0
    transformed_model = ensemble(dummy_model, n_members=invalid_size)

    assert isinstance(transformed_model, nn.ModuleList)
    assert len(transformed_model) == 0


def test_ensemble_transformation_returns_predictor(dummy_model: nn.Module) -> None:
    n_members = 4
    transformed_model = ensemble(dummy_model, n_members=n_members)

    assert isinstance(transformed_model, Predictor)
    assert transformed_model is not dummy_model


def test_ensemble_returns_modulelist_of_correct_size(dummy_model: nn.Module) -> None:
    n_members = 5
    transformed_model = ensemble(dummy_model, n_members=n_members)

    assert isinstance(transformed_model, nn.ModuleList)
    assert len(transformed_model) == n_members
    assert isinstance(transformed_model[0], type(dummy_model))
