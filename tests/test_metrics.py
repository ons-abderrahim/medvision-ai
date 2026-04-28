"""Unit tests for evaluation metrics."""

import torch
import pytest
from src.utils.metrics import compute_auc, compute_f1, compute_ece


def test_compute_auc_perfect():
    labels = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=torch.float32)
    probs = labels.clone()
    auc = compute_auc(labels, probs)
    assert auc == pytest.approx(1.0)


def test_compute_f1_all_correct():
    labels = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
    probs = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    f1 = compute_f1(labels, probs)
    assert f1 == pytest.approx(1.0)


def test_compute_ece_range():
    probs = torch.rand(200)
    labels = (probs > 0.5).float()
    ece = compute_ece(probs, labels)
    assert 0.0 <= ece <= 1.0
