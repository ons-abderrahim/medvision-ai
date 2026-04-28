"""Unit tests for MedVision model classes."""

import pytest
import torch

from src.models.densenet import DenseNetChexpert
from src.models.efficientnet import EfficientNetISIC


# ──────────────────────────────────────────────────────────────
# DenseNet
# ──────────────────────────────────────────────────────────────

class TestDenseNetChexpert:
    @pytest.fixture
    def model(self):
        return DenseNetChexpert(num_classes=14, pretrained=False)

    def test_output_shape(self, model):
        x = torch.randn(2, 3, 320, 320)
        logits = model(x)
        assert logits.shape == (2, 14)

    def test_predict_proba_range(self, model):
        x = torch.randn(1, 3, 320, 320)
        probs = model.predict_proba(x)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_label_count(self, model):
        assert len(model.label_names) == 14

    def test_save_load(self, model, tmp_path):
        path = str(tmp_path / "test.ckpt")
        model.save(path)
        loaded = DenseNetChexpert.load(path)
        x = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            out_orig = model(x)
            out_loaded = loaded(x)
        assert torch.allclose(out_orig, out_loaded, atol=1e-5)


# ──────────────────────────────────────────────────────────────
# EfficientNet
# ──────────────────────────────────────────────────────────────

class TestEfficientNetISIC:
    @pytest.fixture
    def model(self):
        return EfficientNetISIC(num_classes=8, pretrained=False)

    def test_output_shape(self, model):
        x = torch.randn(2, 3, 380, 380)
        logits = model(x)
        assert logits.shape == (2, 8)

    def test_predict_proba_sums_to_one(self, model):
        x = torch.randn(1, 3, 380, 380)
        probs = model.predict_proba(x)
        assert abs(probs.sum().item() - 1.0) < 1e-5

    def test_label_count(self, model):
        assert len(model.label_names) == 8
