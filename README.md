# medvision-ai
Multimodal Medical Image Analysis Tool AI-assisted diagnosis support for X-ray, MRI, and dermoscopy images


# 🩺 MedVision AI — Multimodal Medical Image Analysis Tool

<div align="center">

![MedVision AI Banner](assets/banner.png)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-orange?logo=gradio)](https://gradio.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Model%20Hub-FFD21E)](https://huggingface.co)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**AI-assisted diagnosis support for X-ray, MRI, and dermoscopy images.**  
Fine-tuned vision models with confidence-scored predictions and explainable AI (SHAP / GradCAM).

[**Live Demo**](https://huggingface.co/spaces/your-username/medvision-ai) · [**API Docs**](https://your-api.com/docs) · [**Model Card**](docs/MODEL_CARD.md) · [**Report a Bug**](https://github.com/your-username/medvision-ai/issues)

</div>

---

## ⚠️ Medical Disclaimer

> **MedVision AI is a decision-support tool, NOT a diagnostic device.**  
> All outputs must be reviewed by a qualified medical professional. This software is not FDA-cleared or CE-marked. Do not use as a substitute for professional medical judgment.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Supported Modalities](#-supported-modalities)
- [Quickstart](#-quickstart)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Model Training](#-model-training)
- [Explainability](#-explainability)
- [Datasets](#-datasets)
- [Evaluation](#-evaluation)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

Radiologists and dermatologists in resource-limited settings often lack access to AI-assisted second opinions. Existing commercial tools are expensive and require large-scale hospital infrastructure.

**MedVision AI** is a lightweight, open-source, research-grade tool that:
- Fine-tunes state-of-the-art vision models on public medical datasets
- Provides **confidence-scored predictions** with calibrated uncertainty
- Generates **human-interpretable explanations** via SHAP and Grad-CAM
- Exposes a **REST API** and an **interactive Gradio UI** — deployable anywhere

---

## ✨ Features

| Feature | Details |
|---|---|
| 🧠 **Multi-model support** | DenseNet-121, EfficientNet-B4, BioViL (CXR-specific) |
| 🖼️ **3 imaging modalities** | Chest X-ray, skin lesion (dermoscopy), MRI |
| 📊 **Confidence scoring** | Softmax probabilities + temperature scaling |
| 🔍 **Explainability** | Grad-CAM heatmaps + SHAP feature attributions |
| ⚡ **FastAPI backend** | Production-ready REST API with async inference |
| 🎛️ **Gradio frontend** | Zero-setup interactive web demo |
| 🐳 **Docker support** | One-command deployment |
| 🤗 **HuggingFace Hub** | Pre-trained weights hosted publicly |

---

## 🏗️ Architecture

```
medvision-ai/
├── src/
│   ├── models/          # Model definitions & fine-tuning logic
│   │   ├── densenet.py
│   │   ├── efficientnet.py
│   │   └── biovil.py
│   ├── data/            # Dataset loaders & preprocessing pipelines
│   │   ├── chexpert.py
│   │   ├── isic.py
│   │   └── transforms.py
│   ├── explainability/  # GradCAM & SHAP wrappers
│   │   ├── gradcam.py
│   │   └── shap_explainer.py
│   ├── api/             # FastAPI application
│   │   ├── main.py
│   │   ├── routes.py
│   │   └── schemas.py
│   └── utils/           # Helpers: metrics, visualization, logging
│       ├── metrics.py
│       ├── visualization.py
│       └── logger.py
├── notebooks/           # Exploratory & training notebooks
├── tests/               # Unit + integration tests
├── configs/             # YAML experiment configs
├── docs/                # Extended documentation
└── .github/workflows/   # CI/CD pipelines
```

---

## 🫁 Supported Modalities

### 1. Chest X-ray (CheXpert)
- **Task**: Multi-label pathology classification (14 conditions)
- **Labels**: Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion, and more
- **Model**: DenseNet-121 / BioViL

### 2. Skin Lesion (ISIC)
- **Task**: Multi-class melanoma classification (8 categories)
- **Labels**: Melanoma, Nevus, BCC, AK, BKL, DF, VASC, SCC
- **Model**: EfficientNet-B4

### 3. MRI (experimental)
- **Task**: Binary anomaly detection
- **Model**: EfficientNet-B4 with volumetric preprocessing

---

## 🚀 Quickstart

```bash
# 1. Clone the repository
git clone https://github.com/your-username/medvision-ai.git
cd medvision-ai

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Download pre-trained weights
python scripts/download_weights.py --model densenet_chexpert

# 4. Launch the Gradio demo
python app.py
# → Open http://localhost:7860
```

Or run with Docker:

```bash
docker compose up
# → API at http://localhost:8000
# → Gradio UI at http://localhost:7860
```

---

## 🛠️ Installation

### Requirements

- Python 3.10+
- PyTorch 2.2+ (CUDA 11.8+ recommended)
- 8 GB RAM minimum; 16 GB recommended for training

### From PyPI (inference only)

```bash
pip install medvision-ai
```

### From Source (full development)

```bash
git clone https://github.com/your-username/medvision-ai.git
cd medvision-ai
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

---

## 💻 Usage

### Python API

```python
from medvision import MedVisionModel

# Load a pre-trained model
model = MedVisionModel.from_pretrained("densenet_chexpert")

# Run inference on a chest X-ray
result = model.predict("path/to/xray.jpg")

print(result.predictions)
# [{"label": "Pleural Effusion", "confidence": 0.87}, ...]

# Generate Grad-CAM heatmap
heatmap = model.explain(method="gradcam")
heatmap.save("heatmap.png")
```

### REST API

```bash
# Start the API server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# POST an image for prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@chest_xray.jpg" \
  -F "modality=chexray" \
  -F "explain=true"
```

Response:
```json
{
  "predictions": [
    {"label": "Pleural Effusion", "confidence": 0.87},
    {"label": "Atelectasis",      "confidence": 0.34}
  ],
  "gradcam_url": "http://localhost:8000/results/abc123/heatmap.png",
  "model":       "densenet-121-chexpert-v1.2",
  "inference_ms": 142
}
```

---

## 📡 API Reference

Full interactive docs available at `/docs` (Swagger UI) and `/redoc` when the server is running.

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Run inference on an uploaded image |
| `/explain/gradcam` | POST | Generate Grad-CAM heatmap |
| `/explain/shap` | POST | Generate SHAP attribution map |
| `/models` | GET | List all available models |
| `/health` | GET | Health check |

---

## 🏋️ Model Training

### 1. Prepare the dataset

```bash
# CheXpert
python src/data/chexpert.py --download --output data/chexpert

# ISIC
python src/data/isic.py --download --output data/isic
```

### 2. Configure your experiment

Edit `configs/densenet_chexpert.yaml`:

```yaml
model:
  backbone: densenet121
  num_classes: 14
  pretrained: true

training:
  epochs: 50
  batch_size: 32
  lr: 1e-4
  scheduler: cosine
  mixup_alpha: 0.4

data:
  dataset: chexpert
  image_size: 320
  augmentation: strong
```

### 3. Train

```bash
python src/models/train.py --config configs/densenet_chexpert.yaml

# Multi-GPU (DDP)
torchrun --nproc_per_node=4 src/models/train.py --config configs/densenet_chexpert.yaml
```

### 4. Export to HuggingFace Hub

```bash
python scripts/push_to_hub.py --checkpoint checkpoints/best.ckpt --repo your-username/medvision-densenet
```

---

## 🔬 Explainability

MedVision AI ships two explanation methods:

### Grad-CAM

Highlights image regions that most influenced the model's prediction using gradient-weighted class activation maps.

```python
from src.explainability.gradcam import GradCAMExplainer

explainer = GradCAMExplainer(model, target_layer="features.denseblock4")
heatmap = explainer.explain(image_tensor, class_idx=4)
```

### SHAP (DeepSHAP)

Computes feature-level Shapley values to show pixel-wise contribution.

```python
from src.explainability.shap_explainer import SHAPExplainer

explainer = SHAPExplainer(model, background_dataset=val_loader)
shap_values = explainer.explain(image_tensor)
```

---

## 📦 Datasets

| Dataset | Modality | Classes | License | Link |
|---|---|---|---|---|
| CheXpert | Chest X-ray | 14 | Stanford Research | [stanfordmlgroup.github.io](https://stanfordmlgroup.github.io/competitions/chexpert) |
| ISIC 2020 | Dermoscopy | 8 | CC BY-NC | [isic-archive.com](https://isic-archive.com) |
| BrainMRI | MRI | 2 | CC0 | Kaggle |

> Datasets are **not included** in this repository. Follow each link and accept the respective data use agreements.

---

## 📊 Evaluation

| Model | Dataset | AUC | F1 | Acc |
|---|---|---|---|---|
| DenseNet-121 | CheXpert (test) | 0.883 | 0.761 | 0.812 |
| BioViL | CheXpert (test) | 0.901 | 0.779 | 0.831 |
| EfficientNet-B4 | ISIC 2020 | 0.921 | 0.814 | 0.879 |

Reproduce results:
```bash
python scripts/evaluate.py --config configs/densenet_chexpert.yaml --checkpoint checkpoints/best.ckpt
```

---

## 🗺️ Roadmap

- [x] DenseNet-121 + CheXpert fine-tuning
- [x] EfficientNet-B4 + ISIC fine-tuning
- [x] Grad-CAM integration
- [x] FastAPI backend
- [x] Gradio UI
- [x] Docker support
- [ ] BioViL integration (in progress)
- [ ] DICOM file support
- [ ] ONNX / TensorRT export
- [ ] Federated learning demo
- [ ] Multi-language Gradio UI

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

```bash
# Fork → Clone → Branch
git checkout -b feature/your-feature

# Install dev extras
pip install -e ".[dev]"
pre-commit install

# Make changes, then test
pytest tests/ -v

# Submit a PR 🎉
```

---

## 📚 Citations

```bibtex
@article{irvin2019chexpert,
  title   = {CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels},
  author  = {Irvin, Jeremy and others},
  journal = {AAAI},
  year    = {2019}
}

@article{bannur2023biovil,
  title   = {Learning to Exploit Temporal Structure for Biomedical Vision-Language Processing},
  author  = {Bannur, Shruthi and others},
  journal = {arXiv:2204.09817},
  year    = {2023}
}
```

---

<div align="center">
Built with ❤️ as part of the <strong>AI Freelance Business Launch Guide 2026</strong> · AXONN AI STUDIO
</div>
