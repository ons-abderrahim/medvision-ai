"""
MedVision AI — FastAPI inference server.

Endpoints:
    POST /predict              → run inference, return predictions + confidence
    POST /explain/gradcam      → return Grad-CAM heatmap (PNG)
    POST /explain/shap         → return SHAP attribution map (PNG)
    GET  /models               → list available models
    GET  /health               → health check

Start with:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

from src.api.schemas import HealthResponse, ModelListResponse, PredictionResponse
from src.models.densenet import DenseNetChexpert
from src.models.efficientnet import EfficientNetISIC
from src.explainability.gradcam import GradCAMExplainer
from src.data.transforms import get_val_transforms, get_xray_transforms

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Model registry — populated at startup
# ──────────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, dict] = {
    "densenet_chexpert": {
        "cls": DenseNetChexpert,
        "ckpt": "checkpoints/densenet_chexpert/best.ckpt",
        "gradcam_layer": "features.denseblock4",
        "transforms": lambda: get_xray_transforms(320, "valid"),
        "modality": "chexray",
        "description": "DenseNet-121 fine-tuned on CheXpert (14 pathologies)",
    },
    "efficientnet_isic": {
        "cls": EfficientNetISIC,
        "ckpt": "checkpoints/efficientnet_isic/best.ckpt",
        "gradcam_layer": "model.features",
        "transforms": lambda: get_val_transforms(380),
        "modality": "dermoscopy",
        "description": "EfficientNet-B4 fine-tuned on ISIC 2020 (8 lesion types)",
    },
}

loaded_models: dict[str, torch.nn.Module] = {}
loaded_explainers: dict[str, GradCAMExplainer] = {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models into memory at startup."""
    for name, info in MODEL_REGISTRY.items():
        ckpt = Path(info["ckpt"])
        if ckpt.exists():
            model = info["cls"].load(str(ckpt), device=str(DEVICE))
            model.to(DEVICE).eval()
            loaded_models[name] = model
            loaded_explainers[name] = GradCAMExplainer(model, info["gradcam_layer"])
            logger.info(f"Loaded model: {name}")
        else:
            logger.warning(f"Checkpoint not found for {name}: {ckpt}")
    yield
    loaded_models.clear()
    loaded_explainers.clear()


# ──────────────────────────────────────────────────────────────
# App factory
# ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="MedVision AI",
    description="AI-assisted medical image analysis — decision support only, not diagnosis.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _load_image(file: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(file)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")


def _preprocess(image: Image.Image, model_name: str) -> torch.Tensor:
    transform = MODEL_REGISTRY[model_name]["transforms"]()
    return transform(image).unsqueeze(0).to(DEVICE)


# ──────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return {"status": "ok", "loaded_models": list(loaded_models.keys()), "device": str(DEVICE)}


@app.get("/models", response_model=ModelListResponse)
async def list_models():
    return {
        "models": [
            {
                "name": name,
                "description": info["description"],
                "modality": info["modality"],
                "loaded": name in loaded_models,
            }
            for name, info in MODEL_REGISTRY.items()
        ]
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form("densenet_chexpert"),
):
    if model_name not in loaded_models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not loaded.")

    raw = await file.read()
    image = _load_image(raw)
    tensor = _preprocess(image, model_name)

    model = loaded_models[model_name]
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()

    labels = model.label_names
    predictions = [
        {"label": label, "confidence": round(prob, 4)}
        for label, prob in sorted(zip(labels, probs), key=lambda x: -x[1])
    ]

    return {
        "predictions": predictions,
        "model": model_name,
        "top_label": predictions[0]["label"],
        "top_confidence": predictions[0]["confidence"],
    }


@app.post("/explain/gradcam")
async def explain_gradcam(
    file: UploadFile = File(...),
    model_name: str = Form("densenet_chexpert"),
    class_idx: int = Form(None),
):
    if model_name not in loaded_explainers:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not loaded.")

    raw = await file.read()
    image = _load_image(raw)
    tensor = _preprocess(image, model_name)
    orig_size = image.size  # (W, H)

    explainer = loaded_explainers[model_name]
    heatmap: Image.Image = explainer.explain(tensor, class_idx=class_idx, orig_size=orig_size)

    buf = io.BytesIO()
    heatmap.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.post("/explain/shap")
async def explain_shap(
    file: UploadFile = File(...),
    model_name: str = Form("densenet_chexpert"),
    class_idx: int = Form(None),
):
    """SHAP attribution — requires shap library and a preloaded background dataset."""
    raise HTTPException(
        status_code=501,
        detail="SHAP endpoint requires background dataset. Use the Python API directly.",
    )
