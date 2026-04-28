"""
MedVision AI — Gradio Interactive Demo

Launches a Gradio web interface for:
  - Chest X-ray multi-label classification (CheXpert / DenseNet-121)
  - Skin lesion classification (ISIC / EfficientNet-B4)
  - Grad-CAM heatmap visualisation

Run:
    python app.py
    # → http://localhost:7860
"""

from __future__ import annotations

import io
from pathlib import Path

import gradio as gr
import torch
from PIL import Image

from src.models.densenet import DenseNetChexpert
from src.models.efficientnet import EfficientNetISIC
from src.explainability.gradcam import GradCAMExplainer
from src.data.transforms import get_xray_transforms, get_val_transforms

# ──────────────────────────────────────────────────────────────
# Load models (gracefully handle missing checkpoints)
# ──────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = {}

_ckpt_xray = Path("checkpoints/densenet_chexpert/best.ckpt")
if _ckpt_xray.exists():
    MODELS["chexray"] = {
        "model": DenseNetChexpert.load(str(_ckpt_xray), device=str(DEVICE)),
        "transform": get_xray_transforms(320, "valid"),
        "layer": "features.denseblock4",
    }

_ckpt_isic = Path("checkpoints/efficientnet_isic/best.ckpt")
if _ckpt_isic.exists():
    MODELS["dermoscopy"] = {
        "model": EfficientNetISIC.load(str(_ckpt_isic), device=str(DEVICE)),
        "transform": get_val_transforms(380),
        "layer": "model.features",
    }

for key, info in MODELS.items():
    info["model"].eval()
    info["explainer"] = GradCAMExplainer(info["model"], info["layer"])


# ──────────────────────────────────────────────────────────────
# Inference logic
# ──────────────────────────────────────────────────────────────

def run_inference(image: Image.Image, modality: str, show_gradcam: bool):
    if modality not in MODELS:
        return "⚠️ Model not loaded. Run training first.", None

    info = MODELS[modality]
    transform = info["transform"]
    model = info["model"]
    explainer = info["explainer"]

    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()

    labels = model.label_names
    results = sorted(zip(labels, probs), key=lambda x: -x[1])

    # Format label output as markdown table
    table = "| Condition | Confidence |\n|---|---|\n"
    for label, prob in results:
        bar = "█" * int(prob * 20)
        table += f"| {label} | {prob:.1%} {bar} |\n"

    heatmap_img = None
    if show_gradcam:
        heatmap_img = explainer.explain(tensor, orig_size=image.size)

    return table, heatmap_img


# ──────────────────────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────────────────────

DESCRIPTION = """
## 🩺 MedVision AI — Medical Image Analysis

**Decision-support tool for X-ray and dermoscopy classification.**

> ⚠️ **Not a diagnostic device.** All outputs must be reviewed by a qualified clinician.

Upload a medical image, choose the modality, and optionally enable Grad-CAM to visualise
which regions influenced the prediction.
"""

with gr.Blocks(title="MedVision AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="Upload Image")
            modality = gr.Radio(
                choices=["chexray", "dermoscopy"],
                value="chexray",
                label="Imaging Modality",
            )
            show_cam = gr.Checkbox(value=True, label="Show Grad-CAM Heatmap")
            run_btn = gr.Button("Analyse", variant="primary")

        with gr.Column(scale=1):
            output_table = gr.Markdown(label="Predictions")
            output_heatmap = gr.Image(label="Grad-CAM Heatmap", type="pil")

    run_btn.click(
        fn=run_inference,
        inputs=[img_input, modality, show_cam],
        outputs=[output_table, output_heatmap],
    )

    gr.Examples(
        examples=[
            ["assets/sample_xray.jpg", "chexray", True],
            ["assets/sample_lesion.jpg", "dermoscopy", True],
        ],
        inputs=[img_input, modality, show_cam],
        outputs=[output_table, output_heatmap],
        fn=run_inference,
        cache_examples=False,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
