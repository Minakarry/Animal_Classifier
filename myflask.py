"""Flask web service: Top-3, low-confidence warning, Grad-CAM."""
from __future__ import annotations

import base64
import io

from flask import Flask, jsonify, render_template, request
from PIL import Image

from config import (
    CLASS_DESCRIPTIONS,
    CLASS_NAMES,
    CLASS_NAMES_EN,
    CONFIDENCE_THRESHOLD,
    NUM_CLASSES,
    get_class_description,
)
from predict import get_device, load_model, predict_full

app = Flask(__name__)
device = get_device()
model = None


def ensure_model():
    global model
    if model is None:
        model = load_model(device=device)
    return model


@app.route("/")
def index():
    classes = [
        {
            "en": display_name,
            "folder": folder_name,
            "desc": CLASS_DESCRIPTIONS.get(display_name, ""),
        }
        for display_name, folder_name in zip(CLASS_NAMES, CLASS_NAMES_EN)
    ]
    return render_template(
        "index.html",
        classes=classes,
        num_classes=NUM_CLASSES,
        threshold=CONFIDENCE_THRESHOLD,
    )


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No uploaded file found"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "Please select an image first"}), 400

    try:
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return jsonify({"error": "Cannot read the image. Please try another file."}), 400

    preview_buf = io.BytesIO()
    img.save(preview_buf, format="JPEG", quality=90)
    preview_b64 = base64.b64encode(preview_buf.getvalue()).decode("utf-8")

    try:
        ensure_model()
        result = predict_full(image=img, model=model, device=device, topk=3, with_cam=True)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:
        return jsonify({"error": f"Recognition failed: {exc}"}), 500

    top1 = result["top1"]
    return jsonify({
        "result": top1["name"],
        "confidence": top1["confidence"],
        "description": get_class_description(top1["name"]),
        "topk": result["topk"],
        "uncertain": result["uncertain"],
        "message": result["message"],
        "threshold": result["threshold"],
        "preview": f"data:image/jpeg;base64,{preview_b64}",
        "gradcam": result.get("gradcam"),
    })


if __name__ == "__main__":
    try:
        ensure_model()
        print(f"Model loaded on device: {device}")
    except FileNotFoundError as exc:
        print(f"Warning: {exc}")
    app.run(host="127.0.0.1", port=5000, debug=True)
