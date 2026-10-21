"""Flask Web 端动物识别服务。"""
from __future__ import annotations

import base64
import io

from flask import Flask, jsonify, render_template, request
from PIL import Image

from config import (
    CLASS_DESCRIPTIONS,
    CLASS_NAMES,
    CLASS_NAMES_EN,
    NUM_CLASSES,
    get_class_description,
)
from predict import get_device, load_model, predict_topk

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
            "zh": zh,
            "en": en,
            "desc": CLASS_DESCRIPTIONS.get(zh, ""),
        }
        for zh, en in zip(CLASS_NAMES, CLASS_NAMES_EN)
    ]
    return render_template(
        "index.html",
        classes=classes,
        num_classes=NUM_CLASSES,
    )


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "未找到上传文件"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "请先选择图片"}), 400

    try:
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return jsonify({"error": "图片无法读取，请换一张试试"}), 400

    preview_buf = io.BytesIO()
    img.save(preview_buf, format="JPEG", quality=90)
    preview_b64 = base64.b64encode(preview_buf.getvalue()).decode("utf-8")

    try:
        ensure_model()
        result = predict_topk(image=img, model=model, device=device, topk=1)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:
        return jsonify({"error": f"识别失败: {exc}"}), 500

    top1 = result["top1"]
    return jsonify({
        "result": top1["name"],
        "confidence": top1["confidence"],
        "description": get_class_description(top1["name"]),
        "preview": f"data:image/jpeg;base64,{preview_b64}",
    })


if __name__ == "__main__":
    try:
        ensure_model()
        print(f"模型已加载，设备: {device}")
    except FileNotFoundError as exc:
        print(f"警告: {exc}")
    app.run(host="127.0.0.1", port=5000, debug=True)
