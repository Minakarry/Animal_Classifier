"""单图推理。"""
from __future__ import annotations

from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet18

from config import CLASS_NAMES, IMG_SIZE, MEAN, STD, MODEL_PATH, NUM_CLASSES

normalize = transforms.Normalize(mean=MEAN, std=STD)
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    normalize,
])

_model = None
_device = None


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_path=None, device=None):
    global _model, _device
    device = device or get_device()
    model_path = Path(model_path) if model_path else MODEL_PATH
    if not model_path.exists():
        legacy = Path("output/resnet18_e_best.pth")
        if legacy.exists():
            model_path = legacy
        else:
            raise FileNotFoundError(
                f"未找到模型文件: {MODEL_PATH}\n请先运行: python train.py"
            )

    model = resnet18(num_classes=NUM_CLASSES)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    _model = model
    _device = device
    return model


def preprocess(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB")
    tensor = val_tf(img)
    return torch.unsqueeze(tensor, 0)


@torch.no_grad()
def predict_topk(img_path=None, image=None, model=None, device=None, topk=3):
    """
    预测一张图片，返回 Top-K。
    返回: {
        "top1": {"id", "name", "confidence"},
        "topk": [{"id", "name", "confidence"}, ...],
    }
    """
    device = device or _device or get_device()
    model = model or _model or load_model(device=device)

    if image is not None:
        img = image
    else:
        if img_path is None:
            img_path = "images/target.png"
        img = Image.open(img_path)

    img_tensor = preprocess(img).to(device)
    logits = model(img_tensor)
    probs = torch.softmax(logits, dim=1)[0]
    k = min(topk, probs.numel())
    values, indices = torch.topk(probs, k)

    topk_list = []
    for conf, idx in zip(values.tolist(), indices.tolist()):
        topk_list.append({
            "id": int(idx),
            "name": CLASS_NAMES[int(idx)],
            "confidence": float(conf),
        })

    return {"top1": topk_list[0], "topk": topk_list}


@torch.no_grad()
def predict(img_path=None, model=None, device=None):
    """兼容旧接口: (class_index, class_name_zh, confidence)"""
    result = predict_topk(img_path=img_path, model=model, device=device, topk=1)
    item = result["top1"]
    return item["id"], item["name"], item["confidence"]


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "images/horse.png"
    out = predict_topk(path, topk=3)
    print(f"Top1: {out['top1']['name']} ({out['top1']['confidence']:.4f})")
    for i, item in enumerate(out["topk"], 1):
        print(f"  #{i} {item['name']}: {item['confidence'] * 100:.1f}%")
