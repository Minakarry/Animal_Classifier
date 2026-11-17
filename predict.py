"""Single-image inference, Top-K, low-confidence flag, Grad-CAM."""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet18

from config import (
    CLASS_NAMES,
    CONFIDENCE_THRESHOLD,
    IMG_SIZE,
    MEAN,
    MODEL_PATH,
    NUM_CLASSES,
    STD,
)

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
                f"Model not found: {MODEL_PATH}\nPlease run: python train.py"
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


def _open_image(img_path=None, image=None) -> Image.Image:
    if image is not None:
        return image.convert("RGB")
    if img_path is None:
        img_path = "images/horse.png"
    return Image.open(img_path).convert("RGB")


def predict_topk(img_path=None, image=None, model=None, device=None, topk=3):
    """
    Returns:
      {
        top1, topk,
        uncertain: bool,
        threshold: float,
        message: str,
      }
    """
    device = device or _device or get_device()
    model = model or _model or load_model(device=device)
    img = _open_image(img_path, image)

    img_tensor = preprocess(img).to(device)
    with torch.no_grad():
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

    top1 = topk_list[0]
    uncertain = top1["confidence"] < CONFIDENCE_THRESHOLD
    if uncertain:
        message = (
            f"Low confidence ({top1['confidence'] * 100:.1f}% < "
            f"{CONFIDENCE_THRESHOLD * 100:.0f}%). The prediction may be unreliable "
            "or the animal may be outside the 10 supported classes."
        )
    else:
        message = "Prediction confidence is acceptable."

    return {
        "top1": top1,
        "topk": topk_list,
        "uncertain": uncertain,
        "threshold": CONFIDENCE_THRESHOLD,
        "message": message,
    }


def predict(img_path=None, model=None, device=None):
    result = predict_topk(img_path=img_path, model=model, device=device, topk=1)
    item = result["top1"]
    return item["id"], item["name"], item["confidence"]


def generate_gradcam(img_path=None, image=None, model=None, device=None, target_class=None):
    """
    Grad-CAM over ResNet18 layer4.
    Returns PIL image (original with heatmap overlay).
    """
    device = device or _device or get_device()
    model = model or _model or load_model(device=device)
    model.eval()

    img = _open_image(img_path, image)
    img_tensor = preprocess(img).to(device)
    img_tensor.requires_grad_(True)

    activations = []
    gradients = []

    def fwd_hook(_module, _inp, out):
        activations.append(out)

    def bwd_hook(_module, _gin, gout):
        gradients.append(gout[0])

    handle_f = model.layer4.register_forward_hook(fwd_hook)
    handle_b = model.layer4.register_full_backward_hook(bwd_hook)

    try:
        logits = model(img_tensor)
        if target_class is None:
            target_class = int(logits.argmax(1).item())
        score = logits[0, target_class]
        model.zero_grad(set_to_none=True)
        if img_tensor.grad is not None:
            img_tensor.grad.zero_()
        score.backward()

        act = activations[-1][0].detach()          # C,H,W
        grad = gradients[-1][0].detach()           # C,H,W
        weights = grad.mean(dim=(1, 2))            # C
        cam = torch.relu((weights[:, None, None] * act).sum(dim=0))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(
            cam[None, None],
            size=(img.height, img.width),
            mode="bilinear",
            align_corners=False,
        )[0, 0].cpu().numpy()
    finally:
        handle_f.remove()
        handle_b.remove()

    heatmap = plt_colormap(cam)
    overlay = (0.55 * np.asarray(img).astype(np.float32) + 0.45 * heatmap).clip(0, 255)
    return Image.fromarray(overlay.astype(np.uint8))


def plt_colormap(cam: np.ndarray) -> np.ndarray:
    """Simple jet-like colormap without requiring matplotlib for inference path."""
    # Use OpenCV if available for better colormap; fallback to manual red-blue mix.
    try:
        import cv2

        heat = np.uint8(255 * cam)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32)
    except Exception:
        r = cam
        g = np.zeros_like(cam)
        b = 1.0 - cam
        heat = np.stack([r, g, b], axis=-1)
        return np.uint8(255 * heat).astype(np.float32)


def gradcam_to_base64(pil_img: Image.Image, quality: int = 90) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def predict_full(img_path=None, image=None, model=None, device=None, topk=3, with_cam=True):
    """Convenience for UI: topk + uncertain + optional gradcam."""
    img = _open_image(img_path, image)
    result = predict_topk(image=img, model=model, device=device, topk=topk)
    cam_b64 = None
    if with_cam:
        cam_img = generate_gradcam(
            image=img,
            model=model or _model,
            device=device,
            target_class=result["top1"]["id"],
        )
        cam_b64 = gradcam_to_base64(cam_img)
        result["gradcam_image"] = cam_img
        result["gradcam"] = cam_b64
    return result


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "images/horse.png"
    out = predict_full(path, topk=3, with_cam=True)
    print(f"Top1: {out['top1']['name']} ({out['top1']['confidence']:.4f})")
    print(f"Uncertain: {out['uncertain']} | {out['message']}")
    for i, item in enumerate(out["topk"], 1):
        print(f"  #{i} {item['name']}: {item['confidence'] * 100:.1f}%")
    if out.get("gradcam_image") is not None:
        save = Path("output/figures/gradcam_demo.jpg")
        save.parent.mkdir(parents=True, exist_ok=True)
        out["gradcam_image"].save(save)
        print(f"Grad-CAM saved: {save}")
