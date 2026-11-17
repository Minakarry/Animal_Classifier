"""Generate paper figures that can be auto-inserted (Grad-CAM examples + architecture)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from config import FIGURES_DIR, ROOT
from predict import predict_full


def _load(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def make_gradcam_panel():
    demo = ROOT / "images" / "demo_for_paper"
    success = demo / "01_success_dog_100pct.jpeg"
    fail = demo / "03_fail_true-dog_pred-Elephant_90pct.jpeg"
    topk = demo / "02_topk_sheep_vs_Chicken_67pct.jpeg"
    if not success.exists():
        success = next(demo.glob("01_success_dog*"))
    if not fail.exists():
        fail = next(demo.glob("03_fail_*"))
    if not topk.exists():
        topk = next(demo.glob("02_topk_*"))

    cases = [
        ("Success: Dog (high confidence)", success),
        ("Ambiguous Top-3: Sheep vs Chicken", topk),
        ("Failure: true Dog, predicted Elephant", fail),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(9.5, 11.5))
    for row, (title, path) in enumerate(cases):
        out = predict_full(str(path), topk=3, with_cam=True)
        orig = _load(path)
        cam = out["gradcam_image"]
        top1 = out["top1"]
        topk_str = ", ".join(
            f"{t['name']} {t['confidence'] * 100:.1f}%" for t in out["topk"]
        )
        axes[row, 0].imshow(orig)
        axes[row, 0].set_title(f"{title}\nInput", fontsize=10)
        axes[row, 0].axis("off")
        axes[row, 1].imshow(cam)
        axes[row, 1].set_title(
            f"Grad-CAM\nPred: {top1['name']} ({top1['confidence'] * 100:.1f}%)\n{topk_str}",
            fontsize=9,
        )
        axes[row, 1].axis("off")

    fig.suptitle("Qualitative examples: success / Top-3 ambiguity / failure", fontsize=12, y=0.995)
    fig.tight_layout()
    out_path = FIGURES_DIR / "gradcam_examples.png"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved", out_path)
    return out_path


def make_architecture():
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (0.3, 1.3, 2.2, 1.6, "Data\nAnimals-10\ntrain/test split"),
        (3.0, 1.3, 2.4, 1.6, "Training\nResNet18 + Adam\nearly stopping"),
        (5.9, 1.3, 2.4, 1.6, "Inference\nTop-3 / threshold\nGrad-CAM"),
        (8.8, 2.15, 2.6, 1.2, "Desktop UI\nPyQt5"),
        (8.8, 0.55, 2.6, 1.2, "Web UI\nFlask"),
    ]
    for x, y, w, h, text in boxes:
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.03,rounding_size=0.15",
                linewidth=1.2,
                edgecolor="#1f4d3d",
                facecolor="#e8f2ec",
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    for x0, x1, y in [(2.5, 3.0, 2.1), (5.4, 5.9, 2.1)]:
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="->", color="#1f4d3d", lw=1.5))
    ax.annotate("", xy=(8.8, 2.75), xytext=(8.3, 2.3),
                arrowprops=dict(arrowstyle="->", color="#1f4d3d", lw=1.4))
    ax.annotate("", xy=(8.8, 1.15), xytext=(8.3, 1.8),
                arrowprops=dict(arrowstyle="->", color="#1f4d3d", lw=1.4))

    ax.set_title("System architecture overview", fontsize=12, pad=8)
    out_path = FIGURES_DIR / "system_architecture.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved", out_path)
    return out_path


if __name__ == "__main__":
    make_architecture()
    make_gradcam_panel()
