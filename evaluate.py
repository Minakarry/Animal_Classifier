"""
Evaluate the trained model on the test set.

Outputs:
  - output/metrics.json
  - output/figures/confusion_matrix.png
  - output/figures/per_class_f1.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from config import (
    BATCH_SIZE,
    CLASS_NAMES,
    FIGURES_DIR,
    METRICS_PATH,
    MODEL_PATH,
    NUM_CLASSES,
    OUTPUT_DIR,
    TEST_LIST,
)
from seed_utils import set_seed
from utils import LoadData


def load_eval_model(device: str):
    path = MODEL_PATH if MODEL_PATH.exists() else OUTPUT_DIR / "resnet18_e_best.pth"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    model = resnet18(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model, path


@torch.no_grad()
def collect_predictions(model, loader, device, topk: int = 3):
    y_true, y_pred = [], []
    topk_hit = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(1).cpu().numpy()
        y_np = y.numpy()
        y_true.extend(y_np.tolist())
        y_pred.extend(pred.tolist())

        k = min(topk, probs.size(1))
        topk_idx = torch.topk(probs, k, dim=1).indices.cpu().numpy()
        for i, label in enumerate(y_np):
            total += 1
            if label in topk_idx[i]:
                topk_hit += 1

    return (
        np.array(y_true),
        np.array(y_pred),
        topk_hit / max(total, 1),
    )


def plot_confusion(cm: np.ndarray, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_per_class_f1(f1_list, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(CLASS_NAMES, f1_list, color="#2f6b4f")
    # Zoom y-axis so small gaps among high F1 scores are visible.
    y_min = max(0.0, min(f1_list) - 0.05)
    ax.set_ylim(y_min, 1.02)
    ax.set_ylabel("F1-score")
    ax.set_title("Per-class F1")
    for bar, val in zip(bars, f1_list):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    model, model_path = load_eval_model(device)
    loader = DataLoader(
        LoadData(str(TEST_LIST), False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    y_true, y_pred, top3_acc = collect_predictions(model, loader, device, topk=3)
    top1_acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    per_class = []
    for i, name in enumerate(CLASS_NAMES):
        per_class.append({
            "class": name,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        })

    metrics = {
        "model_path": str(model_path),
        "num_test_samples": int(len(y_true)),
        "top1_accuracy": top1_acc,
        "top3_accuracy": float(top3_acc),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "classification_report": classification_report(
            y_true, y_pred, target_names=CLASS_NAMES, zero_division=0
        ),
    }

    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_confusion(cm, FIGURES_DIR / "confusion_matrix.png")
    plot_per_class_f1([c["f1"] for c in per_class], FIGURES_DIR / "per_class_f1.png")

    print("===== Evaluation =====")
    print(f"Model: {model_path}")
    print(f"Top-1 Acc: {top1_acc * 100:.2f}%")
    print(f"Top-3 Acc: {top3_acc * 100:.2f}%")
    print(f"Macro F1:  {macro_f1:.4f}")
    print(f"Saved: {METRICS_PATH}")
    print(f"Figures: {FIGURES_DIR}")
    print(metrics["classification_report"])


if __name__ == "__main__":
    main()
