"""
Compare pretrained ResNet18 fine-tuning vs training from scratch.

Default uses fewer epochs for a quick fair comparison.
Results: output/compare_results.json and output/figures/compare_accuracy.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

from config import (
    BATCH_SIZE,
    COMPARE_PATH,
    FIGURES_DIR,
    LEARNING_RATE,
    NUM_CLASSES,
    OUTPUT_DIR,
    SEED,
    TEST_LIST,
    TRAIN_LIST,
)
from seed_utils import set_seed
from utils import LoadData


def build_model(pretrained: bool, device: str):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    nn.init.xavier_normal_(model.fc.weight)
    return model.to(device)


def train_one_epoch(loader, model, loss_fn, optimizer, device):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def evaluate_acc(loader, model, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def run_one(name: str, pretrained: bool, epochs: int, device: str, train_loader, valid_loader):
    set_seed(SEED)
    model = build_model(pretrained, device)
    backbone = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    head = [p for n, p in model.named_parameters() if n.startswith("fc.")]
    optimizer = torch.optim.Adam(
        [
            {"params": backbone, "lr": LEARNING_RATE},
            {"params": head, "lr": LEARNING_RATE * 10},
        ]
    )
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = -1.0
    history = []
    t0 = time.time()
    for epoch in range(epochs):
        train_loss = train_one_epoch(train_loader, model, loss_fn, optimizer, device)
        val_acc = evaluate_acc(valid_loader, model, device)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_acc": val_acc})
        print(
            f"[{name}] epoch {epoch + 1}/{epochs} loss={train_loss:.4f} acc={val_acc:.4f}",
            flush=True,
        )
        if val_acc > best_acc:
            best_acc = val_acc
            # Never overwrite the main deployed checkpoint during comparison.
            save_name = (
                "resnet18_compare_pretrained_best.pth"
                if pretrained
                else "resnet18_compare_scratch_best.pth"
            )
            torch.save(model.state_dict(), OUTPUT_DIR / save_name)
        scheduler.step()

    return {
        "name": name,
        "pretrained": pretrained,
        "epochs": epochs,
        "best_val_accuracy": best_acc,
        "seconds": time.time() - t0,
        "history": history,
    }


def plot_compare(results, out_path: Path):
    names = [r["name"] for r in results]
    accs = [r["best_val_accuracy"] * 100 for r in results]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, accs, color=["#2f6b4f", "#c4782a"])
    ax.set_ylabel("Best Val Accuracy (%)")
    ax.set_title("Pretrained Fine-tune vs From Scratch")
    ax.set_ylim(0, 100)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 1, f"{acc:.1f}%", ha="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare pretrained vs scratch ResNet18")
    parser.add_argument("--epochs", type=int, default=10, help="epochs for each setting")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"Compare start | device={device} | epochs={args.epochs} each "
        f"(pretrained + from-scratch). First log appears after epoch 1 finishes.",
        flush=True,
    )
    train_ds = LoadData(str(TRAIN_LIST), True)
    valid_ds = LoadData(str(TEST_LIST), False)
    print(f"Train samples={len(train_ds)} | Test samples={len(valid_ds)}", flush=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    results = []
    print(">>> Training Pretrained run...", flush=True)
    results.append(
        run_one("Pretrained", True, args.epochs, device, train_loader, valid_loader)
    )
    print(">>> Training From-Scratch run...", flush=True)
    results.append(
        run_one("From-Scratch", False, args.epochs, device, train_loader, valid_loader)
    )

    payload = {
        "seed": SEED,
        "epochs": args.epochs,
        "results": results,
    }
    COMPARE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_compare(results, FIGURES_DIR / "compare_accuracy.png")

    print("\n===== Comparison =====")
    for r in results:
        print(f"{r['name']}: best_acc={r['best_val_accuracy'] * 100:.2f}%  time={r['seconds']:.1f}s")
    print(f"Saved: {COMPARE_PATH}")
    print(f"Figure: {FIGURES_DIR / 'compare_accuracy.png'}")


if __name__ == "__main__":
    main()
