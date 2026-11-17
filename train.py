"""Fine-tune ResNet18 on Animals-10 (with seed, early stopping, metrics dump)."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import ResNet18_Weights, resnet18

from config import (
    BATCH_SIZE,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    MODEL_PATH,
    NUM_CLASSES,
    OUTPUT_DIR,
    SEED,
    TEST_LIST,
    TRAIN_LIST,
)
from seed_utils import set_seed
from utils import LoadData


def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    size = len(dataloader.dataset)

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch % 20 == 0:
            current = batch * len(x)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def validate(dataloader, model, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    size = len(dataloader.dataset)

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        total_loss += loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    avg_loss = total_loss / max(len(dataloader), 1)
    accuracy = correct / max(size, 1)
    print(f"Val Accuracy: {(100 * accuracy):>0.1f}%, Avg loss: {avg_loss:>8f}\n")
    return accuracy, avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE)
    args = parser.parse_args()

    set_seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter("logs")

    train_data = LoadData(str(TRAIN_LIST), True)
    valid_data = LoadData(str(TEST_LIST), False)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} | classes={NUM_CLASSES} | seed={args.seed}")

    weights = None if args.no_pretrained else ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    nn.init.xavier_normal_(model.fc.weight)
    model = model.to(device)

    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    head_params = [p for n, p in model.named_parameters() if n.startswith("fc.")]

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [
            {"params": backbone_params, "lr": args.lr},
            {"params": head_params, "lr": args.lr * 10},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = -1.0
    bad_epochs = 0
    history = []
    for t in range(args.epochs):
        print(f"Epoch {t + 1}/{args.epochs}\n-------------------------------")
        t0 = time.time()
        avg_loss = train_one_epoch(train_loader, model, loss_fn, optimizer, device)
        print(f"train time: {time.time() - t0:.1f}s | train_loss={avg_loss:.6f}")

        val_acc, val_loss = validate(valid_loader, model, loss_fn, device)
        writer.add_scalar("avg_train_loss", avg_loss, t)
        writer.add_scalar("val_accuracy", val_acc, t)
        writer.add_scalar("val_loss", val_loss, t)
        history.append({
            "epoch": t + 1,
            "train_loss": avg_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        torch.save(model.state_dict(), OUTPUT_DIR / "resnet18_last.pth")
        if (t + 1) % 5 == 0:
            torch.save(model.state_dict(), OUTPUT_DIR / f"resnet18_epoch{t + 1}.pth")

        if val_acc > best_acc:
            best_acc = val_acc
            bad_epochs = 0
            torch.save(model.state_dict(), MODEL_PATH)
            torch.save(model.state_dict(), OUTPUT_DIR / "resnet18_e_best.pth")
            print(f"Saved best model: {MODEL_PATH} (acc={best_acc:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {t + 1}")
                break

        scheduler.step()

    writer.close()
    run_info = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "epochs_ran": len(history),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "pretrained": not args.no_pretrained,
        "best_val_accuracy": best_acc,
        "history": history,
    }
    (OUTPUT_DIR / "train_run.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Training finished. Best val acc: {best_acc:.4f}")
    print("Next: python evaluate.py")


if __name__ == "__main__":
    main()
