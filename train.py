"""使用 ResNet18 在 Animals-10 上微调训练。"""
from __future__ import annotations

import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import ResNet18_Weights, resnet18

from config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MODEL_PATH,
    NUM_CLASSES,
    OUTPUT_DIR,
    TEST_LIST,
    TRAIN_LIST,
)
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
    print(
        f"Val Accuracy: {(100 * accuracy):>0.1f}%, Avg loss: {avg_loss:>8f}\n"
    )
    return accuracy, avg_loss


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter("logs")

    train_data = LoadData(str(TRAIN_LIST), True)
    valid_data = LoadData(str(TEST_LIST), False)
    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device | classes={NUM_CLASSES}")

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    nn.init.xavier_normal_(model.fc.weight)
    model = model.to(device)

    backbone_params = [
        p for n, p in model.named_parameters()
        if not n.startswith("fc.")
    ]
    head_params = [
        p for n, p in model.named_parameters()
        if n.startswith("fc.")
    ]

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [
            {"params": backbone_params, "lr": LEARNING_RATE},
            {"params": head_params, "lr": LEARNING_RATE * 10},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = -1.0
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}\n-------------------------------")
        t0 = time.time()
        train_loss = train_one_epoch(train_loader, model, loss_fn, optimizer, device)
        print(f"train time: {time.time() - t0:.1f}s | train_loss={train_loss:.6f}")

        val_acc, val_loss = validate(valid_loader, model, loss_fn, device)
        writer.add_scalar("avg_train_loss", train_loss, epoch)
        writer.add_scalar("val_accuracy", val_acc, epoch)
        writer.add_scalar("val_loss", val_loss, epoch)

        torch.save(model.state_dict(), OUTPUT_DIR / "resnet18_last.pth")
        if (epoch + 1) % 5 == 0:
            torch.save(
                model.state_dict(),
                OUTPUT_DIR / f"resnet18_epoch{epoch + 1}.pth",
            )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            # 兼容旧文件名
            torch.save(model.state_dict(), OUTPUT_DIR / "resnet18_e_best.pth")
            print(f"保存最佳模型: {MODEL_PATH} (acc={best_acc:.4f})")

        scheduler.step()

    writer.close()
    print(f"训练结束，最佳验证准确率: {best_acc:.4f}")


if __name__ == "__main__":
    main()
