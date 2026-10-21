"""按类别划分训练集 / 测试集，生成 train.txt 与 test.txt。"""
from __future__ import annotations

import random
from pathlib import Path

from config import (
    DATA_DIR,
    IMAGE_EXTENSIONS,
    TEST_LIST,
    TRAIN_LIST,
    TRAIN_RATIO,
    load_class_names,
)


def collect_samples() -> tuple[list[str], list[str], int]:
    class_names_en, _ = load_class_names()
    train_list: list[str] = []
    test_list: list[str] = []

    for class_id, class_name in enumerate(class_names_en):
        class_dir = DATA_DIR / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"缺少类别目录: {class_dir}\n请先运行: python download_dataset.py"
            )

        images = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise RuntimeError(f"类别目录为空: {class_dir}")

        random.shuffle(images)
        split_idx = max(1, int(len(images) * TRAIN_RATIO))
        # 保证测试集至少有 1 张（类别样本很少时）
        if split_idx >= len(images):
            split_idx = len(images) - 1

        for img in images[:split_idx]:
            train_list.append(f"{img.as_posix()}\t{class_id}")
        for img in images[split_idx:]:
            test_list.append(f"{img.as_posix()}\t{class_id}")

    random.shuffle(train_list)
    random.shuffle(test_list)
    return train_list, test_list, len(class_names_en)


def main() -> None:
    random.seed(42)
    train_list, test_list, num_classes = collect_samples()

    TRAIN_LIST.write_text("\n".join(train_list) + "\n", encoding="utf-8")
    TEST_LIST.write_text("\n".join(test_list) + "\n", encoding="utf-8")

    print(f"训练集: {len(train_list)} -> {TRAIN_LIST.name}")
    print(f"测试集: {len(test_list)} -> {TEST_LIST.name}")
    print(f"类别数: {num_classes}")


if __name__ == "__main__":
    main()
