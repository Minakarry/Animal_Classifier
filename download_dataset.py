"""
下载并整理 Animals-10 动物图像数据集（10 类常见动物）。

原始发布: https://www.kaggle.com/datasets/alessiocorrado99/animals10
镜像: HuggingFace dgrnd4/animals-10

整理为: data/<英文类名>/*.jpg
"""
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from tqdm import tqdm

from config import (
    CLASSES_FILE,
    DATA_DIR,
    IMAGE_EXTENSIONS,
    ITALIAN_TO_EN,
    RAW_DIR,
    ZH_NAME_MAP,
    _DEFAULT_CLASS_NAMES_EN,
)

# 国内可优先尝试 hf-mirror
DATASET_URLS = [
    "https://hf-mirror.com/datasets/dgrnd4/animals-10/resolve/main/dataset.zip",
    "https://huggingface.co/datasets/dgrnd4/animals-10/resolve/main/dataset.zip",
]


class _DownloadProgress:
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if self.pbar is None:
            self.pbar = tqdm(
                total=total_size if total_size > 0 else None,
                unit="B",
                unit_scale=True,
                desc="下载 Animals-10",
            )
        self.pbar.update(block_size)
        if total_size > 0 and self.pbar.n >= total_size:
            self.pbar.close()


def download_zip(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"已存在压缩包，跳过下载: {dest}")
        return dest

    last_error = None
    for url in DATASET_URLS:
        try:
            print(f"尝试下载: {url}")
            urlretrieve(url, dest, reporthook=_DownloadProgress())
            if dest.exists() and dest.stat().st_size > 1_000_000:
                return dest
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"下载失败: {exc}")
            if dest.exists():
                dest.unlink(missing_ok=True)

    raise RuntimeError(
        "无法自动下载 Animals-10。\n"
        "请手动下载后放到 raw_data/animals10.zip，再重新运行本脚本。\n"
        "Kaggle: https://www.kaggle.com/datasets/alessiocorrado99/animals10\n"
        "HF镜像: https://hf-mirror.com/datasets/dgrnd4/animals-10/resolve/main/dataset.zip"
    ) from last_error


def normalize_class_name(folder_name: str) -> str | None:
    name = folder_name.strip().lower().replace(" ", "_")
    if name in _DEFAULT_CLASS_NAMES_EN:
        return name
    if name in ITALIAN_TO_EN:
        return ITALIAN_TO_EN[name]
    # 兼容首字母大写目录
    lower = name.lower()
    if lower in _DEFAULT_CLASS_NAMES_EN:
        return lower
    if lower in ITALIAN_TO_EN:
        return ITALIAN_TO_EN[lower]
    return None


def find_class_dirs(root: Path) -> dict[str, Path]:
    """在解压目录中查找 10 个类别文件夹。"""
    found: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        class_name = normalize_class_name(path.name)
        if class_name is None or class_name in found:
            continue
        # 目录内需有图片
        has_image = any(
            p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            for p in path.iterdir()
        )
        if has_image:
            found[class_name] = path
    return found


def export_to_data(class_dirs: dict[str, Path]) -> int:
    if DATA_DIR.exists():
        for child in DATA_DIR.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    for class_name in _DEFAULT_CLASS_NAMES_EN:
        src_dir = class_dirs.get(class_name)
        if src_dir is None:
            raise RuntimeError(f"未找到类别目录: {class_name}")

        out_dir = DATA_DIR / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        images = [
            p for p in src_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        for img in tqdm(images, desc=f"整理 {class_name}"):
            dst = out_dir / img.name
            if not dst.exists():
                shutil.copy2(img, dst)
            total += 1
    return total


def write_classes_json() -> None:
    classes = [
        {"id": i, "en": en, "zh": ZH_NAME_MAP[en]}
        for i, en in enumerate(_DEFAULT_CLASS_NAMES_EN)
    ]
    CLASSES_FILE.write_text(
        json.dumps(classes, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "animals10.zip"
    extract_dir = RAW_DIR / "animals10_extracted"

    download_zip(zip_path)

    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"解压: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    class_dirs = find_class_dirs(extract_dir)
    missing = [c for c in _DEFAULT_CLASS_NAMES_EN if c not in class_dirs]
    if missing:
        raise RuntimeError(
            f"解压后缺少类别: {missing}\n"
            f"已找到: {sorted(class_dirs.keys())}"
        )

    total = export_to_data(class_dirs)
    write_classes_json()

    print(f"完成：共整理 {total} 张图片到 {DATA_DIR}")
    print(f"类别数: {len(_DEFAULT_CLASS_NAMES_EN)} -> {_DEFAULT_CLASS_NAMES_EN}")
    print(f"类别映射已写入 {CLASSES_FILE}")
    print("下一步请运行: python generate_dataset.py")


if __name__ == "__main__":
    main()
