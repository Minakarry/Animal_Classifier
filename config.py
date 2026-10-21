"""项目全局配置：类别、路径、预处理参数等。"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Animals-10：10 类常见动物（非宠物品种）
# 来源: https://www.kaggle.com/datasets/alessiocorrado99/animals10
_DEFAULT_CLASS_NAMES_EN = [
    "butterfly",
    "cat",
    "chicken",
    "cow",
    "dog",
    "elephant",
    "horse",
    "sheep",
    "spider",
    "squirrel",
]

_DEFAULT_CLASS_NAMES_ZH = [
    "蝴蝶",
    "猫",
    "鸡",
    "牛",
    "狗",
    "象",
    "马",
    "羊",
    "蜘蛛",
    "松鼠",
]

ZH_NAME_MAP = dict(zip(_DEFAULT_CLASS_NAMES_EN, _DEFAULT_CLASS_NAMES_ZH))

# 各类简介（按中文类名索引）
CLASS_DESCRIPTIONS = {
    "蝴蝶": "昆虫纲鳞翅目。翅膀覆盖鳞粉，多数靠花蜜为生，是常见的传粉者。",
    "猫": "小型食肉哺乳动物，行动敏捷、夜视出色，是最常见的家养伴侣动物之一。",
    "鸡": "家禽，鸟纲雉科。善走少飞，广泛养殖，可为人类提供蛋和肉。",
    "牛": "大型反刍家畜，体型壮实。常用于产奶、产肉，也是传统农业役畜。",
    "狗": "最早被人类驯化的家畜之一，嗅觉灵敏、善于协作，常作伴侣或工作犬。",
    "象": "现存陆地上体型最大的哺乳动物，长鼻灵活，群居生活，记忆力出色。",
    "马": "大型草食家畜，四肢修长、奔跑能力强，历史上广泛用于骑乘与运输。",
    "羊": "反刍草食家畜，群居性强，主要提供羊毛、羊肉和羊奶。",
    "蜘蛛": "蛛形纲节肢动物，大多有八足，能结网捕猎，是生态系统中的重要捕食者。",
    "松鼠": "啮齿目松鼠科小动物，尾毛蓬松，擅爬树，常以坚果和种子为食。",
}

# 原始 Animals-10 意大利语目录名 -> 英文标准名
ITALIAN_TO_EN = {
    "farfalla": "butterfly",
    "gatto": "cat",
    "gallina": "chicken",
    "mucca": "cow",
    "cane": "dog",
    "elefante": "elephant",
    "cavallo": "horse",
    "pecora": "sheep",
    "ragno": "spider",
    "scoiattolo": "squirrel",
}

# 数据与模型路径
DATA_DIR = ROOT / "data"
RAW_DIR = ROOT / "raw_data"
OUTPUT_DIR = ROOT / "output"
MODEL_PATH = OUTPUT_DIR / "resnet18_best.pth"
TRAIN_LIST = ROOT / "train.txt"
TEST_LIST = ROOT / "test.txt"
CLASSES_FILE = ROOT / "classes.json"

# 训练超参数
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
TRAIN_RATIO = 0.9

# ImageNet 预训练常用归一化（配合 ResNet 预训练权重）
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_class_names() -> tuple[list[str], list[str]]:
    """优先读取 classes.json，保证与下载脚本导出的目录一致。"""
    if CLASSES_FILE.exists():
        data = json.loads(CLASSES_FILE.read_text(encoding="utf-8"))
        names_en = [item["en"] for item in data]
        names_zh = [
            item.get("zh") or ZH_NAME_MAP.get(item["en"], item["en"])
            for item in data
        ]
        return names_en, names_zh
    return list(_DEFAULT_CLASS_NAMES_EN), list(_DEFAULT_CLASS_NAMES_ZH)


CLASS_NAMES_EN, CLASS_NAMES = load_class_names()
NUM_CLASSES = len(CLASS_NAMES)


def get_class_description(name_zh: str) -> str:
    return CLASS_DESCRIPTIONS.get(name_zh, "暂无该类简介。")
