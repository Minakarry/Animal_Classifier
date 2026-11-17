"""Global project configuration: classes, paths, hyperparameters."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Animals-10: 10 common animal categories
# Source: https://www.kaggle.com/datasets/alessiocorrado99/animals10
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

# English descriptions keyed by display name (Title Case)
CLASS_DESCRIPTIONS = {
    "Butterfly": "An insect of the order Lepidoptera. Its wings are covered with scales; most species feed on nectar and act as pollinators.",
    "Cat": "A small carnivorous mammal that is agile and has excellent night vision. One of the most common companion animals.",
    "Chicken": "A domesticated bird of the pheasant family. It is widely farmed and provides eggs and meat.",
    "Cow": "A large ruminant livestock animal, commonly raised for milk and meat, and historically used as a draft animal.",
    "Dog": "One of the earliest domesticated animals, with a keen sense of smell and strong cooperation ability; often kept as a companion or working animal.",
    "Elephant": "The largest living land mammal, with a flexible trunk and strong memory; typically lives in social groups.",
    "Horse": "A large herbivorous livestock animal with strong running ability; historically used for riding and transport.",
    "Sheep": "A ruminant livestock animal that lives in flocks and provides wool, meat, and milk.",
    "Spider": "An arachnid, usually with eight legs; many species spin webs to catch prey and play an important role in ecosystems.",
    "Squirrel": "A small rodent with a bushy tail; skilled at climbing trees and often feeds on nuts and seeds.",
}

# Original Animals-10 Italian folder names -> English
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

DATA_DIR = ROOT / "data"
RAW_DIR = ROOT / "raw_data"
OUTPUT_DIR = ROOT / "output"
MODEL_PATH = OUTPUT_DIR / "resnet18_best.pth"
TRAIN_LIST = ROOT / "train.txt"
TEST_LIST = ROOT / "test.txt"
CLASSES_FILE = ROOT / "classes.json"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
TRAIN_RATIO = 0.9
SEED = 42
CONFIDENCE_THRESHOLD = 0.50  # below this → uncertain / low-confidence warning
EARLY_STOP_PATIENCE = 8

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
COMPARE_PATH = OUTPUT_DIR / "compare_results.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _to_display_name(name_en: str) -> str:
    return name_en.replace("_", " ").title()


def load_class_names() -> tuple[list[str], list[str]]:
    """Load English folder names and Chinese labels from classes.json if present."""
    if CLASSES_FILE.exists():
        data = json.loads(CLASSES_FILE.read_text(encoding="utf-8"))
        names_en = [item["en"] for item in data]
        names_zh = [
            item.get("zh") or ZH_NAME_MAP.get(item["en"], item["en"])
            for item in data
        ]
        return names_en, names_zh
    return list(_DEFAULT_CLASS_NAMES_EN), list(_DEFAULT_CLASS_NAMES_ZH)


CLASS_NAMES_EN, CLASS_NAMES_ZH = load_class_names()
# UI / prediction display names (English, Title Case)
CLASS_NAMES = [_to_display_name(n) for n in CLASS_NAMES_EN]
NUM_CLASSES = len(CLASS_NAMES)


def get_class_description(name: str) -> str:
    """Look up description by English display name (or raw English folder name)."""
    key = _to_display_name(name) if name and name[0].islower() else name
    return CLASS_DESCRIPTIONS.get(key, CLASS_DESCRIPTIONS.get(name, "No description available."))
