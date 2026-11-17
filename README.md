# Animal Image Classification System (ResNet18 + Animals-10)

Graduate group project: fine-tune **ResNet18** on **Animals-10**, with desktop (PyQt5) and web (Flask) apps.

**Classes (10):** Butterfly, Cat, Chicken, Cow, Dog, Elephant, Horse, Sheep, Spider, Squirrel  

**Note:** Whole-image single-label classification (not multi-object detection).

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline

```bash
# 1) Download & organize dataset
python download_dataset.py

# 2) Build train/test lists
python generate_dataset.py

# 3) Train (seed + early stopping; writes output/train_run.json)
python train.py
# optional: python train.py --epochs 20 --lr 1e-4 --seed 42

# 4) Evaluate (metrics + confusion matrix + per-class F1)
python evaluate.py
# -> output/metrics.json
# -> output/figures/confusion_matrix.png
# -> output/figures/per_class_f1.png

# 5) Compare pretrained fine-tune vs from-scratch (default 10 epochs)
python compare.py --epochs 10
# -> output/compare_results.json
# -> output/figures/compare_accuracy.png

# 6) Apps
python window.py      # desktop GUI
python myflask.py     # web: http://127.0.0.1:5000
python predict.py images/horse.png
```

## App Features

- Upload & auto recognize (English UI)
- **Top-3** predictions with confidence bars
- **Low-confidence warning** (default threshold 50%)
- **Grad-CAM** heatmap (where the model looks)
- Species description + recognition history (delete selected / clear all)

## Project Layout

| File | Role |
|------|------|
| `train.py` | Training with seed / early stop |
| `evaluate.py` | Test metrics & figures |
| `compare.py` | Pretrained vs scratch comparison |
| `predict.py` | Inference + Top-K + Grad-CAM |
| `window.py` | Desktop UI |
| `myflask.py` | Web backend |
| `config.py` | Classes, threshold, paths |

## Tips

- If `python` does nothing on Windows, use your real interpreter path.
- Web UI: hard refresh (Ctrl+F5) after updates.
- Existing best weights stay at `output/resnet18_best.pth`; re-run `evaluate.py` anytime.
