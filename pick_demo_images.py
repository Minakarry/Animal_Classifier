"""Pick success / meaningful Top-3 / failure images for paper screenshots."""
from __future__ import annotations

import random
import shutil
from collections import defaultdict
from pathlib import Path

from config import CLASS_NAMES_EN, ROOT
from predict import predict_full


def main():
    out = ROOT / "images" / "demo_for_paper"
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*"):
        if p.is_file():
            p.unlink()

    rows = Path("test.txt").read_text(encoding="utf-8").splitlines()
    by_cls: dict[int, list[str]] = defaultdict(list)
    for r in rows:
        if not r.strip():
            continue
        path, y = r.split("\t")
        by_cls[int(y)].append(path)

    picks = {"success": [], "topk_close": [], "fail": []}
    random.seed(42)
    candidates: list[str] = []
    for paths in by_cls.values():
        sample = paths[:]
        random.shuffle(sample)
        candidates.extend(sample[:8])
    random.shuffle(candidates)

    path_to_y = {}
    for y, paths in by_cls.items():
        for p in paths:
            path_to_y[p] = y

    for path in candidates:
        true_id = path_to_y[path]
        try:
            r = predict_full(path, topk=3, with_cam=False)
        except Exception:
            continue
        top1 = r["top1"]
        topk = r["topk"]
        conf = float(top1["confidence"])
        correct = top1["id"] == true_id
        top2c = float(topk[1]["confidence"]) if len(topk) > 1 else 0.0
        true_name = CLASS_NAMES_EN[true_id]
        item = {
            "path": path,
            "true": true_name,
            "pred": top1["name"],
            "conf": conf,
            "top2": topk[1]["name"] if len(topk) > 1 else "",
            "topk": [(t["name"], float(t["confidence"])) for t in topk],
        }
        if correct and conf >= 0.95 and len(picks["success"]) < 12:
            picks["success"].append(item)
        elif correct and 0.55 <= conf <= 0.92 and top2c >= 0.05 and len(picks["topk_close"]) < 8:
            picks["topk_close"].append(item)
        elif (not correct) and conf >= 0.65 and len(picks["fail"]) < 6:
            picks["fail"].append(item)

        if (
            len(picks["success"]) >= 10
            and len(picks["topk_close"]) >= 5
            and len(picks["fail"]) >= 4
        ):
            break

    # If too few close Top-3, relax slightly
    if len(picks["topk_close"]) < 4:
        for path in candidates:
            true_id = path_to_y[path]
            try:
                r = predict_full(path, topk=3, with_cam=False)
            except Exception:
                continue
            top1 = r["top1"]
            topk = r["topk"]
            conf = float(top1["confidence"])
            top2c = float(topk[1]["confidence"]) if len(topk) > 1 else 0.0
            if top1["id"] != true_id:
                continue
            if not (0.50 <= conf <= 0.97 and top2c >= 0.02):
                continue
            if any(x["path"] == path for x in picks["success"] + picks["topk_close"]):
                continue
            picks["topk_close"].append({
                "path": path,
                "true": CLASS_NAMES_EN[true_id],
                "pred": top1["name"],
                "conf": conf,
                "top2": topk[1]["name"] if len(topk) > 1 else "",
                "topk": [(t["name"], float(t["confidence"])) for t in topk],
            })
            if len(picks["topk_close"]) >= 6:
                break

    manifest = []
    for kind, items in picks.items():
        for i, it in enumerate(items, 1):
            src = Path(it["path"])
            conf_s = f"{it['conf'] * 100:.0f}"
            if kind == "success":
                name = f"01_success_{it['true']}_{conf_s}pct{src.suffix}"
            elif kind == "topk_close":
                name = f"02_topk_{it['true']}_vs_{it['top2']}_{conf_s}pct{src.suffix}"
            else:
                name = (
                    f"03_fail_true-{it['true']}_pred-{it['pred']}_{conf_s}pct{src.suffix}"
                )
            dst = out / name
            if dst.exists():
                dst = out / f"{dst.stem}_{i}{dst.suffix}"
            shutil.copy2(src, dst)
            line = (
                f"{dst.name}\ttrue={it['true']}\tpred={it['pred']}\t"
                f"conf={it['conf']:.3f}\ttop3={it['topk']}"
            )
            manifest.append(line)
            print(line)

    readme = (
        "论文截图用图（已从测试集筛选并复制）\n"
        "01_success_* : 高置信度正确，适合成功案例\n"
        "02_topk_*    : 正确且 Top-2 也有分量，适合展示 Top-3\n"
        "03_fail_*    : 误分类，适合失败分析 / 图7\n\n"
        + "\n".join(manifest)
        + "\n"
    )
    (out / "README.txt").write_text(readme, encoding="utf-8")
    print("\nSaved to", out)
    print("counts", {k: len(v) for k, v in picks.items()})


if __name__ == "__main__":
    main()
