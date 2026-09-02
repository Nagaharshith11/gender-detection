"""
TrainX training entry point for a YOLO person detector.

The TrainX orchestrator invokes this file with:
  --img --batch --epochs --data --weights --project --name --seed --exist-ok

The script uses Ultralytics YOLO and writes the two artifacts required by
TrainX:
  <project>/<name>/results.csv
  <project>/<name>/weights/best.pt

The generated dataset.yaml is supplied by TrainX from the approved dataset.
Do not hard-code dataset paths or labels here.
"""
import argparse
import csv
import os
import random
import shutil
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="TrainX YOLO person detector")
    p.add_argument("--img", "--imgsz", type=int, default=320)
    p.add_argument("--batch", "--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--data", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exist-ok", action="store_true")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _num(row, *keys):
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            try:
                return float(value)
            except ValueError:
                pass
    return 0.0


def make_trainx_results(ultra_csv: Path, output_csv: Path) -> None:
    """Convert Ultralytics results.csv to the YOLOv5-style columns TrainX reads."""
    with ultra_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"Ultralytics produced an empty results file: {ultra_csv}")

    fields = [
        "epoch",
        "train/box_loss", "train/obj_loss", "train/cls_loss",
        "metrics/precision", "metrics/recall",
        "metrics/mAP_0.5", "metrics/mAP_0.5:0.95",
        "val/box_loss", "val/obj_loss", "val/cls_loss",
        "x/lr0", "x/lr1", "x/lr2",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {key: "" for key in fields}
            out["epoch"] = row.get("epoch", "")
            out["train/box_loss"] = _num(row, "train/box_loss")
            out["train/obj_loss"] = _num(row, "train/obj_loss", "train/dfl_loss")
            out["train/cls_loss"] = _num(row, "train/cls_loss")
            out["metrics/precision"] = _num(row, "metrics/precision(B)", "metrics/precision")
            out["metrics/recall"] = _num(row, "metrics/recall(B)", "metrics/recall")
            out["metrics/mAP_0.5"] = _num(row, "metrics/mAP50(B)", "metrics/mAP_0.5")
            out["metrics/mAP_0.5:0.95"] = _num(row, "metrics/mAP50-95(B)", "metrics/mAP_0.5:0.95")
            out["val/box_loss"] = _num(row, "val/box_loss")
            out["val/obj_loss"] = _num(row, "val/obj_loss", "val/dfl_loss")
            out["val/cls_loss"] = _num(row, "val/cls_loss")
            out["x/lr0"] = _num(row, "lr/pg0", "x/lr0")
            out["x/lr1"] = _num(row, "lr/pg1", "x/lr1")
            out["x/lr2"] = _num(row, "lr/pg2", "x/lr2")
            writer.writerow(out)


def main():
    args = parse_args()
    seed_everything(args.seed)

    run_dir = Path(args.project) / args.name
    if run_dir.exists() and not args.exist_ok:
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print("ERROR: ultralytics is required in the TrainX training environment.", file=sys.stderr)
        print("Install it in the shared venv or coordinate it with the MLOps administrator.", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"[trainx] data={args.data}")
    print(f"[trainx] weights={args.weights}")
    print(f"[trainx] img={args.img} batch={args.batch} epochs={args.epochs} seed={args.seed}")
    print(f"[trainx] output={run_dir}")

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        imgsz=args.img,
        batch=args.batch,
        epochs=args.epochs,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        seed=args.seed,
        pretrained=True,
        verbose=True,
    )

    source_run_dir = Path(model.trainer.save_dir)
    source_best = source_run_dir / "weights" / "best.pt"
    source_results = source_run_dir / "results.csv"

    if not source_best.exists():
        raise RuntimeError(f"Training completed but best.pt was not found at {source_best}")
    if not source_results.exists():
        raise RuntimeError(f"Training completed but results.csv was not found at {source_results}")

    target_weights = run_dir / "weights"
    target_weights.mkdir(parents=True, exist_ok=True)
    target_best = target_weights / "best.pt"
    if source_best.resolve() != target_best.resolve():
        shutil.copy2(source_best, target_best)

    # TrainX expects YOLOv5-style metric names for its acceptance gate.
    make_trainx_results(source_results, run_dir / "results.csv")

    # Preserve the conventional last checkpoint if Ultralytics produced one.
    source_last = source_run_dir / "weights" / "last.pt"
    if source_last.exists():
        shutil.copy2(source_last, target_weights / "last.pt")

    print(f"[trainx] SUCCESS: {target_best}")
    print(f"[trainx] SUCCESS: {run_dir / 'results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
