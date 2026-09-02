```python
"""
TrainX training entry point for a YOLO person/gender detector.

The TrainX orchestrator invokes this file with:

    --img --batch --epochs --data --weights --project --name
    --seed --exist-ok

The script uses Ultralytics YOLO and writes the artifacts required
by TrainX:

    <project>/<name>/results.csv
    <project>/<name>/weights/best.pt
    <project>/<name>/weights/last.pt   (when produced)

Important:
- Dataset paths are NOT hard-coded.
- Labels/classes are NOT hard-coded.
- The dataset.yaml supplied by TrainX is used directly.
- The model weights supplied by TrainX are used directly.
- If Ultralytics already writes into the TrainX run directory,
  artifacts are NOT copied onto themselves.
"""

import argparse
import csv
import os
import random
import shutil
import sys
from pathlib import Path


def parse_args():
    """Parse arguments supplied by the TrainX orchestrator."""

    p = argparse.ArgumentParser(
        description="TrainX YOLO detector training entry point"
    )

    p.add_argument(
        "--img",
        "--imgsz",
        type=int,
        default=320,
        help="Training image size",
    )

    p.add_argument(
        "--batch",
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size",
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs",
    )

    p.add_argument(
        "--data",
        required=True,
        help="Path to dataset YAML supplied by TrainX",
    )

    p.add_argument(
        "--weights",
        required=True,
        help="Initial YOLO model weights",
    )

    p.add_argument(
        "--project",
        required=True,
        help="Training output project directory",
    )

    p.add_argument(
        "--name",
        required=True,
        help="Training run name",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )

    p.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow an existing output directory",
    )

    return p.parse_args()


def seed_everything(seed: int) -> None:
    """Set deterministic/random seeds where possible."""

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
    """
    Read the first available numeric value from a CSV row.

    Returns 0.0 if no usable value is found.
    """

    for key in keys:
        value = row.get(key)

        if value not in (None, ""):
            try:
                return float(value)
            except (ValueError, TypeError):
                pass

    return 0.0


def make_trainx_results(
    ultra_csv: Path,
    output_csv: Path,
) -> None:
    """
    Convert Ultralytics results.csv into the YOLOv5-style columns
    expected by TrainX.

    Ultralytics column names can vary slightly between versions,
    so multiple possible names are supported.
    """

    if not ultra_csv.exists():
        raise RuntimeError(
            f"Ultralytics results file was not found: {ultra_csv}"
        )

    with ultra_csv.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(
            f"Ultralytics produced an empty results file: {ultra_csv}"
        )

    fields = [
        "epoch",
        "train/box_loss",
        "train/obj_loss",
        "train/cls_loss",
        "metrics/precision",
        "metrics/recall",
        "metrics/mAP_0.5",
        "metrics/mAP_0.5:0.95",
        "val/box_loss",
        "val/obj_loss",
        "val/cls_loss",
        "x/lr0",
        "x/lr1",
        "x/lr2",
    ]

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in rows:

            out = {
                key: ""
                for key in fields
            }

            out["epoch"] = row.get(
                "epoch",
                "",
            )

            # Training losses
            out["train/box_loss"] = _num(
                row,
                "train/box_loss",
            )

            out["train/obj_loss"] = _num(
                row,
                "train/obj_loss",
                "train/dfl_loss",
            )

            out["train/cls_loss"] = _num(
                row,
                "train/cls_loss",
            )

            # Detection metrics
            out["metrics/precision"] = _num(
                row,
                "metrics/precision(B)",
                "metrics/precision",
            )

            out["metrics/recall"] = _num(
                row,
                "metrics/recall(B)",
                "metrics/recall",
            )

            out["metrics/mAP_0.5"] = _num(
                row,
                "metrics/mAP50(B)",
                "metrics/mAP_0.5",
            )

            out["metrics/mAP_0.5:0.95"] = _num(
                row,
                "metrics/mAP50-95(B)",
                "metrics/mAP_0.5:0.95",
            )

            # Validation losses
            out["val/box_loss"] = _num(
                row,
                "val/box_loss",
            )

            out["val/obj_loss"] = _num(
                row,
                "val/obj_loss",
                "val/dfl_loss",
            )

            out["val/cls_loss"] = _num(
                row,
                "val/cls_loss",
            )

            # Learning rates
            out["x/lr0"] = _num(
                row,
                "lr/pg0",
                "x/lr0",
            )

            out["x/lr1"] = _num(
                row,
                "lr/pg1",
                "x/lr1",
            )

            out["x/lr2"] = _num(
                row,
                "lr/pg2",
                "x/lr2",
            )

            writer.writerow(out)


def same_file(source: Path, target: Path) -> bool:
    """
    Safely determine whether source and target refer to the same file.

    resolve() is used so that equivalent paths such as:
        /trainx/runs/a/../a/file.pt

    are treated as the same file.
    """

    try:
        return source.resolve() == target.resolve()
    except OSError:
        return os.path.abspath(source) == os.path.abspath(target)


def copy_if_needed(
    source: Path,
    target: Path,
) -> None:
    """
    Copy a file only when source and target are different files.

    This prevents shutil.SameFileError when Ultralytics has already
    written the artifact directly into the TrainX run directory.
    """

    if not source.exists():
        raise FileNotFoundError(
            f"Required source artifact does not exist: {source}"
        )

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if same_file(source, target):
        print(
            f"[trainx] Artifact already in target location: {target}"
        )
        return

    print(
        f"[trainx] Copying artifact:\n"
        f"         source = {source}\n"
        f"         target = {target}"
    )

    shutil.copy2(
        source,
        target,
    )


def main():
    """Main TrainX training workflow."""

    args = parse_args()

    # ---------------------------------------------------------
    # 1. Seed
    # ---------------------------------------------------------

    seed_everything(args.seed)

    # ---------------------------------------------------------
    # 2. Determine TrainX run directory
    # ---------------------------------------------------------

    run_dir = Path(
        args.project
    ) / args.name

    if run_dir.exists() and not args.exist_ok:
        raise FileExistsError(
            f"Run directory already exists: {run_dir}"
        )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # 3. Validate important input paths
    # ---------------------------------------------------------

    data_path = Path(args.data)
    weights_path = Path(args.weights)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset YAML was not found: {data_path}"
        )

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Initial model weights were not found: {weights_path}"
        )

    # ---------------------------------------------------------
    # 4. Import Ultralytics
    # ---------------------------------------------------------

    try:
        from ultralytics import YOLO

    except ImportError as exc:

        print(
            "ERROR: ultralytics is required in the TrainX "
            "training environment.",
            file=sys.stderr,
        )

        print(
            "Install it in the shared venv or coordinate it "
            "with the MLOps administrator.",
            file=sys.stderr,
        )

        raise SystemExit(1) from exc

    # ---------------------------------------------------------
    # 5. Print configuration
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("[trainx] STARTING YOLO TRAINING")
    print("=" * 70)

    print(f"[trainx] data    = {data_path}")
    print(f"[trainx] weights = {weights_path}")
    print(f"[trainx] img     = {args.img}")
    print(f"[trainx] batch   = {args.batch}")
    print(f"[trainx] epochs  = {args.epochs}")
    print(f"[trainx] seed    = {args.seed}")
    print(f"[trainx] project = {args.project}")
    print(f"[trainx] name    = {args.name}")
    print(f"[trainx] output  = {run_dir}")

    print("=" * 70)
    print()

    # ---------------------------------------------------------
    # 6. Load model
    # ---------------------------------------------------------

    model = YOLO(
        str(weights_path)
    )

    # ---------------------------------------------------------
    # 7. Train
    # ---------------------------------------------------------

    model.train(
        data=str(data_path),
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

    # ---------------------------------------------------------
    # 8. Find Ultralytics output directory
    # ---------------------------------------------------------

    source_run_dir = Path(
        model.trainer.save_dir
    )

    print()
    print("=" * 70)
    print("[trainx] TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"[trainx] Ultralytics save directory:"
        f" {source_run_dir}"
    )

    print(
        f"[trainx] TrainX run directory:"
        f" {run_dir}"
    )

    print("=" * 70)
    print()

    # ---------------------------------------------------------
    # 9. Define expected artifacts
    # ---------------------------------------------------------

    source_weights_dir = (
        source_run_dir / "weights"
    )

    source_best = (
        source_weights_dir / "best.pt"
    )

    source_last = (
        source_weights_dir / "last.pt"
    )

    source_results = (
        source_run_dir / "results.csv"
    )

    target_weights = (
        run_dir / "weights"
    )

    target_weights.mkdir(
        parents=True,
        exist_ok=True,
    )

    target_best = (
        target_weights / "best.pt"
    )

    target_last = (
        target_weights / "last.pt"
    )

    target_results = (
        run_dir / "results.csv"
    )

    # ---------------------------------------------------------
    # 10. Validate best.pt
    # ---------------------------------------------------------

    if not source_best.exists():

        raise RuntimeError(
            "Training completed but best.pt was not found at "
            f"{source_best}"
        )

    # ---------------------------------------------------------
    # 11. Validate Ultralytics results.csv
    # ---------------------------------------------------------

    if not source_results.exists():

        raise RuntimeError(
            "Training completed but results.csv was not "
            f"found at {source_results}"
        )

    # ---------------------------------------------------------
    # 12. Handle best.pt
    # ---------------------------------------------------------

    copy_if_needed(
        source_best,
        target_best,
    )

    # ---------------------------------------------------------
    # 13. Convert results.csv for TrainX
    # ---------------------------------------------------------

    print(
        "[trainx] Converting Ultralytics results.csv "
        "to TrainX format..."
    )

    make_trainx_results(
        source_results,
        target_results,
    )

    # ---------------------------------------------------------
    # 14. Handle last.pt
    #
    # IMPORTANT:
    # This fixes the original SameFileError.
    # ---------------------------------------------------------

    if source_last.exists():

        copy_if_needed(
            source_last,
            target_last,
        )

    else:

        print(
            "[trainx] WARNING: last.pt was not produced "
            "by Ultralytics."
        )

    # ---------------------------------------------------------
    # 15. Final artifact validation
    # ---------------------------------------------------------

    if not target_best.exists():

        raise RuntimeError(
            f"TrainX artifact validation failed: "
            f"best.pt is missing at {target_best}"
        )

    if not target_results.exists():

        raise RuntimeError(
            f"TrainX artifact validation failed: "
            f"results.csv is missing at {target_results}"
        )

    # last.pt is optional, so we only report whether it exists.

    # ---------------------------------------------------------
    # 16. Success
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("[trainx] SUCCESS")
    print("=" * 70)

    print(
        f"[trainx] best.pt    = {target_best}"
    )

    print(
        f"[trainx] results.csv = {target_results}"
    )

    if target_last.exists():

        print(
            f"[trainx] last.pt    = {target_last}"
        )

    print("=" * 70)
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
```
