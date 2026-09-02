\"""
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
import random
import shutil
import sys
from pathlib import Path


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="TrainX YOLO person detector"
    )

    parser.add_argument(
        "--img",
        "--imgsz",
        type=int,
        default=320
    )

    parser.add_argument(
        "--batch",
        "--batch-size",
        type=int,
        default=8
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2
    )

    parser.add_argument(
        "--data",
        required=True
    )

    parser.add_argument(
        "--weights",
        required=True
    )

    parser.add_argument(
        "--project",
        required=True
    )

    parser.add_argument(
        "--name",
        required=True
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0
    )

    parser.add_argument(
        "--exist-ok",
        action="store_true"
    )

    return parser.parse_args()


# ============================================================
# RANDOM SEED
# ============================================================

def seed_everything(seed: int) -> None:
    """
    Set random seeds for reproducible training.
    """

    random.seed(seed)

    # NumPy
    try:
        import numpy as np

        np.random.seed(seed)

    except Exception:
        pass

    # PyTorch
    try:
        import torch

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    except Exception:
        pass


# ============================================================
# NUMBER CONVERSION
# ============================================================

def _num(row, *keys):
    """
    Safely get a numeric value from a CSV row.

    Different Ultralytics versions can use slightly different
    column names, so multiple possible names are supported.
    """

    for key in keys:

        value = row.get(key)

        if value not in (None, ""):

            try:
                return float(value)

            except (ValueError, TypeError):
                pass

    return 0.0


# ============================================================
# RESULTS CSV CONVERSION
# ============================================================

def make_trainx_results(
    ultra_csv: Path,
    output_csv: Path
) -> None:
    """
    Convert the Ultralytics results.csv into the metric columns
    expected by TrainX.
    """

    if not ultra_csv.exists():

        raise RuntimeError(
            f"Ultralytics results.csv was not found at {ultra_csv}"
        )

    with ultra_csv.open(
        "r",
        newline="",
        encoding="utf-8"
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
        exist_ok=True
    )

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        for row in rows:

            out = {
                key: ""
                for key in fields
            }

            # Epoch
            out["epoch"] = row.get(
                "epoch",
                ""
            )

            # Training losses
            out["train/box_loss"] = _num(
                row,
                "train/box_loss"
            )

            out["train/obj_loss"] = _num(
                row,
                "train/obj_loss",
                "train/dfl_loss"
            )

            out["train/cls_loss"] = _num(
                row,
                "train/cls_loss"
            )

            # Precision
            out["metrics/precision"] = _num(
                row,
                "metrics/precision(B)",
                "metrics/precision"
            )

            # Recall
            out["metrics/recall"] = _num(
                row,
                "metrics/recall(B)",
                "metrics/recall"
            )

            # mAP 50
            out["metrics/mAP_0.5"] = _num(
                row,
                "metrics/mAP50(B)",
                "metrics/mAP_0.5"
            )

            # mAP 50-95
            out["metrics/mAP_0.5:0.95"] = _num(
                row,
                "metrics/mAP50-95(B)",
                "metrics/mAP_0.5:0.95"
            )

            # Validation losses
            out["val/box_loss"] = _num(
                row,
                "val/box_loss"
            )

            out["val/obj_loss"] = _num(
                row,
                "val/obj_loss",
                "val/dfl_loss"
            )

            out["val/cls_loss"] = _num(
                row,
                "val/cls_loss"
            )

            # Learning rates
            out["x/lr0"] = _num(
                row,
                "lr/pg0",
                "x/lr0"
            )

            out["x/lr1"] = _num(
                row,
                "lr/pg1",
                "x/lr1"
            )

            out["x/lr2"] = _num(
                row,
                "lr/pg2",
                "x/lr2"
            )

            writer.writerow(out)


# ============================================================
# SAFE FILE COPY
# ============================================================

def safe_copy_file(
    source: Path,
    destination: Path
) -> None:
    """
    Copy a file only when source and destination are different.

    This prevents:

        shutil.SameFileError

    when TrainX/Ultralytics has already created the file
    in the desired destination.
    """

    if not source.exists():

        raise FileNotFoundError(
            f"Source file does not exist: {source}"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Resolve both paths before comparing them.
    source_resolved = source.resolve()
    destination_resolved = destination.resolve()

    # --------------------------------------------------------
    # IMPORTANT FIX
    # --------------------------------------------------------
    # If both paths point to exactly the same file,
    # do NOT call shutil.copy2().
    # --------------------------------------------------------

    if source_resolved == destination_resolved:

        print(
            f"[trainx] SKIP COPY: "
            f"{source_resolved} is already at destination"
        )

        return

    print(
        f"[trainx] COPY: "
        f"{source_resolved} -> {destination_resolved}"
    )

    shutil.copy2(
        source_resolved,
        destination_resolved
    )


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():

    # --------------------------------------------------------
    # Read arguments
    # --------------------------------------------------------

    args = parse_args()

    # --------------------------------------------------------
    # Set random seed
    # --------------------------------------------------------

    seed_everything(
        args.seed
    )

    # --------------------------------------------------------
    # Create TrainX output directory
    # --------------------------------------------------------

    run_dir = (
        Path(args.project)
        / args.name
    )

    if (
        run_dir.exists()
        and not args.exist_ok
    ):

        raise FileExistsError(
            f"Run directory already exists: {run_dir}"
        )

    run_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Import Ultralytics
    # --------------------------------------------------------

    try:

        from ultralytics import YOLO

    except ImportError as exc:

        print(
            "ERROR: ultralytics is required "
            "in the TrainX training environment.",
            file=sys.stderr
        )

        print(
            "Install it in the shared venv or "
            "coordinate it with the MLOps administrator.",
            file=sys.stderr
        )

        raise SystemExit(1) from exc

    # --------------------------------------------------------
    # Print training configuration
    # --------------------------------------------------------

    print(
        f"[trainx] data={args.data}"
    )

    print(
        f"[trainx] weights={args.weights}"
    )

    print(
        f"[trainx] img={args.img} "
        f"batch={args.batch} "
        f"epochs={args.epochs} "
        f"seed={args.seed}"
    )

    print(
        f"[trainx] output={run_dir}"
    )

    # --------------------------------------------------------
    # Load YOLO model
    # --------------------------------------------------------

    print(
        "[trainx] Loading YOLO model..."
    )

    model = YOLO(
        args.weights
    )

    # --------------------------------------------------------
    # Train YOLO
    # --------------------------------------------------------

    print(
        "[trainx] Starting YOLO training..."
    )

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

    # --------------------------------------------------------
    # Find Ultralytics output directory
    # --------------------------------------------------------

    source_run_dir = Path(
        model.trainer.save_dir
    )

    print(
        f"[trainx] Ultralytics output: "
        f"{source_run_dir}"
    )

    # --------------------------------------------------------
    # Locate generated files
    # --------------------------------------------------------

    source_best = (
        source_run_dir
        / "weights"
        / "best.pt"
    )

    source_last = (
        source_run_dir
        / "weights"
        / "last.pt"
    )

    source_results = (
        source_run_dir
        / "results.csv"
    )

    # --------------------------------------------------------
    # Verify best.pt
    # --------------------------------------------------------

    if not source_best.exists():

        raise RuntimeError(
            "Training completed but "
            f"best.pt was not found at {source_best}"
        )

    # --------------------------------------------------------
    # Verify results.csv
    # --------------------------------------------------------

    if not source_results.exists():

        raise RuntimeError(
            "Training completed but "
            f"results.csv was not found at {source_results}"
        )

    # --------------------------------------------------------
    # Create target weights directory
    # --------------------------------------------------------

    target_weights = (
        run_dir
        / "weights"
    )

    target_weights.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # BEST.PT
    # ========================================================

    target_best = (
        target_weights
        / "best.pt"
    )

    safe_copy_file(
        source_best,
        target_best
    )

    # ========================================================
    # RESULTS.CSV
    # ========================================================

    target_results = (
        run_dir
        / "results.csv"
    )

    print(
        "[trainx] Converting "
        "Ultralytics results.csv..."
    )

    # If source and target are different,
    # convert normally.
    #
    # The conversion creates a new TrainX-compatible
    # results.csv.

    make_trainx_results(
        source_results,
        target_results
    )

    # ========================================================
    # LAST.PT
    # ========================================================

    if source_last.exists():

        target_last = (
            target_weights
            / "last.pt"
        )

        # IMPORTANT:
        #
        # This is the section that fixes your error.
        #
        # Before:
        #
        # shutil.copy2(source_last, target_last)
        #
        # That failed because source_last and target_last
        # were the same file.
        #
        # Now safe_copy_file() checks first.

        safe_copy_file(
            source_last,
            target_last
        )

    else:

        print(
            "[trainx] WARNING: "
            "last.pt was not produced by Ultralytics."
        )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    if not target_best.exists():

        raise RuntimeError(
            f"TrainX artifact missing: {target_best}"
        )

    if not target_results.exists():

        raise RuntimeError(
            f"TrainX artifact missing: {target_results}"
        )

    # ========================================================
    # SUCCESS
    # ========================================================

    print()
    print(
        "=========================================="
    )

    print(
        "[trainx] TRAINING SUCCESS"
    )

    print(
        f"[trainx] best.pt: "
        f"{target_best}"
    )

    print(
        f"[trainx] results.csv: "
        f"{target_results}"
    )

    if (
        target_weights
        / "last.pt"
    ).exists():

        print(
            f"[trainx] last.pt: "
            f"{target_weights / 'last.pt'}"
        )

    print(
        "=========================================="
    )

    return 0


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    raise SystemExit(
        main()
    )
