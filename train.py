"""
TrainX training entry point for a YOLO detector.

This script is called by the TrainX orchestrator with:

    --img
    --batch
    --epochs
    --data
    --weights
    --project
    --name
    --seed
    --exist-ok

The script uses Ultralytics YOLO and produces the artifacts
expected by TrainX:

    <project>/<name>/results.csv
    <project>/<name>/weights/best.pt
    <project>/<name>/weights/last.pt

The dataset path and class labels are supplied by TrainX.
They are NOT hard-coded in this script.
"""

import argparse
import csv
import os
import random
import shutil
import sys
from pathlib import Path


def parse_args():
    """Parse command-line arguments supplied by TrainX."""

    parser = argparse.ArgumentParser(
        description="TrainX YOLO training entry point"
    )

    parser.add_argument(
        "--img",
        "--imgsz",
        type=int,
        default=320,
        help="Training image size",
    )

    parser.add_argument(
        "--batch",
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to dataset YAML supplied by TrainX",
    )

    parser.add_argument(
        "--weights",
        required=True,
        help="Path to initial YOLO weights",
    )

    parser.add_argument(
        "--project",
        required=True,
        help="Training project/output directory",
    )

    parser.add_argument(
        "--name",
        required=True,
        help="Training run name",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )

    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow existing output directory",
    )

    return parser.parse_args()


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducible training."""

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
    Return the first valid numeric value found for the supplied
    CSV column names.
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
    Convert the Ultralytics results.csv into the metric format
    expected by TrainX.
    """

    if not ultra_csv.exists():
        raise RuntimeError(
            f"Ultralytics results.csv was not found: {ultra_csv}"
        )

    with ultra_csv.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        rows = list(
            csv.DictReader(file)
        )

    if not rows:
        raise RuntimeError(
            f"Ultralytics produced an empty results.csv: {ultra_csv}"
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
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in rows:

            output = {
                field: ""
                for field in fields
            }

            output["epoch"] = row.get(
                "epoch",
                "",
            )

            output["train/box_loss"] = _num(
                row,
                "train/box_loss",
            )

            output["train/obj_loss"] = _num(
                row,
                "train/obj_loss",
                "train/dfl_loss",
            )

            output["train/cls_loss"] = _num(
                row,
                "train/cls_loss",
            )

            output["metrics/precision"] = _num(
                row,
                "metrics/precision(B)",
                "metrics/precision",
            )

            output["metrics/recall"] = _num(
                row,
                "metrics/recall(B)",
                "metrics/recall",
            )

            output["metrics/mAP_0.5"] = _num(
                row,
                "metrics/mAP50(B)",
                "metrics/mAP_0.5",
            )

            output["metrics/mAP_0.5:0.95"] = _num(
                row,
                "metrics/mAP50-95(B)",
                "metrics/mAP_0.5:0.95",
            )

            output["val/box_loss"] = _num(
                row,
                "val/box_loss",
            )

            output["val/obj_loss"] = _num(
                row,
                "val/obj_loss",
                "val/dfl_loss",
            )

            output["val/cls_loss"] = _num(
                row,
                "val/cls_loss",
            )

            output["x/lr0"] = _num(
                row,
                "lr/pg0",
                "x/lr0",
            )

            output["x/lr1"] = _num(
                row,
                "lr/pg1",
                "x/lr1",
            )

            output["x/lr2"] = _num(
                row,
                "lr/pg2",
                "x/lr2",
            )

            writer.writerow(output)


def is_same_file(
    source: Path,
    target: Path,
) -> bool:
    """
    Check whether source and target refer to the same file.

    This prevents shutil.SameFileError when Ultralytics has already
    created the artifact directly inside the TrainX run directory.
    """

    try:
        return source.resolve() == target.resolve()
    except OSError:
        return os.path.abspath(
            source
        ) == os.path.abspath(
            target
        )


def copy_if_needed(
    source: Path,
    target: Path,
) -> None:
    """
    Copy source to target only when they are different files.
    """

    if not source.exists():
        raise FileNotFoundError(
            f"Source artifact does not exist: {source}"
        )

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if is_same_file(
        source,
        target,
    ):
        print(
            "[trainx] Artifact already exists at target:"
        )
        print(
            f"[trainx] {target}"
        )
        return

    print(
        "[trainx] Copying artifact:"
    )

    print(
        f"[trainx] source = {source}"
    )

    print(
        f"[trainx] target = {target}"
    )

    shutil.copy2(
        source,
        target,
    )


def main():
    """Main TrainX training workflow."""

    args = parse_args()

    # ---------------------------------------------------------
    # 1. Set random seed
    # ---------------------------------------------------------

    seed_everything(
        args.seed
    )

    # ---------------------------------------------------------
    # 2. Determine TrainX run directory
    # ---------------------------------------------------------

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
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # 3. Validate input files
    # ---------------------------------------------------------

    data_path = Path(
        args.data
    )

    weights_path = Path(
        args.weights
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset YAML was not found: {data_path}"
        )

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights were not found: {weights_path}"
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
            "Install ultralytics in the shared training "
            "environment.",
            file=sys.stderr,
        )

        raise SystemExit(1) from exc

    # ---------------------------------------------------------
    # 5. Print training configuration
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("Starting YOLO training")
    print("=" * 70)

    print(
        f"[trainx] data={data_path}"
    )

    print(
        f"[trainx] weights={weights_path}"
    )

    print(
        f"[trainx] img={args.img}"
    )

    print(
        f"[trainx] batch={args.batch}"
    )

    print(
        f"[trainx] epochs={args.epochs}"
    )

    print(
        f"[trainx] seed={args.seed}"
    )

    print(
        f"[trainx] project={args.project}"
    )

    print(
        f"[trainx] name={args.name}"
    )

    print(
        f"[trainx] output={run_dir}"
    )

    print("=" * 70)
    print()

    # ---------------------------------------------------------
    # 6. Load YOLO model
    # ---------------------------------------------------------

    print(
        "[trainx] Loading YOLO model..."
    )

    model = YOLO(
        str(weights_path)
    )

    # ---------------------------------------------------------
    # 7. Start training
    # ---------------------------------------------------------

    print(
        "[trainx] Starting model.train()..."
    )

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
    # 8. Determine actual Ultralytics output directory
    # ---------------------------------------------------------

    source_run_dir = Path(
        model.trainer.save_dir
    )

    print()
    print("=" * 70)
    print("YOLO training completed")
    print("=" * 70)

    print(
        f"[trainx] Ultralytics output: {source_run_dir}"
    )

    print(
        f"[trainx] TrainX output:      {run_dir}"
    )

    print("=" * 70)
    print()

    # ---------------------------------------------------------
    # 9. Locate generated artifacts
    # ---------------------------------------------------------

    source_weights = (
        source_run_dir
        / "weights"
    )

    source_best = (
        source_weights
        / "best.pt"
    )

    source_last = (
        source_weights
        / "last.pt"
    )

    source_results = (
        source_run_dir
        / "results.csv"
    )

    target_weights = (
        run_dir
        / "weights"
    )

    target_weights.mkdir(
        parents=True,
        exist_ok=True,
    )

    target_best = (
        target_weights
        / "best.pt"
    )

    target_last = (
        target_weights
        / "last.pt"
    )

    target_results = (
        run_dir
        / "results.csv"
    )

    # ---------------------------------------------------------
    # 10. Validate best.pt
    # ---------------------------------------------------------

    if not source_best.exists():

        raise RuntimeError(
            "Training completed but best.pt was not found.\n"
            f"Expected location: {source_best}"
        )

    print(
        f"[trainx] Found best.pt: {source_best}"
    )

    # ---------------------------------------------------------
    # 11. Validate results.csv
    # ---------------------------------------------------------

    if not source_results.exists():

        raise RuntimeError(
            "Training completed but results.csv was not found.\n"
            f"Expected location: {source_results}"
        )

    print(
        f"[trainx] Found results.csv: {source_results}"
    )

    # ---------------------------------------------------------
    # 12. Copy best.pt only when necessary
    # ---------------------------------------------------------

    copy_if_needed(
        source_best,
        target_best,
    )

    # ---------------------------------------------------------
    # 13. Convert Ultralytics results.csv
    # ---------------------------------------------------------

    print(
        "[trainx] Creating TrainX results.csv..."
    )

    make_trainx_results(
        source_results,
        target_results,
    )

    print(
        f"[trainx] TrainX results.csv created: "
        f"{target_results}"
    )

    # ---------------------------------------------------------
    # 14. Handle last.pt safely
    #
    # This is the important fix for the previous error.
    # ---------------------------------------------------------

    if source_last.exists():

        print(
            f"[trainx] Found last.pt: {source_last}"
        )

        copy_if_needed(
            source_last,
            target_last,
        )

    else:

        print(
            "[trainx] WARNING: last.pt was not produced."
        )

    # ---------------------------------------------------------
    # 15. Final artifact validation
    # ---------------------------------------------------------

    print()
    print(
        "[trainx] Validating final artifacts..."
    )

    if not target_best.exists():

        raise RuntimeError(
            "Final artifact validation failed: "
            f"best.pt does not exist at {target_best}"
        )

    if not target_results.exists():

        raise RuntimeError(
            "Final artifact validation failed: "
            f"results.csv does not exist at {target_results}"
        )

    # last.pt is optional.
    last_status = (
        "present"
        if target_last.exists()
        else "not present"
    )

    # ---------------------------------------------------------
    # 16. Print final success
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINX YOLO TRAINING SUCCESS")
    print("=" * 70)

    print(
        f"[trainx] best.pt     : {target_best}"
    )

    print(
        f"[trainx] results.csv : {target_results}"
    )

    print(
        f"[trainx] last.pt     : {last_status}"
    )

    print("=" * 70)
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
