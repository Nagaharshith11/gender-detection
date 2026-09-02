"""
TrainX export entry point for the trained YOLO model.

TrainX calls:
  python export.py --weights <run_dir>/weights/best.pt --img <size> --include onnx --data <yaml>

The ONNX file must be written next to best.pt as best.onnx.
"""
import argparse
import shutil
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="TrainX YOLO model exporter")
    p.add_argument("--weights", required=True)
    p.add_argument("--img", "--imgsz", type=int, default=320)
    p.add_argument("--include", nargs="+", default=["onnx"])
    p.add_argument("--data")
    return p.parse_args()


def main():
    args = parse_args()
    weights = Path(args.weights).resolve()
    if not weights.exists():
        print(f"ERROR: weights not found: {weights}", file=sys.stderr)
        return 1

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print("ERROR: ultralytics is required for export.", file=sys.stderr)
        raise SystemExit(1) from exc

    model = YOLO(str(weights))
    formats = {fmt.lower() for fmt in args.include}

    if "onnx" in formats:
        exported = model.export(format="onnx", imgsz=args.img, data=args.data)
        exported = Path(exported).resolve()
        target = weights.with_suffix(".onnx")
        if exported != target:
            shutil.copy2(exported, target)
        if not target.exists():
            raise RuntimeError(f"ONNX export did not create {target}")
        print(f"[trainx] ONNX exported: {target}")

    # Optional formats are best-effort in the TrainX contract.
    if "openvino" in formats:
        try:
            model.export(format="openvino", imgsz=args.img, data=args.data)
        except Exception as exc:
            print(f"[trainx] OpenVINO export warning: {exc}", file=sys.stderr)

    if "tflite" in formats:
        try:
            model.export(format="tflite", imgsz=args.img, data=args.data)
        except Exception as exc:
            print(f"[trainx] TFLite export warning: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
