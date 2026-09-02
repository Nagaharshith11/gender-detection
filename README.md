# TrainX Person Detection Model

This repository trains a YOLO object-detection model for **person detection**. It detects people with bounding boxes; it does not classify or infer sensitive attributes about people.

## Class

The dataset should contain one CVAT class:

```text
person
```

## TrainX contract

TrainX calls `train.py` with `--img`, `--batch`, `--epochs`, `--data`, `--weights`, `--project`, `--name`, `--seed`, and `--exist-ok`. The generated dataset YAML comes from the approved CVAT dataset; do not hard-code dataset paths in this repository.

The training entry point writes:

```text
<project>/<name>/results.csv
<project>/<name>/weights/best.pt
```

`results.csv` contains the YOLOv5-style metric names expected by the TrainX acceptance gate.

## Files

- `train.py` - YOLO training entry point.
- `export.py` - exports `best.pt` to `best.onnx` and supports optional OpenVINO/TFLite export.
- `requirements.txt` - Python dependencies.
- `dataset.yaml.example` - reference only; TrainX generates the real YAML.
- `classes.txt` - documents the single class.

## Git

Do not commit datasets, model binaries, or run outputs. TrainX publishes model/data artifacts through its configured artifact/versioning system.
