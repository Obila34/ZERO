# Fine-tuning YOLO on your room

Generic COCO weights miss/mislabel the objects that matter in YOUR space.
200–500 labelled photos of your actual room, objects and lighting reliably
beats the stock model there. Four steps:

## 1. Collect frames (on the Pi / any box with the camera)

```bash
python scripts/capture_dataset.py --out dataset/images --count 300
```

Vary everything between shots: object positions, lighting (day/night/lamp),
camera angle, distances, clutter. Include the objects ZERO currently gets
wrong, from several viewpoints each.

## 2. Label

Any YOLO-format tool works — Roboflow (easiest, browser), labelImg, or CVAT.
Draw boxes, name classes. Export as **YOLO (txt per image)** with a
`data.yaml` listing your classes. Tip: keep COCO names for classes COCO
already knows (person, cup, ...) and add your custom ones after.

## 3. Train (on the GPU node)

```bash
pip install ultralytics
yolo detect train model=yolo11s.pt data=path/to/data.yaml \
     imgsz=640 epochs=80 batch=16
```

Starting from `yolo11s.pt` (transfer learning) converges fast; check
`runs/detect/train*/results.png` — mAP50 above ~0.8 on your validation split
is plenty for remark-grade detection.

## 4. Export + point ZERO at it

```bash
python scripts/export_yolo_onnx.py --model runs/detect/train/weights/best.pt
```

Then in `config.yaml`:

```yaml
vision:
  detect:
    model_path: best.onnx   # your fine-tuned model
    imgsz: 640              # must match the export
```

The exporter writes a sibling `<model>.names.json` with your class names —
the detector auto-loads it. Keep `yolo11s.onnx` around; swapping back is a
one-line config change.

**Note (GPU offload):** with `perception.remote.enabled: true` the ambient
loop uses the server's YOLO11x; your fine-tuned model then serves as the
local fallback. To make the fine-tuned model primary everywhere, also swap
the checkpoint the vision server loads (see `server/vision/`), or set
`perception.remote.enabled: false` to run detection purely local.
