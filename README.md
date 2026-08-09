# pigeon-reid

Local **CPU** pigeon detection with webcam, built from existing tools (not a custom package).

| Layer | Status | Stack |
|-------|--------|--------|
| Environment | **Phase 0 done** (2026-08-09) | Python 3.10 venv, CPU torch, ultralytics, opencv |
| Detect pigeon | Phase 1–2 | [Ultralytics](https://github.com/ultralytics/ultralytics) + [Roboflow](https://universe.roboflow.com/) |
| Webcam | Phase 1 | `yolo predict ... source=0` |
| Same-bird Re-ID | Deferred | wildlife-tools / MegaDescriptor later |

See [PLAN.md](PLAN.md) for the full plan and implementation notes.

## Layout

```
pigeon-reid/
  .venv/            # local venv (gitignored)
  vendor/           # optional cloned reference repos
  data/models/      # YOLO weights (.pt)
  data/gallery/     # reserved for Phase 3
  requirements.txt
  README.md
  PLAN.md
```

## Phase 0 — Setup (Windows / CPU) — done

Verified on this machine (2026-08-09):

- Python **3.10.6**, venv at `.venv/`
- `torch 2.13.0+cpu`, `torchvision 0.28.0+cpu` (`cuda_available=False`)
- `ultralytics 8.4.116`, `opencv-python 5.0.0`
- Webcam `source=0` opens and reads frames (`480×640×3`)

```powershell
cd D:\_myproject_WebsitePorjects\pigeon-reid
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics opencv-python
```

Install **CPU torch first**, then ultralytics/opencv, so pip does not pull a CUDA wheel.

If PyPI fails with `SSL: CERTIFICATE_VERIFY_FAILED` (corporate / self-signed proxy), add trusted hosts:

```powershell
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org --trusted-host download-r2.pytorch.org torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org ultralytics opencv-python
```

Quick webcam check (no YOLO yet):

```powershell
python -c "import cv2; c=cv2.VideoCapture(0); print(c.isOpened()); r,f=c.read(); print(r, None if f is None else f.shape); c.release()"
```

If the first grab fails, retry once or warm up a few frames; DSHOW also works on Windows (`cv2.VideoCapture(0, cv2.CAP_DSHOW)`).

## Phase 1 — Webcam smoke test (bird / pigeon) — not started

Quick check with a COCO pretrained nano model (class includes `bird`):

```powershell
yolo predict model=yolov8n.pt source=0 device=cpu conf=0.25 show=True
```

- `source=0` = default webcam  
- Press `q` in the preview window to quit  
- Accept: birds/pigeons in frame get boxes (may also fire on other COCO classes)

Optional reference clone (webcam / Flask examples):

```powershell
git clone https://github.com/Aamer-Gituser/Bird-Detection.git vendor/Bird-Detection
```

Swap their weights for a pigeon `.pt` when you have one.

## Phase 2 — Pigeon-specific weights (Roboflow) — not started

1. Open a dataset, e.g.  
   - https://universe.roboflow.com/yolov8-group-1/pigeon-detection-l2ivt  
   - https://universe.roboflow.com/pigeon-cn3z7/pigeons-detection-rm5js  
2. Download **YOLOv8** format (or export a trained model / `.pt` if the project provides one).  
3. Place weights at `data/models/pigeon.pt` (or keep the Roboflow export folder and point `model=` at the `.pt`).

Webcam with pigeon weights:

```powershell
yolo predict model=data/models/pigeon.pt source=0 device=cpu conf=0.25 show=True
```

Fine-tune on your own export (example):

```powershell
yolo detect train data=path/to/data.yaml model=yolov8n.pt epochs=50 imgsz=640 device=cpu
# then copy best.pt -> data/models/pigeon.pt
```

## Phase 3 — Same individual (deferred) — not started

Not in scope yet. Later: crop detections → MegaDescriptor / [wildlife-tools](https://github.com/WildlifeDatasets/wildlife-tools) → compare to `data/gallery/<id>/`.

## Remote

```text
https://github.com/MagicDogGuo/pigeon-reid.git
```
