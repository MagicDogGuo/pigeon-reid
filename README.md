# pigeon-reid

Local **CPU** pigeon detection with webcam, built from existing tools (not a custom package).

| Layer | Status | Stack |
|-------|--------|--------|
| Environment | **Phase 0 done** (2026-08-09) | Python 3.10 venv, CPU torch, ultralytics, opencv |
| Detect bird (COCO) | **Phase 1 done** (2026-08-09) | `yolov8n.pt` + webcam `source=0` |
| Detect pigeon | **Phase 2 done** (2026-08-09) | `pigeon.pt` (1-class Roboflow fine-tune) + webcam |
| Same-bird Re-ID | Deferred | wildlife-tools / MegaDescriptor later |

See [PLAN.md](PLAN.md) for the full plan and implementation notes.

## Layout

```
pigeon-reid/
  .venv/            # local venv (gitignored)
  vendor/           # optional cloned reference repos (gitignored contents)
  data/models/      # YOLO weights (.pt, gitignored)
  data/gallery/     # reserved for Phase 3
  data/samples/     # local smoke-test images (gitignored)
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

## Phase 1 — Webcam bird detect (COCO) — done

Verified on this machine (2026-08-09):

- Weights: `data/models/yolov8n.pt` (COCO; class **14** = `bird`)
- Image smoke: sample bird → **`1 bird`** @ `conf=0.25`, CPU ~108 ms
- Webcam: 8 frames inferred successfully via OpenCV DSHOW + Ultralytics
- Reference clone: `vendor/Bird-Detection`

### 1) Get the nano weights

```powershell
.\.venv\Scripts\Activate.ps1
python -c "from ultralytics import YOLO; import shutil; YOLO('yolov8n.pt'); shutil.copy2('yolov8n.pt', 'data/models/yolov8n.pt')"
```

### 2) Live webcam (bird boxes only)

```powershell
yolo predict model=data/models/yolov8n.pt source=0 device=cpu conf=0.25 classes=14 show=True
```

| Parameter | Example | Meaning |
|-----------|---------|---------|
| `model` | `data/models/yolov8n.pt` | Path to YOLO weights. Phase 1 uses COCO nano; later swap to `data/models/pigeon.pt`. |
| `source` | `0` | Input. Webcam index (`0` = first cam, `1` = second, …), or a file/folder path (image/video). |
| `device` | `cpu` | Inference device. This project uses CPU; use `0` for first CUDA GPU if available. |
| `conf` | `0.25` | Confidence threshold (0–1). Lower = more boxes (more false positives); higher = stricter. |
| `classes` | `14` | Only keep these COCO class ids. `14` = `bird`. Omit to show all COCO classes. |
| `show` | `True` | Open a live preview window. Press `q` to quit. |

Also useful later:

| Parameter | Example | Meaning |
|-----------|---------|---------|
| `save` | `True` | Write annotated frames/images under `runs/detect/`. |
| `imgsz` | `640` | Inference image size (default 640). Smaller can be faster on CPU. |

Accept: birds/pigeons in frame get boxes (still COCO-bird, not pigeon-specific).

### 3) Optional image smoke

```powershell
yolo predict model=data/models/yolov8n.pt source=data/samples/bird.jpg device=cpu conf=0.25 classes=14 save=True
```

### Optional reference clone

```powershell
git clone --depth 1 https://github.com/Aamer-Gituser/Bird-Detection.git vendor/Bird-Detection
```

Useful entry points there: `webcam_pilot.py`, Flask `app.py`. Phase 2 weights: `data/models/pigeon.pt`.

## Phase 2 — Pigeon-specific weights — done

Verified on this machine (2026-08-09):

- Weights: `data/models/pigeon.pt` — YOLOv8n fine-tuned on a Roboflow **Pigeon**-only dataset (`nc: 1`, class `Pigeon`)
- Source: [Prajapatidhruv1206/pigeon-yolov8-detection](https://github.com/Prajapatidhruv1206/pigeon-yolov8-detection) `runs/detect/train/weights/best.pt` (local clone: `vendor/pigeon-yolov8-detection`)
- Image smoke (`conf=0.45`): flock sample **11** pigeons; close-up **5**; Phase 1 `bird.jpg` **0** (not pigeon-specific enough); webcam **8** frames inferred on CPU

### 1) Install weights (already copied if you followed this machine’s setup)

```powershell
.\.venv\Scripts\Activate.ps1
git clone --depth 1 https://github.com/Prajapatidhruv1206/pigeon-yolov8-detection.git vendor/pigeon-yolov8-detection
Copy-Item vendor/pigeon-yolov8-detection/runs/detect/train/weights/best.pt data/models/pigeon.pt
```

### 2) Live webcam (pigeon boxes only)

No `classes=` filter needed — the model only knows `Pigeon`. Prefer **`conf=0.45`** (or higher) to cut false positives on non-pigeon scenes.

```powershell
yolo predict model=data/models/pigeon.pt source=0 device=cpu conf=0.45 show=True
```

| Parameter | Example | Meaning |
|-----------|---------|---------|
| `model` | `data/models/pigeon.pt` | Pigeon-only fine-tuned weights (not COCO). |
| `source` | `0` | Webcam index, or image/video path. |
| `device` | `cpu` | This project uses CPU. |
| `conf` | `0.45` | Confidence threshold. Higher = fewer boxes / fewer false positives. |
| `show` | `True` | Preview window; press `q` to quit. |

### 3) Optional image smoke

```powershell
yolo predict model=data/models/pigeon.pt source=data/samples/pigeon_flock.png device=cpu conf=0.45 save=True
```

### Optional: retrain from Roboflow Universe yourself

Needs a free [Roboflow API key](https://roboflow.com/). Example datasets:

- https://universe.roboflow.com/yolov8-group-1/pigeon-detection-l2ivt  
- https://universe.roboflow.com/pigeon-cn3z7/pigeons-detection-rm5js  

```powershell
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org roboflow
# then download YOLOv8 export via Roboflow SDK / CLI into a local folder
yolo detect train data=path/to/data.yaml model=yolov8n.pt epochs=50 imgsz=640 device=cpu
Copy-Item runs/detect/train/weights/best.pt data/models/pigeon.pt
```

Notes:

- Phase 2 labels **`Pigeon` only** (not COCO `bird` / person / chair).
- Some non-pigeon images can still get false positives at low `conf`; raise the threshold if needed.
- CPU FPS stays modest (same as Phase 1).

## Phase 3 — Same individual (deferred) — not started

Not in scope yet. Later: crop detections → MegaDescriptor / [wildlife-tools](https://github.com/WildlifeDatasets/wildlife-tools) → compare to `data/gallery/<id>/`.

## Remote

```text
https://github.com/MagicDogGuo/pigeon-reid.git
```
