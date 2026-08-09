# pigeon-reid 計畫（優先用現成 GitHub）

目標路徑：`D:\_myproject_WebsitePorjects\pigeon-reid`  
環境：**CPU**  
進度：**Phase 3b 完成**（2026-08-09）→ 下一步 **3c**（`saver.py` 自動存照）

---

## 目標拆成三層


| 層級  | 需求         | 做法                             |
| --- | ---------- | ------------------------------ |
| A   | 辨識有沒有鴿子    | 現成 YOLO 鴿／鳥偵測專案                |
| B   | Webcam 即時  | Ultralytics / 現成 webcam script |
| C   | 少樣本判斷是否同一隻 | **延後**（Phase 4）                 |


---

## 建議採用的現成專案

### A. 鴿子／鳥偵測（必做）— **已拍板：Ultralytics + Roboflow**

- 框架：[ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- 資料／權重：[Roboflow 鴿子資料集](https://universe.roboflow.com/yolov8-group-1/pigeon-detection-l2ivt) 或 [Pigeons Detection](https://universe.roboflow.com/pigeon-cn3z7/pigeons-detection-rm5js)
- Webcam：`yolo predict model=xxx.pt source=0 device=cpu`

備援參考（不採用為主幹）：

1. **[jpvoelz/pigeon-detector](https://github.com/jpvoelz/pigeon-detector)** — YOLOv7 + 鴿子資料
2. **[MrAI-NDHU/KeepPigeonsAway](https://github.com/MrAI-NDHU/KeepPigeonsAway)** — YOLOv3-tiny + webcam（較舊）



### B. Webcam 即時（必做）

可直接參考：

- **[Aamer-Gituser/Bird-Detection](https://github.com/Aamer-Gituser/Bird-Detection)**
  - YOLOv8 + `source=0` webcam + 影片／Flask
  - 把偵測類別換成鴿子權重即可
  - 本機已 clone：`vendor/Bird-Detection`（gitignore，僅本機參考）



### C. 少樣本「是不是同一隻」（理想）

沒有現成「鴿子臉 App」，但可組：

1. **[WildlifeDatasets/wildlife-tools](https://github.com/WildlifeDatasets/wildlife-tools)** + **MegaDescriptor**
  - 動物個體 embedding 標準工具
  - 少張照片 → 建 gallery → cosine 比對
2. 流程範本（非鴿子，可抄架構）：
  - **[wadieeee/Camel-Re-Identification-in-Video-using-YOLOv8-and-MegaDescriptor](https://github.com/wadieeee/Camel-Re-Identification-in-Video-using-YOLOv8-and-MegaDescriptor)**
  - YOLO 裁切 → MegaDescriptor 比對
3. 學術小鳥個體辨識（可選參考）：
  - **[AndreCFerreira/Bird_individualID](https://github.com/AndreCFerreira/Bird_individualID)**

---



## 落地架構

### Phase 1–2（已完成）

```
Webcam
  → Ultralytics YOLO（data/models/pigeon.pt，單類 Pigeon）
  → 畫框：有沒有鴿子
```

### Phase 3（計畫中）

```
Webcam
  → Ultralytics YOLO（pigeon.pt）
  → 造訪計數 + 同時隻數 + 自動存照
  → 本機 Flask 網頁（MJPEG + 數字 + 最近照片）
```

本機目錄（Phase 3 完成後預期）：

```
pigeon-reid/
  .venv/           # 本機 venv（gitignore）
  vendor/          # 可選：clone 進來的參考 repo
  app/             # Phase 3：薄 Flask 應用（非大型 package）
    config.py      # 路徑、conf、visit_gap_sec、save_interval_sec
    detector.py
    counter.py
    saver.py
    web.py
  templates/       # index.html 監控頁
  static/          # 可選少量 CSS/JS
  data/models/     # YOLO .pt 權重
  data/captures/   # Phase 3：YYYY-MM-DD/*.jpg（gitignore）
  data/gallery/    # Phase 4 Re-ID 用（目前空著）
  data/samples/    # 本機煙霧測試圖（gitignore）
  requirements.txt
  README.md
  PLAN.md
```

Phase 1–2：**不做**自寫大型 package；以 README + YOLO CLI 為主。  
Phase 3：新增薄 `app/`（設定集中 `config.py` 注入），**不用 React**。

---



## 階段計畫



### Phase 0 — 環境 — **已完成（2026-08-09）**

- Python venv、CPU 版 torch、opencv、ultralytics
- 確認 webcam `source=0` 能開

實作記錄：


| 項目                | 結果                                           |
| ----------------- | -------------------------------------------- |
| Python            | 3.10.6                                       |
| venv              | `.venv/`                                     |
| torch             | `2.13.0+cpu`（`cuda_available=False`）         |
| torchvision       | `0.28.0+cpu`                                 |
| ultralytics       | `8.4.116`                                    |
| opencv-python     | `5.0.0`                                      |
| 目錄                | `data/models/`、`data/gallery/`（含 `.gitkeep`） |
| webcam `source=0` | 可開啟；讀幀成功 `480×640×3`（DEFAULT / DSHOW / MSMF） |


安裝注意（本機）：

- PyPI 遇 `SSL: CERTIFICATE_VERIFY_FAILED`（self-signed in chain）時，需加 `--trusted-host pypi.org --trusted-host files.pythonhosted.org`（以及 PyTorch index 對應 host）
- 安裝順序：先 CPU torch/torchvision，再 `ultralytics` + `opencv-python`
- 首次 `VideoCapture(0)` 可能短暫 grab 失敗，warm-up 數幀後可讀

細節與指令見 [README.md](README.md)。

### Phase 1 — 只做「有沒有鳥／鴿子 + webcam」 — **已完成（2026-08-09）**

- 用 COCO pretrained `yolov8n.pt`（含 `bird` class id **14**）先跑通
- 參考 clone：`vendor/Bird-Detection`
- 驗收：樣本圖有鳥會畫框；webcam 可跑 YOLO 推論

實作記錄：


| 項目      | 結果                                                                                             |
| ------- | ---------------------------------------------------------------------------------------------- |
| 權重      | `data/models/yolov8n.pt`（Ultralytics 自動下載後複製；`*.pt` gitignore）                                 |
| 樣本圖     | `data/samples/bird.jpg` → `1 bird` @ conf=0.25, `classes=14`, CPU ~108ms                       |
| Webcam  | DSHOW `source=0`，連續 8 幀 YOLO 推論成功（`480×640×3`）                                                 |
| 參考 repo | `git clone --depth 1` → `vendor/Bird-Detection`                                                |
| 即時預覽指令  | `yolo predict model=data/models/yolov8n.pt source=0 device=cpu conf=0.25 classes=14 show=True` |


參數意思（精簡；完整表見 [README.md](README.md) Phase 1）：


| 參數        | 意思                                                                                                                                    |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `model`   | 權重路徑（Phase 1：`yolov8n.pt`）                                                                                                            |
| `source`  | 輸入：webcam 編號（`0`/`1`…）或圖／影片路徑                                                                                                         |
| `device`  | 推論裝置（本專案 `cpu`）                                                                                                                       |
| `conf`    | 信心門檻；愈低框愈多                                                                                                                            |
| `classes` | 只保留的 COCO class id；`14` = bird [https://docs.ultralytics.com/datasets/detect/coco](https://docs.ultralytics.com/datasets/detect/coco) |
| `show`    | 開預覽視窗；按 `q` 結束                                                                                                                        |


注意：

- Phase 1 用 COCO **bird**，不是鴿子專用；鏡頭前沒鳥時可能仍偵測到其他 COCO 類別（人／椅等），可加 `classes=14` 只畫鳥
- 真正「只標鴿子」已在 **Phase 2**（`data/models/pigeon.pt`）
- CPU 即時 FPS 預期偏低

細節與指令見 [README.md](README.md)。

### Phase 2 — 換成「鴿子」專用 — **已完成（2026-08-09）**

- 採用現成 Roboflow 單類 **Pigeon** fine-tune 權重（非 COCO `bird`）
- 權重放到 `data/models/pigeon.pt`
- 驗收：只輸出 `Pigeon`；鴿群樣本可畫框；webcam 可推論

實作記錄：


| 項目      | 結果                                                                                                                          |
| ------- | --------------------------------------------------------------------------------------------------------------------------- |
| 權重      | `data/models/pigeon.pt`（~6.0 MB；`names={0: Pigeon}`）                                                                        |
| 來源      | [pigeon-yolov8-detection](https://github.com/Prajapatidhruv1206/pigeon-yolov8-detection) `best.pt`（Roboflow 標註資料 fine-tune） |
| 本機參考    | `vendor/pigeon-yolov8-detection`（gitignore）                                                                                 |
| 圖像煙霧    | `pigeon_flock.png` → **11** @ `conf=0.45`；`pigeon_closeup.png` → **5**；Phase 1 `bird.jpg` → **0**（非鴿專用）                    |
| Webcam  | DSHOW `source=0`，連續 8 幀 YOLO 推論成功（`480×640×3`）                                                                              |
| 建議 conf | **0.45**（過低易在非鴿場景誤報；例如 `person.jpg` 仍可能有 FP）                                                                               |
| 即時預覽指令  | `yolo predict model=data/models/pigeon.pt source=0 device=cpu conf=0.45 show=True`                                          |


參數差異（相對 Phase 1）：


| 參數        | Phase 1                         | Phase 2                                      |
| --------- | ------------------------------- | -------------------------------------------- |
| `model`   | `data/models/yolov8n.pt`（COCO）  | `data/models/pigeon.pt`（Pigeon-only）         |
| `classes` | `14`（COCO bird）                 | **不需要**（模型只有一類）                              |
| `conf`    | `0.25`                          | 建議 `0.45`                                    |


可選後續：用自家 Roboflow API key 下載 Universe 資料集再 `yolo detect train`，覆蓋 `pigeon.pt`。見 [README.md](README.md) Phase 2。

### Phase 3 — 本機網頁監控（Flask）— **進行中（3b 完成）**

目標：瀏覽器看即時畫面，顯示**今日造訪次數**與**目前同時隻數**，並在偵測到鴿子時自動拍照存資料夾。

#### 技術選擇（已拍板）

- **Flask + Jinja／簡單 HTML+JS**（本機 `localhost`）
- **不用 React**（YOLO／存圖本來就在 Python；本機監控頁不需 Node 雙進程）
- 即時畫面：MJPEG stream（可參考 `vendor/Bird-Detection`）

#### 計數規則（已定）

1. **同時隻數（live）**：當前幀 YOLO 框數；畫面上 3 隻就顯示 **3**
2. **今日造訪次數（visit）**：
   - `0 → N`：開始一次造訪，今日計數 **+N**（進場當幀同時隻數）
   - 連續 **N 秒**（預設 30s）偵測為 0：結束造訪
   - 同一造訪期間不重複加計（中途隻數增減不加）
3. **不認個體**（那是 Phase 4）；今日數字是「各波進場同時隻數加總」，不是經 Re-ID 的個體數

#### 資料流

```
Webcam → Detector → Counter → Flask UI
              ↓         ↓
            Stream    Saver → data/captures/YYYY-MM-DD/
```

- **Detector**：OpenCV DSHOW `source=0` + `data/models/pigeon.pt`（`conf=0.45`, CPU）
- **Counter**：同時隻數 + 造訪狀態機 + 依本機本地日切換
- **Saver**：造訪開始必存一張；造訪中每 M 秒（預設 10s）最多再存一張
- **Flask UI**：MJPEG、今日造訪、當前同時隻數、最近存檔縮圖

存檔範例：`data/captures/2026-08-09/visit003_20260809_171530_n2.jpg`

#### 小階段

| 小階段 | 內容 | 驗收 |
| --- | --- | --- |
| **3a** | `detector.py`：webcam + `pigeon.pt` 迴路（無 UI） | **完成**（2026-08-09）：有鴿 `boxes >= 1`，無鴿為 0 |
| **3b** | `counter.py`：`concurrent_count`、`visits_today`、gap=30s | **完成**（2026-08-09）：`0→N` 則 +N；同造訪不重複；離開 30s 再進再加；同時 3 顯示 3 |
| **3c** | `saver.py` → `data/captures/YYYY-MM-DD/`；gitignore | 造訪後有 JPEG；檔名含時間與隻數 |
| **3d** | Flask：`/`、`/video_feed`、`/api/stats`；`requirements.txt` 加 flask | `http://127.0.0.1:5000` 可看流與數字 |
| **3e** | 更新 README 啟動指令與參數；入口 `python -m app.web` | 文件可照做跑起來 |

#### 3a 實作記錄


| 項目 | 結果 |
| --- | --- |
| 檔案 | `app/config.py`（`AppConfig` / `CONFIG` 注入）、`app/detector.py` |
| API | `Detector(config).predict_frame(frame)` / `open`+`read` / `frames()` |
| 圖像煙霧 | `pigeon_flock.png` → **11**；`pigeon_closeup.png` → **5**；`bird.jpg` → **0** |
| Webcam | DSHOW `source=0`，連續 8 幀推論成功（`480×640×3`，現場無鴿 → boxes=0） |
| 無 UI 迴路 | `python -m app.detector`（Ctrl+C 結束） |
| 已知限制 | 非鴿圖（如 `person.jpg`）在 `conf=0.45` 仍可能有 FP（同 Phase 2） |

#### 3b 實作記錄


| 項目 | 結果 |
| --- | --- |
| 檔案 | `app/counter.py`（`VisitCounter` / `VisitStats`） |
| API | `VisitCounter(config, now_fn=…).update(box_count) → VisitStats` |
| 規則 | `0→N` 開始造訪 **+N**；同造訪不重複；連續 `visit_gap_sec`（預設 30）為 0 結束；本機本地日切換重設 |
| 注入 | `visit_gap_sec` 來自 `AppConfig`；可注入 `now_fn` 做確定性測試 |
| 煙霧 | `python -m app.counter` → 進畫面 3 隻 +3、同造訪不加成、+29s 仍在訪、+30s 結束、再進 1 隻 +1、跨日歸零 → **OK** |

#### Phase 3 刻意不做

- React / Node 前端
- 登入、雲端、多使用者、公開部署
- 個體 Re-ID（Phase 4）
- 改動 Phase 0–2 已驗證的權重與 YOLO CLI 流程

### Phase 4 — 少樣本同一隻（**延後，先不做**）

（原 Phase 3 Re-ID，整段後移。）

之後若要做，可：

- clone `wildlife-tools`，用 MegaDescriptor
- 或直接抄 Camel Re-ID 的 YOLO+MegaDescriptor pipeline
- `register`：丟 5–20 張童鴿照片到 `data/gallery/<id>/`
- webcam／單圖：輸出是否同一隻 + 相似度
- 驗收：註冊後再拍，能對上；陌生鴿顯示 unknown

---



## 不做／延後

- 驅鳥硬體（雷射、水槍、Raspberry Pi）
- 從頭訓練大型 Re-ID 模型
- **雲端／公開部署**（本機 Flask OK；不做對外網站）
- Phase 4 少樣本同一隻（延後）

---



## 風險與預期


| 項目             | 預期                    |
| -------------- | --------------------- |
| 偵測 + webcam    | 高，現成很多                |
| CPU 即時 FPS     | 偏低（數 FPS～十來 FPS）      |
| 本機 Flask 監控頁   | 高；MJPEG + 計數狀態機可落地     |
| 造訪次數 ≠ 經 Re-ID 個體數 | 預期內；現為進場同時隻數加總；真正認隻要等 Phase 4 |
| 少樣本同一隻         | 中；角度／光線差會誤判，需多角度照片    |
| 單一 GitHub 全包三點 | **幾乎沒有**，要組 2～3 個現成專案 |


---



## 已拍板

1. **偵測底層**：**Ultralytics + Roboflow**
2. **Phase 3 UI**：**本機 Flask**（不用 React）；造訪次數 + 同時隻數 + 自動存照
3. **Re-ID**：**延後為 Phase 4**（之後可加 wildlife-tools + MegaDescriptor）
4. **專案形態**：Phase 1–2 以 clone + README／CLI 為主；Phase 3 新增薄 `app/`（非大型 package）
