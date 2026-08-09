# pigeon-reid 計畫（優先用現成 GitHub）

目標路徑：`D:\_myproject_WebsitePorjects\pigeon-reid`  
環境：**CPU**  
進度：**Phase 2 完成**（2026-08-09）→ 下一步 Phase 3（延後）

---

## 目標拆成三層


| 層級  | 需求         | 做法                             |
| --- | ---------- | ------------------------------ |
| A   | 辨識有沒有鴿子    | 現成 YOLO 鴿／鳥偵測專案                |
| B   | Webcam 即時  | Ultralytics / 現成 webcam script |
| C   | 少樣本判斷是否同一隻 | **延後**（先不做 Phase 3）            |


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



## 落地架構（clone + README，不自幹骨架）

目前範圍（Phase 1–2）：

```
Webcam
  → Ultralytics YOLO（data/models/pigeon.pt，單類 Pigeon）
  → 畫框：有沒有鴿子
```

本機目錄：

```
pigeon-reid/
  .venv/           # 本機 venv（gitignore）
  vendor/          # 可選：clone 進來的參考 repo
  data/models/     # YOLO .pt 權重
  data/gallery/    # Phase 3 用（目前空著）
  data/samples/    # 本機煙霧測試圖（gitignore）
  requirements.txt
  README.md        # 怎麼安裝與跑 + 實作備註
  PLAN.md          # 本計畫 + 階段實作記錄
```

**不做**自寫 `pigeon_reid/` Python package；指令與流程寫在 README。

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

### Phase 3 — 少樣本同一隻（**延後，先不做**）

之後若要做，可：

- clone `wildlife-tools`，用 MegaDescriptor
- 或直接抄 Camel Re-ID 的 YOLO+MegaDescriptor pipeline
- `register`：丟 5–20 張童鴿照片
- webcam／單圖：輸出是否同一隻 + 相似度
- 驗收：註冊後再拍，能對上；陌生鴿顯示 unknown

---



## 不做／延後

- 驅鳥硬體（雷射、水槍、Raspberry Pi）
- 從頭訓練大型 Re-ID 模型
- 網站／雲端（目前本機 CPU 優先）

---



## 風險與預期


| 項目             | 預期                    |
| -------------- | --------------------- |
| 偵測 + webcam    | 高，現成很多                |
| CPU 即時 FPS     | 偏低（數 FPS～十來 FPS）      |
| 少樣本同一隻         | 中；角度／光線差會誤判，需多角度照片    |
| 單一 GitHub 全包三點 | **幾乎沒有**，要組 2～3 個現成專案 |


---



## 已拍板

1. **偵測底層**：**Ultralytics + Roboflow**
2. **Re-ID**：**先不做 Phase 3**（之後可加 wildlife-tools + MegaDescriptor）
3. **專案形態**：**clone + README**（不保留自寫 `pigeon_reid/` 骨架）

