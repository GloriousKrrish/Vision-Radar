# VisionRadar Phase 9B & 9D — Production Model Baseline & Class Mapping Report

## 1. Production Model Frozen Signature (`BASELINE_V1`)

- **Model Path**: `data/models/yolox_nano.onnx`
- **SHA256 Hash**: `c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d`
- **Architecture**: YOLOX-Nano ONNX (8400 anchor grid decoding)
- **Input Resolution**: `640 x 640` (NCHW layout: `[1, 3, 640, 640]`)
- **Preprocessing**: Letterbox resize with padding, BGR to RGB, standard YOLOX normalization
- **Confidence Threshold**: `0.25`
- **NMS IoU Threshold**: `0.45`
- **Execution Engine**: OpenCV DNN CPU (`DNN_BACKEND_OPENCV` / `DNN_TARGET_CPU`)
- **Average CPU Latency**: `127.68 ms` per frame
- **Average AI Inference FPS**: `7.83 FPS`

---

## 2. Production Class Mapping & Vehicle Taxonomy (`Phase 9D`)

Production YOLOX-Nano is trained on COCO vehicle classes. To ensure accurate evaluation without silently misinterpreting predictions or ground truth, the following explicit mapping is enforced:

| Dataset Class | Dataset Source | Production Class Target | Mapping Status | Rationale |
| :--- | :--- | :--- | :---: | :--- |
| `car` | vehicle-detection-project / Roboflow | `Car` (COCO ID 2) | **EXACT** | Direct 1-to-1 semantic match for passenger cars, sedans, hatchbacks. |
| `bus` | vehicle-detection-project / Roboflow | `Bus` (COCO ID 5) | **EXACT** | Direct 1-to-1 match for transit & commercial buses. |
| `truck` | vehicle-detection-project / Roboflow | `Truck` (COCO ID 7) | **EXACT** | Direct 1-to-1 match for heavy freight vehicles and pickup trucks. |
| `motorcycle` | vehicle-detection-project / Roboflow | `Motorcycle` (COCO ID 3) | **EXACT** | Direct 1-to-1 match for two-wheeled motorized vehicles. |
| `van` | DETRAC / Roboflow fine-tuning | `Car` / `Truck` | **MAPPED** | Vans are structurally evaluated as `Car` (light vans) or `Truck` (box vans) in COCO taxonomy. |
| `auto-rickshaw` | Indian Traffic Dataset (ITD) | `Motorcycle` / `Car` | **EXCLUDED** | Three-wheeled auto-rickshaws do not exist in COCO 80-class taxonomy. Evaluated separately in domain failure mining. |
| `bicycle` | Standard COCO / Roboflow | `N/A` | **EXCLUDED** | Non-motorized vehicles excluded from motor vehicle traffic speed monitoring. |
| `person` | Standard COCO | `N/A` | **EXCLUDED** | Pedestrians excluded from road vehicle tracking evaluation. |
| `other` | UA-DETRAC | `N/A` | **EXCLUDED** | Ambiguous ground truth label; excluded to preserve quantitative rigor. |

---

## 3. Strict Evaluation Guidelines
- No class is silently mapped or merged.
- Excluded classes are documented upfront so model accuracy is evaluated strictly on compatible motor vehicle categories.
- Baseline configuration `BASELINE_V1` remains frozen throughout Phase 9.