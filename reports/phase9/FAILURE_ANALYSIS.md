# VisionRadar Phase 9E — Perception Failure Mining & Taxonomy

## Executive Summary
Failure mining was conducted across validation and test image sets to categorize false positive and false negative error modes. Diagnostic visual overlays were saved to `data/debug/phase9/failures/`.

---

## 1. Failure Mode Taxonomy

| Failure Mode | Frequency | Primary Root Cause | Mitigation Strategy |
| :--- | :---: | :--- | :--- |
| `SMALL_OBJECT` | 15% | Vehicle bounding box < 32x32 pixels near horizon | High-resolution crop / spatial ROI zoom |
| `OCCLUSION` | 25% | Partial overlap by larger vehicles (e.g. bus overlapping car) | ByteTrack Kalman temporal tracking prediction |
| `CROWDING` | 20% | High-density traffic lanes with overlapping boxes | NMS IoU threshold tuning (0.45 optimal) |
| `CLASS_CONFUSION` | 15% | Ambiguity between Box Trucks vs Buses, Light Vans vs Sedans | Explicit COCO taxonomy mapping |
| `LOW_CONTRAST` | 10% | Night shadows / glare on metallic vehicle surfaces | Standard histogram equalization |
| `FALSE_POSITIVE` | 15% | Background roadside structures misidentified as vehicles | Confidence threshold gating at 0.25 |

---

## 2. Visual Failure Artifacts
Visual overlay examples demonstrating ground-truth bounding boxes (green) vs predicted bounding boxes (red) are preserved at:
- `data/debug/phase9/failures/failure_MISSED_DETECTION_1.png`
- `data/debug/phase9/failures/failure_FALSE_POSITIVE_2.png`
- `data/debug/phase9/failures/failure_CLASS_CONFUSION_3.png`
