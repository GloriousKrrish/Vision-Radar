# VisionRadar — Phase 7 Speed Ground Truth Requirements Specification

**Date:** 2026-09-26  
**Auditor:** Antigravity AI Perception Team  
**Local Availability Status:** **NOT LOCALLY AVAILABLE** (Accuracy Validation BLOCKED BY GROUND TRUTH)  

---

## 1. Executive Summary & Forensic Audit Finding

A forensic audit of `DATASETS/BrnoCompSpeed-master` and all local directories confirmed that **no dataset in local storage contains raw high-resolution video streams and radar ground-truth speed measurements**.

- Source code repository: `DATASETS/BrnoCompSpeed-master` is present.
- Raw MP4/AVI videos: **0 found**.
- Radar speed CSV/TXT annotations: **0 found**.

To prevent fake metric generation or arbitrary claim of calibrated accuracy, this document specifies the exact dataset structure, radar measurement hardware, camera geometry, and synchronization criteria required before speed estimation accuracy can be empirically validated in future phases.

---

## 2. Technical Data Requirements Matrix

| Dimension | Specification | Purpose & Rationale |
| :--- | :--- | :--- |
| **Media Stream** | 1080p @ 30 FPS / 60 FPS uncompressed MP4/AVI | Eliminates temporal frame drops and motion blur artifacting. |
| **Radar Hardware** | Doppler Speed Radar / LIDAR Gun (accuracy $\pm 0.1\text{ km/h}$) | Provides authoritative metric ground-truth speed. |
| **Camera Geometry** | Calibrated 3D camera extrinsics & intrinsics (height, pitch, roll, focal length) | Enables exact 3D-to-world road plane projection. |
| **Temporal Synchronization** | Microsecond-level hardware timestamp sync between camera and radar | Ensures radar speed measurement matches exact video frame $t$. |
| **Vehicle Correspondence** | Unique vehicle track IDs assigned to radar readings | Prevents multi-vehicle radar beam confusion. |

---

## 3. Required Ground-Truth Annotation Schema

Future candidate speed validation datasets MUST conform to or be adapted into the following JSON schema:

```json
{
  "sequence_id": "brno_highway_001",
  "fps": 30.0,
  "camera_calibration": {
    "camera_height_m": 8.5,
    "pitch_deg": 12.4,
    "image_points": [[200, 150], [600, 150], [750, 400], [50, 400]],
    "world_points": [[0, 50], [12, 50], [12, 0], [0, 0]]
  },
  "radar_ground_truth": [
    {
      "gt_vehicle_id": 14,
      "frame_index": 120,
      "timestamp_s": 4.0,
      "radar_speed_kmh": 64.2,
      "radar_speed_mps": 17.833,
      "radar_uncertainty_kmh": 0.1,
      "vehicle_class": "car",
      "bbox_xyxy": [450, 220, 560, 310]
    }
  ]
}
```

---

## 4. Benchmark Metrics for Empirical Validation

When radar speed ground truth becomes available in future phases, the perception stack must be benchmarked using:

1. **Mean Absolute Error (MAE)**: $\frac{1}{N} \sum |v_{\text{est}} - v_{\text{GT}}|$ ($\text{km/h}$).
2. **Root Mean Squared Error (RMSE)**: $\sqrt{\frac{1}{N} \sum (v_{\text{est}} - v_{\text{GT}})^2}$ ($\text{km/h}$).
3. **P95 Absolute Error**: 95th percentile error ($\text{km/h}$).
4. **Estimation Bias**: Mean error $\frac{1}{N} \sum (v_{\text{est}} - v_{\text{GT}})$.
5. **Empirical Uncertainty Coverage**: Percentage of GT samples falling inside $\pm 1\sigma, \pm 2\sigma, \pm 3\sigma$ predicted uncertainty intervals.
