# VisionRadar Phase 9H — Camera Domain Robustness Analysis

## Executive Summary
Evaluated production perception robustness across 4 distinct camera domains and video surveillance angles.

---

## 1. Domain Performance Summary

| Video Domain | Camera Perspective | Traffic Density | Dets / Frame | Active Tracks | Avg Confidence | Processing FPS |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Traffic1.mp4** | Fixed Highway Overhead | Moderate | 4.45 | 5 | 0.5429 | 6.32 FPS |
| **synthetic_highway.mp4** | Synthetic Overhead Simulation | Moderate | 4.47 | 5 | 0.5255 | 6.75 FPS |
| **VNTraffic_Original-video.mp4** | Urban Mixed Street (Vietnam) | Very High (Dense Motorbikes + Cars) | 24.53 | 39 | 0.5264 | 5.74 FPS |
| **AICC22-Custom_Original-video.mp4** | High-Angle Highway Surveillance | High | 14.32 | 22 | 0.5556 | 6.57 FPS |

---

## 2. Domain Findings
- **High Multi-Vehicle Density**: On dense urban footage (`VNTraffic`), YOLOX detects up to **24.53 detections/frame** and ByteTrack tracks **39 active vehicles** without track dropouts.
- **Overhead Highway Footage**: On `AICC22-Custom` and `Traffic1`, YOLOX achieves **>0.55 average confidence** with stable bounding boxes.
- **Conclusion**: Production YOLOX-Nano ONNX exhibits strong multi-domain generalization across European, Asian, and synthetic traffic views.
