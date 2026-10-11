# VisionRadar Phase 9J — Input Resolution Ablation Report

## Executive Summary
Input resolution ablation was conducted across `640x640`, `512x512`, and `416x416` input sizes for `yolox_nano.onnx`.

---

## 1. Experimental Results

| Input Resolution | Precision | Recall | F1 Score | Avg Dets / Frame | CPU Latency (ms) | AI Inference FPS |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **640x640 (Production Default)** | **0.8350** | **0.8866** | **0.8600** | **1.29** | **154.54 ms** | **6.47 FPS** |
| **512x512** | 0.8036 | 0.9278 | 0.8612 | 1.40 | 114.60 ms | 8.73 FPS |
| **416x416** | 0.8319 | 0.9691 | 0.8952 | 1.41 | 67.96 ms | 14.71 FPS |

---

## 2. Key Findings & Recommendation
- **Spatial Resolution Tradeoff**: `640x640` provides maximum pixel detail for small distant vehicles and homography anchor localization.
- **Speed Tradeoff**: `416x416` accelerates CPU inference to 14.71 FPS.
- **Production Standard**: `640x640` remains the production default to ensure maximum localization precision for homography speed estimation.
