# VisionRadar — Scene Router Implementation & Verification Report (Stage 2)

**Date**: 2026-10-19  
**Status**: VERIFIED & BENCHMARKED  
**Component**: `packages/visionradar/scene_router/`  

---

## 1. Executive Summary

The Scene Router (Layer B) provides fully automatic, offline, lightweight video scene classification to dynamically recommend the optimal processing module (`traffic` vs `examination`) before processing begins.

- **Supported Classifications**: `TRAFFIC`, `CLASSROOM_EXAMINATION`, `OTHER_UNKNOWN`, `AMBIGUOUS`.
- **Inference Requirements**: 100% local CPU execution. Zero paid APIs. Zero cloud or internet dependencies.
- **Efficiency**: Configurable temporal sampling extracts 6 evenly distributed frames across the video timeline (10% to 90% duration) without full video decoding.
- **Cache Acceleration**: Content-hashed SHA-256 caching reduces re-inspection latency to **11.2ms**.
- **Automated Test Results**: **7 passed, 0 failed** in `tests/test_scene_router.py`.

---

## 2. Multi-Cue Visual Feature Architecture

The classifier combines complementary visual cues to distinguish outdoor roadway environments from indoor classroom/examination settings:

| Visual Cue | Target Domain | Extraction Method | Physical Significance |
| :--- | :--- | :--- | :--- |
| **Asphalt Dominance** | Traffic | HSV mask (low saturation < 65, mid-value 35-195) in lower 65% of frame | Paved road and highway ground surface |
| **Perspective Lines** | Traffic | Canny edge detection + HoughLinesP angle filtering (20° to 75°) | Converging lane lines, highway curbs, and dividers |
| **Sky / Horizon Gradient**| Traffic | Luminance contrast between upper 35% and lower 65% | Open sky illumination vs ground plane |
| **Desk / Paper Geometry**| Examination | Otsu binarization + contour polygonal approximation (aspect 1.1–3.8) | Repetitive rectangular desks and paper exam answer sheets |
| **Indoor Uniformity** | Examination | 3x3 spatial grid standard deviation of grayscale luminance | Controlled fluorescent interior overhead room lighting |
| **Edge Density Guard** | Integrity | Non-zero edge pixel ratio | Rejects flat, blank, underwater, or textureless false positives |

---

## 3. Empirical Verification Results

### A. Real-World Traffic Video Test (`Traffic1.mp4`)
```json
{
  "scene_type": "TRAFFIC",
  "confidence": 0.76,
  "recommended_module": "traffic",
  "requires_user_confirmation": false,
  "reason": "Roadway environment detected with linear perspective lanes and asphalt surface characteristics (confidence: 0.76)",
  "model_version": "scene-router-v1.0",
  "sampled_frames_count": 6,
  "cue_scores": {
    "roadway_composite": 0.7591,
    "classroom_composite": 0.2926,
    "roadway_consistency": 1.0,
    "classroom_consistency": 0.0,
    "asphalt_dominance": 0.7883,
    "linear_perspective_lines": 1.0,
    "indoor_illumination_uniformity": 0.6434,
    "desk_paper_rectangles": 0.0714
  },
  "video_path": "data/videos/Traffic1.mp4",
  "sha256_hash": "6be88f03f103155cb1233c4b3929c506b1339e3de32157c06d765c1fd7ea601b"
}
```
- **Outcome**: 100% of sampled frames voted for `TRAFFIC`. Roadway composite score (0.7591) significantly exceeded classroom composite score (0.2926). Correctly recommended `traffic` without requiring manual user intervention.

### B. Synthetic Classroom Examination Test
- **Input**: Uniform illumination room with brown desks and bright white paper sheets.
- **Output**: `scene_type: CLASSROOM_EXAMINATION`, `confidence: 0.74`, `recommended_module: examination`, `requires_user_confirmation: false`.
- **Outcome**: High desk/paper rectangle count correctly steered routing to the examination pipeline.

### C. Textureless / Blank Frame Test
- **Input**: Uniform solid color background.
- **Output**: `scene_type: OTHER_UNKNOWN`, `confidence: 0.85`, `recommended_module: none`, `requires_user_confirmation: true`.
- **Outcome**: Edge density guard prevented false-positive classroom classification.

### D. Ambiguous Scene Test
- **Input**: Conflicting roadway asphalt texture combined with desk-like contours.
- **Output**: `scene_type: AMBIGUOUS`, `requires_user_confirmation: true`.
- **Outcome**: System correctly flags ambiguity and defers module selection to human operator.

---

## 4. Latency & Performance

| Scenario | Measured Latency | Sampling Details |
| :--- | :--- | :--- |
| **Cold Start Classification** | 7,667 ms | Full video file opening, 6 temporal frame seeks, feature extraction |
| **Warm Cached Lookup** | 11.2 ms | SHA-256 header hash verification + dictionary lookup |
| **Single Frame Feature Extraction** | ~18 ms | Grayscale, HSV, Hough transform, contour analysis at 480x270 |

---

## 5. Test Suite Verification

```text
tests/test_scene_router.py::test_scene_router_schemas PASSED             [ 14%]
tests/test_scene_router.py::test_classify_real_traffic_video PASSED      [ 28%]
tests/test_scene_router.py::test_classify_synthetic_classroom_frames PASSED [ 42%]
tests/test_scene_router.py::test_classify_unknown_blank_frames PASSED    [ 57%]
tests/test_scene_router.py::test_classify_ambiguous_frames PASSED        [ 71%]
tests/test_scene_router.py::test_classify_missing_or_corrupt_file PASSED [ 85%]
tests/test_scene_router.py::test_classifier_caching_performance PASSED   [100%]
============================== 7 passed in 13.47s ==============================
```
