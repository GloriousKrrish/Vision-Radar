# VISIONRADAR PHASE 5.6 — BASELINE AUDIT REPORT

**System:** VisionRadar Monocular Video Intelligence Platform  
**Component:** Examination Event Engine (`packages/visionradar/examination/events.py`)  
**Phase:** 5.6 (Perspective-Aware Proximity Events and False-Positive Reduction)  
**Date:** 2026-10-22  
**Status:** AUDITED & EMPIRICALLY CONFIRMED  

---

## 1. Executive Summary

This audit evaluates the baseline behavior of the Examination Event Engine with specific focus on the `PEER_PROXIMITY` candidate event generator. 

The user reported **223 alert events** on Job #971 (a 54.2-second classroom examination clip, `Exam 2.mp4`). Empirical inspection of the SQLite database records for Job #971 validates that **223 ExaminationEventRecord rows exist**:
- **188 `PEER_PROXIMITY` records (84.3% of all events)**
- **35 `GAZE_DEVIATION` records (15.7% of all events)**

Detailed trajectory, bounding box, and temporal trace analysis confirms that **virtually all 188 proximity records represent false alarms** caused by monocular depth foreshortening, uncalibrated 2D pixel distance thresholds, the absence of baseline distance tracking, and zero hysteresis against detection jitter.

---

## 2. Answers to Specific Architectural Audit Questions

### 1. Coordinate Definitions: Bounding-box centers, feet, or torso?
* **Finding:** The proximity rule uses **2D bounding-box geometric centroids (centers)**.
* **Code Trace (`packages/visionradar/examination/events.py` lines 15–18):**
  ```python
  def bbox_distance(b1: List[float], b2: List[float]) -> float:
      """Computes Euclidean distance between two bounding box centers."""
      c1 = ((b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0)
      c2 = ((b2[0] + b2[2]) / 2.0, (b2[1] + b2[3]) / 2.0)
      return float(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))
  ```
* **Implication:** In elevated oblique camera views, vertical centroid differences reflect depth along the optical axis, not horizontal physical closeness. Centroids of examinees seated in successive rows align vertically on screen, yielding deceptively small Euclidean distances.

### 2. Thresholds: Are they fixed in pixels?
* **Finding:** **Yes, purely in 2D image pixels.**
* **Code Trace (Line 444):**
  ```python
  threshold_dist = max(110.0, (p1_w + p2_w) * 0.75)
  ```
* **Implication:** The threshold has an absolute pixel floor of $110.0\text{ px}$, scaled by bounding box width. There is zero camera calibration, zero perspective normalization, and zero conversion to real-world metric distances.

### 3. Stationary Tracks: Do stationary tracks generate repeated events?
* **Finding:** **YES, massively.**
* **Mechanism:** When two examinees sit stationary at adjacent desks or in adjacent rows, their 2D pixel distance hovers near the threshold (e.g. Pair 3 & 4 has mean distance $126.2\text{ px}$ vs threshold $128.3\text{ px}$). Normal detector bounding-box jitter (1–2 px per frame) causes the measured distance to oscillate above and below the threshold.
* Every time distance exceeds threshold for even a single frame:
  1. `_peer_proximity_streaks[pair_key]` is immediately reset to 0.
  2. The active event is terminated and archived into `_completed_events`.
  3. When distance dips back below threshold for 15 frames ($\approx 0.5\text{ s}$), a **new event is created with a new UUID** (`evt_prox_...`).
* **Evidence:** In Job #971, Pair #3 and #4 generated **10 distinct events**, Pair #3 and #13 generated **7 distinct events**, and Pair #1 and #18 generated **5 distinct events**.

### 4. Deduplication: Are events deduplicated across frames?
* **Finding:** Only within the temporary `_active_events` dictionary during continuous uninterrupted streaks.
* **Code Trace (Lines 503–506):**
  ```python
  unique_evts: Dict[str, ExaminationEvent] = {}
  for e in all_evts:
      unique_evts[e.event_id] = e
  ```
* **Implication:** Deduplication is indexed purely by `event_id` (a unique UUID per occurrence). There is **zero pairwise track deduplication**. Once an event closes, any subsequent re-triggering between the exact same two persons receives a new `event_id` and is permanently saved as an independent database record.

### 5. Event ID Persistence: Do event IDs persist across updates?
* **Finding:** An `event_id` persists while an event remains in `_active_events`. However, as soon as the streak breaks for a single frame, that `event_id` is finalized, and subsequent re-triggers generate brand-new IDs.

### 6. Alert Count Meaning: Distinct events or repeated observations?
* **Finding:** The 223 alert count in Job #971 represents **223 distinct database rows (`ExaminationEventRecord`)**, not multiple observations of a single event. 188 of these records are `PEER_PROXIMITY` rows.

### 7. Missing or Unstable Tracks: How do they affect the rule?
* **Finding:** If an examinee bends their head over the exam paper and the detector momentarily drops the track or reassigns the track ID (ID switch), the pairwise streak resets to 0. This immediately closes the active event and spawns a new event when the track is recovered, compounding the event explosion.

---

## 3. Empirical Baseline Data (Job #971)

* **Video File:** `data/videos/732ce0814da24c459364e9f45f75c9ac.mp4` (`Exam 2.mp4`)
* **Resolution:** 848x480 @ 30.0 FPS
* **Total Frames:** 1627 frames (54.23 seconds)
* **Person Tracks:** 22 total detected tracks (16 active seated students, 1 invigilator)
* **Total Persisted Events:** 223
  - `PEER_PROXIMITY`: 188 (84.3%)
  - `GAZE_DEVIATION`: 35 (15.7%)
* **Top Repeated Proximity Pairs:**
  - Track #3 & Track #4: 10 events (distance oscillates around 126.2 px vs 128.3 px threshold)
  - Track #3 & Track #13: 7 events
  - Track #1 & Track #18: 5 events
  - Track #4 & Track #9: 5 events
  - Track #7 & Track #12: 4 events
  - Track #12 & Track #18: 4 events
  - Track #5 & Track #11: 4 events
  - Track #8 & Track #14 (Invigilator): 4 events
* **Physical Reality via Video Review:** All examinees remain seated quietly writing exams. No examinee leans across desks, touches a peer, or exchanges materials. **True Positive rate for physical peer misconduct in this clip is 0. False Positive rate is virtually 100%.**

---

## 4. Root Causes Identified

1. **Uncalibrated 2D Centroid Metric:** Monocular perspective compression makes students in row $N$ and row $N+1$ appear adjacent in 2D Euclidean pixel space.
2. **Absence of Baseline Distance ($D_0$):** The system evaluates absolute instantaneous distance rather than *relative convergence from normal seated positions*.
3. **Static Equilibrium Triggering:** Normal stationary sitting maintains constant 2D distance. Because the streak threshold is only 15 frames (0.5s), stationary examinees continuously satisfy the trigger condition.
4. **Zero Gap Tolerance / Hysteresis:** A single missed frame or 1-pixel jitter resets the streak and closes the event, fragmenting what should be one observation into dozens of duplicate database records.
5. **Lack of Velocity / Motion Gating:** Genuine proximity convergence involves physical movement. Stationary examinees (mean velocity $\approx 0.8\text{ px/frame}$, consistent with detector noise) should never trigger convergence alerts without displacement.
6. **Invigilator Role Bleed:** Unconfigured roles default to `UNKNOWN`. When the invigilator stood in the aisle, the system paired them with nearby examinees, firing alerts on legitimate invigilation.
