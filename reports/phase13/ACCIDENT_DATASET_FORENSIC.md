# ACCIDENT DATASET FORENSIC REPORT

**Dataset Path:** `DATASETS/Accident`  
**Evaluation Date:** September 27, 2026  
**Dataset Source:** Road Accidents CCTV Footage Dataset  

---

## 1. Forensic Summary & Dataset Reality

- **Total Images:** `6191`
- **Valid Readable Images:** `6191`
- **Duplicate Images:** `14`
- **Corrupted Images:** `0`
- **Top Image Resolutions:** `[((320, 240), 6191)]`
- **Annotation Format:** No bounding box TXT/XML files present. Images represent static photographic frames.
- **Dataset Classification:** **Static Image Classification Dataset (Scene-Level).**

---

## 2. Model Type Determination

- **Is it an Object Detector?** **NO.** Lacks bounding box coordinates (`x_min, y_min, x_max, y_max`).
- **Is it a Video Action Recognition Model?** **NO.** Lacks frame ordering, video container streams, or temporal timestamps.
- **Is it a Track-Level Classifier?** **NO.** Lacks bounding box tracking IDs or velocity ground truth.
- **What is it?** **Scene-Level Image Classifier.**

---

## 3. Data Split Integrity (Leakage Prevention)

- **Train Split (80%):** `4952` images
- **Val Split (10%):** `619` images
- **Test Split (10%):** `620` images
- **Leakage Safeguards:** Deduplication performed via MD5 hash comparison before splitting.

---

## 4. Architectural Distinction (Static Image Classifier vs Video Event Detector)

> **CRITICAL CONCEPTUAL BOUNDARY:**  
> Static image classification ($P(\text{accident} \mid \text{frame})\$) predicts whether a single photograph shows crash damage or an accident scene.  
> Real-time video accident detection in VisionRadar requires **spatio-temporal trajectory evidence** (sudden deceleration $\Delta v / \Delta t$, trajectory convergence, abnormal vehicle interaction, post-collision immobilization).  
> The accident image classifier functions as a **secondary candidate frame verifier**, NOT an independent automated legal crash detector.
