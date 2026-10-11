"""
VisionRadar Phase 5.0 Dataset Audit & Forensics Inspector Script.
Performs read-only discovery, annotation analysis, video/image checks, class extraction,
and redundancy analysis on DATASETS/ folder without modifying any dataset files.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
import cv2

DATASETS_DIR = Path("DATASETS")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
ANNOT_EXTS = {".txt", ".json", ".xml", ".csv", ".yaml", ".yml", ".mat"}


def inspect_entry(path: Path) -> dict:
    info = {
        "name": path.name,
        "rel_path": str(path),
        "abs_path": str(path.resolve()),
        "total_size_bytes": 0,
        "total_files": 0,
        "images": 0,
        "videos": 0,
        "annotations": 0,
        "extensions": {},
        "image_resolutions": set(),
        "video_details": [],
        "annotation_samples": {},
        "readme_files": [],
        "license_files": [],
        "sample_filepaths": [],
        "is_incomplete": False
    }

    if path.is_file():
        info["total_size_bytes"] = path.stat().st_size
        info["total_files"] = 1
        ext = path.suffix.lower()
        info["extensions"][ext] = 1
        info["image_resolutions"] = sorted(list(info["image_resolutions"]))
        if "crdownload" in path.name.lower() or "unconfirmed" in path.name.lower():
            info["is_incomplete"] = True
        return info

    for p in path.glob("**/*"):
        if p.is_file():
            size = p.stat().st_size
            info["total_size_bytes"] += size
            info["total_files"] += 1
            ext = p.suffix.lower()
            info["extensions"][ext] = info["extensions"].get(ext, 0) + 1

            if len(info["sample_filepaths"]) < 20:
                info["sample_filepaths"].append(str(p.relative_to(path)))

            name_lower = p.name.lower()
            if "readme" in name_lower:
                info["readme_files"].append(str(p.relative_to(path)))
            if "license" in name_lower:
                info["license_files"].append(str(p.relative_to(path)))
            if "crdownload" in name_lower or "unconfirmed" in name_lower:
                info["is_incomplete"] = True

            if ext in IMAGE_EXTS:
                info["images"] += 1
                if len(info["image_resolutions"]) < 10:
                    try:
                        img = cv2.imread(str(p))
                        if img is not None:
                            h, w, _ = img.shape
                            info["image_resolutions"].add(f"{w}x{h}")
                    except Exception:
                        pass

            elif ext in VIDEO_EXTS:
                info["videos"] += 1
                try:
                    cap = cv2.VideoCapture(str(p))
                    if cap.isOpened():
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = float(cap.get(cv2.CAP_PROP_FPS))
                        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        dur = frames / fps if fps > 0 else 0
                        info["video_details"].append({
                            "rel_file": str(p.relative_to(path)),
                            "resolution": f"{w}x{h}",
                            "fps": fps,
                            "frames": frames,
                            "duration_sec": round(dur, 2)
                        })
                    cap.release()
                except Exception:
                    pass

            elif ext in ANNOT_EXTS:
                info["annotations"] += 1
                if ext not in info["annotation_samples"]:
                    try:
                        with open(p, "r", encoding="utf-8", errors="ignore") as f:
                            snippet = "".join([f.readline() for _ in range(8)])[:500]
                            info["annotation_samples"][ext] = {
                                "file": str(p.relative_to(path)),
                                "snippet": snippet
                            }
                    except Exception as e:
                        info["annotation_samples"][ext] = {"error": str(e)}

    info["image_resolutions"] = sorted(list(info["image_resolutions"]))
    return info


def run_audit():
    if not DATASETS_DIR.exists():
        print("Error: DATASETS directory not found!")
        return

    audit_data = {}
    for entry in sorted(DATASETS_DIR.iterdir()):
        print(f"Inspecting: {entry.name}...")
        audit_data[entry.name] = inspect_entry(entry)

    out_file = Path("data/dataset_audit_temp.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)

    print(f"\nAudit completed. Temp summary saved to: {out_file}")


if __name__ == "__main__":
    run_audit()
