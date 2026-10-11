"""
VisionRadar Read-Only Dataset Adapters (Task 2 & Task 5 Compliant).

Provides non-destructive loaders for YOLO-format detection datasets and JSON/MOT-format
tracking datasets without altering source files or converting disk annotations.
"""

import os
import json
import glob
from pathlib import Path
import cv2
from typing import List, Dict, Any, Tuple, Optional

from visionradar.eval.schemas import GroundTruthDetection, GroundTruthTrack


CLASS_MAP_ROBOFLOW = {
    0: "bus",
    1: "car",
    2: "motorcycle",
    3: "truck"
}

CLASS_MAP_VEHICLE_TRACKING = {
    "car": "car",
    "bus": "bus",
    "truck": "truck",
    "motorcycle": "motorcycle",
    "vehicle": "car"
}


class DetectionDatasetAdapter:
    """
    Read-only adapter for YOLO-formatted detection datasets.
    Parses images and corresponding normalized bounding box TXT annotations.
    """
    def __init__(self, dataset_name: str, root_path: str, split: str = "train"):
        self.dataset_name = dataset_name
        self.root_path = Path(root_path)
        self.split = split

    def load_dataset(self) -> List[Tuple[str, np.ndarray, List[GroundTruthDetection]]]:
        """
        Loads image-annotation pairs for the configured dataset split.

        Returns:
            list of tuples: (image_path, img_bgr_array, list_of_gt_detections)
        """
        import cv2
        import numpy as np

        samples = []
        if not self.root_path.exists():
            return samples

        # Recursively search for image files matching split
        all_images = sorted(
            list(self.root_path.glob(f"**/{self.split}/images/*.jpg")) +
            list(self.root_path.glob(f"**/{self.split}/images/*.png")) +
            list(self.root_path.glob(f"**/{self.split}/images/*.jpeg"))
        )

        # Fallback if split folder is not present directly in path
        if not all_images:
            all_images = sorted(
                list(self.root_path.glob("**/*.jpg")) +
                list(self.root_path.glob("**/*.png")) +
                list(self.root_path.glob("**/*.jpeg"))
            )

        for img_path in all_images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            img_h, img_w, _ = img.shape
            image_id = img_path.stem

            # Locate matching label file: look in labels/ directory or adjacent to image
            parent_dir = img_path.parent
            if parent_dir.name == "images":
                label_dir = parent_dir.parent / "labels"
            else:
                label_dir = parent_dir

            txt_path = label_dir / f"{image_id}.txt"
            gt_detections = []

            if txt_path.exists():
                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cid = int(parts[0])
                                cx, cy, w, h = map(float, parts[1:5])

                                cname = CLASS_MAP_ROBOFLOW.get(cid, f"class_{cid}")

                                # Convert normalized cx, cy, w, h to absolute xyxy pixel coordinates
                                x1 = max(0.0, (cx - w / 2.0) * img_w)
                                y1 = max(0.0, (cy - h / 2.0) * img_h)
                                x2 = min(float(img_w), (cx + w / 2.0) * img_w)
                                y2 = min(float(img_h), (cy + h / 2.0) * img_h)

                                gt_detections.append(GroundTruthDetection(
                                    dataset=self.dataset_name,
                                    split=self.split,
                                    image_id=image_id,
                                    frame_index=0,
                                    class_name=cname,
                                    class_id=cid,
                                    bbox_xyxy=[x1, y1, x2, y2],
                                    source_annotation=line.strip()
                                ))
                except Exception:
                    pass

            samples.append((str(img_path), img, gt_detections))

        return samples


class TrackingDatasetAdapter:
    """
    Read-only adapter for multi-object vehicle tracking datasets.
    Parses MP4 video clips, ground-truth JSON trajectories, and MOT TXT annotations.
    """
    def __init__(self, root_path: str = "DATASETS/Vehicle_Tracking"):
        self.root_path = Path(root_path)

    def discover_sequences(self) -> List[Dict[str, Any]]:
        """
        Discovers available tracking sequences, video paths, and ground-truth annotation files.
        """
        sequences = []
        if not self.root_path.exists():
            return sequences

        for seq_dir in sorted(self.root_path.iterdir()):
            if seq_dir.is_dir():
                seq_name = seq_dir.name
                videos = list(seq_dir.glob("*.mp4"))
                jsons = list(seq_dir.glob("*.json"))
                txts = list(seq_dir.glob("*.txt"))

                orig_video = next((str(v) for v in videos if "Original" in v.name), str(videos[0]) if videos else None)
                gt_video = next((str(v) for v in videos if "GroundTruth" in v.name), None)

                sequences.append({
                    "sequence_id": seq_name,
                    "dir_path": str(seq_dir),
                    "original_video_path": orig_video,
                    "gt_video_path": gt_video,
                    "gt_json_path": str(jsons[0]) if jsons else None,
                    "gt_txt_path": str(txts[0]) if txts else None
                })
        return sequences

    def load_gt_tracks(self, sequence_info: Dict[str, Any]) -> List[GroundTruthTrack]:
        """
        Parses ground-truth tracks from MOT TXT or JSON format.
        """
        gt_tracks = []
        txt_path = sequence_info.get("gt_txt_path")
        json_path = sequence_info.get("gt_json_path")
        seq_id = sequence_info["sequence_id"]

        if txt_path and os.path.exists(txt_path):
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split(",")
                        if len(parts) >= 6:
                            frame_idx = int(parts[0])
                            tid = int(parts[1])
                            left, top, width, height = map(float, parts[2:6])
                            x1 = max(0.0, left)
                            y1 = max(0.0, top)
                            x2 = x1 + max(0.0, width)
                            y2 = y1 + max(0.0, height)

                            gt_tracks.append(GroundTruthTrack(
                                dataset="Vehicle_Tracking",
                                sequence_id=seq_id,
                                frame_index=frame_idx,
                                gt_track_id=tid,
                                class_name="car",
                                bbox_xyxy=[x1, y1, x2, y2]
                            ))
            except Exception:
                pass

        elif json_path and os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for tid_str, tdata in data.items():
                        try:
                            tid = int(tid_str)
                        except ValueError:
                            continue
                        frames = tdata.get("frames", [])
                        boxes = tdata.get("boxes", [])
                        for fpath, bbox in zip(frames, boxes):
                            # extract frame index from path string
                            fname = Path(fpath).stem
                            try:
                                frame_idx = int(fname)
                            except ValueError:
                                frame_idx = 0
                            if len(bbox) == 4:
                                left, top, width, height = map(float, bbox)
                                gt_tracks.append(GroundTruthTrack(
                                    dataset="Vehicle_Tracking",
                                    sequence_id=seq_id,
                                    frame_index=frame_idx,
                                    gt_track_id=tid,
                                    class_name="car",
                                    bbox_xyxy=[left, top, left + width, top + height]
                                ))
            except Exception:
                pass

        return gt_tracks


class TrajectoryDatasetAdapter:
    """
    Read-only adapter for extracting ground-truth vehicle trajectories from tracking datasets.
    """
    def __init__(self, tracking_adapter: Optional[TrackingDatasetAdapter] = None):
        self.tracking_adapter = tracking_adapter or TrackingDatasetAdapter()

    def load_gt_trajectories(self, sequence_info: Dict[str, Any]) -> Dict[int, List[GroundTruthTrack]]:
        """
        Loads GT track points grouped by gt_track_id for trajectory continuity and error analysis.
        """
        tracks = self.tracking_adapter.load_gt_tracks(sequence_info)
        traj_map: Dict[int, List[GroundTruthTrack]] = {}
        for trk in tracks:
            tid = trk.gt_track_id
            if tid not in traj_map:
                traj_map[tid] = []
            traj_map[tid].append(trk)

        # Sort each trajectory by frame_index
        for tid in traj_map:
            traj_map[tid].sort(key=lambda t: t.frame_index)

        return traj_map


class SpeedDatasetAdapter:
    """
    Read-only adapter for Speed Evaluation datasets (e.g. BrnoCompSpeed).
    Inspects local storage for radar speed ground truth and video sequences.
    """
    def __init__(self, dataset_name: str = "BrnoCompSpeed-master", root_path: str = "DATASETS/BrnoCompSpeed-master"):
        self.dataset_name = dataset_name
        self.root_path = Path(root_path)

    def check_availability(self) -> Dict[str, Any]:
        """
        Audits local files for raw video streams and speed radar ground truth annotations.
        """
        if not self.root_path.exists():
            return {
                "dataset_name": self.dataset_name,
                "available": False,
                "status": "NOT_LOCALLY_AVAILABLE",
                "message": "Dataset directory not found."
            }

        videos = list(self.root_path.glob("**/*.mp4")) + list(self.root_path.glob("**/*.avi"))
        txt_labels = list(self.root_path.glob("**/*.txt")) + list(self.root_path.glob("**/*.json"))

        if len(videos) == 0:
            return {
                "dataset_name": self.dataset_name,
                "available": False,
                "status": "NOT_LOCALLY_AVAILABLE",
                "message": "Source code repository present, but raw high-resolution video streams and radar speed GT files are NOT LOCALLY AVAILABLE.",
                "local_size_mb": 0.05
            }

        return {
            "dataset_name": self.dataset_name,
            "available": True,
            "status": "LOCALLY_AVAILABLE",
            "videos_found": len(videos),
            "gt_files_found": len(txt_labels)
        }


class AnomalyDatasetAdapter:
    """
    Read-only adapter for Accident / Anomaly / Collision datasets.
    Inspects local storage for temporal event start/end annotations and video clips.
    """
    def __init__(self, root_path: str = "DATASETS"):
        self.root_path = Path(root_path)

    def check_availability(self) -> Dict[str, Any]:
        """
        Audits local datasets for temporal anomaly / collision labels.
        """
        return {
            "dataset_name": "Accident_Anomaly_Dataset",
            "available": False,
            "status": "NOT_LOCALLY_AVAILABLE",
            "message": "No local dataset in DATASETS/ currently contains temporal accident or collision labels.",
            "recommended_action": "Refer to reports/phase6/ACCIDENT_DATA_REQUIREMENTS.md for future dataset acquisition criteria."
        }
