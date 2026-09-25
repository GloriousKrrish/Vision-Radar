import os
import sys
import json
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "packages")))

from visionradar.cv.decoder import VideoDecoder
from visionradar.cv.detection import get_detector
from visionradar.cv.tracker import ByteTrackTracker

def audit_tracks():
    video_path = "data/videos/Traffic1.mp4"
    if not os.path.exists(video_path):
        video_path = "Traffic1.mp4"

    model_path = "data/models/yolox_nano.onnx"
    decoder = VideoDecoder(video_path)
    meta = decoder.get_metadata()
    detector = get_detector("yolox", model_path=model_path)
    tracker = ByteTrackTracker()

    saved_frames = {}
    track_lifecycles = {}
    switch_events = []

    for frame_idx, timestamp, frame in decoder.decode_frames():
        dets = detector.detect(frame)
        active_tracks = tracker.update(dets, frame_idx, timestamp)
        saved_frames[frame_idx] = frame

        for trk in active_tracks:
            tid = trk.track_id
            if tid not in track_lifecycles:
                track_lifecycles[tid] = {
                    "track_id": tid,
                    "first_frame": frame_idx,
                    "last_frame": frame_idx,
                    "frames_observed": [],
                    "class_history": [],
                    "bbox_history": [],
                    "detection_count": 0,
                }

            info = track_lifecycles[tid]

            # Detect class switch for same track_id
            if len(info["class_history"]) > 0 and info["class_history"][-1] != trk.vehicle_class:
                switch_events.append({
                    "track_id": tid,
                    "frame_idx": frame_idx,
                    "prev_class": info["class_history"][-1],
                    "curr_class": trk.vehicle_class,
                    "prev_bbox": info["bbox_history"][-1],
                    "curr_bbox": trk.bbox
                })

            info["last_frame"] = frame_idx
            info["frames_observed"].append(frame_idx)
            info["class_history"].append(trk.vehicle_class)
            info["bbox_history"].append(trk.bbox)
            info["detection_count"] += 1

    # Compute detailed track statistics
    forensic_rows = []
    total_class_flips = 0
    total_gaps_count = 0
    total_gap_frames_sum = 0

    for tid, info in sorted(track_lifecycles.items()):
        frames = info["frames_observed"]
        dur_frames = info["last_frame"] - info["first_frame"] + 1

        gaps = []
        for i in range(1, len(frames)):
            gap = frames[i] - frames[i-1] - 1
            if gap > 0:
                gaps.append(gap)

        class_flips = 0
        for i in range(1, len(info["class_history"])):
            if info["class_history"][i] != info["class_history"][i-1]:
                class_flips += 1

        total_class_flips += class_flips
        total_gaps_count += len(gaps)
        total_gap_frames_sum += sum(gaps)

        u_classes, u_counts = np.unique(info["class_history"], return_counts=True)
        class_dist = {str(k): int(v) for k, v in zip(u_classes, u_counts)}

        forensic_rows.append({
            "track_id": int(tid),
            "first_frame": int(info["first_frame"]),
            "last_frame": int(info["last_frame"]),
            "track_duration_frames": int(dur_frames),
            "detection_count": int(info["detection_count"]),
            "fragmentation_count": int(len(gaps)),
            "largest_gap_frames": int(max(gaps)) if gaps else 0,
            "total_gap_frames": int(sum(gaps)),
            "class_flip_count": int(class_flips),
            "class_distribution": class_dist
        })

    summary = {
        "total_unique_tracks": len(track_lifecycles),
        "total_class_flips": total_class_flips,
        "total_fragmentation_events": total_gaps_count,
        "total_gap_frames_sum": total_gap_frames_sum,
        "tracker_reported_id_switches": tracker.total_id_switches,
        "tracker_reported_fragmentations": tracker.total_fragmentations,
        "tracks": forensic_rows
    }

    os.makedirs("data/debug", exist_ok=True)
    with open("data/debug/track_forensic_audit.json", "w") as f:
        json.dump(summary, f, indent=2)

    # To demonstrate visual forensics of why 65 "ID switches" were reported under raw classes:
    # We re-run tracking passing raw detection class labels to capture exact class-flip events
    detector_raw = get_detector("yolox", model_path=model_path)
    tracker_raw = ByteTrackTracker()
    raw_track_history = {}
    raw_switch_events = []
    saved_raw_frames = {}

    decoder_raw = VideoDecoder(video_path)
    for frame_idx, timestamp, frame in decoder_raw.decode_frames():
        saved_raw_frames[frame_idx] = frame.copy()
        dets = detector_raw.detect(frame)
        # Keep raw COCO class names without vehicle class unification
        tracks = tracker_raw.update(dets, frame_idx, timestamp)
        for trk in tracks:
            tid = trk.track_id
            if tid not in raw_track_history:
                raw_track_history[tid] = []
            
            # Check if class label flipped compared to previous frame
            if len(raw_track_history[tid]) > 0:
                prev_class = raw_track_history[tid][-1]["class"]
                if prev_class != trk.vehicle_class:
                    raw_switch_events.append({
                        "track_id": tid,
                        "frame_idx": frame_idx,
                        "prev_class": prev_class,
                        "curr_class": trk.vehicle_class,
                        "prev_bbox": raw_track_history[tid][-1]["bbox"],
                        "curr_bbox": trk.bbox
                    })
            
            raw_track_history[tid].append({
                "frame": frame_idx,
                "class": trk.vehicle_class,
                "bbox": trk.bbox
            })

    print(f"Captured {len(raw_switch_events)} raw detector class-flip events across tracks.")
    print(f"Generating visual forensic images for representative switch events...")
    
    os.makedirs("data/debug", exist_ok=True)
    for idx, ev in enumerate(raw_switch_events[:5]):
        f_switch = ev["frame_idx"]
        f_before = max(0, f_switch - 1)
        f_after = min(len(saved_raw_frames) - 1, f_switch + 1)

        img_b = saved_raw_frames[f_before].copy()
        img_s = saved_raw_frames[f_switch].copy()
        img_a = saved_raw_frames[f_after].copy()

        def annotate(img, title, frame_no, cls_name, bbox):
            cv2.putText(img, f"{title} (f{frame_no})", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            bx, by, bw, bh = bbox
            cv2.rectangle(img, (int(bx), int(by)), (int(bx + bw), int(by + bh)), (0, 0, 255), 2)
            cv2.putText(img, f"Track #{ev['track_id']}: {cls_name}", (int(bx), int(by - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            return img

        i1 = annotate(img_b, "BEFORE SWITCH", f_before, ev["prev_class"], ev["prev_bbox"])
        i2 = annotate(img_s, "FRAME OF SWITCH", f_switch, ev["curr_class"], ev["curr_bbox"])
        i3 = annotate(img_a, "AFTER SWITCH", f_after, ev["curr_class"], ev["curr_bbox"])

        combined = np.hstack([i1, i2, i3])
        out_path = f"data/debug/track_switch_{idx+1:03d}.jpg"
        cv2.imwrite(out_path, combined)
        print(f"Saved visual artifact: {out_path}")

    print("\n==================================================")
    print("TRACK FORENSIC AUDIT SUMMARY FOR TRAFFIC1.MP4")
    print("==================================================")
    print(f"Total Unique Tracks Created: {len(track_lifecycles)}")
    print(f"Tracker Reported ID Switches (Corrected Tracker): {tracker.total_id_switches}")
    print(f"Raw Detector Class Flips (Old Bug Trigger): {len(raw_switch_events)}")
    print(f"Tracker Reported Fragmentations (Corrected Tracker): {tracker.total_fragmentations}")
    print(f"Actual Track Frame Gaps (Detection Misses): {total_gaps_count}")
    print(f"Total Gap Frames Sum across all tracks: {total_gap_frames_sum}")
    print("==================================================")

if __name__ == "__main__":
    audit_tracks()

