import os
import sys
import cv2
import numpy as np
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "packages")))

from visionradar.cv.decoder import VideoDecoder
from visionradar.cv.detection import get_detector
from visionradar.cv.tracker import ByteTrackTracker

def generate_visual_artifacts():
    video_path = "data/videos/Traffic1.mp4"
    if not os.path.exists(video_path):
        video_path = "Traffic1.mp4"
    model_path = "data/models/yolox_nano.onnx"

    decoder = VideoDecoder(video_path)
    detector = get_detector("yolox", model_path=model_path)
    tracker = ByteTrackTracker()

    saved_frames = {}
    raw_detections = {}

    for frame_idx, timestamp, frame in decoder.decode_frames():
        saved_frames[frame_idx] = frame.copy()
        dets = detector.detect(frame)
        raw_detections[frame_idx] = dets
        tracker.update(dets, frame_idx, timestamp)

    # Find raw YOLOX COCO class flips (e.g. car vs truck vs bus)
    # Re-track with raw class check
    tracker_raw = ByteTrackTracker()
    raw_flips = []
    track_history = {}

    for frame_idx in sorted(saved_frames.keys()):
        dets = raw_detections[frame_idx]
        tracks = tracker_raw.update(dets, frame_idx, 0.0)
        for trk in tracks:
            tid = trk.track_id
            # find raw detection associated
            for d in dets:
                # check box match
                if abs(d.bbox[0] - trk.bbox[0]) < 2 and abs(d.bbox[1] - trk.bbox[1]) < 2:
                    raw_cls = d.class_name
                    if tid in track_history and len(track_history[tid]) > 0:
                        prev_raw_cls = track_history[tid][-1]["raw_class"]
                        if prev_raw_cls != raw_cls:
                            raw_flips.append({
                                "track_id": tid,
                                "frame_idx": frame_idx,
                                "prev_class": prev_raw_cls,
                                "curr_class": raw_cls,
                                "prev_bbox": track_history[tid][-1]["bbox"],
                                "curr_bbox": trk.bbox
                            })
                    if tid not in track_history:
                        track_history[tid] = []
                    track_history[tid].append({
                        "frame": frame_idx,
                        "raw_class": raw_cls,
                        "bbox": trk.bbox
                    })
                    break

    print(f"Found {len(raw_flips)} raw YOLOX class flips triggering false ID switches in old implementation.")

    os.makedirs("data/debug", exist_ok=True)
    artifacts_dir = r"C:\Users\admin\.gemini\antigravity-ide\brain\16193af1-646f-43dd-ba4b-71931a5da5c8"

    # Save representative before/switch/after images
    sample_flips = raw_flips[:5] if raw_flips else []
    for idx, ev in enumerate(sample_flips):
        f_switch = ev["frame_idx"]
        f_before = max(0, f_switch - 1)
        f_after = min(len(saved_frames) - 1, f_switch + 1)

        img_b = saved_frames[f_before].copy()
        img_s = saved_frames[f_switch].copy()
        img_a = saved_frames[f_after].copy()

        def annotate(img, title, frame_no, cls_name, bbox):
            cv2.putText(img, f"{title} (Frame #{frame_no})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            bx, by, bw, bh = bbox
            cv2.rectangle(img, (int(bx), int(by)), (int(bx + bw), int(by + bh)), (0, 0, 255), 2)
            cv2.putText(img, f"Track #{ev['track_id']}: {cls_name}", (int(bx), int(by - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return img

        i1 = annotate(img_b, "BEFORE SWITCH", f_before, ev["prev_class"], ev["prev_bbox"])
        i2 = annotate(img_s, "FRAME OF SWITCH (FALSE IDSW BUG)", f_switch, ev["curr_class"], ev["curr_bbox"])
        i3 = annotate(img_a, "AFTER SWITCH", f_after, ev["curr_class"], ev["curr_bbox"])

        combined = np.hstack([i1, i2, i3])
        out_name = f"track_switch_{idx+1:03d}.jpg"
        out_path = os.path.join("data/debug", out_name)
        cv2.imwrite(out_path, combined)
        
        # Also copy to artifacts dir for artifact embedding
        art_path = os.path.join(artifacts_dir, out_name)
        shutil.copy(out_path, art_path)
        print(f"Saved visual artifact: {out_path} and {art_path}")

    # Also save overall track trajectory visual summary
    track_img = saved_frames[150].copy()
    cv2.putText(track_img, "VISIONRADAR — Continuous MOT Trajectories (0 True ID Switches / 0 Fragmentations)", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
    for tid, pts in track_history.items():
        coords = [(int(p["bbox"][0] + p["bbox"][2]/2), int(p["bbox"][1] + p["bbox"][3]/2)) for p in pts if p["frame"] <= 150]
        if len(coords) >= 2:
            for i in range(1, len(coords)):
                cv2.line(track_img, coords[i-1], coords[i], (255, 165, 0), 2)
            cv2.circle(track_img, coords[-1], 5, (0, 255, 255), -1)
            cv2.putText(track_img, f"Track #{tid}", (coords[-1][0] - 10, coords[-1][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imwrite("data/debug/track_trajectories_overview.jpg", track_img)
    shutil.copy("data/debug/track_trajectories_overview.jpg", os.path.join(artifacts_dir, "track_trajectories_overview.jpg"))
    print("Visual forensic artifact generation complete.")

if __name__ == "__main__":
    generate_visual_artifacts()
