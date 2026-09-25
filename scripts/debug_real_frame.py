import os
import sys
import argparse
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "packages")))

from visionradar.cv.decoder import VideoDecoder
from visionradar.cv.detection import get_detector
from visionradar.cv.detection.yolox import YOLOXDetector

def debug_frame(video_path: str, target_frame: int):
    print("==================================================")
    print("VISIONRADAR — PHASE 3.2.3 REAL FRAME FORENSIC DEBUG")
    print("==================================================")
    print(f"Target Video: {video_path}")
    print(f"Target Frame Index: {target_frame}")

    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found: {video_path}")
        sys.exit(1)

    decoder = VideoDecoder(video_path)
    meta = decoder.get_metadata()
    total_frames = meta.total_frames
    fps = meta.fps
    width = meta.width
    height = meta.height
    print(f"Video Properties: {width}x{height} @ {fps:.2f} FPS | Total Frames: {total_frames}")

    target_frame = min(target_frame, total_frames - 1)
    timestamp = target_frame / fps
    print(f"Decoded Frame {target_frame} | Timestamp: {timestamp:.2f}s")

    # Extract target frame
    target_img = None
    for f_idx, t_stamp, img in decoder.decode_frames():
        if f_idx == target_frame:
            target_img = img
            timestamp = t_stamp
            break

    if target_img is None:
        print(f"ERROR: Failed to decode frame {target_frame}")
        sys.exit(1)

    os.makedirs("data/debug/runtime", exist_ok=True)
    raw_save_path = "data/debug/runtime/frame_716_raw.jpg"
    cv2.imwrite(raw_save_path, target_img)
    print(f"Saved raw frame image: {raw_save_path}")

    # Load YOLOX Detector
    model_path = "data/models/yolox_nano.onnx"
    detector = YOLOXDetector(model_path=model_path, confidence_threshold=0.25, nms_threshold=0.45)
    print(f"YOLOX Model Loaded: {model_path}")

    # Process frame
    raw_outputs, decoded_preds, post_nms, vehicle_dets = detector.detect_verbose(target_img)

    print(f"1. Raw Output Matrix Shape: {raw_outputs.shape}")
    print(f"2. Decoded Predictions Count: {len(decoded_preds)}")
    print(f"3. Post-NMS Detections Count: {len(post_nms)}")
    print(f"4. Vehicle Class Detections Count: {len(vehicle_dets)}")

    annotated_img = target_img.copy()

    print("\n--- VEHICLE DETECTIONS FOR FRAME 716 ---")
    for idx, det in enumerate(vehicle_dets):
        x1, y1, x2, y2 = det.bbox
        conf = det.confidence
        cls_name = det.class_name
        cls_id = det.class_id
        print(f"[{idx + 1}] Class: {cls_name} (ID: {cls_id}) | Confidence: {conf:.4f} | BBox (xyxy): [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

        # Draw bounding box
        cv2.rectangle(annotated_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(annotated_img, label, (int(x1), max(20, int(y1) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    det_save_path = "data/debug/runtime/frame_716_detections.jpg"
    cv2.imwrite(det_save_path, annotated_img)
    print(f"Saved annotated detections image: {det_save_path}")

    print("\nFORENSIC VERIFICATION:")
    if len(vehicle_dets) > 0:
        print(f"SUCCESS: YOLOX ONNX detector successfully detected {len(vehicle_dets)} vehicles on frame {target_frame}!")
    else:
        print(f"FAILURE: YOLOX ONNX detector returned 0 vehicle detections on frame {target_frame}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug single real frame YOLOX detection")
    parser.add_argument("--video", type=str, default="data/videos/Traffic1.mp4", help="Path to video file")
    parser.add_argument("--frame", type=int, default=716, help="Target frame index")
    args = parser.parse_args()
    debug_frame(args.video, args.frame)
