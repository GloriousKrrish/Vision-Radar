import os
import sys
sys.path.insert(0, "packages")
import cv2
import numpy as np
import json
from visionradar.examination.detector import ExaminationDetector, COCO_EXAM_CLASSES, DEFAULT_CLASS_THRESHOLDS
from visionradar.examination.tracker import PersonTracker, ExaminationObjectTracker, associate_persons_and_objects
from visionradar.examination.events import ExaminationEventEngine

def audit_video(video_path: str, video_label: str):
    print("=" * 60)
    print(f"AUDITING: {video_label} ({video_path})")
    print("=" * 60)
    
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        return
        
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Dimensions: {w}x{h}, FPS: {fps:.2f}, Frames: {total_frames}, Duration: {total_frames/fps:.2f}s")
    
    detector = ExaminationDetector(confidence_threshold=0.25)
    tracker = PersonTracker()
    obj_tracker = ExaminationObjectTracker()
    
    sample_indices = [0, int(total_frames * 0.1), int(total_frames * 0.25), int(total_frames * 0.5), int(total_frames * 0.75), total_frames - 1]
    sample_indices = [f for f in sample_indices if 0 <= f < total_frames]
    
    print("\n--- SAMPLE FRAME DETECTIONS (Default Conf = 0.25 / Person = 0.30) ---")
    for s_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, s_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        persons, objects = detector.detect(frame)
        print(f"Frame {s_idx} ({s_idx/fps:.2f}s): {len(persons)} persons, {len(objects)} objects")
        for i, p in enumerate(persons):
            bx = [round(c, 1) for c in p["bbox"]]
            area = (bx[2] - bx[0]) * (bx[3] - bx[1])
            print(f"   Person #{i+1}: conf={p['confidence']:.3f}, bbox={bx}, area={area:.0f}px (w={bx[2]-bx[0]:.1f}, h={bx[3]-bx[1]:.1f})")
        for j, o in enumerate(objects):
            bx = [round(c, 1) for c in o.bbox]
            print(f"   Object #{j+1} [{o.object_class}]: conf={o.confidence:.3f}, bbox={bx}")
            
    # Test sensitivity to lower thresholds
    print("\n--- THRESHOLD ABLATION (Frame at 25% duration) ---")
    cap.set(cv2.CAP_PROP_POS_FRAMES, sample_indices[2])
    ret, frame = cap.read()
    if ret:
        for test_thresh in [0.40, 0.30, 0.25, 0.20, 0.15, 0.10]:
            p_test, o_test = detector.detect(frame, conf_thresh=test_thresh)
            print(f"Threshold {test_thresh:.2f}: {len(p_test)} persons, {len(o_test)} objects")

    cap.release()

if __name__ == "__main__":
    audit_video("data/videos/c1b4f50fd4df4fc7843b6389a66cec32.mp4", "exam cheat.mp4")
    audit_video("data/videos/adcfc2a1e1e84047a8639805000a49d0.mp4", "Exam 2.mp4")
