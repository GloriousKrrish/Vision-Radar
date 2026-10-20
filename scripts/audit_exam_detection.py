import sys
sys.path.insert(0, "packages")
import cv2
import onnxruntime as ort
import numpy as np

model_path = "data/models/yolox_nano.onnx"
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
inputs = session.get_inputs()
outputs = session.get_outputs()

print("Model Input:", inputs[0].name, inputs[0].shape, inputs[0].type)
print("Model Output:", outputs[0].name, outputs[0].shape, outputs[0].type)
# If shape is [1, 3549, 85], it's 80 COCO classes.

# Now let's decode frames of 'data/videos/c1b4f50fd4df4fc7843b6389a66cec32.mp4' (exam cheat.mp4)
video_path = "data/videos/c1b4f50fd4df4fc7843b6389a66cec32.mp4"
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"\nVideo info: {total_frames} frames, {fps} fps, {width}x{height}")

# Let's inspect detections across sample frames using ExaminationDetector
from visionradar.examination.detector import ExaminationDetector
detector = ExaminationDetector(model_path=model_path, confidence_threshold=0.15) # lower threshold to see what's detected

frame_detections = []
for f_idx in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ret, frame = cap.read()
    if not ret:
        continue
    persons, objects = detector.detect(frame, conf_thresh=0.10)
    print(f"Frame {f_idx}: {len(persons)} persons, {len(objects)} objects")
    for p in persons:
        print(f"   Person: conf={p['confidence']:.3f}, bbox={p['bbox']}")
    for obj in objects:
        print(f"   Object: class={obj.object_class}, conf={obj.confidence:.3f}, bbox={obj.bbox}")

cap.release()
