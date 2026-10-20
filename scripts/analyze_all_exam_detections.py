import sys
sys.path.insert(0, "packages")
import cv2
from visionradar.examination.detector import ExaminationDetector

video_path = "data/videos/c1b4f50fd4df4fc7843b6389a66cec32.mp4"
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

detector = ExaminationDetector(model_path="data/models/yolox_nano.onnx", confidence_threshold=0.15)

phone_frames = []
book_frames = []
person_frames = []

for f_idx in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        break
    persons, objects = detector.detect(frame, conf_thresh=0.15)
    if persons:
        person_frames.append((f_idx, persons[0]["confidence"], persons[0]["bbox"]))
    for obj in objects:
        if obj.object_class == "cell phone":
            phone_frames.append((f_idx, obj.confidence, obj.bbox))
        elif obj.object_class == "book":
            book_frames.append((f_idx, obj.confidence, obj.bbox))

cap.release()

print(f"Total video frames: {total_frames}")
print(f"Frames with Person: {len(person_frames)} / {total_frames}")
print(f"Frames with Cell Phone (conf>=0.15): {len(phone_frames)}")
for pf in phone_frames[:15]:
    print(f"   Frame {pf[0]}: conf={pf[1]:.3f}, bbox={pf[2]}")
if len(phone_frames) > 15:
    print(f"   ... and {len(phone_frames) - 15} more phone detections")
    for pf in phone_frames[-5:]:
        print(f"   Frame {pf[0]}: conf={pf[1]:.3f}, bbox={pf[2]}")

print(f"Frames with Book/Paper (conf>=0.15): {len(book_frames)}")
