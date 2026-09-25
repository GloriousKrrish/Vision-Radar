import numpy as np
import pytest
from visionradar.cv.detection import Detection, BaseDetector, LightweightMotionDetector, YOLOXDetector, get_detector
from visionradar.cv.tracker import ByteTrackTracker

def test_detector_registry_and_yolox_interface():
    # Test factory retrieval
    motion_det = get_detector("mog2")
    assert isinstance(motion_det, LightweightMotionDetector)

    yolox_det = get_detector("yolox")
    assert isinstance(yolox_det, YOLOXDetector)

    info = yolox_det.get_model_info()
    assert "YOLOX" in info["name"]
    assert "framework" in info or "backend" in info

def test_yolox_detector_execution():
    yolox = YOLOXDetector(confidence_threshold=0.3)
    dummy_frame = np.zeros((450, 800, 3), dtype=np.uint8)

    # Draw dummy vehicle shape on road
    dummy_frame[200:260, 300:400] = [100, 100, 100]

    dets = yolox.detect(dummy_frame)
    assert isinstance(dets, list)
    for d in dets:
        assert isinstance(d, Detection)
        assert d.confidence >= 0.3
        assert len(d.bbox) == 4

def test_bytetrack_tracking_metrics():
    tracker = ByteTrackTracker()
    det1 = Detection(class_id=0, class_name="Car", confidence=0.9, bbox=(100.0, 200.0, 150.0, 250.0))
    det2 = Detection(class_id=0, class_name="Car", confidence=0.92, bbox=(110.0, 205.0, 160.0, 255.0))

    t1 = tracker.update([det1], frame_index=1, timestamp=0.033)
    assert len(t1) == 1
    assert t1[0].track_id == 1
    assert t1[0].direction == "UNKNOWN"

    t2 = tracker.update([det2], frame_index=2, timestamp=0.066)
    assert len(t2) == 1
    assert t2[0].track_id == 1
    assert t2[0].detection_count == 2
    assert t2[0].duration > 0.0

    metrics = tracker.get_tracking_metrics()
    assert metrics["total_tracks_created"] == 1
    assert metrics["total_detections_processed"] == 2
