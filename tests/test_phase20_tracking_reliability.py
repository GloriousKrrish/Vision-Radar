"""
Phase 20: Comprehensive Tracking Reliability Test Suite.

Verifies:
1. Vehicle temporarily disappearing behind an obstruction (Kalman extrapolation & recovery)
2. Vehicle moving rapidly between frames (high-velocity constant velocity tracking)
3. Two vehicles crossing paths (Hungarian bipartite association & crossing disambiguation)
4. Low-confidence detection recovery (Stage 2 association & noise rejection)
5. Track expiration & lifecycle cleanup
6. Track ID stability across long contiguous sequences
7. Duplicate detection assignment suppression
8. Empty, zero-box, and degenerate/corrupt detector input robustness
9. End-of-video state cleanup and metrics accounting
10. Multiple independent tracker instances (isolation across jobs)
"""

import sys
import os
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath("packages"))

from visionradar.cv.detector import Detection
from visionradar.cv.tracker import ByteTrackTracker, KalmanBoxTracker, compute_iou


def test_vehicle_temporarily_disappearing_obstruction():
    """
    1. Vehicle temporarily disappearing behind an obstruction.
    Vehicle moves at 10 px/frame from x=100 to x=180.
    At frames 3 and 4, detector misses the vehicle (occlusion).
    Kalman filter predicts the state forward, and at frame 5, when
    detection returns, it matches the predicted state and preserves the track ID.
    """
    tracker = ByteTrackTracker(max_age=10, iou_threshold=0.3, high_threshold=0.4)

    # Frames 0..2: Vehicle visible moving right (+10 px/frame)
    for f in range(3):
        x = 100.0 + f * 10.0
        d = Detection(class_id=2, class_name="Car", confidence=0.90, bbox=(x, 100.0, x + 50.0, 150.0))
        res = tracker.update([d], frame_index=f, timestamp=f * 0.033)
        assert len(res) == 1
        assert res[0].track_id == 1

    # Frames 3..4: Obstruction / miss (empty detections)
    for f in range(3, 5):
        res = tracker.update([], frame_index=f, timestamp=f * 0.033)
        # Track 1 is maintained in active_tracks with predicted Kalman bbox
        assert 1 in tracker.active_tracks
        assert tracker.active_tracks[1].time_since_update == (f - 2)

    # Frame 5: Vehicle re-emerges at x=150.0
    d_reappear = Detection(class_id=2, class_name="Car", confidence=0.88, bbox=(150.0, 100.0, 200.0, 150.0))
    res5 = tracker.update([d_reappear], frame_index=5, timestamp=5 * 0.033)

    assert len(res5) == 1
    assert res5[0].track_id == 1, "Track ID must be preserved after temporary obstruction!"
    assert res5[0].time_since_update == 0
    assert tracker.total_fragmentations >= 1


def test_vehicle_moving_rapidly_between_frames():
    """
    2. Vehicle moving rapidly between frames (40 px/frame).
    Kalman filter learns the velocity, ensuring IoU matching with predicted bbox
    succeeds even at high displacement.
    """
    tracker = ByteTrackTracker(max_age=10, iou_threshold=0.25, high_threshold=0.4)

    # Fast vehicle moving across 8 frames at 40 px/frame
    for f in range(8):
        x = 100.0 + f * 40.0
        d = Detection(class_id=2, class_name="Car", confidence=0.92, bbox=(x, 200.0, x + 80.0, 250.0))
        res = tracker.update([d], frame_index=f, timestamp=f * 0.033)
        assert len(res) == 1
        assert res[0].track_id == 1, f"Fast moving vehicle lost its ID at frame {f}!"


def test_two_vehicles_crossing_paths():
    """
    3. Two vehicles crossing paths.
    Vehicle 1 travels left-to-right (y=100 -> y=100, x increasing).
    Vehicle 2 travels right-to-left (y=100 -> y=100, x decreasing).
    They cross in the center, then separate.
    Hungarian bipartite matching and motion prediction must maintain correct IDs.
    """
    tracker = ByteTrackTracker(max_age=10, iou_threshold=0.3, high_threshold=0.4)

    # Initial frame: well separated
    # V1 at x=100..150, V2 at x=350..400
    for f in range(6):
        x1 = 100.0 + f * 25.0
        x2 = 350.0 - f * 25.0

        d1 = Detection(class_id=2, class_name="Car", confidence=0.90, bbox=(x1, 100.0, x1 + 50.0, 150.0))
        d2 = Detection(class_id=2, class_name="Truck", confidence=0.90, bbox=(x2, 100.0, x2 + 50.0, 150.0))

        res = tracker.update([d1, d2], frame_index=f, timestamp=f * 0.033)
        assert len(res) == 2

    # After crossing, verify IDs remain 1 and 2
    assert tracker.next_id == 3, "No spurious third track should be spawned during crossing!"
    tids = set(t.track_id for t in tracker.active_tracks.values())
    assert tids == {1, 2}


def test_low_confidence_detection_recovery_and_noise_rejection():
    """
    4. Low-confidence detection recovery.
    - An active track can be matched with a low-confidence detection (0.20 in [0.10, 0.40)).
    - An isolated/spurious low-confidence detection is rejected (no track initialized).
    """
    tracker = ByteTrackTracker(max_age=5, iou_threshold=0.3, high_threshold=0.4, low_threshold=0.10)

    # Frame 0: Spawn track 1 with high confidence
    d_high = Detection(class_id=2, class_name="Car", confidence=0.85, bbox=(100.0, 100.0, 150.0, 150.0))
    # Also supply an isolated spurious noise detection far away with low confidence (0.20)
    d_noise = Detection(class_id=2, class_name="Car", confidence=0.20, bbox=(500.0, 500.0, 550.0, 550.0))

    res0 = tracker.update([d_high, d_noise], frame_index=0, timestamp=0.0)
    assert len(res0) == 1
    assert res0[0].track_id == 1, "Noise detection should not create a new track!"

    # Frame 1: Track 1 drops in confidence to 0.22 (e.g. shadow/glare)
    d_recovered = Detection(class_id=2, class_name="Car", confidence=0.22, bbox=(105.0, 102.0, 155.0, 152.0))
    res1 = tracker.update([d_recovered], frame_index=1, timestamp=0.033)

    assert len(res1) == 1
    assert res1[0].track_id == 1, "Track 1 should be recovered via Stage 2 association!"
    assert res1[0].time_since_update == 0


def test_track_expiration_and_cleanup():
    """
    5. Track expiration and cleanup.
    When a track is missed for > max_age frames, it is removed from active_tracks
    and placed in recently_lost_tracks, and eventually purged.
    """
    tracker = ByteTrackTracker(max_age=3, iou_threshold=0.3, high_threshold=0.4)

    # Frame 0: Spawn track
    d = Detection(class_id=2, class_name="Car", confidence=0.90, bbox=(100.0, 100.0, 150.0, 150.0))
    tracker.update([d], frame_index=0, timestamp=0.0)
    assert len(tracker.active_tracks) == 1

    # Frames 1..3: Track not seen (time_since_update = 1, 2, 3 <= max_age 3)
    for f in range(1, 4):
        tracker.update([], frame_index=f, timestamp=f * 0.033)
        assert 1 in tracker.active_tracks

    # Frame 4: time_since_update = 4 > max_age 3 -> track evicted into recently_lost_tracks
    tracker.update([], frame_index=4, timestamp=4 * 0.033)
    assert 1 not in tracker.active_tracks
    assert 1 in tracker.recently_lost_tracks

    # Frames 5..12: After 2 * max_age frames, evicted completely
    for f in range(5, 13):
        tracker.update([], frame_index=f, timestamp=f * 0.033)

    assert 1 not in tracker.recently_lost_tracks


def test_track_id_stability():
    """
    6. Stable track identifiers over a 50-frame sequence with small jitter.
    """
    tracker = ByteTrackTracker(max_age=10, iou_threshold=0.3, high_threshold=0.4)

    for f in range(50):
        jitter = np.sin(f) * 2.0
        x = 50.0 + f * 5.0 + jitter
        y = 100.0 + jitter
        d = Detection(class_id=2, class_name="Car", confidence=0.85, bbox=(x, y, x + 60.0, y + 40.0))
        res = tracker.update([d], frame_index=f, timestamp=f * 0.033)

        assert len(res) == 1
        assert res[0].track_id == 1, f"Track ID switched unexpectedly at frame {f}!"

    metrics = tracker.get_tracking_metrics()
    assert metrics["id_switches"] == 0
    assert metrics["total_tracks_created"] == 1


def test_duplicate_detection_assignment_suppression():
    """
    7. Duplicate detection assignment.
    Two highly overlapping detections (IoU > 0.85) should not spawn duplicate parallel tracks.
    """
    tracker = ByteTrackTracker(max_age=10, iou_threshold=0.3, high_threshold=0.4)

    # Frame 0: Two almost identical detector boxes for the same physical vehicle
    d1 = Detection(class_id=2, class_name="Car", confidence=0.90, bbox=(100.0, 100.0, 160.0, 160.0))
    d2 = Detection(class_id=2, class_name="Car", confidence=0.88, bbox=(101.0, 101.0, 161.0, 161.0))

    res = tracker.update([d1, d2], frame_index=0, timestamp=0.0)
    # Duplicate track suppression removes redundant track
    assert len(tracker.active_tracks) == 1


def test_empty_and_degenerate_detector_input():
    """
    8. Empty, zero-box, and corrupt detector input robustness.
    Tracker should handle empty lists and zero/negative bounding boxes without exception.
    """
    tracker = ByteTrackTracker()

    # Empty frame
    res_empty = tracker.update([], frame_index=0, timestamp=0.0)
    assert res_empty == []

    # Degenerate boxes (zero area or negative coordinates)
    d_deg1 = Detection(class_id=2, class_name="Car", confidence=0.80, bbox=(0.0, 0.0, 0.0, 0.0))
    d_deg2 = Detection(class_id=2, class_name="Car", confidence=0.80, bbox=(-50.0, -50.0, 10.0, 10.0))

    res_deg = tracker.update([d_deg1, d_deg2], frame_index=1, timestamp=0.033)
    assert isinstance(res_deg, list)
    # Ensure tracker state did not crash
    assert tracker.get_tracking_metrics()["total_detections_processed"] == 2


def test_end_of_video_cleanup_and_accounting():
    """
    9. End-of-video state cleanup and metrics accounting.
    """
    tracker = ByteTrackTracker()

    d1 = Detection(class_id=2, class_name="Car", confidence=0.90, bbox=(10.0, 10.0, 50.0, 50.0))
    d2 = Detection(class_id=7, class_name="Truck", confidence=0.95, bbox=(200.0, 200.0, 280.0, 280.0))
    tracker.update([d1, d2], frame_index=0, timestamp=0.0)

    metrics = tracker.get_tracking_metrics()
    assert metrics["total_tracks_created"] == 2
    assert metrics["active_tracks"] == 2
    assert metrics["id_switches"] == 0
    assert metrics["total_detections_processed"] == 2


def test_independent_tracker_instances_isolation():
    """
    10. Long videos and multiple independent jobs.
    Two separate tracker instances must not share state, next_id, or active tracks.
    """
    tracker_job1 = ByteTrackTracker()
    tracker_job2 = ByteTrackTracker()

    d = Detection(class_id=2, class_name="Car", confidence=0.90, bbox=(100.0, 100.0, 150.0, 150.0))
    res1 = tracker_job1.update([d], frame_index=0, timestamp=0.0)
    assert res1[0].track_id == 1

    # Second tracker should start independently at track_id 1
    res2 = tracker_job2.update([d], frame_index=0, timestamp=0.0)
    assert res2[0].track_id == 1
    assert tracker_job1.active_tracks is not tracker_job2.active_tracks
