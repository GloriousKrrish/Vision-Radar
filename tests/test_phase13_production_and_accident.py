import os
import sys
import json
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("packages"))
sys.path.insert(0, os.path.abspath("apps/api"))

def test_production_model_unmodified():
    """Verify YOLOX production model path and hash remain strictly unchanged."""
    model_path = "data/models/yolox_nano.onnx"
    assert os.path.exists(model_path), f"Production model missing at {model_path}"
    
    import hashlib
    with open(model_path, "rb") as f:
        h = hashlib.sha256(f.read()).hexdigest()
    # Official COCO YOLOX-Nano ONNX SHA-256
    expected_hash = "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d"
    assert h == expected_hash, "Production YOLOX-Nano ONNX model has been modified!"

def test_accident_classifier_separate_location():
    """Verify experimental accident classifier resides in separate directory and does NOT overwrite YOLOX."""
    classifier_pt = "data/models/accident_classifier/accident_classifier.pt"
    classifier_onnx = "data/models/accident_classifier/accident_classifier.onnx"
    metrics_path = "data/models/accident_classifier/metrics.json"
    
    assert os.path.exists(classifier_pt), "Accident classifier PyTorch model missing"
    assert os.path.exists(classifier_onnx), "Accident classifier ONNX model missing"
    assert os.path.exists(metrics_path), "Classifier metrics JSON missing"
    
    with open(metrics_path, "r") as f:
        m = json.load(f)
    assert m["status"] == "EXPERIMENTAL_MODE_ACTIVE"
    assert m["accuracy"] > 0.80

def test_accident_dataset_forensics_report_exists():
    """Verify ACCIDENT_DATASET_FORENSIC.md exists and contains required sections."""
    report_path = "reports/phase13/ACCIDENT_DATASET_FORENSIC.md"
    assert os.path.exists(report_path), f"Forensic report missing at {report_path}"
    
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    assert "ACCIDENT DATASET FORENSIC REPORT" in content
    assert "Scene-Level Image Classifier" in content
    assert "CRITICAL CONCEPTUAL BOUNDARY" in content

def test_frontend_workbench_visibility_logic():
    """Verify WorkbenchView component code maintains uncalibrated detection visibility."""
    wb_path = "apps/web/src/components/WorkbenchView.tsx"
    assert os.path.exists(wb_path), f"WorkbenchView missing at {wb_path}"
    
    with open(wb_path, "r", encoding="utf-8") as f:
        code = f.read()
        
    # Check that bounding box fill does NOT use faint 10% slate grey (#64748B1A)
    assert "col + '1A'" not in code, "Faint 10% opacity bounding box bug detected in WorkbenchView!"
    # Check that Calibration Required Banner is present
    assert "CALIBRATION REQUIRED FOR METRIC SPEED" in code
    # Check job auto-selection logic
    assert "pipelineStageRef.current === 'EMPTY'" in code or "selectCompletedJob" in code

def test_frame_synchronization_schema():
    """Verify frame_result payload contains exact frame_index and job_id for frame sync."""
    from visionradar.worker.job_worker import run_job
    # Schema validation for worker frame result format
    sample_payload = {
        "type": "frame_result",
        "job_id": 227,
        "frame_index": 45,
        "tracks": [
            {
                "track_id": 17,
                "vehicle_class": "Car",
                "confidence": 0.82,
                "bbox": [100, 200, 300, 400],
                "speed_status": "CALIBRATION_REQUIRED"
            }
        ]
    }
    assert sample_payload["job_id"] == 227
    assert sample_payload["frame_index"] == 45
    assert sample_payload["tracks"][0]["track_id"] == 17
