import os
import sys
import json
import glob
import re

def audit_datasets():
    print("==================================================")
    print(" DATASET & MODEL TRAINING USAGE FORENSIC AUDIT")
    print("==================================================")
    
    datasets = [
        "vehicle-detection-project-main",
        "Vehicle_Tracking",
        "training",
        "valid-fine-tuning",
        "valid-self-learning",
        "ITD-Indian-traffic-dataset-main",
        "BrnoCompSpeed-master",
        "DETRAC-Test-Tra",
        "DETRAC-Train-Detections",
        "Accident"
    ]
    
    results = {}
    
    for ds in datasets:
        ds_matches = []
        # Walk python and md files to find references
        for root, dirs, files in os.walk("."):
            if any(x in root for x in [".git", "node_modules", "__pycache__", ".next"]):
                continue
            for f in files:
                if f.endswith((".py", ".md", ".json", ".txt")):
                    fp = os.path.join(root, f)
                    try:
                        with open(fp, "r", encoding="utf-8", errors="ignore") as file_obj:
                            content = file_obj.read()
                            if ds.lower() in content.lower() or ds.replace("-", "_").lower() in content.lower():
                                ds_matches.append(fp)
                    except Exception:
                        pass
        results[ds] = ds_matches
        print(f"\nDataset: {ds}")
        print(f"File matches count: {len(ds_matches)}")
        for m in ds_matches[:10]:
            print(f"  - {m}")
            
    # Check training scripts specifically
    print("\n--- TRAINING / FINE-TUNING SCRIPTS AUDIT ---")
    training_scripts = [
        "scripts/yolox_finetune.py",
        "scripts/yolox_torch_transfer.py",
        "scripts/run_phase5_2_experiment.py",
        "scripts/run_phase6_multi_dataset_eval.py",
        "scripts/run_phase6b_multi_dataset_eval.py",
        "scripts/run_phase12_research_evaluation.py"
    ]
    for ts in training_scripts:
        if os.path.exists(ts):
            print(f"\nScript: {ts}")
            with open(ts, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            for line in lines:
                if any(kw in line.lower() for kw in ["train", "dataset", "epoch", "model", "save", "onnx"]):
                    if len(line.strip()) < 120:
                        print(f"  {line.strip()}")

if __name__ == "__main__":
    audit_datasets()
