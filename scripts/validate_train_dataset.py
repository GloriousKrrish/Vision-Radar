import os
import glob
from PIL import Image

TRAIN_DIR = os.path.join(
    "DATASETS", "vehicle-detection-project-main", "vehicle-detection-project-main", "train"
)
IMAGE_DIR = os.path.join(TRAIN_DIR, "images")
LABEL_DIR = os.path.join(TRAIN_DIR, "labels")

VALID_CLASSES = {0: "bus", 1: "car", 2: "motorcycle", 3: "truck"}

def validate_dataset():
    image_files = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.*")))
    label_files = sorted(glob.glob(os.path.join(LABEL_DIR, "*.txt")))
    
    print(f"Found {len(image_files)} image files and {len(label_files)} label files in training set.")
    
    img_basenames = {os.path.splitext(os.path.basename(f))[0]: f for f in image_files}
    lbl_basenames = {os.path.splitext(os.path.basename(f))[0]: f for f in label_files}
    
    valid_images = 0
    invalid_images = 0
    valid_labels_count = 0
    invalid_labels_count = 0
    
    per_class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    issues = []
    
    for base, img_path in img_basenames.items():
        image_valid = True
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception as e:
            image_valid = False
            issues.append(f"Corrupt image {img_path}: {e}")
            
        if base not in lbl_basenames:
            image_valid = False
            issues.append(f"Missing label file for image {base}")
            invalid_images += 1
            continue
            
        lbl_path = lbl_basenames[base]
        with open(lbl_path, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            
        seen_boxes = set()
        for idx, line in enumerate(lines):
            parts = line.split()
            if len(parts) != 5:
                invalid_labels_count += 1
                image_valid = False
                issues.append(f"Malformed label in {lbl_path} line {idx+1}: {line}")
                continue
                
            try:
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
            except ValueError:
                invalid_labels_count += 1
                image_valid = False
                issues.append(f"Non-numeric values in {lbl_path} line {idx+1}: {line}")
                continue
                
            if cls_id not in VALID_CLASSES:
                invalid_labels_count += 1
                image_valid = False
                issues.append(f"Invalid class ID {cls_id} in {lbl_path} line {idx+1}")
                continue
                
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                invalid_labels_count += 1
                image_valid = False
                issues.append(f"Out of bounds coordinates in {lbl_path} line {idx+1}: {cx, cy, w, h}")
                continue
                
            if w <= 0.0 or h <= 0.0:
                invalid_labels_count += 1
                image_valid = False
                issues.append(f"Zero or negative area box in {lbl_path} line {idx+1}: w={w}, h={h}")
                continue
                
            box_tuple = (cls_id, round(cx, 6), round(cy, 6), round(w, 6), round(h, 6))
            if box_tuple in seen_boxes:
                issues.append(f"Duplicate annotation in {lbl_path} line {idx+1}: {box_tuple}")
            seen_boxes.add(box_tuple)
            
            valid_labels_count += 1
            per_class_counts[cls_id] += 1
            
        if image_valid:
            valid_images += 1
        else:
            invalid_images += 1
            
    print("\n--- DATASET AUDIT RESULT ---")
    print(f"Total Train Images Checked: {len(img_basenames)}")
    print(f"Valid Images: {valid_images}")
    print(f"Invalid Images: {invalid_images}")
    print(f"Valid Bounding Box Labels: {valid_labels_count}")
    print(f"Invalid Bounding Box Labels: {invalid_labels_count}")
    print("Per-Class Object Counts:")
    for cid, name in VALID_CLASSES.items():
        print(f"  Class {cid} ({name}): {per_class_counts[cid]}")
    if issues:
        print(f"\nFound {len(issues)} warnings/issues:")
        for iss in issues[:10]:
            print(f"  - {iss}")
    else:
        print("\nZERO ANNOTATION ERRORS FOUND. DATASET IS 100% CLEAN.")

if __name__ == "__main__":
    validate_dataset()
