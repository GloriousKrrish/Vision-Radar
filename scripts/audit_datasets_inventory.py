import os
import glob
import json
import hashlib

DATASETS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "DATASETS")

def get_dir_size_mb(path):
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return round(total / (1024 * 1024), 2)

def audit_all_datasets():
    inventory = {}
    items = sorted(os.listdir(DATASETS_ROOT))

    for item in items:
        path = os.path.join(DATASETS_ROOT, item)
        if os.path.isdir(path):
            imgs = glob.glob(os.path.join(path, "**", "*.jpg"), recursive=True) + \
                   glob.glob(os.path.join(path, "**", "*.png"), recursive=True) + \
                   glob.glob(os.path.join(path, "**", "*.jpeg"), recursive=True)
            txts = glob.glob(os.path.join(path, "**", "*.txt"), recursive=True)
            xmls = glob.glob(os.path.join(path, "**", "*.xml"), recursive=True)
            jsons = glob.glob(os.path.join(path, "**", "*.json"), recursive=True)
            vids = glob.glob(os.path.join(path, "**", "*.mp4"), recursive=True) + \
                   glob.glob(os.path.join(path, "**", "*.avi"), recursive=True)

            size_mb = get_dir_size_mb(path)

            inventory[item] = {
                "path": path,
                "size_mb": size_mb,
                "images": len(imgs),
                "videos": len(vids),
                "txt_labels": len(txts),
                "xml_labels": len(xmls),
                "json_labels": len(jsons),
            }
        else:
            size_mb = round(os.path.getsize(path) / (1024 * 1024), 2)
            inventory[item] = {
                "path": path,
                "size_mb": size_mb,
                "type": "file / download fragment"
            }

    print(json.dumps(inventory, indent=2))
    return inventory

if __name__ == "__main__":
    audit_all_datasets()
