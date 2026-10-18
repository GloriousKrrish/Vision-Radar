import os
import re

def search_repo():
    patterns = [
        r"synthetic_highway",
        r"default_calib",
        r"calibration_id.*1",
        r"realCalib.*1",
        r"Traffic1\.mp4",
        r"fallback.*demo",
        r"demo.*fallback"
    ]
    
    files_to_check = []
    for root, dirs, files in os.walk("."):
        if any(ignored in root for ignored in [".git", "node_modules", "dist", ".pytest_cache", "__pycache__"]):
            continue
        for f in files:
            if f.endswith((".py", ".ts", ".tsx", ".js", ".json")):
                files_to_check.append(os.path.join(root, f))
                
    for pat in patterns:
        regex = re.compile(pat, re.IGNORECASE)
        print(f"\n=== MATCHES FOR PATTERN: {pat} ===")
        match_count = 0
        for p in files_to_check:
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    for line_no, line in enumerate(f, 1):
                        if regex.search(line):
                            print(f"{p}:{line_no} | {line.strip()[:100]}")
                            match_count += 1
                            if match_count >= 15:
                                print("... (truncated)")
                                break
            except Exception:
                pass
            if match_count >= 15:
                break

if __name__ == '__main__':
    search_repo()
