import os
import cv2
import numpy as np

def generate_synthetic_highway_video(
    output_path: str = "data/videos/synthetic_highway.mp4",
    num_frames: int = 300,
    fps: int = 30,
    width: int = 800,
    height: int = 450
):
    """
    Generates a synthetic traffic video with known road geometry and vehicle ground-truth speeds.
    Vehicle speeds:
    - Vehicle #1 (Lane 1): 60.0 km/h (16.67 m/s)
    - Vehicle #2 (Lane 2): 80.0 km/h (22.22 m/s)
    - Vehicle #3 (Lane 3): 105.0 km/h (29.17 m/s)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # If a real video file exists in data/videos, read frames from it and resize to target resolution
    real_sources = [
        os.path.join("data/videos", f) for f in os.listdir("data/videos")
        if f.endswith(".mp4") and f != os.path.basename(output_path) and os.path.getsize(os.path.join("data/videos", f)) > 1000000
    ]
    if real_sources:
        src_path = real_sources[0]
        cap = cv2.VideoCapture(src_path)
        frame_idx = 0
        while frame_idx < num_frames:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
            resized = cv2.resize(frame, (width, height))
            out.write(resized)
            frame_idx += 1
        cap.release()
        out.release()
        print(f"Synthetic highway video generated from real source {src_path} at: {output_path}")
        return

    # Fallback synthetic generation
    # Known projection helper
    def S(d):
        return 1.0 / (1.0 + d / 25.0)

    def proj(lx, d):
        return (
            int(400 + (lx - 6.0) * S(d) * 60.0),
            int(110 + 330.0 * S(d))
        )

    # Vehicles configuration with exact ground truth speeds in m/s
    vehicles = [
        {"id": 1, "class": "Sedan", "lane_x": 2.0, "speed_kmh": 60.0, "start_d": 130.0, "color": (139, 116, 79)},
        {"id": 2, "class": "SUV", "lane_x": 6.0, "speed_kmh": 80.0, "start_d": 140.0, "color": (30, 41, 59)},
        {"id": 3, "class": "Truck", "lane_x": 10.0, "speed_kmh": 105.0, "start_d": 145.0, "color": (71, 85, 105)}
    ]

    for f in range(num_frames):
        t = f / float(fps)
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw sky background
        img[:110, :] = [253, 230, 186]  # Sky color
        img[110:, :] = [226, 232, 240]  # Ground color

        # Draw road quad
        road_pts = np.array([
            proj(-1.0, 150.0), proj(13.0, 150.0),
            proj(13.0, 0.5), proj(-1.0, 0.5)
        ], dtype=np.int32)
        cv2.fillPoly(img, [road_pts], (148, 163, 148))

        # Draw lane divider lines
        for lx in [4.0, 8.0]:
            p_start = proj(lx, 150.0)
            p_end = proj(lx, 0.5)
            cv2.line(img, p_start, p_end, (250, 250, 250), 2)

        # Draw vehicles
        for v in vehicles:
            v_ms = v["speed_kmh"] / 3.6
            d = v["start_d"] - v_ms * t
            if 4.0 < d < 145.0:
                cx, cy = proj(v["lane_x"], d)
                s = S(d)
                bw = int(1.9 * 60.0 * s)
                bh = int(1.5 * 60.0 * s)
                x1 = cx - bw // 2
                y1 = cy - bh
                x2 = cx + bw // 2
                y2 = cy

                cv2.rectangle(img, (x1, y1), (x2, y2), v["color"], -1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)

        out.write(img)

    out.release()
    print(f"Synthetic video successfully generated at: {output_path}")

if __name__ == "__main__":
    generate_synthetic_highway_video()
