"""Run the detector on every .mp4 in a folder and print alerts.

    python test_detector.py "path/to/videos"
"""
import glob, os, sys
import cv2
from detector import CheatingDetector, save_screenshot

video_folder = sys.argv[1]
detector = CheatingDetector()

for path in sorted(glob.glob(os.path.join(video_folder, "*.mp4"))):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    detector.reset(fps)
    print(f"\n=== {os.path.basename(path)} ===")
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        annotated, events = detector.process_frame(frame, i)
        for ev in events:
            shot = save_screenshot(ev.pop("frame"), student_id=ev["student_id"])
            print(f"  ALERT  t={ev['video_time']:>5}s  ID {ev['student_id']:<3} "
                  f"{ev['behavior']:<30} conf={ev['confidence']}  -> {shot}")
        i += 1
    cap.release()
