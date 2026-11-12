"""
Student tracking and cheating detection.

Each frame: YOLO11-pose + ByteTrack find and track students, YOLO11 finds
phones, and two rules decide who gets flagged:
  - phone: a phone is assigned to the student for PHONE_SECONDS
  - head turn: head yaw moves away from the student's own baseline
    for TURN_SECONDS
"""
import os
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

# --- settings ---
POSE_MODEL = "yolo11n-pose.pt"   # person boxes + keypoints
PHONE_MODEL = "yolo11n.pt"       # COCO model, class 67 = cell phone
PHONE_CLASS_ID = 67

PERSON_CONF = 0.40
PHONE_CONF = 0.40
PHONE_MAX_ASPECT = 3.0    # w/h limit, filters out notebooks
PHONE_SECONDS = 0.5

TURN_THRESHOLD = 0.12     # yaw change vs. baseline
TURN_SECONDS = 0.75
BASELINE_SECONDS = 5.0    # baseline = rolling median

RED_HOLD_SECONDS = 2.0
ALERT_COOLDOWN = 10.0     # per student and behaviour

GREEN = (0, 200, 0)       # BGR
RED = (0, 0, 255)


def head_yaw(keypoints, kp_conf):
    """Horizontal nose offset from the eye midpoint, normalised by eye distance.

    ~0 when facing the camera, +/- when turned. None if the face keypoints
    (nose, left eye, right eye) are not confident enough.
    """
    if min(kp_conf[0], kp_conf[1], kp_conf[2]) < 0.5:
        return None
    nose_x, left_eye_x, right_eye_x = keypoints[0][0], keypoints[1][0], keypoints[2][0]
    eye_distance = abs(left_eye_x - right_eye_x) + 1e-6
    eyes_middle = (left_eye_x + right_eye_x) / 2
    return (nose_x - eyes_middle) / eye_distance


class StudentState:
    """Per-track state kept between frames."""

    def __init__(self, fps):
        self.yaw_history = deque(maxlen=int(fps * BASELINE_SECONDS))
        self.phone_frames = 0
        self.phone_confs = []
        self.turn_frames = 0
        self.red_until = -1.0        # video time in seconds
        self.label = ""
        self.last_alert = {}


class CheatingDetector:
    def __init__(self, fps=25.0):
        self.pose_model = YOLO(POSE_MODEL)
        self.phone_model = YOLO(PHONE_MODEL)
        self.reset(fps)

    def reset(self, fps):
        """Clear tracks and state before a new video."""
        self.fps = fps
        self.students = {}
        self.pose_model = YOLO(POSE_MODEL)   # resets the tracker

    def find_phones(self, frame):
        """Return phone boxes as (x1, y1, x2, y2, conf)."""
        result = self.phone_model.predict(frame, classes=[PHONE_CLASS_ID],
                                          conf=PHONE_CONF, verbose=False)[0]
        phones = []
        for box, conf in zip(result.boxes.xyxy.cpu().numpy(),
                             result.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            if w / (h + 1e-6) > PHONE_MAX_ASPECT:
                continue
            phones.append((x1, y1, x2, y2, float(conf)))
        return phones

    @staticmethod
    def phone_owner(boxes, phone):
        """Return the row index of the student holding the phone, or None.

        Checks the raw boxes first, then boxes padded by 25% (hands often
        stick out). Ties go to the largest box, i.e. the student nearest
        the camera.
        """
        cx, cy = (phone[0] + phone[2]) / 2, (phone[1] + phone[3]) / 2
        for pad in (0.0, 0.25):
            matches = []
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
                if (x1 - pw) <= cx <= (x2 + pw) and (y1 - ph) <= cy <= (y2 + ph):
                    matches.append((i, (x2 - x1) * (y2 - y1)))
            if matches:
                return max(matches, key=lambda m: m[1])[0]
        return None

    def process_frame(self, frame, frame_index):
        """Run detection on one frame.

        Returns (annotated_frame, new_events).
        """
        video_time = frame_index / self.fps
        annotated = frame.copy()
        new_events = []

        # students + track IDs
        result = self.pose_model.track(frame, persist=True, tracker="bytetrack.yaml",
                                       conf=PERSON_CONF, verbose=False)[0]

        phones = self.find_phones(frame)

        if result.boxes.id is None:
            return annotated, new_events

        ids = result.boxes.id.int().cpu().tolist()
        boxes = result.boxes.xyxy.cpu().numpy()
        kps = result.keypoints.xy.cpu().numpy()
        kp_confs = result.keypoints.conf.cpu().numpy()

        # row index -> phones
        owner_of = {}
        for p in phones:
            row = self.phone_owner(boxes, p)
            if row is not None:
                owner_of.setdefault(row, []).append(p)

        for row, (track_id, box, kp, kp_conf) in enumerate(zip(ids, boxes, kps, kp_confs)):
            st = self.students.setdefault(track_id, StudentState(self.fps))

            # phone rule
            my_phones = owner_of.get(row, [])
            if my_phones:
                st.phone_frames += 1
                st.phone_confs.append(max(p[4] for p in my_phones))
            else:
                st.phone_frames, st.phone_confs = 0, []

            if st.phone_frames >= self.fps * PHONE_SECONDS:
                conf = float(np.mean(st.phone_confs))
                self._flag(st, track_id, "Phone detected", conf, video_time, new_events)

            # head-turn rule
            yaw = head_yaw(kp, kp_conf)
            if yaw is not None:
                if len(st.yaw_history) >= self.fps:  # wait for 1 s of history
                    baseline = float(np.median(st.yaw_history))
                    deviation = abs(yaw - baseline)
                    st.turn_frames = st.turn_frames + 1 if deviation > TURN_THRESHOLD else 0
                    if st.turn_frames >= self.fps * TURN_SECONDS:
                        # keypoint confidence
                        conf = float(np.mean(kp_conf[:3]))
                        self._flag(st, track_id, "Head turned",
                                   conf, video_time, new_events)
                st.yaw_history.append(yaw)

            # draw
            is_red = video_time <= st.red_until
            colour = RED if is_red else GREEN
            text = f"ID {track_id}" + (f" - {st.label}" if is_red else "")
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = max(y1, th + 8)
            cv2.rectangle(annotated, (x1, ty - th - 8), (x1 + tw + 8, ty), colour, -1)
            cv2.putText(annotated, text, (x1 + 4, ty - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # attach after drawing so screenshots show the red boxes
        for ev in new_events:
            ev["frame"] = annotated.copy()
        return annotated, new_events

    def _flag(self, st, track_id, behaviour, conf, video_time, new_events):
        """Mark the student red and emit an alert if the cooldown has passed."""
        st.red_until = video_time + RED_HOLD_SECONDS
        st.label = behaviour.split(" (")[0]
        last = st.last_alert.get(behaviour, -1e9)
        if video_time - last >= ALERT_COOLDOWN:
            st.last_alert[behaviour] = video_time
            new_events.append({
                "student_id": track_id,
                "behavior": behaviour,
                "confidence": round(conf, 2),
                "video_time": round(video_time, 1),
                "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })


def save_screenshot(frame, folder="alerts/screenshots", student_id=0):
    """Save frame as JPEG and return the path."""
    os.makedirs(folder, exist_ok=True)
    name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_id{student_id}.jpg"
    path = os.path.join(folder, name)
    cv2.imwrite(path, frame)
    return path
