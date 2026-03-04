# modules/eye_head.py — Component 1: Eye Gaze + Head Pose monitoring

import time
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

from config import (
    WINDOW_SIZE_MINUTES, WINDOW_THRESHOLD_PERCENT, OVERALL_THRESHOLD_PERCENT,
    HEAD_YAW_RATIO_THRESHOLD, HEAD_PITCH_RATIO_THRESHOLD,
    FONT_CV, DANGER, SUCCESS, WARNING, TEXT
)


# ─────────────────────────────────────────────────────────────────────────────
# VIOLATION TRACKER
# ─────────────────────────────────────────────────────────────────────────────
class ViolationTracker:
    def __init__(self, window_size_seconds=300):
        self.window_size             = window_size_seconds
        self.violation_timestamps    = deque()
        self.start_time              = time.time()
        self.total_violation_time    = 0.0
        self.last_update_time        = time.time()
        self.window_alert_triggered  = False
        self.overall_alert_triggered = False

    def update(self, is_violating):
        current_time = time.time()
        delta        = current_time - self.last_update_time
        if is_violating:
            self.total_violation_time += delta
        self.violation_timestamps.append((current_time, is_violating))
        cutoff = current_time - self.window_size
        while self.violation_timestamps and self.violation_timestamps[0][0] < cutoff:
            self.violation_timestamps.popleft()
        self.last_update_time = current_time

    def window_percentage(self):
        if not self.violation_timestamps:
            return 0.0
        current_time  = time.time()
        oldest        = self.violation_timestamps[0][0]
        duration      = current_time - oldest
        if duration <= 0:
            return 0.0
        violation_time = 0.0
        for i in range(len(self.violation_timestamps) - 1):
            t, v   = self.violation_timestamps[i]
            next_t = self.violation_timestamps[i + 1][0]
            if v:
                violation_time += next_t - t
        if self.violation_timestamps[-1][1]:
            violation_time += current_time - self.violation_timestamps[-1][0]
        return min((violation_time / duration) * 100, 100.0)

    def overall_percentage(self):
        total_time = time.time() - self.start_time
        if total_time <= 0:
            return 0.0
        return min((self.total_violation_time / total_time) * 100, 100.0)

    def check_alerts(self):
        wp = self.window_percentage()
        op = self.overall_percentage()
        self.window_alert_triggered  = (wp >= WINDOW_THRESHOLD_PERCENT)
        self.overall_alert_triggered = (op >= OVERALL_THRESHOLD_PERCENT)
        combined = self.window_alert_triggered or self.overall_alert_triggered
        return self.window_alert_triggered, self.overall_alert_triggered, combined

    def get_stats(self):
        self.check_alerts()
        return {
            'window_percentage':    self.window_percentage(),
            'overall_percentage':   self.overall_percentage(),
            'total_violation_time': self.total_violation_time,
            'exam_duration':        time.time() - self.start_time,
            'window_alert':         self.window_alert_triggered,
            'overall_alert':        self.overall_alert_triggered,
        }


# ─────────────────────────────────────────────────────────────────────────────
# EYE GAZE + HEAD POSE MODULE
# ─────────────────────────────────────────────────────────────────────────────
class EyeHeadModule:
    LEFT_EYE   = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
    RIGHT_EYE  = [33, 7, 163,144,145,153,154,155,133,173,157,158,159,160,161,246]
    LEFT_IRIS  = [474,475,476,477]
    RIGHT_IRIS = [469,470,471,472]

    def __init__(self):
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=2, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        self.tracker = ViolationTracker(WINDOW_SIZE_MINUTES * 60)

    def _center(self, lm, indices, w, h):
        pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in indices]
        return np.mean(pts, axis=0).astype(int)

    def _gaze_ratio(self, eye, iris):
        h = np.clip((iris[0] - eye[0]) / 50 + 0.5, 0, 1)
        v = np.clip((iris[1] - eye[1]) / 30 + 0.5, 0, 1)
        return h, v

    def _gaze_direction(self, lr, rr):
        h = (lr[0] + rr[0]) / 2
        v = (lr[1] + rr[1]) / 2
        if h < 0.4: return "LEFT"
        if h > 0.6: return "RIGHT"
        if v < 0.4: return "UP"
        if v > 0.6: return "DOWN"
        return "CENTER"

    def _head_direction(self, lm, w, h):
        nose_tip   = lm[1]
        l_eye_out  = lm[263]
        r_eye_out  = lm[33]
        chin       = lm[152]
        forehead   = lm[10]
        nose_x     = nose_tip.x * w
        l_eye_x    = l_eye_out.x * w
        r_eye_x    = r_eye_out.x * w
        nose_y     = nose_tip.y * h
        chin_y     = chin.y * h
        forehead_y = forehead.y * h
        eye_cx      = (l_eye_x + r_eye_x) / 2
        horiz_off   = nose_x - eye_cx
        face_w      = abs(r_eye_x - l_eye_x)
        yaw_ratio   = (horiz_off / (face_w / 2)) if face_w > 0 else 0
        vert_off    = nose_y - ((forehead_y + chin_y) / 2)
        face_h      = abs(chin_y - forehead_y)
        pitch_ratio = (vert_off / (face_h / 2)) if face_h > 0 else 0
        direction    = "CENTER"
        looking_away = False
        if yaw_ratio < -HEAD_YAW_RATIO_THRESHOLD:
            direction, looking_away = "LEFT",  True
        elif yaw_ratio > HEAD_YAW_RATIO_THRESHOLD:
            direction, looking_away = "RIGHT", True
        if pitch_ratio < -HEAD_PITCH_RATIO_THRESHOLD:
            direction, looking_away = "UP",   True
        elif pitch_ratio > HEAD_PITCH_RATIO_THRESHOLD:
            direction, looking_away = "CENTER", False
        return direction, looking_away, yaw_ratio * 60, pitch_ratio * 45

    @staticmethod
    def _hex2bgr(h):
        h = h.lstrip('#')
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return (b, g, r)

    def _draw_stats(self, frame, w, h):
        stats = self.tracker.get_stats()
        x     = w - 450
        pairs = [
            (f"5-Min Window: {stats['window_percentage']:.1f}%",
             DANGER if stats['window_percentage'] >= WINDOW_THRESHOLD_PERCENT else SUCCESS),
            (f"Overall:      {stats['overall_percentage']:.1f}%",
             DANGER if stats['overall_percentage'] >= OVERALL_THRESHOLD_PERCENT else SUCCESS),
            (f"Viol Time:    {stats['total_violation_time']:.1f}s", WARNING),
            (f"Exam Time:    {stats['exam_duration']:.1f}s",        TEXT),
            (f"Win Alert:    {'YES' if stats['window_alert'] else 'NO'}",
             DANGER if stats['window_alert'] else SUCCESS),
            (f"Ovr Alert:    {'YES' if stats['overall_alert'] else 'NO'}",
             DANGER if stats['overall_alert'] else SUCCESS),
        ]
        for i, (txt, col) in enumerate(pairs):
            if isinstance(col, str):
                col = self._hex2bgr(col)
            cv2.putText(frame, txt, (x, 30 + i * 30), FONT_CV, 0.6, col, 2)

    def process(self, frame):
        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res  = self.mesh.process(rgb)
        is_violation = False
        if not res.multi_face_landmarks:
            is_violation = True
            cv2.putText(frame, "NO FACE DETECTED - VIOLATION",
                        (50, h // 2), FONT_CV, 1.0, (0, 0, 255), 3)
            self.tracker.update(is_violation)
            self._draw_stats(frame, w, h)
            return frame, is_violation
        if len(res.multi_face_landmarks) > 1:
            is_violation = True
            cv2.putText(frame, "MULTIPLE FACES - VIOLATION",
                        (50, h // 2), FONT_CV, 1.0, (0, 0, 255), 3)
            self.tracker.update(is_violation)
            self._draw_stats(frame, w, h)
            return frame, is_violation
        lm = res.multi_face_landmarks[0].landmark
        le = self._center(lm, self.LEFT_EYE,   w, h)
        re = self._center(lm, self.RIGHT_EYE,  w, h)
        li = self._center(lm, self.LEFT_IRIS,  w, h)
        ri = self._center(lm, self.RIGHT_IRIS, w, h)
        lr       = self._gaze_ratio(le, li)
        rr       = self._gaze_ratio(re, ri)
        gaze_dir = self._gaze_direction(lr, rr)
        head_dir, head_away, yaw_deg, pitch_deg = self._head_direction(lm, w, h)
        is_violation = (gaze_dir != "CENTER") or head_away
        self.tracker.update(is_violation)
        self.tracker.check_alerts()
        g_col = (0, 140, 255) if gaze_dir != "CENTER" else (0, 255, 0)
        h_col = (0, 140, 255) if head_away             else (0, 255, 0)
        s_col = (0, 0,   255) if is_violation           else (0, 255, 0)
        cv2.putText(frame, f"Eye: {gaze_dir}", (10, 30),  FONT_CV, 0.7, g_col, 2)
        cv2.putText(frame, f"Head:     {head_dir}", (10, 60),  FONT_CV, 0.7, h_col, 2)
        cv2.putText(frame, f"Status:   {'VIOLATING' if is_violation else 'NORMAL'}",
                    (10, 90), FONT_CV, 0.7, s_col, 2)
        cv2.circle(frame, tuple(li), 3, (255, 255, 0), -1)
        cv2.circle(frame, tuple(ri), 3, (255, 255, 0), -1)
        nose = (int(lm[1].x * w), int(lm[1].y * h))
        cv2.circle(frame, nose, 4, (255, 0, 255), -1)
        self._draw_stats(frame, w, h)
        return frame, is_violation
