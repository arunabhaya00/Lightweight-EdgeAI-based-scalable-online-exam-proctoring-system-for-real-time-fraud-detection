# modules/object_detection.py — Component 3: Object & unauthorized person detection

import os
import csv
import json
import math
import time
import cv2
import numpy as np
from datetime import datetime

from config import (
    CONF_TEXT, CONF_PERSON, CONF_PHONE, TEMPORAL_THRESHOLD, MAX_MISSING_FRAMES,
    TEXT_MODEL_PATH, COCO_MODEL_PATH, FACE_MODEL_PATH,
    C3_CSV_LOG, C3_JSON_LOG, FONT_CV
)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[Component3] ultralytics not installed – object detection disabled.")


# ─────────────────────────────────────────────────────────────────────────────
# PERSON TRACKER
# ─────────────────────────────────────────────────────────────────────────────
class PersonTracker:
    def __init__(self, max_distance=80):
        self.next_id      = 0
        self.objects      = {}
        self.missing      = {}
        self.max_distance = max_distance

    def update(self, detections):
        updated  = {}
        used_ids = set()
        for (x1, y1, x2, y2) in detections:
            cx, cy    = (x1 + x2) // 2, (y1 + y2) // 2
            best_id, best_dist = None, self.max_distance
            for oid, (px, py) in self.objects.items():
                d = math.hypot(cx - px, cy - py)
                if d < best_dist:
                    best_dist, best_id = d, oid
            if best_id is not None:
                updated[best_id] = (cx, cy)
                self.missing[best_id] = 0
                used_ids.add(best_id)
            else:
                updated[self.next_id] = (cx, cy)
                self.missing[self.next_id] = 0
                used_ids.add(self.next_id)
                self.next_id += 1
        for oid in list(self.objects.keys()):
            if oid not in used_ids:
                self.missing[oid] = self.missing.get(oid, 0) + 1
                if self.missing[oid] <= MAX_MISSING_FRAMES:
                    updated[oid] = self.objects[oid]
                else:
                    self.missing.pop(oid, None)
        self.objects = updated
        return self.objects


# ─────────────────────────────────────────────────────────────────────────────
# OBJECT DETECTION MODULE
# ─────────────────────────────────────────────────────────────────────────────
class ObjectDetectionModule:
    def __init__(self, student_name: str):
        self.student_name = student_name
        self.text_model   = None
        self.coco_model   = None
        self.face_model   = None

        if YOLO_AVAILABLE:
            try:
                self.text_model = YOLO(TEXT_MODEL_PATH)
                print(f"[C3] Text model loaded: {TEXT_MODEL_PATH}")
            except Exception as e:
                print(f"[C3] Text model load failed: {e}")
            try:
                self.coco_model = YOLO(COCO_MODEL_PATH)
                print(f"[C3] COCO model loaded: {COCO_MODEL_PATH}")
            except Exception as e:
                print(f"[C3] COCO model load failed: {e}")
            try:
                self.face_model = YOLO(FACE_MODEL_PATH)
                print(f"[C3] Face model loaded: {FACE_MODEL_PATH}")
            except Exception as e:
                print(f"[C3] Face model load failed (yolov8n-face.pt): {e}")
                self.face_model = None

        self.tracker      = PersonTracker()
        self.face_tracker = PersonTracker(max_distance=80)

        self.violation_start  = None
        self.violation_active = False
        self._phone_active    = False
        self._text_active     = False
        self.csv_log          = C3_CSV_LOG
        self.json_log         = C3_JSON_LOG
        self._json_events: list = []

        if os.path.exists(self.json_log):
            try:
                with open(self.json_log, "r") as jf:
                    self._json_events = json.load(jf)
            except Exception:
                self._json_events = []
        if not os.path.exists(self.csv_log):
            with open(self.csv_log, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "violation_type", "student"])

        self.detected_objects: list = []
        self.violation_log:    list = []
        self._phone_detected    = False
        self._notebook_detected = False
        self._unauth_detected   = False

    def log_violation(self, vtype: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.csv_log, "a", newline="") as f:
            csv.writer(f).writerow([ts, vtype, self.student_name])
        self._json_events.append({"timestamp": ts, "violation": vtype,
                                   "student": self.student_name})
        with open(self.json_log, "w") as jf:
            json.dump(self._json_events, jf, indent=2)
        self.violation_log.append({"time": ts, "type": vtype})
        print(f"[C3] Violation logged: {vtype} at {ts}")

    @staticmethod
    def _detect_text_opencv(frame: np.ndarray) -> list:
        MIN_AREA      = 1200
        MAX_AREA_FRAC = 0.30
        MIN_SOLIDITY  = 0.25
        h_f, w_f      = frame.shape[:2]
        MAX_AREA      = int(h_f * w_f * MAX_AREA_FRAC)
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur   = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=15, C=8)
        k_h    = cv2.getStructuringElement(cv2.MORPH_RECT, (22, 3))
        k_v    = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 6))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_h)
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, k_v)
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA or area > MAX_AREA:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bh == 0:
                continue
            aspect   = bw / bh
            if not (0.3 <= aspect <= 20.0):
                continue
            solidity = area / (bw * bh)
            if solidity < MIN_SOLIDITY:
                continue
            boxes.append((x, y, x + bw, y + bh))
        return boxes

    @staticmethod
    def _merge_boxes(boxes: list, gap: int = 20) -> list:
        if not boxes:
            return []
        boxes  = sorted(boxes, key=lambda b: (b[1], b[0]))
        merged = [list(boxes[0])]
        for bx in boxes[1:]:
            last = merged[-1]
            if bx[0] <= last[2] + gap and bx[1] <= last[3] + gap:
                last[0] = min(last[0], bx[0])
                last[1] = min(last[1], bx[1])
                last[2] = max(last[2], bx[2])
                last[3] = max(last[3], bx[3])
            else:
                merged.append(list(bx))
        return [tuple(b) for b in merged]

    def process(self, frame):
        if self.coco_model is None and self.text_model is None and self.face_model is None:
            cv2.putText(frame, "C3: Models not loaded", (10, 180),
                        FONT_CV, 0.55, (128, 128, 128), 1)
            return frame, False

        person_boxes   = []
        phone_detected = False

        if self.coco_model is not None:
            for r in self.coco_model(frame, conf=CONF_PERSON, verbose=False):
                for box in r.boxes:
                    cls          = int(box.cls[0])
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    conf_val     = float(box.conf[0])
                    if cls == 0:
                        person_boxes.append((x1, y1, x2, y2))
                    elif cls == 67 and conf_val >= CONF_PHONE:
                        phone_detected = True
                        self._phone_detected = True
                        if "phone" not in self.detected_objects:
                            self.detected_objects.append("phone")
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                        cv2.putText(frame, f"PHONE ({conf_val:.2f})",
                                    (x1, max(y1 - 5, 14)),
                                    FONT_CV, 0.6, (255, 0, 255), 2)

        face_boxes = []
        if self.face_model is not None:
            for r in self.face_model(frame, conf=CONF_PERSON, verbose=False):
                for box in r.boxes:
                    conf_val     = float(box.conf[0])
                    if conf_val >= CONF_PERSON:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        face_boxes.append((x1, y1, x2, y2))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        cv2.putText(frame, f"FACE ({conf_val:.2f})",
                                    (x1, max(y1 - 5, 14)),
                                    FONT_CV, 0.55, (255, 255, 0), 1)

        tracked_faces = self.face_tracker.update(face_boxes)
        face_count    = len(tracked_faces)

        for fid, (cx, cy) in tracked_faces.items():
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
            cv2.putText(frame, f"F{fid}", (cx - 20, cy - 12),
                        FONT_CV, 0.55, (0, 255, 255), 1)

        tracked      = self.tracker.update(person_boxes)
        person_count = len(tracked)

        for pid, (cx, cy) in tracked.items():
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
            cv2.putText(frame, f"P{pid}", (cx - 20, cy - 10),
                        FONT_CV, 0.6, (255, 0, 0), 2)

        text_detected = False

        if self.text_model is not None:
            for r in self.text_model(frame, conf=CONF_TEXT, verbose=False):
                for box in r.boxes:
                    text_detected = True
                    self._notebook_detected = True
                    if "notebook/notes" not in self.detected_objects:
                        self.detected_objects.append("notebook/notes")
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"NOTES/TEXT (YOLO {float(box.conf[0]):.2f})",
                                (x1, max(y1 - 5, 14)),
                                FONT_CV, 0.55, (0, 0, 255), 2)

        cv_text_boxes = self._detect_text_opencv(frame)
        cv_text_boxes = self._merge_boxes(cv_text_boxes, gap=25)

        for (x1, y1, x2, y2) in cv_text_boxes:
            bw, bh = x2 - x1, y2 - y1
            if bh == 0 or bw / bh < 0.5:
                continue
            text_detected = True
            self._notebook_detected = True
            if "notebook/notes" not in self.detected_objects:
                self.detected_objects.append("notebook/notes")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
            cv2.putText(frame, "TEXT (CV)",
                        (x1, max(y1 - 5, 14)),
                        FONT_CV, 0.5, (0, 140, 255), 1)

        effective_person_count = max(person_count, face_count)

        now          = time.time()
        status_text  = "C3: CLEAR"
        color        = (0, 255, 0)
        is_violation = False

        if effective_person_count >= 2:
            self._unauth_detected = True
            if "unauthorized person" not in self.detected_objects:
                self.detected_objects.append("unauthorized person")
            if self.violation_start is None:
                self.violation_start = now
            elif now - self.violation_start >= TEMPORAL_THRESHOLD:
                status_text  = "C3: EXAM VIOLATION: MULTIPLE PERSONS"
                color        = (0, 0, 255)
                is_violation = True
                if not self.violation_active:
                    self.log_violation("multiple_persons")
                    self.violation_active = True
        else:
            self.violation_start  = None
            self.violation_active = False

        if phone_detected:
            status_text  = "C3: EXAM VIOLATION: MOBILE PHONE"
            color        = (0, 0, 255)
            is_violation = True
            if not self._phone_active:
                self.log_violation("mobile_phone")
                self._phone_active = True
        else:
            self._phone_active = False

        if text_detected:
            self._notebook_detected = True
            if "notebook/notes" not in self.detected_objects:
                self.detected_objects.append("notebook/notes")
            is_violation = True
            if status_text == "C3: CLEAR":
                status_text = "C3: SUSPICIOUS TEXT DETECTED"
                color       = (0, 165, 255)
            elif phone_detected:
                status_text = "C3: VIOLATION: PHONE + NOTES DETECTED"
                color       = (0, 0, 255)
            if not self._text_active:
                self.log_violation("text_detected")
                self._text_active = True
        else:
            self._text_active = False

        cv2.putText(frame, status_text, (20, 210), FONT_CV, 0.75, color, 2)
        cv2.putText(frame, f"C3 Persons (COCO): {person_count}", (20, 240),
                    FONT_CV, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"C3 Faces (YOLOv8-Face): {face_count}", (20, 265),
                    FONT_CV, 0.6, (255, 255, 0), 1)

        secondary = []
        if phone_detected:               secondary.append("PHONE")
        if text_detected:                secondary.append("NOTES/TEXT")
        if effective_person_count >= 2:  secondary.append(f"PERSONS({effective_person_count})")
        if secondary:
            cv2.putText(frame, "Detected: " + "  |  ".join(secondary),
                        (20, 290), FONT_CV, 0.55, (255, 255, 0), 1)

        return frame, is_violation

    def final_result(self) -> bool:
        return self._phone_detected or self._notebook_detected or self._unauth_detected

    def get_summary(self) -> dict:
        return {
            "phone_detected":    self._phone_detected,
            "notebook_detected": self._notebook_detected,
            "unauth_detected":   self._unauth_detected,
            "detected_objects":  self.detected_objects,
            "violation_log":     self.violation_log,
            "total_violations":  len(self.violation_log),
            "cheat":             self.final_result(),
        }
