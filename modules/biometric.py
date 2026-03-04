# modules/biometric.py — Component 2: Biometric auth, liveness, face verification

import os
import time
import threading
import pickle
import collections
import numpy as np
import cv2
import mediapipe as mp
import face_recognition
from PIL import Image, ImageTk

from config import (
    DB_DIR, MATCH_THRESHOLD, LOGIN_VERIFY_FRAMES, LOGIN_VERIFY_REQUIRED,
    REGISTER_FRAMES, EAR_THRESHOLD, BLINK_CONSEC_FRAMES, REQUIRED_BLINKS,
    LIVENESS_TIMEOUT, FACE_WINDOW_SECONDS, FACE_WINDOW_THRESHOLD,
    FONT_CV, DANGER, SUCCESS, WARNING
)


# ─────────────────────────────────────────────────────────────────────────────
# LIVENESS DETECTOR  (blink-based, used at login)
# ─────────────────────────────────────────────────────────────────────────────
class LivenessDetector:
    LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]

    def __init__(self):
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )

    @staticmethod
    def _ear(landmarks, indices, w, h):
        pts = np.array(
            [(landmarks[i].x * w, landmarks[i].y * h) for i in indices],
            dtype=np.float32,
        )
        A = np.linalg.norm(pts[1] - pts[5])
        B = np.linalg.norm(pts[2] - pts[4])
        C = np.linalg.norm(pts[0] - pts[3])
        return (A + B) / (2.0 * C) if C > 0 else 0.0

    def run(self, cap, status_label, cam_label, root):
        result   = [False]
        done_evt = threading.Event()
        blink_count   = 0
        consec_closed = 0
        eye_was_open  = True
        start_time    = time.time()

        def _tick():
            nonlocal blink_count, consec_closed, eye_was_open
            elapsed   = time.time() - start_time
            remaining = max(0, LIVENESS_TIMEOUT - elapsed)
            if elapsed > LIVENESS_TIMEOUT:
                status_label.config(text="✗  Liveness failed: no blink detected in time.", fg=DANGER)
                result[0] = False
                done_evt.set()
                return
            ret, frame = cap.read()
            if not ret:
                cam_label.after(50, _tick)
                return
            h, w = frame.shape[:2]
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res  = self._mesh.process(rgb)
            ear  = 1.0
            if res.multi_face_landmarks:
                lm  = res.multi_face_landmarks[0].landmark
                l_e = self._ear(lm, self.LEFT_EYE_IDX,  w, h)
                r_e = self._ear(lm, self.RIGHT_EYE_IDX, w, h)
                ear = (l_e + r_e) / 2.0
                if ear < EAR_THRESHOLD:
                    consec_closed += 1
                    eye_was_open   = False
                else:
                    if not eye_was_open and consec_closed >= BLINK_CONSEC_FRAMES:
                        blink_count += 1
                    consec_closed = 0
                    eye_was_open  = True
                colour = (0, 0, 255) if ear < EAR_THRESHOLD else (0, 255, 0)
                cv2.putText(frame, f"EAR: {ear:.2f}  Blinks: {blink_count}/{REQUIRED_BLINKS}",
                            (10, 30), FONT_CV, 0.7, colour, 2)
                cv2.putText(frame, f"Time: {remaining:.1f}s",
                            (10, 60), FONT_CV, 0.7, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "No face detected",
                            (10, 30), FONT_CV, 0.8, (0, 0, 255), 2)
            preview = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (240, 180))
            img = ImageTk.PhotoImage(Image.fromarray(preview))
            cam_label.configure(image=img)
            cam_label.image = img
            status_label.config(
                text=f"👁  Blink {blink_count}/{REQUIRED_BLINKS}  —  {remaining:.1f}s remaining",
                fg=WARNING)
            if blink_count >= REQUIRED_BLINKS:
                status_label.config(
                    text=f"✓  Liveness confirmed! ({blink_count} blinks)", fg=SUCCESS)
                result[0] = True
                done_evt.set()
                return
            cam_label.after(50, _tick)

        _tick()
        while not done_evt.is_set():
            root.update()
            time.sleep(0.01)
        return result[0]


# ─────────────────────────────────────────────────────────────────────────────
# FACE VERIFICATION MODULE  (during exam)
# ─────────────────────────────────────────────────────────────────────────────
class FaceVerificationModule:
    FRAME_SKIP           = 3
    EXAM_MATCH_THRESHOLD = 0.55
    NO_FACE_TOLERANCE    = 4

    # 3D depth liveness parameters
    CALIBRATION_FRAMES = 30
    LIVE_FRACTION      = 0.75
    Z_SPREAD_HARD_MIN  = 0.015
    SPOOF_BUFFER_SIZE  = 10
    SPOOF_FLAG_RATIO   = 0.40

    _DEPTH_LANDMARKS = [1, 33, 263, 152, 10, 234, 454, 4, 168, 61, 291]

    def __init__(self, student_name):
        student_file = os.path.join(DB_DIR, f"{student_name}.pkl")
        if not os.path.exists(student_file):
            raise FileNotFoundError(f"No registered data for '{student_name}'")
        with open(student_file, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            self.known_embeddings = data.get("embeddings", [])
            self.known_embedding  = data.get("mean_embedding",
                                             self.known_embeddings[0])
        else:
            self.known_embedding  = data
            self.known_embeddings = [data]

        self.window_start     = time.time()
        self.violation_frames = 0
        self.checked_frames   = 0
        self.frame_counter    = 0
        self.total_windows    = 0
        self.cheat_windows    = 0
        self.window_log       = []
        self._no_face_streak  = 0

        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )

        self._calib_spreads    = []
        self._z_threshold      = None
        self._calibrated       = False
        self._spoof_buffer     = collections.deque(maxlen=self.SPOOF_BUFFER_SIZE)
        self._last_z_spread    = 0.0
        self._liveness_ok      = True
        self._live_status_text = "CALIBRATING..."

    @staticmethod
    def _z_spread(landmarks) -> float:
        z_vals = [landmarks[i].z for i in FaceVerificationModule._DEPTH_LANDMARKS]
        return float(np.std(z_vals))

    def _check_liveness_frame(self, raw_frame: np.ndarray) -> bool:
        rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        res = self._mesh.process(rgb)
        if not res.multi_face_landmarks:
            self._live_status_text = "NO FACE"
            return True
        lm     = res.multi_face_landmarks[0].landmark
        spread = self._z_spread(lm)
        self._last_z_spread = spread

        if not self._calibrated:
            self._calib_spreads.append(spread)
            self._live_status_text = (
                f"CAL {len(self._calib_spreads)}/{self.CALIBRATION_FRAMES}")
            if len(self._calib_spreads) >= self.CALIBRATION_FRAMES:
                mean_spread       = float(np.mean(self._calib_spreads))
                self._z_threshold = max(
                    mean_spread * self.LIVE_FRACTION,
                    self.Z_SPREAD_HARD_MIN
                )
                self._calibrated  = True
                print(f"[Liveness-3D] Calibration complete: "
                      f"mean_spread={mean_spread:.4f}  "
                      f"threshold={self._z_threshold:.4f}")
            return True

        gray    = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var > 150:
            print("High-frequency reflection detected")
            return False

        if spread < self.Z_SPREAD_HARD_MIN:
            is_frame_live = False
        else:
            is_frame_live = spread >= self._z_threshold
        self._spoof_buffer.append(0 if is_frame_live else 1)

        spoof_votes = sum(self._spoof_buffer)
        spoof_ratio = spoof_votes / len(self._spoof_buffer)
        is_live     = spoof_ratio < self.SPOOF_FLAG_RATIO

        self._liveness_ok = is_live
        self._live_status_text = (
            f"LIVE  Z={spread:.4f}" if is_live
            else f"SPOOF Z={spread:.4f}<{self._z_threshold:.4f}"
        )
        print(f"[Liveness-3D] spread={spread:.4f}  "
              f"thr={self._z_threshold:.4f}  "
              f"spoof_ratio={spoof_ratio:.2f}  "
              f"→ {'LIVE' if is_live else 'SPOOF'}")
        return is_live

    def process(self, frame):
        is_live = self._check_liveness_frame(frame)

        self.frame_counter += 1
        if self.frame_counter % self.FRAME_SKIP == 0:
            encodings = face_recognition.face_encodings(frame)
            self.checked_frames += 1

            if not encodings:
                self._no_face_streak += 1
                if self._no_face_streak > self.NO_FACE_TOLERANCE:
                    self.violation_frames += 1
                    print(f"[FaceVerif] No face "
                          f"(streak={self._no_face_streak}) → violation")
            else:
                self._no_face_streak = 0
                dists_all   = face_recognition.face_distance(
                    self.known_embeddings, encodings[0])
                best_dist   = float(min(dists_all))
                dist_mean   = float(face_recognition.face_distance(
                    [self.known_embedding], encodings[0])[0])
                identity_ok = (best_dist <= self.EXAM_MATCH_THRESHOLD)

                print(f"[FaceVerif] Mean={dist_mean:.3f}  "
                      f"Best={best_dist:.3f}  "
                      f"Thr={self.EXAM_MATCH_THRESHOLD}  "
                      f"Identity={'OK' if identity_ok else 'FAIL'}  "
                      f"Liveness={self._live_status_text}")

                if not identity_ok:
                    self.violation_frames += 1
                    print(f"[FaceVerif] VIOLATION: identity mismatch "
                          f"best_dist={best_dist:.3f}")

                if self._calibrated and not is_live:
                    self.violation_frames += 1
                    print(f"[FaceVerif] VIOLATION: liveness SPOOF detected "
                          f"Z-spread={self._last_z_spread:.4f} "
                          f"< thr={self._z_threshold:.4f}")

        if time.time() - self.window_start >= FACE_WINDOW_SECONDS:
            self.total_windows += 1
            id_ratio = (self.violation_frames / self.checked_frames
                        if self.checked_frames > 0 else 0)
            verdict  = "CHEAT" if id_ratio > FACE_WINDOW_THRESHOLD else "CLEAN"
            print(f"[FaceVerif] Window {self.total_windows}: {verdict}  "
                  f"(violation_ratio={id_ratio:.2f}  "
                  f"live={self._live_status_text})")
            self.window_log.append(
                (self.total_windows, round(id_ratio, 3), verdict))
            if verdict == "CHEAT":
                self.cheat_windows += 1
            self.window_start     = time.time()
            self.violation_frames = 0
            self.checked_frames   = 0
            self._no_face_streak  = 0

        id_col = (0, 0, 255) if self._currently_suspicious() else (0, 255, 0)
        cv2.putText(frame,
                    f"Face Win: {self.cheat_windows}/{self.total_windows}",
                    (20, 150), FONT_CV, 0.6, id_col, 2)
        live_col = (0, 255, 0) if self._liveness_ok else (0, 0, 255)
        cv2.putText(frame,
                    f"Live: {self._live_status_text}",
                    (20, 175), FONT_CV, 0.5, live_col, 1)

    def _currently_suspicious(self):
        return (self.total_windows > 0
                and (self.cheat_windows / self.total_windows) > 0.3)

    def final_result(self):
        if self.total_windows == 0:
            return False
        return (self.cheat_windows / self.total_windows) > 0.3

    def cheat_ratio(self):
        if self.total_windows == 0:
            return 0.0
        return round(self.cheat_windows / self.total_windows, 3)


# ─────────────────────────────────────────────────────────────────────────────
# LOGIN-ONLY STRICT FACE VERIFIER
# ─────────────────────────────────────────────────────────────────────────────
def verify_face_strict(cap, known_embedding, n_frames=LOGIN_VERIFY_FRAMES,
                       required=LOGIN_VERIFY_REQUIRED):
    distances = []
    matches   = 0
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        encodings = face_recognition.face_encodings(frame)
        if not encodings:
            distances.append(999.0)
            time.sleep(0.15)
            continue
        dists = face_recognition.face_distance([known_embedding], encodings[0])
        d     = float(dists[0])
        distances.append(round(d, 4))
        if d < MATCH_THRESHOLD:
            matches += 1
        print(f"[FaceVerify] Frame {i+1}/{n_frames}: dist={d:.4f}  "
              f"{'MATCH' if d < MATCH_THRESHOLD else 'NO MATCH'}")
        time.sleep(0.15)
    passed = matches >= required
    print(f"[FaceVerify] Result: {matches}/{n_frames} matched  "
          f"(need {required}) → {'PASS' if passed else 'FAIL'}")
    return passed, matches, distances
