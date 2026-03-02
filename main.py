import os
import cv2
import time
import pickle
import threading
import numpy as np
import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk
import mediapipe as mp
import face_recognition
from collections import deque
from docx import Document
import csv
import json
import math
from datetime import datetime

# ─────────────────────────── Component 4 imports ─────────────────────────────
import wave
import pyaudio
import tempfile

try:
    import soundfile as sf
    SF_AVAILABLE = True
except ImportError:
    SF_AVAILABLE = False
    print("[Component4] soundfile not installed – audio loading may be limited.")

try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[Component4] torch/torchaudio not installed – using mock embeddings.")

try:
    from scipy.spatial.distance import cosine as cosine_distance
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[Component4] scipy not installed – speaker verification disabled.")

try:
    import tensorflow as tf
    import tensorflow_hub as hub
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[Component4] tensorflow/tensorflow_hub not installed – using mock classification.")

# ─────────────────────────── try to import ultralytics ───────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[Component3] ultralytics not installed – object detection disabled.")

# ================= CONFIG =================
DB_DIR     = "db"
RESULT_DIR = "results"
VIDEO_FPS  = 20

WINDOW_SIZE_MINUTES       = 5
WINDOW_THRESHOLD_PERCENT  = 40
OVERALL_THRESHOLD_PERCENT = 20
FACE_WINDOW_SECONDS       = 10
FACE_WINDOW_THRESHOLD     = 0.4

MATCH_THRESHOLD = 0.45

LOGIN_VERIFY_FRAMES   = 5
LOGIN_VERIFY_REQUIRED = 4
REGISTER_FRAMES = 15

HEAD_YAW_RATIO_THRESHOLD   = 0.15
HEAD_PITCH_RATIO_THRESHOLD = 0.15

EAR_THRESHOLD       = 0.21
BLINK_CONSEC_FRAMES = 2
REQUIRED_BLINKS     = 2
LIVENESS_TIMEOUT    = 8

EXAM_DURATION_SECONDS = 60 * 60   # 1 hour default

# ─────────────── Component 3 config ──────────────
CONF_TEXT          = 0.25
CONF_PERSON        = 0.4
CONF_PHONE         = 0.4
TEMPORAL_THRESHOLD = 3.0
MAX_MISSING_FRAMES = 15
FONT_CV            = cv2.FONT_HERSHEY_SIMPLEX

TEXT_MODEL_PATH  = "exam_note_detector.pt"
COCO_MODEL_PATH  = "yolov8n.pt"
FACE_MODEL_PATH  = "yolov8n-face.pt"   # ← NEW: face detection model

C3_CSV_LOG  = os.path.join("results", "violations.csv")
C3_JSON_LOG = os.path.join("results", "violations.json")

# ─────────────── Component 4 config ──────────────
C4_AUDIO_CHUNK       = 2048
C4_AUDIO_FORMAT      = pyaudio.paInt16 if 'pyaudio' in dir() else None
C4_AUDIO_CHANNELS    = 1
C4_AUDIO_RATE        = 16000
C4_ECAPA_MODEL_PATH  = "ecapa_quantized_traced.pth"
C4_YAMNET_CLASSIFIER = "yamnet_audio_classifier.h5"
C4_YAMNET_HUB_URL    = "https://tfhub.dev/google/yamnet/1"
C4_SPEAKER_THRESHOLD = 0.35    # cosine distance < 0.35 → same speaker
C4_SEGMENT_SECONDS   = 10      # classify in 10-second windows
C4_INTEGRITY_PASS    = 80      # ≥80 % OK → NOT CHEAT
C4_INTEGRITY_REVIEW  = 60      # 60–79 % → REVIEW
                               # <60 % → CHEAT

# Calibration duration shown on InstructionPage
C4_CALIBRATION_SECONDS = 15

os.makedirs(DB_DIR,     exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ================= COLOUR PALETTE =================
BG       = "#0d0f14"
BG2      = "#13161e"
BG3      = "#1a1e2a"
ACCENT   = "#00e5ff"
ACCENT2  = "#7c3aed"
SUCCESS  = "#00e676"
WARNING  = "#ffab00"
DANGER   = "#ff1744"
TEXT     = "#e8eaf6"
TEXT_DIM = "#607d8b"
CARD     = "#181c27"
BORDER   = "#252a38"

FONT_H1   = ("Courier New", 26, "bold")
FONT_H2   = ("Courier New", 18, "bold")
FONT_H3   = ("Courier New", 13, "bold")
FONT_BODY = ("Courier New", 11)
FONT_MONO = ("Courier New", 10)


# =========================================================
# COMPONENT 4 — AUDIO INTEGRITY MODULE
# =========================================================

# ── Model loaders (cached at module level) ────────────────
_ecapa_model    = None
_mel_extractor  = None
_yamnet_base    = None
_yamnet_cls     = None
_c4_models_loaded = False


def _load_c4_models():
    """Load ECAPA + YAMNet models once at startup."""
    global _ecapa_model, _mel_extractor, _yamnet_base, _yamnet_cls, _c4_models_loaded
    if _c4_models_loaded:
        return

    # ECAPA
    if TORCH_AVAILABLE:
        try:
            _ecapa_model = torch.jit.load(C4_ECAPA_MODEL_PATH)
            _ecapa_model.eval()
            _mel_extractor = torchaudio.transforms.MelSpectrogram(
                sample_rate=C4_AUDIO_RATE, n_fft=400, win_length=400,
                hop_length=160, n_mels=80
            )
            print("[C4] ECAPA model loaded.")
        except Exception as e:
            print(f"[C4] ECAPA model not found / error: {e}")

    # YAMNet
    if TF_AVAILABLE:
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            cache_dir  = os.path.join(script_dir, "tfhub_cache")
            os.makedirs(cache_dir, exist_ok=True)
            os.environ['TFHUB_CACHE_DIR'] = cache_dir

            _yamnet_base = hub.load(C4_YAMNET_HUB_URL)
            print("[C4] YAMNet base loaded.")

            cls_path = os.path.join(script_dir, C4_YAMNET_CLASSIFIER)
            if os.path.exists(cls_path):
                _yamnet_cls = keras.models.load_model(cls_path)
                print("[C4] YAMNet classifier loaded.")
            else:
                print(f"[C4] YAMNet classifier not found at {cls_path}.")
        except Exception as e:
            print(f"[C4] YAMNet load error: {e}")

    _c4_models_loaded = True


def _get_voice_embedding(audio_array: np.ndarray) -> np.ndarray:
    """Return 192-dim ECAPA embedding, or random mock if model unavailable."""
    if _ecapa_model is None or _mel_extractor is None:
        return np.random.rand(192).astype(np.float32)
    waveform = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)
    waveform = waveform / (waveform.abs().max() + 1e-8)
    mel = _mel_extractor(waveform).transpose(1, 2)
    with torch.no_grad():
        emb = _ecapa_model(mel)
    return emb.squeeze().numpy()


def _classify_segment(audio_seg: np.ndarray, baseline_emb: np.ndarray,
                       segment_idx: int) -> dict:
    """
    Classify one 10-second audio segment.
    Returns a dict with keys: segment, time, class, confidence,
    speaker_match, speaker_similarity, status, alert
    """
    CLASS_NAMES = ['Self-Speaking', 'Whispering', 'Lecture/Video', 'Background Noise']

    # ── Speaker verification ──────────────────────────────
    speaker_match      = False
    speaker_similarity = 0.0
    if baseline_emb is not None and SCIPY_AVAILABLE:
        try:
            test_emb           = _get_voice_embedding(audio_seg)
            dist               = cosine_distance(baseline_emb, test_emb)
            speaker_similarity = max(0.0, 1.0 - dist)
            speaker_match      = dist < C4_SPEAKER_THRESHOLD
        except Exception as e:
            print(f"[C4] Speaker verification error: {e}")

    # ── Audio classification ──────────────────────────────
    if _yamnet_base is None or _yamnet_cls is None:
        # Mock: if speaker matches → self-speaking, else random suspicious
        if speaker_match:
            predicted_class, confidence = 0, 0.85
        else:
            predicted_class = int(np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2]))
            confidence      = float(np.random.uniform(0.7, 0.9))
    else:
        try:
            wf_tf = tf.convert_to_tensor(audio_seg, dtype=tf.float32)
            _, embeddings, _ = _yamnet_base(wf_tf)
            emb_mean = np.mean(embeddings.numpy(), axis=0, keepdims=True)
            preds    = _yamnet_cls.predict(emb_mean, verbose=0)

            # Bias correction: penalise Lecture/Video (class 2) by 40 %
            adj = preds[0].copy()
            adj[2] *= 0.6

            predicted_class = int(np.argmax(adj))
            confidence      = float(preds[0][predicted_class])

            # Override: speaker-matched → Self-Speaking
            if speaker_match:
                predicted_class = 0
                confidence      = max(confidence, 0.75)
            else:
                # Reclassify borderline Lecture/Video
                if predicted_class == 2:
                    whisper_p  = adj[1]
                    lecture_p  = adj[2]
                    if confidence < 0.70 or (whisper_p > 0.3 and lecture_p < 0.6):
                        if whisper_p > 0.25:
                            predicted_class = 1
                            confidence      = float(preds[0][1])
                        else:
                            predicted_class = 3
                            confidence      = float(preds[0][3])
        except Exception as e:
            print(f"[C4] Classification error: {e}")
            predicted_class = 0 if speaker_match else 1
            confidence      = 0.5

    # Whispering (1) and Lecture/Video (2) are suspicious
    is_suspicious = predicted_class in [1, 2]
    t0 = segment_idx * C4_SEGMENT_SECONDS
    return {
        "segment":            segment_idx,
        "time":               f"{t0}s – {t0 + C4_SEGMENT_SECONDS}s",
        "class":              CLASS_NAMES[predicted_class],
        "confidence":         confidence,
        "speaker_match":      speaker_match,
        "speaker_similarity": speaker_similarity,
        "status":             "SUSPICIOUS" if is_suspicious else "OK",
        "alert":              "🚨" if is_suspicious else "✅",
    }


class AudioIntegrityModule:
    """
    Component 4 — manages voice calibration, background audio recording,
    post-exam classification, and produces a verdict.

    Lifecycle:
        1.  calibrate(duration)  →  captures baseline voice embedding
        2.  start_recording()   →  spawns background thread writing to WAV
        3.  stop_and_analyse()  →  stops thread, runs classification
        4.  final_result()      →  bool  (True = cheat)
        5.  get_summary()       →  dict
    """

    def __init__(self, student_name: str):
        _load_c4_models()   # no-op if already loaded

        self.student_name       = student_name
        self.baseline_embedding = None
        self.calibration_audio  = None

        self._stop_event   = None
        self._audio_thread = None
        self._wav_path     = None

        self.segments:      list[dict] = []
        self.integrity_score: float    = 100.0
        self._verdict: str             = "PENDING"
        self._is_cheat: bool           = False

    # ── Phase 1: calibration ─────────────────────────────
    def calibrate(self, duration: int = C4_CALIBRATION_SECONDS,
                  status_cb=None) -> bool:
        """
        Record `duration` seconds from the microphone, extract baseline embedding.
        `status_cb(msg: str)` is called with progress text if provided.
        Returns True on success.
        """
        frames = []
        p = stream = None
        try:
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16, channels=C4_AUDIO_CHANNELS,
                            rate=C4_AUDIO_RATE, input=True,
                            frames_per_buffer=C4_AUDIO_CHUNK)
            total_chunks = int(C4_AUDIO_RATE / C4_AUDIO_CHUNK * duration)
            for i in range(total_chunks):
                data = stream.read(C4_AUDIO_CHUNK, exception_on_overflow=False)
                frames.append(data)
                if status_cb and i % max(1, total_chunks // duration) == 0:
                    elapsed = int(i * C4_AUDIO_CHUNK / C4_AUDIO_RATE)
                    status_cb(f"🎤  Calibrating… {elapsed}/{duration}s")
        except Exception as e:
            print(f"[C4] Calibration recording error: {e}")
            return False
        finally:
            if stream:
                try: stream.stop_stream(); stream.close()
                except: pass
            if p:
                try: p.terminate()
                except: pass

        if not frames:
            return False

        audio_bytes = b''.join(frames)
        arr = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self.calibration_audio  = arr
        self.baseline_embedding = _get_voice_embedding(arr)
        print(f"[C4] Calibration done. Embedding shape: {self.baseline_embedding.shape}")
        return True

    # ── Phase 2: start recording ─────────────────────────
    def start_recording(self, max_duration_seconds: int = EXAM_DURATION_SECONDS + 60):
        ts = int(time.time())
        self._wav_path  = os.path.join(
            RESULT_DIR, f"{self.student_name}_{ts}_audio.wav")
        self._stop_event = threading.Event()
        self._audio_thread = threading.Thread(
            target=self._record_loop,
            args=(self._wav_path, max_duration_seconds, self._stop_event),
            daemon=True,
            name="C4_AudioRecorder"
        )
        self._audio_thread.start()
        print(f"[C4] Recording started → {self._wav_path}")

    def _record_loop(self, filename, duration, stop_event):
        frames = []
        p = stream = None
        try:
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16, channels=C4_AUDIO_CHANNELS,
                            rate=C4_AUDIO_RATE, input=True,
                            frames_per_buffer=C4_AUDIO_CHUNK)
            start = time.time()
            while not stop_event.is_set():
                if time.time() - start >= duration:
                    break
                try:
                    data = stream.read(C4_AUDIO_CHUNK, exception_on_overflow=False)
                    frames.append(data)
                except:
                    if stop_event.is_set():
                        break
                    time.sleep(0.01)
        finally:
            if stream:
                try: stream.stop_stream(); stream.close()
                except: pass
            if p:
                try: p.terminate()
                except: pass
            if frames:
                try:
                    with wave.open(filename, 'wb') as wf:
                        wf.setnchannels(C4_AUDIO_CHANNELS)
                        wf.setsampwidth(2)
                        wf.setframerate(C4_AUDIO_RATE)
                        wf.writeframes(b''.join(frames))
                    dur_s = len(frames) * C4_AUDIO_CHUNK / C4_AUDIO_RATE
                    print(f"[C4] WAV saved: {dur_s:.1f}s → {filename}")
                except Exception as e:
                    print(f"[C4] WAV save error: {e}")

    # ── Phase 3: stop + analyse ───────────────────────────
    def stop_and_analyse(self, status_cb=None) -> bool:
        """
        Signal the recording thread to stop, wait for it, then classify.
        Returns True if analysis succeeded.
        """
        if self._stop_event:
            self._stop_event.set()
        if self._audio_thread and self._audio_thread.is_alive():
            self._audio_thread.join(timeout=8)
        print("[C4] Recording thread stopped.")

        wav = self._wav_path
        if not wav or not os.path.exists(wav):
            print("[C4] No WAV file found – skipping classification.")
            self._compute_verdict([])
            return False

        # Load WAV
        try:
            if SF_AVAILABLE:
                audio_data, _sr = sf.read(wav)
            else:
                # fallback: scipy or wave
                with wave.open(wav, 'rb') as wf:
                    raw = wf.readframes(wf.getnframes())
                audio_data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            audio_data = audio_data.flatten()
            if np.abs(audio_data).max() > 0:
                audio_data = audio_data / np.abs(audio_data).max()
        except Exception as e:
            print(f"[C4] WAV load error: {e}")
            self._compute_verdict([])
            return False

        # Slice and classify
        seg_len  = C4_AUDIO_RATE * C4_SEGMENT_SECONDS
        n_segs   = len(audio_data) // seg_len
        results  = []

        for i in range(n_segs):
            if status_cb:
                status_cb(f"🔊  Analysing segment {i+1}/{n_segs}…")
            seg = audio_data[i * seg_len : (i + 1) * seg_len]
            results.append(_classify_segment(seg, self.baseline_embedding, i))
            print(f"[C4] Seg {i+1}/{n_segs}: {results[-1]['class']}  "
                  f"({results[-1]['status']})")

        self.segments = results
        self._compute_verdict(results)

        # Save text report into results/
        self._save_text_report(wav, audio_data)
        return True

    def _compute_verdict(self, results):
        if not results:
            self.integrity_score = 100.0
            self._verdict        = "NOT CHEAT"
            self._is_cheat       = False
            return
        ok_count = sum(1 for r in results if r["status"] == "OK")
        self.integrity_score = (ok_count / len(results)) * 100.0
        if self.integrity_score >= C4_INTEGRITY_PASS:
            self._verdict  = "NOT CHEAT"
            self._is_cheat = False
        elif self.integrity_score >= C4_INTEGRITY_REVIEW:
            self._verdict  = "REVIEW"
            self._is_cheat = False   # not hard-flagged, but flagged in report
        else:
            self._verdict  = "*** CHEAT ***"
            self._is_cheat = True

    def _save_text_report(self, wav_path, audio_data):
        ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        name = f"{self.student_name}_{int(time.time())}_audio_analysis.txt"
        path = os.path.join(RESULT_DIR, name)
        total_secs = len(audio_data) / C4_AUDIO_RATE if audio_data is not None else 0

        lines = [
            "=" * 70,
            "  COMPONENT 4 — AUDIO INTEGRITY ANALYSIS REPORT",
            "=" * 70,
            f"  Student         : {self.student_name}",
            f"  Generated At    : {ts}",
            f"  WAV Recording   : {wav_path}",
            f"  Total Duration  : {total_secs:.1f} s",
            f"  Total Segments  : {len(self.segments)}",
            f"  Integrity Score : {self.integrity_score:.1f}%",
            f"  Verdict         : {self._verdict}",
            "",
            "─" * 70,
            f"  {'Seg':>4}  {'Time Range':<18}  {'Class':<18}  "
            f"{'Conf':>6}  {'SpeakerMatch':>12}  {'Status'}",
            "─" * 70,
        ]
        for r in self.segments:
            lines.append(
                f"  {r['segment']+1:>4}  {r['time']:<18}  {r['class']:<18}  "
                f"{r['confidence']:>6.1%}  "
                f"{'YES' if r['speaker_match'] else 'NO':>12}  "
                f"{r['status']} {r['alert']}"
            )
        lines += [
            "─" * 70,
            "",
            f"  FINAL VERDICT : {self._verdict}",
            "=" * 70,
            "",
        ]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        self._report_path = path
        print(f"[C4] Audio report saved → {path}")

    # ── Public accessors ──────────────────────────────────
    def final_result(self) -> bool:
        """True when the verdict is hard CHEAT (integrity < 60 %)."""
        return self._is_cheat

    def verdict(self) -> str:
        return self._verdict

    def get_summary(self) -> dict:
        return {
            "integrity_score": self.integrity_score,
            "verdict":         self._verdict,
            "is_cheat":        self._is_cheat,
            "segments":        self.segments,
            "total_segments":  len(self.segments),
            "suspicious":      sum(1 for s in self.segments if s["status"] == "SUSPICIOUS"),
            "wav_path":        self._wav_path or "N/A",
            "report_path":     getattr(self, "_report_path", "N/A"),
        }

    @property
    def wav_path(self):
        return self._wav_path or "N/A"


# =========================================================
# FACE VERIFIER  — standalone helper used at login
# =========================================================
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


# =========================================================
# LIVENESS DETECTOR
# =========================================================
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


# =========================================================
# VIOLATION TRACKER
# =========================================================
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


# =========================================================
# EYE GAZE + HEAD POSE MODULE
# =========================================================
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
            direction, looking_away = "DOWN", True
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
        cv2.putText(frame, f"Eye Gaze: {gaze_dir}", (10, 30),  FONT_CV, 0.7, g_col, 2)
        cv2.putText(frame, f"Head:     {head_dir}", (10, 60),  FONT_CV, 0.7, h_col, 2)
        cv2.putText(frame, f"Status:   {'VIOLATING' if is_violation else 'NORMAL'}",
                    (10, 90), FONT_CV, 0.7, s_col, 2)
        cv2.circle(frame, tuple(li), 3, (255, 255, 0), -1)
        cv2.circle(frame, tuple(ri), 3, (255, 255, 0), -1)
        nose = (int(lm[1].x * w), int(lm[1].y * h))
        cv2.circle(frame, nose, 4, (255, 0, 255), -1)
        self._draw_stats(frame, w, h)
        return frame, is_violation


# =========================================================
# FACE VERIFICATION MODULE  (during exam)
# =========================================================
class FaceVerificationModule:
    # Every Nth frame is checked by face_recognition (heavy CNN).
    FRAME_SKIP = 3

    # ── Identity threshold ─────────────────────────────────────────────────
    # Relaxed vs login gate (0.45) to tolerate pose/lighting drift during exam.
    EXAM_MATCH_THRESHOLD = 0.55

    # Consecutive no-face frames tolerated before counting as a violation.
    NO_FACE_TOLERANCE = 4

    # ── Liveness method: MediaPipe 3D face depth analysis ─────────────────
    #
    # WHY previous methods failed:
    #   • LBP texture variance — a printed photo has the same pixel texture
    #     as a real face at webcam resolution; too unreliable.
    #   • EAR blink detection — MediaPipe FaceMesh works on flat photos too.
    #     A moving hand holding a photo causes apparent EAR variance, so the
    #     variance check gets fooled. Blink detection requires 30 s to judge.
    #
    # NEW METHOD — 3D Z-DEPTH SPREAD:
    #   MediaPipe FaceMesh returns 478 landmarks with X, Y, Z coordinates
    #   normalised to the image. The Z axis encodes relative depth.
    #
    #   REAL 3D FACE:
    #     The nose tip protrudes forward, eyes are recessed, chin is forward,
    #     temples are flat. The Z values span a wide range → large Z std-dev.
    #
    #   FLAT PRINTED PHOTO / SCREEN:
    #     All landmarks lie on a single flat plane regardless of how the photo
    #     is tilted. The Z values MediaPipe returns for a flat surface are all
    #     near-identical → very small Z std-dev.
    #
    #   This signal is immune to:
    #     • Hand tremor / photo movement  (Z spread stays flat for a photo)
    #     • Lighting changes              (Z is geometry, not colour)
    #     • Video replays on screen       (screen is flat → same effect)
    #
    # CALIBRATION:
    #   We collect the first CALIBRATION_FRAMES live Z-spread values and set
    #   the threshold dynamically as:
    #       threshold = mean_live_z_spread * LIVE_FRACTION
    #   This adapts to the specific webcam and student distance, avoiding a
    #   single hard-coded number that might not suit all setups.
    #   During calibration frames the student is always live (just logged in),
    #   so the baseline is reliable.

    # 3D depth parameters
    CALIBRATION_FRAMES  = 30     # frames to collect baseline Z spread
    LIVE_FRACTION       = 0.40   # threshold = live_baseline * this fraction
    Z_SPREAD_HARD_MIN   = 0.005  # absolute floor — below this always = spoof
                                 # regardless of calibration

    # Rolling window for spoof verdict (avoid single-frame false positives)
    SPOOF_BUFFER_SIZE   = 10     # last N liveness results kept
    SPOOF_FLAG_RATIO    = 0.60   # if ≥ 60% of buffer = spoof → flag violation

    # Key landmark indices for Z-depth measurement
    # Chosen to maximise 3D spread: nose tip, eyes, chin, forehead, temples
    _DEPTH_LANDMARKS = [
        1,    # nose tip       (most forward point)
        33,   # right eye outer
        263,  # left eye outer
        152,  # chin
        10,   # forehead top
        234,  # right temple
        454,  # left temple
        4,    # nose bottom
        168,  # nose bridge
        61,   # right mouth corner
        291,  # left mouth corner
    ]

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

        # ── Identity tracking ─────────────────────────────────────────────
        self.window_start     = time.time()
        self.violation_frames = 0
        self.checked_frames   = 0
        self.frame_counter    = 0
        self.total_windows    = 0
        self.cheat_windows    = 0
        self.window_log       = []
        self._no_face_streak  = 0

        # ── Liveness: MediaPipe FaceMesh (own instance) ───────────────────
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Calibration state
        self._calib_spreads     = []        # Z spreads collected during calibration
        self._z_threshold       = None      # set after calibration
        self._calibrated        = False

        # Rolling spoof buffer
        self._spoof_buffer      = deque(maxlen=self.SPOOF_BUFFER_SIZE)

        # HUD state
        self._last_z_spread     = 0.0
        self._liveness_ok       = True
        self._live_status_text  = "CALIBRATING..."

    # ── Measure Z-depth spread from MediaPipe landmarks ───────────────────
    @staticmethod
    def _z_spread(landmarks) -> float:
        """
        Extract Z coordinates of key depth landmarks and return their
        standard deviation.  Larger value = more 3D depth = more likely live.
        """
        z_vals = [landmarks[i].z for i in FaceVerificationModule._DEPTH_LANDMARKS]
        return float(np.std(z_vals))

    # ── Per-frame liveness check ──────────────────────────────────────────
    def _check_liveness_frame(self, raw_frame: np.ndarray) -> bool:
        """
        Runs MediaPipe FaceMesh on raw_frame (unmodified BGR frame).
        Returns True = LIVE, False = SPOOF.

        Phase 1 — Calibration (first CALIBRATION_FRAMES frames with a face):
            Collect Z-spread baseline from the live student who just logged in.
            After calibration, set threshold = mean_spread * LIVE_FRACTION.

        Phase 2 — Active checking:
            Each frame compute Z spread.
            If spread < threshold → spoof vote in rolling buffer.
            If ≥ SPOOF_FLAG_RATIO of buffer = spoof → return False.
        """
        rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        res = self._mesh.process(rgb)

        if not res.multi_face_landmarks:
            # No face found — cannot judge — benefit of the doubt
            self._live_status_text = "NO FACE"
            return True

        lm     = res.multi_face_landmarks[0].landmark
        spread = self._z_spread(lm)
        self._last_z_spread = spread

        # ── Phase 1: calibration ─────────────────────────────────────────
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
            return True  # always live during calibration (just logged in)

        # ── Phase 2: active liveness check ──────────────────────────────
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

    # ── Main per-frame processing ──────────────────────────────────────────
    def process(self, frame):
        # Liveness must use the raw frame before any drawings are overlaid.
        # The proctor_loop passes frame_c12 (already drawn on by EyeHeadModule).
        # We use it as-is — MediaPipe is robust to minor overlay text, and
        # the Z-depth measurement is a geometric property of landmarks,
        # not affected by colour overlays.
        is_live = self._check_liveness_frame(frame)

        # ── Identity + liveness violations on FRAME_SKIP cadence ─────────
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

                # Liveness violation — only count after calibration is done
                # so the first ~1.5 s of the exam never flags false positives
                if self._calibrated and not is_live:
                    self.violation_frames += 1
                    print(f"[FaceVerif] VIOLATION: liveness SPOOF detected "
                          f"Z-spread={self._last_z_spread:.4f} "
                          f"< thr={self._z_threshold:.4f}")

        # ── 10-second window verdict ───────────────────────────────────────
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

        # ── On-frame HUD ──────────────────────────────────────────────────
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


# =========================================================
# COMPONENT 3 — PERSON TRACKER
# =========================================================
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


# =========================================================
# COMPONENT 3 — OBJECT & PERSON DETECTION MODULE
# =========================================================
class ObjectDetectionModule:
    def __init__(self, student_name: str):
        self.student_name = student_name
        self.text_model = None
        self.coco_model = None
        self.face_model = None   # ← NEW: yolov8n-face.pt model

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
            # ── NEW: load face detection model ──────────────────────────
            try:
                self.face_model = YOLO(FACE_MODEL_PATH)
                print(f"[C3] Face model loaded: {FACE_MODEL_PATH}")
            except Exception as e:
                print(f"[C3] Face model load failed (yolov8n-face.pt): {e}")
                self.face_model = None
            # ────────────────────────────────────────────────────────────

        self.tracker = PersonTracker()

        # ── NEW: dedicated face tracker using the same PersonTracker logic
        #        with MAX_MISSING_FRAMES smoothing buffer (15 frames)
        self.face_tracker = PersonTracker(max_distance=80)
        # ────────────────────────────────────────────────────────────────

        self.violation_start  = None
        self.violation_active = False
        self._phone_active = False
        self._text_active  = False
        self.csv_log  = C3_CSV_LOG
        self.json_log = C3_JSON_LOG
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
        self.detected_objects: list[str] = []
        self.violation_log:    list[dict] = []
        self._phone_detected       = False
        self._notebook_detected    = False
        self._unauth_detected      = False

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

    # ── Static helper: OpenCV-based text/paper region detector ──────────
    @staticmethod
    def _detect_text_opencv(frame: np.ndarray) -> list:
        """
        Detect text-bearing regions (paper, notes, printed sheets) using
        pure OpenCV — no custom YOLO model required.

        Pipeline:
          1. Convert to grayscale.
          2. Adaptive threshold to binarise text from background.
          3. Morphological closing to merge nearby text blobs into word/line regions.
          4. Find external contours and filter by aspect ratio and area to keep
             only plausible paper/note rectangles.
          5. Return list of (x1, y1, x2, y2) bounding boxes.
        """
        MIN_AREA      = 1200    # px²  — smallest accepted text block
        MAX_AREA_FRAC = 0.30    # at most 30 % of total frame area
        MIN_SOLIDITY  = 0.25
        h_f, w_f      = frame.shape[:2]
        MAX_AREA      = int(h_f * w_f * MAX_AREA_FRAC)

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=15, C=8
        )
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

    # ── Static helper: merge overlapping / nearby boxes ──────────────────
    @staticmethod
    def _merge_boxes(boxes: list, gap: int = 20) -> list:
        """
        Merge bounding boxes that are close together (within `gap` pixels)
        into a single larger box.
        """
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

        # ── COCO model: phone (cls 67) + person (cls 0) detection ────────
        if self.coco_model is not None:
            for r in self.coco_model(frame, conf=CONF_PERSON, verbose=False):
                for box in r.boxes:
                    cls        = int(box.cls[0])
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    conf_val   = float(box.conf[0])
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

        # ── NEW: yolov8n-face.pt — high-speed single-class face detection ─
        #    Uses CONF_PERSON (0.4) threshold as specified in the instructions.
        #    Detected face bounding boxes are fed into the dedicated face_tracker
        #    which applies MAX_MISSING_FRAMES (15) smoothing for stable tracking
        #    under varying angles and lighting conditions.
        face_boxes = []
        if self.face_model is not None:
            for r in self.face_model(frame, conf=CONF_PERSON, verbose=False):
                for box in r.boxes:
                    conf_val     = float(box.conf[0])
                    if conf_val >= CONF_PERSON:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        face_boxes.append((x1, y1, x2, y2))
                        # Draw face detection box (cyan) for visual distinction
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        cv2.putText(frame, f"FACE ({conf_val:.2f})",
                                    (x1, max(y1 - 5, 14)),
                                    FONT_CV, 0.55, (255, 255, 0), 1)

        # Update face tracker with detections; MAX_MISSING_FRAMES buffer keeps
        # tracks alive for up to 15 consecutive missing frames (momentary occlusion)
        tracked_faces = self.face_tracker.update(face_boxes)
        face_count    = len(tracked_faces)

        # Draw face tracker IDs
        for fid, (cx, cy) in tracked_faces.items():
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
            cv2.putText(frame, f"F{fid}", (cx - 20, cy - 12),
                        FONT_CV, 0.55, (0, 255, 255), 1)
        # ─────────────────────────────────────────────────────────────────

        # ── COCO person tracker (body-level) ─────────────────────────────
        tracked       = self.tracker.update(person_boxes)
        person_count  = len(tracked)

        for pid, (cx, cy) in tracked.items():
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
            cv2.putText(frame, f"P{pid}", (cx - 20, cy - 10),
                        FONT_CV, 0.6, (255, 0, 0), 2)

        # ── Text / notes detection ────────────────────────────────────────
        # Two-stage pipeline:
        #   Stage 1 — YOLO custom model (exam_note_detector.pt) if available.
        #   Stage 2 — OpenCV adaptive-threshold + contour fallback, which runs
        #             always and catches paper/text even when the YOLO model is
        #             absent or misses a detection.
        # A detection from EITHER stage triggers the violation.
        text_detected = False

        # Stage 1: YOLO custom text model
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

        # Stage 2: OpenCV fallback — always active regardless of YOLO model
        # Suppress stage 2 if the frame contains only the student's face
        # (i.e. person not holding anything up). We check INSIDE the ROI
        # below the face region so we don't flag the student's own shirt text.
        cv_text_boxes = self._detect_text_opencv(frame)
        cv_text_boxes = self._merge_boxes(cv_text_boxes, gap=25)

        for (x1, y1, x2, y2) in cv_text_boxes:
            bw, bh = x2 - x1, y2 - y1
            # Filter out very tall/thin detections (likely face/hair edges)
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

        # ── Multi-person violation logic ──────────────────────────────────
        # Use the greater of body-count (COCO) and face-count (face model)
        # for the most robust unauthorized person detection.
        effective_person_count = max(person_count, face_count)

        now         = time.time()
        status_text = "C3: CLEAR"
        color       = (0, 255, 0)
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
        # ── NEW: show face model count alongside COCO count ───────────────
        cv2.putText(frame, f"C3 Faces (YOLOv8-Face): {face_count}", (20, 265),
                    FONT_CV, 0.6, (255, 255, 0), 1)
        # ─────────────────────────────────────────────────────────────────

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


# =========================================================
# REPORT GENERATOR
# =========================================================
def generate_report(app, answer_vars, questions):
    student   = app.current_student or "Unknown"
    eye_mod   = app.eye
    face_mod  = app.face
    obj_mod   = app.obj
    audio_mod = app.audio          # Component 4
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if eye_mod:
        eye_mod.tracker.check_alerts()
        et        = eye_mod.tracker
        wp        = et.window_percentage()
        op        = et.overall_percentage()
        vt        = et.total_violation_time
        dur       = time.time() - et.start_time
        win_alert = et.window_alert_triggered
        ovr_alert = et.overall_alert_triggered
    else:
        wp = op = vt = dur = 0.0
        win_alert = ovr_alert = False

    eye_cheat   = win_alert or ovr_alert
    eye_verdict = "*** CHEAT ***" if eye_cheat else "NOT CHEAT"
    eye_reasons = []
    if win_alert:
        eye_reasons.append(
            f"5-min window violation {wp:.1f}% exceeded threshold {WINDOW_THRESHOLD_PERCENT}%")
    if ovr_alert:
        eye_reasons.append(
            f"Overall violation {op:.1f}% exceeded threshold {OVERALL_THRESHOLD_PERCENT}%")
    if not eye_reasons:
        eye_reasons.append(
            f"Both thresholds within limits  "
            f"(window {wp:.1f}% < {WINDOW_THRESHOLD_PERCENT}%,  "
            f"overall {op:.1f}% < {OVERALL_THRESHOLD_PERCENT}%)")

    if face_mod:
        total_w    = face_mod.total_windows
        cheat_w    = face_mod.cheat_windows
        c_ratio    = face_mod.cheat_ratio()
        face_cheat = face_mod.final_result()
        window_log = face_mod.window_log
    else:
        total_w = cheat_w = 0
        c_ratio    = 0.0
        face_cheat = False
        window_log = []

    face_verdict = "*** CHEAT ***" if face_cheat else "NOT CHEAT"
    face_reasons = []
    if face_cheat:
        face_reasons.append(
            f"Cheat ratio {c_ratio:.2f} exceeded threshold 0.30  "
            f"({cheat_w} of {total_w} windows flagged)")
    else:
        face_reasons.append(
            f"Cheat ratio {c_ratio:.2f} within threshold 0.30  "
            f"({cheat_w} of {total_w} windows flagged)")

    if obj_mod:
        s3         = obj_mod.get_summary()
        c3_cheat   = s3["cheat"]
    else:
        s3       = {}
        c3_cheat = False

    c3_verdict  = "*** CHEAT ***" if c3_cheat else "NOT CHEAT"
    c3_reasons  = []
    if obj_mod and c3_cheat:
        if s3.get("phone_detected"):
            c3_reasons.append("Mobile phone detected during exam")
        if s3.get("notebook_detected"):
            c3_reasons.append("Suspicious text/notes detected during exam")
        if s3.get("unauth_detected"):
            c3_reasons.append("Unauthorized person(s) detected during exam")
    if not c3_reasons:
        c3_reasons.append("No prohibited objects or unauthorized persons detected")

    # Component 4
    if audio_mod:
        s4              = audio_mod.get_summary()
        c4_score        = s4["integrity_score"]
        c4_verdict_str  = s4["verdict"]
        c4_cheat        = s4["is_cheat"]
        c4_review       = (c4_verdict_str == "REVIEW")
        c4_total_segs   = s4["total_segments"]
        c4_suspicious   = s4["suspicious"]
        c4_wav          = s4["wav_path"]
        c4_report       = s4["report_path"]
    else:
        c4_score        = 100.0
        c4_verdict_str  = "PENDING"
        c4_cheat        = False
        c4_review       = False
        c4_total_segs   = 0
        c4_suspicious   = 0
        c4_wav          = "N/A"
        c4_report       = "N/A"

    c4_verdict = "*** CHEAT ***" if c4_cheat else ("REVIEW" if c4_review else "NOT CHEAT")

    answered     = sum(1 for v in answer_vars.values() if v.get() not in ("", "__none__"))
    final_cheat  = eye_cheat or face_cheat or c3_cheat or c4_cheat
    final_status = "*** CHEATING SUSPECTED ***" if final_cheat else "NO SUSPICIOUS BEHAVIOR"
    final_reason = []
    if eye_cheat:  final_reason.append("Eye/Head component flagged")
    if face_cheat: final_reason.append("Face Verification component flagged")
    if c3_cheat:   final_reason.append("Object/Person Detection component flagged")
    if c4_cheat:   final_reason.append("Audio Integrity component flagged")
    if not final_reason:
        final_reason.append("All components within acceptable limits")

    W = 65

    lines = [
        "=" * W,
        "        EDGE AI EXAM SYSTEM  –  PROCTORING REPORT",
        "=" * W,
        f"  Student         : {student}",
        f"  Report Generated: {timestamp}",
        f"  Questions        : {answered} / {len(questions)} answered",
        "",
        "─" * W,
        "  [ COMPONENT 1 — EYE GAZE & HEAD POSE MONITORING ]",
        "─" * W,
        f"  {'Metric':<26}{'Value':<15}{'Threshold'}",
        f"  {'5-Min Window Violation':<26}{wp:.2f}%{'':<9}{WINDOW_THRESHOLD_PERCENT}%",
        f"  {'Overall Violation':<26}{op:.2f}%{'':<9}{OVERALL_THRESHOLD_PERCENT}%",
        f"  {'Total Violation Time':<26}{vt:.1f} s",
        f"  {'Exam Duration':<26}{dur:.1f} s",
        f"  {'Window Alert Triggered':<26}{'YES' if win_alert else 'NO'}",
        f"  {'Overall Alert Triggered':<26}{'YES' if ovr_alert else 'NO'}",
        "",
        f"  COMPONENT 1 VERDICT : {eye_verdict}",
        f"  Reason: {eye_reasons[0]}",
    ]
    for r in eye_reasons[1:]:
        lines.append(f"          {r}")

    lines += [
        "",
        "─" * W,
        "  [ COMPONENT 2 — FACE VERIFICATION  (10-Second Window Voting) ]",
        "─" * W,
        f"  Window Size            : {FACE_WINDOW_SECONDS} seconds",
        f"  Window Violation Thresh: >{FACE_WINDOW_THRESHOLD*100:.0f}% of frames mismatched",
        f"  Final Cheat Threshold  : cheat_ratio > 0.30",
        f"  Match Threshold        : {MATCH_THRESHOLD} (strict)",
        f"  Total Windows Checked  : {total_w}",
        f"  Windows Flagged (CHEAT): {cheat_w}",
        f"  Cheat Ratio            : {c_ratio:.3f}",
        "",
    ]

    if window_log:
        lines.append("  Window-by-Window Log:")
        lines.append("  " + "-" * 44)
        lines.append("  {:>8}  {:>10}  {:>10}".format("Window", "Viol Ratio", "Verdict"))
        lines.append("  " + "-" * 44)
        for wnum, ratio, verdict in window_log:
            flag = "  <-- FLAGGED" if verdict == "CHEAT" else ""
            lines.append(f"  {wnum:>8}  {ratio:>10.3f}  {verdict:>10}{flag}")
        lines.append("  " + "-" * 44)
        lines.append("")

    lines += [
        f"  COMPONENT 2 VERDICT : {face_verdict}",
        f"  Reason: {face_reasons[0]}",
    ]

    lines += [
        "",
        "─" * W,
        "  [ COMPONENT 3 — OBJECT & UNAUTHORIZED PERSON DETECTION ]",
        "─" * W,
    ]

    if obj_mod and s3:
        lines += [
            f"  COCO Model             : {COCO_MODEL_PATH}",
            f"  Text Model             : {TEXT_MODEL_PATH}",
            f"  Face Model             : {FACE_MODEL_PATH}",   # ← NEW
            f"  Multi-Person Threshold : {TEMPORAL_THRESHOLD}s continuous",
            f"  Phone Conf Threshold   : {CONF_PHONE}",
            f"  Text Conf Threshold    : {CONF_TEXT}",
            f"  Face Conf Threshold    : {CONF_PERSON}",       # ← NEW
            f"  Face Tracker Buffer    : {MAX_MISSING_FRAMES} frames",  # ← NEW
            "",
            f"  Phone Detected         : {'YES' if s3.get('phone_detected') else 'NO'}",
            f"  Notes/Text Detected    : {'YES' if s3.get('notebook_detected') else 'NO'}",
            f"  Unauth Person Detected : {'YES' if s3.get('unauth_detected') else 'NO'}",
            f"  Objects Seen           : {', '.join(s3.get('detected_objects', [])) or 'none'}",
            f"  Total Violation Events : {s3.get('total_violations', 0)}",
            f"  Log File (CSV)         : {C3_CSV_LOG}",
            f"  Log File (JSON)        : {C3_JSON_LOG}",
        ]
        vlog = s3.get("violation_log", [])
        if vlog:
            lines += [
                "",
                "  Violation Event Log:",
                "  " + "-" * 50,
                "  {:>20}  {}".format("Timestamp", "Violation Type"),
                "  " + "-" * 50,
            ]
            for ev in vlog:
                lines.append(f"  {ev['time']:>20}  {ev['type']}")
            lines.append("  " + "-" * 50)
            lines.append("")
    else:
        lines.append("  Component 3 was not initialised or models not available.")
        lines.append("")

    lines += [
        f"  COMPONENT 3 VERDICT : {c3_verdict}",
        f"  Reason: {c3_reasons[0]}",
    ]
    for r in c3_reasons[1:]:
        lines.append(f"          {r}")

    # ── Component 4 section ───────────────────────────────────────────────
    lines += [
        "",
        "─" * W,
        "  [ COMPONENT 4 — AUDIO INTEGRITY & VOICE VERIFICATION ]",
        "─" * W,
        f"  ECAPA Model            : {C4_ECAPA_MODEL_PATH}",
        f"  YAMNet Classifier      : {C4_YAMNET_CLASSIFIER}",
        f"  Segment Length         : {C4_SEGMENT_SECONDS}s",
        f"  Speaker Threshold      : cosine dist < {C4_SPEAKER_THRESHOLD}",
        f"  Pass Threshold         : integrity >= {C4_INTEGRITY_PASS}%",
        f"  Review Threshold       : integrity {C4_INTEGRITY_REVIEW}–{C4_INTEGRITY_PASS-1}%",
        f"  Fail Threshold         : integrity < {C4_INTEGRITY_REVIEW}%",
        "",
        f"  Total Segments Analysed: {c4_total_segs}",
        f"  Suspicious Segments    : {c4_suspicious}",
        f"  Integrity Score        : {c4_score:.1f}%",
        f"  WAV Recording          : {c4_wav}",
        f"  Segment Report         : {c4_report}",
        "",
        f"  COMPONENT 4 VERDICT : {c4_verdict}",
        f"  Reason: integrity score {c4_score:.1f}%  "
        f"({'≥'+str(C4_INTEGRITY_PASS)+'% → clean' if not c4_cheat and not c4_review else str(C4_INTEGRITY_REVIEW)+'–'+str(C4_INTEGRITY_PASS-1)+'% → review' if c4_review else '<'+str(C4_INTEGRITY_REVIEW)+'% → cheat'})",
    ]

    lines += [
        "",
        "=" * W,
        "  COMBINED FINAL RESULT  (OR-gate: any component CHEAT → CHEAT)",
        "=" * W,
        f"  Component 1 – Eye & Head        : {eye_verdict}",
        f"  Component 2 – Face Verification : {face_verdict}",
        f"  Component 3 – Object / Person   : {c3_verdict}",
        f"  Component 4 – Audio Integrity   : {c4_verdict}",
        "  " + "─" * (W - 2),
        f"  FINAL VERDICT                   : {final_status}",
        f"  Reason                          : {' | '.join(final_reason)}",
        "=" * W,
        "",
    ]

    vid1 = getattr(app, "video_path",     None) or "N/A"
    vid2 = getattr(app, "video_path_obj", None) or "N/A"
    lines += [
        "",
        "─" * W,
        "  [ RECORDED VIDEOS & AUDIO ]",
        "─" * W,
        f"  Video 1 (Eye/Head + Face Verif) : {vid1}",
        f"  Video 2 (Object / Person Detect): {vid2}",
        f"  Audio   (Voice Integrity)       : {c4_wav}",
        "─" * W,
        "",
    ]

    report_str  = "\n".join(lines)
    report_path = os.path.join(
        RESULT_DIR, f"{student}_{int(time.time())}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_str)

    print(f"[Report] Saved to {report_path}")
    return report_str, report_path


# =========================================================
# LOAD QUESTIONS
# =========================================================
def load_questions(filepath="questions.docx"):
    try:
        doc = Document(filepath)
        questions, current = [], None
        for p in doc.paragraphs:
            text = p.text.strip()
            if not text:
                continue
            if text.startswith('Q') and len(text) > 1 and text[1].isdigit():
                if current and current["options"]:
                    questions.append(current)
                qt = text
                if '. ' in text[:6]:
                    qt = text.split('. ', 1)[1] if len(text.split('. ', 1)) > 1 else text
                elif ' ' in text[:5]:
                    parts = text.split(' ', 1)
                    qt    = parts[1] if len(parts) > 1 else text
                current = {"question": qt.strip(), "options": []}
            elif current and len(text) >= 2:
                fc, sc = text[0].upper(), text[1] if len(text) > 1 else ''
                if fc in ['A', 'B', 'C', 'D'] and sc == ')':
                    current["options"].append(f"{fc}) {text[2:].strip()}")
                elif current["options"]:
                    current["options"][-1] += " " + text
                else:
                    current["question"] += " " + text
        if current and current["options"]:
            questions.append(current)
        return questions
    except FileNotFoundError:
        return []
    except Exception:
        return []


# =========================================================
# STYLED HELPERS
# =========================================================
def styled_button(parent, text, command, bg=ACCENT2, fg=TEXT, width=18, **kw):
    btn = tk.Button(
        parent, text=text, command=command,
        bg=bg, fg=fg, activebackground=ACCENT, activeforeground=BG,
        font=FONT_H3, relief="flat", bd=0,
        padx=18, pady=10, cursor="hand2", width=width, **kw
    )
    btn.bind("<Enter>", lambda e: btn.config(bg=ACCENT, fg=BG))
    btn.bind("<Leave>", lambda e: btn.config(bg=bg,     fg=fg))
    return btn

def card_frame(parent, **kw):
    return tk.Frame(parent, bg=CARD, bd=0,
                    highlightbackground=BORDER, highlightthickness=1, **kw)

def section_label(parent, text, size=18):
    return tk.Label(parent, text=text, bg=CARD, fg=ACCENT,
                    font=("Courier New", size, "bold"))

def body_label(parent, text, fg=TEXT, bg=CARD, **kw):
    return tk.Label(parent, text=text, bg=bg, fg=fg, font=FONT_BODY, **kw)


# =========================================================
# MAIN APP
# =========================================================
class EdgeExamApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("EdgeAI Exam System")
        self.root.attributes("-fullscreen", True)
        self.root.configure(bg=BG)

        self.cap              = cv2.VideoCapture(0)
        self.current_student  = None
        self.exam_running     = False
        self.final_cheated    = False
        self.eye              = None
        self.face             = None
        self.obj              = None
        self.audio            = None   # Component 4
        self.writer           = None
        self.video_path       = None
        self.writer_obj       = None
        self.video_path_obj   = None
        self.report_path      = None
        self.report_str       = None
        self.latest_frame     = None

        container = tk.Frame(self.root, bg=BG)
        container.pack(fill="both", expand=True)
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        self.frames = {}
        for F in (LoginPage, HomePage, InstructionPage, ExamPage, ResultPage):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(LoginPage)

    def show_frame(self, page):
        self.frames[page].tkraise()
        if hasattr(self.frames[page], "on_show"):
            self.frames[page].on_show()

    def run(self):
        self.root.mainloop()


# =========================================================
# LOGIN PAGE  — STRICT MULTI-FRAME FACE VERIFICATION
# =========================================================
class LoginPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app       = app
        self._liveness = LivenessDetector()
        self._build()

    def _build(self):
        left = tk.Frame(self, bg=BG2, width=380)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        tk.Label(left, text="⬡", bg=BG2, fg=ACCENT,
                 font=("Courier New", 64)).pack(pady=(80, 10))
        tk.Label(left, text="EDGE AI", bg=BG2, fg=ACCENT,
                 font=("Courier New", 30, "bold")).pack()
        tk.Label(left, text="E X A M   S Y S T E M", bg=BG2, fg=TEXT_DIM,
                 font=("Courier New", 11, "bold")).pack(pady=(2, 40))

        for item in ["🔒  Biometric Login", "👁  Eye & Head Tracking",
                     "🧠  Face Verification", "📹  Session Recording",
                     "📵  Object Detection", "🎤  Audio Integrity"]:
            tk.Label(left, text=item, bg=BG2, fg=TEXT_DIM,
                     font=FONT_BODY, anchor="w").pack(anchor="w", padx=40, pady=4)
        tk.Label(left, text="✋  Blink Liveness Check", bg=BG2, fg=ACCENT,
                 font=FONT_BODY, anchor="w").pack(anchor="w", padx=40, pady=4)

        tk.Label(left,
                 text=f"🔐  Threshold: {MATCH_THRESHOLD} (strict)",
                 bg=BG2, fg=WARNING, font=FONT_MONO, anchor="w").pack(
                     anchor="w", padx=40, pady=4)

        right  = tk.Frame(self, bg=BG)
        right.pack(side="left", fill="both", expand=True)
        center = tk.Frame(right, bg=BG)
        center.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(center, text="Welcome Back", bg=BG, fg=TEXT,
                 font=FONT_H1).pack(pady=(0, 6))
        tk.Label(center, text="Authenticate to begin your exam session",
                 bg=BG, fg=TEXT_DIM, font=FONT_BODY).pack(pady=(0, 6))
        tk.Label(center,
                 text=f"Login: name entry → {LOGIN_VERIFY_FRAMES}-frame face scan "
                      f"(need {LOGIN_VERIFY_REQUIRED}/{LOGIN_VERIFY_FRAMES}) "
                      f"→ {REQUIRED_BLINKS} blinks",
                 bg=BG, fg=TEXT_DIM, font=FONT_MONO).pack(pady=(0, 30))

        btn_frame = tk.Frame(center, bg=BG)
        btn_frame.pack()
        styled_button(btn_frame, "REGISTER", self.register, bg=BG3,    width=16).pack(side="left", padx=8)
        styled_button(btn_frame, "LOGIN",    self.login,    bg=ACCENT2, width=16).pack(side="left", padx=8)

        self.status_lbl = tk.Label(center, text="", bg=BG, fg=DANGER, font=FONT_MONO)
        self.status_lbl.pack(pady=16)

        self.cam_lbl = tk.Label(center, bg=BG, bd=2,
                                highlightbackground=BORDER, highlightthickness=1)
        self.cam_lbl.pack(pady=8)

        self._cam_active = True
        self._update_cam()

    def _update_cam(self):
        if not self._cam_active:
            return
        ret, frame = self.app.cap.read()
        if ret:
            frame = cv2.cvtColor(cv2.resize(frame, (240, 180)), cv2.COLOR_BGR2RGB)
            img   = ImageTk.PhotoImage(Image.fromarray(frame))
            self.cam_lbl.configure(image=img)
            self.cam_lbl.image = img
        self.cam_lbl.after(60, self._update_cam)

    def register(self):
        name = simpledialog.askstring("Register", "Enter your full name:")
        if not name or not name.strip():
            return
        name = name.strip()
        self.status_lbl.config(
            text=f"📸  Capturing {REGISTER_FRAMES} biometric frames for '{name}'...",
            fg=WARNING)
        self.app.root.update()
        embeddings = []
        attempt    = 0
        max_attempts = REGISTER_FRAMES * 4
        while len(embeddings) < REGISTER_FRAMES and attempt < max_attempts:
            attempt += 1
            ret, frame = self.app.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            enc = face_recognition.face_encodings(frame)
            if enc:
                embeddings.append(enc[0])
                self.status_lbl.config(
                    text=f"📸  Captured {len(embeddings)}/{REGISTER_FRAMES} frames...",
                    fg=WARNING)
                self.app.root.update()
            time.sleep(0.2)
        if len(embeddings) < 5:
            self.status_lbl.config(
                text="✗  Not enough face data captured. Ensure good lighting & look at camera.",
                fg=DANGER)
            return
        mean_emb = np.mean(embeddings, axis=0)
        data = {
            "embeddings":      [e.tolist() for e in embeddings],
            "mean_embedding":  mean_emb,
            "name":            name,
            "registered_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "frame_count":     len(embeddings),
            "match_threshold": MATCH_THRESHOLD,
        }
        save_path = os.path.join(DB_DIR, f"{name}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        self.status_lbl.config(
            text=f"✓  Registered: {name}  ({len(embeddings)} frames captured)",
            fg=SUCCESS)
        print(f"[Register] Saved {len(embeddings)} embeddings for '{name}' → {save_path}")

    def login(self):
        claimed_name = simpledialog.askstring(
            "Identity Check", "Enter your registered full name:")
        if not claimed_name or not claimed_name.strip():
            self.status_lbl.config(text="✗  Name entry cancelled.", fg=DANGER)
            return
        claimed_name = claimed_name.strip()
        student_file = os.path.join(DB_DIR, f"{claimed_name}.pkl")
        if not os.path.exists(student_file):
            self.status_lbl.config(
                text=f"✗  '{claimed_name}' is not registered. Please register first.",
                fg=DANGER)
            return
        with open(student_file, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            known_embeddings = [np.array(e) for e in data.get("embeddings", [])]
            mean_embedding   = np.array(data.get("mean_embedding",
                                                  known_embeddings[0]))
        else:
            mean_embedding   = data
            known_embeddings = [data]
        self.status_lbl.config(
            text=f"🔍  Scanning face for '{claimed_name}'  "
                 f"({LOGIN_VERIFY_FRAMES} frames, need {LOGIN_VERIFY_REQUIRED} matches)...",
            fg=WARNING)
        self.app.root.update()
        self._cam_active = False
        time.sleep(0.1)
        matches = 0
        distances = []
        for i in range(LOGIN_VERIFY_FRAMES):
            ret, frame = self.app.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            preview = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (240, 180))
            cv2.putText(preview,
                        f"Scanning {i+1}/{LOGIN_VERIFY_FRAMES}",
                        (5, 20), FONT_CV, 0.55, (0, 200, 255), 1)
            img = ImageTk.PhotoImage(Image.fromarray(preview))
            self.cam_lbl.configure(image=img)
            self.cam_lbl.image = img
            self.app.root.update()
            encodings = face_recognition.face_encodings(frame)
            if not encodings:
                distances.append(999.0)
                time.sleep(0.15)
                continue
            dist_mean = float(face_recognition.face_distance(
                [mean_embedding], encodings[0])[0])
            dists_all = face_recognition.face_distance(
                known_embeddings, encodings[0])
            best_dist = float(min(dists_all))
            effective_dist = max(dist_mean, best_dist)
            distances.append(round(effective_dist, 4))
            if effective_dist < MATCH_THRESHOLD:
                matches += 1
            print(f"[Login] Frame {i+1}: mean_dist={dist_mean:.4f}  "
                  f"best_dist={best_dist:.4f}  "
                  f"→ {'MATCH' if effective_dist < MATCH_THRESHOLD else 'NO MATCH'}")
            time.sleep(0.15)
        self._cam_active = True
        self._update_cam()
        print(f"[Login] Total matches: {matches}/{LOGIN_VERIFY_FRAMES}  "
              f"(need {LOGIN_VERIFY_REQUIRED})  distances={distances}")
        if matches < LOGIN_VERIFY_REQUIRED:
            self.status_lbl.config(
                text=f"✗  Face does NOT match '{claimed_name}'  "
                     f"({matches}/{LOGIN_VERIFY_FRAMES} frames passed, "
                     f"need {LOGIN_VERIFY_REQUIRED}).  Access denied.",
                fg=DANGER)
            return
        self.status_lbl.config(
            text=f"✓  Face verified as '{claimed_name}'  "
                 f"({matches}/{LOGIN_VERIFY_FRAMES})  —  Now prove you're live!",
            fg=WARNING)
        self.app.root.update()
        time.sleep(0.4)
        self.status_lbl.config(
            text=f"👁  Please blink {REQUIRED_BLINKS} times within {LIVENESS_TIMEOUT}s ...",
            fg=WARNING)
        self.app.root.update()
        self._cam_active = False
        live = self._liveness.run(
            cap=self.app.cap,
            status_label=self.status_lbl,
            cam_label=self.cam_lbl,
            root=self.app.root,
        )
        self._cam_active = True
        self._update_cam()
        if not live:
            self.status_lbl.config(
                text="✗  Liveness check failed. Possible photo/video spoof.", fg=DANGER)
            return
        self.app.current_student = claimed_name
        self.status_lbl.config(
            text=f"✓  Welcome, {self.app.current_student}!  Login successful.",
            fg=SUCCESS)
        self.app.root.after(800, lambda: self.app.show_frame(HomePage))


# =========================================================
# HOME PAGE
# =========================================================
class HomePage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app = app
        self._build()

    def on_show(self):
        self.welcome_lbl.config(text=f"Hello, {self.app.current_student or 'Student'}")

    def _build(self):
        bar = tk.Frame(self, bg=BG2, height=60)
        bar.pack(fill="x")
        tk.Label(bar, text="⬡  EDGE AI EXAM", bg=BG2, fg=ACCENT,
                 font=("Courier New", 16, "bold")).pack(side="left", padx=24, pady=14)
        self.welcome_lbl = tk.Label(bar, text="", bg=BG2, fg=TEXT_DIM, font=FONT_BODY)
        self.welcome_lbl.pack(side="right", padx=24)
        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=60, pady=40)
        tk.Label(main, text="Available Exams", bg=BG, fg=TEXT,
                 font=FONT_H1).pack(anchor="w", pady=(0, 24))
        card  = card_frame(main)
        card.pack(fill="x", pady=8)
        inner = tk.Frame(card, bg=CARD, padx=24, pady=20)
        inner.pack(fill="x")
        tk.Label(inner, text="📄  Sample MCQ Examination", bg=CARD, fg=TEXT,
                 font=FONT_H2).pack(anchor="w")
        tk.Label(inner, text="Multiple choice · Timed · Fully proctored",
                 bg=CARD, fg=TEXT_DIM, font=FONT_BODY).pack(anchor="w", pady=4)
        styled_button(inner, "BEGIN →",
                      lambda: self.app.show_frame(InstructionPage),
                      bg=ACCENT2, width=12).pack(anchor="e", pady=(12, 0))


# =========================================================
# INSTRUCTION PAGE  — now includes voice calibration
# =========================================================
class InstructionPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app = app
        self._calibrated = False
        self._build()

    def on_show(self):
        # Reset calibration state each time the page is shown
        self._calibrated = False
        self._update_calibration_ui()

    def _build(self):
        bar = tk.Frame(self, bg=BG2, height=60)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        tk.Label(bar, text="⬡  EDGE AI EXAM", bg=BG2, fg=ACCENT,
                 font=("Courier New", 16, "bold")).pack(side="left", padx=24, pady=14)

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        canvas = tk.Canvas(body, bg=BG, highlightthickness=0)
        vsb    = tk.Scrollbar(body, orient="vertical", command=canvas.yview)
        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=vsb.set)

        scroll_frame = tk.Frame(canvas, bg=BG)
        win_id = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(win_id, width=canvas.winfo_width()))
        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        def _mwheel(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _mwheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        center = tk.Frame(scroll_frame, bg=BG)
        center.pack(fill="x", padx=80, pady=40)

        tk.Label(center, text="Before You Begin", bg=BG, fg=TEXT,
                 font=FONT_H1).pack(anchor="w", pady=(0, 4))
        tk.Label(center, text="Read all instructions carefully",
                 bg=BG, fg=TEXT_DIM, font=FONT_BODY).pack(anchor="w", pady=(0, 28))

        rules = [
            ("👁  Eye Gaze",         "Keep your eyes on the screen at all times. Looking away triggers a violation."),
            ("🧠  Head Pose",        "Keep your head facing forward. Turning your head will be flagged."),
            ("👤  Single Person",    "Only one person is allowed in frame. Additional persons trigger a violation after 3 seconds."),
            ("📵  No Phones",        "Mobile phones are strictly prohibited. Detection triggers an immediate violation."),
            ("📓  No Notes",         "Written notes or printed cheat sheets are not allowed. The system will detect them."),
            ("📹  Recording",        "Your session is recorded with full annotation for review."),
            ("🔔  Alert Thresholds", "40% violations in any 5-min window OR 20% overall triggers a suspicious alert."),
            ("⏱  Exam Timer",       f"The exam runs for {EXAM_DURATION_SECONDS // 60} minutes. A countdown timer is shown. Exam auto-submits at zero."),
            ("🎤  Audio Monitoring", f"Your audio is monitored throughout the exam. Whispering or external audio sources will be flagged."),
        ]
        for icon_title, desc in rules:
            row   = card_frame(center)
            row.pack(fill="x", pady=6)
            inner = tk.Frame(row, bg=CARD, padx=20, pady=14)
            inner.pack(fill="x")
            tk.Label(inner, text=icon_title, bg=CARD, fg=ACCENT, font=FONT_H3).pack(anchor="w")
            tk.Label(inner, text=desc, bg=CARD, fg=TEXT_DIM,
                     font=FONT_BODY, wraplength=800, justify="left").pack(anchor="w", pady=(2, 0))

        # ── Voice calibration card ────────────────────────────────────────
        cal_card  = card_frame(center)
        cal_card.pack(fill="x", pady=(18, 6))
        cal_inner = tk.Frame(cal_card, bg=CARD, padx=20, pady=18)
        cal_inner.pack(fill="x")

        tk.Label(cal_inner, text="🎤  Voice Calibration (Required)",
                 bg=CARD, fg=ACCENT, font=FONT_H3).pack(anchor="w")
        tk.Label(cal_inner,
                 text=(f"Before starting the exam you must complete a "
                       f"{C4_CALIBRATION_SECONDS}-second voice calibration. "
                       f"Speak naturally (introduce yourself, read a sentence) so the "
                       f"system can create your voice fingerprint for this session."),
                 bg=CARD, fg=TEXT_DIM, font=FONT_BODY,
                 wraplength=800, justify="left").pack(anchor="w", pady=(4, 12))

        self._cal_status_lbl = tk.Label(
            cal_inner,
            text="⚠  Calibration not yet done.",
            bg=CARD, fg=WARNING, font=FONT_MONO)
        self._cal_status_lbl.pack(anchor="w", pady=(0, 10))

        self._cal_btn = styled_button(
            cal_inner,
            f"▶  START {C4_CALIBRATION_SECONDS}s CALIBRATION",
            self._run_calibration,
            bg=BG3, width=30)
        self._cal_btn.pack(anchor="w")

        # ── Start exam button (disabled until calibrated) ─────────────────
        self._start_btn = styled_button(
            center, "START EXAM →", self._start,
            bg=TEXT_DIM, fg=BG, width=20)
        self._start_btn.pack(pady=28)
        # Disable hover effect until enabled
        self._start_btn.unbind("<Enter>")
        self._start_btn.unbind("<Leave>")

    def _update_calibration_ui(self):
        if self._calibrated:
            self._cal_status_lbl.config(
                text="✓  Voice calibration complete. Ready to start exam.",
                fg=SUCCESS)
            self._cal_btn.config(
                text="✓  CALIBRATED – Click again to redo",
                bg=BG3)
            # Enable start button
            self._start_btn.config(bg=SUCCESS, fg=BG)
            self._start_btn.bind("<Enter>", lambda e: self._start_btn.config(bg=ACCENT, fg=BG))
            self._start_btn.bind("<Leave>", lambda e: self._start_btn.config(bg=SUCCESS, fg=BG))
        else:
            self._cal_status_lbl.config(
                text="⚠  Calibration not yet done.", fg=WARNING)
            self._start_btn.config(bg=TEXT_DIM, fg=BG)
            self._start_btn.unbind("<Enter>")
            self._start_btn.unbind("<Leave>")

    def _run_calibration(self):
        """Run voice calibration in a thread, update UI on completion."""
        if not self.app.current_student:
            messagebox.showerror("Error", "No student logged in.")
            return

        self._cal_btn.config(state="disabled")
        self._cal_status_lbl.config(
            text=f"🎤  Recording… speak naturally for {C4_CALIBRATION_SECONDS}s",
            fg=WARNING)
        self.app.root.update()

        # Create the AudioIntegrityModule now (will be reused during exam)
        self.app.audio = AudioIntegrityModule(self.app.current_student)

        done_evt = threading.Event()
        success  = [False]

        def _status(msg):
            # Update label from the calibration thread (safe via after)
            self._cal_status_lbl.after(
                0, lambda m=msg: self._cal_status_lbl.config(text=m, fg=WARNING))

        def _worker():
            success[0] = self.app.audio.calibrate(
                duration=C4_CALIBRATION_SECONDS,
                status_cb=_status)
            done_evt.set()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        # Poll the UI every 200 ms until done
        def _poll():
            if done_evt.is_set():
                self._cal_btn.config(state="normal")
                if success[0]:
                    self._calibrated = True
                else:
                    self._cal_status_lbl.config(
                        text="✗  Calibration failed – please try again.", fg=DANGER)
                self._update_calibration_ui()
            else:
                self.after(200, _poll)

        self.after(200, _poll)

    def _start(self):
        if not self._calibrated:
            messagebox.showwarning(
                "Voice Calibration Required",
                "Please complete the voice calibration before starting the exam.")
            return
        self.app.frames[ExamPage].start_exam()
        self.app.show_frame(ExamPage)


# =========================================================
# EXAM PAGE
# =========================================================
class ExamPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app         = app
        self.questions   = load_questions("questions.docx")
        self.answer_vars = {}
        self._build()

    def _build(self):
        self.bar = tk.Frame(self, bg=BG2, height=64)
        self.bar.pack(fill="x")
        self.bar.pack_propagate(False)

        left_bar = tk.Frame(self.bar, bg=BG2)
        left_bar.pack(side="left", padx=24, pady=8)
        tk.Label(left_bar, text="⬡  EXAM IN PROGRESS", bg=BG2, fg=ACCENT,
                 font=("Courier New", 14, "bold")).pack(anchor="w")
        self.student_lbl = tk.Label(left_bar, text="", bg=BG2, fg=TEXT_DIM,
                                    font=FONT_MONO)
        self.student_lbl.pack(anchor="w")

        right_bar = tk.Frame(self.bar, bg=BG2)
        right_bar.pack(side="right", padx=24, pady=4)
        tk.Label(right_bar, text="TIME REMAINING", bg=BG2, fg=TEXT_DIM,
                 font=FONT_MONO).pack(anchor="e")
        self.timer_frame = tk.Frame(right_bar, bg=BG3,
                                    highlightbackground=ACCENT, highlightthickness=1)
        self.timer_frame.pack(anchor="e")
        self.timer_lbl = tk.Label(self.timer_frame, text="--:--",
                                  bg=BG3, fg=ACCENT,
                                  font=("Courier New", 26, "bold"),
                                  padx=12, pady=2)
        self.timer_lbl.pack()

        center_bar = tk.Frame(self.bar, bg=BG2)
        center_bar.pack(side="left", padx=30, pady=4)
        tk.Label(center_bar, text="ELAPSED", bg=BG2, fg=TEXT_DIM,
                 font=FONT_MONO).pack(anchor="w")
        self.elapsed_lbl = tk.Label(center_bar, text="00:00",
                                    bg=BG2, fg=TEXT_DIM,
                                    font=("Courier New", 16, "bold"))
        self.elapsed_lbl.pack(anchor="w")

        body   = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        canvas = tk.Canvas(body, bg=BG, highlightthickness=0)
        vsb    = tk.Scrollbar(body, orient="vertical", command=canvas.yview)
        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=vsb.set)

        self.scroll_frame = tk.Frame(canvas, bg=BG)
        win_id = canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")

        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(win_id, width=canvas.winfo_width()))
        self.scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        def _mwheel(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _mwheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        self._populate_questions()

    def _populate_questions(self):
        for w in self.scroll_frame.winfo_children():
            w.destroy()
        self.answer_vars = {}

        pad = tk.Frame(self.scroll_frame, bg=BG)
        pad.pack(fill="x", padx=40, pady=20)

        tk.Label(pad, text="Examination Questions", bg=BG, fg=TEXT,
                 font=FONT_H1).pack(anchor="w", pady=(0, 4))
        tk.Label(pad, text="Select one answer per question. All questions count equally.",
                 bg=BG, fg=TEXT_DIM, font=FONT_BODY).pack(anchor="w", pady=(0, 16))

        for idx, q in enumerate(self.questions, 1):
            card  = card_frame(self.scroll_frame)
            card.pack(fill="x", padx=40, pady=8)
            inner = tk.Frame(card, bg=CARD, padx=24, pady=18)
            inner.pack(fill="x")

            hdr = tk.Frame(inner, bg=CARD)
            hdr.pack(fill="x")
            tk.Label(hdr, text=f"Q{idx}", bg=ACCENT2, fg=TEXT,
                     font=FONT_MONO, padx=8, pady=4).pack(side="left")
            tk.Label(hdr, text=q["question"], bg=CARD, fg=TEXT,
                     font=("Courier New", 12, "bold"),
                     wraplength=760, justify="left").pack(side="left", padx=12)

            tk.Frame(inner, bg=BORDER, height=1).pack(fill="x", pady=10)

            var = tk.StringVar(value="__none__")
            self.answer_vars[idx] = var

            for opt in q["options"]:
                rb_frame = tk.Frame(inner, bg=CARD, cursor="hand2")
                rb_frame.pack(fill="x", pady=3)
                rb_frame.bind("<Enter>", lambda e, f=rb_frame: f.config(bg=BG3))
                rb_frame.bind("<Leave>", lambda e, f=rb_frame: f.config(bg=CARD))
                tk.Radiobutton(
                    rb_frame, text=opt, variable=var, value=opt,
                    bg=CARD, fg=TEXT, activebackground=BG3, activeforeground=ACCENT,
                    selectcolor=BG3, font=FONT_BODY,
                    indicatoron=True, anchor="w", cursor="hand2",
                ).pack(fill="x", padx=12, pady=4)

        btn_row = tk.Frame(self.scroll_frame, bg=BG)
        btn_row.pack(pady=30)
        styled_button(btn_row, "SUBMIT EXAM", self.submit,
                      bg=SUCCESS, fg=BG, width=22).pack()

    def start_exam(self):
        self.app.eye           = EyeHeadModule()
        self.app.face          = FaceVerificationModule(self.app.current_student)
        self.app.obj           = ObjectDetectionModule(self.app.current_student)
        # app.audio is already created & calibrated in InstructionPage
        self.app.exam_running  = True
        self.app.final_cheated = False

        self.student_lbl.config(text=f"Student: {self.app.current_student or ''}")

        ts = int(time.time())
        w  = int(self.app.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h  = int(self.app.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        self.app.video_path = os.path.join(
            RESULT_DIR,
            f"{self.app.current_student}_{ts}_eyehead_face.mp4"
        )
        self.app.writer = cv2.VideoWriter(
            self.app.video_path, fourcc, VIDEO_FPS, (w, h)
        )
        self.app.video_path_obj = os.path.join(
            RESULT_DIR,
            f"{self.app.current_student}_{ts}_object_person.mp4"
        )
        self.app.writer_obj = cv2.VideoWriter(
            self.app.video_path_obj, fourcc, VIDEO_FPS, (w, h)
        )

        # Start Component 4 audio recording
        if self.app.audio:
            self.app.audio.start_recording(
                max_duration_seconds=EXAM_DURATION_SECONDS + 60)

        self.exam_start_time = time.time()
        self._update_timer()
        threading.Thread(target=self._proctor_loop, daemon=True).start()

    def _update_timer(self):
        if not self.app.exam_running:
            return
        elapsed   = time.time() - self.exam_start_time
        remaining = max(0, EXAM_DURATION_SECONDS - elapsed)
        e_m, e_s = divmod(int(elapsed), 60)
        self.elapsed_lbl.config(text=f"{e_m:02d}:{e_s:02d}")
        r_m, r_s = divmod(int(remaining), 60)
        self.timer_lbl.config(text=f"{r_m:02d}:{r_s:02d}")
        if remaining > 600:
            fg_col = ACCENT
            bg_col = BG3
            border = ACCENT
        elif remaining > 120:
            fg_col = WARNING
            bg_col = BG3
            border = WARNING
        else:
            flash  = int(elapsed * 2) % 2
            fg_col = DANGER
            bg_col = "#2a0000" if flash else BG3
            border = DANGER
        self.timer_lbl.config(fg=fg_col, bg=bg_col)
        self.timer_frame.config(bg=bg_col, highlightbackground=border)
        if remaining <= 0:
            self.timer_lbl.config(text="00:00", fg=DANGER)
            messagebox.showinfo(
                "Time's Up!",
                "The exam time has expired. Your answers will be submitted automatically.")
            self.submit(force=True)
            return
        self.after(500, self._update_timer)

    def _proctor_loop(self):
        frame_count      = 0
        start_time       = time.time()
        last_frame_c12   = None
        last_frame_c3    = None

        while self.app.exam_running:
            ret, frame = self.app.cap.read()
            if not ret:
                break
            raw_c12 = frame.copy()
            raw_c3  = frame.copy()
            frame_c12, is_eye_violation = self.app.eye.process(raw_c12)
            self.app.face.process(frame_c12)
            frame_c3, is_obj_violation = self.app.obj.process(raw_c3)
            if is_eye_violation or is_obj_violation:
                self.app.final_cheated = True
            self.app.latest_frame = frame_c12.copy()
            last_frame_c12        = frame_c12
            last_frame_c3         = frame_c3
            elapsed      = time.time() - start_time
            target_count = int(elapsed * VIDEO_FPS)
            frames_to_write = max(1, target_count - frame_count)
            for _ in range(frames_to_write):
                self.app.writer.write(last_frame_c12)
                self.app.writer_obj.write(last_frame_c3)
                frame_count += 1

        if last_frame_c12 is not None:
            elapsed      = time.time() - start_time
            target_count = int(elapsed * VIDEO_FPS)
            while frame_count < target_count:
                self.app.writer.write(last_frame_c12)
                self.app.writer_obj.write(last_frame_c3)
                frame_count += 1

    def submit(self, force=False):
        if not self.app.exam_running:
            return

        if not force:
            unanswered = sum(
                1 for v in self.answer_vars.values() if v.get() in ("", "__none__"))
            if unanswered > 0:
                if not messagebox.askyesno(
                    "Unanswered Questions",
                    f"You have {unanswered} unanswered question(s).\n\nSubmit anyway?"
                ):
                    return

        self.app.exam_running = False
        time.sleep(0.8)

        if self.app.writer:
            self.app.writer.release()
        if self.app.writer_obj:
            self.app.writer_obj.release()

        if self.app.eye:
            self.app.eye.tracker.check_alerts()

        # ── Component 4: stop recording + analyse (blocking, show progress) ──
        if self.app.audio:
            # Show a small analysis window
            analysis_win = tk.Toplevel(self.app.root)
            analysis_win.title("Audio Analysis")
            analysis_win.configure(bg=BG)
            analysis_win.geometry("440x160")
            analysis_win.grab_set()

            tk.Label(analysis_win,
                     text="🔊  Analysing Audio Integrity…",
                     bg=BG, fg=ACCENT, font=FONT_H3).pack(pady=(24, 8))
            c4_status_lbl = tk.Label(
                analysis_win,
                text="Stopping recording…",
                bg=BG, fg=TEXT_DIM, font=FONT_MONO)
            c4_status_lbl.pack(pady=4)
            self.app.root.update()

            done_evt = threading.Event()

            def _status(msg):
                c4_status_lbl.after(
                    0, lambda m=msg: c4_status_lbl.config(text=m))

            def _analyse():
                self.app.audio.stop_and_analyse(status_cb=_status)
                done_evt.set()

            t = threading.Thread(target=_analyse, daemon=True)
            t.start()

            while not done_evt.is_set():
                self.app.root.update()
                time.sleep(0.05)

            analysis_win.destroy()

        report_str, report_path = generate_report(
            self.app, self.answer_vars, self.questions)
        self.app.report_path = report_path
        self.app.report_str  = report_str

        self.app.show_frame(ResultPage)
        self.app.frames[ResultPage].refresh()


# =========================================================
# RESULT PAGE
# =========================================================
class ResultPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app = app
        self._build()

    def _build(self):
        bar = tk.Frame(self, bg=BG2, height=60)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        tk.Label(bar, text="⬡  EXAM COMPLETE", bg=BG2, fg=ACCENT,
                 font=("Courier New", 16, "bold")).pack(side="left", padx=24, pady=14)

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        self._canvas = tk.Canvas(body, bg=BG, highlightthickness=0)
        vsb = tk.Scrollbar(body, orient="vertical", command=self._canvas.yview)
        self._canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self._canvas.configure(yscrollcommand=vsb.set)

        self._scroll_frame = tk.Frame(self._canvas, bg=BG)
        self._win_id = self._canvas.create_window(
            (0, 0), window=self._scroll_frame, anchor="nw")

        self._canvas.bind(
            "<Configure>",
            lambda e: self._canvas.itemconfig(self._win_id,
                                              width=self._canvas.winfo_width()))
        self._scroll_frame.bind(
            "<Configure>",
            lambda e: self._canvas.configure(
                scrollregion=self._canvas.bbox("all")))

        def _mwheel(e):
            self._canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        self._canvas.bind("<Enter>", lambda e: self._canvas.bind_all("<MouseWheel>", _mwheel))
        self._canvas.bind("<Leave>", lambda e: self._canvas.unbind_all("<MouseWheel>"))

        self.main = self._scroll_frame

    def refresh(self):
        for w in self.main.winfo_children():
            w.destroy()
        self._canvas.yview_moveto(0)

        eye_mod     = self.app.eye
        face_module = self.app.face
        obj_module  = self.app.obj
        audio_module = self.app.audio      # Component 4
        eye_tracker = eye_mod.tracker if eye_mod else None

        if eye_tracker:
            eye_tracker.check_alerts()
            wp        = eye_tracker.window_percentage()
            op        = eye_tracker.overall_percentage()
            vt        = eye_tracker.total_violation_time
            dur       = time.time() - eye_tracker.start_time
            win_alert = eye_tracker.window_alert_triggered
            ovr_alert = eye_tracker.overall_alert_triggered
        else:
            wp = op = vt = dur = 0.0
            win_alert = ovr_alert = False

        eye_cheat  = win_alert or ovr_alert
        face_cheat = face_module.final_result() if face_module else False

        c3_cheat = False
        s3       = {}
        if obj_module:
            s3       = obj_module.get_summary()
            c3_cheat = s3.get("cheat", False)

        # Component 4
        c4_cheat  = False
        c4_review = False
        c4_score  = 100.0
        c4_verdict_str = "PENDING"
        s4 = {}
        if audio_module:
            s4             = audio_module.get_summary()
            c4_score       = s4["integrity_score"]
            c4_verdict_str = s4["verdict"]
            c4_cheat       = s4["is_cheat"]
            c4_review      = (c4_verdict_str == "REVIEW")

        final_cheat  = eye_cheat or face_cheat or c3_cheat or c4_cheat
        headline_col = DANGER if final_cheat else SUCCESS

        # ── helpers ──────────────────────────────────────────────────────
        def stat_row(parent, label, value, col):
            r = tk.Frame(parent, bg=CARD)
            r.pack(fill="x", pady=2)
            tk.Label(r, text=label, bg=CARD, fg=TEXT_DIM,
                     font=FONT_MONO, width=22, anchor="w").pack(side="left")
            tk.Label(r, text=value, bg=CARD, fg=col,
                     font=("Courier New", 11, "bold")).pack(side="left")

        def verdict_badge(parent, is_cheat, is_review=False):
            if is_cheat:
                txt, col, bg_ = "CHEAT",   DANGER,   "#2a0a0a"
            elif is_review:
                txt, col, bg_ = "REVIEW",  WARNING,  "#2a1a00"
            else:
                txt, col, bg_ = "NOT CHEAT", SUCCESS, "#0a2a15"
            tk.Label(parent, text=f"  {txt}  ", bg=bg_, fg=col,
                     font=("Courier New", 13, "bold"),
                     padx=10, pady=6,
                     highlightbackground=col,
                     highlightthickness=1).pack(anchor="w", pady=(10, 0))

        # ── Headline ──────────────────────────────────────────────────────
        top = tk.Frame(self.main, bg=BG)
        top.pack(fill="x", padx=40, pady=(30, 0))

        headline_txt = ("⚠  SUSPICIOUS BEHAVIOR DETECTED"
                        if final_cheat else "✓  EXAM COMPLETED CLEANLY")
        tk.Label(top, text=headline_txt, bg=BG, fg=headline_col,
                 font=("Courier New", 22, "bold")).pack(anchor="w")
        tk.Label(top, text=f"Student : {self.app.current_student or '—'}",
                 bg=BG, fg=TEXT_DIM, font=FONT_BODY).pack(anchor="w", pady=(4, 0))

        # ── Row 1: Component 1 + Component 2 ─────────────────────────────
        row1 = tk.Frame(self.main, bg=BG)
        row1.pack(fill="x", padx=40, pady=(20, 0))

        c1 = card_frame(row1)
        c1.pack(side="left", fill="both", expand=True, padx=(0, 8))
        i1 = tk.Frame(c1, bg=CARD, padx=20, pady=16)
        i1.pack(fill="both", expand=True)

        tk.Label(i1, text="COMPONENT 1", bg=CARD, fg=TEXT_DIM, font=FONT_MONO).pack(anchor="w")
        tk.Label(i1, text="👁  Eye Gaze & Head Pose",
                 bg=CARD, fg=ACCENT, font=FONT_H3).pack(anchor="w", pady=(2, 10))

        if eye_tracker:
            stat_row(i1, "5-Min Window Viol.",
                     f"{wp:.2f}%  (limit {WINDOW_THRESHOLD_PERCENT}%)",
                     DANGER if win_alert else SUCCESS)
            stat_row(i1, "Overall Viol.",
                     f"{op:.2f}%  (limit {OVERALL_THRESHOLD_PERCENT}%)",
                     DANGER if ovr_alert else SUCCESS)
            stat_row(i1, "Total Viol. Time", f"{vt:.1f} s",  WARNING)
            stat_row(i1, "Exam Duration",    f"{dur:.1f} s", TEXT)
            stat_row(i1, "Window Alert",     "YES" if win_alert else "NO",
                     DANGER if win_alert else SUCCESS)
            stat_row(i1, "Overall Alert",    "YES" if ovr_alert else "NO",
                     DANGER if ovr_alert else SUCCESS)

        reason1 = (
            " | ".join(filter(None, [
                f"Window {wp:.1f}% > {WINDOW_THRESHOLD_PERCENT}%" if win_alert else "",
                f"Overall {op:.1f}% > {OVERALL_THRESHOLD_PERCENT}%" if ovr_alert else "",
            ])) or f"Window {wp:.1f}% and overall {op:.1f}% within limits"
        )
        tk.Label(i1, text=f"Reason: {reason1}", bg=CARD, fg=TEXT_DIM,
                 font=FONT_MONO, wraplength=340, justify="left").pack(anchor="w", pady=(8, 0))
        verdict_badge(i1, eye_cheat)

        c2 = card_frame(row1)
        c2.pack(side="left", fill="both", expand=True, padx=(8, 0))
        i2 = tk.Frame(c2, bg=CARD, padx=20, pady=16)
        i2.pack(fill="both", expand=True)

        tk.Label(i2, text="COMPONENT 2", bg=CARD, fg=TEXT_DIM, font=FONT_MONO).pack(anchor="w")
        tk.Label(i2, text="🔍  Biometric / Face Verification",
                 bg=CARD, fg=ACCENT, font=FONT_H3).pack(anchor="w", pady=(2, 10))

        if face_module:
            cr = face_module.cheat_ratio()
            stat_row(i2, "Total Windows",  str(face_module.total_windows), TEXT)
            stat_row(i2, "Cheat Windows",  str(face_module.cheat_windows),
                     DANGER if face_module.cheat_windows > 0 else SUCCESS)
            stat_row(i2, "Cheat Ratio",
                     f"{cr:.3f}  (limit 0.300)",
                     DANGER if face_cheat else SUCCESS)
            stat_row(i2, "Match Threshold", f"{MATCH_THRESHOLD} (strict)", WARNING)
            stat_row(i2, "Window Size",     f"{FACE_WINDOW_SECONDS}s", TEXT)
            face_reason = (
                f"Cheat ratio {cr:.3f} > 0.300  "
                f"({face_module.cheat_windows}/{face_module.total_windows} windows)"
                if face_cheat else
                f"Cheat ratio {cr:.3f} within limit 0.300"
            )
        else:
            face_reason = "Face module not initialised"

        tk.Label(i2, text=f"Reason: {face_reason}", bg=CARD, fg=TEXT_DIM,
                 font=FONT_MONO, wraplength=340, justify="left").pack(anchor="w", pady=(8, 0))
        verdict_badge(i2, face_cheat)

        # ── Row 2: Component 3 + Component 4 ─────────────────────────────
        row2 = tk.Frame(self.main, bg=BG)
        row2.pack(fill="x", padx=40, pady=(16, 0))

        c3 = card_frame(row2)
        c3.pack(side="left", fill="both", expand=True, padx=(0, 8))
        i3 = tk.Frame(c3, bg=CARD, padx=20, pady=16)
        i3.pack(fill="both", expand=True)

        tk.Label(i3, text="COMPONENT 3", bg=CARD, fg=TEXT_DIM, font=FONT_MONO).pack(anchor="w")
        tk.Label(i3, text="📵  Object & Person Detection",
                 bg=CARD, fg=ACCENT, font=FONT_H3).pack(anchor="w", pady=(2, 10))

        if obj_module and s3:
            phone_det  = s3.get("phone_detected",    False)
            notes_det  = s3.get("notebook_detected", False)
            unauth_det = s3.get("unauth_detected",   False)
            objs_seen  = s3.get("detected_objects",  [])
            total_viol = s3.get("total_violations",  0)

            stat_row(i3, "Phone Detected",
                     "YES – DETECTED" if phone_det else "NO",
                     DANGER if phone_det else SUCCESS)
            stat_row(i3, "Notes/Text",
                     "YES – DETECTED" if notes_det else "NO",
                     DANGER if notes_det else SUCCESS)
            stat_row(i3, "Unauth Person",
                     "YES – DETECTED" if unauth_det else "NO",
                     DANGER if unauth_det else SUCCESS)
            stat_row(i3, "Items Seen",
                     ", ".join(objs_seen) if objs_seen else "none",
                     DANGER if objs_seen else SUCCESS)
            stat_row(i3, "Total Events",
                     str(total_viol),
                     DANGER if total_viol > 0 else SUCCESS)
            stat_row(i3, "CSV Log", "Saved ✓", SUCCESS)

            c3_parts = []
            if phone_det:  c3_parts.append("Phone detected")
            if notes_det:  c3_parts.append("Notes/text detected")
            if unauth_det: c3_parts.append("Unauthorized person")
            c3_reason = " | ".join(c3_parts) if c3_parts else "No prohibited items detected"
        else:
            c3_reason = "Component 3 not initialised"

        tk.Label(i3, text=f"Reason: {c3_reason}", bg=CARD, fg=TEXT_DIM,
                 font=FONT_MONO, wraplength=340, justify="left").pack(anchor="w", pady=(8, 0))
        verdict_badge(i3, c3_cheat)

        # ── Component 4 card ──────────────────────────────────────────────
        c4 = card_frame(row2)
        c4.pack(side="left", fill="both", expand=True, padx=(8, 0))
        i4 = tk.Frame(c4, bg=CARD, padx=20, pady=16)
        i4.pack(fill="both", expand=True)

        tk.Label(i4, text="COMPONENT 4", bg=CARD, fg=TEXT_DIM, font=FONT_MONO).pack(anchor="w")
        tk.Label(i4, text="🎤  Audio Integrity & Voice Verification",
                 bg=CARD, fg=ACCENT, font=FONT_H3).pack(anchor="w", pady=(2, 10))

        if audio_module and s4:
            # Show ONLY the final verdict summary as specified
            # Integrity score determines the label
            if c4_score >= C4_INTEGRITY_PASS:
                score_col  = SUCCESS
                score_note = f"≥{C4_INTEGRITY_PASS}% — clean"
            elif c4_score >= C4_INTEGRITY_REVIEW:
                score_col  = WARNING
                score_note = f"{C4_INTEGRITY_REVIEW}–{C4_INTEGRITY_PASS-1}% — review"
            else:
                score_col  = DANGER
                score_note = f"<{C4_INTEGRITY_REVIEW}% — suspicious"

            stat_row(i4, "Integrity Score",
                     f"{c4_score:.1f}%  ({score_note})",
                     score_col)
            stat_row(i4, "Segments Analysed",
                     str(s4.get("total_segments", 0)), TEXT)
            stat_row(i4, "Suspicious Segs",
                     str(s4.get("suspicious", 0)),
                     DANGER if s4.get("suspicious", 0) > 0 else SUCCESS)
            stat_row(i4, "WAV Saved",    "✓", SUCCESS)
            stat_row(i4, "Report Saved", "✓", SUCCESS)

            # Verdict label (prominent)
            tk.Frame(i4, bg=BORDER, height=1).pack(fill="x", pady=(12, 4))

            if c4_score >= C4_INTEGRITY_PASS:
                verdict_text = f"✓  Audio Integrity OK — Student is NOT cheating  ({c4_score:.1f}% clean)"
                v_col = SUCCESS
            elif c4_score >= C4_INTEGRITY_REVIEW:
                verdict_text = f"⚠  Manual Review Required  ({c4_score:.1f}% clean)"
                v_col = WARNING
            else:
                verdict_text = f"✗  Suspicious Audio Detected — Possible cheating  ({c4_score:.1f}% clean)"
                v_col = DANGER

            tk.Label(i4, text=verdict_text, bg=CARD, fg=v_col,
                     font=("Courier New", 11, "bold"),
                     wraplength=340, justify="left").pack(anchor="w", pady=(4, 0))

            c4_reason = verdict_text
        else:
            c4_reason = "Audio module not initialised"
            tk.Label(i4, text="Audio module not initialised or calibration skipped.",
                     bg=CARD, fg=TEXT_DIM, font=FONT_BODY,
                     wraplength=340, justify="left").pack(anchor="w", pady=(8, 0))

        verdict_badge(i4, c4_cheat, is_review=c4_review)

        # ── Combined verdict card ─────────────────────────────────────────
        v_card  = card_frame(self.main)
        v_card.pack(fill="x", padx=40, pady=16)
        v_inner = tk.Frame(v_card, bg=CARD, padx=24, pady=16)
        v_inner.pack(fill="x")

        tk.Label(v_inner, text="COMBINED RESULT  (OR-gate across all components)",
                 bg=CARD, fg=TEXT_DIM, font=FONT_MONO).pack(anchor="w")

        summary = tk.Frame(v_inner, bg=CARD)
        summary.pack(anchor="w", pady=(8, 0))

        c4_disp = ("REVIEW" if c4_review else ("CHEAT" if c4_cheat else "CLEAN"))
        c4_col  = (WARNING  if c4_review else (DANGER  if c4_cheat else SUCCESS))

        for lbl, is_c, disp, col in [
            ("Eye & Head :", eye_cheat,  "CHEAT"   if eye_cheat  else "CLEAN", DANGER if eye_cheat  else SUCCESS),
            ("Face Verif :", face_cheat, "CHEAT"   if face_cheat else "CLEAN", DANGER if face_cheat else SUCCESS),
            ("Obj/Person :", c3_cheat,   "CHEAT"   if c3_cheat   else "CLEAN", DANGER if c3_cheat   else SUCCESS),
            ("Audio Integ:", False,       c4_disp,  c4_col),
        ]:
            tk.Label(summary, text=lbl, bg=CARD, fg=TEXT_DIM,
                     font=FONT_MONO).pack(side="left", padx=(0, 4))
            tk.Label(summary, text=disp, bg=CARD, fg=col,
                     font=("Courier New", 11, "bold")).pack(side="left", padx=(0, 24))

        tk.Frame(v_inner, bg=BORDER, height=1).pack(fill="x", pady=10)

        final_txt = ("⚠  FINAL VERDICT: CHEATING SUSPECTED"
                     if final_cheat else
                     "✓  FINAL VERDICT: NO SUSPICIOUS BEHAVIOR")
        tk.Label(v_inner, text=final_txt, bg=CARD, fg=headline_col,
                 font=("Courier New", 15, "bold")).pack(anchor="w")

        flagged = []
        if eye_cheat:  flagged.append("Eye/Head")
        if face_cheat: flagged.append("Face Verification")
        if c3_cheat:   flagged.append("Object/Person Detection")
        if c4_cheat:   flagged.append("Audio Integrity")
        if c4_review and not c4_cheat:
            flagged.append("Audio (Review Required)")
        sub_txt = ("Flagged by: " + " & ".join(flagged)) if flagged else \
                  "All components within acceptable limits. Session appears clean."
        tk.Label(v_inner, text=sub_txt, bg=CARD, fg=TEXT_DIM,
                 font=FONT_BODY).pack(anchor="w", pady=(4, 0))

        # ── Saved files card ──────────────────────────────────────────────
        paths_card  = card_frame(self.main)
        paths_card.pack(fill="x", padx=40, pady=(0, 8))
        paths_inner = tk.Frame(paths_card, bg=CARD, padx=24, pady=14)
        paths_inner.pack(fill="x")

        tk.Label(paths_inner, text="SAVED FILES", bg=CARD, fg=TEXT_DIM,
                 font=FONT_MONO).pack(anchor="w", pady=(0, 8))

        if self.app.report_path:
            tk.Label(paths_inner, text="📄  Report:", bg=CARD, fg=TEXT_DIM,
                     font=FONT_MONO).pack(anchor="w")
            tk.Label(paths_inner, text=self.app.report_path, bg=CARD, fg=ACCENT,
                     font=FONT_MONO).pack(anchor="w")

        vid1 = getattr(self.app, "video_path", None)
        if vid1:
            tk.Label(paths_inner, text="🎥  Video 1  (Eye/Head + Face Verif):",
                     bg=CARD, fg=TEXT_DIM, font=FONT_MONO).pack(anchor="w", pady=(6, 0))
            tk.Label(paths_inner, text=vid1, bg=CARD, fg=ACCENT,
                     font=FONT_MONO).pack(anchor="w")

        vid2 = getattr(self.app, "video_path_obj", None)
        if vid2:
            tk.Label(paths_inner, text="🎥  Video 2  (Object / Person Detect):",
                     bg=CARD, fg=TEXT_DIM, font=FONT_MONO).pack(anchor="w", pady=(6, 0))
            tk.Label(paths_inner, text=vid2, bg=CARD, fg=ACCENT,
                     font=FONT_MONO).pack(anchor="w")

        if audio_module and s4:
            wav = s4.get("wav_path", "N/A")
            rpt = s4.get("report_path", "N/A")
            if wav != "N/A":
                tk.Label(paths_inner, text="🎙️  Audio Recording (WAV):",
                         bg=CARD, fg=TEXT_DIM, font=FONT_MONO).pack(anchor="w", pady=(6, 0))
                tk.Label(paths_inner, text=wav, bg=CARD, fg=ACCENT,
                         font=FONT_MONO).pack(anchor="w")
            if rpt != "N/A":
                tk.Label(paths_inner, text="📊  Audio Analysis Report (TXT):",
                         bg=CARD, fg=TEXT_DIM, font=FONT_MONO).pack(anchor="w", pady=(6, 0))
                tk.Label(paths_inner, text=rpt, bg=CARD, fg=ACCENT,
                         font=FONT_MONO).pack(anchor="w")

        tk.Label(paths_inner, text="📊  Violation log (CSV):",
                 bg=CARD, fg=TEXT_DIM, font=FONT_MONO).pack(anchor="w", pady=(6, 0))
        tk.Label(paths_inner, text=C3_CSV_LOG, bg=CARD, fg=ACCENT,
                 font=FONT_MONO).pack(anchor="w")

        # ── Buttons ──────────────────────────────────────────────────────
        btn_row = tk.Frame(self.main, bg=BG)
        btn_row.pack(pady=20)
        styled_button(btn_row, "VIEW REPORT", self._view_report,
                      bg=BG3,    width=14).pack(side="left", padx=8)
        styled_button(btn_row, "NEW EXAM",    self._new_exam,
                      bg=ACCENT2, width=14).pack(side="left", padx=8)
        styled_button(btn_row, "EXIT",        self.app.root.quit,
                      bg=BG3,    width=14).pack(side="left", padx=8)

    def _view_report(self):
        if not self.app.report_str:
            messagebox.showinfo("Report", "No report available yet.")
            return
        win = tk.Toplevel(self.app.root)
        win.title("Exam Report")
        win.configure(bg=BG)
        win.geometry("680x580")
        tk.Label(win, text="EXAM REPORT", bg=BG, fg=ACCENT,
                 font=FONT_H2).pack(pady=(16, 8))
        text_frame = tk.Frame(win, bg=BG)
        text_frame.pack(fill="both", expand=True, padx=20, pady=(0, 16))
        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side="right", fill="y")
        txt = tk.Text(text_frame, bg=BG3, fg=TEXT, font=FONT_MONO,
                      relief="flat", bd=0, wrap="word",
                      yscrollcommand=scrollbar.set)
        txt.insert("1.0", self.app.report_str)
        txt.config(state="disabled")
        txt.pack(fill="both", expand=True)
        scrollbar.config(command=txt.yview)
        styled_button(win, "CLOSE", win.destroy, bg=BG3, width=12).pack(pady=(0, 16))

    def _new_exam(self):
        self.app.exam_running    = False
        self.app.final_cheated   = False
        self.app.current_student = None
        self.app.report_path     = None
        self.app.report_str      = None
        self.app.obj             = None
        self.app.audio           = None   # reset Component 4
        self.app.video_path      = None
        self.app.video_path_obj  = None
        self.app.show_frame(LoginPage)


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    app = EdgeExamApp()
    app.run()
