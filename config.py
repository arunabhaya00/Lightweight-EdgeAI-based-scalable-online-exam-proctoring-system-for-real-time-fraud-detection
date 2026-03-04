# config.py — All constants and configuration for the Edge AI Exam System

import os
import sys
import cv2
import pyaudio

# ─── Directory paths ─────────────────────────────────────────────────────────
DB_DIR     = "db"
RESULT_DIR = "results"

os.makedirs(DB_DIR,     exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ─── Silent-Face Anti-Spoofing path ──────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__),
                             "Silent-Face-Anti-Spoofing"))

# ─── Video ────────────────────────────────────────────────────────────────────
VIDEO_FPS = 20

# ─── Component 1: Eye / Head thresholds ───────────────────────────────────────
WINDOW_SIZE_MINUTES       = 5
WINDOW_THRESHOLD_PERCENT  = 40
OVERALL_THRESHOLD_PERCENT = 20
FACE_WINDOW_SECONDS       = 10
FACE_WINDOW_THRESHOLD     = 0.4

HEAD_YAW_RATIO_THRESHOLD   = 0.15
HEAD_PITCH_RATIO_THRESHOLD = 0.15

# ─── Component 2: Face verification ───────────────────────────────────────────
MATCH_THRESHOLD       = 0.45
LOGIN_VERIFY_FRAMES   = 5
LOGIN_VERIFY_REQUIRED = 4
REGISTER_FRAMES       = 15

# ─── Liveness (blink) ─────────────────────────────────────────────────────────
EAR_THRESHOLD       = 0.21
BLINK_CONSEC_FRAMES = 2
REQUIRED_BLINKS     = 2
LIVENESS_TIMEOUT    = 8

# ─── Exam ─────────────────────────────────────────────────────────────────────
EXAM_DURATION_SECONDS = 60 * 60   # 1 hour default

# ─── Component 3: Object / person detection ───────────────────────────────────
CONF_TEXT          = 0.25
CONF_PERSON        = 0.4
CONF_PHONE         = 0.4
TEMPORAL_THRESHOLD = 3.0
MAX_MISSING_FRAMES = 15

TEXT_MODEL_PATH = "exam_note_detector.pt"
COCO_MODEL_PATH = "yolov8n.pt"
FACE_MODEL_PATH = "yolov8n-face.pt"

C3_CSV_LOG  = os.path.join("results", "violations.csv")
C3_JSON_LOG = os.path.join("results", "violations.json")

# ─── Component 4: Audio integrity ─────────────────────────────────────────────
C4_AUDIO_CHUNK       = 2048
C4_AUDIO_FORMAT      = pyaudio.paInt16
C4_AUDIO_CHANNELS    = 1
C4_AUDIO_RATE        = 16000
C4_ECAPA_MODEL_PATH  = "ecapa_quantized_traced.pth"
C4_YAMNET_CLASSIFIER = "yamnet_audio_classifier_new.keras"
C4_YAMNET_HUB_URL    = "https://tfhub.dev/google/yamnet/1"
C4_SPEAKER_THRESHOLD = 0.35
C4_SEGMENT_SECONDS   = 10
C4_INTEGRITY_PASS    = 80
C4_INTEGRITY_REVIEW  = 60
C4_CALIBRATION_SECONDS = 15

# ─── Colour palette ───────────────────────────────────────────────────────────
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

# ─── Fonts ────────────────────────────────────────────────────────────────────
FONT_H1   = ("Courier New", 26, "bold")
FONT_H2   = ("Courier New", 18, "bold")
FONT_H3   = ("Courier New", 13, "bold")
FONT_BODY = ("Courier New", 11)
FONT_MONO = ("Courier New", 10)

# ─── OpenCV font ──────────────────────────────────────────────────────────────
FONT_CV = cv2.FONT_HERSHEY_SIMPLEX
