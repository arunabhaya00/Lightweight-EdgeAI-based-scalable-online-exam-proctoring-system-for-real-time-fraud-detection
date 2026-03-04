# modules/audio_integrity.py — Component 4: Audio integrity & voice verification

import os
import time
import wave
import threading
import numpy as np
from datetime import datetime

import pyaudio

from config import (
    RESULT_DIR, EXAM_DURATION_SECONDS,
    C4_AUDIO_CHUNK, C4_AUDIO_CHANNELS, C4_AUDIO_RATE,
    C4_ECAPA_MODEL_PATH, C4_YAMNET_CLASSIFIER, C4_YAMNET_HUB_URL,
    C4_SPEAKER_THRESHOLD, C4_SEGMENT_SECONDS,
    C4_INTEGRITY_PASS, C4_INTEGRITY_REVIEW, C4_CALIBRATION_SECONDS
)

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


# ─── Module-level model cache ─────────────────────────────────────────────────
_ecapa_model      = None
_mel_extractor    = None
_yamnet_base      = None
_yamnet_cls       = None
_c4_models_loaded = False


def _load_c4_models():
    global _ecapa_model, _mel_extractor, _yamnet_base, _yamnet_cls, _c4_models_loaded
    if _c4_models_loaded:
        return

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
    CLASS_NAMES = ['Self-Speaking', 'Whispering', 'Lecture/Video', 'Background Noise']

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

    if _yamnet_base is None or _yamnet_cls is None:
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
            adj = preds[0].copy()
            adj[2] *= 0.6
            predicted_class = int(np.argmax(adj))
            confidence      = float(preds[0][predicted_class])
            if speaker_match:
                predicted_class = 0
                confidence      = max(confidence, 0.75)
            else:
                if predicted_class == 2:
                    whisper_p = adj[1]
                    lecture_p = adj[2]
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


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO INTEGRITY MODULE
# ─────────────────────────────────────────────────────────────────────────────
class AudioIntegrityModule:
    def __init__(self, student_name: str):
        _load_c4_models()
        self.student_name       = student_name
        self.baseline_embedding = None
        self.calibration_audio  = None
        self._stop_event        = None
        self._audio_thread      = None
        self._wav_path          = None
        self.segments           = []
        self.integrity_score    = 100.0
        self._verdict           = "PENDING"
        self._is_cheat          = False

    def calibrate(self, duration: int = C4_CALIBRATION_SECONDS,
                  status_cb=None) -> bool:
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

    def start_recording(self, max_duration_seconds: int = EXAM_DURATION_SECONDS + 60):
        ts = int(time.time())
        self._wav_path   = os.path.join(
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

    def stop_and_analyse(self, status_cb=None) -> bool:
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

        try:
            if SF_AVAILABLE:
                audio_data, _sr = sf.read(wav)
            else:
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

        seg_len = C4_AUDIO_RATE * C4_SEGMENT_SECONDS
        n_segs  = len(audio_data) // seg_len
        results = []
        for i in range(n_segs):
            if status_cb:
                status_cb(f"🔊  Analysing segment {i+1}/{n_segs}…")
            seg = audio_data[i * seg_len : (i + 1) * seg_len]
            results.append(_classify_segment(seg, self.baseline_embedding, i))
            print(f"[C4] Seg {i+1}/{n_segs}: {results[-1]['class']}  "
                  f"({results[-1]['status']})")

        self.segments = results
        self._compute_verdict(results)
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
            self._is_cheat = False
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

    def final_result(self) -> bool:
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
