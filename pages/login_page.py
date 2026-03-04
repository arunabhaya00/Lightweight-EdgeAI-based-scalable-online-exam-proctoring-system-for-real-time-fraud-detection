# pages/login_page.py — Login & Registration page

import os
import time
import pickle
import numpy as np
import tkinter as tk
from tkinter import simpledialog
from datetime import datetime
from PIL import Image, ImageTk
import cv2
import face_recognition

from config import (
    DB_DIR, MATCH_THRESHOLD, LOGIN_VERIFY_FRAMES, LOGIN_VERIFY_REQUIRED,
    REGISTER_FRAMES, REQUIRED_BLINKS, LIVENESS_TIMEOUT,
    BG, BG2, BG3, ACCENT, ACCENT2, SUCCESS, WARNING, DANGER, TEXT, TEXT_DIM, BORDER,
    FONT_H1, FONT_BODY, FONT_MONO, FONT_CV
)
from modules.biometric import LivenessDetector
from modules.utils import styled_button


class LoginPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app       = app
        self._liveness = LivenessDetector()
        self._build()

    def _build(self):
        # ── Left sidebar ──────────────────────────────────────────────────
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

        # ── Right panel ───────────────────────────────────────────────────
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
        embeddings   = []
        attempt      = 0
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

        matches   = 0
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
        from pages.home_page import HomePage
        self.app.root.after(800, lambda: self.app.show_frame(HomePage))
