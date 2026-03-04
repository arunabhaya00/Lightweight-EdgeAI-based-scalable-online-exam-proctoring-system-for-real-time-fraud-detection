# pages/exam_page.py — Exam page with MCQ questions and proctor loop

import os
import time
import threading
import tkinter as tk
from tkinter import messagebox
import cv2

from config import (
    RESULT_DIR, VIDEO_FPS, EXAM_DURATION_SECONDS,
    BG, BG2, BG3, ACCENT, ACCENT2, SUCCESS, WARNING, DANGER, TEXT, TEXT_DIM,
    CARD, BORDER,
    FONT_H1, FONT_H3, FONT_BODY, FONT_MONO
)
from modules.utils import styled_button, card_frame, load_questions
from modules.eye_head import EyeHeadModule
from modules.biometric import FaceVerificationModule
from modules.object_detection import ObjectDetectionModule
from modules.report import generate_report


class ExamPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app         = app
        self.questions   = load_questions("questions.docx")
        self.answer_vars = {}
        self._build()

    def _build(self):
        # ── Top bar ───────────────────────────────────────────────────────
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

        # ── Scrollable question area ──────────────────────────────────────
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
        self.app.exam_running  = True
        self.app.final_cheated = False

        self.student_lbl.config(text=f"Student: {self.app.current_student or ''}")

        # Silent-Face Anti-Spoofing setup
        from src.anti_spoof_predict import AntiSpoofPredict
        self.anti_spoof_model_path = os.path.join(
            "Silent-Face-Anti-Spoofing",
            "resources",
            "anti_spoof_models",
            "2.7_80x80_MiniFASNetV2.pth"
        )
        self.anti_spoof = AntiSpoofPredict(
            device_id=0,
            model_path=self.anti_spoof_model_path
        )

        self.violation_frames = 0
        self.checked_frames   = 0
        self.window_start     = time.time()

        ts     = int(time.time())
        w      = int(self.app.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(self.app.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
        e_m, e_s  = divmod(int(elapsed),   60)
        self.elapsed_lbl.config(text=f"{e_m:02d}:{e_s:02d}")
        r_m, r_s  = divmod(int(remaining), 60)
        self.timer_lbl.config(text=f"{r_m:02d}:{r_s:02d}")
        if remaining > 600:
            fg_col = ACCENT;  bg_col = BG3;          border = ACCENT
        elif remaining > 120:
            fg_col = WARNING; bg_col = BG3;          border = WARNING
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
        frame_count    = 0
        start_time     = time.time()
        last_frame_c12 = None
        last_frame_c3  = None

        while self.app.exam_running:
            ret, frame = self.app.cap.read()
            if not ret:
                break

            self.checked_frames += 1

            # Silent-Face spoof detection every 3rd frame
            if self.checked_frames % 3 == 0:
                bbox = self.anti_spoof.get_bbox(frame)
                if bbox is not None:
                    x, y, w, h = bbox
                    face = frame[y:y+h, x:x+w]
                    if face.size > 0:
                        face_resized = cv2.resize(face, (80, 80))
                        result       = self.anti_spoof.predict(face_resized)
                        real_score   = result[0][0]
                        spoof_score  = result[0][1]
                        if spoof_score > real_score:
                            print("MiniFASNet SPOOF detected")
                            self.app.face.violation_frames += 1

            raw_c12 = frame.copy()
            raw_c3  = frame.copy()
            frame_c12, is_eye_violation = self.app.eye.process(raw_c12)
            self.app.face.process(frame_c12)
            frame_c3, is_obj_violation  = self.app.obj.process(raw_c3)

            if is_eye_violation or is_obj_violation:
                self.app.final_cheated = True

            self.app.latest_frame = frame_c12.copy()
            last_frame_c12        = frame_c12
            last_frame_c3         = frame_c3

            elapsed         = time.time() - start_time
            target_count    = int(elapsed * VIDEO_FPS)
            frames_to_write = max(1, target_count - frame_count)
            for _ in range(frames_to_write):
                self.app.writer.write(last_frame_c12)
                self.app.writer_obj.write(last_frame_c3)
                frame_count += 1

        # Flush remaining frames
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

        # Component 4: stop recording + analyse
        if self.app.audio:
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

        from pages.result_page import ResultPage
        self.app.show_frame(ResultPage)
        self.app.frames[ResultPage].refresh()
