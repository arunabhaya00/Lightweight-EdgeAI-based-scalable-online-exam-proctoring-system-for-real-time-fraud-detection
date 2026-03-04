# pages/instruction_page.py — Instructions and voice calibration page

import threading
import tkinter as tk
from tkinter import messagebox

from config import (
    BG, BG2, BG3, ACCENT, SUCCESS, WARNING, DANGER, TEXT, TEXT_DIM, CARD,
    FONT_H1, FONT_H3, FONT_BODY, FONT_MONO,
    C4_CALIBRATION_SECONDS
)
from modules.utils import styled_button, card_frame
from modules.audio_integrity import AudioIntegrityModule


class InstructionPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app         = app
        self._calibrated = False
        self._build()

    def on_show(self):
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
            ("⏱  Exam Timer",       "The exam runs for 60 minutes. A countdown timer is shown. Exam auto-submits at zero."),
            ("🎤  Audio Monitoring", "Your audio is monitored throughout the exam. Whispering or external audio sources will be flagged."),
        ]
        for icon_title, desc in rules:
            row   = card_frame(center)
            row.pack(fill="x", pady=6)
            inner = tk.Frame(row, bg=CARD, padx=20, pady=14)
            inner.pack(fill="x")
            tk.Label(inner, text=icon_title, bg=CARD, fg=ACCENT,
                     font=FONT_H3).pack(anchor="w")
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
        if not self.app.current_student:
            messagebox.showerror("Error", "No student logged in.")
            return

        self._cal_btn.config(state="disabled")
        self._cal_status_lbl.config(
            text=f"🎤  Recording… speak naturally for {C4_CALIBRATION_SECONDS}s",
            fg=WARNING)
        self.app.root.update()

        self.app.audio = AudioIntegrityModule(self.app.current_student)

        done_evt = threading.Event()
        success  = [False]

        def _status(msg):
            self._cal_status_lbl.after(
                0, lambda m=msg: self._cal_status_lbl.config(text=m, fg=WARNING))

        def _worker():
            success[0] = self.app.audio.calibrate(
                duration=C4_CALIBRATION_SECONDS,
                status_cb=_status)
            done_evt.set()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

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
        from pages.exam_page import ExamPage
        self.app.frames[ExamPage].start_exam()
        self.app.show_frame(ExamPage)
