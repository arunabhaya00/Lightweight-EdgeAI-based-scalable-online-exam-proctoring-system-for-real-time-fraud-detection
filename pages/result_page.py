# pages/result_page.py — Results page showing all 4 component verdicts

import time
import tkinter as tk
from tkinter import messagebox

from config import (
    BG, BG2, BG3, ACCENT, ACCENT2, SUCCESS, WARNING, DANGER, TEXT, TEXT_DIM,
    CARD, BORDER,
    FONT_H1, FONT_H2, FONT_H3, FONT_BODY, FONT_MONO,
    MATCH_THRESHOLD, FACE_WINDOW_SECONDS, FACE_WINDOW_THRESHOLD,
    WINDOW_THRESHOLD_PERCENT, OVERALL_THRESHOLD_PERCENT,
    C3_CSV_LOG,
    C4_INTEGRITY_PASS, C4_INTEGRITY_REVIEW
)
from modules.utils import styled_button, card_frame


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

        eye_mod      = self.app.eye
        face_module  = self.app.face
        obj_module   = self.app.obj
        audio_module = self.app.audio
        eye_tracker  = eye_mod.tracker if eye_mod else None

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

        c4_cheat       = False
        c4_review      = False
        c4_score       = 100.0
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

        # ── helpers ───────────────────────────────────────────────────────
        def stat_row(parent, label, value, col):
            r = tk.Frame(parent, bg=CARD)
            r.pack(fill="x", pady=2)
            tk.Label(r, text=label, bg=CARD, fg=TEXT_DIM,
                     font=FONT_MONO, width=22, anchor="w").pack(side="left")
            tk.Label(r, text=value, bg=CARD, fg=col,
                     font=("Courier New", 11, "bold")).pack(side="left")

        def verdict_badge(parent, is_cheat, is_review=False):
            if is_cheat:
                txt, col, bg_ = "CHEAT",    DANGER,  "#2a0a0a"
            elif is_review:
                txt, col, bg_ = "REVIEW",   WARNING, "#2a1a00"
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
        else:
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
            ("Eye & Head :", eye_cheat,  "CHEAT"    if eye_cheat  else "CLEAN", DANGER if eye_cheat  else SUCCESS),
            ("Face Verif :", face_cheat, "CHEAT"    if face_cheat else "CLEAN", DANGER if face_cheat else SUCCESS),
            ("Obj/Person :", c3_cheat,   "CHEAT"    if c3_cheat   else "CLEAN", DANGER if c3_cheat   else SUCCESS),
            ("Audio Integ:", False,       c4_disp,   c4_col),
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

        # ── Buttons ───────────────────────────────────────────────────────
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
        txt = tk.Text(text_frame, bg=BG2, fg=TEXT, font=FONT_MONO,
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
        self.app.audio           = None
        self.app.video_path      = None
        self.app.video_path_obj  = None
        from pages.login_page import LoginPage
        self.app.show_frame(LoginPage)
