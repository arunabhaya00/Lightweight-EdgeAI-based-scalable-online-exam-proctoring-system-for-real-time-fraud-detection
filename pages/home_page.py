# pages/home_page.py — Home page showing available exams

import tkinter as tk
from config import (
    BG, BG2, ACCENT, ACCENT2, TEXT, TEXT_DIM, CARD,
    FONT_H1, FONT_H2, FONT_BODY
)
from modules.utils import styled_button, card_frame


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

        from pages.instruction_page import InstructionPage
        styled_button(inner, "BEGIN →",
                      lambda: self.app.show_frame(InstructionPage),
                      bg=ACCENT2, width=12).pack(anchor="e", pady=(12, 0))
