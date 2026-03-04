# modules/utils.py — Shared UI helper functions and question loader

import tkinter as tk
from config import (ACCENT, ACCENT2, BG3, TEXT, TEXT_DIM, CARD, BORDER,
                    FONT_H3, FONT_BODY)


def styled_button(parent, text, command, bg=ACCENT2, fg=TEXT, width=18, **kw):
    btn = tk.Button(
        parent, text=text, command=command,
        bg=bg, fg=fg, activebackground=ACCENT, activeforeground=BG3,
        font=FONT_H3, relief="flat", bd=0,
        padx=18, pady=10, cursor="hand2", width=width, **kw
    )
    btn.bind("<Enter>", lambda e: btn.config(bg=ACCENT, fg=BG3))
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


def load_questions(filepath="questions.docx"):
    """Parse MCQ questions from a .docx file."""
    from docx import Document
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
