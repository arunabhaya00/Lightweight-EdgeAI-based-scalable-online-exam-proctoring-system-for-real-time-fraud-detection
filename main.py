# main.py — Entry point for the Edge AI Exam System

import cv2
import tkinter as tk

from config import BG

from pages.login_page       import LoginPage
from pages.home_page        import HomePage
from pages.instruction_page import InstructionPage
from pages.exam_page        import ExamPage
from pages.result_page      import ResultPage


class EdgeExamApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("EdgeAI Exam System")
        self.root.attributes("-fullscreen", True)
        self.root.configure(bg=BG)

        self.cap              = cv2.VideoCapture(0) # Camera capture object
        self.current_student  = None    # Current logged in student
        self.exam_running     = False   # Flag to track exam state
        self.final_cheated    = False   # Final cheating decision
        self.eye              = None
        self.face             = None
        self.obj              = None    # Detection models
        self.audio            = None
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


if __name__ == "__main__":
    app = EdgeExamApp()
    app.run()
