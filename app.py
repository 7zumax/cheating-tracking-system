"""
Tkinter desktop app for the cheating tracking system.

    python app.py              # pick a video with Upload Video
    python app.py exam.mp4     # start processing straight away
"""
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

# keep relative paths (models, alerts/, alerts.db) next to the app
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

import database as db                                   # noqa: E402
from detector import CheatingDetector, save_screenshot  # noqa: E402

SCREENSHOT_DIR = os.path.join("alerts", "screenshots")
VIEW_W, VIEW_H = 768, 432

BG = "#0f172a"
PANEL = "#1e293b"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
ACCENT = "#2563eb"
DANGER = "#dc2626"
OK = "#16a34a"


class CheatingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cheating Tracking & Alert System")
        self.root.configure(bg=BG)
        self.root.geometry("1400x660")
        self.root.minsize(1200, 620)
        self.set_icon()

        db.create_table()
        self.detector = None  # lazy-loaded on first video
        self.frame_queue = queue.Queue(maxsize=4)
        self.stop_flag = threading.Event()
        self.worker = None
        self.popups = []
        self.session_alerts = 0

        self.build_ui()
        self.load_history()
        self.root.after(15, self.poll_queue)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # --- UI ---
    def set_icon(self):
        """Set window and taskbar icon."""
        try:
            if sys.platform.startswith("win"):
                self.root.iconbitmap(os.path.join("assets", "icon.ico"))
            self._icon_img = tk.PhotoImage(file=os.path.join("assets", "icon.png"))
            self.root.iconphoto(True, self._icon_img)
        except Exception:
            pass

    def build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background=PANEL, fieldbackground=PANEL,
                        foreground=TEXT, rowheight=26, borderwidth=0)
        style.configure("Treeview.Heading", background=BG, foreground=TEXT,
                        font=("Segoe UI", 9, "bold"))
        style.map("Treeview", background=[("selected", ACCENT)])
        style.configure("Horizontal.TProgressbar", troughcolor=PANEL, background=ACCENT)

        # top bar
        top = tk.Frame(self.root, bg=BG)
        top.pack(fill="x", padx=16, pady=(14, 8))
        tk.Label(top, text="Cheating Tracking & Alert System", bg=BG, fg=TEXT,
                 font=("Segoe UI", 18, "bold")).pack(side="left")
        self.stop_btn = self.make_button(top, "Stop", self.stop_video, DANGER)
        self.stop_btn.pack(side="right", padx=(8, 0))
        self.stop_btn.config(state="disabled")
        self.make_button(top, "Upload Video", self.upload_video, ACCENT).pack(side="right")

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=(0, 14))

        # video + status
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="y")
        video_box = tk.Frame(left, bg="black", width=VIEW_W, height=VIEW_H)
        video_box.pack()
        video_box.pack_propagate(False)
        self.video_label = tk.Label(video_box, bg="black",
                                    text="Upload an exam video to start",
                                    fg=MUTED, font=("Segoe UI", 13), compound="center")
        self.video_label.pack(fill="both", expand=True)
        self.progress = ttk.Progressbar(left, length=VIEW_W, mode="determinate",
                                        style="Horizontal.TProgressbar")
        self.progress.pack(pady=(8, 4))
        self.status = tk.Label(left, text="Ready.", bg=BG, fg=MUTED,
                               font=("Segoe UI", 10), anchor="w")
        self.status.pack(fill="x")

        legend = tk.Frame(left, bg=BG)
        legend.pack(fill="x", pady=(6, 0))
        tk.Label(legend, text="■ Normal student", fg=OK, bg=BG,
                 font=("Segoe UI", 10, "bold")).pack(side="left")
        tk.Label(legend, text="   ■ Suspected cheating", fg=DANGER, bg=BG,
                 font=("Segoe UI", 10, "bold")).pack(side="left")
        self.count_label = tk.Label(legend, text="Alerts this video: 0", fg=TEXT, bg=BG,
                                    font=("Segoe UI", 10, "bold"))
        self.count_label.pack(side="right")

        # alert history
        right = tk.Frame(body, bg=PANEL)
        right.pack(side="left", fill="both", expand=True, padx=(14, 0))
        tk.Label(right, text="Alert History", bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=12, pady=(10, 6))

        cols = ("datetime", "student", "behavior", "confidence", "screenshot")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", selectmode="browse")
        for col, title, width in [("datetime", "Date / Time", 150), ("student", "ID", 35),
                                  ("behavior", "Behavior", 180), ("confidence", "Conf.", 50),
                                  ("screenshot", "Screenshot", 120)]:
            self.tree.heading(col, text=title)
            self.tree.column(col, width=width, anchor="w", stretch=(col == "screenshot"))
        self.tree.pack(fill="both", expand=True, padx=12)
        self.tree.bind("<Double-1>", lambda e: self.view_screenshot())

        btns = tk.Frame(right, bg=PANEL)
        btns.pack(fill="x", padx=12, pady=10)
        self.make_button(btns, "View Screenshot", self.view_screenshot, ACCENT).pack(side="left")
        self.make_button(btns, "Open Folder", self.open_folder, "#475569").pack(side="left", padx=8)
        self.make_button(btns, "Clear History", self.clear_history, "#475569").pack(side="right")

    @staticmethod
    def make_button(parent, text, command, colour):
        return tk.Button(parent, text=text, command=command, bg=colour, fg="white",
                         activebackground=colour, activeforeground="white",
                         relief="flat", font=("Segoe UI", 10, "bold"),
                         padx=14, pady=6, cursor="hand2", bd=0)

    # --- history ---
    def load_history(self):
        self.tree.delete(*self.tree.get_children())
        for a in db.get_all_alerts():
            self.add_history_row(a, at_top=False)

    def add_history_row(self, a, at_top=True):
        values = (a["datetime"], a["student_id"], a["behavior"],
                  f'{a["confidence"]:.2f}', os.path.basename(a["screenshot_path"] or ""))
        self.tree.insert("", 0 if at_top else "end", iid=str(a["id"]), values=values)

    def clear_history(self):
        if messagebox.askyesno("Clear history",
                               "Delete all alerts from the database?\n"
                               "(Screenshot files are kept.)"):
            db.clear_alerts()
            self.load_history()

    # --- screenshot viewer ---
    def view_screenshot(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Screenshot viewer", "Select an alert in the table first.")
            return
        alert = db.get_alert(int(sel[0]))
        path = alert["screenshot_path"] if alert else None
        if not path or not os.path.exists(path):
            messagebox.showerror("Screenshot viewer", f"File not found:\n{path}")
            return

        win = tk.Toplevel(self.root, bg=BG)
        win.title(f"Alert #{alert['id']} - {alert['behavior']}")
        img = Image.open(path)
        img.thumbnail((960, 540))
        photo = ImageTk.PhotoImage(img)
        lbl = tk.Label(win, image=photo, bg=BG)
        lbl.image = photo  # keep a reference
        lbl.pack(padx=10, pady=10)
        info = (f"Student ID {alert['student_id']}   |   {alert['behavior']}   |   "
                f"confidence {alert['confidence']:.2f}   |   {alert['datetime']}\n"
                f"Video: {alert['video_name']}  at {alert['video_time']} s")
        tk.Label(win, text=info, bg=BG, fg=TEXT, font=("Segoe UI", 10)).pack(pady=(0, 10))

    def open_folder(self):
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        if sys.platform.startswith("win"):
            os.startfile(os.path.abspath(SCREENSHOT_DIR))
        else:
            subprocess.Popen(["xdg-open", SCREENSHOT_DIR])

    # --- notifications ---
    def notify(self, alert):
        """Show an alert popup in the bottom-right corner for 5 s."""
        self.root.bell()
        pop = tk.Toplevel(self.root, bg=DANGER)
        pop.overrideredirect(True)
        pop.attributes("-topmost", True)
        w, h = 330, 120
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        y = sh - h - 60 - len(self.popups) * (h + 10)
        pop.geometry(f"{w}x{h}+{sw - w - 20}+{max(y, 0)}")

        time_txt = datetime.strptime(alert["datetime"], "%Y-%m-%d %H:%M:%S").strftime("%I:%M %p")
        lines = [("⚠  Cheating Alert Detected", ("Segoe UI", 12, "bold")),
                 (f"Behavior: {alert['behavior']}  (ID {alert['student_id']})", ("Segoe UI", 10)),
                 (f"Time: {time_txt}", ("Segoe UI", 10)),
                 ("Screenshot saved successfully.", ("Segoe UI", 10))]
        for text, font in lines:
            tk.Label(pop, text=text, bg=DANGER, fg="white", font=font,
                     anchor="w").pack(fill="x", padx=14, pady=1)
        pop.bind("<Button-1>", lambda e: self.close_popup(pop))
        self.popups.append(pop)
        self.root.after(5000, lambda: self.close_popup(pop))

    def close_popup(self, pop):
        if pop in self.popups:
            self.popups.remove(pop)
            pop.destroy()

    # --- video processing (worker thread) ---
    def upload_video(self):
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("Busy", "A video is already being processed.")
            return
        path = filedialog.askopenfilename(
            title="Choose an exam video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
        if path:
            self.start_video(path)

    def start_video(self, path):
        self.stop_flag.clear()
        self.session_alerts = 0
        self.count_label.config(text="Alerts this video: 0")
        self.stop_btn.config(state="normal")
        self.status.config(text="Loading AI models...")
        self.worker = threading.Thread(target=self.process_video, args=(path,), daemon=True)
        self.worker.start()

    def stop_video(self):
        self.stop_flag.set()

    def process_video(self, path):
        """Worker thread: detect, save alerts, push results to the UI queue."""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self.frame_queue.put(("error", f"Could not open video:\n{path}"))
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        if self.detector is None:
            self.detector = CheatingDetector(fps)
        self.detector.reset(fps)
        video_name = os.path.basename(path)

        i = 0
        while not self.stop_flag.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            annotated, events = self.detector.process_frame(frame, i)
            saved = []
            for ev in events:
                shot = save_screenshot(ev.pop("frame"), SCREENSHOT_DIR, ev["student_id"])
                alert_id = db.save_alert(ev, video_name, shot)
                saved.append(db.get_alert(alert_id))
            self.frame_queue.put(("frame", annotated, saved, i + 1, total, video_name))
            i += 1
        cap.release()
        self.frame_queue.put(("done", self.stop_flag.is_set()))

    def poll_queue(self):
        """Drain the worker queue on the UI thread."""
        try:
            while True:
                msg = self.frame_queue.get_nowait()
                if msg[0] == "frame":
                    _, frame, alerts, idx, total, name = msg
                    self.show_frame(frame)
                    self.progress["value"] = idx / total * 100
                    self.status.config(text=f"Processing {name}  -  frame {idx}/{total}")
                    for a in alerts:
                        self.session_alerts += 1
                        self.add_history_row(a)
                        self.notify(a)
                    self.count_label.config(text=f"Alerts this video: {self.session_alerts}")
                elif msg[0] == "done":
                    stopped = msg[1]
                    self.status.config(text=("Stopped." if stopped else "Finished.") +
                                       f"  {self.session_alerts} alert(s) saved to the database.")
                    self.stop_btn.config(state="disabled")
                elif msg[0] == "error":
                    messagebox.showerror("Error", msg[1])
                    self.stop_btn.config(state="disabled")
        except queue.Empty:
            pass
        self.root.after(15, self.poll_queue)

    def show_frame(self, frame):
        h, w = frame.shape[:2]
        scale = min(VIEW_W / w, VIEW_H / h)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_label.config(image=photo, text="")
        self.video_label.image = photo

    def on_close(self):
        self.stop_flag.set()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = CheatingApp(root)
    if len(sys.argv) > 1:
        root.after(300, lambda: app.start_video(sys.argv[1]))
    root.mainloop()
