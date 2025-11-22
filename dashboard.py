"""
Interactive GPU-accelerated Shape Detection Dashboard
Features:
 - Webcam real-time detection (YOLOv8) using GPU
 - Hand-draw canvas: draw shapes with mouse and predict them
 - FPS counter and GPU usage display
 - File upload for videos and images
 - Optional voice feedback (fixed)
"""

import os
import threading
import time
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
from tkinter import filedialog, messagebox
import GPUtil
import pyttsx3

# ================== CONFIG ==================
MODEL_PATH = r"C:\Coding\GPU Project\shape_train_main\weights\best.pt"  # <- UPDATE if needed
CAM_INDEX = 0
CONFIDENCE = 0.25
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600
VOICE_FEEDBACK = True
SAVE_DIR = os.path.join(os.getcwd(), "dashboard_outputs")
os.makedirs(SAVE_DIR, exist_ok=True)
# ============================================

print("Loading model:", MODEL_PATH)
model = YOLO(MODEL_PATH)

# --- FIXED: reliable TTS ---
def speak_async(text):
    def _speak():
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 170)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print("TTS error:", e)
    threading.Thread(target=_speak, daemon=True).start()


# ========== Main App ==========
class DashboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive Shape Detection Dashboard")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.cap = None
        self.running = False
        self.frame = None
        self.annotated = None
        self.last_time = 0
        self.fps = 0.0
        self.voice = VOICE_FEEDBACK

        self.build_gui()
        self.update_gpu_stats()

    def build_gui(self):
        self.video_panel = tk.Label(self.root)
        self.video_panel.grid(row=0, column=0, rowspan=8, padx=5, pady=5)

        btn_start = tk.Button(self.root, text="Start Webcam", command=self.start_webcam, width=20)
        btn_start.grid(row=0, column=1, padx=5, pady=4)

        btn_stop = tk.Button(self.root, text="Stop Webcam", command=self.stop_webcam, width=20)
        btn_stop.grid(row=1, column=1, padx=5, pady=4)

        btn_draw = tk.Button(self.root, text="Open Draw Pad", command=self.open_draw_pad, width=20)
        btn_draw.grid(row=2, column=1, padx=5, pady=4)

        btn_open_video = tk.Button(self.root, text="Open Video File", command=self.open_video_file, width=20)
        btn_open_video.grid(row=3, column=1, padx=5, pady=4)

        btn_open_image = tk.Button(self.root, text="Upload Image", command=self.open_image_file, width=20)
        btn_open_image.grid(row=4, column=1, padx=5, pady=4)

        self.voice_btn = tk.Button(self.root, text=f"Voice: {'ON' if self.voice else 'OFF'}", command=self.toggle_voice, width=20)
        self.voice_btn.grid(row=5, column=1, padx=5, pady=4)

        self.info_text = tk.StringVar()
        self.info_text.set("FPS: 0.00 | GPU: n/a | VRAM: n/a")
        lbl_info = tk.Label(self.root, textvariable=self.info_text, width=40)
        lbl_info.grid(row=6, column=1, padx=5, pady=4)

        self.log = tk.Text(self.root, height=10, width=40)
        self.log.grid(row=7, column=1, padx=5, pady=4)
        self.log.insert(tk.END, "Ready.\n")

    # Webcam handling
    def start_webcam(self):
        if self.running:
            self.log_insert("Webcam already running.")
            return
        try:
            self.cap = cv2.VideoCapture(CAM_INDEX)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam.")
                return
            self.running = True
            threading.Thread(target=self.webcam_loop, daemon=True).start()
            self.log_insert("Webcam started.")
        except Exception as e:
            self.log_insert(f"Error starting webcam: {e}")

    def stop_webcam(self):
        if not self.running:
            self.log_insert("Webcam not running.")
            return
        self.running = False
        time.sleep(0.2)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.log_insert("Webcam stopped.")

    def webcam_loop(self):
        self.last_time = time.time()
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            start = time.time()
            results = model(rgb, conf=CONFIDENCE)
            annotated = results[0].plot()
            end = time.time()
            cur_fps = 1.0 / (end - start) if (end - start) > 0 else 0.0
            self.fps = 0.85 * self.fps + 0.15 * cur_fps if self.fps else cur_fps
            self.update_image_in_gui(annotated)
        self.log_insert("Webcam loop ended.")

    def update_image_in_gui(self, annotated_rgb):
        img = Image.fromarray(annotated_rgb)
        img = img.resize((DISPLAY_WIDTH, DISPLAY_HEIGHT))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_panel.imgtk = imgtk
        self.video_panel.config(image=imgtk)
        gpu_load, vram = self.get_gpu_stats()
        self.info_text.set(f"FPS: {self.fps:.2f} | GPU Load: {gpu_load:.1f}% | VRAM: {vram}")
        self.root.update_idletasks()

    def open_video_file(self):
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if not path:
            return
        if self.running:
            self.stop_webcam()
        threading.Thread(target=self.process_video_file, args=(path,), daemon=True).start()

    def process_video_file(self, path):
        self.log_insert(f"Processing video: {path}")
        cap = cv2.VideoCapture(path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
        out_path = os.path.join(SAVE_DIR, f"annotated_{os.path.basename(path)}")
        out_writer = cv2.VideoWriter(out_path, fourcc, fps_in, (W, H))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, conf=CONFIDENCE)
            annotated = results[0].plot()
            out_writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            self.update_image_in_gui(annotated)
        cap.release()
        out_writer.release()
        self.log_insert(f"Saved annotated video to: {out_path}")
        if self.voice:
            speak_async("Video processing complete.")

    def open_image_file(self):
        path = filedialog.askopenfilename(
            title="Select an image file",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not path:
            return

        if self.running:
            self.stop_webcam()

        self.log_insert(f"Processing image: {path}")
        frame = cv2.imread(path)
        results = model(frame, conf=CONFIDENCE)
        annotated = results[0].plot()

        # Save annotated image
        out_path = os.path.join(SAVE_DIR, f"annotated_{os.path.basename(path)}")
        cv2.imwrite(out_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        self.log_insert(f"Saved annotated image to: {out_path}")

        # Display result
        self.update_image_in_gui(annotated)

        # Voice feedback
        if len(results[0].boxes) > 0:
            detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls]
            unique_classes = list(set(detected_classes))
            text = f"Detected: {', '.join(unique_classes)}"
            self.log_insert(text)
            if self.voice:
                speak_async(text)
        else:
            self.log_insert("No shapes detected.")
            if self.voice:
                speak_async("No shapes detected.")

    def open_draw_pad(self):
        DrawPad(self)

    def get_gpu_stats(self):
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return 0.0, "n/a"
            g = gpus[0]
            load = g.load * 100
            vram = f"{g.memoryUsed}MB/{g.memoryTotal}MB"
            return load, vram
        except Exception:
            return 0.0, "n/a"

    def update_gpu_stats(self):
        gpu_load, vram = self.get_gpu_stats()
        self.info_text.set(f"FPS: {self.fps:.2f} | GPU Load: {gpu_load:.1f}% | VRAM: {vram}")
        self.root.after(1000, self.update_gpu_stats)

    def toggle_voice(self):
        self.voice = not self.voice
        self.voice_btn.config(text=f"Voice: {'ON' if self.voice else 'OFF'}")
        self.log_insert(f"Voice set to: {'ON' if self.voice else 'OFF'}")

    def log_insert(self, text):
        ts = time.strftime("%H:%M:%S")
        self.log.insert(tk.END, f"[{ts}] {text}\n")
        self.log.see(tk.END)

    def on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()


# ========== Draw Pad ==========
class DrawPad:
    def __init__(self, app):
        self.app = app
        self.pad_win = tk.Toplevel()
        self.pad_win.title("Draw Pad - Draw a shape")
        self.width = 400
        self.height = 400
        self.canvas = tk.Canvas(self.pad_win, width=self.width, height=self.height, bg="white")
        self.canvas.pack()
        self.image = Image.new("RGB", (self.width, self.height), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last_y = None
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
        btn_frame = tk.Frame(self.pad_win)
        btn_frame.pack(fill=tk.X)
        tk.Button(btn_frame, text="Predict Drawing", command=self.predict_drawing).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(btn_frame, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=5, pady=5)

    def paint(self, event):
        x, y = event.x, event.y
        r = 6
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, x, y, width=8, capstyle=tk.ROUND, fill="black")
            self.draw.line([self.last_x, self.last_y, x, y], fill="black", width=8)
        else:
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")
            self.draw.ellipse([x - r, y - r, x + r, y + r], fill="black", outline="black")
        self.last_x = x
        self.last_y = y

    def reset(self, event):
        self.last_x, self.last_y = None, None

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.width, self.height], fill=(255, 255, 255))

    def predict_drawing(self):
        img = np.array(self.image.convert("RGB"))
        self.app.log_insert("Predicting drawn image...")
        results = model(img, conf=CONFIDENCE)
        annotated = results[0].plot()
        win = tk.Toplevel()
        win.title("Prediction Result")
        img_pil = Image.fromarray(annotated).resize((500, 500))
        imgtk = ImageTk.PhotoImage(img_pil)
        lbl = tk.Label(win, image=imgtk)
        lbl.image = imgtk
        lbl.pack()
        boxes = results[0].boxes
        labels = []
        for b in boxes:
            cl = int(b.cls[0])
            conf = float(b.conf[0]) if hasattr(b, "conf") else 0.0
            name = model.names[cl] if cl in model.names else str(cl)
            labels.append(f"{name}: {conf:.2f}")
        if labels:
            txt = "\n".join(labels)
            self.app.log_insert(f"Draw prediction: {txt}")
            if self.app.voice:
                speak_async(txt)
        else:
            self.app.log_insert("No objects detected in drawing.")
            if self.app.voice:
                speak_async("No shapes detected.")


# ========== Run App ==========
if __name__ == "__main__":
    root = tk.Tk()
    app = DashboardApp(root)
    root.mainloop()
