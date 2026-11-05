"""
hair_tracker_app.py
-------------------
Hair Progress Tracker - Desktop GUI Application

Features:
  - Capture baseline photos via webcam OR load from file
  - Capture follow-up photos to compare
  - U-Net hair segmentation: shows hair coverage % per photo
  - AKAZE+RANSAC image alignment of follow-up to baseline
  - Visual change map: Red=hair lost, Green=hair gained
  - Quantitative coverage change report
  - Export full comparison PNG report
  - Session history saved to JSON

Usage:
    python app/hair_tracker_app.py --model outputs/models/best_model.pth
"""

import os
import sys
import json
import argparse
import threading
import datetime
import traceback

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox

import torch

# ── Add src/ to path ──────────────────────────────────────────
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.dirname(__file__))

from model import UNet
from predict import preprocess_image, predict_hair_mask, compute_coverage
from utils import overlay_mask
from pipeline_visualizer import TrainingPipelineVisualizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ──────────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────────

IMG_SIZE  = 256
VIEWS     = ["Front", "Left 45deg", "Right 45deg"]

OUTPUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "outputs",
                 "results", "comparison_outputs"))
SESSIONS_FILE = os.path.join(OUTPUT_DIR, "sessions.json")


# ──────────────────────────────────────────────────────────────
#  Hierarchical Multi-Tier Registration & Haar Cascades
# ──────────────────────────────────────────────────────────────

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def hierarchical_align(base_rgb: np.ndarray, query_rgb: np.ndarray) -> tuple:
    """
    Hierarchical multi-tier image registration engine:
    1. Tier 1: Rigid Facial Biometric Alignment (Haar Cascade detection of face centers & scale)
    2. Tier 2: AKAZE + RANSAC Feature Matching (for vertex/scalp-only views without visible face)
    3. Tier 3: Clean Fallback with True Unwarped SSIM Calculation
    Returns: (aligned_rgb, method_name, ssim_val, align_ok)
    """
    from skimage.metrics import structural_similarity
    h, w = IMG_SIZE, IMG_SIZE
    base_rs  = cv2.resize(base_rgb,  (w, h))
    query_rs = cv2.resize(query_rgb, (w, h))

    base_g  = cv2.cvtColor(base_rs,  cv2.COLOR_RGB2GRAY)
    query_g = cv2.cvtColor(query_rs, cv2.COLOR_RGB2GRAY)

    try:
        raw_ssim, _ = structural_similarity(base_g, query_g, full=True)
        raw_ssim = max(0.0, float(raw_ssim))
    except Exception:
        raw_ssim = 0.50

    # Tier 1: Facial Biometric Alignment
    def get_primary_face(gray):
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            return None
        return max(faces, key=lambda f: f[2] * f[3])

    f1 = get_primary_face(base_g)
    f2 = get_primary_face(query_g)

    if f1 is not None and f2 is not None:
        cx1, cy1 = f1[0] + f1[2] / 2.0, f1[1] + f1[3] / 2.0
        cx2, cy2 = f2[0] + f2[2] / 2.0, f2[1] + f2[3] / 2.0
        scale = float((f1[2] + f1[3]) / (f2[2] + f2[3]))

        if 0.5 <= scale <= 2.0:
            tx = cx1 - cx2 * scale
            ty = cy1 - cy2 * scale
            M = np.array([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float32)
            aligned_face = cv2.warpAffine(query_rs, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

            try:
                ssim_face, _ = structural_similarity(base_g, cv2.cvtColor(aligned_face, cv2.COLOR_RGB2GRAY), full=True)
                ssim_face = max(0.0, float(ssim_face))
            except Exception:
                ssim_face = raw_ssim

            if ssim_face >= raw_ssim * 0.85:
                return aligned_face, "Facial Biometric Alignment", ssim_face, True

    # Tier 2: AKAZE + RANSAC Homography
    try:
        akaze = cv2.AKAZE_create()
        kp1, d1 = akaze.detectAndCompute(base_g,  None)
        kp2, d2 = akaze.detectAndCompute(query_g, None)

        if d1 is not None and d2 is not None and len(kp1) >= 8 and len(kp2) >= 8:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = sorted(matcher.match(d1, d2), key=lambda m: m.distance)
            good    = matches[:max(8, int(len(matches) * 0.25))]

            if len(good) >= 8:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

                if H is not None and mask is not None:
                    inliers = int(mask.sum())
                    det = float(np.linalg.det(H[:2, :2]))
                    if inliers >= 8 and 0.4 <= det <= 2.5:
                        aligned_akaze = cv2.warpPerspective(query_rs, H, (w, h), borderMode=cv2.BORDER_REFLECT_101)
                        if np.mean(aligned_akaze) >= 5.0:
                            try:
                                ssim_akaze, _ = structural_similarity(base_g, cv2.cvtColor(aligned_akaze, cv2.COLOR_RGB2GRAY), full=True)
                                ssim_akaze = max(0.0, float(ssim_akaze))
                            except Exception:
                                ssim_akaze = raw_ssim
                            return aligned_akaze, f"AKAZE Homography ({inliers} inliers)", ssim_akaze, True
    except Exception:
        pass

    # Tier 3: Reference Frame (Unaligned)
    return query_rs, "Reference Frame (Unaligned)", raw_ssim, False


def akaze_align(base_rgb: np.ndarray, query_rgb: np.ndarray) -> np.ndarray:
    """Legacy wrapper returning aligned RGB array."""
    aligned, _, _, _ = hierarchical_align(base_rgb, query_rgb)
    return aligned


# ──────────────────────────────────────────────────────────────
#  Session helpers
# ──────────────────────────────────────────────────────────────

def load_sessions() -> dict:
    if os.path.exists(SESSIONS_FILE):
        with open(SESSIONS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_sessions(sessions: dict):
    os.makedirs(os.path.dirname(SESSIONS_FILE), exist_ok=True)
    with open(SESSIONS_FILE, "w") as f:
        json.dump(sessions, f, indent=2)


# ──────────────────────────────────────────────────────────────
#  Image display helpers
# ──────────────────────────────────────────────────────────────

def pil_to_ctk_image(pil_img: Image.Image,
                     width: int, height: int) -> ctk.CTkImage:
    """Converts a PIL image to a CTkImage for display in labels."""
    pil_resized = pil_img.resize((width, height), Image.LANCZOS)
    return ctk.CTkImage(light_image=pil_resized, dark_image=pil_resized,
                        size=(width, height))


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        return Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8), "L")
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def make_placeholder(width: int, height: int,
                     text: str = "No Image") -> ctk.CTkImage:
    """Creates a dark grey placeholder image with centered text."""
    img = Image.new("RGB", (width, height), color=(45, 45, 45))
    draw = ImageDraw.Draw(img)
    draw.text((width // 2 - len(text) * 3, height // 2 - 8),
              text, fill=(120, 120, 120))
    return ctk.CTkImage(light_image=img, dark_image=img,
                        size=(width, height))


# ──────────────────────────────────────────────────────────────
#  Main App
# ──────────────────────────────────────────────────────────────

class HairTrackerApp(ctk.CTk):

    PW = 280   # panel image width
    PH = 280   # panel image height

    def __init__(self, model_path: str):
        super().__init__()

        # ── App settings ──────────────────────────────────────
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.title("Hair Progress Tracker  |  DL Capstone")
        self.geometry("1280x820")
        self.minsize(1100, 720)

        # ── State variables ───────────────────────────────────
        self.model_path  = model_path
        self.model       = None
        self.device      = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.cap          = None
        self.live_running = False
        self.sessions     = load_sessions()
        self.current_view = tk.StringVar(value="Front")

        # Image data  (numpy RGB uint8)
        self.baseline_rgb  = None
        self.baseline_mask = None  # numpy (H,W) float 0/1
        self.current_rgb   = None
        self.current_mask  = None
        self.aligned_rgb   = None
        self.aligned_mask  = None

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # ── Build UI ──────────────────────────────────────────
        self._build_ui()

        # ── Load model in background ──────────────────────────
        threading.Thread(target=self._load_model, daemon=True).start()

    # ──────────────────────────────────────────────────────────
    #  UI Construction
    # ──────────────────────────────────────────────────────────

    def _build_ui(self):
        # Top status and action bar with Visualize button in corner
        top_bar = ctk.CTkFrame(self, fg_color=("gray78", "gray22"), height=36, corner_radius=0)
        top_bar.pack(fill="x", padx=4, pady=(4, 0))

        self.status_var = tk.StringVar(value="  Loading model...")
        self.status_bar = ctk.CTkLabel(
            top_bar, textvariable=self.status_var,
            anchor="w", font=ctk.CTkFont(size=12))
        self.status_bar.pack(side="left", fill="x", expand=True, padx=(8, 4))

        # "Visualize" button in the corner
        self.btn_visualize_top = ctk.CTkButton(
            top_bar, text="  Visualize Training Pipeline  ",
            command=self._open_visualizer,
            fg_color="#1f538d", hover_color="#14375e", height=28,
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.btn_visualize_top.pack(side="right", padx=(4, 8), pady=4)

        # Root paned layout: left controls | right panels
        root_frame = ctk.CTkFrame(self, fg_color="transparent")
        root_frame.pack(fill="both", expand=True, padx=6, pady=6)
        root_frame.columnconfigure(0, weight=0)
        root_frame.columnconfigure(1, weight=1)
        root_frame.rowconfigure(0, weight=1)

        self._build_left(root_frame)
        self._build_right(root_frame)

    def _open_visualizer(self):
        if hasattr(self, "vis_window") and self.vis_window is not None and self.vis_window.winfo_exists():
            self.vis_window.focus()
            return
        self.vis_window = TrainingPipelineVisualizer(master=self)

    # ── Left control panel ─────────────────────────────────────
    def _build_left(self, parent):
        left = ctk.CTkFrame(parent, width=300)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 6))
        left.pack_propagate(False)

        # Title
        ctk.CTkLabel(
            left, text="Hair Progress Tracker",
            font=ctk.CTkFont(size=15, weight="bold")
        ).pack(pady=(14, 2))
        ctk.CTkLabel(
            left, text="DL Capstone | U-Net Segmentation",
            font=ctk.CTkFont(size=10), text_color="gray60"
        ).pack(pady=(0, 6))

        # Secondary Visualize button in left panel
        ctk.CTkButton(
            left, text="  Visualize Training Flow  ",
            command=self._open_visualizer,
            fg_color="#234a70", hover_color="#183652", height=30,
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(fill="x", padx=14, pady=(0, 8))

        self._divider(left)

        # View selector
        ctk.CTkLabel(left, text="Select View",
                     font=ctk.CTkFont(size=12, weight="bold")).pack(
            anchor="w", padx=14, pady=(8, 2))
        vf = ctk.CTkFrame(left, fg_color="transparent")
        vf.pack(fill="x", padx=14, pady=2)
        for v in VIEWS:
            ctk.CTkRadioButton(
                vf, text=v, variable=self.current_view, value=v,
                command=self._on_view_change
            ).pack(anchor="w", pady=2)

        self._divider(left)

        # Camera preview
        ctk.CTkLabel(left, text="Live Camera Preview",
                     font=ctk.CTkFont(size=12, weight="bold")).pack(
            anchor="w", padx=14, pady=(8, 2))

        placeholder = make_placeholder(264, 200, "Camera Off")
        self.cam_label = ctk.CTkLabel(left, image=placeholder, text="")
        self.cam_label.pack(padx=14, pady=4)
        self.cam_label.image = placeholder

        cam_btns = ctk.CTkFrame(left, fg_color="transparent")
        cam_btns.pack(fill="x", padx=14, pady=4)
        cam_btns.columnconfigure(0, weight=1)
        cam_btns.columnconfigure(1, weight=1)

        self.btn_cam_start = ctk.CTkButton(
            cam_btns, text="Start Camera",
            command=self._start_camera, height=30)
        self.btn_cam_start.grid(row=0, column=0, padx=(0, 3), sticky="ew")

        self.btn_cam_stop = ctk.CTkButton(
            cam_btns, text="Stop", state="disabled",
            command=self._stop_camera, height=30, fg_color="gray40")
        self.btn_cam_stop.grid(row=0, column=1, padx=(3, 0), sticky="ew")

        self._divider(left)

        # Capture buttons
        ctk.CTkLabel(left, text="Capture / Load",
                     font=ctk.CTkFont(size=12, weight="bold")).pack(
            anchor="w", padx=14, pady=(8, 4))

        ctk.CTkButton(
            left, text="Set as BASELINE (Before / Day 0)",
            command=self._capture_baseline,
            fg_color="#1a5e35", hover_color="#144d2b", height=36
        ).pack(fill="x", padx=14, pady=3)

        ctk.CTkButton(
            left, text="Capture NEW Photo (After / Follow-up)",
            command=self._capture_current,
            fg_color="#1a3e6e", hover_color="#142f57", height=36
        ).pack(fill="x", padx=14, pady=3)

        ctk.CTkButton(
            left, text="Load BASELINE from File (Before)",
            command=lambda: self._load_file("baseline"),
            height=32
        ).pack(fill="x", padx=14, pady=3)

        ctk.CTkButton(
            left, text="Load NEW Photo from File (After)",
            command=lambda: self._load_file("current"),
            height=32
        ).pack(fill="x", padx=14, pady=3)

        self._divider(left)

        # Analysis + Save
        self.btn_analyze = ctk.CTkButton(
            left, text="Run Analysis",
            command=self._run_analysis,
            fg_color="#6e1a1a", hover_color="#5a1414", height=44,
            font=ctk.CTkFont(size=14, weight="bold"))
        self.btn_analyze.pack(fill="x", padx=14, pady=(8, 4))

        self.btn_save = ctk.CTkButton(
            left, text="Export Report PNG",
            command=self._save_report, height=34)
        self.btn_save.pack(fill="x", padx=14, pady=4)

        ctk.CTkButton(
            left, text="Clear All",
            command=self._clear_all,
            fg_color="gray30", hover_color="gray25", height=30
        ).pack(fill="x", padx=14, pady=(2, 12))

    def _divider(self, parent):
        ctk.CTkFrame(parent, height=2, fg_color="gray35").pack(
            fill="x", padx=14, pady=6)

    # ── Right display panels ───────────────────────────────────
    def _build_right(self, parent):
        right = ctk.CTkFrame(parent, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.columnconfigure(1, weight=1)
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        # Baseline panel (top-left)
        bp = ctk.CTkFrame(right)
        bp.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        ctk.CTkLabel(bp, text="BASELINE (Before / Day 0)",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color="#5fba5f").pack(pady=(10, 2))

        ph = make_placeholder(self.PW, self.PH, "No Baseline")
        self.lbl_base = ctk.CTkLabel(bp, image=ph, text="")
        self.lbl_base.pack(padx=8, pady=4)
        self.lbl_base.image = ph

        self.lbl_base_cov = ctk.CTkLabel(
            bp, text="Coverage: --",
            font=ctk.CTkFont(size=13))
        self.lbl_base_cov.pack(pady=(2, 10))

        # Current photo panel (top-right)
        cp = ctk.CTkFrame(right)
        cp.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
        ctk.CTkLabel(cp, text="CURRENT / ALIGNED (After / Follow-up)",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color="#5f9fda").pack(pady=(10, 2))

        ph2 = make_placeholder(self.PW, self.PH, "No Current Photo")
        self.lbl_cur = ctk.CTkLabel(cp, image=ph2, text="")
        self.lbl_cur.pack(padx=8, pady=4)
        self.lbl_cur.image = ph2

        self.lbl_cur_cov = ctk.CTkLabel(
            cp, text="Coverage: --",
            font=ctk.CTkFont(size=13))
        self.lbl_cur_cov.pack(pady=(2, 10))

        # Results panel (bottom, spans both columns)
        rp = ctk.CTkFrame(right)
        rp.grid(row=1, column=0, columnspan=2, sticky="nsew",
                padx=4, pady=(0, 4))
        rp.columnconfigure(0, weight=1)
        rp.columnconfigure(1, weight=2)
        rp.rowconfigure(0, weight=1)

        # Change map
        map_frame = ctk.CTkFrame(rp)
        map_frame.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)
        ctk.CTkLabel(map_frame, text="Change Map (Overlaid on Scalp)",
                     font=ctk.CTkFont(size=12, weight="bold")).pack(pady=(8, 2))

        ph3 = make_placeholder(self.PW, self.PH, "Run Analysis First")
        self.lbl_map = ctk.CTkLabel(map_frame, image=ph3, text="")
        self.lbl_map.pack(padx=8, pady=4)
        self.lbl_map.image = ph3

        ctk.CTkLabel(map_frame,
                     text="Red = Hair Lost (Receded)  |  Green = Hair Gained (Regrown)",
                     font=ctk.CTkFont(size=10, weight="bold"), text_color="#cbd5e1").pack(pady=(0, 8))

        # Metrics
        metrics_frame = ctk.CTkFrame(rp)
        metrics_frame.grid(row=0, column=1, sticky="nsew",
                           padx=(4, 8), pady=8)
        ctk.CTkLabel(metrics_frame, text="Analysis Results",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(pady=(12, 6))

        self.lbl_result_main = ctk.CTkLabel(
            metrics_frame, text="--",
            font=ctk.CTkFont(size=32, weight="bold"),
            text_color="gray60")
        self.lbl_result_main.pack(pady=4)

        self.lbl_result_sub = ctk.CTkLabel(
            metrics_frame, text="Run Analysis to see results",
            font=ctk.CTkFont(size=12), text_color="gray60")
        self.lbl_result_sub.pack(pady=2)

        self._divider(metrics_frame)

        labels = [
            ("Baseline Coverage", "lbl_m_base"),
            ("Current Coverage",  "lbl_m_cur"),
            ("Absolute Change",   "lbl_m_abs"),
            ("Relative Change",   "lbl_m_rel"),
            ("Changed Area",      "lbl_m_area"),
            ("Alignment SSIM",    "lbl_m_ssim"),
        ]
        for title, attr in labels:
            row = ctk.CTkFrame(metrics_frame, fg_color="transparent")
            row.pack(fill="x", padx=20, pady=2)
            ctk.CTkLabel(row, text=f"{title}:",
                         font=ctk.CTkFont(size=11),
                         text_color="gray60", width=160, anchor="w").pack(side="left")
            lbl = ctk.CTkLabel(row, text="--",
                               font=ctk.CTkFont(size=11, weight="bold"),
                               anchor="w")
            lbl.pack(side="left")
            setattr(self, attr, lbl)

        self._divider(metrics_frame)

        ctk.CTkLabel(metrics_frame, text="Session History",
                     font=ctk.CTkFont(size=11, weight="bold")).pack(
            anchor="w", padx=20, pady=(4, 2))
        self.lbl_history = ctk.CTkLabel(
            metrics_frame, text="No sessions recorded yet.",
            font=ctk.CTkFont(size=10), text_color="gray60",
            justify="left", wraplength=340)
        self.lbl_history.pack(anchor="w", padx=20, pady=(0, 10))
        self._refresh_history()

    # ──────────────────────────────────────────────────────────
    #  Model loading
    # ──────────────────────────────────────────────────────────

    def _load_model(self):
        try:
            self._status("Loading U-Net model...")
            checkpoint = torch.load(self.model_path,
                                    map_location=self.device)
            model = UNet(in_channels=3, out_channels=1,
                         features=[32, 64, 128, 256]).to(self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            self.model = model
            epoch = checkpoint.get("epoch", "?")
            dice  = checkpoint.get("val_dice", "?")
            dice_str = f"{dice:.4f}" if isinstance(dice, float) else str(dice)
            self._status(
                f"Model ready  |  Epoch {epoch}  |  Val Dice {dice_str}"
                f"  |  Device: {self.device}")
        except Exception as e:
            self._status(f"ERROR loading model: {e}")

    # ──────────────────────────────────────────────────────────
    #  Camera
    # ──────────────────────────────────────────────────────────

    def _start_camera(self):
        if self.cap is not None:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror(
                "Camera Error",
                "Cannot open webcam.\n"
                "Please use 'Load from File' buttons instead.")
            self.cap = None
            return
        self.live_running = True
        self.btn_cam_start.configure(state="disabled")
        self.btn_cam_stop.configure(state="normal")
        self._update_camera()

    def _stop_camera(self):
        self.live_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        ph = make_placeholder(264, 200, "Camera Off")
        self.cam_label.configure(image=ph, text="")
        self.cam_label.image = ph
        self.btn_cam_start.configure(state="normal")
        self.btn_cam_stop.configure(state="disabled")

    def _update_camera(self):
        if not self.live_running or self.cap is None:
            return
        ret, frame = self.cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            ctk_img = pil_to_ctk_image(pil, 264, 200)
            self.cam_label.configure(image=ctk_img, text="")
            self.cam_label.image = ctk_img
        self.after(33, self._update_camera)   # ~30 fps

    def _grab_frame(self) -> np.ndarray | None:
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    # ──────────────────────────────────────────────────────────
    #  Capture / Load
    # ──────────────────────────────────────────────────────────

    def _capture_baseline(self):
        arr = self._grab_frame()
        if arr is None:
            messagebox.showwarning("No Frame",
                                   "Start camera first, or use 'Load from File'.")
            return
        self._process_image(arr, "baseline")

    def _capture_current(self):
        arr = self._grab_frame()
        if arr is None:
            messagebox.showwarning("No Frame",
                                   "Start camera first, or use 'Load from File'.")
            return
        self._process_image(arr, "current")

    def _load_file(self, target: str):
        path = filedialog.askopenfilename(
            title=f"Select {'Baseline' if target == 'baseline' else 'New'} Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")])
        if not path:
            return
        arr = np.array(Image.open(path).convert("RGB"))
        self._process_image(arr, target)

    def _process_image(self, rgb_arr: np.ndarray, target: str):
        label = "Baseline" if target == "baseline" else "Current"
        self._status(f"Running segmentation on {label}...")

        def worker():
            try:
                mask, cov = self._segment(rgb_arr)
                if target == "baseline":
                    self.baseline_rgb  = rgb_arr
                    self.baseline_mask = mask
                    self.after(0, lambda: self._show_panel(
                        "base", rgb_arr, mask, cov))
                    self._status(
                        f"Baseline set  |  Hair coverage: {cov:.1f}%")
                else:
                    self.current_rgb  = rgb_arr
                    self.current_mask = mask
                    self.after(0, lambda: self._show_panel(
                        "cur", rgb_arr, mask, cov))
                    self._status(
                        f"Current photo set  |  Hair coverage: {cov:.1f}%")
            except Exception as e:
                self._status(f"ERROR in segmentation: {e}")
                traceback.print_exc()

        threading.Thread(target=worker, daemon=True).start()

    # ──────────────────────────────────────────────────────────
    #  Segmentation
    # ──────────────────────────────────────────────────────────

    def _segment(self, rgb_arr: np.ndarray) -> tuple:
        """Runs U-Net with Test-Time Augmentation (TTA). Returns (mask, coverage%)."""
        if self.model is None:
            raise RuntimeError("Model not loaded yet.")

        tmp = os.path.join(OUTPUT_DIR, "_tmp_input.png")
        Image.fromarray(rgb_arr.astype(np.uint8)).save(tmp)
        tensor, _ = preprocess_image(tmp, size=IMG_SIZE)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            logit1 = self.model(tensor)
            prob1 = torch.sigmoid(logit1)
            tensor_flip = torch.flip(tensor, dims=[3])
            logit2 = self.model(tensor_flip)
            prob2 = torch.flip(torch.sigmoid(logit2), dims=[3])
            prob = (0.5 * (prob1 + prob2)).squeeze().cpu().numpy()
            mask = (prob >= 0.5).astype(np.float32)

        cov = compute_coverage(mask)
        return mask, cov

    # ──────────────────────────────────────────────────────────
    #  Display helpers
    # ──────────────────────────────────────────────────────────

    def _show_panel(self, which: str, rgb: np.ndarray,
                    mask: np.ndarray, cov: float):
        """Updates the baseline or current panel with overlay image."""
        resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        ovl     = overlay_mask(resized, mask,
                               color=(80, 220, 80) if which == "base"
                               else (80, 150, 220),
                               alpha=0.40)
        ctk_img = pil_to_ctk_image(numpy_to_pil(ovl), self.PW, self.PH)
        if which == "base":
            self.lbl_base.configure(image=ctk_img, text="")
            self.lbl_base.image = ctk_img
            self.lbl_base_cov.configure(
                text=f"Hair Coverage: {cov:.1f}%",
                text_color="#5fba5f")
        else:
            self.lbl_cur.configure(image=ctk_img, text="")
            self.lbl_cur.image = ctk_img
            self.lbl_cur_cov.configure(
                text=f"Hair Coverage: {cov:.1f}%",
                text_color="#5f9fda")

    def _set_metric_labels(self, base_cov, cur_cov,
                           change, change_pct, changed_area, ssim_val):
        direction  = "INCREASE (Growth)" if change >= 0 else "DECREASE (Loss)"
        arrow      = "+" if change >= 0 else ""
        color      = "#5fba5f" if change >= 0 else "#e05555"

        self.lbl_result_main.configure(
            text=f"{arrow}{change:.1f}%", text_color=color)
        self.lbl_result_sub.configure(
            text=f"Hair {direction} detected",
            text_color=color)

        self.lbl_m_base.configure(text=f"{base_cov:.2f}%")
        self.lbl_m_cur.configure(text=f"{cur_cov:.2f}%")
        self.lbl_m_abs.configure(
            text=f"{arrow}{change:.2f}% ({direction})", text_color=color)
        self.lbl_m_rel.configure(
            text=f"{arrow}{change_pct:.1f}%", text_color=color)
        self.lbl_m_area.configure(text=f"{changed_area:.2f}%")
        self.lbl_m_ssim.configure(text=f"{ssim_val:.4f}")

    # ──────────────────────────────────────────────────────────
    #  Analysis
    # ──────────────────────────────────────────────────────────

    def _run_analysis(self):
        if self.baseline_rgb is None or self.current_rgb is None:
            messagebox.showwarning(
                "Missing Images",
                "Please set BOTH a Baseline and a Current photo first.")
            return
        if self.model is None:
            messagebox.showwarning("Model Not Ready",
                                   "Model is still loading. Please wait.")
            return

        self._status("Running hierarchical alignment and TTA hair analysis...")
        self.btn_analyze.configure(state="disabled", text="Analyzing...")

        def worker():
            try:
                # Step 1: Hierarchical registration (Biometric face -> AKAZE -> Reference)
                aligned_rgb, reg_method, ssim_val, align_ok = hierarchical_align(self.baseline_rgb, self.current_rgb)
                self.aligned_rgb = aligned_rgb

                # Step 2: Segment aligned image with TTA
                aligned_mask, cur_cov = self._segment(aligned_rgb)
                self.aligned_mask = aligned_mask

                # Step 3: Metrics
                base_cov  = compute_coverage(self.baseline_mask)
                change    = cur_cov - base_cov
                change_pct = (change / base_cov * 100) if base_cov > 1e-6 else 0.0

                xor_map = np.logical_xor(
                    self.baseline_mask > 0, aligned_mask > 0).astype(np.uint8)
                changed_area = float(xor_map.sum() / xor_map.size * 100)

                # Step 4: Refine SSIM estimate
                try:
                    from skimage.metrics import structural_similarity
                    base_gray = cv2.cvtColor(
                        cv2.resize(self.baseline_rgb, (IMG_SIZE, IMG_SIZE)),
                        cv2.COLOR_RGB2GRAY)
                    cur_gray  = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2GRAY)
                    calc_ssim, _ = structural_similarity(
                        base_gray, cur_gray, full=True)
                    ssim_val = max(0.0, float(calc_ssim))
                except Exception:
                    pass  # keep ssim_val from hierarchical_align

                # Step 5: Build change map overlaid on scalp image
                gray_face = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2GRAY)
                dim_face = cv2.cvtColor(gray_face, cv2.COLOR_GRAY2RGB) // 2  # 50% dimmed grayscale
                change_map = dim_face.copy()

                lost_mask   = np.logical_and(
                    xor_map > 0, self.baseline_mask > 0)
                gained_mask = np.logical_and(
                    xor_map > 0, aligned_mask > 0)
                change_map[lost_mask]   = [230, 60,  60]   # Bright Red   = Lost
                change_map[gained_mask] = [50,  230, 70]   # Bright Green = Gained

                # Update UI on main thread
                self.after(0, lambda: self._apply_analysis_results(
                    base_cov, cur_cov, change, change_pct,
                    changed_area, ssim_val, change_map,
                    aligned_rgb, aligned_mask))

            except Exception as e:
                self._status(f"ERROR in analysis: {e}")
                traceback.print_exc()
            finally:
                self.after(0, lambda: self.btn_analyze.configure(
                    state="normal", text="Run Analysis"))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_analysis_results(self, base_cov, cur_cov, change,
                                 change_pct, changed_area, ssim_val,
                                 change_map, aligned_rgb, aligned_mask):
        # Update current panel with aligned image
        self._show_panel("cur", aligned_rgb, aligned_mask, cur_cov)

        # Update change map panel
        ctk_map = pil_to_ctk_image(numpy_to_pil(change_map),
                                   self.PW, self.PH)
        self.lbl_map.configure(image=ctk_map, text="")
        self.lbl_map.image = ctk_map

        # Update metric labels
        self._set_metric_labels(base_cov, cur_cov, change,
                                change_pct, changed_area, ssim_val)

        # Save session
        view     = self.current_view.get()
        date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        if view not in self.sessions:
            self.sessions[view] = []
        self.sessions[view].append({
            "date": date_str,
            "base_cov":     round(base_cov, 2),
            "cur_cov":      round(cur_cov, 2),
            "change":       round(change, 2),
            "changed_area": round(changed_area, 2),
        })
        save_sessions(self.sessions)
        self._refresh_history()

        arrow = "+" if change >= 0 else ""
        self._status(
            f"Analysis complete  |  Change: {arrow}{change:.2f}%"
            f"  |  Changed area: {changed_area:.1f}%"
            f"  |  View: {view}")

    # ──────────────────────────────────────────────────────────
    #  Session history
    # ──────────────────────────────────────────────────────────

    def _refresh_history(self):
        lines = []
        for view, entries in self.sessions.items():
            for e in entries[-3:]:   # show last 3 per view
                arrow = "+" if e["change"] >= 0 else ""
                lines.append(
                    f"{e['date']}  [{view}]  "
                    f"{arrow}{e['change']:.1f}%")
        if lines:
            self.lbl_history.configure(
                text="\n".join(lines[-6:]))  # show max 6 lines
        else:
            self.lbl_history.configure(
                text="No sessions recorded yet.")

    def _on_view_change(self):
        self._status(f"View set to: {self.current_view.get()}")

    # ──────────────────────────────────────────────────────────
    #  Export report
    # ──────────────────────────────────────────────────────────

    def _save_report(self):
        if self.baseline_rgb is None or self.aligned_rgb is None:
            messagebox.showwarning("Nothing to Save",
                                   "Run analysis first.")
            return

        path = filedialog.asksaveasfilename(
            title="Save Report",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png")],
            initialfile=(
                f"hair_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.png"))
        if not path:
            return

        try:
            base_cov = compute_coverage(self.baseline_mask)
            cur_cov  = compute_coverage(self.aligned_mask)
            change   = cur_cov - base_cov

            base_rs  = cv2.resize(self.baseline_rgb,  (IMG_SIZE, IMG_SIZE))
            align_rs = cv2.resize(self.aligned_rgb,   (IMG_SIZE, IMG_SIZE))

            base_ovl  = overlay_mask(base_rs,  self.baseline_mask,
                                     color=(80, 220, 80),  alpha=0.40)
            align_ovl = overlay_mask(align_rs, self.aligned_mask,
                                     color=(80, 150, 220), alpha=0.40)

            xor_map = np.logical_xor(
                self.baseline_mask > 0, self.aligned_mask > 0)
            gray_face = cv2.cvtColor(align_rs, cv2.COLOR_RGB2GRAY)
            dim_face = cv2.cvtColor(gray_face, cv2.COLOR_GRAY2RGB) // 2
            cmap = dim_face.copy()
            cmap[np.logical_and(xor_map, self.baseline_mask > 0)] = [230, 60, 60]
            cmap[np.logical_and(xor_map, self.aligned_mask  > 0)] = [50, 230, 70]

            view = self.current_view.get()
            date = datetime.datetime.now().strftime("%Y-%m-%d")
            arrow = "+" if change >= 0 else ""
            direction = "INCREASE (Growth)" if change >= 0 else "DECREASE (Loss)"

            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.patch.set_facecolor("#1a1a2e")
            fig.suptitle(
                f"Hair Progress Report   |   {view}   |   {date}",
                fontsize=16, fontweight="bold", color="white", y=0.98)

            panels = [
                (base_rs,   f"Baseline Photo\nCoverage: {base_cov:.1f}%"),
                (align_rs,  f"Current Photo (Aligned)\nCoverage: {cur_cov:.1f}%"),
                (cmap,      "Change Map (Overlaid on Scalp)\nRed=Hair Lost | Green=Hair Gained"),
                (base_ovl,  "Baseline Mask Overlay"),
                (align_ovl, "Current Mask Overlay"),
                (None,      f"Summary\n\nChange: {arrow}{change:.1f}%\n"
                            f"({direction})\n\n"
                            f"Baseline: {base_cov:.1f}%\n"
                            f"Current:  {cur_cov:.1f}%\n\n"
                            f"Model: U-Net 7.8M params\n"
                            f"Dice: 95.19%"),
            ]

            for ax, (img, title) in zip(axes.flatten(), panels):
                ax.set_facecolor("#1a1a2e")
                if img is None:
                    ax.text(0.5, 0.5, title, ha="center", va="center",
                            fontsize=13, color="white",
                            transform=ax.transAxes,
                            linespacing=1.8)
                else:
                    ax.imshow(img)
                ax.set_title(title, fontsize=11, color="white", pad=8)
                ax.axis("off")

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            messagebox.showinfo("Saved", f"Report saved:\n{path}")
            self._status(f"Report exported to: {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save report:\n{e}")
            traceback.print_exc()

    # ──────────────────────────────────────────────────────────
    #  Clear all
    # ──────────────────────────────────────────────────────────

    def _clear_all(self):
        self.baseline_rgb  = None
        self.baseline_mask = None
        self.current_rgb   = None
        self.current_mask  = None
        self.aligned_rgb   = None
        self.aligned_mask  = None

        ph = make_placeholder(self.PW, self.PH, "No Baseline")
        self.lbl_base.configure(image=ph, text="")
        self.lbl_base.image = ph
        self.lbl_base_cov.configure(text="Coverage: --",
                                    text_color="gray60")

        ph2 = make_placeholder(self.PW, self.PH, "No Current Photo")
        self.lbl_cur.configure(image=ph2, text="")
        self.lbl_cur.image = ph2
        self.lbl_cur_cov.configure(text="Coverage: --",
                                   text_color="gray60")

        ph3 = make_placeholder(self.PW, self.PH, "Run Analysis First")
        self.lbl_map.configure(image=ph3, text="")
        self.lbl_map.image = ph3

        self.lbl_result_main.configure(text="--", text_color="gray60")
        self.lbl_result_sub.configure(
            text="Run Analysis to see results", text_color="gray60")
        for attr in ("lbl_m_base", "lbl_m_cur", "lbl_m_abs",
                     "lbl_m_rel", "lbl_m_area", "lbl_m_ssim"):
            getattr(self, attr).configure(text="--", text_color="white")

        self._status("Cleared. Load new images to start.")

    # ──────────────────────────────────────────────────────────
    #  Status bar
    # ──────────────────────────────────────────────────────────

    def _status(self, msg: str):
        self.after(0, lambda: self.status_var.set(f"  {msg}"))

    # ──────────────────────────────────────────────────────────
    #  Close
    # ──────────────────────────────────────────────────────────

    def on_close(self):
        self._stop_camera()
        self.destroy()


# ──────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Hair Progress Tracker")
    parser.add_argument(
        "--model", type=str,
        default=os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "outputs", "models", "best_model.pth")))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_path = os.path.abspath(args.model)

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at: {model_path}")
        sys.exit(1)

    app = HairTrackerApp(model_path=model_path)
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
