"""Sign Bridge — train static + keyframe signs, live translate."""

from __future__ import annotations

import gc
import json
import threading
import time
from collections import deque
from pathlib import Path
from tkinter import filedialog, colorchooser
import zipfile

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageTk
import tkinter as tk

from camera import HandCamera, list_cameras
from features import POSE_SIZE
from trainer import SignTrainer

try:
    import winsound
except ImportError:
    winsound = None

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

DATA_DIR = Path(__file__).resolve().parent / "data"
EXAMPLES_DIR = Path(__file__).resolve().parent / "signexamples"
SETTINGS_PATH = DATA_DIR / "settings.json"

DARK = {
    "BG": "#050508", "CARD": "#0e0e14", "ACCENT": "#5eead4", "ACCENT_DIM": "#2dd4bf",
    "TEXT": "#e2e8f0", "MUTED": "#94a3b8", "DANGER": "#f43f5e",
}
LIGHT = {
    "BG": "#f1f5f9", "CARD": "#ffffff", "ACCENT": "#0d9488", "ACCENT_DIM": "#14b8a6",
    "TEXT": "#0f172a", "MUTED": "#64748b", "DANGER": "#e11d48",
}

VIDEO_W, VIDEO_H = 480, 360
RECORD_SECONDS = 2.5  # shorter hold for keyframe poses
MAX_CAPTURE = 45
MIN_SAMPLES_WARN = 6


class SignBridgeApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Sign Bridge")
        self.geometry("1280x780")
        self.minsize(1080, 680)

        self.theme = dict(DARK)
        self.settings = {
            "theme": dict(DARK),
            "bg_image_path": None,
            "camera_index": 0,
            "mirror": True,
            "confidence_threshold": 0.58,
            "unknown_below": 0.22,
            "appearance": "dark",
            "tts_volume": 1.0,
            "tts_voice": "",
            "practice_mode": False,
            "kf_count": 3,
            "bg_blur": True,
        }
        self._load_settings()
        self.configure(fg_color=self.theme["BG"])
        ctk.set_appearance_mode(self.settings.get("appearance", "dark"))

        self.trainer = SignTrainer()
        self.latest_features = None
        self.photo = None
        self._latest_frame = None
        self._ui_pending = False
        self._frame_i = 0
        self._vid_w, self._vid_h = VIDEO_W, VIDEO_H
        self._fps_times: deque = deque(maxlen=40)
        self._hand_count = 0

        self.sentence_parts: list[str] = []
        self.last_committed = None
        self.last_commit_at = 0.0
        self.hold_started = None
        self.static_smooth: deque = deque(maxlen=5)

        self.confidence_threshold = float(self.settings.get("confidence_threshold", 0.58))
        self.unknown_below = float(self.settings.get("unknown_below", 0.22))
        self.mirror = bool(self.settings.get("mirror", True))
        self.camera_index = int(self.settings.get("camera_index", 0))
        self.tts_volume = float(self.settings.get("tts_volume", 1.0))
        self.tts_voice = str(self.settings.get("tts_voice", "") or "")
        self.practice_mode = bool(self.settings.get("practice_mode", False))
        self.bg_image_path = self.settings.get("bg_image_path")
        self.is_training = False
        self.neural_waiting = False
        self.neural_guess = None
        self._neural_corrections = 0
        self._last_live_features = None
        self._tts_lock = threading.Lock()
        self._tts_stop = threading.Event()
        self._tts_busy = False
        self._last_feedback = ""

        # Train mode: "static" | "keyframe"
        self.train_kind = "static"
        self.kf_total = int(self.settings.get("kf_count", 3))
        self.bg_blur = bool(self.settings.get("bg_blur", True))
        self.kf_total = max(2, min(5, self.kf_total))
        self.kf_current = 0  # which keyframe we're recording

        self.capturing = False
        self.capture_buffer: list = []
        self.capture_start = None

        self._bg_label = None
        self._bg_photo = None

        self._build_ui()
        self.after(150, self._apply_bg_image)
        self.after(300, self._on_resize)

        self.camera = HandCamera(self._on_frame, camera_index=self.camera_index)
        self.camera.set_mirror(self.mirror)
        self.camera.start()

        self.bind("<Configure>", self._on_resize)
        self.bind_all("<KeyPress>", self._on_key)
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(40, self._tick)
        self.after(400, self._update_fps_label)

    # ── settings ──
    def _load_settings(self) -> None:
        if SETTINGS_PATH.exists():
            try:
                data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
                self.settings.update(data)
                if isinstance(data.get("theme"), dict):
                    self.theme.update(data["theme"])
            except Exception as e:
                print("settings:", e)

    def _save_settings(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.settings.update({
            "theme": dict(self.theme),
            "bg_image_path": self.bg_image_path,
            "camera_index": self.camera_index,
            "mirror": self.mirror,
            "confidence_threshold": self.confidence_threshold,
            "unknown_below": self.unknown_below,
            "tts_volume": self.tts_volume,
            "tts_voice": self.tts_voice,
            "practice_mode": self.practice_mode,
            "kf_count": self.kf_total,
            "bg_blur": self.bg_blur,
        })
        try:
            SETTINGS_PATH.write_text(json.dumps(self.settings, indent=2), encoding="utf-8")
        except Exception as e:
            print("save settings:", e)

    # ── UI ──
    def _build_ui(self) -> None:
        t = self.theme
        self.configure(fg_color=t["BG"])
        # Background label sits on the WINDOW (self), under all UI.
        # Do NOT put a full-size opaque frame over it — that hid the image every time.
        self._bg_label = None
        self._bg_photo = None

        self.root_frame = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        self.root_frame.pack(fill="both", expand=True, padx=0, pady=0)
        self.content = ctk.CTkFrame(self.root_frame, fg_color="transparent")
        self.content.pack(fill="both", expand=True, padx=14, pady=14)
        self.content.grid_columnconfigure(0, weight=3)
        self.content.grid_columnconfigure(1, weight=2)
        self.content.grid_rowconfigure(2, weight=1)

        header = ctk.CTkFrame(self.content, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        self.title_lbl = ctk.CTkLabel(header, text="Sign Bridge", font=ctk.CTkFont(size=24, weight="bold"),
                     text_color=t["ACCENT"])
        self.title_lbl.pack(side="left")
        self.subtitle_lbl = ctk.CTkLabel(header, text="  static · keyframe sequences · live",
                     font=ctk.CTkFont(size=12), text_color=t["MUTED"])
        self.subtitle_lbl.pack(side="left", pady=(6, 0))
        self.fps_label = ctk.CTkLabel(header, text="FPS —  |  hands —", text_color=t["MUTED"],
                                      font=ctk.CTkFont(size=11))
        self.fps_label.pack(side="right")

        # Video
        self.wallpaper = ctk.CTkLabel(
            self.content,
            text="No background — Settings → Browse image → Apply background",
            text_color=t["MUTED"], fg_color=t["CARD"], corner_radius=10, height=120,
        )
        self.wallpaper.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        self._wallpaper_photo = None

        self.video_card = ctk.CTkFrame(self.content, fg_color=t["CARD"], corner_radius=12)
        self.video_card.grid(row=2, column=0, sticky="nsew", padx=(0, 8))
        self.video = ctk.CTkLabel(self.video_card, text="Starting camera…", text_color=t["MUTED"])
        self.video.pack(fill="both", expand=True, padx=8, pady=8)

        self.overlay = ctk.CTkFrame(self.video_card, fg_color="#050508", corner_radius=8)
        self.overlay.pack(fill="x", padx=10, pady=(0, 8))
        self.live_sign = ctk.CTkLabel(self.overlay, text="show a trained sign",
                                      font=ctk.CTkFont(size=20, weight="bold"), text_color=t["TEXT"])
        self.live_sign.pack(anchor="w", padx=10, pady=(6, 0))
        self.conf_bar = ctk.CTkProgressBar(self.overlay, height=5, progress_color=t["ACCENT_DIM"])
        self.conf_bar.pack(fill="x", padx=10, pady=(5, 2))
        self.conf_bar.set(0)
        self.conf_label = ctk.CTkLabel(self.overlay, text="confidence  0%", text_color=t["MUTED"])
        self.conf_label.pack(anchor="w", padx=10, pady=(0, 2))
        self.feedback_lbl = ctk.CTkLabel(self.overlay, text="", text_color=t["ACCENT"], font=ctk.CTkFont(size=11))
        self.feedback_lbl.pack(anchor="w", padx=10, pady=(0, 6))
        self.capture_prog = ctk.CTkProgressBar(self.video_card, height=8, progress_color=t["ACCENT"])
        self.capture_prog.set(0)

        # Side
        side = ctk.CTkFrame(self.content, fg_color="transparent")
        side.grid(row=2, column=1, sticky="nsew")

        self.sentence_card = ctk.CTkFrame(side, fg_color=t["CARD"], corner_radius=12)
        self.sentence_card.pack(fill="x")
        self.translation_hdr = ctk.CTkLabel(self.sentence_card, text="TRANSLATION", text_color=t["MUTED"],
                     font=ctk.CTkFont(size=10, weight="bold"))
        self.translation_hdr.pack(anchor="w", padx=10, pady=(6, 0))
        self.sentence = ctk.CTkLabel(self.sentence_card, text="Your sentence will appear here.",
                                     wraplength=380, justify="left", font=ctk.CTkFont(size=15),
                                     text_color=t["TEXT"])
        self.sentence.pack(anchor="w", padx=10, pady=4)

        actions = ctk.CTkFrame(self.sentence_card, fg_color="transparent")
        actions.pack(fill="x", padx=6, pady=(0, 4))
        for txt, cmd in [("Space", self._space), ("Delete", self._backspace),
                         ("Undo", self._undo), ("Clear", self._clear)]:
            ctk.CTkButton(actions, text=txt, width=64, height=26, command=cmd,
                          fg_color="#1e1e2a").pack(side="left", padx=2)
        ctk.CTkButton(actions, text="Speak", width=64, height=26, command=self._speak,
                      fg_color=t["ACCENT_DIM"], text_color="#042f2e").pack(side="left", padx=2)
        ctk.CTkButton(actions, text="Stop", width=56, height=26, command=self._stop_tts,
                      fg_color="#1e1e2a").pack(side="left", padx=2)

        sl = ctk.CTkFrame(self.sentence_card, fg_color="transparent")
        sl.pack(fill="x", padx=10, pady=(0, 4))
        ctk.CTkLabel(sl, text="Threshold", text_color=t["MUTED"], font=ctk.CTkFont(size=10)).pack(side="left")
        self.conf_slider = ctk.CTkSlider(sl, from_=0.4, to=0.9, number_of_steps=50,
                                         command=self._on_conf, width=110)
        self.conf_slider.set(self.confidence_threshold)
        self.conf_slider.pack(side="left", padx=4)
        self.conf_thresh_label = ctk.CTkLabel(sl, text=f"{self.confidence_threshold:.0%}", text_color=t["MUTED"])
        self.conf_thresh_label.pack(side="left")
        ctk.CTkLabel(sl, text=" Vol", text_color=t["MUTED"], font=ctk.CTkFont(size=10)).pack(side="left", padx=(8, 0))
        self.vol_slider = ctk.CTkSlider(sl, from_=0, to=1, number_of_steps=20,
                                        command=self._on_vol, width=80)
        self.vol_slider.set(self.tts_volume)
        self.vol_slider.pack(side="left", padx=2)

        self.practice_var = ctk.BooleanVar(value=self.practice_mode)
        self.practice_sw = ctk.CTkSwitch(self.sentence_card, text="Practice mode", variable=self.practice_var,
                      command=self._on_practice, text_color=t["MUTED"])
        self.practice_sw.pack(anchor="w", padx=10, pady=(0, 6))

        self.tabs = ctk.CTkTabview(side, fg_color=t["CARD"], corner_radius=12,
                                   segmented_button_selected_color=t["ACCENT_DIM"])
        self.tabs.pack(fill="both", expand=True, pady=(8, 0))
        for name in ("Train", "Neural", "Library", "Settings", "Misc", "How to"):
            self.tabs.add(name)

        self._build_train()
        self._build_neural()
        self._build_library()
        self._build_settings()
        self._build_misc()
        self._build_howto()

    def _build_train(self) -> None:
        t = self.theme
        tab = self.tabs.tab("Train")
        ctk.CTkLabel(
            tab,
            text="Static = one pose. Keyframe = 2–5 poses in order (motion).\n"
                 "Hands: 1 or 2 both work automatically.",
            wraplength=380, justify="left", text_color=t["MUTED"],
        ).pack(anchor="w", padx=8, pady=6)

        self.kind_switch = ctk.CTkSegmentedButton(
            tab, values=["Static", "Keyframe"], command=self._on_kind,
            selected_color=t["ACCENT_DIM"],
        )
        self.kind_switch.set("Static")
        self.kind_switch.pack(fill="x", padx=8, pady=4)

        self.sign_name = ctk.CTkEntry(tab, placeholder_text="sign name e.g. hello", height=34)
        self.sign_name.pack(fill="x", padx=8, pady=4)

        # Keyframe controls
        self.kf_frame = ctk.CTkFrame(tab, fg_color="transparent")
        ctk.CTkLabel(self.kf_frame, text="Keyframes (2–5)", text_color=t["MUTED"]).pack(side="left")
        self.kf_slider = ctk.CTkSlider(self.kf_frame, from_=2, to=5, number_of_steps=3,
                                       command=self._on_kf_count, width=100)
        self.kf_slider.set(self.kf_total)
        self.kf_slider.pack(side="left", padx=6)
        self.kf_count_lbl = ctk.CTkLabel(self.kf_frame, text=str(self.kf_total), text_color=t["TEXT"])
        self.kf_count_lbl.pack(side="left")
        self.kf_step_lbl = ctk.CTkLabel(tab, text="", text_color=t["ACCENT"])
        # shown when keyframe mode

        self.record_btn = ctk.CTkButton(
            tab, text="Record pose  (R)", height=40,
            fg_color=t["ACCENT_DIM"], hover_color=t["ACCENT"], text_color="#042f2e",
            font=ctk.CTkFont(weight="bold"), command=self._start_record,
        )
        self.record_btn.pack(fill="x", padx=8, pady=6)
        self._accent_buttons = [self.record_btn]
        self.capture_status = ctk.CTkLabel(tab, text="Ready.", text_color=t["MUTED"])
        self.capture_status.pack(anchor="w", padx=8)

        self.train_btn = ctk.CTkButton(tab, text="Train model  (T)", height=36, command=self._train,
                                       fg_color="#1e1e2a")
        self.train_btn.pack(fill="x", padx=8, pady=(10, 4))
        self.train_status = ctk.CTkLabel(tab, text=self._blurb(), text_color=t["MUTED"],
                                         wraplength=380, justify="left")
        self.train_status.pack(anchor="w", padx=8)


    def _build_neural(self) -> None:
        t = self.theme
        tab = self.tabs.tab("Neural")
        ctk.CTkLabel(
            tab,
            text="Show a sign → one guess → Correct or Wrong.\n"
                 "Static: type the sign name if wrong.\n"
                 "Keyframe: hold ONE step pose, then guess — Correct reinforces that step.\n"
                 "If wrong, type the sign name (step is kept). Auto-retrain after a few fixes.",
            wraplength=380, justify="left", text_color=t["MUTED"],
        ).pack(anchor="w", padx=8, pady=6)
        self.neural_labels_box = ctk.CTkLabel(tab, text="Available: —", wraplength=380,
                                              justify="left", text_color=t["TEXT"])
        self.neural_labels_box.pack(anchor="w", padx=8)
        self._refresh_neural_labels()

        self.neural_guess_label = ctk.CTkLabel(
            tab, text="Guess: —",
            font=ctk.CTkFont(size=18, weight="bold"), text_color=t["ACCENT"],
        )
        self.neural_guess_label.pack(anchor="w", padx=8, pady=8)

        row = ctk.CTkFrame(tab, fg_color="transparent")
        row.pack(fill="x", padx=8)
        self.neural_start_btn = ctk.CTkButton(
            row, text="Make a guess  (N)", width=120, command=self._neural_guess,
            fg_color=t["ACCENT_DIM"], text_color="#042f2e",
        )
        self.neural_start_btn.pack(side="left", padx=(0, 4))
        self.neural_correct_btn = ctk.CTkButton(
            row, text="Correct ✓", width=85, command=self._neural_correct,
            fg_color="#166534", state="disabled",
        )
        self.neural_correct_btn.pack(side="left", padx=2)
        self.neural_wrong_btn = ctk.CTkButton(
            row, text="Wrong ✗", width=85, command=self._neural_wrong,
            fg_color="#7f1d1d", state="disabled",
        )
        self.neural_wrong_btn.pack(side="left", padx=2)

        self.neural_fix_entry = ctk.CTkEntry(
            tab, placeholder_text="If wrong — type correct label", height=32,
        )
        self.neural_fix_entry.pack(fill="x", padx=8, pady=8)
        self.neural_fix_entry.configure(state="disabled")
        self.neural_status = ctk.CTkLabel(tab, text="", text_color=t["MUTED"])
        self.neural_status.pack(anchor="w", padx=8)

    def _refresh_neural_labels(self) -> None:
        labels = self.trainer.labels()
        try:
            self.neural_labels_box.configure(
                text="Available: " + (", ".join(labels) if labels else "(train some first)")
            )
        except Exception:
            pass

    def _neural_guess(self) -> None:
        if self.neural_waiting:
            return
        feat = self.latest_features
        self._last_live_features = feat
        raw, conf = self.trainer.predict_raw(feat)
        if raw is None or feat is None:
            self.neural_status.configure(text="No model or no hand detected.")
            return
        display = raw.split("::k")[0]
        self.neural_guess = raw
        self.neural_waiting = True
        self.neural_guess_label.configure(text=f"Guess: {display}  ({conf*100:.0f}%)")
        self.neural_correct_btn.configure(state="normal")
        self.neural_wrong_btn.configure(state="normal")
        self.neural_fix_entry.configure(state="normal")
        self.neural_status.configure(text="Correct or Wrong?")

    def _neural_correct(self) -> None:
        if not self.neural_waiting or not self.neural_guess:
            return
        feat = self._last_live_features
        if feat is not None:
            import numpy as np
            self.trainer.X.append(np.asarray(feat, dtype=np.float32))
            self.trainer.y.append(self.neural_guess)
            self.trainer.save_samples()
        display = self.neural_guess.split("::k")[0]
        self.neural_status.configure(text=f"Reinforced “{display}”.")
        self._neural_reset()
        self._refresh_library()
        self._refresh_neural_labels()
        self.train_status.configure(text=self._blurb())
        self._maybe_auto_retrain()

    def _neural_wrong(self) -> None:
        if not self.neural_waiting:
            return
        correct = self.neural_fix_entry.get().strip().lower()
        if not correct:
            self.neural_status.configure(text="Type the correct label, then Wrong.")
            return
        feat = self._last_live_features
        if feat is None:
            self.neural_status.configure(text="No features to learn from.")
            return
        import numpy as np
        if self.neural_guess and "::k" in self.neural_guess:
            step = self.neural_guess.split("::k")[-1]
            if correct in self.trainer.keyframe_signs:
                label = f"{correct}::k{step}"
            else:
                label = correct
        else:
            label = correct
        self.trainer.X.append(np.asarray(feat, dtype=np.float32))
        self.trainer.y.append(label)
        self.trainer.save_samples()
        self.neural_status.configure(text=f"Learned as “{correct}”. Re-train when ready.")
        self._neural_reset()
        self._refresh_library()
        self._refresh_neural_labels()
        self.train_status.configure(text=self._blurb())
        self._maybe_auto_retrain()

    def _neural_reset(self) -> None:
        self.neural_waiting = False
        self.neural_guess = None
        self.neural_guess_label.configure(text="Guess: —")
        self.neural_correct_btn.configure(state="disabled")
        self.neural_wrong_btn.configure(state="disabled")
        self.neural_fix_entry.delete(0, "end")
        self.neural_fix_entry.configure(state="disabled")

    def _maybe_auto_retrain(self) -> None:
        self._neural_corrections += 1
        if self._neural_corrections >= 3 and self.trainer.can_train():
            self._neural_corrections = 0
            self.neural_status.configure(text="Auto-retraining…")
            self._train()

    def _build_library(self) -> None:
        self.lib_box = ctk.CTkScrollableFrame(self.tabs.tab("Library"), fg_color="transparent")
        self.lib_box.pack(fill="both", expand=True, padx=4, pady=4)
        self._refresh_library()

    def _build_settings(self) -> None:
        t = self.theme
        tab = self.tabs.tab("Settings")
        ctk.CTkLabel(
            tab,
            text="Presets, colors, and full-window blurred background image.",
            text_color=t["MUTED"],
        ).pack(anchor="w", padx=8, pady=6)

        row = ctk.CTkFrame(tab, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=4)
        ctk.CTkButton(row, text="Dark", width=80, command=lambda: self._preset("dark"),
                      fg_color="#1e1e2a").pack(side="left", padx=2)
        ctk.CTkButton(row, text="Light", width=80, command=lambda: self._preset("light"),
                      fg_color="#1e1e2a").pack(side="left", padx=2)

        self.color_vars = {}
        for key, label in [("BG", "Window BG"), ("CARD", "Cards / panels"),
                           ("ACCENT", "Accent"), ("ACCENT_DIM", "Accent dim"),
                           ("TEXT", "Text"), ("MUTED", "Muted text")]:
            r = ctk.CTkFrame(tab, fg_color="transparent")
            r.pack(fill="x", padx=8, pady=2)
            ctk.CTkLabel(r, text=label, width=110, anchor="w").pack(side="left")
            var = ctk.StringVar(value=t.get(key, "#000000"))
            ctk.CTkEntry(r, textvariable=var, width=90).pack(side="left")
            ctk.CTkButton(
                r, text="Pick", width=48, height=26,
                command=lambda k=key: self._pick_color(k),
                fg_color="#1e1e2a",
            ).pack(side="left", padx=4)
            self.color_vars[key] = var

        ctk.CTkButton(
            tab, text="Apply colors", command=self._apply_colors,
            fg_color=t["ACCENT_DIM"], text_color="#042f2e",
        ).pack(anchor="w", padx=8, pady=(8, 4))

        ctk.CTkLabel(
            tab,
            text="Background image (full window + blur)",
            text_color=t["MUTED"], font=ctk.CTkFont(weight="bold"),
        ).pack(anchor="w", padx=8, pady=(10, 4))
        r = ctk.CTkFrame(tab, fg_color="transparent")
        r.pack(fill="x", padx=8, pady=4)
        ctk.CTkLabel(r, text="Image path", width=110, anchor="w").pack(side="left")
        self.bg_path_var = ctk.StringVar(value=self.bg_image_path or "")
        ctk.CTkEntry(r, textvariable=self.bg_path_var, width=160).pack(side="left", padx=4)
        ctk.CTkButton(r, text="Browse…", width=70, command=self._browse_bg).pack(side="left")
        r = ctk.CTkFrame(tab, fg_color="transparent")
        r.pack(fill="x", padx=8, pady=4)
        ctk.CTkLabel(r, text="Blur image", width=110, anchor="w").pack(side="left")
        self.bg_blur_sw = ctk.CTkSwitch(r, text="on = soft blur / off = sharp", command=self._on_bg_blur)
        (self.bg_blur_sw.select() if self.bg_blur else self.bg_blur_sw.deselect())
        self.bg_blur_sw.pack(side="left")

        ctk.CTkButton(tab, text="Apply background", command=self._apply_bg_from_ui,
                      fg_color=t["ACCENT_DIM"], text_color="#042f2e").pack(anchor="w", padx=8, pady=4)
        ctk.CTkButton(tab, text="Clear background image", command=self._clear_bg,
                      fg_color="#1e1e2a").pack(anchor="w", padx=8, pady=2)
        ctk.CTkLabel(
            tab,
            text="Image shows in the TOP BANNER across the app (reliable in CTk). "
                 "Browse → Apply. Blur on/off. Text color = Text row → Apply colors.",
            wraplength=380, justify="left", text_color=t["MUTED"],
        ).pack(anchor="w", padx=8, pady=4)

    def _build_misc(self) -> None:
        t = self.theme
        tab = self.tabs.tab("Misc")
        r = ctk.CTkFrame(tab, fg_color="transparent")
        r.pack(fill="x", padx=8, pady=8)
        ctk.CTkLabel(r, text="Camera", width=90, anchor="w").pack(side="left")
        cams = list_cameras()
        vals = [f"Camera {i}" for i in cams] or ["Camera 0"]
        self.cam_var = ctk.StringVar(
            value=f"Camera {self.camera_index}" if f"Camera {self.camera_index}" in vals else vals[0]
        )
        ctk.CTkOptionMenu(r, values=vals, variable=self.cam_var,
                          command=self._on_camera, width=120).pack(side="left")

        r = ctk.CTkFrame(tab, fg_color="transparent")
        r.pack(fill="x", padx=8, pady=4)
        ctk.CTkLabel(r, text="Mirror", width=90, anchor="w").pack(side="left")
        self.mirror_sw = ctk.CTkSwitch(r, text="", command=self._on_mirror)
        (self.mirror_sw.select() if self.mirror else self.mirror_sw.deselect())
        self.mirror_sw.pack(side="left")

        ctk.CTkLabel(tab, text="Export / import trained data", text_color=t["MUTED"]).pack(
            anchor="w", padx=8, pady=(12, 4))
        r = ctk.CTkFrame(tab, fg_color="transparent")
        r.pack(fill="x", padx=8)
        ctk.CTkButton(r, text="Export zip", width=100, command=self._export,
                      fg_color=t["ACCENT_DIM"], text_color="#042f2e").pack(side="left", padx=(0, 4))
        ctk.CTkButton(r, text="Import zip", width=100, command=self._import,
                      fg_color="#1e1e2a").pack(side="left")
        self.export_status = ctk.CTkLabel(tab, text="", text_color=t["MUTED"])
        self.export_status.pack(anchor="w", padx=8, pady=4)

        ctk.CTkLabel(tab, text="Keys: R record · T train · S speak · U undo · Esc cancel",
                     text_color=t["MUTED"], font=ctk.CTkFont(size=11)).pack(anchor="w", padx=8, pady=10)

    def _build_howto(self) -> None:
        tips = (
            "STATIC\n"
            "1. Select Static, type a name\n"
            "2. Record pose (hold still ~2.5s)\n"
            "3. Capture a few samples → Train\n\n"
            "KEYFRAME (motion)\n"
            "1. Select Keyframe, pick 2–5 steps\n"
            "2. Type name (e.g. hello)\n"
            "3. Record pose for step 1, then 2, …\n"
            "4. Train when all steps have samples\n"
            "5. Live: do poses in order quickly → sign commits\n\n"
            "1 or 2 hands both work — missing hand is ignored.\n"
            "Hybrid is automatic: static + keyframe work together live.\n"
            "Neural tab: correct wrong guesses so the model learns."
        )
        ctk.CTkLabel(self.tabs.tab("How to"), text=tips, justify="left", wraplength=380,
                     text_color=self.theme["TEXT"]).pack(anchor="w", padx=8, pady=10)

    # ── handlers ──
    def _on_kind(self, v: str) -> None:
        self.train_kind = "keyframe" if v == "Keyframe" else "static"
        self.kf_current = 0
        if self.train_kind == "keyframe":
            self.kf_frame.pack(fill="x", padx=8, pady=4)
            self.kf_step_lbl.pack(anchor="w", padx=8)
            self._update_kf_step_label()
            self.record_btn.configure(text="Record keyframe pose  (R)")
        else:
            self.kf_frame.pack_forget()
            self.kf_step_lbl.pack_forget()
            self.record_btn.configure(text="Record pose  (R)")

    def _on_kf_count(self, val: float) -> None:
        self.kf_total = int(round(val))
        self.kf_count_lbl.configure(text=str(self.kf_total))
        self.kf_current = min(self.kf_current, self.kf_total - 1)
        self._update_kf_step_label()
        self._save_settings()

    def _update_kf_step_label(self) -> None:
        self.kf_step_lbl.configure(
            text=f"Recording step {self.kf_current + 1} of {self.kf_total}"
        )

    def _on_conf(self, v: float) -> None:
        self.confidence_threshold = float(v)
        self.conf_thresh_label.configure(text=f"{self.confidence_threshold:.0%}")
        self._save_settings()

    def _on_vol(self, v: float) -> None:
        self.tts_volume = float(v)
        self._save_settings()

    def _on_practice(self) -> None:
        self.practice_mode = bool(self.practice_var.get())
        self._save_settings()

    def _on_mirror(self) -> None:
        self.mirror = bool(self.mirror_sw.get())
        self.camera.set_mirror(self.mirror)
        self._save_settings()

    def _on_camera(self, value: str) -> None:
        try:
            idx = int(value.replace("Camera ", "").strip())
        except ValueError:
            return
        self.camera_index = idx
        self.camera.set_camera_index(idx)
        self._save_settings()


    def _pick_color(self, key: str) -> None:
        """OS color picker (closest thing to a color wheel in Tk)."""
        initial = self.theme.get(key, "#ffffff")
        result = colorchooser.askcolor(color=initial, title=f"Pick {key} color")
        if not result or not result[1]:
            return
        hx = result[1]
        self.theme[key] = hx
        if key in getattr(self, "color_vars", {}):
            self.color_vars[key].set(hx)
        self._save_settings()
        self._recolor_ui()
        self._set_feedback(f"{key} → {hx}")

    def _show_example(self, label: str) -> None:
        """100% path: find signexamples/<name>.(png|jpg|...) and show it."""
        EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        name = label.strip().lower()
        path = None
        exts = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
        # exact name match
        for ext in exts:
            cand = EXAMPLES_DIR / f"{name}{ext}"
            if cand.is_file():
                path = cand
                break
        # case-insensitive scan
        if path is None and EXAMPLES_DIR.exists():
            for p in EXAMPLES_DIR.iterdir():
                if p.is_file() and p.suffix.lower() in exts and p.stem.lower() == name:
                    path = p
                    break

        win = ctk.CTkToplevel(self)
        win.title(f"Example — {label}")
        win.geometry("480x520")
        try:
            win.attributes("-topmost", True)
        except Exception:
            pass
        win.configure(fg_color=self.theme["BG"])
        ctk.CTkLabel(
            win, text=str(label),
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=self.theme["ACCENT"],
        ).pack(pady=(14, 8))

        if path is None:
            ctk.CTkLabel(
                win,
                text=(
                    f"No image for “{name}”.\n\n"
                    f"Add a file here:\n{EXAMPLES_DIR}\n\n"
                    f"Name it:\n  {name}.png\n  or {name}.jpg"
                ),
                text_color=self.theme["MUTED"],
                justify="center",
            ).pack(expand=True, padx=20)
        else:
            try:
                img = Image.open(path).convert("RGB")
                img.thumbnail((420, 380), Image.Resampling.LANCZOS)
                photo = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
                lbl = ctk.CTkLabel(win, text="", image=photo)
                lbl.pack(padx=12, pady=8)
                win._photo = photo
                ctk.CTkLabel(win, text=path.name, text_color=self.theme["MUTED"]).pack()
            except Exception as e:
                ctk.CTkLabel(
                    win, text=f"Failed to open image:\n{e}",
                    text_color=self.theme["DANGER"],
                ).pack(padx=12, pady=20)

        ctk.CTkButton(win, text="Close", width=120, command=win.destroy).pack(pady=14)


    def _stop_tts(self) -> None:
        self._tts_stop.set()
        self._tts_busy = False
        self._set_feedback("TTS stopped")

    def _norm_hex(self, val: str) -> str | None:
        val = val.strip()
        if not val.startswith("#"):
            return None
        if len(val) == 4:  # #RGB -> #RRGGBB
            return "#" + "".join(c * 2 for c in val[1:])
        if len(val) == 7:
            return val
        return None

    def _apply_colors(self) -> None:
        for key, var in getattr(self, "color_vars", {}).items():
            hx = self._norm_hex(var.get())
            if hx:
                self.theme[key] = hx
                var.set(hx)
        self._save_settings()
        self._recolor_ui()
        self._set_feedback("Colors applied")
        self.after(80, self._apply_bg_image)

    def _recolor_ui(self) -> None:
        """Apply ALL theme keys: BG, CARD, ACCENT, ACCENT_DIM, TEXT, MUTED."""
        t = self.theme
        try:
            has_img = bool(self.bg_image_path and Path(str(self.bg_image_path)).is_file())
            self.configure(fg_color=t["BG"])
            if has_img:
                self.root_frame.configure(fg_color="transparent")
                if getattr(self, "content", None) is not None:
                    self.content.configure(fg_color="transparent")
                if getattr(self, "_bg_label", None) is not None:
                    try:
                        self._bg_label.lower()
                        self.root_frame.lift()
                    except Exception:
                        pass
            else:
                self.root_frame.configure(fg_color=t["BG"])
                if getattr(self, "content", None) is not None:
                    self.content.configure(fg_color="transparent")

            # Cards / panels
            if getattr(self, "video_card", None):
                self.video_card.configure(fg_color=t["CARD"])
            if getattr(self, "sentence_card", None):
                self.sentence_card.configure(fg_color=t["CARD"])
            if getattr(self, "tabs", None):
                self.tabs.configure(
                    fg_color=t["CARD"],
                    segmented_button_selected_color=t["ACCENT_DIM"],
                    segmented_button_selected_hover_color=t["ACCENT"],
                    text_color=t["TEXT"],
                )
            if getattr(self, "overlay", None):
                # slightly darker than card for contrast
                self.overlay.configure(fg_color=t["BG"])

            # Title / accents
            if getattr(self, "title_lbl", None):
                self.title_lbl.configure(text_color=t["ACCENT"])
            if getattr(self, "subtitle_lbl", None):
                self.subtitle_lbl.configure(text_color=t["MUTED"])
            if getattr(self, "feedback_lbl", None):
                self.feedback_lbl.configure(text_color=t["ACCENT"])
            if getattr(self, "fps_label", None):
                self.fps_label.configure(text_color=t["MUTED"])

            # Main text
            if getattr(self, "live_sign", None):
                self.live_sign.configure(text_color=t["TEXT"])
            if getattr(self, "sentence", None):
                self.sentence.configure(text_color=t["TEXT"])
            if getattr(self, "translation_hdr", None):
                self.translation_hdr.configure(text_color=t["MUTED"])
            if getattr(self, "conf_label", None):
                self.conf_label.configure(text_color=t["MUTED"])
            if getattr(self, "conf_thresh_label", None):
                self.conf_thresh_label.configure(text_color=t["MUTED"])
            if getattr(self, "capture_status", None):
                self.capture_status.configure(text_color=t["MUTED"])
            if getattr(self, "train_status", None):
                self.train_status.configure(text_color=t["MUTED"])
            if getattr(self, "practice_sw", None):
                self.practice_sw.configure(text_color=t["MUTED"])

            # Progress / accent dim
            if getattr(self, "conf_bar", None):
                self.conf_bar.configure(progress_color=t["ACCENT_DIM"])
            if getattr(self, "capture_prog", None):
                self.capture_prog.configure(progress_color=t["ACCENT"])

            # Primary action buttons use ACCENT_DIM
            if getattr(self, "record_btn", None):
                self.record_btn.configure(fg_color=t["ACCENT_DIM"], hover_color=t["ACCENT"])
            if getattr(self, "neural_start_btn", None):
                try:
                    self.neural_start_btn.configure(fg_color=t["ACCENT_DIM"])
                except Exception:
                    pass

            if getattr(self, "_accent_bar", None) is not None:
                self._accent_bar.configure(fg_color=t["ACCENT_DIM"])

            # Keep color field entries in sync
            for key, var in getattr(self, "color_vars", {}).items():
                if key in t:
                    var.set(t[key])
        except Exception as e:
            print("recolor:", e)


    def _preset(self, name: str) -> None:
        self.theme = dict(DARK if name == "dark" else LIGHT)
        self.settings["appearance"] = name
        ctk.set_appearance_mode(name)
        self._save_settings()
        for key, var in getattr(self, "color_vars", {}).items():
            if key in self.theme:
                var.set(self.theme[key])
        self._recolor_ui()
        self.after(80, self._apply_bg_image)

    def _browse_bg(self) -> None:
        path = filedialog.askopenfilename(
            title="Background image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.webp;*.bmp"), ("All", "*.*")],
        )
        if path:
            path = str(Path(path).resolve())
            self.bg_path_var.set(path)
            self.bg_image_path = path
            self._save_settings()
            self._set_feedback("Loading background…")
            self.after(50, self._apply_bg_image)

    def _on_bg_blur(self) -> None:
        self.bg_blur = bool(self.bg_blur_sw.get())
        self._save_settings()
        self._apply_bg_image()

    def _apply_bg_from_ui(self) -> None:
        path = self.bg_path_var.get().strip().strip('"')
        ok = bool(path and Path(path).is_file())
        self.bg_image_path = path if ok else None
        self._save_settings()
        if not ok:
            self._set_feedback("Image path not found — use Browse")
            print("bg path missing:", path)
            return
        self._apply_bg_image()

    def _clear_bg(self) -> None:
        self.bg_image_path = None
        try:
            self.bg_path_var.set("")
        except Exception:
            pass
        self._save_settings()
        self._clear_bg_canvas()
        try:
            self.root_frame.configure(fg_color=self.theme["BG"])
            self.configure(fg_color=self.theme["BG"])
        except Exception:
            pass


    def _apply_bg_image(self) -> None:
        """Show background image on the full-width wallpaper strip (always visible)."""
        path = self.bg_image_path
        if not path or not Path(str(path)).is_file():
            print("bg: no file", path)
            if getattr(self, "wallpaper", None) is not None:
                self.wallpaper.configure(
                    image=None,
                    text="No background — Settings → Browse image → Apply background",
                    fg_color=self.theme["CARD"],
                )
            self._wallpaper_photo = None
            return

        try:
            self.update_idletasks()
            # strip size: full content width x ~120-160px
            cw = max(int(self.content.winfo_width()), int(self.winfo_width()) - 40, 400)
            ch = 140
            img = Image.open(str(path)).convert("RGB")
            # cover-crop to strip aspect
            tw, th = cw, ch
            scale = max(tw / img.width, th / img.height)
            nw, nh = int(img.width * scale), int(img.height * scale)
            img = img.resize((nw, nh), Image.Resampling.LANCZOS)
            left = max(0, (nw - tw) // 2)
            top = max(0, (nh - th) // 2)
            img = img.crop((left, top, left + tw, top + th))
            if getattr(self, "bg_blur", True):
                img = img.filter(ImageFilter.GaussianBlur(radius=8))
            img = ImageEnhance.Brightness(img).enhance(0.70)

            photo = ctk.CTkImage(light_image=img, dark_image=img, size=(tw, th))
            self._wallpaper_photo = photo
            self.wallpaper.configure(image=photo, text="", fg_color=self.theme["CARD"])
            self._bg_size = (cw, ch)
            self._set_feedback("Background image showing in top banner")
            print(f"bg OK: {path} -> wallpaper {tw}x{th}")
        except Exception as e:
            print("bg ERROR:", e)
            self._set_feedback(f"BG failed: {e}")


    def _clear_bg_canvas(self) -> None:
        if getattr(self, "_bg_label", None) is not None:
            try:
                self._bg_label.destroy()
            except Exception:
                pass
            self._bg_label = None
        self._bg_photo = None


    def _on_frame(self, frame_bgr, features, n_hands: int) -> None:
        self._fps_times.append(time.time())
        self.latest_features = features
        self._hand_count = n_hands
        self._frame_i += 1

        if self.capturing and features is not None and len(self.capture_buffer) < MAX_CAPTURE:
            self.capture_buffer.append(features.copy())

        self._latest_frame = frame_bgr
        if not self._ui_pending:
            self._ui_pending = True
            try:
                self.after(0, self._flush_video)
            except Exception:
                self._ui_pending = False


    def _on_resize(self, event=None) -> None:
        """Scale video to fill the video card when window is maximized/resized."""
        try:
            # Only react to main window resizes (avoid child spam)
            if event is not None and event.widget is not self:
                return
            self.update_idletasks()
            # Measure the actual card the video lives in
            cw = int(self.video_card.winfo_width())
            ch = int(self.video_card.winfo_height())
            if cw < 100 or ch < 100:
                return
            # leave space for bottom overlay (~90px)
            max_w = max(320, cw - 24)
            max_h = max(240, ch - 100)
            # fit 4:3 inside available box
            if max_w / max_h > 4 / 3:
                h = max_h
                w = int(h * 4 / 3)
            else:
                w = max_w
                h = int(w * 3 / 4)
            w, h = max(320, w), max(240, h)
            if abs(w - self._vid_w) > 8 or abs(h - self._vid_h) > 8:
                self._vid_w, self._vid_h = w, h
            if self.bg_image_path:
                ww, hh = max(self.winfo_width(), 1), max(self.winfo_height(), 1)
                prev = getattr(self, "_bg_size", (0, 0))
                if abs(ww - prev[0]) > 50 or abs(hh - prev[1]) > 50:
                    if getattr(self, "_bg_resize_after", None) is not None:
                        try:
                            self.after_cancel(self._bg_resize_after)
                        except Exception:
                            pass
                    self._bg_resize_after = self.after(200, self._apply_bg_image)
        except Exception:
            pass

    def _debounced_bg_resize(self) -> None:
        self._bg_resize_after = None
        self._apply_bg_image()

    def _flush_video(self) -> None:
        self._ui_pending = False
        frame = self._latest_frame
        if frame is None:
            return
        try:
            vw, vh = self._vid_w, self._vid_h
            small = cv2.resize(frame, (vw, vh), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            self.photo = ctk.CTkImage(light_image=image, dark_image=image, size=(vw, vh))
            self.video.configure(image=self.photo, text="")
        except Exception:
            pass

    def _update_fps_label(self) -> None:
        if len(self._fps_times) >= 2:
            dt = self._fps_times[-1] - self._fps_times[0]
            fps = (len(self._fps_times) - 1) / dt if dt > 0 else 0
        else:
            fps = 0
        try:
            self.fps_label.configure(text=f"FPS {fps:.0f}  |  hands {self._hand_count}")
        except Exception:
            pass
        self.after(400, self._update_fps_label)

    def _tick(self) -> None:
        if self.capturing and self.capture_start is not None:
            elapsed = time.time() - self.capture_start
            prog = min(1.0, elapsed / RECORD_SECONDS)
            try:
                self.capture_prog.set(prog)
                self.capture_status.configure(
                    text=f"Hold pose… {elapsed:.1f}s  ({len(self.capture_buffer)} frames)"
                )
            except Exception:
                pass
            if elapsed >= RECORD_SECONDS:
                self._finish_record()

        if not self.capturing:
            now = time.time()
            commit, conf, display = self.trainer.predict_live(
                self.latest_features,
                now,
                conf_threshold=self.confidence_threshold,
                unknown_below=self.unknown_below,
            )

            # display
            if conf >= self.confidence_threshold:
                color = "#4ade80"
            elif conf >= self.confidence_threshold - 0.12:
                color = "#fbbf24"
            elif display == "unknown":
                color = "#94a3b8"
            else:
                color = self.theme["TEXT"]
            self.live_sign.configure(text=display, text_color=color)
            self.conf_bar.set(conf)
            self.conf_label.configure(text=f"confidence  {conf * 100:.0f}%")

            # Keyframe complete → commit immediately
            if commit and self.trainer.is_keyframe_sign(commit) and not self.practice_mode:
                self._commit(commit)
                self.hold_started = None
            # Static → hold
            elif commit and not self.trainer.is_keyframe_sign(commit) and not self.practice_mode:
                if conf >= self.confidence_threshold:
                    if self.hold_started is None:
                        self.hold_started = now
                    elif now - self.hold_started >= 0.7:
                        self._commit(commit)
                        self.hold_started = None
                else:
                    self.hold_started = None
            else:
                if not commit:
                    self.hold_started = None

        self.after(40, self._tick)

    def _blurb(self) -> str:
        n = len(self.trainer.labels())
        samples = len(self.trainer.y)
        kf = len(self.trainer.keyframe_signs)
        ready = "Model ready" if self.trainer.model else "Not trained"
        return f"{ready}  ·  {n} signs ({kf} keyframe)  ·  {samples} samples"

    def _set_feedback(self, msg: str) -> None:
        self._last_feedback = msg
        try:
            self.feedback_lbl.configure(text=msg)
        except Exception:
            pass
        self.after(1600, lambda: self.feedback_lbl.configure(text="") if self._last_feedback == msg else None)

    def _commit(self, name: str) -> None:
        now = time.time()
        if name == self.last_committed and now - self.last_commit_at < 1.4:
            return
        self.last_committed = name
        self.last_commit_at = now
        self.sentence_parts.append(name)
        self._render_sentence()
        self._set_feedback(f"Committed: {name}")
        self.after(1400, lambda: setattr(self, "last_committed", None))

    def _render_sentence(self) -> None:
        text = " ".join(self.sentence_parts).strip() or "Your sentence will appear here."
        self.sentence.configure(text=text)

    def _space(self) -> None:
        if self.sentence_parts and self.sentence_parts[-1] != "":
            self.sentence_parts.append("")
        self._render_sentence()

    def _backspace(self) -> None:
        if self.sentence_parts:
            self.sentence_parts.pop()
        self._render_sentence()

    def _undo(self) -> None:
        if self.sentence_parts:
            removed = self.sentence_parts.pop()
            self._render_sentence()
            self._set_feedback(f"Undid: {removed or '(space)'}")

    def _clear(self) -> None:
        self.sentence_parts.clear()
        self._render_sentence()

    def _beep(self, kind: str = "start") -> None:
        def run():
            try:
                if winsound:
                    winsound.Beep(880 if kind == "start" else 660, 90)
            except Exception:
                pass
        threading.Thread(target=run, daemon=True).start()

    def _start_record(self) -> None:
        name = self.sign_name.get().strip()
        if not name:
            self.capture_status.configure(text="Type a sign name first.")
            return
        if self.capturing:
            return
        if self.train_kind == "static" and self.trainer.is_keyframe_sign(name.lower()):
            self.capture_status.configure(text="That name is a keyframe sign. Use Keyframe mode or another name.")
            return
        if self.train_kind == "keyframe" and name.lower() in self.trainer.counts() and not self.trainer.is_keyframe_sign(name.lower()):
            self.capture_status.configure(text="That name is a static sign. Use another name for keyframes.")
            return

        self.capturing = True
        self.capture_buffer = []
        self.capture_start = time.time()
        self.record_btn.configure(state="disabled", text="Recording…")
        try:
            self.capture_prog.pack(fill="x", padx=10, pady=(0, 6))
            self.capture_prog.set(0)
        except Exception:
            pass
        if self.train_kind == "keyframe":
            self.capture_status.configure(
                text=f"Hold keyframe {self.kf_current + 1}/{self.kf_total}…"
            )
        else:
            self.capture_status.configure(text="Hold still pose…")
        self._beep("start")

    def _finish_record(self) -> None:
        self.capturing = False
        self.capture_start = None
        name = self.sign_name.get().strip()
        buf = self.capture_buffer
        self.capture_buffer = []
        n = len(buf)

        if self.train_kind == "keyframe":
            added = self.trainer.add_keyframe_samples(
                name, self.kf_current, buf, self.kf_total
            )
        else:
            added = self.trainer.add_static_samples(name, buf)

        self.record_btn.configure(state="normal", text="Record pose  (R)" if self.train_kind == "static" else "Record keyframe pose  (R)")
        try:
            self.capture_prog.pack_forget()
        except Exception:
            pass
        self._beep("end")

        if added == 0:
            self.capture_status.configure(text=f"No usable frames ({n}). Hands in view?")
        else:
            if self.train_kind == "keyframe":
                self.capture_status.configure(
                    text=f"Saved {added} samples for “{name}” step {self.kf_current + 1}/{self.kf_total}."
                )
                # advance to next keyframe
                if self.kf_current + 1 < self.kf_total:
                    self.kf_current += 1
                    self._update_kf_step_label()
                    self.capture_status.configure(
                        text=f"Step {self.kf_current}/{self.kf_total} done. Record step {self.kf_current + 1} next."
                    )
                else:
                    self.capture_status.configure(
                        text=f"All {self.kf_total} keyframes sampled for “{name}”. Train when ready."
                    )
                    self.kf_current = 0
                    self._update_kf_step_label()
            else:
                self.capture_status.configure(text=f"Saved {added} static samples for “{name.lower()}”.")

        self.train_status.configure(text=self._blurb())
        self._refresh_library()
        gc.collect()

    def _train(self) -> None:
        if self.is_training:
            return
        if not self.trainer.can_train():
            self.train_status.configure(text="Need ≥8 samples total. Record more.")
            return
        self.is_training = True
        self.train_btn.configure(state="disabled", text="Training…")
        self.train_status.configure(text="Training…")

        def worker():
            try:
                stats = self.trainer.train()
                msg = (f"Trained {stats['signs']} signs / {stats['samples']} samples "
                       f"· fit {stats['accuracy']*100:.0f}%")
            except Exception as e:
                msg = str(e)

            def done():
                self.is_training = False
                self.train_btn.configure(state="normal", text="Train model  (T)")
                self.train_status.configure(text=msg)
                self._set_feedback("Model updated")
                try:
                    self._refresh_neural_labels()
                except Exception:
                    pass
                gc.collect()
            self.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def _refresh_library(self) -> None:
        for c in self.lib_box.winfo_children():
            c.destroy()
        counts = self.trainer.counts()
        if not counts:
            ctk.CTkLabel(self.lib_box, text="No signs yet.", text_color=self.theme["MUTED"]).pack(
                anchor="w", padx=4, pady=6)
            return
        ctk.CTkLabel(
            self.lib_box,
            text="Click a sign name to view example image from /signexamples",
            text_color=self.theme["MUTED"], font=ctk.CTkFont(size=11),
        ).pack(anchor="w", padx=4, pady=(0, 6))
        for label, n in counts.items():
            row = ctk.CTkFrame(self.lib_box, fg_color="transparent")
            row.pack(fill="x", pady=1)
            kind = "KF" if self.trainer.is_keyframe_sign(label) else "S"
            warn = n < MIN_SAMPLES_WARN
            color = self.theme["DANGER"] if warn else self.theme["TEXT"]
            extra = f"  [{kind}]"
            if self.trainer.is_keyframe_sign(label):
                extra += f" ×{self.trainer.keyframe_signs[label]}"
            if warn:
                extra += "  ⚠"
            name_btn = ctk.CTkButton(
                row, text=f"{label}  ({n}){extra}",
                anchor="w", fg_color="transparent", hover_color="#1a1a24",
                text_color=color, height=28,
                command=lambda l=label: self._show_example(l),
            )
            name_btn.pack(side="left", fill="x", expand=True, padx=2)
            ctk.CTkButton(row, text="Delete", width=55, height=22, fg_color="#3a2030",
                          hover_color=self.theme["DANGER"],
                          command=lambda l=label: self._delete(l)).pack(side="right", padx=3)

    def _delete(self, label: str) -> None:
        self.trainer.delete_label(label)
        self.train_status.configure(text=self._blurb())
        self._refresh_library()

    def _speak(self) -> None:
        text = " ".join(p for p in self.sentence_parts if p).strip()
        if not text:
            return
        if self._tts_busy:
            self._stop_tts()
            return
        vol = self.tts_volume
        voice = self.tts_voice
        self._tts_stop.clear()
        self._tts_busy = True
        self._set_feedback("Speaking… (Stop to cancel)")

        def run():
            eng = None
            try:
                with self._tts_lock:
                    import pyttsx3
                    eng = pyttsx3.init()
                    eng.setProperty("volume", max(0.0, min(1.0, vol)))
                    try:
                        rate = eng.getProperty("rate")
                        eng.setProperty("rate", max(130, int(rate * 0.88)))
                    except Exception:
                        pass
                    if voice:
                        for v in eng.getProperty("voices") or []:
                            if (v.name or "") == voice or (v.id or "") == voice:
                                eng.setProperty("voice", v.id)
                                break
                    if self._tts_stop.is_set():
                        return
                    eng.say(text)
                    eng.runAndWait()
            except Exception as e:
                print("TTS:", e)
            finally:
                if eng is not None:
                    try:
                        eng.stop()
                    except Exception:
                        pass
                self._tts_busy = False

        threading.Thread(target=run, daemon=True).start()

    def _export(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".zip", filetypes=[("Zip", "*.zip")],
            initialfile="signbridge_data.zip",
        )
        if not path:
            return
        try:
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
                for name in ("samples.npz", "model.joblib", "meta.json", "keyframes.json", "settings.json"):
                    p = DATA_DIR / name
                    if p.exists():
                        zf.write(p, name)
            self.export_status.configure(text=f"Exported → {Path(path).name}")
        except Exception as e:
            self.export_status.configure(text=f"Failed: {e}")

    def _import(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Zip", "*.zip")])
        if not path:
            return
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(DATA_DIR)
            self.trainer = SignTrainer()
            self.train_status.configure(text=self._blurb())
            self._refresh_library()
            self.export_status.configure(text=f"Imported {Path(path).name}")
        except Exception as e:
            self.export_status.configure(text=f"Failed: {e}")

    def _on_key(self, event) -> None:
        key = event.keysym.lower()
        w = self.focus_get()
        if w is not None and w.__class__.__name__ in ("CTkEntry", "Entry"):
            return
        if key == "r":
            self._start_record()
        elif key == "t":
            self._train()
        elif key == "n":
            self._neural_guess()
        elif key == "s":
            self._speak()
        elif key == "u":
            self._undo()
        elif key == "space":
            self._space()
        elif key == "backspace":
            self._backspace()
        elif key == "escape":
            if self.capturing:
                self.capturing = False
                self.capture_start = None
                self.capture_buffer.clear()
                self.record_btn.configure(state="normal", text="Record pose  (R)")
                try:
                    self.capture_prog.pack_forget()
                except Exception:
                    pass
                self.capture_status.configure(text="Cancelled.")

    def _close(self) -> None:
        self._save_settings()
        try:
            self.camera.stop()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    SignBridgeApp().mainloop()
