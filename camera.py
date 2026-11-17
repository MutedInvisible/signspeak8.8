"""Webcam + MediaPipe hands — steady FPS, no UI work on this thread."""

from __future__ import annotations

import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from features import HAND_CONNECTIONS, landmarks_to_features

FrameCallback = Callable[[np.ndarray, np.ndarray | None, int], None]
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"


def ensure_model() -> Path:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    min_size = 3_000_000
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size >= min_size:
        return MODEL_PATH
    if MODEL_PATH.exists():
        try:
            MODEL_PATH.unlink()
        except Exception:
            pass
    print(f"Downloading hand model → {MODEL_PATH}")
    tmp = MODEL_PATH.with_suffix(".task.part")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    if not tmp.exists() or tmp.stat().st_size < min_size:
        raise RuntimeError("Model download failed / corrupt.")
    tmp.replace(MODEL_PATH)
    return MODEL_PATH


def list_cameras(max_test: int = 5) -> list[int]:
    found = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                found.append(i)
            cap.release()
        else:
            try:
                cap.release()
            except Exception:
                pass
    return found or [0]


def _draw_hands(frame: np.ndarray, hands) -> None:
    h, w = frame.shape[:2]
    for hand in hands:
        pts = hand.landmark if hasattr(hand, "landmark") else hand
        pixels = [(int(lm.x * w), int(lm.y * h)) for lm in pts]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pixels[a], pixels[b], (45, 212, 191), 1, cv2.LINE_AA)
        for x, y in pixels:
            cv2.circle(frame, (x, y), 2, (110, 231, 208), -1, cv2.LINE_AA)


class HandCamera:
    def __init__(self, on_frame: FrameCallback, camera_index: int = 0) -> None:
        self.on_frame = on_frame
        self.camera_index = camera_index
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None
        self.mirror = True

    def set_mirror(self, enabled: bool) -> None:
        self.mirror = enabled

    def set_camera_index(self, index: int) -> None:
        if index == self.camera_index:
            return
        self.camera_index = index
        self.stop()
        time.sleep(0.15)
        self.start()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="hand-cam")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def _make_landmarker(self):
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker,
            HandLandmarkerOptions,
            RunningMode,
        )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model())),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return HandLandmarker.create_from_options(options)

    def _run(self) -> None:
        self._cap = self._open()
        if not self._cap.isOpened():
            blank = np.zeros((360, 480, 3), dtype=np.uint8)
            cv2.putText(blank, "Camera not found", (40, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 255), 2)
            self.on_frame(blank, None, 0)
            return

        import mediapipe as mp

        try:
            ensure_model()
            landmarker = self._make_landmarker()
        except Exception as e:
            blank = np.zeros((360, 480, 3), dtype=np.uint8)
            cv2.putText(blank, "Model error", (40, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 255), 2)
            self.on_frame(blank, None, 0)
            print("landmarker:", e)
            return

        stamp = 0
        target_dt = 1.0 / 30.0  # aim ~30 detect FPS steady

        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    time.sleep(0.005)
                    continue

                if self.mirror:
                    frame = cv2.flip(frame, 1)

                h, w = frame.shape[:2]
                if w > 640:
                    frame = cv2.resize(frame, (640, int(h * 640 / w)), interpolation=cv2.INTER_LINEAR)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if not rgb.flags["C_CONTIGUOUS"]:
                    rgb = np.ascontiguousarray(rgb)

                stamp += 33
                features = None
                n_hands = 0
                try:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = landmarker.detect_for_video(mp_image, stamp)
                    hands = result.hand_landmarks
                    handed = result.handedness
                    if hands:
                        n_hands = len(hands)
                        features = landmarks_to_features(hands, handed)
                        _draw_hands(frame, hands)
                except Exception as ex:
                    print("detect:", ex)

                self.on_frame(frame, features, n_hands)
            except Exception as ex:
                print("camera loop:", ex)
                time.sleep(0.01)

            # pace to ~30fps without burning CPU into thermal throttle
            elapsed = time.perf_counter() - t0
            sleep_for = target_dt - elapsed
            if sleep_for > 0.001:
                time.sleep(sleep_for)

        try:
            landmarker.close()
        except Exception:
            pass
        if self._cap is not None:
            self._cap.release()
            self._cap = None
