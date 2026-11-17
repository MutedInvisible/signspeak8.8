"""Samples, classifier, static + keyframe-sequence recognition."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import POSE_SIZE

DATA_DIR = Path(__file__).resolve().parent / "data"
SAMPLES_PATH = DATA_DIR / "samples.npz"
MODEL_PATH = DATA_DIR / "model.joblib"
META_PATH = DATA_DIR / "meta.json"
KEYFRAME_META = DATA_DIR / "keyframes.json"

# Internal label: "hello" static, "hello::k0" / "hello::k1" keyframe steps
KF_SEP = "::k"


class SignTrainer:
    def __init__(self) -> None:
        self.X: list[np.ndarray] = []
        self.y: list[str] = []  # includes ::kN for keyframe steps
        self.model: Pipeline | None = None
        # keyframe_signs: { "hello": 3 }  number of keyframes
        self.keyframe_signs: dict[str, int] = {}
        self._history: deque[str] = deque(maxlen=5)
        # live sequence state: sign -> next keyframe index, last advance time
        self._seq_progress: dict[str, int] = {}
        self._seq_last_t: dict[str, float] = {}
        self.load()

    def counts(self) -> dict[str, int]:
        """User-facing sign counts (static + keyframe groups)."""
        counts: dict[str, int] = {}
        for label in self.y:
            base = label.split(KF_SEP)[0]
            counts[base] = counts.get(base, 0) + 1
        return dict(sorted(counts.items()))

    def labels(self) -> list[str]:
        return list(self.counts().keys())

    def is_keyframe_sign(self, name: str) -> bool:
        return name.strip().lower() in self.keyframe_signs

    def add_static_samples(self, label: str, samples: list[np.ndarray]) -> int:
        name = label.strip().lower()
        if not name or not samples:
            return 0
        # Don't allow same name as keyframe sign
        if name in self.keyframe_signs:
            return 0
        added = 0
        for sample in samples:
            if sample is None or sample.shape != (POSE_SIZE,):
                continue
            self.X.append(sample.astype(np.float32))
            self.y.append(name)
            added += 1
        if added:
            self.save_samples()
        return added

    def add_keyframe_samples(self, label: str, keyframe_index: int, samples: list[np.ndarray], total_keyframes: int) -> int:
        name = label.strip().lower()
        if not name or not samples or total_keyframes < 2:
            return 0
        internal = f"{name}{KF_SEP}{keyframe_index}"
        added = 0
        for sample in samples:
            if sample is None or sample.shape != (POSE_SIZE,):
                continue
            self.X.append(sample.astype(np.float32))
            self.y.append(internal)
            added += 1
        if added:
            self.keyframe_signs[name] = total_keyframes
            self.save_samples()
            self._save_keyframe_meta()
        return added

    def delete_label(self, label: str) -> None:
        name = label.strip().lower()
        kept_x, kept_y = [], []
        for x, y in zip(self.X, self.y):
            base = y.split(KF_SEP)[0]
            if base != name:
                kept_x.append(x)
                kept_y.append(y)
        self.X, self.y = kept_x, kept_y
        self.keyframe_signs.pop(name, None)
        self.model = None
        self.save_samples()
        self._save_keyframe_meta()
        if MODEL_PATH.exists():
            try:
                MODEL_PATH.unlink()
            except Exception:
                pass

    def can_train(self) -> bool:
        bases = set(y.split(KF_SEP)[0] for y in self.y)
        return len(bases) >= 1 and len(self.y) >= 8

    def train(self) -> dict:
        if not self.can_train():
            raise ValueError("Need at least 8 samples across your signs to train.")

        # Validate keyframe signs have samples for every step
        for name, n_kf in list(self.keyframe_signs.items()):
            for i in range(n_kf):
                tag = f"{name}{KF_SEP}{i}"
                if not any(y == tag for y in self.y):
                    raise ValueError(f"Keyframe sign “{name}” missing samples for step {i + 1}/{n_kf}.")

        X = np.stack(self.X)
        y = np.array(self.y)
        forest = RandomForestClassifier(
            n_estimators=120,
            max_depth=16,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        )
        self.model = Pipeline([("scaler", StandardScaler()), ("clf", forest)])
        self.model.fit(X, y)
        acc = float(self.model.score(X, y))

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, MODEL_PATH, compress=3)
        META_PATH.write_text(
            json.dumps(
                {
                    "train_accuracy": acc,
                    "n_samples": int(len(y)),
                    "labels": self.labels(),
                    "keyframes": self.keyframe_signs,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self._save_keyframe_meta()
        self._history.clear()
        self._seq_progress.clear()
        return {"accuracy": acc, "samples": len(y), "signs": len(self.labels())}

    def predict_raw(self, features: np.ndarray | None) -> tuple[str | None, float]:
        """Raw class label (may include ::kN) + confidence."""
        if self.model is None or features is None or features.shape != (POSE_SIZE,):
            return None, 0.0
        try:
            proba = self.model.predict_proba([features])[0]
            idx = int(np.argmax(proba))
            return str(self.model.classes_[idx]), float(proba[idx])
        except Exception:
            return None, 0.0

    def predict_top(self, features: np.ndarray | None, k: int = 3) -> list[tuple[str, float]]:
        if self.model is None or features is None or features.shape != (POSE_SIZE,):
            return []
        try:
            proba = self.model.predict_proba([features])[0]
            classes = [str(c) for c in self.model.classes_]
            order = list(np.argsort(proba)[::-1][:k])
            return [(classes[i], float(proba[i])) for i in order]
        except Exception:
            return []

    def predict_live(
        self,
        features: np.ndarray | None,
        now: float,
        conf_threshold: float = 0.55,
        kf_step_timeout: float = 2.2,
        unknown_below: float = 0.22,
        kf_min_gap: float = 0.28,
        kf_conf_boost: float = 0.06,
    ) -> tuple[str | None, float, str]:
        """
        Returns (commit_label_or_None, display_confidence, display_text).
        Keyframes need order + min time between steps so fast flailing
        does not skip yes/no into the wrong sign.
        """
        raw, conf = self.predict_raw(features)
        if raw is None:
            self._expire_sequences(now, kf_step_timeout)
            return None, 0.0, "show a trained sign"

        if conf < unknown_below:
            self._expire_sequences(now, kf_step_timeout)
            return None, conf, "unknown"

        # If top is a keyframe step but a STATIC class is close, prefer static
        # (fixes "s" showing as "yes" when yes is a keyframe sign)
        if KF_SEP in raw:
            tops = self.predict_top(features, k=4)
            static_alt = None
            for lab, p in tops:
                if KF_SEP not in lab and lab not in self.keyframe_signs:
                    static_alt = (lab, p)
                    break
            if static_alt and static_alt[1] >= conf_threshold * 0.92:
                # static is strong enough — treat as static this frame
                raw, conf = static_alt[0], static_alt[1]
            else:
                pass  # keep keyframe raw

        # Keyframe step hit?
        if KF_SEP in raw:
            base, _, rest = raw.partition(KF_SEP)
            try:
                step = int(rest)
            except ValueError:
                return None, conf, raw
            n_kf = self.keyframe_signs.get(base, 0)
            if n_kf < 2:
                return None, conf, raw

            # Keyframes need slightly higher confidence than static
            need = min(0.92, conf_threshold + kf_conf_boost)
            if conf < need:
                expected = self._seq_progress.get(base, 0)
                return None, conf, f"{base} [{expected}/{n_kf}] …"

            expected = self._seq_progress.get(base, 0)
            last = self._seq_last_t.get(base, 0.0)
            if expected > 0 and now - last > kf_step_timeout:
                self._seq_progress[base] = 0
                expected = 0

            if step == expected:
                # Require a short gap between steps (stops same-frame spam / too-fast blur)
                if expected > 0 and (now - last) < kf_min_gap:
                    return None, conf, f"{base} [{expected}/{n_kf}]"
                self._seq_progress[base] = step + 1
                self._seq_last_t[base] = now
                if step + 1 >= n_kf:
                    self._seq_progress[base] = 0
                    return base, conf, base
                return None, conf, f"{base} [{step + 1}/{n_kf}]"
            elif step < expected:
                return None, conf, f"{base} [{expected}/{n_kf}]"
            else:
                return None, conf, f"{base} [{expected}/{n_kf}]"

        # Static class
        if raw in self.keyframe_signs:
            # predicted a keyframe base without step — ignore
            return None, conf, raw

        self._history.append(raw)
        if conf < conf_threshold:
            return None, conf, raw
        if self._history.count(raw) < 3:
            return None, conf, raw
        return raw, conf, raw  # caller still applies hold-time for static commit

    def _expire_sequences(self, now: float, timeout: float) -> None:
        for base in list(self._seq_progress.keys()):
            if self._seq_progress[base] > 0 and now - self._seq_last_t.get(base, 0) > timeout:
                self._seq_progress[base] = 0

    def save_samples(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not self.X:
            if SAMPLES_PATH.exists():
                try:
                    SAMPLES_PATH.unlink()
                except Exception:
                    pass
            return
        np.savez_compressed(
            SAMPLES_PATH,
            X=np.array(self.X, dtype=object),
            y=np.array(self.y),
        )

    def _save_keyframe_meta(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        KEYFRAME_META.write_text(json.dumps(self.keyframe_signs, indent=2), encoding="utf-8")

    def load(self) -> None:
        if SAMPLES_PATH.exists():
            try:
                blob = np.load(SAMPLES_PATH, allow_pickle=True)
                self.X = [np.asarray(row, dtype=np.float32) for row in blob["X"]]
                self.y = [str(label) for label in blob["y"]]
            except Exception as e:
                print("load samples:", e)
                self.X, self.y = [], []
        if KEYFRAME_META.exists():
            try:
                self.keyframe_signs = json.loads(KEYFRAME_META.read_text(encoding="utf-8"))
            except Exception:
                self.keyframe_signs = {}
        # recover keyframe_signs from labels if meta missing
        if not self.keyframe_signs:
            recovered: dict[str, int] = {}
            for y in self.y:
                if KF_SEP in y:
                    base, _, rest = y.partition(KF_SEP)
                    try:
                        step = int(rest)
                        recovered[base] = max(recovered.get(base, 0), step + 1)
                    except ValueError:
                        pass
            self.keyframe_signs = recovered
        if MODEL_PATH.exists():
            try:
                payload = joblib.load(MODEL_PATH)
                if isinstance(payload, dict):
                    # old format
                    self.model = payload.get("static") or payload.get("model")
                else:
                    self.model = payload
            except Exception as e:
                print("load model:", e)
                self.model = None
