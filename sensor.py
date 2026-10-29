"""Accelerometer-based pothole / speed-breaker detection.

A pothole produces a short, sharp drop-and-rebound spike (~0.2-0.4 s).
A speed breaker produces a longer, smoother bump (~0.8-1.5 s).
We slice the vertical acceleration into windows, extract shape features and
classify each window with a Random Forest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

FS = 50  # sampling rate (Hz) everything is resampled to
LABELS = {0: "normal", 1: "pothole", 2: "speed_breaker"}
WIN = int(1.5 * FS)
STEP = int(0.5 * FS)
FEATURES = [
    "std", "ptp", "min", "max", "peak_jerk",
    "rms", "skew", "kurtosis", "n_above_1ms2",
]
G = 9.80665


# --------------------------------------------------------------------------
# Synthetic data
# --------------------------------------------------------------------------
def make_signal(duration_s: int = 600, seed: int = 0) -> pd.DataFrame:
    """Simulate a vertical accelerometer (m/s^2) with labelled events."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * FS)
    t = np.arange(n) / FS
    z = 9.81 + rng.normal(0, 0.25, n) + 0.3 * np.sin(2 * np.pi * 0.5 * t)
    y = np.zeros(n, dtype=int)
    pos = 5 * FS
    while pos < n - 4 * FS:
        kind = int(rng.choice([1, 2]))
        if kind == 1:  # pothole: sharp drop + rebound
            dur = int(rng.uniform(0.2, 0.4) * FS)
            amp = rng.uniform(2.5, 5.0)
            k = np.arange(dur)
            pulse = -amp * np.sin(2 * np.pi * k / dur) * np.exp(-2 * k / dur)
        else:  # speed breaker: smooth, longer bump
            dur = int(rng.uniform(0.8, 1.5) * FS)
            amp = rng.uniform(1.5, 3.0)
            k = np.arange(dur)
            pulse = amp * np.sin(np.pi * k / dur) - 0.5 * amp * np.sin(2 * np.pi * k / dur)
        z[pos:pos + dur] += pulse
        y[pos:pos + dur] = kind
        pos += dur + int(rng.uniform(6, 15) * FS)
    return pd.DataFrame({"t": t, "az": z, "label": y})


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def _window_features(w: np.ndarray) -> list[float]:
    w = w - w.mean()
    jerk = np.diff(w) * FS
    return [
        w.std(), np.ptp(w), w.min(), w.max(),
        np.abs(jerk).max(), np.sqrt((w ** 2).mean()),
        skew(w), kurtosis(w), int((np.abs(w) > 1.0).sum()),
    ]


def build_windows(az: np.ndarray, labels: np.ndarray | None = None):
    X, y, starts = [], [], []
    for s in range(0, len(az) - WIN, STEP):
        X.append(_window_features(az[s:s + WIN]))
        starts.append(s)
        if labels is not None:
            counts = np.bincount(labels[s:s + WIN], minlength=3)
            y.append(int(np.argmax(counts[1:]) + 1) if counts[1:].max() >= 10 else 0)
    X = pd.DataFrame(X, columns=FEATURES)
    return X, np.array(y), np.array(starts)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def train_model(duration_s: int = 900, seed: int = 0) -> RandomForestClassifier:
    df = make_signal(duration_s, seed)
    X, y, _ = build_windows(df["az"].to_numpy(), df["label"].to_numpy())
    clf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
    )
    clf.fit(X, y)
    return clf


def evaluate(clf: RandomForestClassifier, seed: int = 123) -> dict:
    """Window-level metrics on a separate synthetic drive."""
    df = make_signal(300, seed)
    X, y, _ = build_windows(df["az"].to_numpy(), df["label"].to_numpy())
    pred = clf.predict(X)
    rep = classification_report(
        y, pred, labels=[0, 1, 2], target_names=list(LABELS.values()),
        output_dict=True, zero_division=0,
    )
    return {
        "accuracy": rep["accuracy"],
        "pothole_recall": rep["pothole"]["recall"],
        "pothole_precision": rep["pothole"]["precision"],
        "speed_breaker_recall": rep["speed_breaker"]["recall"],
        "speed_breaker_precision": rep["speed_breaker"]["precision"],
    }


# --------------------------------------------------------------------------
# Real data helpers
# --------------------------------------------------------------------------
def prepare_signal(
    raw: pd.DataFrame,
    time_col: str,
    az_col: str,
    time_unit: str = "seconds",
    accel_unit: str = "m/s²",
) -> pd.DataFrame:
    """Clean a raw CSV and resample it to FS Hz. Returns columns t, az (m/s^2)."""
    t = pd.to_numeric(raw[time_col], errors="coerce").to_numpy(float)
    a = pd.to_numeric(raw[az_col], errors="coerce").to_numpy(float)
    ok = np.isfinite(t) & np.isfinite(a)
    t, a = t[ok], a[ok]
    if len(t) < 2:
        raise ValueError("Not enough valid numeric rows in the selected columns.")
    t = t / {"seconds": 1.0, "milliseconds": 1e3, "nanoseconds": 1e9}[time_unit]
    order = np.argsort(t)
    t, a = t[order], a[order]
    keep = np.diff(t, prepend=t[0] - 1) > 0
    t, a = t[keep], a[keep]
    t = t - t[0]
    if t[-1] < (WIN / FS) + 0.5:
        raise ValueError("Recording is too short - need at least a few seconds of data.")
    if accel_unit == "g":
        a = a * G
    grid = np.arange(0, t[-1], 1 / FS)
    return pd.DataFrame({"t": grid, "az": np.interp(grid, t, a)})


def _severity(peak: float) -> str:
    return "Low" if peak < 2.5 else "Medium" if peak < 4.0 else "High"


def detect_events(
    az: np.ndarray, clf: RandomForestClassifier, min_conf: float = 0.5
) -> pd.DataFrame:
    """Classify windows and merge consecutive ones into events."""
    az = np.asarray(az, dtype=float)
    X, _, starts = build_windows(az)
    if len(X) == 0:
        raise ValueError("Recording is too short to analyse.")
    proba = clf.predict_proba(X)
    pred = clf.classes_[proba.argmax(1)]
    conf = proba.max(1)
    pred = np.where((pred != 0) & (conf < min_conf), 0, pred)

    spans, cur = [], None
    for p, c, s in zip(pred, conf, starts):
        t0, t1 = s / FS, (s + WIN) / FS
        if p != 0 and cur is not None and cur["cls"] == p and t0 <= cur["end"]:
            cur["end"] = t1
            cur["confs"].append(c)
        else:
            if cur is not None:
                spans.append(cur)
                cur = None
            if p != 0:
                cur = {"cls": int(p), "start": t0, "end": t1, "confs": [c]}
    if cur is not None:
        spans.append(cur)

    dev = az - np.median(az)
    rows = []
    for sp in spans:
        i0, i1 = int(sp["start"] * FS), min(len(az), int(sp["end"] * FS))
        seg = dev[i0:i1]
        k = int(np.argmax(np.abs(seg)))
        peak = float(abs(seg[k]))
        rows.append({
            "start_s": sp["start"],
            "end_s": sp["end"],
            "peak_time_s": (i0 + k) / FS,
            "class_id": sp["cls"],
            "type": LABELS[sp["cls"]],
            "peak_ms2": peak,
            "severity": _severity(peak),
            "confidence": float(np.mean(sp["confs"])),
        })
    cols = ["start_s", "end_s", "peak_time_s", "class_id", "type",
            "peak_ms2", "severity", "confidence"]
    return pd.DataFrame(rows, columns=cols)
