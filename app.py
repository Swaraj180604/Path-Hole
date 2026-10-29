"""RoadScan AI - pothole & speed-breaker detection (single-file Streamlit app).

Everything (Python logic + HTML/CSS front end) lives in this one file, so it
deploys with just app.py and requirements.txt.
"""
from __future__ import annotations

import hashlib
import io
import tempfile
from html import escape
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ===========================================================================
# FRONT END: CSS + HTML templates
# ===========================================================================
CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');

:root {
  --bg: #0B0F17;
  --panel: #131A26;
  --panel-2: #182131;
  --line: #243049;
  --text: #E8ECF3;
  --muted: #8B97AD;
  --amber: #FFB020;
  --red: #FF5C5C;
  --teal: #2DD4BF;
  --blue: #6EA8FE;
  --radius: 14px;
}

html, body, .stApp, [class*="css"] {
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
}
#MainMenu, footer { visibility: hidden; }
.block-container { padding: 1.6rem 1.5rem 3rem !important; max-width: 1200px; }
h4 { letter-spacing: -0.01em; }

/* ---------- Hero ---------- */
.hero {
  position: relative;
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: 20px;
  background:
    radial-gradient(900px 300px at 90% -20%, rgba(255,176,32,.18), transparent 60%),
    radial-gradient(700px 300px at -10% 120%, rgba(110,168,254,.14), transparent 60%),
    var(--panel);
  margin-bottom: 1.4rem;
}
.hazard-bar {
  height: 10px;
  background: repeating-linear-gradient(-45deg, var(--amber) 0 14px, #10131a 14px 28px);
}
.hero-inner { padding: 2rem 2.2rem 2.2rem; }
.eyebrow {
  font-size: .72rem; font-weight: 700; letter-spacing: .18em;
  color: var(--amber); text-transform: uppercase;
}
.hero h1 {
  margin: .35rem 0 .5rem; padding: 0;
  font-size: clamp(2rem, 5vw, 3.2rem); font-weight: 800; letter-spacing: -0.03em;
  color: var(--text);
}
.hero h1 span { color: var(--amber); }
.hero p { color: var(--muted); max-width: 640px; font-size: 1.02rem; line-height: 1.6; margin: 0 0 1.1rem; }
.chips { display: flex; flex-wrap: wrap; gap: .5rem; }
.chip {
  font-size: .76rem; font-weight: 600; color: var(--text);
  background: rgba(255,255,255,.05); border: 1px solid var(--line);
  padding: .3rem .7rem; border-radius: 999px;
}

/* ---------- Stat cards ---------- */
.stat-grid {
  display: grid; gap: .9rem; margin: .6rem 0 1.2rem;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
}
.stat {
  background: var(--panel); border: 1px solid var(--line);
  border-radius: var(--radius); padding: 1rem 1.1rem;
  transition: transform .15s ease, border-color .15s ease;
}
.stat:hover { transform: translateY(-2px); border-color: #34456b; }
.stat-label { font-size: .74rem; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); font-weight: 600; }
.stat-value { font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; margin-top: .2rem; color: var(--text); }
.stat-value.red { color: var(--red); }
.stat-value.amber { color: var(--amber); }
.stat-value.teal { color: var(--teal); }
.stat-value.blue { color: var(--blue); }
.stat-sub { font-size: .8rem; color: var(--muted); margin-top: .15rem; }

/* ---------- Events table ---------- */
.table-wrap {
  max-height: 420px; overflow: auto;
  border: 1px solid var(--line); border-radius: var(--radius); background: var(--panel);
}
.evt-table { width: 100%; border-collapse: collapse; font-size: .9rem; }
.evt-table th {
  position: sticky; top: 0; background: var(--panel-2); z-index: 1;
  text-align: left; padding: .7rem 1rem; font-size: .72rem;
  text-transform: uppercase; letter-spacing: .1em; color: var(--muted);
  border-bottom: 1px solid var(--line);
}
.evt-table td { padding: .65rem 1rem; border-bottom: 1px solid rgba(36,48,73,.6); color: var(--text); vertical-align: middle; }
.evt-table tr:last-child td { border-bottom: none; }
.evt-table tbody tr:hover { background: rgba(255,255,255,.03); }
.mono { font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: .85rem; }
.table-note { color: var(--muted); font-size: .8rem; margin-top: .5rem; }

.badge {
  display: inline-block; padding: .2rem .6rem; border-radius: 999px;
  font-size: .76rem; font-weight: 700;
}
.badge.pothole { background: rgba(255,92,92,.15); color: var(--red); border: 1px solid rgba(255,92,92,.4); }
.badge.speed_breaker { background: rgba(255,176,32,.14); color: var(--amber); border: 1px solid rgba(255,176,32,.4); }

.sev { font-size: .76rem; font-weight: 700; padding: .15rem .5rem; border-radius: 6px; }
.sev.low { background: rgba(45,212,191,.14); color: var(--teal); }
.sev.medium { background: rgba(255,176,32,.14); color: var(--amber); }
.sev.high { background: rgba(255,92,92,.16); color: var(--red); }

.meter {
  display: inline-block; vertical-align: middle; width: 80px; height: 6px;
  border-radius: 99px; background: var(--line); margin-right: .6rem; overflow: hidden;
}
.meter span { display: block; height: 100%; background: linear-gradient(90deg, var(--blue), var(--teal)); }

/* ---------- Callout ---------- */
.callout {
  background: var(--panel); border: 1px solid var(--line);
  border-left: 4px solid var(--amber); border-radius: var(--radius);
  padding: 1rem 1.2rem; margin: .8rem 0;
}
.callout h5 { margin: 0 0 .3rem; font-size: 1rem; color: var(--text); }
.callout p { margin: 0; color: var(--muted); font-size: .92rem; line-height: 1.55; }
.callout code { background: rgba(255,255,255,.07); padding: .1rem .35rem; border-radius: 5px; font-size: .84rem; }

/* ---------- Feature cards ---------- */
.feature-grid {
  display: grid; gap: .9rem; margin: .6rem 0 1.2rem;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
}
.feature {
  background: var(--panel); border: 1px solid var(--line);
  border-radius: var(--radius); padding: 1.2rem;
}
.feature .icon {
  width: 40px; height: 40px; border-radius: 10px; display: grid; place-items: center;
  background: rgba(255,176,32,.12); font-size: 1.3rem; margin-bottom: .7rem;
}
.feature h5 { margin: 0 0 .3rem; font-size: 1rem; color: var(--text); }
.feature p { margin: 0; font-size: .88rem; color: var(--muted); line-height: 1.55; }

/* ---------- Footer ---------- */
.site-footer {
  margin-top: 2.5rem; padding-top: 1.1rem; border-top: 1px solid var(--line);
  color: var(--muted); font-size: .82rem; text-align: center;
}

/* ---------- Streamlit widget polish ---------- */
.stTabs [data-baseweb="tab-list"] { gap: .4rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] {
  height: 46px; padding: 0 1.1rem; border-radius: 10px 10px 0 0; font-weight: 600;
}
.stTabs [aria-selected="true"] { color: var(--amber) !important; }
.stButton > button, .stDownloadButton > button {
  border-radius: 10px; font-weight: 600; border: 1px solid var(--line);
  background: var(--panel-2); transition: all .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: var(--amber); color: var(--amber);
}
[data-testid="stSidebar"] { border-right: 1px solid var(--line); }
[data-testid="stFileUploader"] section { border-radius: var(--radius); border: 1px dashed #34456b; background: var(--panel); }

@media (max-width: 640px) {
  .hero-inner { padding: 1.4rem 1.2rem 1.6rem; }
  .meter { width: 50px; }
}

/* ---- self-contained theme (works without .streamlit/config.toml) ---- */
.stApp { background: var(--bg); color: var(--text); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: var(--panel); }
.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp h5,.stApp label,.stApp p,.stApp li,
.stApp [data-testid="stMarkdownContainer"], .stApp .stCaption { color: var(--text); }
.stApp [data-testid="stCaptionContainer"] { color: var(--muted); }

"""

TEMPLATES = {
    'hero.html': '<div class="hero">\n  <div class="hazard-bar"></div>\n  <div class="hero-inner">\n    <span class="eyebrow">Road intelligence</span>\n    <h1>Road<span>Scan</span> AI</h1>\n    <p>\n      Detect potholes and speed breakers from phone accelerometer logs or\n      dashcam photos. Upload a recording, see every road defect on a timeline,\n      and export the results.\n    </p>\n    <div class="chips">\n      <span class="chip">Python</span>\n      <span class="chip">scikit-learn</span>\n      <span class="chip">YOLOv8</span>\n      <span class="chip">Streamlit</span>\n    </div>\n  </div>\n</div>\n',
    'stat_card.html': '<div class="stat">\n  <div class="stat-label">$label</div>\n  <div class="stat-value $tone">$value</div>\n  <div class="stat-sub">$sub</div>\n</div>\n',
    'feature_card.html': '<div class="feature">\n  <div class="icon">$icon</div>\n  <h5>$title</h5>\n  <p>$body</p>\n</div>\n',
    'callout.html': '<div class="callout">\n  <h5>$title</h5>\n  <p>$body</p>\n</div>\n',
    'footer.html': '<div class="site-footer">\n  RoadScan AI &middot; Built with Python, scikit-learn, YOLOv8 and Streamlit\n  &middot; Results are indicative, not a substitute for a professional road survey.\n</div>\n',
}

# ===========================================================================
# SENSOR MODEL (accelerometer)
# ===========================================================================
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


# ===========================================================================
# UI HELPERS + OPTIONAL YOLO
# ===========================================================================

PRETTY = {"pothole": "Pothole", "speed_breaker": "Speed breaker"}


def _compact(text: str) -> str:
    """Strip indentation / blank lines so Markdown never treats HTML as a code block."""
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def inject_css() -> None:
    st.markdown(f"<style>{_compact(CSS)}</style>", unsafe_allow_html=True)


def _template(name: str, **values) -> str:
    return _compact(Template(TEMPLATES[name]).safe_substitute(**values))


def render(name: str, **values) -> None:
    st.markdown(_template(name, **values), unsafe_allow_html=True)


def stat_grid(items: list[dict]) -> None:
    cards = "".join(
        _template("stat_card.html", label=escape(str(i["label"])), value=escape(str(i["value"])),
                  sub=escape(str(i.get("sub", ""))), tone=i.get("tone", ""))
        for i in items
    )
    st.markdown(f'<div class="stat-grid">{cards}</div>', unsafe_allow_html=True)


def feature_grid(items: list[dict]) -> None:
    cards = "".join(
        _template("feature_card.html", icon=i["icon"], title=escape(i["title"]), body=escape(i["body"]))
        for i in items
    )
    st.markdown(f'<div class="feature-grid">{cards}</div>', unsafe_allow_html=True)


def events_table(events: pd.DataFrame, limit: int = 200) -> None:
    rows = []
    for r in events.head(limit).itertuples():
        pct = int(round(r.confidence * 100))
        rows.append(
            "<tr>"
            f'<td class="mono">{r.peak_time_s:.1f} s</td>'
            f'<td><span class="badge {r.type}">{PRETTY[r.type]}</span></td>'
            f'<td class="mono">{r.peak_ms2:.1f} m/s²</td>'
            f'<td><span class="sev {r.severity.lower()}">{r.severity}</span></td>'
            f'<td><div class="meter"><span style="width:{pct}%"></span></div>'
            f'<span class="mono">{pct}%</span></td>'
            "</tr>"
        )
    note = ""
    if len(events) > limit:
        note = f'<div class="table-note">Showing first {limit} of {len(events)} events - download the CSV for all.</div>'
    html = (
        '<div class="table-wrap"><table class="evt-table"><thead><tr>'
        "<th>Time</th><th>Type</th><th>Peak shock</th><th>Severity</th><th>Confidence</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>{note}"
    )
    st.markdown(html, unsafe_allow_html=True)


def yolo_available() -> bool:
    try:
        import ultralytics  # noqa: F401
        return True
    except Exception:
        return False


def yolo_load(weights_path: str):
    from ultralytics import YOLO
    return YOLO(weights_path)


def yolo_predict(model, rgb: np.ndarray, conf: float = 0.4):
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    res = model.predict(bgr, conf=conf, verbose=False)[0]
    annotated = np.ascontiguousarray(res.plot()[..., ::-1])
    rows = []
    for box in res.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        rows.append({"class": res.names[int(box.cls[0])], "confidence": float(box.conf[0]),
                     "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return annotated, pd.DataFrame(rows, columns=["class", "confidence", "x1", "y1", "x2", "y2"])


# ===========================================================================
# APP
# ===========================================================================
ROOT = Path(__file__).parent
DEFAULT_WEIGHTS = ROOT / "models" / "best.pt"
COLORS = {"pothole": "#FF5C5C", "speed_breaker": "#FFB020"}

st.set_page_config(page_title="RoadScan AI", page_icon="🚧", layout="wide")
inject_css()
render("hero.html")


# --------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training the detector on synthetic road data…")
def get_model():
    return train_model()


@st.cache_resource(show_spinner=False)
def get_report():
    return evaluate(get_model())


@st.cache_data(show_spinner=False)
def demo_drive(seed: int, minutes: int) -> pd.DataFrame:
    return make_signal(minutes * 60, seed)


@st.cache_resource(show_spinner="Loading YOLO weights…")
def load_yolo(path: str):
    return yolo_load(path)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Detection settings")
    min_conf = st.slider(
        "Minimum confidence", 0.30, 0.95, 0.50, 0.05,
        help="Sensor events below this confidence are ignored. "
             "Raise it for fewer false alarms, lower it to catch more bumps.",
    )
    st.caption("Applies to the accelerometer analysis.")
    st.divider()
    st.markdown("##### Tips for recording")
    st.caption(
        "Mount the phone flat and fixed in the vehicle. Log vertical (z) "
        "acceleration at 50 Hz or more. Keep speed fairly steady."
    )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _guess(cols: list[str], keys: tuple[str, ...], fallback: int) -> int:
    low = [c.lower() for c in cols]
    for k in keys:
        if k in low:
            return low.index(k)
    for k in keys:
        if len(k) > 1:
            for i, c in enumerate(low):
                if k in c:
                    return i
    return min(fallback, len(cols) - 1)


def signal_chart(df: pd.DataFrame, events: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=df["t"], y=df["az"], mode="lines", name="Vertical acceleration",
        line=dict(color="#6EA8FE", width=1),
    ))
    for r in events.itertuples():
        fig.add_vrect(
            x0=r.start_s, x1=r.end_s, fillcolor=COLORS[r.type],
            opacity=0.28, line_width=0, layer="below",
        )
    for t, label in (("pothole", "Pothole"), ("speed_breaker", "Speed breaker")):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=label,
            marker=dict(size=12, color=COLORS[t], symbol="square"),
        ))
    fig.update_layout(
        template="plotly_dark", height=400,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="Time (s)", rangeslider=dict(visible=True, thickness=0.06),
                   range=[0, min(60, float(df["t"].iloc[-1]))]),
        yaxis=dict(title="Acceleration (m/s²)"),
        legend=dict(orientation="h", y=1.08, x=0),
    )
    return fig


def show_sensor_results(signal: pd.DataFrame, events: pd.DataFrame, note: str = "") -> None:
    n_pot = int((events["type"] == "pothole").sum())
    n_sb = int((events["type"] == "speed_breaker").sum())
    avg_conf = f"{events['confidence'].mean() * 100:.0f}%" if len(events) else "–"
    duration = float(signal["t"].iloc[-1])
    stat_grid([
        {"label": "Recording", "value": f"{duration / 60:.1f} min", "sub": f"{len(signal):,} samples @ {FS} Hz", "tone": "blue"},
        {"label": "Potholes", "value": n_pot, "sub": "sharp, short shocks", "tone": "red"},
        {"label": "Speed breakers", "value": n_sb, "sub": "smooth, longer bumps", "tone": "amber"},
        {"label": "Avg confidence", "value": avg_conf, "sub": note or "across detected events", "tone": "teal"},
    ])
    st.plotly_chart(signal_chart(signal, events), width="stretch")
    if events.empty:
        render("callout.html", title="No events found",
                  body="Nothing crossed the detection threshold. Try lowering the minimum confidence in the sidebar.")
        return
    st.markdown("##### Detected events")
    events_table(events)
    st.write("")
    out = events.drop(columns=["class_id"]).round(3)
    st.download_button("⬇ Download events (CSV)", out.to_csv(index=False).encode(),
                       "road_events.csv", "text/csv")


# --------------------------------------------------------------------------
# Tab 1 - accelerometer
# --------------------------------------------------------------------------
def sensor_tab() -> None:
    st.markdown("#### Accelerometer analysis")
    source = st.radio("Data source", ["Demo drive", "Upload CSV"], horizontal=True)
    note = ""

    if source == "Demo drive":
        c1, c2 = st.columns(2)
        minutes = c1.slider("Length (minutes)", 1, 10, 5)
        seed = int(c2.number_input("Random seed", 0, 9999, 7))
        d = demo_drive(seed, minutes)
        true_events = int((np.diff((d["label"].to_numpy() > 0).astype(int)) == 1).sum())
        signal = d[["t", "az"]]
        note = f"{true_events} true events in demo"
    else:
        sample = ROOT / "sample_data" / "sample_drive.csv"
        up = st.file_uploader("Accelerometer CSV", type=["csv"])
        if sample.exists():
            st.download_button("⬇ Download a sample CSV", sample.read_bytes(),
                               "sample_drive.csv", "text/csv")
        if up is None:
            render("callout.html", title="Upload a recording to begin",
                      body="Export a CSV from a phone sensor app such as Phyphox or Sensor Logger. "
                           "It needs a time column and a vertical (z) acceleration column.")
            return
        try:
            raw = pd.read_csv(io.BytesIO(up.getvalue()))
        except Exception:
            st.error("Could not read that file as CSV.")
            return
        num_cols = raw.select_dtypes("number").columns.tolist()
        if len(num_cols) < 2:
            st.error("The CSV needs at least two numeric columns (time and acceleration).")
            return
        c1, c2, c3, c4 = st.columns(4)
        tcol = c1.selectbox("Time column", num_cols,
                            index=_guess(num_cols, ("time", "t", "timestamp", "seconds"), 0))
        acol = c2.selectbox("Vertical (z) acceleration", num_cols,
                            index=_guess(num_cols, ("az", "z", "accel_z", "acc_z"), len(num_cols) - 1))
        tunit = c3.selectbox("Time unit", ["seconds", "milliseconds", "nanoseconds"])
        aunit = c4.selectbox("Acceleration unit", ["m/s²", "g"])
        try:
            signal = prepare_signal(raw, tcol, acol, tunit, aunit)
        except ValueError as e:
            st.error(str(e))
            return

    try:
        events = detect_events(signal["az"].to_numpy(), get_model(), min_conf)
    except ValueError as e:
        st.error(str(e))
        return
    show_sensor_results(signal, events, note)


# --------------------------------------------------------------------------
# Tab 2 - camera
# --------------------------------------------------------------------------
def _resolve_weights() -> str | None:
    if DEFAULT_WEIGHTS.exists():
        return str(DEFAULT_WEIGHTS)
    with st.expander("Model weights", expanded=True):
        st.caption("Upload your trained YOLOv8 weights (best.pt). Only upload files you trust - "
                   "model files can contain executable code.")
        w = st.file_uploader("best.pt", type=["pt"], key="weights")
    if w is None:
        return None
    data = w.getvalue()
    path = Path(tempfile.gettempdir()) / f"roadscan_{hashlib.sha1(data).hexdigest()[:12]}.pt"
    if not path.exists():
        path.write_bytes(data)
    return str(path)


def camera_tab() -> None:
    st.markdown("#### Camera detection")
    if not yolo_available():
        render(
            "callout.html", title="Vision module not installed",
            body="Camera detection needs <code>torch</code> and <code>ultralytics</code>. Add them to "
                 "<code>requirements.txt</code> (uncomment the optional lines at the bottom) "
                 "and redeploy. The accelerometer tab works without it.",
        )
        return
    weights = _resolve_weights()
    if weights is None:
        render(
            "callout.html", title="No trained weights yet",
            body="Train a YOLOv8 model on a pothole dataset (see the notebook), then place the "
                 "result at <code>models/best.pt</code> in the repo or upload it above.",
        )
        return

    conf = st.slider("Detection confidence", 0.10, 0.90, 0.40, 0.05, key="vconf")
    mode = st.radio("Input", ["Upload images", "Take a photo"], horizontal=True)
    if mode == "Upload images":
        files = st.file_uploader("Road images", type=["png", "jpg", "jpeg", "webp"],
                                 accept_multiple_files=True)
        images = [(f.name, Image.open(f).convert("RGB")) for f in files or []]
    else:
        shot = st.camera_input("Point the camera at the road")
        images = [("photo", Image.open(shot).convert("RGB"))] if shot else []
    if not images:
        return

    model = load_yolo(weights)
    total = 0
    for name, img in images:
        annotated, dets = yolo_predict(model, np.array(img), conf)
        total += len(dets)
        left, right = st.columns([3, 2])
        left.image(annotated, caption=name, width="stretch")
        with right:
            stat_grid([{"label": "Detections", "value": len(dets), "tone": "amber"}])
            if len(dets):
                st.dataframe(dets[["class", "confidence"]].round(2), hide_index=True,
                             width="stretch")
            else:
                st.caption("Nothing detected at this confidence.")
        st.divider()


# --------------------------------------------------------------------------
# Tab 3 - how it works
# --------------------------------------------------------------------------
def about_tab() -> None:
    st.markdown("#### How it works")
    feature_grid([
        {"icon": "📈", "title": "Windowing", "body": "The vertical acceleration is resampled to 50 Hz and cut into 1.5 s windows that overlap."},
        {"icon": "🧮", "title": "Shape features", "body": "Each window gets features such as peak jerk, kurtosis and RMS. Potholes are sharp and short; speed breakers are smooth and long."},
        {"icon": "🌲", "title": "Random Forest", "body": "A classifier labels each window as normal, pothole or speed breaker, then neighbouring windows merge into events."},
        {"icon": "📷", "title": "YOLOv8 vision", "body": "Optionally, a YOLOv8 model finds defects in dashcam photos before the vehicle reaches them."},
    ])
    rep = get_report()
    st.markdown("##### Model quality on a held-out synthetic drive")
    stat_grid([
        {"label": "Window accuracy", "value": f"{rep['accuracy'] * 100:.0f}%", "tone": "teal"},
        {"label": "Pothole recall", "value": f"{rep['pothole_recall'] * 100:.0f}%", "sub": f"precision {rep['pothole_precision'] * 100:.0f}%", "tone": "red"},
        {"label": "Speed breaker recall", "value": f"{rep['speed_breaker_recall'] * 100:.0f}%", "sub": f"precision {rep['speed_breaker_precision'] * 100:.0f}%", "tone": "amber"},
    ])
    render("callout.html", title="Synthetic data caveat",
              body="These scores come from simulated roads and will look better than real-world results. "
                   "Train on labelled real drives, at different speeds and vehicles, before relying on it.")


tab1, tab2, tab3 = st.tabs(["📈 Accelerometer", "📷 Camera", "ℹ️ How it works"])
with tab1:
    sensor_tab()
with tab2:
    camera_tab()
with tab3:
    about_tab()

render("footer.html")
