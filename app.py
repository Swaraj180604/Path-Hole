"""RoadScan AI - pothole & speed-breaker detection (Streamlit app)."""
from __future__ import annotations

import hashlib
import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from core import sensor, ui, vision

ROOT = Path(__file__).parent
DEFAULT_WEIGHTS = ROOT / "models" / "best.pt"
COLORS = {"pothole": "#FF5C5C", "speed_breaker": "#FFB020"}

st.set_page_config(page_title="RoadScan AI", page_icon="🚧", layout="wide")
ui.inject_css()
ui.render("hero.html")


# --------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training the detector on synthetic road data…")
def get_model():
    return sensor.train_model()


@st.cache_resource(show_spinner=False)
def get_report():
    return sensor.evaluate(get_model())


@st.cache_data(show_spinner=False)
def demo_drive(seed: int, minutes: int) -> pd.DataFrame:
    return sensor.make_signal(minutes * 60, seed)


@st.cache_resource(show_spinner="Loading YOLO weights…")
def load_yolo(path: str):
    return vision.load_model(path)


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
    ui.stat_grid([
        {"label": "Recording", "value": f"{duration / 60:.1f} min", "sub": f"{len(signal):,} samples @ {sensor.FS} Hz", "tone": "blue"},
        {"label": "Potholes", "value": n_pot, "sub": "sharp, short shocks", "tone": "red"},
        {"label": "Speed breakers", "value": n_sb, "sub": "smooth, longer bumps", "tone": "amber"},
        {"label": "Avg confidence", "value": avg_conf, "sub": note or "across detected events", "tone": "teal"},
    ])
    st.plotly_chart(signal_chart(signal, events), width="stretch")
    if events.empty:
        ui.render("callout.html", title="No events found",
                  body="Nothing crossed the detection threshold. Try lowering the minimum confidence in the sidebar.")
        return
    st.markdown("##### Detected events")
    ui.events_table(events)
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
            ui.render("callout.html", title="Upload a recording to begin",
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
            signal = sensor.prepare_signal(raw, tcol, acol, tunit, aunit)
        except ValueError as e:
            st.error(str(e))
            return

    try:
        events = sensor.detect_events(signal["az"].to_numpy(), get_model(), min_conf)
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
    if not vision.is_available():
        ui.render(
            "callout.html", title="Vision module not installed",
            body="Camera detection needs <code>ultralytics</code>. Add it to "
                 "<code>requirements.txt</code> (see README, section Enable camera detection) "
                 "and redeploy. The accelerometer tab works without it.",
        )
        return
    weights = _resolve_weights()
    if weights is None:
        ui.render(
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
        annotated, dets = vision.predict(model, np.array(img), conf)
        total += len(dets)
        left, right = st.columns([3, 2])
        left.image(annotated, caption=name, width="stretch")
        with right:
            ui.stat_grid([{"label": "Detections", "value": len(dets), "tone": "amber"}])
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
    ui.feature_grid([
        {"icon": "📈", "title": "Windowing", "body": "The vertical acceleration is resampled to 50 Hz and cut into 1.5 s windows that overlap."},
        {"icon": "🧮", "title": "Shape features", "body": "Each window gets features such as peak jerk, kurtosis and RMS. Potholes are sharp and short; speed breakers are smooth and long."},
        {"icon": "🌲", "title": "Random Forest", "body": "A classifier labels each window as normal, pothole or speed breaker, then neighbouring windows merge into events."},
        {"icon": "📷", "title": "YOLOv8 vision", "body": "Optionally, a YOLOv8 model finds defects in dashcam photos before the vehicle reaches them."},
    ])
    rep = get_report()
    st.markdown("##### Model quality on a held-out synthetic drive")
    ui.stat_grid([
        {"label": "Window accuracy", "value": f"{rep['accuracy'] * 100:.0f}%", "tone": "teal"},
        {"label": "Pothole recall", "value": f"{rep['pothole_recall'] * 100:.0f}%", "sub": f"precision {rep['pothole_precision'] * 100:.0f}%", "tone": "red"},
        {"label": "Speed breaker recall", "value": f"{rep['speed_breaker_recall'] * 100:.0f}%", "sub": f"precision {rep['speed_breaker_precision'] * 100:.0f}%", "tone": "amber"},
    ])
    ui.render("callout.html", title="Synthetic data caveat",
              body="These scores come from simulated roads and will look better than real-world results. "
                   "Train on labelled real drives, at different speeds and vehicles, before relying on it.")


tab1, tab2, tab3 = st.tabs(["📈 Accelerometer", "📷 Camera", "ℹ️ How it works"])
with tab1:
    sensor_tab()
with tab2:
    camera_tab()
with tab3:
    about_tab()

ui.render("footer.html")
