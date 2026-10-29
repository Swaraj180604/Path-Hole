# RoadScan AI - Pothole & Speed Breaker Detection

A Streamlit app with a custom HTML/CSS front end. It detects potholes and speed
breakers from phone accelerometer logs (scikit-learn) and, optionally, from
dashcam photos (YOLOv8).

## Project layout

```
roadscan-ai/
├── app.py                    # Streamlit app (pages, charts, upload handling)
├── roadscan/
│   ├── sensor.py             # accelerometer features + Random Forest
│   ├── vision.py             # optional YOLOv8 wrapper
│   └── ui.py                 # renders the HTML/CSS templates
├── assets/
│   ├── style.css             # all custom styling
│   └── templates/*.html      # hero, stat cards, callouts, footer, feature cards
├── notebooks/                # Jupyter notebook (training + experiments)
├── sample_data/              # example accelerometer CSV
├── models/                   # put best.pt here for the Camera tab
├── tests/                    # pytest tests
├── .streamlit/config.toml    # dark theme + upload limit
├── requirements.txt
├── packages.txt              # apt packages for OpenCV on Streamlit Cloud
└── .github/workflows/ci.yml  # runs tests on every push
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. Run the tests with `pip install pytest && pytest`.

## Deploy with GitHub + Streamlit Community Cloud

1. Create an empty repository on GitHub (for example `roadscan-ai`).
2. Push this folder:
   ```bash
   cd roadscan-ai
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/<your-username>/roadscan-ai.git
   git push -u origin main
   ```
3. Go to https://share.streamlit.io and sign in with GitHub.
4. Click **Create app** → **Deploy a public app from GitHub**.
5. Choose your repository, branch `main`, and main file path `app.py`.
6. Open **Advanced settings** and pick Python 3.11 or 3.12, then click **Deploy**.

Every `git push` to `main` redeploys the app automatically.

## Enable camera detection (optional)

1. Train YOLOv8 on a pothole dataset using the notebook and copy the result to
   `models/best.pt` (or upload it in the app).
2. In `requirements.txt`, uncomment the `torch` / `ultralytics` lines.
3. Commit and push.

PyTorch is large. If Streamlit Community Cloud runs out of resources, deploy the
same repo to Hugging Face Spaces (Streamlit SDK), which has more room.

## Using your own accelerometer data

Record with a phone app (Phyphox, Sensor Logger) with the phone fixed flat in the
vehicle. Export a CSV with a time column and a vertical (z) acceleration column,
then upload it in the **Accelerometer** tab and pick the columns and units.

## Notes

- The bundled model is trained on **synthetic** data, so real-world accuracy will
  be lower. Train on labelled real drives for production use.
- Only upload YOLO `.pt` weights you trust; model files can contain executable code.
