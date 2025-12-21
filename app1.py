# =============================================================
# AI SOLAR FAULT & PANEL-WISE PERFORMANCE ESTIMATION
# DEPLOYMENT-SAFE VERSION (NO OPENCV)
# =============================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import requests
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
from datetime import date

# -------------------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------------------
st.set_page_config(
    page_title="AI Solar Fault Detection & Energy Estimation",
    layout="wide"
)

# -------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------
FAULT_LOSS = {
    "Non-Defective": (0.00, 0.00),
    "Clean": (0.00, 0.00),
    "Bird-drop": (0.12, 0.22),
    "Dusty": (0.11, 0.15),
    "Snow-Covered": (0.10, 0.34),
    "Physical Damage": (0.25, 0.50),
    "Electrical-Damage": (0.30, 0.70),
}

USAGE_CAPACITY = {
    "Home": 5,
    "Office / Commercial": 20,
    "Factory / Warehouse": 200,
    "Agriculture / Village": 10,
    "Solar Power Plant": 1000,
}

# -------------------------------------------------------------
# LOCATION → LAT/LON (OPEN-METEO)
# -------------------------------------------------------------
def get_lat_lon(city):
    try:
        url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {"name": city, "count": 1, "language": "en", "format": "json"}
        data = requests.get(url, params=params, timeout=10).json()
        if "results" not in data:
            return None, None
        r = data["results"][0]
        return r["latitude"], r["longitude"]
    except:
        return None, None

# -------------------------------------------------------------
# SUNLIGHT HOURS
# -------------------------------------------------------------
def get_sunlight(lat, lon, d):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": d,
            "end_date": d,
            "daily": "sunshine_duration",
            "timezone": "auto",
        }
        data = requests.get(url, params=params, timeout=10).json()
        return round(data["daily"]["sunshine_duration"][0] / 3600, 2)
    except:
        return None

# -------------------------------------------------------------
# CLOUD COVER
# -------------------------------------------------------------
def get_cloud(lat, lon, d):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": d,
            "end_date": d,
            "hourly": "cloudcover",
            "timezone": "auto",
        }
        data = requests.get(url, params=params, timeout=10).json()
        return sum(data["hourly"]["cloudcover"]) / len(data["hourly"]["cloudcover"])
    except:
        return 0

# -------------------------------------------------------------
# LOAD YOLO MODELS (NO CACHE FOR SAFETY)
# -------------------------------------------------------------
@st.cache_resource
def load_model(path):
    return YOLO(path, task="detect")

with st.spinner("🔄 Loading AI Models..."):
    try:
        fault_model = load_model("models/best.pt")
        snow_model = load_model("models/snow.pt")
        panel_model = load_model("models/panel_detect.pt")
    except Exception as e:
        st.error("❌ Failed to load YOLO models")
        st.exception(e)
        st.stop()

# -------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------
def normalize_label(lbl):
    mapping = {
        "bird": "Bird-drop",
        "clean": "Non-Defective",
        "dusty": "Dusty",
        "snow": "Snow-Covered",
        "physical damage": "Physical Damage",
        "electrical-damage": "Electrical-Damage",
    }
    return mapping.get(lbl.lower(), lbl)

def avg_loss(lbl):
    lo, hi = FAULT_LOSS.get(lbl, (0, 0))
    return (lo + hi) / 2

def iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0
    areaA = (a[2]-a[0])*(a[3]-a[1])
    areaB = (b[2]-b[0])*(b[3]-b[1])
    return inter / (areaA + areaB - inter)

# -------------------------------------------------------------
# PANEL DETECTION
# -------------------------------------------------------------
def detect_panels(img, conf):
    arr = np.array(img)
    res = panel_model.predict(arr, conf=conf, verbose=False)[0]

    boxes = []
    for b in res.boxes:
        name = res.names[int(b.cls[0])].lower()
        if "panel" in name:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            boxes.append((float(b.conf[0]), (x1, y1, x2, y2)))

    boxes.sort(key=lambda x: x[1][0])
    return boxes

# -------------------------------------------------------------
# FAULT DETECTION
# -------------------------------------------------------------
def detect_faults(img, conf):
    arr = np.array(img)
    res = fault_model.predict(arr, conf=conf, verbose=False)[0]

    dets = [(normalize_label(res.names[int(b.cls[0])]),
             float(b.conf[0]),
             tuple(map(int, b.xyxy[0])))
            for b in res.boxes]

    if dets:
        return dets, Image.fromarray(res.plot())

    res2 = snow_model.predict(arr, conf=conf, verbose=False)[0]
    dets2 = [(normalize_label(res2.names[int(b.cls[0])]),
              float(b.conf[0]),
              tuple(map(int, b.xyxy[0])))
             for b in res2.boxes]

    return dets2, Image.fromarray(res2.plot())

# -------------------------------------------------------------
# UI
# -------------------------------------------------------------
st.title("☀️ AI Solar Fault Detection & Energy Estimation")

usage = st.selectbox("Site Type", list(USAGE_CAPACITY.keys()))
capacity = st.number_input(
    "System Capacity (kW)",
    value=float(USAGE_CAPACITY[usage]),
    min_value=0.1
)

city = st.text_input("City / Village")
selected_date = st.date_input("Select Date", value=date.today(), max_value=date.today())

sunlight = None
if city:
    lat, lon = get_lat_lon(city)
    if lat:
        d = selected_date.strftime("%Y-%m-%d")
        raw = get_sunlight(lat, lon, d)
        cloud = get_cloud(lat, lon, d)
        sunlight = round(raw * (1 - cloud / 100), 2)
        st.success(f"🌤 Effective Sunlight: {sunlight} hrs")

panel_conf = st.slider("Panel Detection Confidence", 0.1, 0.9, 0.4)
fault_conf = st.slider("Fault Detection Confidence", 0.1, 0.9, 0.5)

file = st.file_uploader("Upload Solar Panel Image", type=["jpg", "png", "jpeg"])

# -------------------------------------------------------------
# ANALYSIS
# -------------------------------------------------------------
if file and sunlight and st.button("Analyze"):
    img = Image.open(file).convert("RGB")

    panels = detect_panels(img, panel_conf)
    faults, fault_img = detect_faults(img, fault_conf)

    draw_img = img.copy()
    draw = ImageDraw.Draw(draw_img)

    for i, (_, (x1, y1, x2, y2)) in enumerate(panels, 1):
        draw.rectangle([x1, y1, x2, y2], outline="yellow", width=4)
        draw.text((x1, max(y1 - 15, 0)), f"Panel {i}", fill="yellow")

    col1, col2 = st.columns(2)
    col1.image(draw_img, caption="🟦 Panel Detection", use_container_width=True)
    col2.image(fault_img, caption="🔴 Fault Detection", use_container_width=True)

    total_loss = 0
    st.subheader("📊 Panel-wise Loss")

    for i, (_, pb) in enumerate(panels):
        loss = 0
        for lbl, conf, fb in faults:
            if iou(pb, fb) > 0.5:
                loss += avg_loss(lbl) * conf * capacity * sunlight
        total_loss += loss
        st.write(f"Panel {i+1}: **{loss:.2f} kWh/day loss**")

    total_energy = len(panels) * capacity
    st.success(f"🟢 Usable Energy: {max(total_energy - total_loss, 0):.2f} kWh/day")
