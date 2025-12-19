# =============================================================
# AI SOLAR FAULT & PANEL-WISE PERFORMANCE ESTIMATION
# CITY INPUT → AUTO LAT/LON (OPEN-METEO)
# DATE → LIVE SUNLIGHT HOURS (OPEN-METEO)
# CLOUD COVER → REDUCED SUNLIGHT
# DATE VALIDATION → NO FUTURE DATES ALLOWED
# =============================================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import requests
from PIL import Image
import numpy as np
import cv2
from datetime import date
from ultralytics import YOLO

# -------------------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------------------
st.set_page_config(page_title="AI Solar Fault Detection & Performance Estimation", layout="wide")

# -------------------------------------------------------------
# FAULT LOSS CONFIG
# -------------------------------------------------------------
FAULT_LOSS = {
    "Non-Defective": (0.00, 0.00),
    "Clean": (0.00, 0.00),
    "Bird-drop": (0.121, 0.221),
    "Dusty": (0.115, 0.150),
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
# CITY → LAT/LON (OPEN-METEO GEOCODING)
# -------------------------------------------------------------
def get_lat_lon_from_city(city):
    try:
        url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {
            "name": city,
            "count": 1,
            "language": "en",
            "format": "json"
        }
        data = requests.get(url, params=params, timeout=10).json()

        if "results" not in data or not data["results"]:
            return None, None

        return data["results"][0]["latitude"], data["results"][0]["longitude"]
    except:
        return None, None

# -------------------------------------------------------------
# LIVE SUNSHINE HOURS (OPEN-METEO)
# -------------------------------------------------------------
def get_sunlight_hours(lat, lon, date_str):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": date_str,
            "end_date": date_str,
            "daily": "sunshine_duration",
            "timezone": "auto"
        }
        data = requests.get(url, params=params, timeout=10).json()

        sunshine_seconds = data["daily"]["sunshine_duration"][0]
        return round(sunshine_seconds / 3600, 2)
    except:
        return None

# -------------------------------------------------------------
# LIVE CLOUD COVER (OPEN-METEO)
# -------------------------------------------------------------
def get_cloud_cover(lat, lon, date_str):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": date_str,
            "end_date": date_str,
            "hourly": "cloudcover",
            "timezone": "auto"
        }
        data = requests.get(url, params=params, timeout=10).json()

        clouds = data["hourly"]["cloudcover"]
        return sum(clouds) / len(clouds)
    except:
        return 0

# -------------------------------------------------------------
# LOAD YOLO MODELS
# -------------------------------------------------------------
# @st.cache_resource
def load_fault_model(path):
    return YOLO(path)

# @st.cache_resource
def load_panel_model():
    return YOLO("models/panel_detect.pt")

with st.spinner("Loading AI Models..."):
    primary_model = load_fault_model("models/best.pt")
    snow_model = load_fault_model("snow.pt")
    panel_model = load_panel_model()

# -------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------
def normalize_label(raw):
    mapping = {
        "bird": "Bird-drop",
        "clean": "Non-Defective",
        "dusty": "Dusty",
        "snow": "Snow-Covered",
        "physical damage": "Physical Damage",
        "electrical-damage": "Electrical-Damage",
    }
    return mapping.get(raw.lower(), raw)

def avg_severity(label):
    low, high = FAULT_LOSS.get(label, (0, 0))
    return (low + high) / 2

def calculate_iou(a, b):
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
        lbl = res.names[int(b.cls[0])].lower()
        if "panel" in lbl:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            boxes.append([x1, y1, x2, y2, float(b.conf[0])])

    if not boxes:
        return []

    boxes.sort(key=lambda x: x[4], reverse=True)

    final = []
    for b in boxes:
        if not any(calculate_iou(b[:4], f[:4]) > 0.4 for f in final):
            final.append(b)

    final.sort(key=lambda x: x[0])
    return [(b[4], (b[0], b[1], b[2], b[3])) for b in final]

# -------------------------------------------------------------
# FAULT DETECTION
# -------------------------------------------------------------
def detect_faults(img, conf):
    res = primary_model.predict(img, conf=conf, verbose=False)[0]

    dets = [
        (normalize_label(res.names[int(b.cls[0])]), float(b.conf[0]), tuple(map(int, b.xyxy[0])))
        for b in res.boxes
    ]

    if any(lbl not in ("Clean", "Non-Defective") for lbl,_,_ in dets):
        return dets, np.asarray(res.plot())

    res2 = snow_model.predict(img, conf=conf, verbose=False)[0]
    dets2 = [
        (normalize_label(res2.names[int(b.cls[0])]), float(b.conf[0]), tuple(map(int, b.xyxy[0])))
        for b in res2.boxes
    ]
    return dets2, np.asarray(res2.plot())

# -------------------------------------------------------------
# PANEL LOSS (UNCHANGED)
# -------------------------------------------------------------
def styled_card(lbl, conf, loss_frac, cap, sun):
    return f"""
    <div style='background:#8B0000;padding:12px;color:white;border-radius:10px;margin:5px;'>
        <b>{lbl}</b> ({conf*100:.1f}%)
        <br>Loss: {loss_frac*100:.2f}% | {(loss_frac*cap*sun):.2f} kWh/day
    </div>
    """

def show_panels_and_loss(dets, cap, sun, panel_boxes):
    total_loss = 0
    fault_map = {i: [] for i in range(len(panel_boxes))}

    for lbl, c, (fx1, fy1, fx2, fy2) in dets:
        fault_area = max((fx2-fx1)*(fy2-fy1), 1)
        best_panel = -1
        best_ratio = 0

        for i, (_, (px1, py1, px2, py2)) in enumerate(panel_boxes):
            ix1, iy1 = max(px1, fx1), max(py1, fy1)
            ix2, iy2 = min(px2, fx2), min(py2, fy2)

            if ix2 > ix1 and iy2 > iy1:
                overlap = (ix2-ix1)*(iy2-iy1)
                ratio = overlap / fault_area
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_panel = i

        if best_panel != -1 and best_ratio >= 0.5:
            fault_map[best_panel].append((lbl, c))

    for idx, (conf, box) in enumerate(panel_boxes):
        st.markdown(f"## 🟦 Panel {idx+1}")
        panel_faults = fault_map[idx]

        if panel_faults:
            for lbl, c in panel_faults:
                lf = avg_severity(lbl) * c
                lk = lf * cap * sun
                total_loss += lk
                st.markdown(styled_card(lbl, c, lf, cap, sun), unsafe_allow_html=True)
        else:
            st.success("Clean — No Faults ✔")

    total_energy = len(panel_boxes) * cap
    usable = max(total_energy - total_loss, 0)

    st.subheader("📊 Final Energy Summary")
    st.info(f"🔋 Total Energy Stored: **{total_energy:.2f} kWh/day**")
    st.error(f"⚡ Total Damaged Energy: **{total_loss:.2f} kWh/day**")
    st.success(f"🟢 Final Usable Energy: **{usable:.2f} kWh/day**")

# -------------------------------------------------------------
# USER INTERFACE
# -------------------------------------------------------------
st.title("☀️ AI Solar Fault Detection & Energy Estimation")
st.caption("City → Date → Live Sunlight Hours → Cloud Reduction → AI Fault Detection")

usage = st.selectbox("Site Type", list(USAGE_CAPACITY.keys()))
capacity = st.number_input("System Capacity (kW)", 0.1, value=float(USAGE_CAPACITY[usage]))

st.subheader("📍 Enter Location")
city = st.text_input("City or Village Name")

selected_date = st.date_input(
    "Select Date",
    value=date.today(),
    max_value=date.today()
)

sunlight = None

if city.strip():
    lat, lon = get_lat_lon_from_city(city.strip())

    if lat is not None and lon is not None:
        st.success(f"📌 Location Found: {city}")

        date_str = selected_date.strftime("%Y-%m-%d")
        sunlight_raw = get_sunlight_hours(lat, lon, date_str)
        cloud = get_cloud_cover(lat, lon, date_str)

        if sunlight_raw is not None:
            sunlight = round(sunlight_raw * (1 - cloud / 100), 2)

            st.info(f"🌞 Raw Sunshine: {sunlight_raw} hours")
            st.warning(f"☁ Cloud Cover: {cloud:.1f}%")
            st.success(f"🌤 Effective Sunlight: **{sunlight} hours**")
        else:
            st.error("Could not fetch sunlight hours.")
            sunlight = st.number_input("Sunlight Hours", 1, 12, 5)
    else:
        st.error("❌ City not found.")
        sunlight = st.number_input("Sunlight Hours", 1, 12, 5)

panel_conf = st.slider("Panel Detection Confidence", 0.1, 0.9, 0.4)
fault_conf = st.slider("Fault Detection Confidence", 0.05, 0.99, 0.5)

file = st.file_uploader("Upload Solar Panel Image", type=["jpg", "jpeg", "png"])

if file and sunlight and st.button("Analyze"):
    img = Image.open(file).convert("RGB")

    panel_boxes = detect_panels(img, panel_conf)
    dets, fault_img = detect_faults(img, fault_conf)

    st.subheader("📦 Panels Detected")
    st.info(f"Total Panels: {len(panel_boxes)}")

    arr = np.array(img).copy()
    for i, (conf, (x1, y1, x2, y2)) in enumerate(panel_boxes, 1):
        cv2.rectangle(arr, (x1, y1), (x2, y2), (255, 255, 0), 3)
        cv2.putText(arr, f"Panel {i}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 3)

    col1, col2 = st.columns(2)
    col1.image(arr, caption="🟦 Panel Detection", use_container_width=True)
    col2.image(fault_img, caption="🔴 Fault Detection", use_container_width=True)

    show_panels_and_loss(dets, capacity, sunlight, panel_boxes)
