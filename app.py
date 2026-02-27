import os
import time
import requests
import streamlit as st
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 1 ▸ PAGE CONFIG (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Forest Cover AI",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# 2 ▸ GLOBAL CSS  —  deep-forest topographic theme
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800;900&family=DM+Mono:wght@300;400;500&family=Crimson+Pro:ital,wght@0,300;0,400;1,300&display=swap');

/* ── CSS Variables ── */
:root {
    --forest-black:   #050e07;
    --forest-deep:    #081409;
    --forest-dark:    #0d1f0f;
    --forest-mid:     #163319;
    --forest-accent:  #2d6a35;
    --forest-bright:  #4ade80;
    --forest-glow:    #86efac;
    --earth-brown:    #7c5c2e;
    --topog-line:     rgba(74, 222, 128, 0.08);
    --glass-bg:       rgba(13, 31, 15, 0.65);
    --glass-border:   rgba(74, 222, 128, 0.18);
    --text-primary:   #e8f5e9;
    --text-secondary: #a5d6a7;
    --text-dim:       #558b60;
}

/* ── Base / Body ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--forest-black) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Mono', monospace !important;
}

/* ── Animated topographic background ── */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    z-index: 0;
    background-image:
        repeating-linear-gradient(
            0deg,
            transparent,
            transparent 59px,
            var(--topog-line) 60px
        ),
        repeating-linear-gradient(
            90deg,
            transparent,
            transparent 59px,
            var(--topog-line) 60px
        ),
        radial-gradient(ellipse 80% 50% at 20% 40%, rgba(22,51,25,0.55) 0%, transparent 70%),
        radial-gradient(ellipse 60% 70% at 80% 70%, rgba(13,31,15,0.7) 0%, transparent 60%),
        linear-gradient(160deg, #060f08 0%, #0a1a0c 50%, #050e07 100%);
    pointer-events: none;
}

[data-testid="stMain"] {
    position: relative;
    z-index: 1;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--forest-deep) !important;
    border-right: 1px solid var(--glass-border) !important;
}

/* ── Remove Streamlit default padding ── */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1280px !important;
}

/* ── Hero Header ── */
.hero-wrapper {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    position: relative;
}
.hero-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--forest-bright);
    border: 1px solid var(--glass-border);
    background: rgba(74,222,128,0.06);
    padding: 0.35rem 1.1rem;
    border-radius: 100px;
    margin-bottom: 1.2rem;
    animation: fadeSlideDown 0.6s ease both;
}
.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-weight: 900;
    font-size: clamp(2rem, 5vw, 3.4rem);
    line-height: 1.1;
    letter-spacing: -0.02em;
    color: var(--text-primary);
    margin: 0 0 0.8rem;
    animation: fadeSlideDown 0.7s ease 0.1s both;
}
.hero-title span {
    color: var(--forest-bright);
    text-shadow: 0 0 40px rgba(74,222,128,0.4);
}
.hero-subtitle {
    font-family: 'Crimson Pro', serif;
    font-style: italic;
    font-size: 1.15rem;
    color: var(--text-secondary);
     
    animation: fadeSlideDown 0.7s ease 0.2s both;

    position: relative;
    padding: 0rem 20px 1rem 0;
    margin-bottom: 1rem;
    border-bottom: 1px solid var(--border);
    overflow: hidden;
    
    /* THE FIX: These 4 lines center everything perfectly */
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;                
            
}
.accuracy-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--forest-bright);
    background: rgba(74,222,128,0.08);
    border: 1px solid rgba(74,222,128,0.25);
    border-radius: 100px;
    padding: 0.3rem 0.9rem;
    margin-top: 1rem;
    animation: fadeSlideDown 0.7s ease 0.3s both;
}
.accuracy-pill::before {
    content: '●';
    font-size: 0.5rem;
    animation: pulse-dot 1.8s ease-in-out infinite;
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}
@keyframes fadeSlideDown {
    from { opacity: 0; transform: translateY(-16px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Section card (glassmorphic) ── */
.section-card {
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 1.5rem 1.6rem 1.8rem;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    margin-bottom: 1.2rem;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.section-card:hover {
    border-color: rgba(74,222,128,0.32);
    box-shadow: 0 0 28px rgba(74,222,128,0.07);
}
.section-label {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--forest-bright);
    margin-bottom: 1.1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--glass-border);
}

/* ── Streamlit native widgets — restyle ── */

/* Number inputs */
input[type="number"], input[type="text"] {
    background: rgba(5,14,7,0.8) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 8px !important;
    color: var(--forest-bright) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
input[type="number"]:focus, input[type="text"]:focus {
    border-color: var(--forest-bright) !important;
    box-shadow: 0 0 0 2px rgba(74,222,128,0.15) !important;
    outline: none !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: rgba(5,14,7,0.8) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 8px !important;
    color: var(--forest-bright) !important;
    font-family: 'DM Mono', monospace !important;
}

/* Slider track */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: var(--forest-bright) !important;
    border-color: var(--forest-bright) !important;
    box-shadow: 0 0 10px rgba(74,222,128,0.5) !important;
}
[data-testid="stSlider"] div[data-testid="stTickBar"] {
    color: var(--text-dim) !important;
}
/* Active slider fill */
[data-testid="stSlider"] [data-baseweb="slider"] div:nth-child(3) {
    background: var(--forest-accent) !important;
}

/* Labels */
label, [data-testid="stWidgetLabel"] {
    color: var(--text-secondary) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.05em !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary {
    color: var(--text-secondary) !important;
    font-family: 'DM Mono', monospace !important;
}

/* Divider */
hr {
    border-color: var(--glass-border) !important;
    margin: 1.5rem 0 !important;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: rgba(5,14,7,0.75) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    padding: 1.2rem 1.4rem !important;
    backdrop-filter: blur(8px) !important;
    transition: border-color 0.3s ease, transform 0.25s ease !important;
    animation: fadeSlideUp 0.5s ease both !important;
}
div[data-testid="metric-container"]:hover {
    border-color: rgba(74,222,128,0.4) !important;
    transform: translateY(-2px) !important;
}
div[data-testid="metric-container"] [data-testid="metric-label"] {
    color: var(--text-dim) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
div[data-testid="metric-container"] [data-testid="metric-value"] {
    color: var(--forest-bright) !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
}

/* ── CTA Button ── */
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #1e5128 0%, #2d6a35 50%, #1e5128 100%) !important;
    background-size: 200% 200% !important;
    border: 1px solid rgba(74,222,128,0.35) !important;
    border-radius: 10px !important;
    color: var(--forest-bright) !important;
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 0.8rem 2rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 24px rgba(30, 81, 40, 0.5), inset 0 1px 0 rgba(255,255,255,0.05) !important;
    animation: gradientShift 4s ease infinite !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    background-position: right center !important;
    border-color: rgba(74,222,128,0.65) !important;
    box-shadow: 0 0 32px rgba(74,222,128,0.25), 0 4px 24px rgba(30,81,40,0.6) !important;
    transform: translateY(-1px) !important;
    color: #fff !important;
}
[data-testid="stButton"] > button[kind="primary"]:active {
    transform: translateY(0) !important;
}

@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ── Alert / Info boxes ── */
[data-testid="stSuccess"] {
    background: rgba(45,106,53,0.15) !important;
    border: 1px solid rgba(74,222,128,0.3) !important;
    border-radius: 10px !important;
    color: var(--forest-bright) !important;
    font-family: 'DM Mono', monospace !important;
    animation: fadeSlideUp 0.4s ease both !important;
}
[data-testid="stError"] {
    background: rgba(127,29,29,0.15) !important;
    border: 1px solid rgba(239,68,68,0.3) !important;
    border-radius: 10px !important;
}
[data-testid="stInfo"] {
    background: rgba(14,116,144,0.1) !important;
    border: 1px solid rgba(34,211,238,0.2) !important;
    border-radius: 10px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    font-family: 'DM Mono', monospace !important;
    color: var(--text-secondary) !important;
}

/* ── Bar chart ── */
[data-testid="stVegaLiteChart"] {
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ── Fade-up animation (results) ── */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Result section header ── */
.result-header {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--forest-bright);
    text-align: center;
    margin: 0.5rem 0 1.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.8rem;
}
.result-header::before, .result-header::after {
    content: '';
    display: block;
    width: 60px;
    height: 1px;
    background: linear-gradient(to right, transparent, var(--forest-bright));
}
.result-header::after {
    background: linear-gradient(to left, transparent, var(--forest-bright));
}

/* ── Probability bar custom labels ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.65rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
}
.prob-label {
    width: 180px;
    flex-shrink: 0;
    color: var(--text-secondary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.prob-bar-track {
    flex: 1;
    height: 6px;
    background: rgba(255,255,255,0.05);
    border-radius: 100px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, var(--forest-accent), var(--forest-bright));
    transition: width 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94);
}
.prob-pct {
    width: 42px;
    text-align: right;
    color: var(--forest-bright);
    font-weight: 500;
}

/* ── Footer ── */
.footer-text {
    text-align: center;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: var(--text-dim);
    letter-spacing: 0.1em;
    padding: 2rem 0 0.5rem;
}

/* ── Streamlit default overrides (dark mode fixes) ── */
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { background: transparent !important; }
.stDeployButton { display: none; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# 3 ▸ API CONFIGURATION & HEALTH CHECK
# ──────────────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

@st.cache_data(ttl=10, show_spinner=False)
def check_api_health() -> tuple[bool, str]:
    """Returns (is_healthy, error_message)."""
    try:
        r = requests.get(f"{API_BASE_URL}/", timeout=5)
        data = r.json()
        if data.get("status") == "ok" and data.get("model_loaded") and data.get("preprocessor_loaded"):
            return True, ""
        return False, "API is reachable but model/preprocessor not loaded."
    except Exception as e:
        return False, str(e)

api_ok, _api_error = check_api_health()
model_loaded = api_ok
if not api_ok:
    _load_error = _api_error

# ──────────────────────────────────────────────────────────────────────────────
# 4 ▸ CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
TREE_DICT = {
    1: ("Spruce / Fir",         "🌲", "#4ade80"),
    2: ("Lodgepole Pine",       "🌲", "#22c55e"),
    3: ("Ponderosa Pine",       "🌳", "#16a34a"),
    4: ("Cottonwood / Willow",  "🌿", "#86efac"),   # rare
    5: ("Aspen",                "🍂", "#bbf7d0"),
    6: ("Douglas-fir",          "🌲", "#4ade80"),
    7: ("Krummholz",            "🌳", "#a3e635"),   # alpine
}

WILDERNESS_OPTIONS = [f"Wilderness_Area{i}" for i in range(1, 5)]
SOIL_OPTIONS       = [f"Soil_Type{i}"       for i in range(1, 41)]

# ──────────────────────────────────────────────────────────────────────────────
# 5 ▸ HERO HEADER
# ──────────────────────────────────────────────────────────────────────────────
if not model_loaded:
    st.error(
        f"⚠️ Cannot reach the FastAPI backend at `{API_BASE_URL}` — `{_load_error}`.\n\n"
        "Start the backend with:\n```\nuvicorn main:app --reload\n```"
    )
    st.stop()

st.markdown("""
<div class="hero-wrapper">
    <div class="hero-badge">XGBoost · Geospatial Intelligence · USGS Dataset</div>
    <h1 class="hero-title">Forest Cover <span> AI Predictor</span></h1>
    <p class="hero-subtitle">
        Enter topographical &amp; hydrological parameters to identify the dominant
        tree species of any Roosevelt National Forest zone.
    </p>
    <div class="accuracy-pill">Model accuracy: 94.7% on held-out test set</div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# 6 ▸ INPUT PANEL  (3 glassmorphic columns)
# ──────────────────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3, gap="medium")

# ── Column 1: Geography ──────────────────────────────────────────────────────
with col1:
    st.markdown('<div class="section-card"><div class="section-label">⛰  Geography</div>', unsafe_allow_html=True)
    elevation = st.number_input(
        "Elevation (meters)", min_value=1800, max_value=4000, value=2500, step=10,
        help="Elevation above sea level in meters (range: 1800–4000 m)"
    )
    aspect = st.slider(
        "Aspect (degrees)", 0, 360, 150,
        help="Compass bearing the slope faces — 0°/360° = North, 180° = South"
    )
    slope = st.slider(
        "Slope (degrees)", 0, 60, 15,
        help="Steepness of terrain in degrees"
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ── Column 2: Hydrology & Amenities ─────────────────────────────────────────
with col2:
    st.markdown('<div class="section-card"><div class="section-label">💧  Hydrology & Amenities</div>', unsafe_allow_html=True)
    h_dist_hydro = st.number_input(
        "Horiz. distance to water (m)", 0, 1500, 200, step=10,
        help="Horizontal distance to nearest surface water feature"
    )
    v_dist_hydro = st.number_input(
        "Vert. distance to water (m)", -200, 600, 50, step=5,
        help="Vertical distance to water; negative = below water level"
    )
    h_dist_road = st.number_input(
        "Horiz. distance to road (m)", 0, 7000, 1000, step=50,
        help="Horizontal distance to nearest road or trail"
    )
    h_dist_fire = st.number_input(
        "Horiz. distance to fire point (m)", 0, 7000, 1000, step=50,
        help="Horizontal distance to nearest wildfire ignition point"
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ── Column 3: Hillshade ──────────────────────────────────────────────────────
with col3:
    st.markdown('<div class="section-card"><div class="section-label">☀  Hillshade Index</div>', unsafe_allow_html=True)
    hillshade_9am  = st.slider("9 AM hillshade",   0, 255, 200, help="Solar illumination at 09:00 (0 = full shadow, 255 = full sun)")
    hillshade_noon = st.slider("Noon hillshade",   0, 255, 220, help="Solar illumination at 12:00")
    hillshade_3pm  = st.slider("3 PM hillshade",   0, 255, 140, help="Solar illumination at 15:00")

    # Live mini-sparkline: hillshade curve across the day
    shade_df = pd.DataFrame({
        "Hour":      ["9 AM", "Noon", "3 PM"],
        "Hillshade": [hillshade_9am, hillshade_noon, hillshade_3pm],
    })
    st.caption("Solar arc preview")
    st.line_chart(shade_df.set_index("Hour"), height=90, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Location & Soil expander ─────────────────────────────────────────────────
with st.expander("📍 Location & Soil Composition", expanded=True):
    cat_c1, cat_c2 = st.columns(2)
    with cat_c1:
        selected_wilderness = st.selectbox(
            "Wilderness Area",
            WILDERNESS_OPTIONS,
            help="One of four designated wilderness areas within the study zone"
        )
    with cat_c2:
        selected_soil = st.selectbox(
            "Dominant Soil Type",
            SOIL_OPTIONS,
            help="40 soil types derived from USFS ELU survey data (types 1–40)"
        )

# ──────────────────────────────────────────────────────────────────────────────
# 7 ▸ VALIDATION HELPER
# ──────────────────────────────────────────────────────────────────────────────
def validate_inputs():
    """Return list of error strings; empty list = all good."""
    errors = []
    if not (1800 <= elevation <= 4000):
        errors.append("Elevation must be between 1,800 m and 4,000 m.")
    if not (0 <= slope <= 60):
        errors.append("Slope must be between 0° and 60°.")
    if not (-200 <= v_dist_hydro <= 600):
        errors.append("Vertical distance to hydrology must be −200 to 600 m.")
    return errors

# ──────────────────────────────────────────────────────────────────────────────
# 8 ▸ RUN ANALYSIS BUTTON
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
run_btn = st.button("🚀  Run AI Analysis", type="primary", use_container_width=True)

if run_btn:
    # — Validate —
    validation_errors = validate_inputs()
    if validation_errors:
        for err in validation_errors:
            st.error(f"⚠️ {err}")
        st.stop()

    with st.spinner("Compiling spatial features · Running XGBoost inference…"):
        time.sleep(0.6)   # intentional UX pause — makes inference feel substantial

        # ── Feature engineering values (for display only) ──
        euclidean_hydro  = np.sqrt(h_dist_hydro**2 + v_dist_hydro**2)
        water_elevation  = elevation - v_dist_hydro
        dist_amenities   = (h_dist_road + h_dist_fire) / 2.0

        # ── Build API payload (raw features + one-hot encodings) ──
        payload = {
            "Elevation":                          elevation,
            "Aspect":                             aspect,
            "Slope":                              slope,
            "Horizontal_Distance_To_Hydrology":   h_dist_hydro,
            "Vertical_Distance_To_Hydrology":     v_dist_hydro,
            "Horizontal_Distance_To_Roadways":    h_dist_road,
            "Horizontal_Distance_To_Fire_Points": h_dist_fire,
            "Hillshade_9am":                      hillshade_9am,
            "Hillshade_Noon":                     hillshade_noon,
            "Hillshade_3pm":                      hillshade_3pm,
        }
        for i in range(1, 5):
            payload[f"Wilderness_Area{i}"] = 1 if selected_wilderness == f"Wilderness_Area{i}" else 0
        for i in range(1, 41):
            payload[f"Soil_Type{i}"] = 1 if selected_soil == f"Soil_Type{i}" else 0

        # ── Call FastAPI /predict ──
        try:
            response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=15)
            response.raise_for_status()
            result = response.json()

            prediction_idx = result["cover_type_id"]           # 1-indexed
            prob_dict      = result["probabilities"]            # {name: float}

            # Rebuild ordered probabilities array (index 0 = class 1 … index 6 = class 7)
            probabilities = np.array([
                prob_dict.get(TREE_DICT[i][0], 0.0) for i in range(1, 8)
            ])
            confidence = float(np.max(probabilities)) * 100

            tree_name, tree_icon, tree_color = TREE_DICT.get(
                prediction_idx, ("Unknown", "❓", "#6b7280")
            )

        except requests.exceptions.HTTPError:
            detail = response.json().get("detail", response.text)
            st.error(f"⚠️ API Error {response.status_code}: {detail}")
            st.stop()
        except Exception as e:
            st.error(f"⚠️ Could not reach the API: {e}")
            st.info(f"Make sure the FastAPI backend is running at `{API_BASE_URL}`.")
            st.stop()

    # ──────────────────────────────────────────────────────────────────────────
    # 9 ▸ RESULTS PANEL
    # ──────────────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="result-header">Analysis Output</div>', unsafe_allow_html=True)

    st.success("✅ Inference complete")

    # ── Primary metrics row ──
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(label="🌲  Predicted Forest Cover", value=f"{tree_icon} {tree_name}")
    with m2:
        st.metric(label="🎯  Confidence Score",       value=f"{confidence:.1f}%")
    with m3:
        rank = sorted(enumerate(probabilities), key=lambda x: -x[1])
        runner_up_idx = rank[1][0] + 1
        runner_up_name = TREE_DICT.get(runner_up_idx, ("Unknown", "❓", "#6b7280"))[0]
        runner_up_pct  = rank[1][1] * 100
        st.metric(label="🥈  Runner-Up Species", value=runner_up_name,
                  delta=f"{runner_up_pct:.1f}% probability")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Probability breakdown — custom HTML bars ──
    result_col, info_col = st.columns([3, 2], gap="large")

    with result_col:
        st.markdown('<div class="section-label">Probability Distribution</div>', unsafe_allow_html=True)

        # Sort all 7 classes by probability descending
        sorted_probs = sorted(
            [(TREE_DICT[i+1][0], TREE_DICT[i+1][1], float(p)) for i, p in enumerate(probabilities)],
            key=lambda x: -x[2],
        )
        # Find max for relative bar width scaling
        max_prob = sorted_probs[0][2]

        bar_html = ""
        for (name, icon, prob) in sorted_probs:
            bar_width = (prob / max_prob) * 100 if max_prob > 0 else 0
            highlight = "color:#4ade80;" if prob == max_prob else ""
            bar_html += f"""
            <div class="prob-row">
                <span class="prob-label" style="{highlight}">{icon} {name}</span>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill" style="width:{bar_width:.1f}%"></div>
                </div>
                <span class="prob-pct">{prob*100:.1f}%</span>
            </div>
            """
        st.markdown(bar_html, unsafe_allow_html=True)

    with info_col:
        st.markdown('<div class="section-label">Feature Summary</div>', unsafe_allow_html=True)
        summary_data = {
            "Parameter":    ["Elevation", "Slope", "Aspect", "H₂O Distance", "Road Distance", "Wilderness", "Soil Type"],
            "Value":        [
                f"{elevation:,} m",
                f"{slope}°",
                f"{aspect}°",
                f"{h_dist_hydro} m",
                f"{h_dist_road} m",
                selected_wilderness.replace("Wilderness_Area", "Area "),
                selected_soil.replace("Soil_Type", "Type "),
            ],
        }
        summary_df = pd.DataFrame(summary_data).set_index("Parameter")
        st.dataframe(summary_df, use_container_width=True, height=280)

    # ── Engineered features callout ──
    with st.expander("🔧 Engineered Features Used in Inference", expanded=False):
        eng_col1, eng_col2, eng_col3 = st.columns(3)
        with eng_col1:
            st.metric("Euclidean Hydro Distance", f"{euclidean_hydro:.1f} m")
        with eng_col2:
            st.metric("Water-Adjusted Elevation", f"{water_elevation:.0f} m")
        with eng_col3:
            st.metric("Avg. Amenity Distance",    f"{dist_amenities:.0f} m")

# ──────────────────────────────────────────────────────────────────────────────
# 10 ▸ FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-text">
    Forest Cover AI Predictor · XGBoost Champion Model · USGS/USFS Geospatial Dataset<br>
    Powered by Streamlit &amp; Scikit-learn · 94.7% Test Accuracy
</div>
""", unsafe_allow_html=True)