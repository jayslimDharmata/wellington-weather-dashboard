import re
from datetime import datetime, timezone
import matplotlib.pyplot as plt
from metpy.calc import bulk_shear, cape_cin, parcel_profile
from metpy.plots import SkewT
from metpy.units import units
import numpy as np
import pandas as pd
import requests
from siphon.simplewebservice.wyoming import WyomingUpperAir
import streamlit as st

# Configure the Streamlit page layout
st.set_page_config(
    page_title="SE Florida Severe Weather Dashboard",
    page_icon="⛈️",
    layout="wide",
)

st.title("⛈️ Southeast Florida Severe Convective Monitor")
st.caption(
    "Tailored thermodynamic & synoptic analysis for Wellington / Palm Beach"
    " County"
)


# --- FUNCTION: Fetch & Parse NWS Miami AFD ---
@st.cache_data(ttl=1800)
def fetch_nws_afd():
  url = "https://api.weather.gov/products/types/AFD/locations/MFL"
  headers = {
      "User-Agent": (
          "(WellingtonWeatherDashboard, contact@localweatherdashboard.org)"
      )
  }
  try:
    resp = requests.get(url, headers=headers, timeout=10)
    data = resp.json()
    product_url = data["@graph"][0]["@id"]

    prod_resp = requests.get(product_url, headers=headers, timeout=10)
    afd_text = prod_resp.json()["productText"]

    # Extract Short Term section using regex
    short_term_match = re.search(
        r"\.SHORT TERM\.\.\.(.*?)(?=\n\.[A-Z\s]+\.\.\.|$)", afd_text, re.DOTALL
    )
    short_term = (
        short_term_match.group(1).strip()
        if short_term_match
        else "Short term discussion currently unavailable."
    )

    return short_term, afd_text
  except Exception as e:
    return f"Unable to fetch AFD: {e}", ""


# --- FUNCTION: Fetch Miami (KMFL) Sounding Data ---
@st.cache_data(ttl=3600)
def fetch_sounding_data(year, month, day, hour):
  station = "MFL"  # Miami NWS Station
  dt = datetime(year, month, day, hour)
  try:
    df = WyomingUpperAir.request_data(dt, station)
    return df, None
  except Exception as e:
    return None, str(e)


# --- SECTION 1: NWS Miami Discussion (Human Insight) ---
st.subheader("📋 NWS Miami: Short-Term Operational Thinking")
short_term_afd, full_afd = fetch_nws_afd()

st.info(short_term_afd)

with st.expander("📄 View Full Unedited NWS Miami AFD"):
  st.text(full_afd)

st.divider()

# --- SECTION 2: Upper-Air Thermodynamics (KMFL) ---
st.subheader("🎈 KMFL (Miami) Sounding & Severe Parameters")

# Sidebar Controls
st.sidebar.header("Controls & Diagnostics")
cycle_choice = st.sidebar.radio("Select Sounding Cycle", ["Latest 12z", "00z"])
now_utc = datetime.now(timezone.utc)

hour = 12 if "12z" in cycle_choice else 0
sounding_df, err = fetch_sounding_data(
    now_utc.year, now_utc.month, now_utc.day, hour
)

if err:
  st.warning(
      f"Note: Current {hour:02d}z sounding not yet processed or delayed"
      f" ({err}). Showing fallback/historical parameters if available."
  )
elif sounding_df is not None:
  # Extract clean arrays with MetPy units
  p = sounding_df["pressure"].values * units.hPa
  t = sounding_df["temperature"].values * units.degC
  td = sounding_df["dewpoint"].values * units.degC
  u = sounding_df["u_wind"].values * units.knots
  v = sounding_df["v_wind"].values * units.knots
  height = sounding_df["height"].values * units.meter

  # Calculate Surface-Based Parcel Profile & CAPE/CIN
  prof = parcel_profile(p, t[0], td[0])
  sbcape, sbcin = cape_cin(p, t, td, prof)

  # Calculate Kinematics (0-6km Shear)
  u_shear_6k, v_shear_6k = bulk_shear(
      p, u, v, height=height, depth=6000 * units.meter
  )
  shear_6k_mag = np.hypot(u_shear_6k, v_shear_6k)

  # Display High-Contrast KPI Cards
  col1, col2, col3, col4 = st.columns(4)
  col1.metric("SBCAPE (Thermodynamics)", f"{sbcape.magnitude:.0f} J/kg")
  col2.metric("SBCIN (Cap Strength)", f"{sbcin.magnitude:.0f} J/kg")
  col3.metric("0-6 km Bulk Shear", f"{shear_6k_mag.magnitude:.1f} kts")
  col4.metric("Surface Dew Point", f"{td[0].to('degF').magnitude:.1f} °F")

  # Plain-English Diagnostics for Wellington / South Florida
  st.markdown("#### 🔍 Automated Convective Assessment")
  regime = []
  if shear_6k_mag.magnitude >= 40:
    regime.append(
        "**Kinematics:** High deep-layer shear (>40 kts) supports organized"
        " convective bands or fast-moving QLCS segments."
    )
  else:
    regime.append(
        "**Kinematics:** Marginal deep-layer shear (<40 kts). Fast-moving"
        " organized supercells are less likely."
    )

  if sbcape.magnitude < 500:
    regime.append(
        "**Thermodynamics:** Meager surface buoyancy (<500 J/kg). Classic"
        " winter high-shear/low-CAPE setup where severe risk relies heavily on"
        " dynamic forcing."
    )
  elif sbcape.magnitude < 1500:
    regime.append(
        "**Thermodynamics:** Moderate buoyancy (500–1500 J/kg). Ample fuel"
        " during cool-season setups for isolated severe wind gusts."
    )
  else:
    regime.append(
        "**Thermodynamics:** Robust buoyancy (>1500 J/kg). Highly energized"
        " boundary layer."
    )

  for note in regime:
    st.markdown(f"- {note}")

  # Plot the Skew-T
  fig = plt.figure(figsize=(10, 8))
  skew = SkewT(fig, rotation=45)
  skew.plot(p, t, "r", linewidth=2, label="Temperature")
  skew.plot(p, td, "g", linewidth=2, label="Dew Point")
  skew.plot(p, prof, "k--", linewidth=1.5, label="Parcel Profile")
  skew.plot_barbs(p[::3], u[::3], v[::3])
  skew.ax.set_ylim(1000, 100)
  skew.ax.set_xlim(-30, 40)
  skew.ax.set_xlabel("Temperature (°C)")
  skew.ax.set_ylabel("Pressure (hPa)")
  skew.ax.legend(loc="upper left")

  st.pyplot(fig)
