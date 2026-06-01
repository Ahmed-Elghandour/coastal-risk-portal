import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet

st.set_page_config(page_title="Coastal Risk Portal", layout="wide")

st.title("🌊 Coastal Risk Geoportal")
st.markdown("Interactive platform for sea level rise monitoring, AI-based forecasting, and coastal flood hazard assessment")

# ─── PSMSL Stations ───────────────────────────────────────────────────────────
STATIONS = {
    "IJMUIDEN (Netherlands)": {"id": 32,  "lat": 52.46,  "lon":   4.55},
    "Port Said (Egypt)":      {"id": 253, "lat": 31.26,  "lon":  32.28},
    "Dakar (Senegal)":        {"id": 476, "lat": 14.69,  "lon": -17.44},
    "Sydney (Australia)":     {"id": 65,  "lat": -33.87, "lon": 151.21},
}

# ─── Load TWL flood data ───────────────────────────────────────────────────────
@st.cache_data
def load_twl_data():
    df = pd.read_csv(r"C:\Coastal_portal\data\EU_TWL_storms.csv")
    high   = df[df['RP100'] > 3]
    medium = df[(df['RP100'] >= 1.5) & (df['RP100'] <= 3)].iloc[::5]
    low    = df[df['RP100'] < 1.5].iloc[::10]
    return pd.concat([high, medium, low]).reset_index(drop=True)

# ─── Fetch PSMSL ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_psmsl(station_id, station_name):
    url = f"https://psmsl.org/data/obtaining/rlr.annual.data/{station_id}.rlrdata"
    try:
        response = requests.get(url, timeout=10)
        records = []
        for line in response.text.strip().split("\n"):
            parts = line.strip().split(";")
            if len(parts) >= 2:
                year  = int(float(parts[0].strip()))
                level = float(parts[1].strip())
                if level != -99999:
                    records.append({"year": year, "level_mm": level, "station": station_name})
        return pd.DataFrame(records)
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_all_stations():
    all_data = {}
    for name, info in STATIONS.items():
        df = fetch_psmsl(info["id"], name)
        if not df.empty:
            baseline = df[df["year"] <= 1990]["level_mm"].mean()
            df["anomaly_mm"] = df["level_mm"] - baseline
            x     = df["year"] - df["year"].mean()
            trend = (x * df["anomaly_mm"]).sum() / (x ** 2).sum()
            all_data[name] = {
                "df":           df,
                "trend_mm_yr":  round(trend, 2),
                "latest_year":  int(df["year"].max()),
                "n_records":    len(df),
            }
    return all_data

# ─── Prophet forecast ─────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def run_forecast(df_anomaly, forecast_year, scenario):
    df_prophet = pd.DataFrame({
        "ds": pd.to_datetime(df_anomaly["year"], format="%Y"),
        "y":  df_anomaly["anomaly_mm"]
    })
    m = Prophet(yearly_seasonality=False, weekly_seasonality=False,
                daily_seasonality=False, changepoint_prior_scale=0.1)
    m.fit(df_prophet)
    latest  = df_anomaly["year"].max()
    periods = forecast_year - latest
    future  = m.make_future_dataframe(periods=periods, freq="YE")
    forecast = m.predict(future)
    if "8.5" in scenario:
        extra = ((forecast["ds"].dt.year - latest) * 0.5).clip(lower=0)
        forecast["yhat"]       += extra
        forecast["yhat_upper"] += extra
        forecast["yhat_lower"] += extra
    return forecast

# ─── Load data ────────────────────────────────────────────────────────────────
with st.spinner("Loading coastal hazard data..."):
    twl_df   = load_twl_data()
    all_data = load_all_stations()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("Controls")

map_layer = st.sidebar.radio(
    "Map layer",
    ["🌊 TWL Flood Hazard (Europe)", "📍 Tide Gauge Stations"]
)

scenario = st.sidebar.radio("Climate scenario", ["RCP 4.5 (moderate)", "RCP 8.5 (severe)"])
year     = st.sidebar.slider("Projection year", 2025, 2100, 2050, step=5)

if map_layer == "📍 Tide Gauge Stations":
    selected_station = st.sidebar.selectbox("Select station", list(STATIONS.keys()))

st.sidebar.markdown("---")
st.sidebar.markdown("**Data sources**")
st.sidebar.markdown("🔴 Live — [PSMSL](https://www.psmsl.org)")
st.sidebar.markdown("📄 [Cotrim et al., 2026](https://doi.org/10.5194/nhess-26-1859-2026)")

# ─── Map ──────────────────────────────────────────────────────────────────────
st.subheader("🗺️ Coastal Risk Map")
m = folium.Map(location=[48, 10], zoom_start=4, tiles="CartoDB positron")

if map_layer == "🌊 TWL Flood Hazard (Europe)":

    def twl_color(rp100):
        if rp100 > 4:   return "#d73027"
        elif rp100 > 3: return "#fc8d59"
        elif rp100 > 2: return "#fee08b"
        elif rp100 > 1.5: return "#d9ef8b"
        else:           return "#91cf60"

    # Dominant storm type per point
    def dominant_st(row):
        sts = {"ST_A": row["ST_A_perc"], "ST_B": row["ST_B_perc"],
               "ST_C": row["ST_C_perc"], "ST_D": row["ST_D_perc"]}
        return max(sts, key=sts.get)

    for _, row in twl_df.iterrows():
        color = twl_color(row["RP100"])
        dom   = dominant_st(row)
        popup = f"""
        <b>Coastal Target Point</b><br>
        Lon: {row['Lon']:.3f} | Lat: {row['Lat']:.3f}<br>
        <b>RP100 TWL: {row['RP100']:.2f} m</b><br>
        RP50 TWL: {row['RP50']:.2f} m<br>
        Storm duration (RP100): {row['Dur_RP100']:.1f} hrs<br>
        Dominant storm type: {dom}
        """
        folium.CircleMarker(
            location=[row["Lat"], row["Lon"]],
            radius=3,
            color=color, fill=True, fill_color=color, fill_opacity=0.8,
            popup=folium.Popup(popup, max_width=240),
            weight=0
        ).add_to(m)

    # Legend
    legend_html = """
    <div style='position:fixed;bottom:30px;left:30px;z-index:1000;
         background:white;padding:12px;border-radius:8px;
         border:1px solid #ccc;font-size:12px;'>
    <b>RP100 Total Water Level</b><br>
    <span style='color:#d73027'>●</span> &gt; 4m — Extreme<br>
    <span style='color:#fc8d59'>●</span> 3–4m — Very High<br>
    <span style='color:#fee08b'>●</span> 2–3m — High<br>
    <span style='color:#d9ef8b'>●</span> 1.5–2m — Medium<br>
    <span style='color:#91cf60'>●</span> &lt; 1.5m — Low
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    col1, col2 = st.columns([3, 1])
    with col1:
        st_folium(m, width=None, height=540)
    with col2:
        st.markdown("**Summary**")
        st.metric("Total coastal points", f"{len(twl_df):,}")
        st.metric("Max RP100 TWL", f"{twl_df['RP100'].max():.2f} m")
        st.metric("Mean RP100 TWL", f"{twl_df['RP100'].mean():.2f} m")
        st.markdown("---")
        st.markdown("**Risk distribution**")
        risk_counts = pd.DataFrame({
            "Risk":  ["Extreme", "Very High", "High", "Medium", "Low"],
            "Points": [
                len(twl_df[twl_df['RP100'] > 4]),
                len(twl_df[(twl_df['RP100'] > 3) & (twl_df['RP100'] <= 4)]),
                len(twl_df[(twl_df['RP100'] > 2) & (twl_df['RP100'] <= 3)]),
                len(twl_df[(twl_df['RP100'] > 1.5) & (twl_df['RP100'] <= 2)]),
                len(twl_df[twl_df['RP100'] <= 1.5]),
            ]
        })
        st.dataframe(risk_counts, hide_index=True)

else:
    # Tide gauge stations map
    for name, info in STATIONS.items():
        if name not in all_data:
            continue
        trend = all_data[name]["trend_mm_yr"]
        color = "red" if trend > 3 else "orange" if trend > 2 else "green"
        scale = 1.3 if "8.5" in scenario else 1.0
        years_ahead = year - all_data[name]["latest_year"]
        projected   = round((trend * years_ahead * scale) / 10, 1)
        popup = f"""
        <b>{name}</b><br>
        Trend: <b>{trend} mm/yr</b><br>
        Records: {all_data[name]['n_records']} years<br>
        Projected SLR by {year}: <b>{projected} cm</b><br>
        Scenario: {scenario}
        """
        folium.CircleMarker(
            location=[info["lat"], info["lon"]],
            radius=10 + abs(trend) * 2,
            color=color, fill=True, fill_color=color, fill_opacity=0.7,
            popup=folium.Popup(popup, max_width=260)
        ).add_to(m)

    col1, col2 = st.columns([3, 1])
    with col1:
        st_folium(m, width=None, height=540)
    with col2:
        st.markdown("**Stations summary**")
        summary = pd.DataFrame([
            {"Station": name.split("(")[0].strip(),
             "Trend (mm/yr)": all_data[name]["trend_mm_yr"],
             "Records": all_data[name]["n_records"]}
            for name in all_data
        ])
        st.dataframe(summary, hide_index=True)

# ─── Station detail (only when tide gauge layer selected) ─────────────────────
if map_layer == "📍 Tide Gauge Stations" and selected_station in all_data:
    st.markdown("---")
    st.subheader(f"📈 Station Detail — {selected_station}")

    d  = all_data[selected_station]
    df = d["df"]

    scale       = 1.3 if "8.5" in scenario else 1.0
    years_ahead = year - d["latest_year"]
    projected   = round((d["trend_mm_yr"] * years_ahead * scale) / 10, 1)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Trend",          f"{d['trend_mm_yr']} mm/yr")
    m2.metric("Records",        f"{d['n_records']} years")
    m3.metric("Latest year",    d["latest_year"])
    m4.metric(f"Projected SLR by {year}", f"{projected} cm")

    st.markdown("#### 🤖 AI Forecast (Prophet)")
    with st.spinner("Running forecast model..."):
        forecast = run_forecast(df[["year", "anomaly_mm"]], year, scenario)

    latest_year  = d["latest_year"]
    forecast_only = forecast[forecast["ds"].dt.year > latest_year]
    target_row    = forecast[forecast["ds"].dt.year == year]

    if not target_row.empty:
        fa, fb, fc = st.columns(3)
        fa.metric("AI forecast (central)", f"{round(target_row['yhat'].values[0]/10,1)} cm")
        fb.metric("Lower bound",           f"{round(target_row['yhat_lower'].values[0]/10,1)} cm")
        fc.metric("Upper bound",           f"{round(target_row['yhat_upper'].values[0]/10,1)} cm")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["year"], y=df["anomaly_mm"],
        name="Observed", mode="lines",
        line=dict(color="#1f77b4", width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=forecast_only["ds"].dt.year, y=forecast_only["yhat"],
        name="AI Forecast", mode="lines",
        line=dict(color="red", width=2, dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_only["ds"].dt.year, forecast_only["ds"].dt.year[::-1]]),
        y=pd.concat([forecast_only["yhat_upper"], forecast_only["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(255,0,0,0.1)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Uncertainty range"
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
    fig.add_vline(x=latest_year, line_dash="dot", line_color="gray", opacity=0.6,
                  annotation_text="Forecast start", annotation_position="top right")
    fig.update_layout(
        title=f"Sea Level Anomaly & AI Forecast — {selected_station} ({scenario})",
        xaxis_title="Year", yaxis_title="Anomaly (mm)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Sea level data: PSMSL. Flood hazard data: Cotrim et al. (2026), NHESS, doi:10.5194/nhess-26-1859-2026. Forecasts: Meta Prophet model.")