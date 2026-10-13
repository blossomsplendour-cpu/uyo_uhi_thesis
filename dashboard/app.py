
import streamlit as st
import pandas as pd
import geopandas as gpd
import rioxarray as rxr
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
from PIL import Image

st.set_page_config(
    page_title="Uyo UHI Surface Thermal Explorer",
    page_icon="🌡️",
    layout="wide"
)

PROJECT_DIR = Path(__file__).parent.parent.resolve()
out_dir = PROJECT_DIR / 'data' / 'outputs'
fig_dir = PROJECT_DIR / 'figures'

@st.cache_data
def load_summary():
    return pd.read_csv(out_dir / 'summary_by_date.csv')

@st.cache_data
def load_persistence_zones():
    return gpd.read_file(out_dir / 'hotspot_persistence_zones.geojson')

@st.cache_data
def load_zonal_for_scene(scene_id):
    return pd.read_csv(out_dir / f'zonal_stats_{scene_id}.csv')

@st.cache_data
def load_ml_metrics():
    ml_file = out_dir / 'ml_model_evaluation.csv'
    if ml_file.exists():
        return pd.read_csv(ml_file)
    return None

summary_df = load_summary()
zones_persistence = load_persistence_zones()
ml_metrics_df = load_ml_metrics()

st.title("🌡️ Uyo Urban Heat Island & Thermal Microclimate Explorer")
st.caption("A Reproducible Multi-Temporal Analysis (2015–2025) Using Landsat 8/9 Collection 2 Level-2 Imagery & Machine Learning | Uyo LGA, Nigeria")

st.sidebar.header("🕹️ Explorer Modes")
mode = st.sidebar.radio(
    "Select Mode", 
    ["Overview (Study Area & LULC)", "Single-Date Microclimate", "Dual-Date Comparison", "ML & Hotspot Persistence"]
)

date_options = summary_df['date'].tolist()
scene_map = dict(zip(summary_df['date'], summary_df['scene_id']))

# ---------------------------------------------------------
# MODE 1: OVERVIEW (STUDY AREA & LULC)
# ---------------------------------------------------------
if mode == "Overview (Study Area & LULC)":
    st.subheader("📍 Study Area & Urban Sprawl Overview")
    
    col_img1, col_img2 = st.columns(2)
    study_map_path = fig_dir / '01_study_area_map_publication_refined.png'
    lulc_map_path = fig_dir / '07_lulc_5class_calibrated.png'
    
    with col_img1:
        st.markdown("#### Publication Study Area Map")
        if study_map_path.exists():
            st.image(Image.open(study_map_path), use_container_width=True)
            st.info("📌 **Map Guide:** Main map displays Uyo LGA boundary in yellow/red with key settlements and major road network. Insets A and B provide Akwa Ibom State and Nigeria location context.")
            
    with col_img2:
        st.markdown("#### 10-Year 5-Class LULC Urban Sprawl (2015 vs. 2023)")
        if lulc_map_path.exists():
            st.image(Image.open(lulc_map_path), use_container_width=True)
            st.info("📌 **Map Guide:** **Red** = Built-Up/Impervious, **Dark Green** = Forest/Trees, **Light Green** = Crops, **Yellow** = Bare Soil, **Blue** = Wetlands/Streams.")

# ---------------------------------------------------------
# MODE 2: SINGLE-DATE ANALYSIS
# ---------------------------------------------------------
elif mode == "Single-Date Microclimate":
    selected_date = st.sidebar.selectbox("Select Observation Date", date_options, index=len(date_options)-1)
    selected_scene = scene_map[selected_date]
    zonal_df = load_zonal_for_scene(selected_scene)
    
    selected_layer = st.sidebar.selectbox(
        "Select Spatial Layer",
        ["Land Surface Temperature (°C)", "Vegetation Index (NDVI)", "Built-Up Index (NDBI)", "Priority Hotspot Zones"]
    )
    p_threshold = st.sidebar.slider("Thermal Priority Threshold (Percentile)", 75, 98, 90, 1)

    curr_row = summary_df[summary_df['date'] == selected_date].iloc[0]
    st.markdown(f"### 📊 Metrics for {selected_date}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mean LST", f"{curr_row['mean_lst']:.2f} °C")
    m2.metric("Max LST", f"{curr_row['max_lst']:.2f} °C")
    m3.metric("Surface UHI Gap", f"+{curr_row['uhi_gap']:.2f} °C")
    m4.metric("Mean NDVI", f"{curr_row['mean_ndvi']:.3f}")
    
    st.divider()
    left_col, right_col = st.columns([1.2, 1])
    
    merged_gdf = zones_persistence.merge(zonal_df, on='zone_id')
    wgs_gdf = merged_gdf.to_crs(epsg=4326)
    centroid = wgs_gdf.union_all().centroid
    
    with left_col:
        st.subheader(f"🗺️ Spatial Layer: {selected_layer}")
        m = folium.Map(location=[centroid.y, centroid.x], zoom_start=12, tiles="OpenStreetMap")
        
        if selected_layer == "Land Surface Temperature (°C)":
            folium.Choropleth(
                geo_data=wgs_gdf, data=wgs_gdf, columns=["zone_id", "lst_mean"],
                key_on="feature.properties.zone_id", fill_color="YlOrRd", fill_opacity=0.75,
                line_opacity=0.3, nan_fill_color="#f7f7f7", nan_fill_opacity=0.2, legend_name="LST (°C)"
            ).add_to(m)
            st_folium(m, width="100%", height=420)
            st.info("💡 **Map Guide (LST):** **Dark Red/Orange** = Hot Surface Microclimates (>38°C), **Yellow** = Cooler Vegetated Surfaces (<34°C). Scale bar in top-left shows temperature range.")
            
        elif selected_layer == "Vegetation Index (NDVI)":
            folium.Choropleth(
                geo_data=wgs_gdf, data=wgs_gdf, columns=["zone_id", "ndvi_mean"],
                key_on="feature.properties.zone_id", fill_color="YlGn", fill_opacity=0.75,
                line_opacity=0.3, nan_fill_color="#f7f7f7", nan_fill_opacity=0.2, legend_name="NDVI"
            ).add_to(m)
            st_folium(m, width="100%", height=420)
            st.info("💡 **Map Guide (NDVI):** **Dark Green** = Dense Vegetation/Forest (>0.4), **Yellow** = Built-up / Bare Soil (<0.2).")
            
        elif selected_layer == "Built-Up Index (NDBI)":
            folium.Choropleth(
                geo_data=wgs_gdf, data=wgs_gdf, columns=["zone_id", "ndbi_mean"],
                key_on="feature.properties.zone_id", fill_color="Oranges", fill_opacity=0.75,
                line_opacity=0.3, nan_fill_color="#f7f7f7", nan_fill_opacity=0.2, legend_name="NDBI"
            ).add_to(m)
            st_folium(m, width="100%", height=420)
            st.info("💡 **Map Guide (NDBI):** **Dark Orange/Red** = High Built-Up Density / Concrete (>0.0), **Light Yellow** = Vegetated/Non-built (<-0.1).")
            
        else:
            cutoff = np.nanpercentile(zonal_df['lst_mean'], p_threshold)
            p_zones = wgs_gdf[wgs_gdf['lst_mean'] >= cutoff]
            folium.GeoJson(wgs_gdf, style_function=lambda x: {'fillColor': '#e0e0e0', 'color': '#999999', 'weight': 0.5, 'fillOpacity': 0.2}).add_to(m)
            folium.GeoJson(
                p_zones, style_function=lambda x: {'fillColor': '#d95f02', 'color': '#900000', 'weight': 1.5, 'fillOpacity': 0.75},
                tooltip=folium.GeoJsonTooltip(fields=['zone_id', 'lst_mean', 'ndvi_mean'], aliases=['Zone:', 'LST (°C):', 'NDVI:'])
            ).add_to(m)
            st_folium(m, width="100%", height=420)
            st.info(f"💡 **Map Guide (Priority Zones):** **Orange/Red Cells** = Priority targets ranking in the upper **{p_threshold}th percentile** for surface temperature requiring immediate greening interventions.")

    with right_col:
        st.subheader("📈 Microclimate OLS Regression")
        scatter_var = st.radio("Scatter Variable", ["NDVI (Vegetation)", "NDBI (Built-Up)"], horizontal=True)
        if scatter_var == "NDVI (Vegetation)":
            fig = px.scatter(zonal_df, x="ndvi_mean", y="lst_mean", trendline="ols", labels={"ndvi_mean": "Mean NDVI", "lst_mean": "Mean LST (°C)"}, color_discrete_sequence=["#2ca02c"])
        else:
            fig = px.scatter(zonal_df, x="ndbi_mean", y="lst_mean", trendline="ols", labels={"ndbi_mean": "Mean NDBI", "lst_mean": "Mean LST (°C)"}, color_discrete_sequence=["#d95f02"])
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# MODE 3: DUAL-DATE COMPARISON
# ---------------------------------------------------------
elif mode == "Dual-Date Comparison":
    col_d1, col_d2 = st.sidebar.columns(2)
    date1 = col_d1.selectbox("Date 1 (Baseline)", date_options, index=0)
    date2 = col_d2.selectbox("Date 2 (Comparison)", date_options, index=len(date_options)-1)
    scene1, scene2 = scene_map[date1], scene_map[date2]
    zonal1, zonal2 = load_zonal_for_scene(scene1), load_zonal_for_scene(scene2)
    
    st.subheader(f"🔄 Side-by-Side Surface Temperature Comparison: {date1} vs. {date2}")
    col_m1, col_m2 = st.columns(2)
    wgs_gdf = zones_persistence.to_crs(epsg=4326)
    centroid = wgs_gdf.union_all().centroid
    
    with col_m1:
        st.markdown(f"#### Baseline Date: {date1}")
        m1 = folium.Map(location=[centroid.y, centroid.x], zoom_start=11, tiles="OpenStreetMap")
        m1_gdf = wgs_gdf.merge(zonal1, on='zone_id')
        folium.Choropleth(geo_data=m1_gdf, data=m1_gdf, columns=["zone_id", "lst_mean"], key_on="feature.properties.zone_id", fill_color="YlOrRd", fill_opacity=0.75, nan_fill_color="#f7f7f7", nan_fill_opacity=0.2, legend_name=f"LST {date1} (°C)").add_to(m1)
        st_folium(m1, width="100%", height=400, key="map1")
        
    with col_m2:
        st.markdown(f"#### Comparison Date: {date2}")
        m2 = folium.Map(location=[centroid.y, centroid.x], zoom_start=11, tiles="OpenStreetMap")
        m2_gdf = wgs_gdf.merge(zonal2, on='zone_id')
        folium.Choropleth(geo_data=m2_gdf, data=m2_gdf, columns=["zone_id", "lst_mean"], key_on="feature.properties.zone_id", fill_color="YlOrRd", fill_opacity=0.75, nan_fill_color="#f7f7f7", nan_fill_opacity=0.2, legend_name=f"LST {date2} (°C)").add_to(m2)
        st_folium(m2, width="100%", height=400, key="map2")
        
    st.info("💡 **Comparison Guide:** Compare spatial shifts in high-temperature red zones between the baseline year and comparison year. Scale bars in top-left corners reflect exact observation ranges.")

# ---------------------------------------------------------
# MODE 4: ML & HOTSPOT PERSISTENCE
# ---------------------------------------------------------
else:
    c_ml1, c_ml2 = st.columns([1.2, 1])
    with c_ml1:
        st.subheader("🔥 10-Year Hotspot Persistence Map (Getis-Ord Gi*)")
        wgs_gdf = zones_persistence.to_crs(epsg=4326)
        centroid = wgs_gdf.union_all().centroid
        
        m_p = folium.Map(location=[centroid.y, centroid.x], zoom_start=12, tiles="OpenStreetMap")
        folium.Choropleth(
            geo_data=wgs_gdf, data=wgs_gdf, columns=["zone_id", "hotspot_persistence"],
            key_on="feature.properties.zone_id", fill_color="YlOrRd", fill_opacity=0.8,
            nan_fill_color="#f7f7f7", nan_fill_opacity=0.2, legend_name="Hotspot Persistence Count"
        ).add_to(m_p)
        st_folium(m_p, width="100%", height=420)
        st.info("💡 **Persistence Map Guide:** **Dark Maroon (5–6 Dates)** = Highly Critical Persistent Thermal Hotspots (Ibom Plaza / Central Business District), **Light Yellow (0–1 Dates)** = Non-hotspot rural areas.")
        
    with c_ml2:
        st.subheader("🤖 Machine Learning Model Evaluation (Option B)")
        if ml_metrics_df is not None:
            st.dataframe(ml_metrics_df, use_container_width=True)
            st.info("💡 **Methodological Rigor:** Evaluated using **Spatial Blocked Cross-Validation (2 km x 2 km blocks)** to eliminate spatial autocorrelation data leakage.")
        
        st.subheader("📌 Key Predictive Drivers (SHAP Analysis)")
        st.markdown('''
        * **NDVI (Vegetation):** Primary cooling driver. High NDVI reduces surface heat by up to **-2.5°C**.
        * **NDBI (Built-Up):** Primary warming driver. High built-up intensity increases surface heat by up to **+2.0°C**.
        * **Spatial Position:** Captures urban density vs. rural hinterland gradient across Uyo LGA.
        ''')

st.divider()

st.subheader("📈 10-Year Multi-Temporal Microclimate Trajectory (2015–2025)")
fig_ts = go.Figure()
fig_ts.add_trace(go.Scatter(x=summary_df['date'], y=summary_df['mean_lst'], mode='lines+markers', name='Mean LST (°C)', line=dict(color='#d95f02', width=3)))
fig_ts.add_trace(go.Scatter(x=summary_df['date'], y=summary_df['mean_ndvi']*50, mode='lines+markers', name='Mean NDVI (Scaled x50)', line=dict(color='#2ca02c', width=2, dash='dash')))
fig_ts.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig_ts, use_container_width=True)

st.subheader("📋 Master Summary Data Table")
st.dataframe(summary_df, use_container_width=True)
st.download_button(
    label="📥 Download Master Summary CSV",
    data=summary_df.to_csv(index=False),
    file_name="uyo_uhi_master_summary_2015_2025.csv",
    mime="text/csv"
)

with st.expander("📚 Scientific Methodology & Thesis Limitations (Click to Expand)"):
    st.markdown('''
    1. **LST vs. Air Temperature:** Landsat measures Radiative Land Surface Temperature (LST), which is distinct from 2 m air temperature measured by meteorological weather stations.
    2. **Satellite Overpass Timing:** Landsat overpasses mid-morning (~10:00 AM local time), capturing pre-peak thermal conditions.
    3. **Spatial Autocorrelation:** Addressed via Getis-Ord $Gi^*$ spatial clustering and Spatial Blocked Cross-Validation.
    4. **Level-2 Calibration:** Processed using USGS Collection 2 Level-2 scale factors ($DN \times 0.00341802 + 149.0 - 273.15$). No Planck law conversion applied.
    ''')
