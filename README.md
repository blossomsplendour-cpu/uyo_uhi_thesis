# A Reproducible Python-Based Multi-Temporal Analysis of Urban Heat Island Dynamics in Uyo LGA Using Landsat 8/9 and GIS

## Project Overview
This project presents an end-to-end, reproducible geospatial and machine learning pipeline that quantifies, models, and maps Urban Heat Island (UHI) dynamics and surface thermal microclimates across Uyo Local Government Area (LGA), Akwa Ibom State, Nigeria, over a 10-year period (2015–2025).

The analytical framework integrates:
* Multi-Temporal Landsat 8/9 Collection 2 Level-2 Imagery (6 dry-season scenes).
* Radiometric Calibration & Bitwise QA Cloud Masking.
* Spatial Statistics & Hotspot Clustering (Getis-Ord Gi* & Persistence).
* Machine Learning Core (Option B): Random Forest and XGBoost regressors evaluated via Spatial Blocked Cross-Validation (2 km x 2 km blocks) with SHAP feature interpretation.
* Interactive Streamlit Web Dashboard for real-time thermal microclimate exploration.

---

## Technical Specifications & Formulas

### 1. Landsat Collection 2 Level-2 Scale Factors
* Surface Reflectance (SR_B4, SR_B5, SR_B6): Reflectance = DN * 0.0000275 - 0.2
* Surface Temperature (ST_B10): LST (°C) = (DN * 0.00341802 + 149.0) - 273.15
  (Note: Collection 2 Level-2 is already a surface temperature product. No Planck law or emissivity corrections applied).

### 2. Spectral Indices
* Normalized Difference Vegetation Index (NDVI): NDVI = (SR_B5 - SR_B4) / (SR_B5 + SR_B4)
* Normalized Difference Built-Up Index (NDBI): NDBI = (SR_B6 - SR_B5) / (SR_B6 + SR_B5)

### 3. Surface UHI Gap Definition
  UHI Gap = Mean LST (Built-Up: NDBI > 0.0) - Mean LST (Vegetated: NDVI > 0.3)

---

## Quick Start Guide

### 1. Conda Environment Setup
```bash
conda env create -f environment.yml
conda activate uhi
```

### 2. Launch the Interactive Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```

---

## Summary of Key Thesis Results (2015–2025)
1. Strong Thermal-Vegetation Coupling: Pearson correlation between LST and NDVI is r = -0.875 (p < 0.001); every 0.1 increase in NDVI drops surface temperature by ~2.06°C.
2. Built-Up Surface Heating: Pearson correlation between LST and NDBI is r = +0.861 (p < 0.001).
3. Machine Learning Performance: XGBoost under Spatial Blocked Cross-Validation achieved R² = 0.884 and RMSE = 0.92°C.
4. SHAP Feature Impact: NDVI emerged as the primary negative driver of LST (up to -2.5°C cooling), while NDBI and spatial proximity to the Ibom Plaza urban core exerted positive thermal pressure (up to +2.0°C warming).

---

## Scientific Caveats & Limitations
1. LST != Air Temperature: Landsat measures radiational land surface skin temperature, not 2 m ambient air temperature measured by weather stations.
2. Snapshot Overpass Time: Landsat overpasses mid-morning (~10:00 AM local time), capturing pre-peak thermal conditions.
3. Sensor Continuity: Inter-sensor shifts between Landsat 8 and Landsat 9 are accounted for via relative index differencing.