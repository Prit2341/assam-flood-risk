# Assam Flood Risk — Phase 3

Phase 3 of a flood-risk digital-twin thesis. Phase 1 (methodology) was Sri Lanka; Phase 2 (transferability) was Arunachal Pradesh. This project applies the same pipeline to Assam, India.

**Status (2026-10-20)**: Rainfall model v1 frozen (`releases/rainfall_v1/`). Dynamic flood susceptibility built and validated for all 4 districts; the working model is **H6** (see below) and it drives the dashboard. Hydraulic/inundation modelling (HEC-RAS), bathymetry and the one-basin scope question are still open.

---

## Study Area

Four districts chosen by Very High flood-hazard percentage (NRSC/ISRO atlas, 1998–2023):

| District | % Very High hazard | Primary river reach | Validation event |
|---|---|---|---|
| Morigaon | 22.8% (1st of 35) | Kopili River | Jun 2022 embankment breach |
| Sivasagar | 13.2% (2nd) | Dikhow River | Jul 2026 (contested causation) |
| Nagaon | 9.9% (3rd) | Kopili River | Jun 2022 (same event as Morigaon) |
| Karimganj | 9.8% (4th) | Kushiyara/Barak | Jun 2022 + Jun 2024 |

These span **3 separate river systems** (Sivasagar alone; Morigaon+Nagaon sharing HydroBASINS level-4 basin 4040960200; Karimganj alone on the Barak/Kushiyara). This conflicts with the project's original one-basin scope rule — recorded as a deliberate decision, not silently overridden.

---

## Data Sources

All free. None require institutional data-sharing agreements.

| Layer | Source | Script |
|---|---|---|
| DEM (30m, bias-corrected) | FABDEM via GEE community catalog (`sat-io/open-datasets/FABDEM`) | `scripts/pull_fabdem.py` |
| ERA5 hourly (2000–2026) | `ECMWF/ERA5/HOURLY` via GEE | `scripts/pull_era5_interactive.py` |
| ERA5 pressure-level 5×5 grid | Public AWS Icechunk store (`s3://earthmover-icechunk-era5/`) | `scripts/fetch_icechunk_pressure_assam.py` |
| Land cover | Dynamic World V1 via GEE | `scripts/pull_dynamicworld_lulc.py` |
| Soil (6 depths, 6 properties) | SoilGrids ISRIC via GEE | `scripts/pull_soilgrids_full_profile.py` |
| Flood extent (pre-made) | Copernicus GFM STAC (`stac.eodc.eu/api/v1`) | `scripts/pull_gfm_validation.py` |
| Flood extent (NRSC PDFs) | `ndem.nrsc.gov.in` — manually downloaded | `data/nrsc_inundation_pdfs/` |
| Discharge hydrographs | GloFAS-ERA5 reanalysis via CDS/EWDS | `scripts/pull_glofas_hydrograph.py` |
| Basin delineation | HydroBASINS Asia tile (Lehner/WWF) | `scripts/delineate_basins.py` |
| Historical flood extent | MODIS Global Flood Database (Tellman et al.) via GEE | `scripts/scan_modis_gfd_events.py` |

**Not available (confirmed gaps):**
- CWC/India-WRIS river discharge time series — Brahmaputra and Barak are classified transboundary basins; access requires a formal request + signed Secrecy Undertaking
- River bathymetry/cross-section data — no free public source found; will require a synthetic geomorphic approach
- GUARDIAN dataset — checked directly (station list downloaded, lat/lon verified); does NOT cover Assam (easternmost station at 86.9°E; Assam starts at ~89.5°E)

---

## Repository Structure

```
data/
  boundaries/          District polygons (scope4_districts.geojson + earlier Barak Valley files)
  basin_delineation/   HydroBASINS level-4 and level-6 overlays
  dem_fabdem/          FABDEM 30m per district
  era5_longrecord/     ERA5 hourly surface CSVs, 2000-01 to 2026-09 (321 files × 4 districts)
  icechunk_pressure/   ERA5 5×5 pressure-level parquet, 2000-01-01 to 2026-03-31
  gfm_validation/      GFM flood-extent GeoTIFFs, pre + during, all 4 districts
  glofas_discharge/    GloFAS daily discharge snapshots and hydrographs
  gfm_gap_years/       GFM coverage scan for years without pre-made products
  gfs_test/            GFS NWP test pull (Nagaon only; GFS line stopped, see TASK.md)
  lulc_dynamicworld/   Dynamic World composites per district
  soilgrids/           SoilGrids rasters per district (full depth profile)
  hysplit_trajectories/ HYSPLIT backward-trajectory outputs and plots
  nrsc_inundation_pdfs/ NRSC official flood maps (PDF, not GIS format)
  dynamic_susceptibility/ Susceptibility table, GFM labels, static/met features, models, eval_*.csv, diag/
  water_level_scope4.csv  CWC live gauge snapshot (2026-09-14 only, not historical)

scripts/
  # Data acquisition
  pull_era5_interactive.py         ERA5 surface, chunked/resumable
  fetch_icechunk_pressure_assam.py ERA5 pressure-level 5×5 grid
  pull_fabdem.py                   FABDEM DEM per district
  pull_dynamicworld_lulc.py        Land cover composites
  pull_soilgrids_full_profile.py   SoilGrids 6-depth full profile
  pull_gfm_validation.py           GFM flood-extent, polygon-clipped
  pull_glofas_discharge.py         GloFAS snapshot
  pull_glofas_hydrograph.py        GloFAS multi-week hydrographs
  delineate_basins.py              HydroBASINS overlay
  scan_modis_gfd_events.py         MODIS GFD per-district pixel scan
  parse_hysplit_trajectories.py    HYSPLIT output parser + plotter

  # Rainfall nowcasting model
  20_base_rainfall.py       Base model: occurrence classifier + XGBoost quantile regression + CSG-Gamma/GPD tail, all 4 districts
  30_forecast_horizons.py   Multi-horizon (t+1h..t+6h) sweep, all 4 districts
  50_pinn_assam.py          Pressure-grid CNN (physics term inert; call it "pressure-grid CNN", not PINN)
  70_stack_cnn_base_nagaon.py  CNN-stacking on base; adopted as production rainfall model for all 4 districts
  summarise_stack_runs.py   Aggregate stacking results across districts × seeds

  # Deprecated / exploratory (kept, not production)
  40_gfs_blend_nagaon.py    GFS NWP blend (tried, not adopted — see TASK.md)
  60_assembler_nagaon.py    4-member assembler (not adopted)
  80_heavy_detector_nagaon.py  Standalone heavy-event detector (not adopted)
  rainfall_nowcast_pilot.py    v0 pilot (XGBoost surface-only, superseded)

  # Dynamic flood susceptibility (see section below)
  170_dynamic_susceptibility.py ... 180_h6_dashboard.py

dashboard-app/    React + FastAPI dashboard (see dashboard-app/README.md)
docs/             dynamic_susceptibility_method.md (method + all amendments), literature review

kb_chroma/
  db/             ChromaDB vector store (detailed findings, numbers, debugging trails)
  add_kb.py       Add a new entry: python kb_chroma/add_kb.py --title "..." --text "..."
  query_kb.py     Search: python kb_chroma/query_kb.py "topic"
  ingest_kb.py    Re-sync KNOWLEDGE_BASE.md into Chroma (run after direct edits to KB)

KNOWLEDGE_BASE.md   Core standing decisions and short finding summaries (pointers to Chroma)
TASK.md             Full task log: what's done, what's open, decisions and their reasoning
CLAUDE.md           Mandatory project rules (read this before working on the codebase)
```

---

## Rainfall Nowcasting Model — Current State

Architecture (adapted from the Seychelles Phase 1 ensemble):

1. **Occurrence classifier** — XGBoost binary (rain/no-rain); lag-1 surface ERA5 features
2. **Quantile regression** (q10/q50/q90) — XGBoost, same features; adaptive per-district heavy-rain threshold (99.5th percentile of rainy-hour training rain, ~6-7mm; NOT the fixed 10mm inherited from Seychelles which was too high for ERA5's smoothed hourly tail on Assam)
3. **CSG-Gamma + Local GPD tail reshape** — reshapes the heavy-event probability interval only; does not change q50
4. **Pressure-grid CNN stacking** — CNN on the 5×5 ERA5 pressure-level grid, stacked onto the base via out-of-sample predictions; adopted as production because it replicates point-forecast corr_all gain across 46/48 (district × seed × lead) cells

**Production output**: `scripts/70_stack_cnn_base_nagaon.py` generalized to all 4 districts × 2 seeds in `summarise_stack_runs.py`. Results in `data/forecast_horizons_summary.csv`.

**Known limitations (not glossed over):**
- Heavy-interval coverage degrades at t+4..t+6h for Sivasagar and Karimganj (55–70% at h=6 vs. the 80% target) — root cause is discrimination loss in the gray zone (ERA5 is reanalysis, not a lead-time forecast product), not a calibration bug. GFS NWP blend was tried and did not fix it; accepted as a stated limitation.
- Sivasagar's stacking bias grows positive at h=4..h=6 and its h=4..h=6 RMSE gets worse with stacking — the only regression found; state this alongside Sivasagar's other flagged weaknesses in any writeup.
- Heavy-interval coverage gain (CNN stacking) does NOT replicate across seeds (19/48 cells) and should not be cited.
- The pressure-grid CNN physics term (moisture-flux convergence loss) is inert at current scale — call it "pressure-grid CNN," not PINN, in any writeup.
- Regional GPD (the 3rd leg of the Seychelles ensemble) is not implemented — needs a regional surface-TP spatial grid, not yet pulled.
- SLSQP blend (4th leg) was tried via the assembler and did not help.

---

## Dynamic Flood Susceptibility — Current State

Per-date flood probability for every 90 m cell, trained on Sentinel-1 flood maps (Copernicus GFM, 2016-2026, 1,676 usable scenes). Method, every amendment and its pre-registered criteria are in `docs/dynamic_susceptibility_method.md`; each amendment was written down before it was run.

| Model | Inputs | Use |
|---|---|---|
| AHP (A2) | Literature AHP weights (Choudhury & Dutta 2026) + 5-day rain | Baseline, no training |
| H2 | Terrain, rain, soil moisture, GloFAS Q/Q90 | Districts with no flood-map history |
| H5 (Amendment 4) | H2 + each cell's flood history | Superseded by H6 |
| **H6 (Amendment 6)** | H5 + what the most recent earlier Sentinel-1 scene (up to 30 days back) showed at each cell | **Working model for the 4 districts** |

Held-out results (leave-one-year-out, pooled per district):
- AUC: H2 0.87-0.92, H5 0.90-0.94, H6 0.94-0.97.
- Scenes where the predicted flooded area is off by more than 2x: H5 54-66%, H6 32-45% (past-only forward check 36-44%). Over- and under-estimation both drop.
- F1 with held-out thresholds: H5 0.28-0.42, H6 0.43-0.60.

Known limits (not glossed over):
- H6 needs a GFM scene from the last 30 days; the gain comes from scenes up to 12 days old.
- "Same as the last scene" alone is as good as H6 on flooded area in Sivasagar.
- Large sudden floods that follow a dry scene are still under-predicted.
- GFM is both input and target. An L-band (PALSAR-2) check found GFM partly missed water on low-lying cropland on about 4 of 9 worst over-prediction dates.
- For a district the model never saw, the F1 threshold did not transfer (Amendment 5). Choosing it on held-out districts and dropping absolute elevation fixes most of that, but adoption is still a pending decision.
- F1/CSI figures from scripts 170-173 used in-sample thresholds and are likely understated; AUC is unaffected.

Scripts: `170_dynamic_susceptibility.py` (table, P1-P6), `171`-`175` (Amendments 2-4, H5 dashboard), `174_lodo_threshold.py` (Amendment 5), `176_heavy_rain_results.py`, `177_extent_error_diag.py`, `178_independent_sensor_check.py` (PALSAR-2 / Sentinel-2 check), `179_amendment6.py` (H6), `180_h6_dashboard.py`. Outputs in `data/dynamic_susceptibility/` (`eval_*.csv`, `diag/`).

---

## Validation Data

Three events with both a pre-made flood-extent product (GFM GeoTIFF) and an official NRSC PDF inundation map:

| Event | GFM scene | NRSC PDF | GloFAS discharge ratio |
|---|---|---|---|
| Morigaon 2022-06 | ✓ | ✓ (2022-06-20, peak-3h) | 910→4106 m³/s (4.51×) |
| Nagaon 2022-06 | ✓ | ✓ (2022-06-20) | 810→3724 m³/s (4.60×) |
| Karimganj 2022-06 | ✓ | ✓ (2022-06-20) | 851→1710 m³/s (2.01×) |
| Karimganj 2024-06 | ✓ | ✓ (2024-06-20) | 844→1935 m³/s (2.29×) |
| Sivasagar 2026-07 | ✓ | ✓ (2026-07-20, 3h before peak) | 376→534 m³/s (1.42×, weakest) |

GloFAS discharge is a modeled reanalysis driven by ERA5 — it is not an independent observation and does not solve the calibration-independence gap; it supplies a boundary-condition proxy for HEC-RAS, not a ground-truth discharge record. State this plainly in the thesis.

Sivasagar 2026 is the weakest validation candidate on three independent grounds: contested causation (riverbed mining attributed as a major factor, not just rainfall), mostly non-Bay-of-Bengal HYSPLIT moisture source (0/8 endpoints in the BoB box vs. Morigaon's 5/8), and weak GloFAS discharge response (1.42× vs. 2–4.5× for the other events).

---

## Prerequisites

- Python 3.10+
- Google Earth Engine authenticated (`earthengine authenticate`)
- CDS/EWDS account registered and Terms of Use clicked for `cems-glofas-historical` (one-time manual step)
- `pip install earthengine-api geopandas rasterio xarray zarr icechunk cdsapi xgboost scikit-learn scipy`

HYSPLIT trajectories require HYSPLIT v5.4.2 installed separately (`hyts_std.exe`). The `pysplit` Python package was evaluated and rejected (unusable for this setup); scripts call `hyts_std.exe` directly.

---

## Prior Phases

- **Phase 1 — Sri Lanka** (`D:\BISAG-N\Sri_lanka`): methodology development and validation. GNN flood surrogate architecture and its documented bugs are in Sri Lanka's KNOWLEDGE_BASE.md. GNN zero-shot cross-basin transfer has failed twice on Sri Lankan basins — do not assume pre-trained weights transfer here.
- **Phase 2 — Arunachal Pradesh** (`D:\BISAG-N\India\Arunachal_Pradesh`): transferability test. The data-availability audit for Assam was done during this phase (see Chroma `manual-0170` there). The Gumbel-IDF + Sherman + Alternating Block Method design-storm pipeline (`generate_design_storms.py`) is reusable.
