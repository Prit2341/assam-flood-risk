# Rainfall prediction model, version 1 (frozen 2026-10-09)

**What v1 is:** the Assam rainfall prediction model as it stood before the 2026-10-09 Seychelles-method
experiments. It is the Seychelles *architecture* adapted for Assam (decisions of 2026-09-16), not the full
Seychelles pipeline.

- Base model (`scripts/20_base_rainfall.py`): occurrence classifier + quantile regression (q10/q50/q90)
  + heavy-rain gate with soft blending, centre-grid-point features only.
- Interval: CSG-Gamma + Local GPD, mean of 2 (no Regional GPD).
- Horizons t+1..t+6 h (`scripts/30_forecast_horizons.py`).
- Production stack: base model + out-of-fold pressure-grid CNN prediction as an extra feature
  (`scripts/50_pinn_assam.py` is the CNN; its physics term is inert, so it is a plain CNN and is called one;
  `scripts/70_stack_cnn_base_nagaon.py`). Adopted for all 4 districts (point-forecast gain in 46/48 cells).
- Exploratory Nagaon-only scripts kept for the record: 40 (GFS blend), 60 (assembler), 80 (heavy detector).

**Not in v1:** Seychelles grid-distillation features (`scripts/90_distill_features.py`), Regional GPD
(`scripts/100_regional_gpd.py`), any real physics-informed model. Those belong to v2 candidates.

**Contents:** `scripts/` (code), `models/` (saved base models per district), `results/` (predictions, summaries,
cached out-of-fold CNN features). `MANIFEST.sha256` holds a SHA-256 for every file; verified on creation.
Files are marked read-only.

**Inputs not copied (too large, unchanged):** `data/era5_longrecord/`, `data/icechunk_pressure/`,
`data/dem_fabdem/`, `data/lulc_dynamicworld_latest/`. v1 depends on them at their existing paths.

**To revert to v1:** copy `scripts/*` back over `D:\BISAG-N\India\Assam\scripts\`, `models/*` over `models\`, and
`results/*` over `data\rainfall_nowcast\`; then verify with
`awk '{print $1"  "$3}' MANIFEST.sha256 | sha256sum -c`. (The live scripts 20/30/50/70 were byte-identical
to this snapshot at creation.) Do not edit files in this folder; make v2 elsewhere.
