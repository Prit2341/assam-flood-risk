"""
RF Flood Susceptibility Model — Assam 4 Districts
Tehrany et al. 2014: flood inventory labels + terrain features + RF classifier

Features per pixel:
  - elevation  (FABDEM 30m)
  - slope      (degrees, derived from FABDEM)
  - TWI        (Topographic Wetness Index = ln(flow_acc / tan(slope)))
  - flow_acc   (upstream drainage area, derived from FABDEM via pysheds)

Labels:
  - 1 = flood   (from data/gfm_validation/*.tif)
  - 0 = no-flood (from data/gfm_nonflood/*.tif)

Output:
  - flood_susceptibility/<district>_susceptibility_rf.tif  (prob raster)
  - flood_susceptibility/<district>_susceptibility_rf.geojson
  - flood_susceptibility/rf_model_summary.txt
"""

import json
import os
import warnings
from pathlib import Path

import numpy as np

# pysheds uses np.in1d which was removed in NumPy 2.0 — restore it
if not hasattr(np, "in1d"):
    np.in1d = np.isin

import pandas as pd
import rasterio
import rasterio.warp
from pyproj import Transformer
from pysheds.grid import Grid
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from shapely.geometry import mapping, shape
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).parent.parent
FABDEM_DIR = BASE_DIR / "data" / "dem_fabdem"
GFM_FLOOD  = BASE_DIR / "data" / "gfm_validation"
GFM_DRY    = BASE_DIR / "data" / "gfm_nonflood"
OUT_DIR    = Path(__file__).parent
BOUNDS_GJ  = BASE_DIR / "data" / "boundaries" / "assam_districts.geojson"

DISTRICTS = ["Morigaon", "Nagaon", "Karimganj", "Sivasagar"]
SAMPLES_PER_CLASS = 8000   # per district per label
RANDOM_STATE      = 42
GFM_CRS           = "EPSG:27703"   # Equi7 Asia — native GFM CRS
FABDEM_CRS        = "EPSG:4326"

# ── helpers ──────────────────────────────────────────────────────────────────

def fabdem_path(district: str) -> Path:
    name = "Morigaon" if district == "Marigaon" else district
    return FABDEM_DIR / f"{name}_fabdem_30m.tif"


def gfm_rasters(district: str, label: int) -> list[Path]:
    """Return all GFM tif paths for a district with the given label (0 or 1)."""
    src = GFM_FLOOD if label == 1 else GFM_DRY
    sub = src / district
    if not sub.exists():
        return []
    return sorted(sub.glob("*.tif"))


def compute_terrain(dem_path: Path):
    """Return (elevation, slope_deg, flow_acc, twi) arrays in FABDEM's own CRS/grid."""
    grid = Grid.from_raster(str(dem_path))
    dem  = grid.read_raster(str(dem_path))

    # Fill pits and resolve flats
    pit_filled  = grid.fill_pits(dem)
    flooded     = grid.fill_depressions(pit_filled)
    inflated    = grid.resolve_flats(flooded)

    # Flow direction + accumulation
    fdir     = grid.flowdir(inflated)
    flow_acc = grid.accumulation(fdir).astype(np.float32) + 1  # +1 avoids log(0)

    # Elevation array
    with rasterio.open(dem_path) as src:
        elev      = src.read(1).astype(np.float32)
        transform = src.transform
        profile   = src.profile
        nodata    = src.nodata if src.nodata is not None else -9999

    # Slope from numpy gradient (degrees)
    res_x = abs(transform.a)
    res_y = abs(transform.e)
    # convert degrees-per-pixel to metres-per-pixel (~111km per degree)
    res_m = ((res_x + res_y) / 2) * 111_000
    dy, dx  = np.gradient(elev, res_m, res_m)
    slope_r = np.sqrt(dx**2 + dy**2)            # rise/run
    slope_d = np.degrees(np.arctan(slope_r))     # degrees

    # TWI = ln(flow_acc / tan(slope))
    slope_safe = np.where(slope_r < 1e-6, 1e-6, slope_r)
    twi = np.log(flow_acc / slope_safe).astype(np.float32)

    mask = elev == nodata
    for arr in (slope_d, flow_acc, twi):
        arr[mask] = np.nan

    return elev, slope_d.astype(np.float32), flow_acc, twi, transform, profile


def sample_pixels(raster_paths: list[Path], label: int, n: int,
                  elev: np.ndarray, slope: np.ndarray,
                  flow_acc: np.ndarray, twi: np.ndarray,
                  dem_transform, dem_crs: str, gfm_crs: str) -> pd.DataFrame:
    """
    Sample n pixels from GFM rasters (flood or nonflood), extract terrain
    features from the already-computed DEM arrays at matching locations.
    """
    transformer = Transformer.from_crs(gfm_crs, dem_crs, always_xy=True)
    rows_list   = []

    for path in raster_paths:
        with rasterio.open(path) as src:
            arr   = src.read(1)
            nd    = src.nodata
            valid = arr != nd if nd is not None else np.ones_like(arr, bool)
            if label == 1:
                pixel_mask = (arr == 1) & valid
            else:
                pixel_mask = (arr == 0) & valid

        ys, xs = np.where(pixel_mask)
        if len(ys) == 0:
            continue

        # Random subsample to avoid one raster dominating
        n_take = min(len(ys), n // max(len(raster_paths), 1) + 1)
        idx    = np.random.default_rng(RANDOM_STATE).choice(len(ys), n_take, replace=False)
        ys, xs = ys[idx], xs[idx]

        # Convert pixel centres to GFM CRS coordinates
        with rasterio.open(path) as src:
            gfm_xs, gfm_ys = rasterio.transform.xy(src.transform, ys, xs)

        # Reproject to FABDEM CRS
        dem_xs, dem_ys = transformer.transform(gfm_xs, gfm_ys)

        # Convert to FABDEM pixel indices
        rows_px, cols_px = rasterio.transform.rowcol(dem_transform, dem_xs, dem_ys)
        rows_px = np.clip(rows_px, 0, elev.shape[0] - 1)
        cols_px = np.clip(cols_px, 0, elev.shape[1] - 1)

        e  = elev[rows_px, cols_px]
        sl = slope[rows_px, cols_px]
        fa = flow_acc[rows_px, cols_px]
        tw = twi[rows_px, cols_px]

        valid_feat = ~(np.isnan(e) | np.isnan(sl) | np.isnan(fa) | np.isnan(tw))
        for i in np.where(valid_feat)[0]:
            rows_list.append({
                "elevation": float(e[i]),
                "slope":     float(sl[i]),
                "flow_acc":  float(fa[i]),
                "twi":       float(tw[i]),
                "label":     label,
            })

    df = pd.DataFrame(rows_list)
    if len(df) > n:
        df = df.sample(n, random_state=RANDOM_STATE)
    return df


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    all_train = []
    terrain_cache = {}

    print("=" * 60)
    print("Step 1: Compute terrain features for all 4 districts")
    print("=" * 60)

    for district in DISTRICTS:
        dp = fabdem_path(district)
        if not dp.exists():
            print(f"  {district}: FABDEM not found at {dp} — skip")
            continue
        print(f"  {district}: computing terrain from {dp.name} ...")
        elev, slope, flow_acc, twi, transform, profile = compute_terrain(dp)
        terrain_cache[district] = (elev, slope, flow_acc, twi, transform, profile)
        print(f"    elev  min={np.nanmin(elev):.1f}  max={np.nanmax(elev):.1f} m")
        print(f"    slope min={np.nanmin(slope):.1f} max={np.nanmax(slope):.1f} deg")
        print(f"    TWI   min={np.nanmin(twi):.2f}  max={np.nanmax(twi):.2f}")

    print("\n" + "=" * 60)
    print("Step 2: Sample training pixels")
    print("=" * 60)

    for district in DISTRICTS:
        if district not in terrain_cache:
            continue
        elev, slope, flow_acc, twi, transform, _ = terrain_cache[district]

        flood_paths   = gfm_rasters(district, label=1)
        nonflood_paths = gfm_rasters(district, label=0)
        print(f"\n  {district}: {len(flood_paths)} flood rasters, "
              f"{len(nonflood_paths)} nonflood rasters")

        if not flood_paths or not nonflood_paths:
            print(f"  WARNING: missing rasters for {district}, skipping")
            continue

        df_flood = sample_pixels(flood_paths, 1, SAMPLES_PER_CLASS,
                                  elev, slope, flow_acc, twi,
                                  transform, FABDEM_CRS, GFM_CRS)
        df_dry   = sample_pixels(nonflood_paths, 0, SAMPLES_PER_CLASS,
                                  elev, slope, flow_acc, twi,
                                  transform, FABDEM_CRS, GFM_CRS)

        print(f"    flood samples:    {len(df_flood)}")
        print(f"    nonflood samples: {len(df_dry)}")
        df_flood["district"] = district
        df_dry["district"]   = district
        all_train.extend([df_flood, df_dry])

    train_df = pd.concat(all_train, ignore_index=True)
    print(f"\nTotal training samples: {len(train_df)}  "
          f"(flood={train_df.label.sum()}, nonflood={(train_df.label==0).sum()})")

    # Save training data
    train_csv = OUT_DIR / "rf_training_data.csv"
    train_df.to_csv(train_csv, index=False)
    print(f"Training data saved: {train_csv}")

    print("\n" + "=" * 60)
    print("Step 3: Train Random Forest")
    print("=" * 60)

    FEATURES = ["elevation", "slope", "flow_acc", "twi"]
    X = train_df[FEATURES].values
    y = train_df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)

    report = classification_report(y_test, y_pred,
                                    target_names=["nonflood", "flood"])
    print(report)
    print(f"ROC-AUC: {auc:.4f}")

    importances = pd.Series(rf.feature_importances_, index=FEATURES)
    print("\nFeature importances:")
    print(importances.sort_values(ascending=False).to_string())

    summary_lines = [
        "RF Flood Susceptibility — Assam 4 Districts",
        f"Training samples: {len(X_train)} | Test samples: {len(X_test)}",
        f"ROC-AUC: {auc:.4f}",
        "", report,
        "\nFeature importances:",
        importances.sort_values(ascending=False).to_string(),
    ]
    (OUT_DIR / "rf_model_summary.txt").write_text("\n".join(summary_lines))

    print("\n" + "=" * 60)
    print("Step 4: Predict susceptibility maps for all 4 districts")
    print("=" * 60)

    for district in DISTRICTS:
        if district not in terrain_cache:
            continue
        elev, slope, flow_acc, twi, transform, profile = terrain_cache[district]

        h, w = elev.shape
        feat_flat = np.stack([
            elev.ravel(), slope.ravel(),
            flow_acc.ravel(), twi.ravel()
        ], axis=1).astype(np.float32)

        valid_mask = ~np.isnan(feat_flat).any(axis=1)
        prob_flat  = np.full(h * w, np.nan, dtype=np.float32)

        if valid_mask.sum() > 0:
            prob_flat[valid_mask] = rf.predict_proba(
                feat_flat[valid_mask])[:, 1]

        prob_map = prob_flat.reshape(h, w)

        # Save GeoTIFF
        out_tif = OUT_DIR / f"{district}_susceptibility_rf.tif"
        out_profile = profile.copy()
        out_profile.update(dtype=rasterio.float32, count=1, nodata=-9999)
        prob_save = np.where(np.isnan(prob_map), -9999, prob_map)
        with rasterio.open(out_tif, "w", **out_profile) as dst:
            dst.write(prob_save.astype(np.float32), 1)
        print(f"  {district}: saved {out_tif.name}  "
              f"(prob range {np.nanmin(prob_map):.3f}–{np.nanmax(prob_map):.3f})")

        # Save GeoJSON (vectorised grid at 0.1 threshold steps)
        out_geojson = OUT_DIR / f"{district}_susceptibility_rf.geojson"
        save_geojson(prob_map, transform, FABDEM_CRS, out_geojson)
        print(f"           GeoJSON: {out_geojson.name}")

    print("\nDone. All outputs in:", OUT_DIR)


def save_geojson(prob_map: np.ndarray, transform, crs: str, out_path: Path):
    """Convert susceptibility raster to a GeoJSON grid (downsampled 5x for size)."""
    import rasterio.features
    from rasterio.crs import CRS

    # Downsample 5x to keep GeoJSON manageable
    step = 5
    pm   = prob_map[::step, ::step]
    h, w = pm.shape
    new_transform = rasterio.transform.from_origin(
        transform.c, transform.f,
        abs(transform.a) * step, abs(transform.e) * step
    )

    features = []
    for r in range(h):
        for c in range(w):
            v = pm[r, c]
            if np.isnan(v):
                continue
            x, y = rasterio.transform.xy(new_transform, r, c)
            dx = abs(new_transform.a) / 2
            dy = abs(new_transform.e) / 2
            coords = [[
                [x - dx, y - dy], [x + dx, y - dy],
                [x + dx, y + dy], [x - dx, y + dy],
                [x - dx, y - dy],
            ]]
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": coords},
                "properties": {
                    "susceptibility": round(float(v), 4),
                    "class": (
                        "Very High" if v >= 0.8 else
                        "High"      if v >= 0.6 else
                        "Medium"    if v >= 0.4 else
                        "Low"       if v >= 0.2 else
                        "Very Low"
                    ),
                },
            })

    gj = {"type": "FeatureCollection", "features": features}
    with open(out_path, "w") as f:
        json.dump(gj, f)


if __name__ == "__main__":
    main()
