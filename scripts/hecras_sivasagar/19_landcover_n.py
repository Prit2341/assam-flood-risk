"""Land-cover Manning's n test: copy hecras_c600_d4 -> hecras_c600_d4_lcn, add an ESA WorldCover (100 m) land-cover layer with the NLCD-style starter n table
(ras_commander default values, mapped to WorldCover classes), associate it to the geometry, run 07-05 + 432 h at runoff 30%, score on 07-17.
  python -u scripts/hecras_sivasagar/19_landcover_n.py"""
import shutil, sys, time
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import hecras_io as rio
from ras_commander import init_ras_project, RasCmdr
from ras_commander.RasMap import RasMap
ROOT = HERE.parents[1]; SV = ROOT / "digital_twin/sivasagar"; src = SV / "hecras_c600_d4"; h = SV / "hecras_c600_d4_lcn"; NAME = "Sivasagar_2D"
if h.exists(): shutil.rmtree(h)
shutil.copytree(src, h, ignore=shutil.ignore_patterns("*.p01.hdf", "*.u01.hdf", "*.log", "*.b01", "*.bco01", "*.ic.o01", "*.x01", "*.tmp.hdf"))
N = {10: ("Tree cover", 0.12), 20: ("Shrubland", 0.08), 30: ("Grassland", 0.04), 40: ("Cropland", 0.04), 50: ("Built-up", 0.09), 60: ("Bare", 0.035),
     70: ("Snow", 0.03), 80: ("Open water", 0.025), 90: ("Herbaceous wetland", 0.07), 95: ("Mangroves", 0.10), 100: ("Moss", 0.05)}
tab = pd.DataFrame([dict(source_value=k, class_id=k, class_name=v[0], mannings_n=v[1], percent_impervious=0.0) for k, v in N.items()])
ras = init_ras_project(str(h), "7.0.1")
lc = RasMap.add_landcover_layer(ras_project_path=h / f"{NAME}.prj", source_path=SV / "data/worldcover_100m.tif", classification_table=tab, cell_size=100.0,
                                layer_name="LandCover", output_hdf_path=h / "LandCover" / "LandCover.hdf")
print("landcover hdf", lc, flush=True)
RasMap.associate_geometry_layers(ras_project_path=h / f"{NAME}.prj", geom_file=h / f"{NAME}.g01", landcover_hdf_path=lc, terrain_hdf_path=h / "Terrain" / "Terrain.hdf")
t0 = pd.Timestamp("2026-07-05 00:00"); t1 = t0 + pd.Timedelta(hours=432)
rain = pd.read_csv(SV / "data/era5land_domain_hourly_2026-07-01_2026-08-26.csv", parse_dates=["timestamp"]).set_index("timestamp").tp_mm[t0:t1]
rio.set_simulation_window(h / f"{NAME}.p01", t0, t1); rio.rewrite_precipitation(h / f"{NAME}.u01", (rain.values * 0.3).tolist())
ras = init_ras_project(str(h), "7.0.1"); t = time.time()
ok = RasCmdr.compute_plan("01", ras_object=ras, verify=True, dialog_watchdog=False, clear_geompre=True); print("done", ok, round(time.time() - t), flush=True)
shutil.copy(h / f"{NAME}.p01.hdf", SV / "results/c600_lcn_r30_0705_432h.p01.hdf")
