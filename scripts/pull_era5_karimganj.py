"""
Pilot pull: 32 of the 33 A10-A42 forecasting variables (AP spec) for Karimganj,
sourced entirely from GEE's ECMWF/ERA5/HOURLY collection (single consistent
source, no multi-provider patchwork). A42 (convection-permitting <=3km
rainfall) is excluded - not derivable from ERA5 at any resolution, a native
limit of the reanalysis product, not a GEE gap.

Pilot window: 2024-06-10 to 2024-06-25 (spans pre-flood baseline through the
documented Karimganj 2024 flood onset, ~19 June 2024 per news research).
Area-averaged over the real Karimganj district polygon (not a bounding box).
"""
import json
import ee
import pandas as pd

ee.Initialize(project='assamflood-508209')

BANDS = [
    "total_precipitation",
    "convective_available_potential_energy",
    "convective_inhibition",
    "total_column_water_vapour",
    "temperature_2m",
    "dewpoint_temperature_2m",
    "surface_pressure",
    "mean_sea_level_pressure",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "u_component_of_wind_100m",
    "v_component_of_wind_100m",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
    "boundary_layer_height",
    "instantaneous_10m_wind_gust",
    "skin_temperature",
    "sea_surface_temperature",
    "low_cloud_cover",
    "medium_cloud_cover",
    "high_cloud_cover",
    "total_cloud_cover",
    "cloud_base_height",
    "evaporation",
    "surface_solar_radiation_downwards",
    "surface_thermal_radiation_downwards",
    "surface_net_solar_radiation",
    "surface_net_thermal_radiation",
    "mean_surface_direct_short_wave_radiation_flux",
]
print(f"Pulling {len(BANDS)} ERA5 bands (covers 32 of the 33 A10-A42 spec "
      f"variables; A42 convection-permitting <=3km rainfall excluded - not "
      f"derivable from ERA5 at any resolution)")

with open(r"D:\BISAG-N\India\Assam\data\boundaries\barak_valley_districts.geojson") as f:
    gj = json.load(f)
karimganj_geom = next(f["geometry"] for f in gj["features"] if f["properties"]["dtname"] == "Karimganj")
region = ee.Geometry(karimganj_geom)

start, end = "2024-06-10", "2024-06-26"
col = (ee.ImageCollection("ECMWF/ERA5/HOURLY")
       .filterDate(start, end)
       .filterBounds(region)
       .select(BANDS))

def reduce_image(img):
    stats = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=27830, maxPixels=1e9)
    return ee.Feature(None, stats).set("system:time_start", img.get("system:time_start"))

fc = col.map(reduce_image)
result = fc.getInfo()

rows = []
for f in result["features"]:
    props = f["properties"]
    ts = props.pop("system:time_start")
    props["timestamp"] = pd.to_datetime(ts, unit="ms")
    rows.append(props)

df = pd.DataFrame(rows).sort_values("timestamp")
out_path = r"D:\BISAG-N\India\Assam\data\rainfall_nowcast\era5_karimganj_pilot_202406.csv"
df.to_csv(out_path, index=False)
print(f"Pulled {len(df)} hourly timesteps, {len(BANDS)} variables")
print(f"Saved to {out_path}")
print(df[["timestamp", "total_precipitation", "temperature_2m"]].head())
print(df[["timestamp", "total_precipitation", "temperature_2m"]].tail())
