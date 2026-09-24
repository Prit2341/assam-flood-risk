"""
Verification pull only (not the production script): real GloFAS-ERA5
historical discharge near the Kampur/Kopili gauge (Nagaon district) for a
pre-flood date and the verified 2022-06-20 flood peak, to confirm CDS/EWDS
access works and shows a real before/after signal before this data source
is treated as usable for the project.
"""
import cdsapi

client = cdsapi.Client()
dataset = "cems-glofas-historical"

# small box around Kampur, Nagaon (26.20N, 92.63E), N/W/S/E
area = [26.35, 92.50, 26.05, 92.80]

for label, y, m, d in [("pre", "2022", "06", "01"), ("during", "2022", "06", "20")]:
    request = {
        "system_version": ["version_4_0"],
        "hydrological_model": ["lisflood"],
        "product_type": ["consolidated"],
        "timespan": ["time_mean"],
        "variable": ["average_river_discharge_in_the_last_24_hours"],
        "year": [y],
        "month": [m],
        "day": [d],
        "data_format": "grib2",
        "download_format": "unarchived",
        "area": area,
    }
    target = f"data/glofas_discharge/glofas_test_kampur_{label}_{y}{m}{d}.grib"
    client.retrieve(dataset, request, target)
    print(f"Downloaded {label}: {target}")
