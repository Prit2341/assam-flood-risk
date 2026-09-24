"""
Parse HYSPLIT tdump backward-trajectory files and summarize/plot moisture
source endpoints, for all 3 validation events run so far (Morigaon 2022,
Karimganj 2024, Sivasagar 2026). See KNOWLEDGE_BASE.md manual-0024/0025/
0026/0027/0029 for context -- this fills the "no visualization or
quantitative endpoint summary" gap flagged there.

tdump format (HYSPLIT ASCII trajectory output), data line fields:
  traj# grid# YY MM DD HH mm fwd# age(h) lat lon height(m) pressure(hPa)
Header lines before the data vary in count; we locate the data block by
finding the line "PRESSURE"/"THETA"/"AIR_TEMP" and reading from the next
line onward.
"""
import glob
import os
import re

import matplotlib.pyplot as plt

OUTPUT_DIR = r"D:\HYSPLIT_extract\HYSPLIT\run_assam\output"
RESULT_DIR = r"D:\BISAG-N\India\Assam\data\hysplit_trajectories"

# Bay of Bengal rough bounding box used only to classify endpoints, not a
# precise coastline -- literature-cited moisture source region (manual-0024).
BOB_BOX = {"lat_min": 5.0, "lat_max": 22.0, "lon_min": 80.0, "lon_max": 95.0}

EVENTS = {
    "morigaon_2022": {
        "pattern": "tdump_2206*",
        "label": "Morigaon, 2022-06-17/18 (Kopili breach)",
    },
    "karimganj_2024": {
        "pattern": "tdump_karimganj_2024_*",
        "label": "Karimganj, 2024-06-19/24 (Kushiyara danger-level)",
    },
    "sivasagar_2026": {
        "pattern": "tdump_sivasagar_2026_*",
        "label": "Sivasagar, 2026-07-19/21 (Dikhow record flood)",
    },
}


def parse_tdump(path):
    with open(path) as fh:
        lines = fh.readlines()

    data_start = None
    for i, line in enumerate(lines):
        if re.search(r"PRESSURE|THETA|AIR_TEMP", line):
            data_start = i + 1
            break
    if data_start is None:
        raise ValueError(f"Could not find data header in {path}")

    points = []
    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) < 12:
            continue
        age_h = float(parts[8])
        lat = float(parts[9])
        lon = float(parts[10])
        height = float(parts[11])
        points.append((age_h, lat, lon, height))
    return points


def in_bay_of_bengal(lat, lon):
    return (
        BOB_BOX["lat_min"] <= lat <= BOB_BOX["lat_max"]
        and BOB_BOX["lon_min"] <= lon <= BOB_BOX["lon_max"]
    )


def process_event(event_key, event_cfg):
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, event_cfg["pattern"])))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        print(f"[{event_key}] No tdump files found matching {event_cfg['pattern']}")
        return None

    fig, ax = plt.subplots(figsize=(8, 8))
    summary_lines = [
        f"HYSPLIT backward-trajectory endpoint summary -- {event_cfg['label']}",
        "Source tdump files: " + OUTPUT_DIR,
        "",
        f"{'file':45s} {'start_lat':>9s} {'start_lon':>9s} {'end_lat':>9s} {'end_lon':>9s} {'in_BoB_box':>10s}",
    ]

    bob_count = 0
    for path in files:
        name = os.path.basename(path)
        pts = parse_tdump(path)
        if not pts:
            continue
        lats = [p[1] for p in pts]
        lons = [p[2] for p in pts]
        start_lat, start_lon = lats[0], lons[0]
        end_lat, end_lon = lats[-1], lons[-1]
        is_bob = in_bay_of_bengal(end_lat, end_lon)
        bob_count += int(is_bob)

        ax.plot(lons, lats, marker=".", markersize=2, linewidth=1, label=name.replace("tdump_", ""))
        ax.scatter([end_lon], [end_lat], marker="x", s=40, color="black", zorder=5)

        summary_lines.append(
            f"{name:45s} {start_lat:9.3f} {start_lon:9.3f} {end_lat:9.3f} {end_lon:9.3f} {str(is_bob):>10s}"
        )

    summary_lines.append("")
    summary_lines.append(
        f"{bob_count}/{len(files)} trajectory endpoints fall inside the rough Bay-of-Bengal "
        f"box (lat {BOB_BOX['lat_min']}-{BOB_BOX['lat_max']}, lon {BOB_BOX['lon_min']}-{BOB_BOX['lon_max']})."
    )
    summary_lines.append(
        "This is a coarse geographic check only (endpoint-in-box), NOT a moisture-flux "
        "or back-trajectory-clustering analysis -- see manual-0024/0025 for the caveat "
        "that quantitative source attribution (%% BoB vs Arabian Sea vs local recycling) "
        "needs a dedicated moisture-tracking method, not demonstrated here."
    )

    ax.axvspan(BOB_BOX["lon_min"], BOB_BOX["lon_max"], ymin=0, ymax=1, color="blue", alpha=0.03)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"120h backward trajectories -- {event_cfg['label']}")
    ax.legend(fontsize=6, loc="upper left")
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(RESULT_DIR, f"{event_key}_trajectories.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{event_key}] Saved plot:", plot_path)

    summary_path = os.path.join(RESULT_DIR, f"{event_key}_trajectory_summary.txt")
    with open(summary_path, "w") as fh:
        fh.write("\n".join(summary_lines) + "\n")
    print(f"[{event_key}] Saved summary:", summary_path)
    print()
    print("\n".join(summary_lines))
    print()
    return bob_count, len(files)


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    for event_key, event_cfg in EVENTS.items():
        process_event(event_key, event_cfg)


if __name__ == "__main__":
    main()
