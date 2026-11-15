"""
Small text-file helpers for the Namsai HEC-RAS project's .p01 (plan) and .u01
(unsteady flow) files. Shared by setup, event wiring, and the batch runner so
the fixed-width precipitation format lives in exactly one place.

Precipitation Hydrograph format (confirmed against the Kameng .u01 that
HEC-RAS itself wrote): `Precipitation Hydrograph= N`, then N values as
8-char right-justified fields, 10 per line.
"""
import re
from pathlib import Path


def format_precip_block(values):
    lines = []
    for i in range(0, len(values), 10):
        lines.append("".join(f"{v:8.2f}" for v in values[i:i + 10]))
    return "\n".join(lines)


def rewrite_precipitation(u01_path, values):
    """Replace the Precipitation Hydrograph block in-place."""
    u01_path = Path(u01_path)
    lines = u01_path.read_text().split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith("Precipitation Hydrograph="))
    n = int(lines[start].split("=")[1].strip())
    end, count = start + 1, 0
    while count < n:
        count += len(lines[end]) // 8
        end += 1
    new = lines[:start] + [f"Precipitation Hydrograph= {len(values)}"] + format_precip_block(values).split("\n") + lines[end:]
    u01_path.write_text("\n".join(new))


def read_precipitation(u01_path):
    lines = Path(u01_path).read_text().split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith("Precipitation Hydrograph="))
    n = int(lines[start].split("=")[1].strip())
    vals, i = [], start + 1
    while len(vals) < n:
        row = lines[i]
        vals += [float(row[j:j + 8]) for j in range(0, len(row), 8)]
        i += 1
    return vals


def _pad_fields(line, flow_area, bc_name):
    """Rebuild a `Boundary Location=` line keeping the fixed field widths of
    the template line (field 6 = flow area, field 8 = BC line name)."""
    head, rest = line.split("=", 1)
    fields = rest.split(",")
    widths = [len(f) for f in fields]
    fields[5] = flow_area.ljust(widths[5])
    fields[7] = bc_name.ljust(widths[7]) if bc_name else " " * widths[7]
    return head + "=" + ",".join(fields)


def write_flow_file(template, out_path, flow_area, outlet_bc, friction_slope, n_hours, title):
    """Render the .u01 from the template: flow-area/BC names, normal-depth
    friction slope at the outlet, and a zero precipitation series of n_hours
    (real forcing is written later with rewrite_precipitation)."""
    lines = Path(template).read_text().split("\n")
    out, seen_bc = [], 0
    for l in lines:
        if l.startswith("Flow Title="):
            l = f"Flow Title={title}"
        elif l.startswith("Boundary Location="):
            seen_bc += 1
            # 1st = outlet (normal depth) BC line, 2nd = precipitation on the flow area
            l = _pad_fields(l, flow_area, outlet_bc if seen_bc == 1 else "")
        elif l.startswith("Friction Slope="):
            l = f"Friction Slope={friction_slope:g}"
        out.append(l)
    Path(out_path).write_text("\n".join(out))
    rewrite_precipitation(out_path, [0.0] * n_hours)


def write_plan_file(template, out_path, title, short_id, start, end, comp_interval="1SEC"):
    """start/end are pandas-style timestamps -> HEC 'DDMMMYYYY,HHMM' format."""
    import pandas as pd

    def hec(t):
        t = pd.Timestamp(t)
        return f"{t:%d%b%Y}".upper() + f",{t:%H%M}"

    lines = Path(template).read_text().split("\n")
    out = []
    for l in lines:
        if l.startswith("Plan Title="):
            l = f"Plan Title={title}"
        elif l.startswith("Short Identifier="):
            l = f"Short Identifier={short_id:<80}"
        elif l.startswith("Simulation Date="):
            l = f"Simulation Date={hec(start)},{hec(end)}"
        elif l.startswith("Computation Interval="):
            l = f"Computation Interval={comp_interval}"
        out.append(l)
    Path(out_path).write_text("\n".join(out))


def set_simulation_window(p01_path, start, end):
    import pandas as pd

    def hec(t):
        t = pd.Timestamp(t)
        return f"{t:%d%b%Y}".upper() + f",{t:%H%M}"

    p = Path(p01_path)
    p.write_text(re.sub(r"(?m)^Simulation Date=.*$", f"Simulation Date={hec(start)},{hec(end)}", p.read_text()))
