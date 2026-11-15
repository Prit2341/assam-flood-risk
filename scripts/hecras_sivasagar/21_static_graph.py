"""Static mesh graph of the frozen teacher (c600 d4) for the GNN student: node features (cell x,y, min elevation, area, Manning n), edges = faces joining two cells.
Cells beyond ncell (=26038) are boundary ghost cells of the 2D area. Out: digital_twin/sivasagar/surrogate/static_graph.npz"""
import sys
from pathlib import Path
import h5py, numpy as np
ROOT = Path(__file__).resolve().parents[2]
import os  # overrides (2026-11-06): STATIC_SRC (result hdf name in results/), STATIC_OUT (file name in surrogate/)
src = ROOT / "digital_twin/sivasagar/results" / os.environ.get("STATIC_SRC", "c600_d4_r30_0705_432h_cal.p01.hdf"); A = "Geometry/2D Flow Areas/Sivasagar_Domain"
with h5py.File(src) as f:
    g = f[A]; xy = g["Cells Center Coordinate"][:]; zmin = g["Cells Minimum Elevation"][:]; area = g["Cells Surface Area"][:]; mann = g["Cells Center Manning's n"][:]
    fc = g["Faces Cell Indexes"][:]; fl = g["Faces NormalUnitVector and Length"][:]; fz = g["Faces Minimum Elevation"][:]
inner = (fc[:, 0] >= 0) & (fc[:, 1] >= 0) & (fc[:, 0] < 26038) & (fc[:, 1] < 26038)
print("cells", len(xy), "real", 26038, "faces", len(fc), "faces between two real cells", int(inner.sum()))
np.savez_compressed(ROOT / "digital_twin/sivasagar/surrogate" / os.environ.get("STATIC_OUT", "static_graph.npz"), xy=xy, zmin=zmin, area=area, mann=mann, face_cells=fc, face_nx_ny_len=fl, face_zmin=fz, n_real=26038)
