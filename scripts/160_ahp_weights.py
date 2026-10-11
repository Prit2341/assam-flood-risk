"""
AHP weights from a PUBLISHED Assam pairwise-comparison matrix -- no pairwise judgements invented here.

Source: Choudhury, S. & Dutta, M. (2026) "Flood susceptibility prioritization in Dimoria block Assam using
GIS and analytical hierarchy process", Discover Environment 4:138, doi:10.1007/s44274-026-00623-5
(open access). Table 4 (pairwise matrix, 11 factors), Table 6 (weights), CR reported 0.061.
Method: Saaty (1980) principal-eigenvector weights, CI = (lambda_max - n)/(n - 1), CR = CI/RI,
RI from Saaty (1980): n=10 -> 1.49, n=11 -> 1.51.

Steps:
  1. reproduce the paper's weights and CR from its own matrix (sanity check of the transcription);
  2. drop lithology (no usable digital geology map for our districts) by taking the 10x10 sub-matrix of
     the SAME judgements and recomputing the eigenvector and CR (judgements unchanged, only renormalised).
The weights were derived for Dimoria block (Kamrup Metro, Brahmaputra valley), not for our districts;
this is stated, and a one-at-a-time weight sensitivity test is run in script 170 (Chen et al. 2021,
arXiv:2107.13368 style +-perturbation).

Out: data/dynamic_susceptibility/ahp_weights.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dyn_common as dc

F = ["slope", "elevation", "rainfall", "lulc", "ndvi", "soil", "drain_dens", "dist_river", "lithology", "tri", "sti"]
t = 1 / 3; h = 1 / 2; f = 1 / 5
M = np.array([
    # Slope Elev Rain LULC NDVI Soil DD  Dist Lith TRI STI
    [1, 1, t, h, h, t, h, t, t, h, t],      # Slope
    [1, 1, t, h, h, t, h, t, t, h, t],      # Elevation
    [3, 3, 1, 2, 2, 3, 2, 3, 3, 2, 5],      # Rainfall
    [2, 2, h, 1, 1, 2, 1, 2, 2, 1, 3],      # LULC
    [2, 2, h, 1, 1, 2, 1, 2, 2, 1, 3],      # NDVI
    [3, 3, t, h, h, 1, h, t, t, h, t],      # Soil
    [2, 2, h, 1, 1, 2, 1, 2, 2, 1, 3],      # Drainage density
    [3, 3, t, h, h, 3, h, 1, 1, h, 1],      # Distance from river
    [3, 3, t, h, h, 3, h, 1, 1, h, 1],      # Lithology
    [2, 2, h, 1, 1, 2, 1, 2, 2, 1, 3],      # TRI
    [3, 3, f, t, t, 3, t, 1, 1, t, 1],      # STI
], dtype=float)
PUBLISHED_W = {"slope": .041, "elevation": .041, "rainfall": .193, "lulc": .111, "ndvi": .111, "soil": .068,
               "drain_dens": .111, "dist_river": .078, "lithology": .078, "tri": .111, "sti": .057}
RI = {3: .58, 4: .90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51}


def ahp(A):
    vals, vecs = np.linalg.eig(A)
    k = np.argmax(vals.real)
    w = np.abs(vecs[:, k].real); w /= w.sum()
    n = len(A)
    lam = vals[k].real
    ci = (lam - n) / (n - 1)
    return w, lam, ci, ci / RI[n]


def main():
    recip = np.allclose(M * M.T, 1.0)
    bad = [(F[i], F[j], M[i, j], M[j, i]) for i in range(11) for j in range(11) if not np.isclose(M[i, j] * M[j, i], 1)]
    w11, lam11, ci11, cr11 = ahp(M)
    print(f"reciprocal: {recip}; non-reciprocal pairs: {bad}")
    print(f"11-factor: lambda_max={lam11:.4f} CI={ci11:.4f} CR={cr11:.4f} (paper: 0.061)")
    for k, v in zip(F, w11):
        print(f"   {k:11s} ours {v:.3f}  paper {PUBLISHED_W[k]:.3f}")
    keep = [i for i, k in enumerate(F) if k != "lithology"]
    w10, lam10, ci10, cr10 = ahp(M[np.ix_(keep, keep)])
    F10 = [F[i] for i in keep]
    print(f"10-factor (lithology dropped): lambda_max={lam10:.4f} CI={ci10:.4f} CR={cr10:.4f}")
    for k, v in zip(F10, w10):
        print(f"   {k:11s} {v:.3f}")
    out = {"source": "Choudhury & Dutta 2026, Discover Environment 4:138, Table 4",
           "matrix_reciprocal": bool(recip), "non_reciprocal_pairs": [list(map(str, b)) for b in bad],
           "cr_11": cr11, "weights_11": dict(zip(F, w11.round(4).tolist())),
           "cr_10": cr10, "weights_10": dict(zip(F10, w10.round(4).tolist()))}
    os.makedirs(dc.DYN, exist_ok=True)
    json.dump(out, open(os.path.join(dc.DYN, "ahp_weights.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
