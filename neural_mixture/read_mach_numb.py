"""
read_mach_numb – plot isentropic Mach and total-pressure loss for
OpenFOAM RANS cases versus a reference LES simulation.

CLI usage (after ``pip install -e``)::

    read-mach-numb cases.json [options]

JSON cases format::

    {
        "baseline": {
            "case":  "case.foam",
            "fName": "Baseline",
            "color": "tab:green",
            "label": "sim_1"
        }
    }
"""

import argparse
import json
import os
import re
from pathlib import Path

# Absolute path to the neural_mixture package directory (where extract_data_all.py lives).
_PKG_DIR = Path(__file__).parent

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# ── Module-level defaults (overridable at runtime via main() CLI args) ────────
c_ref = 0.07
SURFACE_Z = 0.01
SURFACE_ATOL = 1e-4

CAMBER_SMOOTH_POLY = 2
CAMBER_SMOOTH_FRAC = 0.08

mp1_pattern = "{case}_MP1.csv"
mp2_pattern = "{case}_MP2.csv"


# ── Helper functions (unchanged from standalone script) ───────────────────────

def stagnation_pressure(p, T, Umag, gamma, R):
    a = np.sqrt(gamma * R * T)
    M = Umag / a
    return p * (1.0 + 0.5 * (gamma - 1.0) * M**2) ** (gamma / (gamma - 1.0))


def read_expe_lrn_ogv_Mis(path_txt):
    if not os.path.exists(path_txt):
        return pd.DataFrame(columns=["x_over_l", "Mis"])
    x_vals, m_vals = [], []
    with open(path_txt, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or "x/l" in s.lower():
                continue
            parts = [p for p in re.split(r"[\t ]+", s.replace(",", ".")) if p]
            if len(parts) >= 2:
                try:
                    x_vals.append(float(parts[0]))
                    m_vals.append(float(parts[1]))
                except ValueError:
                    pass
    df = pd.DataFrame({"x_over_l": x_vals, "Mis": m_vals})
    return df.sort_values("x_over_l", kind="mergesort").reset_index(drop=True)


def read_expe_lrn_ogv_loss(filename):
    """Read experimental LRN OGV data (handles comma decimals)."""
    try:
        df = pd.read_csv(filename, sep=";")
    except Exception:
        df = pd.read_csv(filename, sep=",")
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.replace(",", ".").astype(float)
    return df["eta"].to_numpy(), df["Losses"].to_numpy()


def _pca_chord_frame(xy):
    ctr = xy.mean(axis=0)
    X = xy - ctr
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return ctr, Vt[0], Vt[1]


def _project_sn(xy, ctr, e_s, e_n):
    X = xy - ctr
    return X @ e_s, X @ e_n


def _savgol_window(nbins, frac=CAMBER_SMOOTH_FRAC, poly=CAMBER_SMOOTH_POLY):
    w = int(max(5, round(nbins * frac)))
    if w % 2 == 0:
        w += 1
    w = max(w, poly + 3 if (poly + 3) % 2 == 1 else poly + 4)
    w = min(w, nbins - (1 - nbins % 2))
    if w < 5:
        w = 5 if nbins >= 5 else nbins - (1 - nbins % 2)
    if w % 2 == 0:
        w = max(5, w - 1)
    return max(5, w)


def _build_camber_from_surface(surface_df, nbins=200):
    xy = surface_df[["Points_0", "Points_1"]].to_numpy(float)
    ctr, e_s, e_n = _pca_chord_frame(xy)
    s, n = _project_sn(xy, ctr, e_s, e_n)

    s0, s1 = np.min(s), np.max(s)
    span = max(s1 - s0, 1e-12)
    s_norm = (s - s0) / span

    edges = np.linspace(0.0, 1.0, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.floor(s_norm * nbins).astype(int), 0, nbins - 1)

    n_min = np.full(nbins, np.nan)
    n_max = np.full(nbins, np.nan)
    for k in range(nbins):
        mask = idx == k
        if not np.any(mask):
            continue
        n_k = n[mask]
        n_min[k], n_max[k] = np.nanmin(n_k), np.nanmax(n_k)

    if np.nanmean(n_max) < -np.nanmean(n_min):
        e_n = -e_n
        _, n = _project_sn(xy, ctr, e_s, e_n)
        for k in range(nbins):
            mask = idx == k
            if not np.any(mask):
                continue
            n_k = n[mask]
            n_min[k], n_max[k] = np.nanmin(n_k), np.nanmax(n_k)

    def _fill_nan(x, y):
        m = ~np.isnan(y)
        if m.sum() < 2:
            return np.where(np.isnan(y), 0.0, y)
        return np.interp(x, x[m], y[m])

    n_min_f = _fill_nan(centers, n_min)
    n_max_f = _fill_nan(centers, n_max)

    w = _savgol_window(nbins)
    n_min_s = savgol_filter(n_min_f, window_length=w, polyorder=CAMBER_SMOOTH_POLY, mode="interp")
    n_max_s = savgol_filter(n_max_f, window_length=w, polyorder=CAMBER_SMOOTH_POLY, mode="interp")

    n_c = 0.5 * (n_max_s + n_min_s)
    frame = {"ctr": ctr, "e_s": e_s, "e_n": e_n, "s0": s0, "span": span}
    return s_norm, n, centers, n_c, frame


def _interp_n_c(s_norm_points, centers, n_c):
    return np.interp(np.clip(s_norm_points, 0.0, 1.0), centers, n_c)


def build_side_curves_and_classify(air_df, nbins=200):
    mask_surface = np.isclose(air_df["Points_2"].to_numpy(float), SURFACE_Z, atol=SURFACE_ATOL)
    surface = air_df.loc[mask_surface].copy()
    if surface.empty:
        raise RuntimeError(f"No surface points found (|z-{SURFACE_Z}| < {SURFACE_ATOL}).")

    s_norm_surf, n_surf, centers, n_c_centers, frame = _build_camber_from_surface(
        surface, nbins=nbins
    )

    n_c_surf = _interp_n_c(s_norm_surf, centers, n_c_centers)

    P = air_df[["Points_0", "Points_1"]].to_numpy(float)
    s_all, n_all = _project_sn(P, frame["ctr"], frame["e_s"], frame["e_n"])
    s_norm_all = (s_all - frame["s0"]) / frame["span"]
    n_c_all = _interp_n_c(s_norm_all, centers, n_c_centers)

    air_cls = air_df.copy()
    air_cls["zone"] = np.where(n_all >= n_c_all, "suctSide", "presSide")

    extra_surf = air_cls[(air_cls["zone"] == "suctSide") & mask_surface].copy()
    intra_surf = air_cls[(air_cls["zone"] == "presSide") & mask_surface].copy()
    extra_surf.sort_values("x_over_l", inplace=True, kind="mergesort")
    intra_surf.sort_values("x_over_l", inplace=True, kind="mergesort")

    return extra_surf, intra_surf, air_cls


def isentropic_mach_from_p(p, p0, gamma):
    """Return isentropic Mach from static pressure *p* and total pressure *p0*."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    ok = p > 0
    term = np.maximum(np.power(p0 / p[ok], (gamma - 1.0) / gamma) - 1.0, 0.0)
    out[ok] = np.sqrt(2.0 / (gamma - 1.0) * term)
    return out


def compute_massaverage_pressure_loss(file_mp1, file_mp2, gamma, R):
    """Compute total-pressure loss coefficient from MP1/MP2 slices."""
    df1 = pd.read_csv(file_mp1, usecols=["y", "p", "rho", "u", "v", "w", "T"])
    df2 = pd.read_csv(file_mp2, usecols=["y", "p", "rho", "u", "v", "w", "T"])

    U1 = np.sqrt(df1["u"] ** 2 + df1["v"] ** 2 + df1["w"] ** 2)
    U2 = np.sqrt(df2["u"] ** 2 + df2["v"] ** 2 + df2["w"] ** 2)  # noqa: F841

    p01_loc = stagnation_pressure(df1["p"], df1["T"], U1, gamma, R)
    p02_loc = stagnation_pressure(df2["p"], df2["T"], U2, gamma, R)

    w1 = df1["rho"] * df1["u"]
    p01 = np.sum(w1 * p01_loc) / np.sum(w1)
    p1  = np.sum(w1 * df1["p"]) / np.sum(w1)

    loss = (p01 - p02_loc) / (p01 - p1)
    return pd.DataFrame({"y": df2["y"], "loss": loss}).sort_values("y")


def extract_keys_from_file(path: str, file_list: list[str], keys: list[list[str]]) -> pd.DataFrame:
    qty_list: list[list[float]] = []
    header_list: list[str] = []
    for fid, file in enumerate(file_list):
        if file.endswith(".csv"):
            print(f"found {file} in {path}")
            return pd.read_csv(os.path.join(path, file))
        file_lines = list(filter(None, open(os.path.join(path, file)).read().splitlines()))
        headers = file_lines[0][2:].split()
        print(f"headers: {headers}")
        for key in keys[fid]:
            try:
                idx = headers.index(key) if key != "y+" else 10
                qty_list.append([float(line.split()[idx]) for line in file_lines[1:]])
                header_list.append(key)
            except Exception as e:
                print(f"ERROR -- could not read {key} in {headers}\nexception: {e}")
    return pd.DataFrame({header_list[i]: pd.Series(qty_list[i]) for i in range(len(qty_list))})


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        prog="read_mach_numb",
        description=(
            "Plot isentropic Mach and total-pressure loss for OpenFOAM RANS cases vs LES.\n\n"
            "JSON cases format:\n"
            '  {"tag": {"case": "case.foam", "fName": "Output",\n'
            '           "color": "tab:green", "label": "sim_1"}}'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("config", help="Path to JSON file containing the cases dict.")

    ap.add_argument(
        "--c-ref", type=float, default=0.07, metavar="C",
        help="Reference chord for y/c axis in the loss plot (default: 0.07)",
    )
    ap.add_argument(
        "--surface-z", type=float, default=0.01, metavar="Z",
        help="Z coordinate of the 2-D surface slice used for Mis extraction (default: 0.01)",
    )
    ap.add_argument(
        "--gamma", type=float, default=1.4,
        help="Ratio of specific heats (default: 1.4)",
    )
    ap.add_argument(
        "--R", type=float, default=287.05,
        help="Specific gas constant [J/kg/K] (default: 287.05)",
    )
    ap.add_argument(
        "--output", default="results.png", metavar="FILE",
        help="Output figure filename (default: results.png)",
    )
    args = ap.parse_args()

    # Override module-level globals so all helper functions see the correct values.
    global c_ref, SURFACE_Z, gamma, R
    c_ref     = args.c_ref
    SURFACE_Z = args.surface_z
    gamma     = args.gamma
    R         = args.R

    plt.close("all")

    # ── Plotting ──────────────────────────────────────────────────────────────
    fig, (ax_mis, ax_loss) = plt.subplots(1, 2, figsize=(16, 5.5))

    with open(args.config) as fh:
        cases = json.load(fh)

    plt.rcParams.update({
        "font.size": 20, "font.family": "serif", "mathtext.fontset": "cm",
        "axes.grid": True, "grid.color": "0.9", "grid.linestyle": "-", "grid.linewidth": 0.9,
    })

    MP1_compare = ""

    for c in cases:
        if cases[c].get("case"):
            _script = _PKG_DIR / "extract_data_all.py"
            os.system(f'pvpython {_script} {cases[c]["case"]} {cases[c]["fName"]}')

        df_profile  = pd.read_csv(cases[c]["fName"] + "_Mis.csv")
        df_upstream = pd.read_csv(cases[c]["fName"] + "_MP1.csv")

        if cases["LES"]:
            df_LES_MP1 = pd.read_csv(f"{cases["LES"]["fName"]}_MP1.csv")
            MP1_compare += (f'Case {c}: M_in RANS = {np.mean(df_upstream["M"]):.6g}; M_in LES = {np.mean(df_LES_MP1["M"]):.6g}\n')
            MP1_compare += (f'Case {c}: T_in RANS = {np.mean(df_upstream["T"]):.6g}; T_in LES = {np.mean(df_LES_MP1["T"]):.6g}\n')
            MP1_compare += (f'Case {c}: p_in RANS = {np.mean(df_upstream["p"]):.6g}; p_in LES = {np.mean(df_LES_MP1["p"]):.6g}\n')
            MP1_compare += (f'Case {c}: rho_in RANS = {np.mean(df_upstream["rho"]):.6g}; rho_in LES = {np.mean(df_LES_MP1["rho"]):.6g}\n')

        if cases[c]['solver']=="OpenFOAM":
            air = df_profile[
                df_profile["Block Name"].astype(str).str.strip().str.lower() == "airfoil"
            ].copy()
            if air.empty:
                print(f"[INFO] {c}: no 'airfoil' block in Mis CSV")

            air["x_over_l"] = air["Points_0"] / np.max(air["Points_0"].to_numpy(float))
            P0 = np.mean(
                df_upstream["p"].to_numpy()
                * (1 + (gamma - 1) / 2 * df_upstream["M"].to_numpy() ** 2) ** (gamma / (gamma - 1))
            )
            air["Mis_from_p"] = isentropic_mach_from_p(air["p"].to_numpy(), P0, gamma)

            try:
                suctSide, presSide, air = build_side_curves_and_classify(air)
            except Exception as e:
                print(f"[WARN] {c}: suction/pressure side classification failed ({e}) -> fallback")
                air["zone"] = "suctSide"
                mask_surface = np.isclose(
                    air["Points_2"].to_numpy(float), SURFACE_Z, atol=SURFACE_ATOL
                )
                suctSide = air[mask_surface].sort_values("x_over_l")
                presSide = air.head(0)

            cases[c]["suctSide"] = suctSide
            cases[c]["presSide"] = presSide
            
            ax_mis.plot(
                cases[c]["presSide"]["x_over_l"], cases[c]["presSide"]["Mis_from_p"],
                linestyle="-.", linewidth=3,
                color=cases[c]["color"], label=cases[c]["label"],
            )
            ax_mis.plot(
                cases[c]["suctSide"]["x_over_l"], cases[c]["suctSide"]["Mis_from_p"],
                linestyle="-.", linewidth=3, color=cases[c]["color"],
            )
        else:

            ax_mis.plot(
                df_profile["xw"] / np.max(df_profile["xw"]),
                df_profile["M_is"],
                color=cases[c]["color"], label="LES", linewidth=3,
            )

    ax_mis.set_xlabel(r"$x/c~[-]$")
    ax_mis.set_ylabel(r"$M_{is}~[-]$")
    ax_mis.set_xlim(left=0.0)
    ax_mis.grid(True, alpha=1)
    ax_mis.legend(ncols=2)

    for c in cases:
        mp1 = mp1_pattern.format(case=cases[c]["fName"])
        mp2 = mp2_pattern.format(case=cases[c]["fName"])
        if cases[c]["solver"] == 'MUSICAA':
            div = 1000.0
        else:
            div = 1.0
        df = compute_massaverage_pressure_loss(mp1, mp2, gamma, R)
        ax_loss.plot(df["y"] / c_ref/div, df["loss"],
                     label=cases[c]["label"], color=cases[c]["color"], linewidth=3)

    ax_loss.set_xlabel(r"$y/c~[-]$")
    ax_loss.set_ylabel(r"$\omega~[-]$")
    fig.tight_layout()
    plt.savefig(args.output)
    print(MP1_compare)
    with open("report.txt", "w") as out:
        out.write(MP1_compare)
    plt.show()


if __name__ == "__main__":
    main()
