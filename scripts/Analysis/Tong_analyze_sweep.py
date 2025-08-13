#!/usr/bin/env python3
"""
Facet Id–Vgs plots by a chosen sweep variable.
Each subplot shows *all devices' curves* that share the same value of that variable
(while the other parameters are held constant). Also detects NMOS/PMOS from folder names.

Folder example:
  gax_mod_fet_tlm_nmos_lc_0.16_lch_0.12_lov_0.06_gateasym_0.00_w_4.0_21_2025_08_11_12_25_58

Usage example:
  python plot_idvgs_sweep_facet.py \
      --root ./scripts/die1-1_full_sweep \
      --type nmos \
      --sweep lc --lch 0.12 --lov 0.06 --gateasym 0.00 --w 4.0 \
      --target_vds 0.8 --dir_indices 2,3 --out ./facet_lc.png
"""

import os
import re
import csv
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt

# ---------- optional: use controller.util.io if available ----------
try:
    from controller.util.io import import_hdf5
except ImportError:
    import h5py
    def import_hdf5(path):
        out = {}
        with h5py.File(path, "r") as f:
            def pull(g):
                obj = {}
                for k, v in g.items():
                    if isinstance(v, h5py.Dataset):
                        obj[k] = v[()]
                    else:
                        obj[k] = pull(v)
                return obj
            for k, v in f.items():
                if isinstance(v, h5py.Dataset):
                    out[k] = v[()]
                else:
                    out[k] = pull(v)
        return out

PARAMS = ["lc", "lch", "lov", "gateasym", "w"]

# Robust float: -0.12, 0.12, 4, 4.0
F = r"(-?\d+(?:\.\d+)?)"

# Extract type (nmos|pmos) and dimensions from dirname.
# Example: gax_mod_fet_tlm_nmos_lc_0.16_lch_0.12_lov_0.06_gateasym_0.00_w_4.0_21_2025_08_11_12_25_58
NAME_RE = re.compile(
    rf"^gax_.*?_(nmos|pmos).*?_lc_{F}_lch_{F}_lov_{F}_gateasym_{F}_w_{F}_(\d+)_\d{{4}}_\d{{2}}_\d{{2}}_\d{{2}}_\d{{2}}_\d{{2}}$",
    re.IGNORECASE
)

def scan_device_dirs(root):
    """
    Return list of records:
    {'path','dirname','type','lc','lch','lov','gateasym','w','device_id'}
    """
    recs = []
    for dirpath, dirnames, _ in os.walk(root):
        for name in dirnames:
            m = NAME_RE.match(name)
            if not m:
                continue
            dev_type, lc, lch, lov, gateasym, w, device_id = m.groups()
            recs.append({
                "path": os.path.join(dirpath, name),
                "dirname": name,
                "type": dev_type.lower(),
                "lc": float(lc),
                "lch": float(lch),
                "lov": float(lov),
                "gateasym": float(gateasym),
                "w": float(w),
                "device_id": int(device_id),
            })
    return recs

def write_csv(records, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if not records:
        return
    keys = ["dirname","path","type","lc","lch","lov","gateasym","w","device_id"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in records:
            w.writerow({k: r[k] for k in keys})

def filter_records(records, dev_type, sweep, constants):
    """
    Filter by device type and by constants (except the sweep param).
    dev_type in {'nmos','pmos','both'}.
    constants: dict param->value (floats).
    """
    tol = 1e-9
    out = []
    for r in records:
        if dev_type != "both" and r["type"] != dev_type:
            continue
        ok = True
        for k, v in constants.items():
            if k == sweep:
                continue
            if v is None:
                ok = False
                break
            if abs(r[k] - float(v)) > tol:
                ok = False
                break
        if ok:
            out.append(r)
    return out

def parse_dir_indices(s):
    # "2" or "2,3"
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return [int(p) for p in parts]

def find_bias_index_for_vds(vds, dir_index, target_vds):
    """
    vds shape: (bias, dir, points)
    Return bias index with start-point vds closest to target_vds for the given dir.
    """
    nbias = vds.shape[0]
    vals = np.array([vds[b, dir_index, 0] for b in range(nbias)])
    return int(np.nanargmin(np.abs(vals - target_vds)))

def load_idvgs_curve(h5_path, dir_indices, target_vds, use_abs=False, average_dirs=False):
    """
    Load Id–Vgs for the specified dir_indices at the bias closest to target_vds.
    If multiple dir_indices are given:
      - average_dirs=False: return first dir that loads successfully
      - average_dirs=True: average Id across dirs (requires same Vgs vector)
    Returns (vgs, id_array) on success, else (None, None).
    """
    if not os.path.exists(h5_path):
        return None, None
    try:
        d = import_hdf5(h5_path)
        i_d = np.array(d["i_d"])
        v_gs = np.array(d["v_gs"])
        v_ds = np.array(d["v_ds"])
    except Exception:
        return None, None

    if i_d.ndim != 3:
        return None, None

    got = []
    for d_idx in dir_indices:
        if d_idx < 0 or d_idx >= i_d.shape[1]:
            continue
        b = find_bias_index_for_vds(v_ds, d_idx, target_vds)
        vgs = v_gs[b, d_idx, :]
        idc = i_d[b, d_idx, :]
        if use_abs:
            idc = np.abs(idc)
        got.append((vgs, idc))

    if not got:
        return None, None

    if average_dirs and len(got) > 1:
        ref = got[0][0]
        if all(np.array_equal(ref, g[0]) for g in got[1:]):
            ids = np.stack([g[1] for g in got], axis=0)
            return ref, np.nanmean(ids, axis=0)
        else:
            return got[0]
    else:
        return got[0]

def facet_plot_idvgs(records, sweep, program, target_vds, dir_indices,
                     use_abs=False, logy=False, average_dirs=False,
                     max_cols=4, out_png="facet.png"):
    """
    Build a subplot grid with one subplot per unique sweep value.
    On each subplot, plot *all devices'* curves for that value.

    Assumes FRFR sweep order:
      dir 0 = forward-1, dir 1 = reverse-1, dir 2 = forward-2, dir 3 = reverse-2
    """
    import matplotlib.lines as mlines

    # Pick your colors/styles per dir
    DIR_COLOR = {0:"cyan", 1: "red", 2: "navy", 3: "orange"}   # second forward, second reverse
    DIR_LABEL = {        
        0: "Fwd-1 (dir 0)",
        1: "Rev-1 (dir 1)",
        2: "Fwd-2 (dir 2)",
        3: "Rev-2 (dir 3)",
    }
    DIR_STYLE = {2: "-", 3: "--"}          # optional: different linestyles

    # Group by sweep value
    by_val = {}
    for r in records:
        val = r[sweep]
        by_val.setdefault(val, []).append(r)

    vals_sorted = sorted(by_val.keys())
    n = len(vals_sorted)
    if n == 0:
        raise SystemExit("No sweep values found after filtering.")

    ncols = min(max_cols, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(4.2*ncols, 3.3*nrows), squeeze=False)

    for idx, val in enumerate(vals_sorted):
        ax = axes[idx // ncols][idx % ncols]
        group = by_val[val]
        plotted = 0

        # plot both directions separately so colors differ
        for rec in group:
            h5_path = os.path.join(rec["path"], f"{program}.h5")

            # loop each requested dir (e.g., [2,3])
            for d_idx in dir_indices:
                vgs, idc = load_idvgs_curve(
                    h5_path,
                    dir_indices=[d_idx],          # IMPORTANT: single dir at a time
                    target_vds=target_vds,
                    use_abs=use_abs,
                    average_dirs=False,           # don't average; we want distinct dirs
                )
                if vgs is None:
                    continue

                color = DIR_COLOR.get(d_idx, None)
                style = DIR_STYLE.get(d_idx, "-")
                ax.plot(vgs, idc, alpha=0.8, color=color, linestyle=style, linewidth=1.0)
                plotted += 1

        ax.set_title(f"{sweep}={val:g}  (n={plotted})", fontsize=10)
        ax.set_xlabel("V_GS [V]")
        ax.set_ylim(bottom=1e-9, top=1e-3)  # Ensure y-axis starts at 0
        ax.set_ylabel("|I_D| [A]" if use_abs else "I_D [A]")
        if logy:
            ax.set_yscale("log")
        ax.grid(True, which="both", ls=":")
        # Tight autoscale on Y
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)

    # Hide any unused axes
    for j in range(n, nrows*ncols):
        axes[j // ncols][j % ncols].axis("off")

    # Figure-level legend (one legend for all facets)
    legend_handles = []
    for d_idx in dir_indices:
        legend_handles.append(
            mlines.Line2D([], [], color=DIR_COLOR.get(d_idx, "C0"),
                          linestyle=DIR_STYLE.get(d_idx, "-"),
                          label=DIR_LABEL.get(d_idx, f"dir {d_idx}"))
        )
    if legend_handles:
        fig.legend(handles=legend_handles, loc="lower center", ncols=len(legend_handles), frameon=False)

    fig.suptitle(f"Id–Vgs at Vds≈{target_vds:g} V • sweep={sweep}", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[ok] wrote facet plot -> {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./scripts/die1-1_full_sweep/tlm", help="Root folder to scan (recursively)")
    ap.add_argument("--sweep", default = "lch", choices=PARAMS, help="Which variable to sweep")
    ap.add_argument("--type", choices=["nmos","pmos","both"], default="both", help="Device type filter (from folder name)")
    # constants for the remaining parameters (optional; if omitted, we print options and exit)
    ap.add_argument("--lc", default = "0.4", type=float)
    ap.add_argument("--lch", default = "0.16", type=float)
    ap.add_argument("--lov", default = "0.06", type=float)
    ap.add_argument("--gateasym", default = "0.00", type=float)
    ap.add_argument("--w", default = "4.0", type=float)

    # extraction/plot options
    ap.add_argument("--program", default="keysight_id_vgs", help="HDF5 file name w/o .h5")
    ap.add_argument("--target_vds", type=float, default=0.8, help="Target Vds (closest bias picked)")
    ap.add_argument("--dir_indices", default="0,1,2,3", help="Comma list of dir indices (e.g., '2' or '2,3')")
    ap.add_argument("--avg_dirs", action="store_true", help="Average dir curves if Vgs align")
    ap.add_argument("--abs_current", action="store_true", help="Plot |Id|")
    ap.add_argument("--logy", action="store_true", help="Log y-scale")
    ap.add_argument("--max_cols", type=int, default=4, help="Max subplot columns")
    ap.add_argument("--out", default="./scripts/die1-1_full_sweep/_config_summary/facet_idvgs.png", help="Output PNG path")
    ap.add_argument("--csv_out", default="./scripts/die1-1_full_sweep/_config_summary/all_device_configs.csv", help="Output CSV path")
    args = ap.parse_args()

    # scan and write CSV (including device type)
    recs = scan_device_dirs(args.root)
    if not recs:
        raise SystemExit("No matching device directories found.")
    write_csv(recs, args.csv_out)
    print(f"[csv] wrote -> {args.csv_out}  ({len(recs)} rows)")

    # Build constants dict (must supply all non-swept params)
    constants = {p: getattr(args, p) for p in PARAMS if p != args.sweep}
    missing = [p for p, v in constants.items() if v is None]
    if missing:
        raise SystemExit(f"Please provide constants for: {missing}. Example: --{missing[0]} <value>")

    dir_indices = parse_dir_indices(args.dir_indices)

    # filter by type and constants
    filtered = filter_records(recs, args.type, args.sweep, constants)
    if not filtered:
        raise SystemExit("No devices matched type/constants.")

    # -------- NEW: Build output PNG file name --------
    const_str = "_".join(f"{k}_{constants[k]:g}" for k in sorted(constants))
    out_dir = os.path.dirname(args.out) if args.out else "."
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"{args.sweep}_sweep_{const_str}.png")
    # -------------------------------------------------

    # facet plot (one subplot per unique value of sweep variable; each subplot plots ALL curves for that value)
    facet_plot_idvgs(
        filtered,
        sweep=args.sweep,
        program=args.program,
        target_vds=args.target_vds,
        dir_indices=dir_indices,
        use_abs=args.abs_current,
        logy=args.logy,
        average_dirs=args.avg_dirs,
        max_cols=args.max_cols,
        out_png=out_png,
    )


if __name__ == "__main__":
    main()




