#!/usr/bin/env python3
import os, re, csv, math, argparse
import numpy as np
import matplotlib.pyplot as plt

# ---------- HDF5 loader ----------
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
                    obj[k] = v[()] if isinstance(v, h5py.Dataset) else pull(v)
                return obj
            for k, v in f.items():
                out[k] = v[()] if isinstance(v, h5py.Dataset) else pull(v)
        return out

PARAMS = ["lc", "lch", "lov", "gateasym", "w", "nf"]

# ---------------- parsing ----------------
F = r"(-?\d+(?:\.\d+)?)"
I = r"(\d+)"
TAIL = rf"_{I}(?:_.+)?$"     # device_id + optional _anything
RE_NORMAL = re.compile(
    rf"^gax_.*?_(nmos|pmos).*?_lc_{F}_lch_{F}_lov_{F}_gateasym_{F}_w_{F}_{I}_\d{{4}}_\d{{2}}_\d{{2}}_\d{{2}}_\d{{2}}_\d{{2}}$",
    re.IGNORECASE
)
RE_WSWEEP = re.compile(
    rf"^gax_.*?wsweep.*?_(nmos|pmos).*?_lc_{F}_lch_{F}_lov_{F}_w_{F}_{I}_\d{{4}}_\d{{2}}_\d{{2}}_\d{{2}}_\d{{2}}_\d{{2}}$",
    re.IGNORECASE
)
RE_OV = re.compile(
    rf"^gax_.*?ov.*?_(nmos|pmos).*?_lc_{F}_lch_{F}_lov_{F}_gateasym_{F}_w_{F}_{I}_\d{{4}}_\d{{2}}_\d{{2}}_\d{{2}}_\d{{2}}_\d{{2}}$",
    re.IGNORECASE
)
RE_LONG = re.compile(
    rf"^gax_.*?long.*?_(nmos|pmos).*?_lc_{F}_lch_{F}_lov_{F}_w_{F}_{I}_\d{{4}}_\d{{2}}_\d{{2}}_\d{{2}}_\d{{2}}_\d{{2}}$",
    re.IGNORECASE
)
RE_FINGERED = re.compile(
    rf"^gax_.*?fingered(_fet_tlm)?_.*?_(nmos|pmos).*?_nf_{I}_lc_{F}_lch_{F}_lov_{F}_gateasym_{F}_w_{F}{TAIL}",
    re.IGNORECASE
)

def parse_device_dirname(dirname):
    m = RE_FINGERED.match(dirname)
    if m:
        _fet_tlm, dev_type, nf, lc, lch, lov, gateasym, w, dev_id = m.groups()
        return dict(type=dev_type.lower(), nf=int(nf), lc=float(lc), lch=float(lch),
                    lov=float(lov), gateasym=float(gateasym), w=float(w),
                    device_id=int(dev_id), dirname=dirname)
    m = RE_OV.match(dirname)
    if m:
        dev_type, lc, lch, lov, gateasym, w, dev_id = m.groups()
        return dict(type=dev_type.lower(), nf=1, lc=float(lc), lch=float(lch),
                    lov=float(lov), gateasym=float(gateasym), w=float(w),
                    device_id=int(dev_id), dirname=dirname)
    m = RE_LONG.match(dirname)
    if m:
        dev_type, lc, lch, lov, w, dev_id = m.groups()
        return dict(type=dev_type.lower(), nf=1, lc=float(lc), lch=float(lch),
                    lov=float(lov), gateasym=0.0, w=float(w),
                    device_id=int(dev_id), dirname=dirname)
    m = RE_WSWEEP.match(dirname)
    if m:
        dev_type, lc, lch, lov, w, dev_id = m.groups()
        return dict(type=dev_type.lower(), nf=1, lc=float(lc), lch=float(lch),
                    lov=float(lov), gateasym=0.0, w=float(w),
                    device_id=int(dev_id), dirname=dirname)
    m = RE_NORMAL.match(dirname)
    if m:
        dev_type, lc, lch, lov, gateasym, w, dev_id = m.groups()
        return dict(type=dev_type.lower(), nf=1, lc=float(lc), lch=float(lch),
                    lov=float(lov), gateasym=float(gateasym), w=float(w),
                    device_id=int(dev_id), dirname=dirname)
    return None

def scan_device_dirs(root):
    recs = []
    for dirpath, dirnames, _ in os.walk(root):
        for name in dirnames:
            meta = parse_device_dirname(name)
            if meta:
                meta["path"] = os.path.join(dirpath, name)
                recs.append(meta)
    return recs

def write_csv(records, out_csv):
    if not out_csv:
        return
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    keys = ["dirname","path","type","lc","lch","lov","gateasym","w","nf","device_id"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k) for k in keys})

def parse_dir_indices(arg: str):
    a = arg.strip().lower()
    if a == "all":
        return [0,1,2,3]
    return [int(s) for s in a.split(",") if s.strip()!=""]

def parse_vds_list(arg: str):
    if not arg:
        return [0.05, 0.8]
    vals = []
    for s in arg.split(","):
        s = s.strip()
        if s:
            vals.append(float(s))
    return vals or [0.05, 0.8]

def filter_constants(records, dev_type, consts, tol=1e-9):
    """
    Keep rows that match dev_type and all constants in `consts`.
    """
    out = []
    for r in records:
        if dev_type != "both" and r["type"] != dev_type:
            continue
        ok = True
        for k, v in consts.items():
            if v is None:
                continue
            if k not in r:
                continue
            if abs(r[k] - float(v)) > tol:
                ok = False
                break
        if ok:
            out.append(r)
    return out

# ---------------- data ----------------
def find_bias_index_for_vds(vds, dir_index, target_vds):
    nbias = vds.shape[0]
    vals = np.array([vds[b, dir_index, 0] for b in range(nbias)])
    return int(np.nanargmin(np.abs(vals - target_vds)))

def load_idvgs_curve(h5_path, dir_index, target_vds, use_abs=False):
    if not os.path.exists(h5_path):
        return None, None
    try:
        d = import_hdf5(h5_path)
        i_d, v_gs, v_ds = np.array(d["i_d"]), np.array(d["v_gs"]), np.array(d["v_ds"])
    except Exception:
        return None, None
    if i_d.ndim != 3 or dir_index < 0 or dir_index >= i_d.shape[1]:
        return None, None
    b = find_bias_index_for_vds(v_ds, dir_index, target_vds)
    vgs = v_gs[b, dir_index, :]
    idc = np.abs(i_d[b, dir_index, :]) if use_abs else i_d[b, dir_index, :]
    return vgs, idc

# ---------------- matrix plotting ----------------
def unique_sorted(values):
    return sorted(set(float(v) for v in values))

def get_unique_axes_values(records, dev_type="both"):
    lcs, lchs = [], []
    for r in records:
        if dev_type != "both" and r["type"] != dev_type:
            continue
        if "lc" in r and "lch" in r:
            lcs.append(r["lc"]); lchs.append(r["lch"])
    return unique_sorted(lcs), unique_sorted(lchs)

def build_constants(args):
    # Constants that must match for all curves inside a cell
    return {
        "lov": args.lov,
        "gateasym": args.gateasym,
        "w": args.w,
        "nf": args.nf
    }

def simple_matrix_plot(recs, args, dir_indices, vds_list):
    # Determine grid axes
    lc_vals, lch_vals = get_unique_axes_values(recs, dev_type=args.type)
    if not lc_vals or not lch_vals:
        raise SystemExit("No lc/lch values found under --root with current filters.")

    ncols, nrows = len(lc_vals), len(lch_vals)

    # Figure size heuristic: ~3.2x2.6 inches per subplot
    fig_w, fig_h = max(6, 3.2*ncols), max(4, 2.6*nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), squeeze=False)

    # Colors for the two Vds curves
    # Ensure two distinct colors regardless of style cycle
    vds_colors = {}
    default_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3'])
    for i, v in enumerate(vds_list):
        vds_colors[v] = default_colors[i % len(default_colors)]

    # Build constants shared across cells
    base_consts = build_constants(args)

    any_plotted = False

    for row, lch in enumerate(lch_vals):      # y-axis increases downward by row index
        ax_left = axes[row, 0]
        ax_left.text(
            -0.26, 0.5, f"Lch={lch:g}",
            transform=ax_left.transAxes, ha="right", va="center",
            rotation=90, fontweight="bold", clip_on=False, fontsize=20
        )

        for col, lc in enumerate(lc_vals):    # x-axis increases with column
            axes[0, col].text(0.5, 1.02, f"Lc={lc:g}", transform=axes[0, col].transAxes,
                      ha="center", va="bottom", fontweight="bold", fontsize=20)
            ax = axes[row, col]
            # Filter records for this cell
            consts = dict(base_consts)
            consts["lc"] = lc
            consts["lch"] = lch
            cell_recs = filter_constants(recs, args.type, consts)

            # Plot: for each record, each dir, both Vds (in different COLORS)
            for rec in sorted(cell_recs, key=lambda r: (r.get("device_id",0), r.get("nf",0), r.get("w",0))):
                h5 = os.path.join(rec["path"], f"{args.program}.h5")
                for d_idx in dir_indices:
                    for vi, vds in enumerate(vds_list):
                        vgs, idc = load_idvgs_curve(h5, d_idx, vds, args.abs_current)
                        if vgs is None:
                            continue
                        # Normalize to μA/μm for W=4 μm
                        y = idc / 4e-6
                        ax.plot(vgs, y, linewidth=1.2, alpha=0.9, color=vds_colors[vds])
                        ax.set_ylim(bottom=1e-6,top=200)
                        any_plotted = True

            # Styling: simple, no title, minimal ticks
            if args.logy:
                ax.set_yscale("log")

            # Only outer labels to keep clean
            if row == nrows - 1:
                ax.set_xlabel("V_GS [V]")
            if col == 0:
                ax.set_ylabel("I_D [μA/μm]")

            ax.grid(True, which="both", linestyle=":", alpha=0.5)


    # Global legend for the two Vds colors (shown once)
    legend_handles = []
    for v in vds_list:
        # create dummy lines for legend
        line, = axes[0,0].plot([], [], color=vds_colors[v], label=f"V_DS = {v:g} V")
        legend_handles.append(line)
    fig.legend(handles=legend_handles, loc="upper center", ncol=len(vds_list), frameon=False, fontsize=20)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # room for legend
    out_path = args.out
    if os.path.isdir(os.path.dirname(out_path)) and (os.path.splitext(out_path)[1]=="" or out_path.endswith("/")):
        os.makedirs(out_path, exist_ok=True)
        out_path = os.path.join(out_path, "matrix_lc_lch.png")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    if not any_plotted:
        raise SystemExit("Found devices but failed to load any curves.")
    print(f"[ok] wrote -> {out_path}")

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Matrix of Id–Vgs across (lc, lch).")
    ap.add_argument("--root", default="./scripts/Analysis/Al/tlm", help="Root folder to scan")
    # ap.add_argument("--root", default="./scripts/Analysis/W/die_x_0_y_0/tlm2", help="Root folder to scan")

    ap.add_argument("--type", choices=["nmos","pmos","both"], default="both")

    # constants (used to filter within each cell; lc/lch come from grid)
    ap.add_argument("--lov", type=float, default=0.06)
    ap.add_argument("--gateasym", type=float, default=0.00)
    ap.add_argument("--w", type=float, default=4.0)
    ap.add_argument("--nf", type=float, default=None)

    ap.add_argument("--program", default="keysight_id_vgs")
    ap.add_argument("--vds_list", default="0.05,0.8", help="Comma list of two Vds values, e.g. '0.05,0.8'")
    ap.add_argument("--dir_indices", default="2", help="'all' or comma list like '2,3'")
    ap.add_argument("--abs_current", action="store_true", default=True)
    ap.add_argument("--logy", action="store_true", default=True)
    ap.add_argument("--csv_out", default="")
    ap.add_argument("--out", default="./scripts/Analysis/W/_config_summary", help="Output file OR folder")

    args = ap.parse_args()
    dir_indices = parse_dir_indices(args.dir_indices)
    vds_list = parse_vds_list(args.vds_list)

    recs = scan_device_dirs(args.root)
    if args.csv_out:
        write_csv(recs, args.csv_out)
        print(f"[csv] wrote -> {args.csv_out} ({len(recs)} rows)")

    simple_matrix_plot(recs, args, dir_indices, vds_list)

if __name__ == "__main__":
    main()

