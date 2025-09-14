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
    # Most specific FIRST to avoid wrong matches
    m = RE_FINGERED.match(dirname)
    if m:
        # groups: optional _fet_tlm, type, nf, lc, lch, lov, gateasym, w, device_id
        _fet_tlm, dev_type, nf, lc, lch, lov, gateasym, w, dev_id = m.groups()
        return dict(
            type=dev_type.lower(), nf=int(nf),
            lc=float(lc), lch=float(lch), lov=float(lov),
            gateasym=float(gateasym), w=float(w),
            device_id=int(dev_id), dirname=dirname
        )

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

def filter_constants(records, dev_type, consts, sweep_key=None, tol=1e-9):
    """
    Keep rows that match dev_type and all constants in `consts`,
    except the field named by `sweep_key` (ignored so it can vary).
    """
    out = []
    for r in records:
        if dev_type != "both" and r["type"] != dev_type:
            continue
        ok = True
        for k, v in consts.items():
            if k == sweep_key or v is None:
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

# ---------------- plotting ----------------
DIR_STYLE = {0:"-.", 1:"--", 2:"-", 3:":"}

def plot_group_overlay(grouped, program, target_vds, dir_indices, use_abs, logy,
                       title, group_key, color_map='viridis_r', out_png="plot.png",
                       legend_cols=4):
    """
    grouped: dict { group_value -> [records] }, all curves in the same group share one color.
    Color darkens as group value increases (sequential colormap).
    Returns (devices_plotted, curves_plotted).
    """
    # sort group keys numerically if possible
    keys_sorted = sorted(grouped.keys(), key=lambda k: float(k))
    if not keys_sorted:
        raise SystemExit("Nothing to plot: empty groups.")

    # cmap = getattr(plt.cm, color_map)
    # kmin, kmax = min(keys_sorted), max(keys_sorted)
    # if kmax > kmin:
    #     key_to_color = {k: cmap((k - kmin) / (kmax - kmin)+0.3) for k in keys_sorted}
    # else:
    #     key_to_color = {k: cmap(0.5) for k in keys_sorted}
    # Use a high-contrast palette from matplotlib
    base_colors = plt.cm.tab10.colors  # strong, saturated colors
    # If you have more curves than tab10 colors, cycle through with varied line styles
    from itertools import cycle
    color_cycle = cycle(base_colors)

    key_to_color = {k: next(color_cycle) for k in keys_sorted}

    plt.figure(figsize=(10, 6))
    devices_plotted = 0
    curves_plotted = 0
    any_plotted = False

    for k in keys_sorted:
        color = key_to_color[k]
        labeled_this_group = False

        for rec in sorted(grouped[k], key=lambda r: r.get("device_id", 0)):
            h5 = os.path.join(rec["path"], f"{program}.h5")
            rec_plotted = False

            for d_idx in dir_indices:
                vgs, idc = load_idvgs_curve(h5, d_idx, target_vds, use_abs)
                if vgs is None:
                    continue

                plt.plot(
                    vgs, idc,
                    color=color,
                    linestyle=DIR_STYLE.get(d_idx, "-"),
                    linewidth=1.3,        # slightly thicker for visibility
                    alpha=0.8,            # no transparency for strong color
                    label=(f"{group_key}={k:g}" if not labeled_this_group else None)
                )
                labeled_this_group = True
                rec_plotted = True
                curves_plotted += 1
                any_plotted = True

            if rec_plotted:
                devices_plotted += 1

    
    if not any_plotted:
        raise SystemExit("Found devices but failed to load any curves.")

    # Bigger font for axis labels
    plt.xlabel("V_GS [V]", fontsize=16)
    plt.ylabel("|I_D| [A]" if use_abs else "I_D [A]", fontsize=16)

    if logy:
        plt.yscale("log")
    
    # Only show device count as title
    plt.title(f"Devices Plotted per Category= {devices_plotted/len(grouped):.1f}", fontsize=18)

    plt.grid(True, which="both", ls=":")

    # Increase tick label font size
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    ax = plt.gca()
    ax.relim()
    ax.autoscale_view(scalex=False, scaley=True)
    plt.ylim(bottom=2e-11, top=1e-2)  # Set y-limits for better visibility

    # Bigger legend font
    plt.legend(ncols=legend_cols, fontsize=15)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"[ok] wrote -> {out_png}")
    return devices_plotted, curves_plotted


# ------------- tlm facet -------------
def mode_tlm(recs, args, dir_indices):
    """
    Overlay-style TLM:
      - Group by args.sweep (e.g., 'lch' or 'lc')
      - Different colors per sweep value
      - All devices on one axes
      - Figure title includes plotted counts
    """
    # Build constants, but allow the sweep variable to vary
    constants = {p: getattr(args, p) for p in PARAMS if p != args.sweep}
    filtered = filter_constants(recs, args.type, constants, sweep_key=args.sweep)
    if not filtered:
        print_available_values(recs)
        raise SystemExit("No devices matched type/constants for tlm mode.")

    # Group records by the sweep key
    by_val = {}
    for r in filtered:
        by_val.setdefault(r[args.sweep], []).append(r)

    print(f"[debug] matched folders (total) = {len(filtered)}")
    from collections import Counter
    print("[debug] by type:", Counter(r["type"] for r in filtered))
    print("[debug] by sweep key:", Counter(r[args.sweep] for r in filtered) if hasattr(args, "sweep") else "n/a")
    # list a few paths to verify scope
    for r in sorted(filtered, key=lambda z: z["path"])[:10]:
        print("  ", r["path"])

    # Output naming: if --out is a directory, construct a descriptive filename
    out_png = args.out
    if os.path.isdir(os.path.dirname(out_png)) and (os.path.splitext(out_png)[1]=="" or out_png.endswith("/")):
        bits = [f"type_{args.type}", f"{args.sweep}_sweep"]
        for k in PARAMS:
            if k != args.sweep and getattr(args, k) is not None:
                bits.append(f"{k}_{getattr(args,k):g}")
        bits += [f"vds_{args.target_vds:g}", f"dirs_{'-'.join(map(str,dir_indices))}"]
        os.makedirs(args.out, exist_ok=True)
        out_png = os.path.join(args.out, "_".join(bits) + ".png")

    # Build a descriptive title line summarizing the constants
    const_bits = []
    for k in PARAMS:
        if k != args.sweep and getattr(args, k) is not None:
            const_bits.append(f"{k}={getattr(args,k):g}")
    # title_label = f"type={args.type}" + (", " + ", ".join(const_bits) if const_bits else "")
    # title = f"Id–Vgs • sweep={args.sweep} @ Vds≈{args.target_vds:g} V • dirs {','.join(map(str, dir_indices))}\n{title_label}"
    title_label =  ( ", ".join(const_bits) if const_bits else "")
    DIR_LABEL = {0:"Forward 1st", 1:"Reverse 1st", 2:"Forward 2nd", 3:"Reverse 2nd"}
    dirs_str = ', '.join(DIR_LABEL.get(i, str(i)) for i in dir_indices)
    title = (
        f"Id–Vgs • sweep={args.sweep} @ Vds={args.target_vds:g} V • dirs {dirs_str}\n"
        f"{title_label}"
    )
    # Use the same overlay engine as other modes
    plot_group_overlay(
        grouped=by_val,
        program=args.program,
        target_vds=args.target_vds,
        dir_indices=dir_indices,
        use_abs=args.abs_current,
        logy=args.logy,
        title=title,
        group_key=args.sweep,
        out_png=out_png,
        legend_cols=4
    )


# ------------- generic overlay -------------
def mode_group_overlay(recs, args, dir_indices, group_key, title_label):
    constants = {p: getattr(args, p) for p in PARAMS if p != group_key}
    filtered = filter_constants(recs, args.type, constants, sweep_key=group_key)  # <-- FIXED
    if not filtered:
        print_available_values(recs)
        raise SystemExit(f"No devices matched type/constants for {group_key} mode.")

    by_key = {}
    for r in filtered:
        by_key.setdefault(r[group_key], []).append(r)

    out_png = args.out
    if os.path.isdir(os.path.dirname(out_png)) and (os.path.splitext(out_png)[1]=="" or out_png.endswith("/")):
        bits = [f"type_{args.type}", f"{group_key}_sweep"]
        for k in PARAMS:
            if k != group_key and getattr(args, k) is not None:
                bits.append(f"{k}_{getattr(args,k):g}")
        bits += [f"vds_{args.target_vds:g}", f"dirs_{'-'.join(map(str,dir_indices))}"]
        os.makedirs(args.out, exist_ok=True)
        out_png = os.path.join(args.out, "_".join(bits) + ".png")

    # title = f"Id–Vgs • sweep={group_key} @ Vds≈{args.target_vds:g} V • dirs {','.join(map(str,dir_indices))}\n{title_label}"

    DIR_LABEL = {0:"Forward 1st", 1:"Reverse 1st", 2:"Forward 2nd", 3:"Reverse 2nd"}
    dirs_str = ', '.join(DIR_LABEL.get(i, str(i)) for i in dir_indices)
    title = (
        f"Id–Vgs • sweep={args.sweep} @ Vds={args.target_vds:g} V • dirs {dirs_str}\n"
        f"{title_label}"
    )
    plot_group_overlay(
        by_key, program=args.program, target_vds=args.target_vds, dir_indices=dir_indices,
        use_abs=args.abs_current, logy=args.logy, title=title, group_key=group_key,
        out_png=out_png, legend_cols=4
    )

def print_available_values(records):
    from collections import defaultdict
    vals = defaultdict(set)
    for r in records:
        for k in ("type","lc","lch","lov","gateasym","w","nf"):
            if k in r:
                vals[k].add(r[k])
    print("\n[debug] Available values under --root:")
    for k in ["type","lc","lch","lov","gateasym","w","nf"]:
        if k in vals:
            print(f"  {k}: {sorted(vals[k])}")

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Analyze IdVgs with multiple sweep families.")
    ap.add_argument("--mode", choices=["tlm","w","ov","long","tlm", "fingered"], default="fingered")
    ap.add_argument("--root", default="./scripts/Analysis/W/die_x_0_y_0/tlm2", help="Root folder to scan")
    ap.add_argument("--type", choices=["nmos","pmos","both"], default="both")
    ap.add_argument("--sweep", choices=["lc","lch","lov","gateasym","w","nf"], default="nf")
    ap.add_argument("--max_cols", type=int, default=6)

    ap.add_argument("--lc", type=float, default=0.2)
    ap.add_argument("--lch", type=float, default=0.4)
    ap.add_argument("--lov", type=float, default=0.06)
    ap.add_argument("--gateasym", type=float, default=0.00)
    ap.add_argument("--w", type=float, default=4.0)
    ap.add_argument("--nf", type=float, default=None)

    ap.add_argument("--program", default="keysight_id_vgs")
    ap.add_argument("--target_vds", type=float, default=0.8)
    ap.add_argument("--dir_indices", default="2", help="'all' or comma list like '2,3'") #direction
    ap.add_argument("--abs_current", default="true", action="store_true")
    ap.add_argument("--logy", default="true", action="store_true")
    ap.add_argument("--csv_out", default="")
    ap.add_argument("--out", default="./scripts/Analysis/W/_config_summary", help="Output PNG path OR folder")

    args = ap.parse_args()

    dir_indices = parse_dir_indices(args.dir_indices)
    recs = scan_device_dirs(args.root)
    print_available_values(recs)
    if not recs:
        # show the first 10 folder names we *tried* to match
        try:
            children = [d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root,d))]
            print("[debug] first subfolders under --root:", children[:10])
        except Exception as e:
            print("[debug] cannot list --root:", e)

    if args.csv_out:
        write_csv(recs, args.csv_out)
        print(f"[csv] wrote -> {args.csv_out} ({len(recs)} rows)")

    if args.mode == "tlm":
        mode_tlm(recs, args, dir_indices)
    elif args.mode == "w":
        mode_group_overlay(recs, args, dir_indices, group_key="w",
                           title_label=f"type={args.type}, lc={args.lc:g}, lch={args.lch:g}, lov={args.lov:g}" +
                                       (f", gateasym={args.gateasym:g}" if args.gateasym is not None else ""))
    elif args.mode == "ov":
        mode_group_overlay(recs, args, dir_indices, group_key="lov",
                           title_label=f"type={args.type}, lc={args.lc:g}, lch={args.lch:g}, w={args.w:g}" +
                                       (f", gateasym={args.gateasym:g}" if args.gateasym is not None else ""))
    elif args.mode == "long":
        mode_group_overlay(recs, args, dir_indices, group_key="lch",
                           title_label=f"type={args.type}, lc={args.lc:g}, lov={args.lov:g}, w={args.w:g}" +
                                       (f", gateasym={args.gateasym:g}" if args.gateasym is not None else ""))
    elif args.mode == "fingered":
        mode_group_overlay(recs, args, dir_indices, group_key="nf",
                           title_label=f"type={args.type}, lc={args.lc:g}, lch={args.lch:g}, lov={args.lov:g}, w={args.w:g}" +
                                       (f", gateasym={args.gateasym:g}" if args.gateasym is not None else ""))

if __name__ == "__main__":
    main()