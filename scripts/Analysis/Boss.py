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

def parse_vds_single(arg: str):
    if not arg:
        return 0.8
    try:
        return float(arg.split(",")[0].strip())
    except Exception:
        return 0.8

def filter_constants(records, dev_type, consts, tol=1e-9):
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

# --------- selection helpers (choose exact device_id if provided) ---------
def available_device_ids(root, dev_type, lc, lch, lov, gateasym, w, nf):
    recs = scan_device_dirs(root)
    consts = {"lc": lc, "lch": lch, "lov": lov, "gateasym": gateasym, "w": w, "nf": nf}
    matches = filter_constants(recs, dev_type, consts)
    return sorted(set(int(m.get("device_id")) for m in matches if "device_id" in m))

def find_matching_record(root, dev_type, lc, lch, lov, gateasym, w, nf, device_id=None):
    recs = scan_device_dirs(root)
    consts = {"lc": lc, "lch": lch, "lov": lov, "gateasym": gateasym, "w": w, "nf": nf}
    matches = filter_constants(recs, dev_type, consts)
    if device_id is not None:
        matches = [m for m in matches if int(m.get("device_id", -1)) == int(device_id)]
    matches.sort(key=lambda r: (r.get("path",""), r.get("device_id", 0)))
    return matches[0] if matches else None

# ---------------- boss plot (specific sources with device choice) ----------------
def mode_boss_specific(args, dir_indices, vds):
    """
    Plot only one VDS (default 0.8 V):
      - PMOS from args.p_root with (Lc,Lch)=(args.p_lc,args.p_lch) and optional --p_device_id
      - NMOS from args.n_root with (Lc,Lch)=(args.n_lc,args.n_lch) and optional --n_device_id
    PMOS: plot only VGS <= 0; NMOS: plot only VGS >= 0
    """
    # PMOS record
    rec_p = find_matching_record(
        root=args.p_root, dev_type="pmos",
        lc=args.p_lc, lch=args.p_lch, lov=args.lov,
        gateasym=args.gateasym, w=args.w, nf=args.nf,
        device_id=args.p_device_id
    )
    if not rec_p and args.p_device_id is not None:
        ids = available_device_ids(args.p_root, "pmos", args.p_lc, args.p_lch, args.lov, args.gateasym, args.w, args.nf)
        print(f"[warn] PMOS device_id={args.p_device_id} not found. Available: {ids}")

    # NMOS record
    rec_n = find_matching_record(
        root=args.n_root, dev_type="nmos",
        lc=args.n_lc, lch=args.n_lch, lov=args.lov,
        gateasym=args.gateasym, w=args.w, nf=args.nf,
        device_id=args.n_device_id
    )
    if not rec_n and args.n_device_id is not None:
        ids = available_device_ids(args.n_root, "nmos", args.n_lc, args.n_lch, args.lov, args.gateasym, args.w, args.nf)
        print(f"[warn] NMOS device_id={args.n_device_id} not found. Available: {ids}")

    if not rec_p and not rec_n:
        raise SystemExit("Did not find a matching PMOS or NMOS record for the requested (Lc, Lch) and device_id.")

    plt.figure(figsize=(7.0, 4.2))
    ax = plt.gca()
    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1'])
    color_n = colors[0]
    color_p = colors[1] if len(colors) > 1 else "C1"

    def plot_one(rec, color, is_nmos):
        if not rec:
            return False
        h5 = os.path.join(rec["path"], f"{args.program}.h5")
        for d_idx in dir_indices:
            vgs, idc = load_idvgs_curve(h5, d_idx, vds, args.abs_current)
            if vgs is None:
                continue
            # Keep only the desired VGS side
            if is_nmos:
                mask = vgs >= 0
            else:
                mask = vgs <= 0
            if not np.any(mask):
                continue
            vgs_plot = vgs[mask]
            y = (idc / 4e-6)[mask]  # μA/μm for W=4 μm
            if is_nmos:
                ax.plot(vgs_plot, y, linewidth=1.8, alpha=0.95, color=color,
                        label=f"NMOS")
            else:
                ax.plot(vgs_plot, y, linewidth=1.2, alpha=0.95, color=color,
                        marker="^", markevery=max(1, len(vgs_plot)//18), markersize=3.5,
                        label=f"PMOS")
            return True
        return False

    ok_p = plot_one(rec_p, color_p, is_nmos=False)
    ok_n = plot_one(rec_n, color_n, is_nmos=True)
    if not (ok_p or ok_n):
        raise SystemExit("Failed to load curves at VDS for both devices.")

    # Axes/labels
    ax.set_yscale("log" if args.logy else "linear")
    ax.set_xlabel(r"$V_{GS}$ [V]")
    ax.set_ylabel(r"$I_D$ [$\mu$A/$\mu$m]")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.axvline(0.0, color="k", linewidth=1.0, alpha=0.4)
    ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=9)

    # Save
    out_path = args.out
    if os.path.isdir(os.path.dirname(out_path)) and (os.path.splitext(out_path)[1] == "" or out_path.endswith("/")):
        os.makedirs(out_path, exist_ok=True)
        out_path = os.path.join(out_path, f"boss_nmospmos_vds{str(vds).replace('.','p')}.png")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"[ok] wrote -> {out_path}")

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Boss plot with device selection (PMOS VGS≤0, NMOS VGS≥0).")
    # Fixed roots + targets from your request (override as needed)
    ap.add_argument("--p_root", default="./scripts/Analysis/W/die_x_0_y_0/tlm2", help="Root for PMOS selection")
    ap.add_argument("--n_root", default="./scripts/Analysis/Al/tlm", help="Root for NMOS selection")
    ap.add_argument("--p_lc", type=float, default=0.2)
    ap.add_argument("--p_lch", type=float, default=0.4)
    ap.add_argument("--n_lc", type=float, default=0.4)
    ap.add_argument("--n_lch", type=float, default=0.4)
    ap.add_argument("--p_device_id", type=int, default=4, help="PMOS device_id (1..24) to plot")
    ap.add_argument("--n_device_id", type=int, default=3, help="NMOS device_id (1..24) to plot")

    # Shared measurement filters (set to None to ignore)
    ap.add_argument("--lov", type=float, default=0.06)
    ap.add_argument("--gateasym", type=float, default=0.00)
    ap.add_argument("--w", type=float, default=4.0)
    ap.add_argument("--nf", type=float, default=None)

    # IO / plotting
    ap.add_argument("--program", default="keysight_id_vgs")
    ap.add_argument("--vds", default="0.8", help="Single VDS value (default 0.8 V)")
    ap.add_argument("--dir_indices", default="2", help="'all' or comma list like '2,3'")
    ap.add_argument("--abs_current", action="store_true", default=True)
    ap.add_argument("--logy", action="store_true", default=True)
    ap.add_argument("--csv_out", default="")
    ap.add_argument("--out", default="./scripts/Analysis/W/_config_summary", help="Output file OR folder")

    args = ap.parse_args()
    dir_indices = parse_dir_indices(args.dir_indices)
    vds_single = parse_vds_single(args.vds)

    # Optional: dump matches to CSV for debugging
    if args.csv_out:
        recs = scan_device_dirs(args.p_root) + scan_device_dirs(args.n_root)
        write_csv(recs, args.csv_out)
        print(f"[csv] wrote -> {args.csv_out} ({len(recs)} rows)")

    mode_boss_specific(args, dir_indices, vds_single)

if __name__ == "__main__":
    main()
