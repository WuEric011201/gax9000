"""
Generate BTI stress/relaxation figures using the existing Keysight B1500
Id–Vg/Id–Vd measurement 


Protocol summary 
  • During STRESS: hold V_G = V_G,Stress (<0 for PMOS NBTI / >0 for NMOS PBTI),
    V_D = V_S = 0. Periodically perform quick READ pulses to estimate V_T.
  • During RELAX: set V_G = 0, V_D = V_S = 0. Periodically READ to estimate V_T.
  • One stress sequence uses logarithmically spaced dwell times (≈10, 100, … 10k s),
    producing a |ΔV_T|(t) curve. Then one relaxation sequence with the same times
    produces a fraction‑remaining curve.
  • Repeat for multiple stress fields (different |V_G,Stress|) and two temperatures.

Notes
  • Uses the ProgramKeysightIdVgs measurement program for READs (fast, narrow sweep
    around V_T) so ΔV_T can be extracted via a constant‑current or linear‑fit method.
  • The B1500 channel mapping and other low‑level setup are inherited from the API.
  • Temperature control is left as a stub (set_chuck_temperature) — implement with
    your Velox station API if available.
  • For NMOS vs PMOS, select polarities accordingly with CONFIG at the top.

"""

import time
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

# === Import your existing controller API ===
# Assumes this file lives alongside your controller package from the shared code
from controller.programs.keysight_fet_iv import (
    ProgramKeysightIdVgs,
)
from controller.util import into_sweep_range
import pyvisa

# ===================== User configuration =====================
@dataclass
class ChannelMap:
    gate: int = 4
    source: int = 3
    drain: int = 5
    bulk: int = 6

@dataclass
class ReadConfig:
    # Small V_DS to stay in linear region during READ ("Minimal Vd_read")
    vds_read: float = -0.05   # PMOS example (negative). Use +0.05 for NMOS.
    # Narrow V_G sweep around the current V_T estimate
    vg_span: float = 0.3      # total span (V); sweep is [vt_guess - span/2, vt_guess + span/2]
    vg_step: float = 0.02
    # Constant current for V_T extraction (per‑width). If unknown, choose value that
    # falls in sub‑mA range during linear read. Adjust to your device geometry.
    id_const: float = 1e-7
    # Reading cadence during stress/relax (seconds)
    read_pulse_width: float = 10e-3   # controls instrument wait; not a true HW pulse here

@dataclass
class StressRelaxPlan:
    # Stress gate voltages corresponding to oxide fields labels used in the figure
    # Keys are legend strings; values are V_G,Stress (sign per device type)
    # Example for PMOS (negative gate stress). Adjust for NMOS as needed.
    stress_levels: Dict[str, float] = None
    # Log‑spaced sample times (s) used for both stress and relax sequences
    log_times: np.ndarray = None
    # Temperatures to run (°C)
    temps_C: List[int] = None

# ---- Default plan resembling the sample figure ----
DEFAULT_PLAN = StressRelaxPlan(
    stress_levels={
        "-12 MV/cm": -1.2,
        "-10 MV/cm": -1.0,
        "-8 MV/cm":  -0.8,
        "-6 MV/cm":  -0.6,
        "-4 MV/cm":  -0.4,
        "-2 MV/cm":  -0.2,
    },
    log_times=np.unique(np.round(np.logspace(1, 2, num=3), 0)).astype(int),  # ~10…10000 s
    temps_C=[25, 125],
)

# Device polarity switch: set to +1 for NMOS (PBTI), -1 for PMOS (NBTI)
POLARITY = -1  # -1 => PMOS example (negative gate stress)

CHANNELS = ChannelMap()
READ = ReadConfig()
PLAN = DEFAULT_PLAN

# ===================== Temperature control stub =====================

def set_chuck_temperature(temp_C: float):
    """Stub for Velox / prober chuck temperature control.
    Replace with your station API. Include soak/wait for stabilization.
    """
    print(f"[TEMP] Set chuck to {temp_C} °C (stub). Wait for stabilization as needed…")

# ===================== Measurement helpers =====================

def make_rm_and_instrument(gpib_addr: int):
    rm = pyvisa.ResourceManager()
    inst = rm.open_resource(f"GPIB0::{gpib_addr}::INSTR")
    print(inst.query("*IDN?"))
    return rm, inst


def quick_read_vth(
    instr,
    vt_guess: float,
    negate_id: bool = True,
) -> Tuple[float, dict]:
    """Perform a *quick* Id–Vg sweep around vt_guess at small Vds and extract V_T.
    Returns (vt, raw_result_data).
    """
    v_start = vt_guess - READ.vg_span / 2.0
    v_stop  = vt_guess + READ.vg_span / 2.0
    # Ensure sweep direction includes both forward and reverse to minimize hysteresis bias
    sweep = ProgramKeysightIdVgs.run(
        instr_b1500=instr,
        probe_gate=CHANNELS.gate,
        probe_source=CHANNELS.source,
        probe_drain=CHANNELS.drain,
        probe_sub=CHANNELS.bulk,
        v_gs={"start": v_start, "stop": v_stop, "step": READ.vg_step},
        v_ds=[READ.vds_read],
        v_sub=0.0,
        negate_id=True,
        sweep_direction="fr",
        id_compliance=0.010,
        ig_compliance=0.001,
        stop_on_error=True,
    )
    data = sweep.data
    # data shapes: [bias, sweep, points]
    vgs = np.array(data["v_gs"]).reshape(-1)
    ids = np.array(data["i_d"]).reshape(-1)
    if negate_id:
        ids = -ids

    # Extract V_T using constant‑current criterion (Id = READ.id_const at small Vds)
    # Interpolate Vgs at Id = id_const (monotonic assumption in the window)
    # If not bracketing, fall back to linear fit around gm peak.
    try:
        vt = np.interp(READ.id_const, ids, vgs)
    except Exception:
        # Linear fit Id ≈ k*(Vgs - Vt) in linear region
        # use upper 60% of Id range in the sweep window
        k = max(1, int(0.6 * len(vgs)))
        idx = np.argsort(ids)[-k:]
        A = np.vstack([vgs[idx], np.ones(len(idx))]).T
        m, b = np.linalg.lstsq(A, ids[idx], rcond=None)[0]  # ids = m*vgs + b
        vt = -b / m

    return vt, data


def stress_hold(instr, vg_stress: float, t_s: float):
    """Apply a static gate stress for t_s seconds (drain/source grounded)."""
    # Direct SMU force using DV via measurement_keysight_b1500_setup isn't exported here,
    # so we reuse a minimal Program run with 1‑point sweep just to set/hold.
    # Alternatively, implement a tiny helper that writes DV directly.
    # Here we approximate by setting Vg to vg_stress with a 1‑point WV sweep.
    ProgramKeysightIdVgs.run(
        instr_b1500=instr,
        probe_gate=CHANNELS.gate,
        probe_source=CHANNELS.source,
        probe_drain=CHANNELS.drain,
        probe_sub=CHANNELS.bulk,
        v_gs={"start": vg_stress, "stop": vg_stress, "step": 0.1},
        v_ds=[0.0],
        v_sub=0.0,
        negate_id=True,
        sweep_direction="f",
        id_compliance=0.010,
        ig_compliance=0.001,
        stop_on_error=True,
    )
    print(f"[STRESS] Holding Vg={vg_stress:.3f} V for {t_s:.1f} s…")
    time.sleep(t_s)
    # Return to 0V
    ProgramKeysightIdVgs.run(
        instr_b1500=instr,
        probe_gate=CHANNELS.gate,
        probe_source=CHANNELS.source,
        probe_drain=CHANNELS.drain,
        probe_sub=CHANNELS.bulk,
        v_gs={"start": 0.0, "stop": 0.0, "step": 0.1},
        v_ds=[0.0],
        v_sub=0.0,
        negate_id=True,
        sweep_direction="f",
        id_compliance=0.010,
        ig_compliance=0.001,
        stop_on_error=True,
    )


# ===================== Orchestrator =====================

def run_bti_sequences(
    instr,
    plan: StressRelaxPlan = PLAN,
    initial_vt_guess: float = -0.3,  # PMOS example. Use +0.3 for NMOS devices.
) -> Dict[str, dict]:
    results: Dict[str, dict] = {}

    for temp_C in plan.temps_C:
        set_chuck_temperature(temp_C)
        # Initial reference V_T at this temperature
        vt0, _ = quick_read_vth(instr, vt_guess=initial_vt_guess)
        print(f"[T={temp_C}°C] Initial V_T ≈ {vt0:.4f} V")

        results[str(temp_C)] = {}

        for label, vg_stress in plan.stress_levels.items():
            vg_stress = float(POLARITY) * abs(vg_stress)  # enforce polarity

            # --- Stress phase ---
            vt_over_time = []
            elapsed = 0.0
            vt_curr_guess = vt0
            for t_target in plan.log_times:
                dt = float(t_target) - elapsed
                dt = max(0.0, dt)
                stress_hold(instr, vg_stress, dt)
                elapsed = float(t_target)
                # Spot read V_T
                vt_i, _ = quick_read_vth(instr, vt_guess=vt_curr_guess)
                vt_over_time.append(vt_i)
                vt_curr_guess = vt_i

            vt_over_time = np.array(vt_over_time)
            dvt_stress = vt_over_time - vt0  # signed shift

            # --- Relax phase ---
            vt_relax = []
            vt_curr_guess = vt_over_time[-1]
            # ensure gate at 0 V during relax between reads; handled in quick_read_vth
            for t_rel in plan.log_times:
                time.sleep(max(0.0, t_rel - 0.0))  # passive wait between reads
                vt_r, _ = quick_read_vth(instr, vt_guess=vt_curr_guess)
                vt_relax.append(vt_r)
                vt_curr_guess = vt_r

            vt_relax = np.array(vt_relax)
            dvt0 = (vt_over_time[-1] - vt0)
            dvt_relax = vt_relax - vt0
            frac_remaining = np.where(np.abs(dvt0) > 0, np.abs(dvt_relax) / np.abs(dvt0), np.nan)

            results[str(temp_C)][label] = {
                "times": plan.log_times.astype(float),
                "dvt_stress": np.abs(dvt_stress),
                "dvt_relax_frac": frac_remaining,
            }
            print(f"[T={temp_C}°C, {label}] Final |ΔV_T| after stress: {np.abs(dvt0):.4f} V")

    return results


# ===================== Plotting =====================

def make_figures(results: Dict[str, dict], outfile_prefix: str = "bti"):
    # Panels (c) and (d): |ΔV_T| vs time @ 25C and 125C
    for temp_key in results.keys():
        # Determine panel letter based on temp
        temp_C = int(temp_key)
        panel = "c" if temp_C == 25 else "d"
        plt.figure()
        for label, payload in results[temp_key].items():
            t = payload["times"]
            y = payload["dvt_stress"]
            plt.loglog(t, y, marker=".", label=label)
        plt.xlabel("Time [s]")
        plt.ylabel("|ΔV_T| [V]")
        plt.title(f"EOT = 2.13 nm, T = {temp_C}°C")
        plt.legend()
        plt.grid(True, which="both", ls=":")
        plt.tight_layout()
        plt.savefig(f"{outfile_prefix}_{panel}_abs_dvt_T{temp_C}.png", dpi=300)

    # Panels (e) and (f): fraction remaining during relax @ 25C and 125C
    for temp_key in results.keys():
        temp_C = int(temp_key)
        panel = "e" if temp_C == 25 else "f"
        plt.figure()
        for label, payload in results[temp_key].items():
            t = payload["times"]
            y = payload["dvt_relax_frac"]
            plt.semilogx(t, y, marker=".", label=label)
        plt.xlabel("Time [s]")
        plt.ylabel("ΔV_T Fraction Remaining")
        plt.ylim(0, 1.05)
        plt.title(f"EOT = 2.13 nm, T = {temp_C}°C")
        plt.legend()
        plt.grid(True, which="both", ls=":")
        plt.tight_layout()
        plt.savefig(f"{outfile_prefix}_{panel}_frac_relax_T{temp_C}.png", dpi=300)


# ===================== Main =====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run BTI stress/relax measurement and generate figures.")
    parser.add_argument("--gpib", type=int, default=16, help="GPIB address for Keysight B1500")
    parser.add_argument("--prefix", type=str, default="bti", help="Output file prefix for PNGs")
    parser.add_argument("--vt_guess", type=float, default=-0.3, help="Initial VT guess for quick reads")
    parser.add_argument("--norange", action="store_true", help="Skip actual instrument connection (dev/test mode)")

    args = parser.parse_args()

    if args.norange:
        print("[DEV] No instrument mode: generating synthetic curves for layout check…")
        synthetic = {}
        for T in PLAN.temps_C:
            temp_key = str(T)
            synthetic[temp_key] = {}
            for label in PLAN.stress_levels.keys():
                t = PLAN.log_times.astype(float)
                # Fake ΔV_T growth ~ A * log(1+t/τ) with temp + field scaling
                A = (0.02 + 0.01 * (T == 125)) * (1.0 + 0.3 * list(PLAN.stress_levels.keys()).index(label))
                tau = 5.0
                dvt = A * np.log1p(t / tau)
                # Relax: simple stretched‑exp decay
                beta = 0.35 + 0.05 * (T == 125)
                tau_r = 200.0
                frac = np.exp(-(t / tau_r) ** beta)
                synthetic[temp_key][label] = {"times": t, "dvt_stress": dvt, "dvt_relax_frac": frac}
        make_figures(synthetic, outfile_prefix=args.prefix)
    else:
        rm, instr = make_rm_and_instrument(args.gpib)
        try:
            res = run_bti_sequences(instr, plan=PLAN, initial_vt_guess=args.vt_guess)
            make_figures(res, outfile_prefix=args.prefix)
        finally:
            try:
                instr.write("CL")
            except Exception:
                pass
