# compressor_series_off_design.py

import os
import numpy as np
import pandas as pd
import CoolProp.CoolProp as CP
from scipy.optimize import root
from scipy.optimize import minimize
from plot_ts_dome_compressor_series import plot_ts_dome_compressor_series
from simulate_compressor_series import simulate_compressor_series
from plot_compressor_series import plot_compressor_series_summary


# Inputs
fluid = "CO2"
backend = "REFPROP"
p01_0 = 30e5
T01_sweep = [26, 27, 28]
mdot_des = 1354  # kg/s

# Optimisation flag
optimize_design = True     # <<< set False to use fixed design values

# User-defined target overall pressure ratio
PR_target = 10            # <<< you can change this as needed
n_stages = 5

# User option: choose what the optimizer should maximize
objective_flag = "eta_weighted"   # options: "eta_weighted" or "work_total"

# Save series data to CSV
dump_table = True          # print/save the summary at the end
csv_name = r".\compressor_series_pictures\compressor_series_summary.csv"

# ------------------------ Design (fixed or optimized) ------------------------

# Default (fixed) design values = current baseline / also used as optimizer seed
phi_fixed = np.array([0.045] * n_stages, dtype=float)
Ma_fixed  = np.array([0.65] * n_stages, dtype=float)

# phi_fixed = np.array([0.0706, 0.0696, 0.0711, 0.0653, 0.0692], dtype=float)
# Ma_fixed  = np.array([0.4846, 0.5775, 0.6344, 0.5058, 0.6389], dtype=float)

def _run_design(phi_list, Ma_list):
    """Run a single design evaluation and return the full result dict."""
    return simulate_compressor_series(
        n_stages=n_stages,
        mdot=mdot_des,
        phi_list=list(phi_list),
        p01_0=p01_0,
        T01_list=[T01_sweep[0] + 273.15] * n_stages,
        Ma_list=list(Ma_list),
        fluid=fluid,
        backend=backend,
        show_figure=False,
    )

def _design_objective(x, obj_flag=objective_flag):
    """
    Objective function for compressor design optimization.

    Parameters
    ----------
    x : array
        [phi_1...phi_n, Ma_1...Ma_n]
    obj_flag : str
        "eta_weighted" -> maximize weighted average of stage efficiencies
        "work_total"   -> maximize total stage work

    Returns
    -------
    float
        Value to minimize (negative of objective).
    """
    phi = x[:n_stages]
    Ma  = x[n_stages:]
    try:
        res = _run_design(phi, Ma)
        stage = res["stage"]

        if obj_flag == "eta_weighted":
            # Weighted-average of eta_is_ts by stage work
            eta = np.asarray(stage["eta_is_ts"], float)
            w   = np.asarray(stage["W"], float)
            if not np.isfinite(eta).all() or not np.isfinite(w).all() or np.all(w <= 0):
                return 1e7
            obj = np.average(eta, weights=w)

        elif obj_flag == "work_total":
            # Maximize total stage work (sum)
            w = np.asarray(stage["W"], float) / 1e3
            if not np.isfinite(w).all():
                return 1e7
            obj = np.sum(w)

        else:
            raise ValueError(f"Unknown objective_flag: {obj_flag}")

        # Negative because we minimize in scipy.optimize.minimize
        return -float(obj)

    except Exception:
        # Infeasible / failed simulation → large penalty
        return 1e7


def _design_constraint_eq(x):
    """Equality constraint: overall PR_tt_series == PR_target"""
    phi = x[:n_stages]
    Ma  = x[n_stages:]
    try:
        res = _run_design(phi, Ma)
        pr = float(res["overall"]["PR_tt_series"])
        return pr - PR_target
    except Exception:
        # Push solver away from failures (treat as far from target)
        return 1e4

# Bounds
phi_lo, phi_hi = 0.03, 0.08
Ma_lo,  Ma_hi  = 0.4, 0.9
bounds = [(phi_lo, phi_hi)] * n_stages + [(Ma_lo, Ma_hi)] * n_stages

# Initial guess from your current design
x0 = np.concatenate([phi_fixed, Ma_fixed])

if not optimize_design:
    # --- Fixed design path (original behavior) ---
    res_des = _run_design(phi_fixed, Ma_fixed)

else:
    # -------- Iteration logging --------
    def _make_opt_callback():
        """Stateful callback that prints a line each SLSQP iteration."""
        state = {"krpm": 0}
        header_printed = {"flag": False}

        def _callback(xk):
            state["krpm"] += 1
            n_stages = xk.size // 2
            phi = xk[:n_stages]
            Ma = xk[n_stages:]

            # Objective we are maximizing (minimize returns negative)
            try:
                obj = -_design_objective(xk)  # flip sign back to 'maximize'
            except Exception:
                obj = float("nan")

            # Equality constraint residual (PR - PR_target), should -> 0
            try:
                c_eq = _design_constraint_eq(xk)
            except Exception:
                c_eq = float("nan")

            # Current PR (for readability)
            try:
                res_tmp = _run_design(phi, Ma)
                PR_now = float(res_tmp["overall"]["PR_tt_series"])
            except Exception:
                PR_now = float("nan")

            # Print header once
            if not header_printed["flag"]:
                print(
                    "\nIter |     objective |     c_eq(PR-PR*) |      PR_now |")
                header_printed["flag"] = True

            # Compact arrays
            phi_str = np.array2string(phi, precision=4, separator=',', suppress_small=False)
            Ma_str = np.array2string(Ma, precision=4, separator=',', suppress_small=False)

            # Console line
            print(f"{state['krpm']:4d} | {obj:13.5f} | {c_eq:16.3e} | {PR_now:11.5f} |")

        return _callback

    # --- Optimized design path ---
    cons = [{'type': 'eq', 'fun': _design_constraint_eq}]
    opt = minimize(
        _design_objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options=dict(maxiter=500, ftol=1e-4, disp=True),    # disp=True prints final summary
        callback=_make_opt_callback(),                      # per-iteration console log
    )

    # Use optimal if successful; otherwise fall back to seed
    if opt.success and np.isfinite(opt.fun):
        x_best = opt.x
        phi_opt, Ma_opt = x_best[:n_stages], x_best[n_stages:]
        res_des = _run_design(phi_opt, Ma_opt)
        print("[design] Optimization successful.")
        print(f"         phi* = {phi_opt.round(5)}")
        print(f"         Ma*  = {Ma_opt.round(5)}")
        print(f"         PR_tt_series = {res_des['overall']['PR_tt_series']:.4f}")
        # For convenience, also print weighted-avg eta_is_ts achieved
        eta = np.asarray(res_des["stage"]["eta_is_ts"], float)
        w   = np.asarray(res_des["stage"]["W"], float)
        w_tot = np.sum(w) / 1e3
        eta_w = float(np.average(eta, weights=w))
        print(f"         weighted ⟨η_is,ts⟩ = {eta_w:.3f}")
        print(f"         Overall ⟨W⟩ = {w_tot:.2f} kJ/kg")
        print(f"         Power input ⟨Wdot⟩ = {w_tot * mdot_des / 1e3:.1f} MW")
    else:
        print("[design] Optimization failed or did not converge; using fixed design.")
        res_des = _run_design(phi_fixed, Ma_fixed)

# Extract rpm and r2 from the chosen design for subsequent off-design sims
rpm_des = res_des["stage"]["rpm"]
r2_des  = res_des["stage"]["r2"]
pr_des  = res_des["overall"]["PR_tt_series"]

# ---------------------------------------------------------------------------

# --- Off-design ---

# Utilities

def _simulation(mdot, T01):
    res_cmp = simulate_compressor_series(
        n_stages=n_stages,
        mdot=mdot,
        r2_list=r2_des,
        p01_0=p01_0,
        T01_list=[T01 + 273.15] * n_stages,
        N_list=rpm_des,
        fluid=fluid,
        backend=backend,
        show_figure=False,
    )
    return res_cmp

def _mdot_residual(mdot_guess, T01):
    # Ensure scalar even if root passes a 1-element ndarray
    mdot_guess = np.atleast_1d(mdot_guess).item()
    res_cmp = _simulation(mdot_guess,T01)
    pr = res_cmp["overall"]["PR_tt_series"]
    mdot_actual = mdot_des * pr / pr_des
    residual = mdot_guess - mdot_actual
    return np.array([residual]) # keep array shape for scipy.root

# Offdesign 1
convergence_od1 = root(_mdot_residual, mdot_des, args=(T01_sweep[1],), tol=1e-6)
if convergence_od1.success: # Final update
    res_od1 = _simulation(convergence_od1.x.item(),T01_sweep[1])
    print(f"Offdesign converged for {T01_sweep[1]}°C")
else:
    print(f"Offdesign did not converged for {T01_sweep[1]}°C")

# Offdesign 2
convergence_od2 = root(_mdot_residual, mdot_des, args=(T01_sweep[2],), tol=1e-6)
if convergence_od2.success: # Final update
    res_od2 = _simulation(convergence_od2.x.item(),T01_sweep[2])
    print(f"Offdesign converged for {T01_sweep[2]}°C")
else:
    print(f"Offdesign did not converged for {T01_sweep[2]}°C")


# -------- Plot --------
in_p01   = np.vstack([res_des["stage"]["p01"],
                      res_od1["stage"]["p01"],
                      res_od2["stage"]["p01"]])
out_p05  = np.vstack([res_des["stage"]["p05"],
                      res_od1["stage"]["p05"],
                      res_od2["stage"]["p05"]])
in_T01   = np.vstack([res_des["stage"]["T01"],
                      res_od1["stage"]["T01"],
                      res_od2["stage"]["T01"]])
out_T05  = np.vstack([res_des["stage"]["T05"],
                      res_od1["stage"]["T05"],
                      res_od2["stage"]["T05"]])

labels = [f"{T01_sweep[0]} °C (design)",
          f"{T01_sweep[1]} °C (off-design)",
          f"{T01_sweep[2]} °C (off-design)"]

fig, ax, data = plot_ts_dome_compressor_series(
    in_p01=in_p01,
    out_p05=out_p05,
    in_T01=in_T01,
    out_T05=out_T05,
    fluid=fluid,
    backend=backend,
    show_isobars=True,
    N_iso=30,
    save_path="./compressor_series_pictures/compare_trains_ts.png",
    labels=labels,
    show_markers=True,
)

# ------------------------ Tabular summary ------------------------
if dump_table:

    def tip_Mach_list(res, fluid, backend):
        """Ma_tip_i = u2_i / a01_i at each stage inlet."""
        AS = CP.AbstractState(backend, fluid)
        Ma = []
        for p01_i, T01_i, u2_i in zip(res["stage"]["p01"], res["stage"]["T01"], res["stage"]["u2"]):
            AS.update(CP.PT_INPUTS, float(p01_i), float(T01_i))
            a01 = AS.speed_sound()
            Ma.append(u2_i / a01 if a01 > 0 else np.nan)
        return np.asarray(Ma, float)

    # Calculates the turbine power output and calculates a simplified cycle net power output
    def _allam_cycle_power_simple(cmp):
        eta_el = 0.99
        eta_t = 0.94
        p_out_t = cmp["overall"]["p_in"]
        p_in_t = cmp["overall"]["p_out"]
        T_in_t = 1200 + 273.15  # Assumption, can be changed

        s_in_t = CP.PropsSI("S", "T", T_in_t, "P", p_in_t, fluid)
        h_in_t = CP.PropsSI("H", "T", T_in_t, "P", p_in_t, fluid)

        h_out_t_is = CP.PropsSI("H", "S", s_in_t, "P", p_out_t, fluid)

        h_out_t = h_in_t - eta_t * (h_in_t - h_out_t_is)

        mdot = cmp["inputs"]["mdot"]

        Wdot_t = mdot * (h_in_t - h_out_t) / 1e6 * eta_el
        Wdot_c = cmp["overall"]["Wdot"] / 1e6 / eta_el

        Wdot_net = Wdot_t - Wdot_c
        return Wdot_net

    series = [
        (f"{T01_sweep[0]} °C (design)",     res_des),
        (f"{T01_sweep[1]} °C (off-design)", res_od1),
        (f"{T01_sweep[2]} °C (off-design)", res_od2),
    ]

    # Infer number of stages from the first result
    n_stages = len(res_des["stage"]["p01"])

    rows = []
    for name, res in series:
        overallPR = float(res["overall"]["PR_tt_series"])
        p_out_bar = float(res["overall"]["p_out"]) / 1e5
        Wdot_MW   = float(res["overall"]["Wdot"]) / 1e6
        overallMdot = float(res["inputs"]["mdot"])
        Wdot_cycle_net_MW = float(_allam_cycle_power_simple(res))

        PR_ts = np.asarray(res["stage"]["PR_ts"], float)
        phi   = np.asarray(res["stage"]["phi"], float)
        u2    = np.asarray(res["stage"]["u2"], float)
        Ma    = tip_Mach_list(res, fluid, res["inputs"]["backend"])
        etaIS = np.asarray(res["stage"]["eta_is_ts"], float)
        rho_in = np.asarray(res["stage"]["rho_01"])

        row = {
            "Series": name,
            "PR_tt_overall": overallPR,
            "p_out_bar": p_out_bar,
            "Wdot_MW": Wdot_MW,
            "mdot_kgs": overallMdot,
            "Wdot_cycle_net_MW": Wdot_cycle_net_MW,
        }

        # Add per-stage columns
        for i in range(n_stages):
            row[f"PR_ts_stage{i+1}"]   = PR_ts[i]
            row[f"phi_stage{i+1}"]     = phi[i]
            row[f"Ma_tip_stage{i+1}"]  = Ma[i]
            row[f"u2_stage{i+1}_mps"]  = u2[i]
            row[f"eta_is_stage{i+1}"]  = etaIS[i]
            row[f"rho_in{i+1}"]        = rho_in[i]

        rows.append(row)

    df = pd.DataFrame(rows)

    # Order columns: overall, then each stage block
    col_overall = ["Series", "PR_tt_overall", "p_out_bar", "Wdot_MW", "mdot_kgs", "Wdot_cycle_net_MW"]
    col_stages = []
    for i in range(n_stages):
        col_stages += [
            f"PR_ts_stage{i+1}",
            f"phi_stage{i+1}",
            f"Ma_tip_stage{i+1}",
            f"u2_stage{i+1}_mps",
            f"eta_is_stage{i+1}",
            f"rho_in{i + 1}",
        ]
    df = df[col_overall + col_stages]

    # Nicely formatted print (rounded copies)
    df_print = df.copy()
    df_print["PR_tt_overall"] = df_print["PR_tt_overall"].map(lambda v: f"{v:.2f}")
    df_print["p_out_bar"]     = df_print["p_out_bar"].map(lambda v: f"{v:.1f}")
    df_print["Wdot_MW"]       = df_print["Wdot_MW"].map(lambda v: f"{v:.1f}")
    df_print["mdot_kgs"]      = df_print["mdot_kgs"].map(lambda v: f"{v:.1f}")
    for i in range(n_stages):
        df_print[f"PR_ts_stage{i+1}"]   = df_print[f"PR_ts_stage{i+1}"].map(lambda v: f"{v:.2f}")
        df_print[f"phi_stage{i+1}"]     = df_print[f"phi_stage{i+1}"].map(lambda v: f"{v:.3f}")
        df_print[f"Ma_tip_stage{i+1}"]  = df_print[f"Ma_tip_stage{i+1}"].map(lambda v: f"{v:.3f}")
        df_print[f"u2_stage{i+1}_mps"]  = df_print[f"u2_stage{i+1}_mps"].map(lambda v: f"{v:.1f}")
        df_print[f"eta_is_stage{i+1}"]  = df_print[f"eta_is_stage{i+1}"].map(lambda v: f"{v:.3f}")
        df_print[f"rho_in{i + 1}"] = df_print[f"rho_in{i + 1}"].map(lambda v: f"{v:.2f}")

    print("\n=== Compressor Series Summary (wide) ===")
    print(df_print.to_string(index=False))

    # Save CSV (unrounded numeric values)
    csv_path = csv_name
    df.to_csv(csv_path, index=False)
    print(f"\nTable saved to: {os.path.abspath(csv_name)}")

    # Plot CSV data to pictures
    result = plot_compressor_series_summary(csv_path, outdir="./compressor_series_pictures")
    print(f"\nFigures saved to: {os.path.abspath(csv_path)}")
