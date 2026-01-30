"""
Four-compressor series simulator with intercooling.

Requirements
- CoolProp (with HEOS or REFPROP) 
- The local module `compressor_model_vaned.py` in the same folder.

This module exposes one main function:

    simulate_compressor_series(
        n_stages: int = 4,
        mdot=300.0,
        base_geometry=None,
        phi_list=None,
        r2_list=None,
        p01_0=60e5,
        T01_list=None,
        N_list=None,
        Ma_list=None,
        fluid="CO2",
        backend="auto",
        show_figure=False,
    ) -> dict

Inputs (summary)
- mdot: single mass flow rate through all four compressors [kg/s].
- base_geometry: dict of geometry parameters common to all stages EXCEPT r2, which
  is handled per stage. If None, a sensible baseline scaling with r2 is used.
- Either `phi_list` (4 values) or `r2_list` (4 values), but not both.
- p01_0: total inlet pressure to the first compressor [Pa].
- T01_list: list/tuple of 4 total inlet temperatures [K], one for each stage. If None,
  use four equal values at 35°C.
- Either `N_list` (RPM for each stage) or `Ma_list` (tip Mach numbers at each stage),
  but not both. If both are None, default to four equal Mach=0.8.

Behavior
- For each stage i, inlet state is (p01_i, T01_list[i]). The outlet *total* pressure p05
  from stage i is used as the inlet total pressure p01 for stage i+1. Temperatures at
  each inlet are taken from T01_list (intercooling).
- If `phi_list` is given, r2 is solved from the mdot relation. If `r2_list` is given,
  phi is computed from the same relation. The relation is

    mdot = phi * rho_01 * u2 * (2 * r2)^2

  where u2 is the impeller tip speed for that stage. When N is specified, u2 = ω*r2
  with ω = 2π N/60, which makes r2 appear also in u2; in this case the solution is

    r2 = ( mdot / (8 * phi * rho_01 * ω) )^(1/3)

  When Ma is specified, u2 = Ma * a_01 and

    r2 = 0.5 * sqrt( mdot / (phi * rho_01 * u2) ).

Outputs
- Dict with per-stage arrays for:
  * eta_is_ts  (isentropic efficiency, total-static)
  * PR_ts      (pressure ratio total-static, p5/p1 of the stage model)
  * dh0        (total enthalpy rise h05 - h01 for the stage) [J/kg]
- Also returns derived per-stage r2, phi, u2, rpm, and the cumulative overall PR_tt
  based on p05 chaining.
"""
from __future__ import annotations
from typing import Dict, Any, Iterable, Optional, Tuple
import math
import numpy as np

# CoolProp
import CoolProp.CoolProp as CP

# Local model
from compressor_model_vaned import CentrifugalCompressor

# Figures
from dvdt_contour import plot_alpha_TS_from_AS
from matplotlib import pyplot as plt
from plot_ts_dome_compressor_series import plot_ts_dome_compressor_series


# --- Backend selection helper (adapted from compressor_ma_phi_sweep.py) ---

def _select_backend(fluid: str, prefer: str = "auto") -> Tuple[str, str]:
    prefer = (prefer or "auto").upper()
    if prefer not in {"AUTO", "REFPROP", "HEOS"}:
        prefer = "AUTO"

    def _ok(backend: str) -> bool:
        try:
            CP.AbstractState(backend, fluid)
            return True
        except Exception:
            return False

    if prefer in {"AUTO", "REFPROP"}:
        if _ok("REFPROP"):
            return "REFPROP", "Using REFPROP backend."
        if _ok("HEOS"):
            return "HEOS", "REFPROP unavailable; using HEOS."
    elif prefer == "HEOS":
        if _ok("HEOS"):
            return "HEOS", "Using HEOS backend."

    raise RuntimeError(
        f"CoolProp backends REFPROP/HEOS unavailable for fluid '{fluid}'. "
        "Check installation and fluid name."
    )

# --- Default geometry template (fields that scale with r2 will be set per stage) ---

def _default_geometry_for_r2(r2: float) -> Dict[str, Any]:
    g = dict(
        N_blades_impeller=12,
        N_blades_diffuser=17,
        imp_blade_thickness_inlet=2e-3,
        imp_blade_thickness_outlet=2e-3,
        diff_blade_thickness_inlet=2e-3,
        diff_blade_thickness_outlet=6e-3,
        roughness=3.6e-6,
        beta2_blade=-40.0,
        beta1_blade_rms=-50.0,
        alpha1=0.0,
        alpha3_blade=70.0,
        alpha4_blade=65.0,
        r2=r2,
        r4=1.5 * r2,
        r1_hub=0.3 * r2,
        r1_shroud=0.7 * r2,
        b2=0.17 * r2,
    )
    return g

# --- Utility: ensure list of 4 ---

def _ensure_len4(x: Iterable[float] | None, default_val: float) -> Tuple[float, float, float, float]:
    if x is None:
        return (default_val, default_val, default_val, default_val)
    arr = list(x)
    if len(arr) != 4:
        raise ValueError("Expected 4 values.")
    return tuple(float(v) for v in arr)

def _ensure_length(x, n_expected: int, default_val: float) -> tuple[float, ...]:
    """Ensure the input has length n_expected; otherwise fill with default_val."""
    if x is None:
        return tuple([default_val] * n_expected)
    arr = list(x)
    if len(arr) != n_expected:
        raise ValueError(f"Expected {n_expected} values, got {len(arr)}.")
    return tuple(float(v) for v in arr)


# --- Main function ---

def simulate_compressor_series(
        *,
        n_stages: int = 4,
        mdot: float = 300.0,
        base_geometry: Optional[Dict[str, Any]] = None,
        # Either phi_list or r2_list (but not both)
        phi_list: Optional[Iterable[float]] = None,
        r2_list: Optional[Iterable[float]] = None,
        # Inlet total pressure to stage 1
        p01_0: float = 60e5,
        # Intercooler: total inlet temperatures per stage
        T01_list: Optional[Iterable[float]] = None,
        # Either N_list [rpm] or Ma_list (tip Mach) (but not both)
        N_list: Optional[Iterable[float]] = None,
        Ma_list: Optional[Iterable[float]] = None,
        # Fluid
        fluid: str = "CO2",
        backend: str = "auto",
        show_figure: bool = True,
) -> Dict[str, Any]:
    """Simulate four centrifugal compressors in series with intercooling.

    See module docstring for details.
    """
    # --- validate exclusivity ---
    if (phi_list is None) == (r2_list is None):
        raise ValueError("Provide either phi_list OR r2_list (exclusively).")
    if (N_list is None) == (Ma_list is None):
        # both None -> default to Mach; both not None -> invalid
        if N_list is None and Ma_list is None:
            Ma_list = (0.8, 0.8, 0.8, 0.8)
        else:
            raise ValueError("Provide either N_list OR Ma_list (exclusively).")

    phi_list = None if phi_list is None else _ensure_length(phi_list, n_stages, 0.0)
    r2_list  = None if r2_list  is None else _ensure_length(r2_list, n_stages, 0.0)
    T01_list = _ensure_length(T01_list, n_stages, 35.0 + 273.15)
    N_list   = None if N_list is None else _ensure_length(N_list, n_stages, 0.0)
    Ma_list  = None if Ma_list is None else _ensure_length(Ma_list, n_stages, 0.0)

    # --- backend and fluid key ---
    backend, note = _select_backend(fluid, backend)

    # Storage per stage
    eta_is_ts = np.zeros(n_stages)
    PR_ts = np.zeros(n_stages)
    dh0 = np.zeros(n_stages)
    W = np.zeros(n_stages)
    Wdot = np.zeros(n_stages)
    # useful derived
    in_p01 = np.zeros(n_stages)   # track inlet total pressure to each stage
    in_rho01 = np.zeros(n_stages)  # track inlet total density to each stage
    out_p05 = np.zeros(n_stages)
    out_T05 = np.zeros(n_stages)
    used_phi = np.zeros(n_stages)
    used_r2 = np.zeros(n_stages)
    used_u2 = np.zeros(n_stages)
    used_rpm = np.zeros(n_stages)

    p01 = float(p01_0)

    for i in range(n_stages):
        T01 = float(T01_list[i])
        in_p01[i] = p01
        AS = CP.AbstractState(backend, fluid)
        AS.update(CP.PT_INPUTS, p01, T01)
        rho_01 = AS.rhomass()
        in_rho01[i] = rho_01
        a_01 = AS.speed_sound()

        # Determine u2 and rpm depending on provided set (Ma vs N)
        if Ma_list is not None:
            Ma = float(Ma_list[i])
            u2 = Ma * a_01
            # r2 will be needed to get rpm later
            omega = None  # compute after r2 is known
        else:
            N = float(N_list[i])
            omega = 2.0 * math.pi * N / 60.0
            u2 = None  # depends on r2

        # Determine r2 and phi depending on which set is provided
        if phi_list is not None:
            phi = float(phi_list[i])
            if Ma_list is not None:
                # r2 = 0.5 * sqrt( mdot / (phi * rho_01 * u2) )
                r2 = 0.5 * math.sqrt(mdot / (phi * rho_01 * u2))
            else:
                # r2 = ( mdot / (8 * phi * rho_01 * omega) )^(1/3)
                r2 = (mdot / (8.0 * phi * rho_01 * omega)) ** (1.0 / 3.0)
                u2 = omega * r2
            # rpm from u2 and r2
            rpm = (u2 / r2) * 60.0 / (2.0 * math.pi)
        else:
            # r2 is provided; compute phi
            r2 = float(r2_list[i])
            if Ma_list is not None:
                rpm = (u2 / r2) * 60.0 / (2.0 * math.pi)
            else:
                u2 = omega * r2
                rpm = N_list[i]
            phi = mdot / (rho_01 * u2 * (2.0 * r2) ** 2)

        # Build geometry for this stage (base + r2-scaled defaults)
        geom_default = _default_geometry_for_r2(r2)
        geom = dict(geom_default)
        if base_geometry:
            # apply user base geometry overrides that are not r2-dependent;
            # if user supplied r2-dependent fields, keep compatibility by recomputing them here
            for k, v in base_geometry.items():
                # We allow user to override, but then ensure consistency for scaled fields
                geom[k] = v
            # enforce r2-coupled fields
            geom["r2"] = r2
            geom["r4"] = base_geometry.get("r4", 1.5 * r2)
            geom["r1_hub"] = base_geometry.get("r1_hub", 0.3 * r2)
            geom["r1_shroud"] = base_geometry.get("r1_shroud", 0.7 * r2)
            geom["b2"] = base_geometry.get("b2", 0.1 * r2)

        # Simulate this stage
        cmp = CentrifugalCompressor()
        cmp.set_fluid(fluid, backend)
        cmp.set_inlet_conditions(p01, T01)
        cmp.set_geometry(geom)
        cmp.set_operating_conditions(mdot, rpm)
        cmp.simulate(verbose=False)

        # Collect stage results
        eta_is_ts[i] = float(cmp.eta_is_ts)
        PR_ts[i] = float(cmp.PR_ts)
        dh0[i] = float(cmp.h_05 - cmp.h_01)
        W[i] = float(cmp.W)
        Wdot[i] = float(cmp.Wdot)
        out_p05[i] = float(cmp.p_05)
        out_T05[i] = float(cmp.T_05)
        used_phi[i] = phi
        used_r2[i] = r2
        used_u2[i] = u2
        used_rpm[i] = rpm

        # Chain pressure to next stage (total pressure)
        p01 = out_p05[i]

    results = {
        "inputs": {
            "mdot": mdot,
            "p01_0": p01_0,
            "T01_list": T01_list,
            "phi_list": phi_list,
            "r2_list": r2_list,
            "N_list": N_list,
            "Ma_list": Ma_list,
            "fluid": fluid,
            "backend": backend,
            "backend_note": note,
        },
        "stage": {
            "eta_is_ts": eta_is_ts,
            "PR_ts": PR_ts,
            "dh0": dh0,
            "W": W,
            "Wdot": Wdot,
            "p01": in_p01,
            "p05": out_p05,
            "T01": T01_list,
            "T05": out_T05,
            "rho_01": in_rho01,
            "phi": used_phi,
            "r2": used_r2,
            "u2": used_u2,
            "rpm": used_rpm,
        },
        "overall": {
            "PR_tt_series": float(out_p05[-1] / p01_0), # cumulative total-pressure ratio
            "p_in": float(in_p01[0]),
            "p_out": float(out_p05[-1]),
            "W": np.sum(W),
            "Wdot": np.sum(Wdot),

        },
    }

    if show_figure:
        dome_ts_path = r"C:\Users\guido\OneDrive - University of Pisa\01_conferenze\2026_ASME\Analisi\allam_compressor\results\dome_ts_one_compressor_series.png"
        plot_ts_dome_compressor_series(in_p01=in_p01, out_p05=out_p05, in_T01=T01_list, out_T05=out_T05, fluid=fluid,
                                       backend=backend, show_isobars=True, N_iso=30, save_path=dome_ts_path)

    return results


# --- Convenience example (quick smoke test) ---
if __name__ == "__main__":
    res = simulate_compressor_series(n_stages = 5,
                                     mdot=1592.0,
                                     phi_list=[0.05] * 5,
                                     p01_0=30e5,
                                     T01_list=[35 + 273.15] * 5,
                                     Ma_list=[0.7] * 5,
                                     fluid="CO2",
                                     backend="auto",
                                     show_figure=True)

    print("Stage eta_is_ts:", res["stage"]["eta_is_ts"])
    print("Stage PR_ts:", res["stage"]["PR_ts"])
    print("Stage Δh0 [kJ/kg]:", res["stage"]["dh0"]/1000.0)
    print("Stage kRPM [-]:", res["stage"]["rpm"]/1000.0)
    print("Overall PR_tt (p05_4 / p01_0):", res["overall"]["PR_tt_series"]) 
