import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP


def _saturation_dome_TS_from_AS(AS, n=150):
    """
    Return arrays of T (K), s_liq (J/kg-K), s_vap (J/kg-K) along the saturation line
    using the provided AbstractState.
    """
    T_triple = AS.Ttriple()
    T_crit = AS.T_critical()
    T_sat = np.linspace(T_triple + 1e-6, T_crit - 1e-10, n)

    s_liq, s_vap = np.empty_like(T_sat), np.empty_like(T_sat)
    for i, T in enumerate(T_sat):
        try:
            AS.update(CP.QT_INPUTS, 0.0, float(T))
            s_liq[i] = AS.smass()
        except Exception:
            s_liq[i] = np.nan
        try:
            AS.update(CP.QT_INPUTS, 1.0, float(T))
            s_vap[i] = AS.smass()
        except Exception:
            s_vap[i] = np.nan
    return T_sat, s_liq, s_vap, T_triple, T_crit


def _critical_isobar_TS_from_AS(AS, T_max, n=150):
    """
    Return arrays for the critical isobar (P=Pc) in T–s coordinates using AS.
    """
    T_crit = AS.T_critical()
    P_crit = AS.p_critical()
    T_iso = np.linspace(max(T_crit * 0.75, AS.Ttriple() + 1e-3), T_max, n)
    s_iso = np.empty_like(T_iso)
    for i, T in enumerate(T_iso):
        try:
            AS.update(CP.PT_INPUTS, P_crit, float(T))
            s_iso[i] = AS.smass()
        except Exception:
            s_iso[i] = np.nan
    # Critical-point entropy
    try:
        AS.update(CP.PT_INPUTS, P_crit, T_crit)
        s_crit = AS.smass()
    except Exception:
        s_crit = np.nan
    return T_iso, s_iso, T_crit, P_crit, s_crit


def _twophase_mask_from_AS(AS, S_grid, T_vec, Tcrit):
    """
    Fast mask for two-phase region in T–s plane:
    for each row at fixed T<Tc, mark s between s_liq(T) and s_vap(T).
    """
    s_liq_T = np.full_like(T_vec, np.nan, dtype=float)
    s_vap_T = np.full_like(T_vec, np.nan, dtype=float)
    T_triple = AS.Ttriple()

    # Collect saturation entropies row-wise; no global nanmin/nanmax
    for i, T in enumerate(T_vec):
        if (T > T_triple) and (T < Tcrit):
            try:
                AS.update(CP.QT_INPUTS, 0.0, float(T))
                s_liq_T[i] = AS.smass()
            except Exception:
                pass
            try:
                AS.update(CP.QT_INPUTS, 1.0, float(T))
                s_vap_T[i] = AS.smass()
            except Exception:
                pass

    # For rows where both are finite, define [low, high]; else mark as invalid
    valid_row = np.isfinite(s_liq_T) & np.isfinite(s_vap_T)
    s_low = np.where(valid_row, np.minimum(s_liq_T, s_vap_T), np.nan)
    s_high = np.where(valid_row, np.maximum(s_liq_T, s_vap_T), np.nan)

    s_low_grid = s_low[:, None]
    s_high_grid = s_high[:, None]

    eps = 1e-9
    base = (T_vec[:, None] < Tcrit - eps)
    # Apply mask only on rows where we have valid saturation bounds
    mask = base & np.isfinite(s_low_grid) & np.isfinite(s_high_grid) & (S_grid >= s_low_grid - eps) & (
                S_grid <= s_high_grid + eps)
    return mask


def _alpha_from_TS_grid(AS, T_grid, S_grid):
    """
    Compute isobaric expansion coefficient alpha(T,s) [1/K] on a T–S grid via AS.
    """
    nT, nS = T_grid.shape
    A = np.full((nT, nS), np.nan, dtype=float)

    for i in range(nT):
        for j in range(nS):
            T = float(T_grid[i, j])
            s = float(S_grid[i, j])
            try:
                AS.update(CP.SmassT_INPUTS, s, T)
                A[i, j] = AS.isobaric_expansion_coefficient()
            except Exception:
                A[i, j] = np.nan
    return A


def _estimate_entropy_window_from_P_bounds(AS, T_min, T_max, P_min, P_max, nT_probe=60):
    """
    Probe s(T,P) at the two pressure bounds across T_min..T_max to infer a good entropy window.
    Returns (s_min, s_max).
    """
    T_vec = np.linspace(T_min, T_max, nT_probe)
    s_vals = []
    for T in T_vec:
        for P in (P_min, P_max):
            try:
                AS.update(CP.PT_INPUTS, float(P), float(T))
                s_vals.append(AS.smass())
            except Exception:
                pass
    if len(s_vals) == 0:
        # Fallback: try along the critical isobar within the range
        try:
            Pcrit = AS.p_critical()
            for T in T_vec:
                AS.update(CP.PT_INPUTS, Pcrit, float(T))
                s_vals.append(AS.smass())
        except Exception:
            pass
    if len(s_vals) == 0:
        # As a last resort, use saturation line in-range temperatures
        T_triple, T_crit = AS.Ttriple(), AS.T_critical()
        T_probe = np.linspace(max(T_triple, T_min), min(T_crit, T_max), max(5, nT_probe // 4))
        for T in T_probe:
            for Q in (0.0, 1.0):
                try:
                    AS.update(CP.QT_INPUTS, Q, float(T))
                    s_vals.append(AS.smass())
                except Exception:
                    pass
    if len(s_vals) == 0:
        raise RuntimeError("Could not determine entropy bounds from the provided T/P range.")
    s_min = float(np.nanmin(s_vals))
    s_max = float(np.nanmax(s_vals))
    # Add a small padding
    pad = 0.03 * (s_max - s_min if s_max > s_min else max(abs(s_min), 1.0))
    return s_min - pad, s_max + pad


def plot_alpha_TS_from_AS(
        AS,
        T_min, T_max,
        P_min, P_max,
        nT=240, nS=360,
        alpha_thresh=0.10,
        cmap_fill=True,
        show=True
):
    """
    Plot alpha(T,s) >= alpha_thresh in T–s plane using a CoolProp AbstractState.

    Parameters
    ----------
    AS : CoolProp.AbstractState
        Pre-constructed AbstractState (e.g., CP.AbstractState('HEOS', 'CO2')).
    T_min, T_max : float [K]
        Temperature bounds for the plot window.
    P_min, P_max : float [Pa]
        Pressure bounds (used only to infer a sensible entropy window).
    nT, nS : int
        Grid resolution in T and s, respectively.
    alpha_thresh : float [1/K]
        The contour threshold for alpha.
    cmap_fill : bool
        If True, lightly fill the region where alpha >= threshold.
    show : bool
        If True, call plt.show(). The function always returns (fig, ax).
    """
    # --- Saturation dome ---
    T_sat, s_liq, s_vap, T_triple, T_crit = _saturation_dome_TS_from_AS(AS)

    # --- Entropy window inferred from the P-bounds ---
    s_lo, s_hi = _estimate_entropy_window_from_P_bounds(AS, T_min, T_max, P_min, P_max)

    # --- Regular grids ---
    T_vec = np.linspace(T_min, T_max, nT)
    s_vec = np.linspace(s_lo, s_hi, nS)
    S_grid, T_grid = np.meshgrid(s_vec, T_vec)

    # --- Alpha on the grid (TS inputs) ---
    ALPHA = _alpha_from_TS_grid(AS, T_grid, S_grid)

    # --- Mask out two-phase region (under the dome) ---
    mask_2ph = _twophase_mask_from_AS(AS, S_grid, T_vec, T_crit)
    ALPHA = np.where(mask_2ph, np.nan, ALPHA)

    # --- Critical isobar & point ---
    T_iso, s_iso, Tcrit, Pcrit, s_crit = _critical_isobar_TS_from_AS(AS, T_max)

    # --- Celsius for plotting only ---
    T_grid_C = T_grid - 273.15
    T_sat_C = T_sat - 273.15
    T_iso_C = T_iso - 273.15
    Tcrit_C = Tcrit - 273.15
    T_lo_C = T_min - 273.15
    T_hi_C = T_max - 273.15

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 6), dpi=90)

    finite = np.isfinite(ALPHA)
    has_high = np.any(finite) and (np.nanmax(ALPHA[finite]) >= alpha_thresh)

    if has_high:
        if cmap_fill:
            ax.contourf(
                S_grid / 1000.0, T_grid_C, ALPHA,
                levels=[alpha_thresh, np.nanmax(ALPHA[finite])],
                alpha=0.25
            )
        ax.contour(
            S_grid / 1000.0, T_grid_C, ALPHA,
            levels=[alpha_thresh],
            colors='crimson', linewidths=2.3
        )
    else:
        if np.any(finite):
            level = float(np.nanpercentile(ALPHA[finite], 98))
            ax.contour(S_grid / 1000.0, T_grid_C, ALPHA, levels=[level], colors='crimson', linewidths=2.3)
            ax.text(0.02, 0.98, rf'No α ≥ {alpha_thresh:g} 1/K; showing ~98th pct ({level:.3g})',
                    transform=ax.transAxes, va='top', ha='left',
                    bbox=dict(boxstyle='round', fc='w', alpha=0.75))
        else:
            ax.text(0.02, 0.98, 'No valid α values in this window',
                    transform=ax.transAxes, va='top', ha='left',
                    bbox=dict(boxstyle='round', fc='w', alpha=0.75))

    # --- Saturation dome ---
    ax.plot(s_liq / 1000.0, T_sat_C, 'krpm-', lw=2.0, label='Saturation dome')
    ax.plot(s_vap / 1000.0, T_sat_C, 'krpm-', lw=2.0)

    # --- Critical isobar & point ---
    ax.plot(s_iso / 1000.0, T_iso_C, ls='-.', lw=1.6, color='black', label=r'Critical isobar  $P=P_c$')
    ax.plot([s_crit / 1000.0], [Tcrit_C], 'o', ms=6, color='red', label='Critical point')

    # --- Labels & cosmetics ---
    ax.set_xlabel(r'Entropy  $s$  [kJ/(kg\cdot K)]')
    ax.set_ylabel(r'Temperature  $T$  [°C]')
    ax.set_xlim(s_lo / 1000.0, s_hi / 1000.0)
    ax.set_ylim(T_lo_C, T_hi_C)
    ax.grid(alpha=0.25)
    ax.legend(loc='upper left', framealpha=0.85)
    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


# ------ Example usage (only runs if this file is executed directly) ------
if __name__ == "__main__":
    # Build an AbstractState for CO2
    AS = CP.AbstractState("HEOS", "CO2")
    # Plot window (example): T in [283.15, Tcrit+45 K], P in [60 bar, 90 bar]
    Tcrit = AS.T_critical()
    fig, ax = plot_alpha_TS_from_AS(
        AS,
        T_min=max(283.15, AS.Ttriple() + 0.25),  # ~10 °C
        T_max=Tcrit + 45.0,
        P_min=60e5,  # 60 bar
        P_max=90e5,  # 90 bar
        nT=240, nS=360,
        alpha_thresh=0.10,
        cmap_fill=True,
        show=True
    )