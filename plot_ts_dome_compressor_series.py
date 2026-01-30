import numpy as np
import CoolProp.CoolProp as CP
from dvdt_contour import plot_alpha_TS_from_AS
from matplotlib import pyplot as plt
from typing import Iterable, Optional, Tuple, Dict, Any
from typing import Sequence, List

def init_ts_dome_axes(
    *,
    fluid: str,
    backend: str,
    T_min: Optional[float] = None,
    T_max: Optional[float] = None,
    P_min: Optional[float] = None,
    P_max: Optional[float] = None,
    nT: int = 400,
    nS: int = 400,
    alpha_thresh: float = 0.10,
    cmap_fill: bool = True,
) -> Tuple[plt.Figure, plt.Axes, CP.AbstractState, Dict[str, Any]]:
    """
    Create a T–s dome background once. Returns (fig, ax, AS, meta).
    If min/max bounds are not given, they will be inferred from the fluid's critical/triple data.
    """
    AS = CP.AbstractState(backend, fluid)

    # Reasonable default bounds if not provided
    Tcrit = AS.T_critical()
    Tmin_default = AS.Ttriple() + 50.0
    Tmax_default = Tcrit + 50.0

    T_min = float(T_min if T_min is not None else Tmin_default)
    T_max = float(T_max if T_max is not None else Tmax_default)

    # Pressure bounds are only used to label/size the dome canvas in plot_alpha_TS_from_AS;
    # they do not draw isobars by themselves.
    if P_min is None or P_max is None:
        # choose something broad if not provided
        P_min = float(max(1e4, AS.p_triple() * 1.05))  # avoid exactly at triple
        P_max = float(min(AS.p_critical() * 3.0, 3.0e8))

    fig, ax = plot_alpha_TS_from_AS(
        AS,
        T_min=T_min,
        T_max=T_max,
        P_min=P_min,
        P_max=P_max,
        nT=nT,
        nS=nS,
        alpha_thresh=alpha_thresh,
        cmap_fill=cmap_fill,
        show=False,
    )

    meta = dict(T_min=T_min, T_max=T_max, P_min=P_min, P_max=P_max)
    return fig, ax, AS, meta


def _plot_isobars_dedup(
    ax: plt.Axes,
    AS: CP.AbstractState,
    pressures: Iterable[float],
    T_min: float,
    T_max: float,
    *,
    linestyle: str = "--",
    linewidth: float = 0.9,
    alpha: float = 0.7,
    color: str = "grey",
) -> None:
    """Draw deduplicated constant-P lines between T_min and T_max."""
    all_P = np.sort(np.array(list(pressures), dtype=float))
    unique_P = []
    tol_rtol = 1e-6
    for P in all_P:
        if not unique_P or not np.isclose(P, unique_P[-1], rtol=tol_rtol, atol=0.0):
            unique_P.append(P)

    for P in unique_P:
        s_col, T_col = [], []
        try:
            AS.update(CP.PT_INPUTS, float(P), T_min); s_min = AS.smass()
            AS.update(CP.PT_INPUTS, float(P), T_max); s_max = AS.smass()
            for s in np.linspace(s_min, s_max, 100):
                AS.update(CP.PSmass_INPUTS, float(P), s)
                s_col.append(AS.smass() / 1000.0)   # kJ/kg/K
                T_col.append(AS.T() - 273.15)       # °C
        except Exception:
            continue
        if len(s_col) >= 2:
            ax.plot(s_col, T_col, linestyle=linestyle, linewidth=linewidth, alpha=alpha, color=color)


def plot_ts_dome_compressor_series(
    *,
    in_p01: Iterable[Iterable[float]],
    out_p05: Iterable[Iterable[float]],
    in_T01: Iterable[Iterable[float]],
    out_T05: Iterable[Iterable[float]],
    fluid: str,
    backend: str,
    show_isobars: bool = True,
    N_iso: int = 30,
    save_path: Optional[str] = None,
    labels: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[str]] = None,
    show_markers: bool = True,
    add_legend: bool = True,
):
    """
    Vectorized convenience wrapper.
    Inputs are 2-D (rows = different compressor trains, cols = stages).
    Draws the T–s dome once, optional grey isobars once, then overlays all trains.

    Returns (fig, ax, series_data_list)
    """
    # Convert to 2-D arrays
    in_p01   = np.atleast_2d(np.asarray(in_p01, dtype=float))
    out_p05  = np.atleast_2d(np.asarray(out_p05, dtype=float))
    in_T01 = np.atleast_2d(np.asarray(in_T01, dtype=float))
    out_T05  = np.atleast_2d(np.asarray(out_T05, dtype=float))

    n_trains, n_stages = in_p01.shape
    assert out_p05.shape == (n_trains, n_stages), "out_p05 shape mismatch"
    assert in_T01.shape == (n_trains, n_stages), "T01_list shape mismatch"
    assert out_T05.shape == (n_trains, n_stages), "out_T05 shape mismatch"

    # Global bounds for the dome
    AS_tmp = CP.AbstractState(backend, fluid)
    Tcrit  = AS_tmp.T_critical()
    Tmin   = float(min(np.min(in_T01), AS_tmp.Ttriple() + 50.0))
    Tmax   = float(max(np.max(out_T05),  Tcrit + 50.0))
    Pmin   = float(np.nanmin(in_p01))
    Pmax   = float(np.nanmax(out_p05))

    # Create the dome once
    fig, ax, AS, _ = init_ts_dome_axes(
        fluid=fluid, backend=backend,
        T_min=Tmin, T_max=Tmax, P_min=Pmin, P_max=Pmax,
        nT=400, nS=400, alpha_thresh=0.10, cmap_fill=True
    )

    # Optional grey isobars (dedup across ALL trains)
    if show_isobars:
        all_pressures = list(in_p01.ravel()) + list(out_p05.ravel())
        _plot_isobars_dedup(ax, AS, all_pressures, T_min=Tmin, T_max=Tmax)

    # Default labels/colors
    if labels is None:
        labels = [f"Series {k+1}" for k in range(n_trains)]
    if colors is None:
        # cycle through default Matplotlib prop cycle
        prop_cycle = plt.rcParams.get("axes.prop_cycle")
        base_colors: List[str] = (prop_cycle.by_key().get("color", []) or
                                  ["tab:blue", "tab:orange", "tab:green", "tab:red",
                                   "tab:purple", "tab:brown", "tab:pink", "tab:gray",
                                   "tab:olive", "tab:cyan"])
        # repeat if needed
        reps = int(np.ceil(n_trains / len(base_colors)))
        colors = (base_colors * reps)[:n_trains]

    series_data: List[Dict[str, np.ndarray]] = []

    # Small helper to compute one series' path (same logic as your single-series)
    def _compute_series_paths(AS_loc, inP, outP, Tin, Tout):
        def _s_kJ_per_kgK(P, T):
            try:
                AS_loc.update(CP.PT_INPUTS, float(P), float(T))
                return AS_loc.smass() / 1000.0
            except Exception:
                return None

        s01 = []; T01C = []; s05 = []; T05C = []
        for i in range(n_stages):
            s_in  = _s_kJ_per_kgK(inP[i],  Tin[i])
            s_out = _s_kJ_per_kgK(outP[i], Tout[i])
            if s_in is None or s_out is None:
                continue
            s01.append(s_in);              T01C.append(Tin[i]  - 273.15)
            s05.append(s_out);             T05C.append(Tout[i] - 273.15)

        path_s = []; path_T = []
        for i in range(len(s01)):  # len may be < n_stages if some states failed
            if i == 0:
                path_s.append(s01[i]); path_T.append(T01C[i])
            path_s.append(s05[i]);     path_T.append(T05C[i])

            if i < len(s01) - 1:
                P_iso   = outP[i]
                T_start = Tout[i]
                T_end   = Tin[i+1]
                T_samples = np.linspace(T_start, T_end, N_iso + 2)[1:-1]
                for T_iso in T_samples:
                    try:
                        AS_loc.update(CP.PT_INPUTS, float(P_iso), float(T_iso))
                        s_iso = AS_loc.smass() / 1000.0
                        path_s.append(s_iso); path_T.append(T_iso - 273.15)
                    except Exception:
                        pass
                path_s.append(s01[i+1]); path_T.append(T01C[i+1])

        return dict(
            path_s=np.array(path_s), path_T=np.array(path_T),
            s01=np.array(s01), T01C=np.array(T01C),
            s05=np.array(s05), T05C=np.array(T05C)
        )

    # Plot each train
    for k in range(n_trains):
        data_k = _compute_series_paths(
            AS, in_p01[k, :], out_p05[k, :], in_T01[k, :], out_T05[k, :]
        )
        series_data.append(data_k)

        ax.plot(data_k["path_s"], data_k["path_T"], '-', lw=2.0,
                color=colors[k], label=labels[k])
        if show_markers:
            ax.plot(data_k["s01"],  data_k["T01C"], 'o', ms=5.5, color=colors[k], alpha=0.9)
            ax.plot(data_k["s05"],  data_k["T05C"], 's', ms=5.5, color=colors[k], alpha=0.9)

    if add_legend:
        ax.legend(loc='upper left', framealpha=0.85)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=400)

    return fig, ax, series_data
