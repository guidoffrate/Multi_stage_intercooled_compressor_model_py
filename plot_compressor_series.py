
import re
import os
from typing import Union, Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_compressor_series_summary(
    csv_or_df: Union[str, pd.DataFrame],
    outdir: str = ".",
    series_label_col: str = "Series",
    pr_regex: str = r"^PR_ts_stage\d+$",
    eta_regex: str = r"^eta_is_stage\d+$",
    font_size: int = 24,
    spine_width: float = 1.4,
    figsize=(12, 7),
    dpi: int = 300,
    legend_title: str = "Inlet temperature series",
    prefix: str = "",
) -> Dict[str, Any]:
    """
    Create grouped bar charts for stage pressure ratio and isentropic efficiency
    from a compressor-series summary table, plus a 3-panel overview.

    Required columns (exact names) in the CSV/DataFrame:
      - 'Series' (string, includes inlet temperature, e.g., "34 °C (off-design)")
      - 'PR_tt_overall' (overall total-total PR)
      - 'Wdot_MW' (overall power in MW)
      - 'p_out_bar' (outlet total pressure in bar)
      - Stage columns matching regex: PR_ts_stageN and eta_is_stageN

    Styling: 24 pt fonts and 1.4 pt spine width.
    """

    # ---- Load data ----
    if isinstance(csv_or_df, str):
        df = pd.read_csv(csv_or_df)
    else:
        df = csv_or_df.copy()

    # ---- Validate required columns ----
    required_cols = {"Series", "PR_tt_overall", "Wdot_MW", "p_out_bar"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # ---- Detect stage columns strictly by regex ----
    stage_pr_cols = sorted([c for c in df.columns if re.match(pr_regex, c)],
                           key=lambda x: int(re.findall(r"\d+", x)[0]))
    stage_eta_cols = sorted([c for c in df.columns if re.match(eta_regex, c)],
                            key=lambda x: int(re.findall(r"\d+", x)[0]))
    if not stage_pr_cols or not stage_eta_cols:
        raise KeyError("No stage columns found. Expect names like 'PR_ts_stage1' and 'eta_is_stage1'.")

    n_stages = len(stage_pr_cols)
    series_labels = df[series_label_col].astype(str).tolist()
    n_series = len(series_labels)

    PR = df[stage_pr_cols].to_numpy(dtype=float)   # (n_series, n_stages)
    ETA = df[stage_eta_cols].to_numpy(dtype=float) # (n_series, n_stages)

    # ---- Styling helper ----
    def style_axes(ax):
        ax.tick_params(axis='both', which='both', labelsize=font_size, width=spine_width)
        for side in ["left", "right", "bottom", "top"]:
            ax.spines[side].set_linewidth(spine_width)

    # ---- Global rc ----
    plt.rcParams.update({
        "font.size": font_size,
        "axes.labelsize": font_size * 0.85,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": int(font_size * 0.85),
    })

    x = np.arange(1, n_stages + 1)  # stage numbers start at 1
    group_width = 0.8
    bar_w = group_width / max(n_series, 1)
    offsets = (np.arange(n_series) - (n_series - 1) / 2.0) * bar_w  # centered groups

    os.makedirs(outdir, exist_ok=True)
    prefix = (prefix + "_") if prefix else ""

    # ---- Plot PR ----
    fig1, ax1 = plt.subplots(figsize=figsize, dpi=dpi)
    for i in range(n_series):
        ax1.bar(x + offsets[i], PR[i], width=bar_w, align='center',
                label=series_labels[i], edgecolor='black', linewidth=1.0)

    ax1.set_xlabel("Stage #")
    ax1.set_ylabel(r"Stage Pressure Ratio  $PR_{ts}$ [-]")
    ax1.set_xticks(x)
    ax1.legend(title=legend_title, frameon=False, ncol=1, loc="best")
    style_axes(ax1)
    fig1.tight_layout()
    pr_path = os.path.join(outdir, f"{prefix}pr_ts_by_stage.png")
    fig1.savefig(pr_path, bbox_inches="tight")
    plt.close(fig1)

    # ---- Plot eta ----
    fig2, ax2 = plt.subplots(figsize=figsize, dpi=dpi)
    for i in range(n_series):
        ax2.bar(x + offsets[i], ETA[i], width=bar_w, align='center',
                label=series_labels[i], edgecolor='black', linewidth=1.0)

    ax2.set_xlabel("Stage #")
    ax2.set_ylabel(r"Isentropic Efficiency  $\eta_{is,ts}$ [-]")
    ax2.set_xticks(x)
    ax2.legend(title=legend_title, frameon=False, ncol=1, loc="best")
    style_axes(ax2)
    fig2.tight_layout()
    eta_path = os.path.join(outdir, f"{prefix}eta_is_by_stage.png")
    fig2.savefig(eta_path, bbox_inches="tight")
    plt.close(fig2)

    # ---- Overview figure (3 panels) ----
    # Extract inlet temperatures from 'Series', e.g., "34 °C (off-design)"
    labels = df[series_label_col].astype(str).tolist()
    temps = []
    for s in labels:
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*°?\s*C", s)
        if not m:
            raise ValueError(f"Could not parse inlet temperature from Series label: '{s}'")
        temps.append(float(m.group(1)))

    overall_PR = df["PR_tt_overall"].to_numpy(dtype=float)
    overall_P_MW = df["Wdot_MW"].to_numpy(dtype=float)      # use directly (MW)
    p_out_bar = df["p_out_bar"].to_numpy(dtype=float)       # use directly (bar)
    Wdot_cycle_net_MW = df["Wdot_cycle_net_MW"].to_numpy(dtype=float)
    Wdot_cycle_norm = Wdot_cycle_net_MW / Wdot_cycle_net_MW[0]  # Normliased for the design power output

    fig3, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=dpi, constrained_layout=True)
    panels = [
        ("Overall Power", r"$\dot{W}$ [MW]", overall_P_MW),
        ("Overall Pressure Ratio", r"$PR_{series}$ [-]", overall_PR),
        ("Outlet Pressure", r"$p_{out}$ [bar]", p_out_bar),
        ("Cycle Norm. Power Output", r"$\dot{W}/\dot{W}_{des}$ [-]", Wdot_cycle_norm),
    ]

    for ax, (title, ylabel, data) in zip(axes.flat, panels):
        ax.bar(range(len(temps)), data, edgecolor="black", linewidth=1.0)
        ax.set_title(title,fontsize=font_size * 0.85)
        ax.set_xlabel("Inlet temperature [°C]")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(temps)))
        xtlbls = [f"{int(t)}" if float(t).is_integer() else f"{t:.1f}" for t in temps]
        ax.set_xticklabels(xtlbls)
        ax.tick_params(axis='both', which='both', labelsize=font_size, width=spine_width)

        # Increment y ticks and add a grid
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6))
        ax.grid(axis="y")

        # Spine width
        for side in ["left", "right", "bottom", "top"]:
            ax.spines[side].set_linewidth(spine_width)

        # Ax aspect ratio
        # ax.set_box_aspect(1)

    overview_path = os.path.join(outdir, f"{prefix}overall_PR_Wdot_Pout_WdotCycle.png")
    fig3.savefig(overview_path, bbox_inches="tight")
    plt.close(fig3)

    return {
        "paths": {"pr": pr_path, "eta": eta_path, "overview": overview_path},
        "n_stages": n_stages,
        "n_series": n_series,
        "stage_pr_cols": stage_pr_cols,
        "stage_eta_cols": stage_eta_cols,
        "series_labels": series_labels,
    }


# ---------------------------------------------------------------------------
# Standalone execution example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    csv_path = "compressor_series_pictures/compressor_series_summary.csv"  # expects CSV next to this script
    outdir = "./compressor_series_pictures"
    os.makedirs(outdir, exist_ok=True)
    res = plot_compressor_series_summary(csv_path, outdir=outdir)
    print("--- Saved plots ---")
    for k, v in res["paths"].items():
        print(f"{k}: {v}")
