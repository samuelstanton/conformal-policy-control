import os
import csv

import numpy as np
import matplotlib.pyplot as plt

BLUE = "#2166ac"   # CPC+CBO
RED = "#d73027"    # CBO (no CPC)
TEAL = '#5ab4ac'           # Calibration data (complements blue/red)


def _method_style(method):
    """Return a dict of plot kwargs (color, marker, label, alpha) for a method name."""
    if method == "Uniform Random":
        return {"color": "gray", "marker": "s", "label": "Uniform Random", "alpha": 0.8}
    if method == "Gaussian Random":
        return {"color": TEAL, "marker": "^", "label": "Gaussian Random (Safe Policy)", "alpha": 0.9}
    if method.startswith("CPC+CBO"):
        alpha_str = method.split("alpha=")[-1] if "alpha=" in method else ""
        return {"color": BLUE, "marker": "o",
                "label": "CPC ("+ r"$\alpha$=" + f"{alpha_str}) + Constrained Bayes Opt,", "alpha": 1.0}
    if method == "CBO (no CPC)":
        return {"color": RED, "marker": "X", "label": "Constrained Bayes Opt (no CPC)", "alpha": 1.0}
    # Fallback for any unrecognized method name.
    return {"color": None, "marker": "o", "label": method, "alpha": 1.0}


def _ordered_methods(methods, include_gaussian=True):
    """Return the methods in a consistent plotting order, optionally dropping Gaussian Random."""
    preferred = ["Uniform Random", "Gaussian Random", "CBO (no CPC)"]
    cpc = sorted(m for m in methods if m.startswith("CPC+CBO"))
    ordered = [m for m in preferred if m in methods]
    others = [m for m in methods
              if m not in preferred and not m.startswith("CPC+CBO")]
    ordered = ordered + cpc + others
    if not include_gaussian:
        ordered = [m for m in ordered if m != "Gaussian Random"]
    return ordered


def _load_csv(path):
    """Read a long-format results CSV into a list of dict rows (values kept as strings)."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _series(rows, method, ycol, secol):
    """Extract (iteration, mean, se) arrays for a given method, sorted by iteration."""
    d = [r for r in rows if r["method"] == method]
    d.sort(key=lambda r: int(r["iteration"]))
    it = np.array([int(r["iteration"]) for r in d])
    y = np.array([float(r[ycol]) for r in d])
    se = np.array([float(r[secol]) for r in d])
    return it, y, se


def _savefig_tight_subplot(ax, path, pad_inches=0.1, dpi=300):
    """Save just the region of ``ax`` (labels, title, legend included) to ``path``."""
    parent_fig = ax.get_figure()
    parent_fig.canvas.draw()  # finalize text/legend layout before measuring
    renderer = parent_fig.canvas.get_renderer()
    tight_bbox = ax.get_tightbbox(renderer)
    extent = tight_bbox.transformed(parent_fig.dpi_scale_trans.inverted())
    parent_fig.savefig(path, bbox_inches=extent.padded(pad_inches), dpi=dpi)


def plot_results_from_csv(objective_csv, constraint_csv, include_gaussian=True,
                          show_se=True, save_dir=None):
    """Re-create the averaged-results figure from the CSVs written by cpc_cbo.py.

    Parameters
    ----------
    objective_csv, constraint_csv : str
        Paths to the ``Objective_*.csv`` and ``Constraint_*.csv`` files for one run.
    include_gaussian : bool
        If True, also plot the "Gaussian Random" baseline.
    show_se : bool
        If True, shade the +/- standard-error bands stored in the CSVs.
    save_dir : str or None
        If given, also save the two individual panels, the combined figure, and a
        standalone legend figure there.

    Returns
    -------
    fig, ax_con, ax_obj, fig_legend
    """
    label_fs = 18

    obj_rows = _load_csv(objective_csv)
    con_rows = _load_csv(constraint_csv)

    temperature = obj_rows[0]["temperature"]

    # optimal is constant across rows: best_obj_mean = optimal - regret_mean
    optimal = float(np.mean([float(r["best_obj_mean"]) + float(r["regret_mean"])
                             for r in obj_rows]))

    obj_methods = _ordered_methods(list(dict.fromkeys(r["method"] for r in obj_rows)),
                                   include_gaussian)
    con_methods = _ordered_methods(list(dict.fromkeys(r["method"] for r in con_rows)),
                                   include_gaussian)

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    # fig.suptitle(f"Temperature = {temperature}", fontsize=18, fontweight="bold")

    ax_con = axes[0]  # constraint panel (top)
    ax_obj = axes[1]  # objective panel (bottom)

    # band_note = f"shaded = +/- SE"

    # ----- Objective panel: best objective value found over time -----
    for method in obj_methods:
        it, mean, se = _series(obj_rows, method, "best_obj_mean", "best_obj_se")
        style = _method_style(method)
        ax_obj.plot(it, mean, linewidth=2.5, markersize=8,
                    color=style["color"], marker=style["marker"],
                    label=style["label"], alpha=style["alpha"])
        if show_se:
            ax_obj.fill_between(it, mean - se, mean + se, color=style["color"], alpha=0.2)

    # ax_obj.axhline(y=optimal, color="black", linestyle=":", alpha=0.75,
    #                label="True Optimum", linewidth=1.5)
    ax_obj.set_xlabel("Timestep of Policy Improvement", fontsize=label_fs)
    ax_obj.set_ylabel(r"Best Objective Value Found [$\rightarrow$]", fontsize=label_fs)
    ax_obj.set_title(f"Average Best Objective Value Over Time", fontsize=label_fs)
    ax_obj.grid(True, alpha=0.3)
    ax_obj.set_ylim([1.35, 1.7])
    ax_obj.spines[['top', 'right']].set_visible(False)


    # ----- Constraint panel: per-round constraint violations -----
    for method in con_methods:
        it, mean, se = _series(con_rows, method,
                               "violations_per_round_mean", "violations_per_round_se")
        style = _method_style(method)
        ax_con.plot(it, mean, linewidth=2.5, markersize=8,
                    color=style["color"], marker=style["marker"],
                    label=style["label"], alpha=style["alpha"])
        if show_se:
            ax_con.fill_between(it, mean - se, mean + se, color=style["color"], alpha=0.2)
        if method.startswith("CPC+CBO") and "alpha=" in method:
            alpha_val = float(method.split("alpha=")[-1])
            ax_con.axhline(y=alpha_val, color=style["color"], linestyle="--", linewidth=2.5,
                           label=r"$\alpha$=" + f"{alpha_val}")

    ax_con.set_xlabel("Iteration", fontsize=label_fs)
    ax_con.set_ylabel(r"Constraint Violations per Round [$\leftarrow$]", fontsize=label_fs)
    ax_con.set_title(f"Average Per-Round Constraint Violations", fontsize=label_fs)
    ax_con.grid(True, alpha=0.3)
    ax_con.set_ylim([0.0, 1.01])
    ax_con.spines[['top', 'right']].set_visible(False)
    ax_con.axhline(y=1.0, color="black", linestyle=":", alpha=0.75,label="B", linewidth=1.5)


    ax_con.tick_params(axis='both', which='major', labelsize=label_fs - 4)
    ax_obj.tick_params(axis='both', which='major', labelsize=label_fs - 4)

    plt.tight_layout()

    # ----- Standalone legend (drawn separately from the subplots) -----
    # Prefer constraint handles so the alpha threshold line is included; keep
    # first occurrence of each label so objective-only entries are still kept.
    legend_entries = {}
    for ax in (ax_obj, ax_con):
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in legend_entries:
                legend_entries[label] = handle
    handles = list(legend_entries.values())
    labels = list(legend_entries.keys())

    fig_legend = plt.figure(figsize=(4, 2.5))
    fig_legend.legend(handles, labels, loc="center", fontsize=label_fs - 2, frameon=True, ncols=6, reverse=True)
    fig_legend.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(objective_csv))[0]
        base = base[len("Objective_"):] if base.startswith("Objective_") else base
        _savefig_tight_subplot(ax_obj, os.path.join(save_dir, f"Objective_{base}.pdf"))
        _savefig_tight_subplot(ax_con, os.path.join(save_dir, f"Constraint_{base}.pdf"))
        # fig.savefig(os.path.join(save_dir, f"CPC_vs_CBO_{base}.pdf"), dpi=300)
        # Legend is shared across temperatures for a given alpha: drop any
        # "temp*" tokens so all temps overwrite the same Legend_*.pdf, while
        # different alphas still get distinct files.
        legend_base = "_".join(p for p in base.split("_") if not p.startswith("temp"))
        fig_legend.savefig(os.path.join(save_dir, f"Legend_{legend_base}.pdf"),
                           dpi=300, bbox_inches="tight")

    return fig, ax_con, ax_obj, fig_legend


def _savefig_tight_subplot(ax, path, pad_inches=0.2, dpi=300):
    """Save just the region of ``ax`` (labels, title, legend included) to ``path``.

    ``Axes.get_window_extent`` only covers the bare axes frame, so cropping to
    it (as ``bbox_inches=`` requires) cuts off axis labels, titles, and
    legends. ``Axes.get_tightbbox`` additionally accounts for those
    decorations, so use it instead to compute the crop box.
    """
    parent_fig = ax.get_figure()
    parent_fig.canvas.draw()  # ensure text/legend layout is up to date before measuring
    renderer = parent_fig.canvas.get_renderer()
    tight_bbox = ax.get_tightbbox(renderer)
    extent = tight_bbox.transformed(parent_fig.dpi_scale_trans.inverted())
    parent_fig.savefig(path, bbox_inches=extent.padded(pad_inches), dpi=dpi)
