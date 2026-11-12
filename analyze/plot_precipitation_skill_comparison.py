"""Plot saved annual LMR v2.1 versus LMR4D-Var metrics; no assimilation rerun."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from lmr4dvar.verification.precipitation_comparison import paired_skill_difference

PRODUCTS = ("GPCPv2.3", "20CRv3", "ERA5", "ERA20C")


def map_figure(panels, kind, interval, path):
    fig, axes = plt.subplots(4, 3, figsize=(16, 12), constrained_layout=True)
    images = {}
    for row, (product, (maps, scores, period)) in enumerate(panels.items()):
        for column, name in enumerate(("lmr", "fdvar", "difference")):
            limit = 0.5 if name == "difference" and kind == "correlation" else 1.0
            ax = axes[row, column]
            images[column] = ax.pcolormesh(
                maps.lon,
                maps.lat,
                maps[name],
                cmap="RdBu_r",
                shading="auto",
                vmin=-limit,
                vmax=limit,
            )
            key = {
                "lmr": "lmr_area_mean",
                "fdvar": "fdvar_area_mean",
                "difference": "difference_area_mean",
            }[name]
            label = {
                "lmr": "LMR v2.1",
                "fdvar": "LMR4D-Var",
                "difference": "4D-Var minus LMR v2.1",
            }[name]
            ax.set_title(f"{label}; area mean={scores[key]:.3f}", fontsize=11)
            if column == 0:
                ax.set_ylabel(f"{product}\n{period[0]}-{period[1]}\nLatitude")
            if row == 3:
                ax.set_xlabel("Longitude")
            ax.set_xlim(0, 360)
            ax.set_ylim(-90, 90)
            ax.set_xticks([0, 90, 180, 270, 360])
            ax.set_yticks([-60, 0, 60])
    for column in range(3):
        fig.colorbar(images[column], ax=axes[:, column].tolist(), shrink=0.85, extend="both")
    metric = (
        "Gridpoint temporal correlation" if kind == "correlation" else "Coefficient of efficiency"
    )
    fig.suptitle(
        f"Annual precipitation: {metric}; {interval} overlap\n"
        "Positive difference favors 4D-Var; saturated colors do not clip saved metrics",
        fontsize=14,
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def summary_figure(summary, interval, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    metrics = [
        ("gridpoint_temporal_r", "Area-mean temporal correlation R"),
        ("spatial_pattern_r", "Mean spatial anomaly-pattern correlation"),
        ("gridpoint_ce", "Area-mean local CE"),
        ("gridpoint_rmse_mm_day", "Area-mean local RMSE (mm/day)"),
    ]
    x = np.arange(len(PRODUCTS))
    labels = []
    for product in PRODUCTS:
        period = summary["comparisons"][f"{product}/djf_year_reference/{interval}"]["global"][
            "LMR4DVar"
        ]["period"]
        labels.append(f"{product}\n{period[0]}-{period[1]}")
    for ax, (metric, title) in zip(axes.flat, metrics, strict=True):
        for offset, estimate, color in (
            (-0.18, "LMR4DVar", "#0072B2"),
            (0.18, "LMRv2.1", "#D55E00"),
        ):
            values = [
                summary["comparisons"][f"{product}/djf_year_reference/{interval}"]["global"][
                    estimate
                ][metric]
                for product in PRODUCTS
            ]
            ax.bar(x + offset, values, width=0.36, color=color, label=estimate)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xticks(x, labels, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0, 0].legend()
    fig.suptitle(
        f"Global annual precipitation skill: {interval} overlap\n"
        "Within-reference matched comparison; single 4D-Var trajectory vs 20-MCrun LMR mean"
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison_root", type=Path)
    args = parser.parse_args()
    root = args.comparison_root
    summary = json.loads((root / "summary.json").read_text())
    output = root / "figures"
    output.mkdir(exist_ok=True)
    result = {"difference": "LMR4DVar minus LMRv2.1", "comparisons": {}}
    for interval in ("common", "longest"):
        groups = {"correlation": {}, "ce": {}}
        for product in PRODUCTS:
            directory = root / product / "djf_year_reference" / interval
            with xr.open_dataset(directory / "global_LMR4DVar_metrics.nc") as left:
                fdvar = left.load()
            with xr.open_dataset(directory / "global_LMRv2.1_metrics.nc") as right:
                lmr = right.load()
            # The same years/reference means must underlie both saved reconstructions.
            xr.testing.assert_equal(fdvar.time, lmr.time)
            xr.testing.assert_equal(fdvar.reference_mean, lmr.reference_mean)
            period = summary["comparisons"][f"{product}/djf_year_reference/{interval}"]["global"][
                "LMR4DVar"
            ]["period"]
            for kind, variable in (("correlation", "gridpoint_temporal_correlation"), ("ce", "ce")):
                maps, scores = paired_skill_difference(fdvar, lmr, variable)
                maps.attrs.update(
                    reference=product, period=f"{period[0]}-{period[1]}", baseline="1981-2000"
                )
                maps.to_netcdf(directory / f"{kind}_difference.nc", engine="h5netcdf")
                groups[kind][product] = (maps, scores, period)
                result["comparisons"][f"{product}/{interval}/{kind}"] = scores
        for kind, panels in groups.items():
            map_figure(panels, kind, interval, output / f"{kind}_maps_{interval}.png")
        summary_figure(summary, interval, output / f"skill_summary_{interval}.png")
    (output / "difference_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved six comparison figures and untruncated difference fields to {output}", flush=True)


if __name__ == "__main__":
    main()
