"""Standalone figures from recorded held-out predictions, without model imports."""
from pathlib import Path

import numpy as np


def write_report(output, config, data, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    figure_dir = output / "figures"
    figure_dir.mkdir(exist_ok=True)
    spots = result["test_spots"]
    limit = int(config.get("max_plot_slides", 6))
    if limit < 0:
        raise ValueError("max_plot_slides must be nonnegative")
    figures = []
    # Deterministic selection by ID, not by best-looking performance.
    for i, (slide, group) in enumerate(spots.groupby("slide_id", sort=True)):
        if i >= limit:
            break
        idx = group.index.to_numpy()
        for j, target in enumerate(data["names"]):
            maps = {"Measured ST": result["truth"][idx, j],
                    **{name: pred[idx, j] for name, pred in result["predictions"].items()}}
            vmin, vmax = np.min(list(maps.values())), np.max(list(maps.values()))
            fig, axes = plt.subplots(1, len(maps), figsize=(4 * len(maps), 4), squeeze=False, constrained_layout=True)
            for ax, (name, values) in zip(axes[0], maps.items()):
                points = ax.scatter(group.x, group.y, c=values, s=9, vmin=vmin, vmax=vmax, cmap="viridis")
                ax.set_title(name)
                ax.set_aspect("equal")
                ax.invert_yaxis()
                ax.set_xlabel("level-0 x (pixels)")
            fig.colorbar(points, ax=axes[0].tolist(), label=target)
            fig.suptitle(f"{slide} — {target}")
            stem = f"map_{i:03d}_{j:03d}"
            for extension in ("png", "pdf"):
                fig.savefig(figure_dir / f"{stem}.{extension}", dpi=160)
            plt.close(fig)
            figures.append(f"![{slide} / {target}](figures/{stem}.png)")
    lines = ["# Spatial biology evaluation", "",
             "Only held-out patients are evaluated. Program scores are proxies, not cell fractions or causal mechanisms.", "",
             "See patient_summary.csv for patient-weighted means and paired 95% bootstrap intervals.",
             "Positive deltas favour the candidate for correlations/coverage; negative deltas favour it for errors.",
             "Nonsignificant differences do not establish equivalence. Coverage is not causal importance.", "",
             "## Comparators", ""]
    for method in config["methods"]:
        lines.append(f"- {method['name']}: {method['citation']}; supervision: {method['supervision']}; "
                     f"pretraining overlap: {method['pretraining_overlap']}.")
    coverage = result["coverage"]
    if not coverage.empty:
        # Balance patients before aggregating the coverage chart.
        table = coverage.groupby(["patient_id", "method", "niche"]).coverage.mean().reset_index()
        table = table.groupby(["niche", "method"]).coverage.mean().unstack("method")
        fig, ax = plt.subplots(figsize=(max(6, len(table) * 1.5), 4), constrained_layout=True)
        table.plot.bar(ax=ax)
        ax.set_ylabel("Patient-averaged spot coverage")
        ax.set_ylim(0, 1.05)
        for extension in ("png", "pdf"):
            fig.savefig(figure_dir / f"niche_coverage.{extension}", dpi=160)
        plt.close(fig)
        figures.append("![Niche coverage](figures/niche_coverage.png)")
    lines += ["", "## Maps", "", "Slides are selected in sorted identifier order, not by performance.", "", *figures]
    (output / "report.md").write_text("\n\n".join(lines) + "\n")
