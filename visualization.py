import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Optional
import matplotlib.cm as cm
import numpy as np
from matplotlib.colors import Normalize


def _ordered_experiments(area_data: Dict[str, Dict[int, float]], layout: Optional[pd.DataFrame]) -> list:
    if layout is not None:
        exps = []
        for r in layout.index:
            for c in layout.columns:
                val = layout.loc[r, c]
                if pd.notna(val):
                    exps.append(int(val))
        return exps
    exps = sorted({e for mapping in area_data.values() for e in mapping})
    return exps


def generate_pie_plots(
    area_data: Dict[str, Dict[int, float]],
    yield_data: Dict[str, Dict[int, float]],
    calibrations: Dict[str, Dict[str, float]],
    internal_std: Optional[str],
    layout: Optional[pd.DataFrame],
) -> None:
    calibrated = list(calibrations.keys())
    ordered_exps = _ordered_experiments(area_data, layout)

    if layout is not None:
        rows = list(layout.index)
        cols = list(layout.columns)
        nrows, ncols = len(rows), len(cols)
    else:
        nrows, ncols = 1, len(ordered_exps)
        rows = ["row"]
        cols = ordered_exps

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    fig_other, axes_other = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)
    axes_other = np.array(axes_other).reshape(nrows, ncols)

    for r_i in range(nrows):
        for c_i in range(ncols):
            ax = axes[r_i, c_i]
            ax_other = axes_other[r_i, c_i]
            exp = None
            if layout is not None:
                val = layout.iloc[r_i, c_i]
                if pd.notna(val):
                    exp = int(val)
            else:
                if c_i < len(ordered_exps):
                    exp = ordered_exps[c_i]
            if exp is None or exp not in ordered_exps:
                ax.axis("off")
                ax_other.axis("off")
                continue

            slices = []
            labels = []
            colors = []
            for comp in calibrated:
                area = area_data.get(comp, {}).get(exp, 0.0)
                slices.append(area)
                yv = yield_data.get(comp, {}).get(exp, 0.0)
                labels.append(f"{comp} ({yv:.1f}%)")
                colors.append(cm.viridis(np.clip(yv, 0.0, 100.0) / 100.0))
            other_area = 0.0
            for comp, amap in area_data.items():
                if comp in calibrated or comp == internal_std:
                    continue
                other_area += amap.get(exp, 0.0)
            if other_area:
                slices.append(other_area)
                labels.append("Others")
                colors.append("lightgrey")
            if any(slices):
                ax.pie(slices, labels=labels, colors=colors, startangle=90)
            ax.set_title(f"Exp {exp}")
            ax.axis("equal")

            # other-only pie chart
            other_labels = []
            other_sizes = []
            other_colors = []
            cmap = cm.tab20
            idx = 0
            for comp, amap in area_data.items():
                if comp in calibrated or comp == internal_std:
                    continue
                area = amap.get(exp, 0.0)
                if area > 0:
                    other_labels.append(str(comp))
                    other_sizes.append(area)
                    other_colors.append(cmap(idx % 20))
                    idx += 1
            if other_sizes:
                ax_other.pie(other_sizes, labels=other_labels, colors=other_colors, startangle=90)
            ax_other.set_title(f"Exp {exp} others")
            ax_other.axis("equal")

    sm = cm.ScalarMappable(cmap="viridis", norm=Normalize(vmin=0, vmax=100))
    fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, label="Yield (%)")
    fig.tight_layout()
    fig.savefig("pie_plots.png")
    plt.close(fig)

    fig_other.tight_layout()
    fig_other.savefig("pie_plots_others.png")
    plt.close(fig_other)


def generate_heatmaps(yield_df: pd.DataFrame) -> None:
    if not isinstance(yield_df.columns, pd.MultiIndex):
        return
    for comp in yield_df.columns.levels[0]:
        mat = yield_df[comp].astype(float)
        plt.figure()
        im = plt.imshow(mat, vmin=0, vmax=100, cmap="viridis")
        plt.title(f"Yield heatmap: {comp}")
        plt.colorbar(im, label="Yield (%)")
        plt.xticks(range(len(mat.columns)), mat.columns)
        plt.yticks(range(len(mat.index)), mat.index)
        plt.xlabel("Column")
        plt.ylabel("Row")
        plt.tight_layout()
        plt.savefig(f"heatmap_{comp}.png")
        plt.close()
