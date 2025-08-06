import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Optional
import matplotlib.cm as cm


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
    for exp in ordered_exps:
        slices = []
        labels = []
        colors = []
        for comp in calibrated:
            area = area_data.get(comp, {}).get(exp, 0.0)
            slices.append(area)
            yv = yield_data.get(comp, {}).get(exp, 0.0)
            labels.append(f"{comp} ({yv:.1f}%)")
            colors.append(cm.viridis(min(max(yv, 0.0), 100.0) / 100.0))
        other_area = 0.0
        for comp, amap in area_data.items():
            if comp in calibrated or comp == internal_std:
                continue
            other_area += amap.get(exp, 0.0)
        if other_area:
            slices.append(other_area)
            labels.append("Others")
            colors.append("lightgrey")
        plt.figure()
        if any(slices):
            plt.pie(slices, labels=labels, colors=colors, startangle=90)
            plt.title(f"Experiment {exp}")
            plt.tight_layout()
            plt.savefig(f"experiment_{exp}_pie.png")
        plt.close()
        # second pie for others individually
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
            plt.figure()
            plt.pie(other_sizes, labels=other_labels, colors=other_colors, startangle=90)
            plt.title(f"Experiment {exp} - other peaks")
            plt.tight_layout()
            plt.savefig(f"experiment_{exp}_pie_others.png")
            plt.close()


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
