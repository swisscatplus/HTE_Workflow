import argparse
import math
import re
from typing import Dict, List, Tuple
import difflib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_layout(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return experiment layout and per-well dataframe from actual.xlsx."""
    per = pd.read_excel(path, sheet_name="per_well", header=[0, 1])
    rows = per.iloc[1:, 0].astype(str).tolist()
    first_reagent = per.columns[1][0]
    cols = [int(c[1]) for c in per.columns if c[0] == first_reagent]
    layout = pd.DataFrame(index=rows, columns=cols)
    exp = 1
    for c in cols:
        for r in rows:
            layout.loc[r, c] = exp
            exp += 1
    return layout, per


def parse_actual_dispenses(per: pd.DataFrame, layout: pd.DataFrame) -> Tuple[Dict[str, Dict[int, float]], List[str]]:
    """Parse intended dispenses from ``actual.xlsx``."""
    rows = list(layout.index)
    cols = list(layout.columns)
    reagents = sorted({c[0] for c in per.columns if c[0] != "Unnamed: 0_level_0"})
    reagents = [r for r in reagents if "as stock solution" not in r.lower()]
    data: Dict[str, Dict[int, float]] = {}
    for reagent in reagents:
        df = per.loc[1:, per.columns[per.columns.get_level_values(0) == reagent]].copy()
        df.index = rows
        df.columns = cols
        mapping: Dict[int, float] = {}
        for r in rows:
            for c in cols:
                exp = int(layout.loc[r, c])
                val = df.loc[r, c]
                if not pd.isna(val):
                    mapping[exp] = float(val)
        data[reagent] = mapping
    return data, reagents


def _normalize(name: str) -> str:
    name = str(name).lower()
    name = re.sub(r"\(.+?\)", "", name)  # remove parentheses
    name = name.replace("_", " ").replace("-", " ")
    replacements = {
        "meoh": "methanol",
        "mecn": "acetonitrile",
        "thf": "tetrahydrofuran",
        "dioxane": "1,4 dioxane",
        "toluene": "toluene",
    }
    for k, v in replacements.items():
        name = re.sub(k, v, name)
    name = re.sub(r"\d+", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def parse_runstatistics(path: str, actual_reagents: List[str]) -> Dict[str, Dict[int, float]]:
    """Parse actual dispenses from RunStatistics file."""
    df = pd.read_excel(path, sheet_name="Run Statistic", header=2)
    df = df[["Experiment", "Dispense Material", "Dispensed Amount", "Unit"]]
    df = df.dropna(subset=["Dispense Material", "Dispensed Amount"])
    df["Experiment"] = df["Experiment"].astype(str).str.extract(r"Experiment (\d+)").astype(int)
    actual_map = {_normalize(r): r for r in actual_reagents}
    data: Dict[str, Dict[int, float]] = {}
    for _, row in df.iterrows():
        norm = _normalize(row["Dispense Material"])
        match = difflib.get_close_matches(norm, actual_map.keys(), n=1, cutoff=0.6)
        if not match:
            continue
        reagent = actual_map[match[0]]
        exp = int(row["Experiment"])
        val = float(row["Dispensed Amount"])
        data.setdefault(reagent, {})[exp] = val
    return data


def build_matrix(data: Dict[str, Dict[int, float]], layout: pd.DataFrame) -> pd.DataFrame:
    """Arrange data into plate layout."""
    exp_to_rc: Dict[int, Tuple[str, int]] = {}
    for r in layout.index:
        for c in layout.columns:
            exp = int(layout.loc[r, c])
            exp_to_rc[exp] = (r, c)
    per_comp: Dict[str, pd.DataFrame] = {}
    for comp, exp_map in data.items():
        mat = pd.DataFrame(0.0, index=layout.index, columns=layout.columns)
        for exp, val in exp_map.items():
            if exp in exp_to_rc:
                r, c = exp_to_rc[exp]
                mat.loc[r, c] = val
        per_comp[comp] = mat
    if not per_comp:
        return pd.DataFrame(0.0, index=layout.index, columns=layout.columns)
    return pd.concat(per_comp, axis=1)


def generate_heatmaps(per_reagent: Dict[str, pd.DataFrame], prefix: str, center_zero: bool = False) -> None:
    if not per_reagent:
        return
    n = len(per_reagent)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    for ax, (name, df) in zip(axes, per_reagent.items()):
        mat = df.values.astype(float)
        if center_zero:
            vmax = np.max(np.abs(mat))
            im = ax.imshow(mat, cmap="coolwarm", origin="upper", vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(mat, cmap="viridis", origin="upper")
        ax.set_xticks(range(df.shape[1]))
        ax.set_xticklabels(df.columns)
        ax.set_yticks(range(df.shape[0]))
        ax.set_yticklabels(df.index)
        ax.set_title(name)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    out_path = f"{prefix}.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse HTE dispenses")
    parser.add_argument("runstatistics", help="RunStatistics excel file")
    parser.add_argument("actual", help="actual excel file")
    parser.add_argument("--output", default="dispense_analysis.xlsx", help="Output Excel file")
    args = parser.parse_args()

    layout, per = load_layout(args.actual)
    intended_map, reagents = parse_actual_dispenses(per, layout)
    run_map = parse_runstatistics(args.runstatistics, reagents)

    intended_df = build_matrix(intended_map, layout).fillna(0)
    run_df = build_matrix(run_map, layout)
    run_df = run_df.reindex(columns=intended_df.columns).fillna(0)
    relative = (run_df - intended_df) / intended_df.replace(0, np.nan)
    relative = relative.fillna(0)

    with pd.ExcelWriter(args.output) as writer:
        run_df.to_excel(writer, sheet_name="actual_dispenses")
        relative.to_excel(writer, sheet_name="relative_difference")
    print(f"Saved analysis to {args.output}")

    run_dict = {comp: run_df[comp] for comp in run_df.columns.levels[0]}
    rel_dict = {comp: relative[comp] for comp in relative.columns.levels[0]}
    generate_heatmaps(run_dict, "actual_dispenses_heatmap")
    generate_heatmaps(rel_dict, "relative_difference_heatmap", center_zero=True)


if __name__ == "__main__":
    main()
