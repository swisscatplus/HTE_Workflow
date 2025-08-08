import argparse
import os
import subprocess
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

from visualization import generate_heatmaps


def _run_analysis(default_output: str) -> str:
    """Run analysis.py with calibration and visualization."""
    folder = input("Folder with analysis CSV files: ").strip()
    layout = input("Path to actual.xlsx for layout ordering (blank for none): ").strip()
    out = input(f"Output Excel file [{default_output}]: ").strip() or default_output
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "analysis.py"),
        folder,
        "--calibration",
        "--visualize",
        "--output",
        out,
    ]
    if layout:
        cmd.extend(["--layout", layout])
    subprocess.run(cmd, check=True)
    return out


def _run_dispense(default_output: str) -> str:
    """Run dispense_analyser.py to create dispense analysis."""
    runstats = input("RunStatistics Excel file: ").strip()
    actual = input("actual.xlsx file: ").strip()
    out = input(f"Output Excel file [{default_output}]: ").strip() or default_output
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "dispense_analyser.py"),
        runstats,
        actual,
        "--output",
        out,
    ]
    subprocess.run(cmd, check=True)
    return out


def _load_yield(path: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name="yield", header=[0, 1], index_col=0)


def _load_dispense(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    actual = pd.read_excel(path, sheet_name="actual_dispenses", header=[0, 1], index_col=0)
    rel = pd.read_excel(path, sheet_name="relative_difference", header=[0, 1], index_col=0)
    return actual, rel


def _experiment_number(rows: List[str], cols: List[int | str], row: str, col: int | str) -> int:
    r_idx = rows.index(row)
    c_idx = cols.index(col)
    return r_idx * len(cols) + c_idx + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize yields with dispense data")
    parser.add_argument("--analysis", help="analysis output Excel file")
    parser.add_argument("--dispense", help="dispense analysis Excel file")
    args = parser.parse_args()

    analysis_path = args.analysis
    if not analysis_path or not os.path.exists(analysis_path):
        print("Analysis Excel not provided or missing; running analysis.py")
        analysis_path = _run_analysis("analysis_output.xlsx")
    yield_df = _load_yield(analysis_path)

    dispense_path = args.dispense
    if not dispense_path or not os.path.exists(dispense_path):
        print("Dispense Excel not provided or missing; running dispense_analyser.py")
        dispense_path = _run_dispense("dispense_analysis.xlsx")
    actual_df, rel_df = _load_dispense(dispense_path)

    compounds = list(actual_df.columns.levels[0])
    print("Available compounds:")
    for c in compounds:
        print(f" - {c}")
    choice = input(
        "Normalize against which compound? (enter name or 'Compound group'): "
    ).strip()

    rows = list(yield_df.index)
    cols = list(yield_df.columns.levels[1])

    if choice.lower() == "compound group":
        group_in = input("Enter compounds to include, separated by commas: ").split(",")
        selected = [g.strip() for g in group_in if g.strip() in compounds]
        if not selected:
            print("No valid compounds selected; aborting")
            return
        factors = pd.DataFrame(np.nan, index=rows, columns=cols)
        for r in rows:
            for c in cols:
                present = [comp for comp in selected if actual_df[comp].loc[r, c] > 0]
                exp = _experiment_number(rows, cols, r, c)
                if not present:
                    print(
                        f"Warning: experiment {exp} (row {r}, col {c}) has none of the selected compounds"
                    )
                    continue
                if len(present) > 1:
                    print(
                        f"Experiment {exp} (row {r}, col {c}) has multiple compounds: {', '.join(present)}"
                    )
                    chosen = input("Choose compound for normalization: ").strip()
                    if chosen not in present:
                        chosen = present[0]
                else:
                    chosen = present[0]
                factors.loc[r, c] = 1 + rel_df[chosen].loc[r, c]
        norm_label = "group"
    else:
        if choice not in compounds:
            print("Unknown compound; aborting")
            return
        factors = 1 + rel_df[choice]
        for r in rows:
            for c in cols:
                if actual_df[choice].loc[r, c] == 0:
                    exp = _experiment_number(rows, cols, r, c)
                    print(
                        f"Warning: experiment {exp} (row {r}, col {c}) has no {choice} dispensed"
                    )
                    factors.loc[r, c] = np.nan
        norm_label = choice

    norm_dfs: Dict[str, pd.DataFrame] = {}
    for comp in yield_df.columns.levels[0]:
        norm_dfs[comp] = yield_df[comp] / factors
    norm_df = pd.concat(norm_dfs, axis=1)

    sheet_name = f"normalized_{norm_label}"[:31]
    with pd.ExcelWriter(analysis_path, mode="a", if_sheet_exists="replace") as writer:
        norm_df.to_excel(writer, sheet_name=sheet_name)
    print(f"Saved normalized yields to sheet '{sheet_name}' in {analysis_path}")

    generate_heatmaps(norm_df)


if __name__ == "__main__":
    main()
