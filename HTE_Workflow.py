import os
import sys
import shutil
import datetime
import subprocess
from typing import Optional, Tuple

import pandas as pd

import hte_calculator
import reaction_analyser as ra


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _rename(src: str, dst: str) -> Optional[str]:
    """Rename *src* to *dst* if it exists and return the destination path."""
    if os.path.exists(src):
        os.rename(src, dst)
        return dst
    return None


# ---------------------------------------------------------------------------
# Step 1: HTE calculator
# ---------------------------------------------------------------------------

def run_hte_calculator(prefix: str, preload: Optional[str]) -> str:
    """Run ``hte_calculator`` and return the path to the generated Excel file."""
    output_file = f"{prefix}_calculator.xlsx"

    original_argv = sys.argv
    sys.argv = ["hte_calculator.py", "--output", output_file]
    if preload:
        sys.argv.extend(["--preload", preload])

    import builtins

    builtin_input = builtins.input

    def patched_input(prompt: str = "") -> str:
        if prompt.startswith("Reaction name"):
            # Automatically supply our prefix for the reaction name
            print(f"{prompt}{prefix}")
            return prefix
        return builtin_input(prompt)

    builtins.input = patched_input
    try:
        hte_calculator.main()
    finally:
        builtins.input = builtin_input
        sys.argv = original_argv

    return output_file


# ---------------------------------------------------------------------------
# Step 2: Workflow checker
# ---------------------------------------------------------------------------

def run_workflow_checker(prefix: str, calculator_file: str) -> str:
    """Fill and visualise the workflow based on the calculator output."""
    template = input("Workflow Excel template file: ").strip()
    workflow_file = f"{prefix}_workflow.xlsx"
    shutil.copyfile(template, workflow_file)

    subprocess.run(
        [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "workflow_checker.py"),
            calculator_file,
            workflow_file,
            "--visualize",
            "--fill",
        ],
        check=True,
    )

    _rename("layout.png", f"{prefix}_workflow_layout.png")
    _rename("experiment_map.png", f"{prefix}_experiment_map.png")
    _rename("workflow.png", f"{prefix}_workflow_diagram.png")
    return workflow_file


# ---------------------------------------------------------------------------
# Step 3: Reaction analysis
# ---------------------------------------------------------------------------

def _rename_analysis_images(prefix: str, yield_df: pd.DataFrame) -> None:
    _rename("pie_plots.png", f"{prefix}_pie_plots.png")
    _rename("pie_plots_others.png", f"{prefix}_pie_plots_others.png")
    for comp in yield_df.columns.levels[0]:
        _rename(f"heatmap_{comp}.png", f"{prefix}_heatmap_{comp}.png")


def _rename_dispense_images(prefix: str, actual_df: pd.DataFrame, rel_df: pd.DataFrame) -> None:
    for comp in actual_df.columns.levels[0]:
        _rename(
            f"actual_dispenses_heatmap_{comp}.png",
            f"{prefix}_actual_dispenses_heatmap_{comp}.png",
        )
    for comp in rel_df.columns.levels[0]:
        _rename(
            f"relative_difference_heatmap_{comp}.png",
            f"{prefix}_relative_difference_heatmap_{comp}.png",
        )


def run_reaction_analysis(prefix: str, limiting: str) -> Tuple[str, pd.DataFrame]:
    """Run the full reaction analysis workflow.

    Returns the path to the analysis Excel file and the normalised yield DataFrame.
    """
    analysis_path, layout_path = ra._run_analysis(f"{prefix}_analysis.xlsx")
    yield_df = ra._load_yield(analysis_path)
    _rename_analysis_images(prefix, yield_df)

    dispense_path = ra._run_dispense(f"{prefix}_dispense.xlsx", layout_path)
    actual_df, rel_df = ra._load_dispense(dispense_path)
    _rename_dispense_images(prefix, actual_df, rel_df)

    factors = 1 + rel_df[limiting]
    for r in factors.index:
        for c in factors.columns:
            if actual_df[limiting].loc[r, c] == 0:
                factors.loc[r, c] = float("nan")

    norm_dfs = {comp: yield_df[comp] / factors for comp in yield_df.columns.levels[0]}
    norm_df = pd.concat(norm_dfs, axis=1)

    sheet_name = f"normalized_{limiting}"[:31]
    with pd.ExcelWriter(analysis_path, mode="a", if_sheet_exists="replace") as writer:
        norm_df.to_excel(writer, sheet_name=sheet_name)
    print(f"Saved normalized yields to sheet '{sheet_name}' in {analysis_path}")

    ra.generate_heatmaps(norm_df, prefix=f"{prefix}_heatmap_normalized_{limiting}")
    return analysis_path, norm_df


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    exp_name = input("Experiment name: ").strip()
    exp_number = input("Experiment number: ").strip()
    limiting = input("Limiting reagent name: ").strip()
    date_str = datetime.date.today().strftime("%Y%m%d")
    prefix = f"{date_str}_{exp_name}_{exp_number}"

    preload = input("Preloaded reagents file (blank if none): ").strip() or None

    calculator_file = run_hte_calculator(prefix, preload)
    workflow_file = run_workflow_checker(prefix, calculator_file)

    input("Run the reaction according to the workflow now. Press Enter to continue once finished...")

    analysis_path, yields = run_reaction_analysis(prefix, limiting)

    print("\nFinal normalized yields:")
    print(yields)

    # Placeholder for forwarding results to external systems
    # e.g., send ``yields`` to a Bayesian optimizer in future iterations


if __name__ == "__main__":
    main()
