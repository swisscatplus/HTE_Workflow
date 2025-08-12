import os
import sys
import shutil
import datetime
import subprocess
import re

from importlib import import_module
from typing import Optional, Tuple

import pandas as pd

from hte_workflow import hte_calculator, reaction_analyser


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
    sys.argv = ["-m",
                "hte_workflow.hte_calculator",
                "--output", output_file]
    if preload:
        sys.argv.extend(["--preload", preload])

    import builtins

    AUTO_ANSWERS: list[tuple[re.Pattern, callable]] = [
        # Example: “Reaction name …”
        (re.compile(r"^Reaction name", re.I), lambda: prefix),

        # This will answer "y" if preload is provided, otherwise "n"
        (re.compile(r"^Use preloaded compounds\?\s*\[y/n\]", re.I), lambda: "y" if preload else "n"),

        # Example: limiting reagent prompt


        # Add more patterns here as needed...
    ]

    if preload:
        AUTO_ANSWERS.extend([

            # No need to add more as autofilled correctly
            (re.compile(r"^Reagent name", re.I), lambda: ""),
            (re.compile(r"^Add more reagents/solvents", re.I), lambda: "N"),
        ])

    builtin_input = builtins.input

    def patched_input(prompt: str = "") -> str:
        for pattern, answer_fn in AUTO_ANSWERS:
            if pattern.search(prompt or ""):
                ans = str(answer_fn())
                # Echo what we’re “typing” so logs remain readable:
                print(f"{prompt}{ans}")
                return ans
        # Fallback: ask the user normally
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
            "-m",
            "hte_workflow.workflow_checker",
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
    _rename("parsed_layout.xlsx", f"{prefix}_parsed_workflow_layout.xlsx")
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


def run_reaction_analysis(prefix: str, limiting: str, calculator_file: str) -> str:
    """Run the full reaction analysis workflow.

    Returns the path to the analysis Excel file and the normalised yield DataFrame.
    """
    analysis_output_file = f"{prefix}_analysis.xlsx"
    dispense_output_file = f"{prefix}_dispense.xlsx"

    original_argv = sys.argv
    sys.argv = ["-m",
                "hte_workflow.analysis",]

    import builtins

    AUTO_ANSWERS: list[tuple[re.Pattern, callable]] = [
        # Example: “Reaction name …”
        (re.compile(r"^Output Excel file; default: analysis_output.xlsx", re.I), lambda: analysis_output_file),
        (re.compile(r"^Output Excel file; default: dispense_analysis.xlsx", re.I), lambda: dispense_output_file),

        (re.compile(r"^Analysis start time in min", re.I), lambda: ""),
        # Default to empty string for analysis start time; could be changed to different defaults if needed

        (re.compile(r"^Path to actual.xlsx for layout ordering", re.I), lambda: calculator_file),

        (re.compile(r"^Normalize against which compound?", re.I), lambda: limiting),
    ]

    builtin_input = builtins.input

    def patched_input(prompt: str = "") -> str:
        for pattern, answer_fn in AUTO_ANSWERS:
            if pattern.search(prompt or ""):
                ans = str(answer_fn())
                # Echo what we’re “typing” so logs remain readable:
                print(f"{prompt}{ans}")
                return ans
        # Fallback: ask the user normally
        return builtin_input(prompt)

    builtins.input = patched_input
    try:
        reaction_analyser.main()
    finally:
        builtins.input = builtin_input
        sys.argv = original_argv

    return analysis_output_file

# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    exp_name = input("Experiment name: ").strip()
    exp_number = input("Experiment number: ").strip()
    limiting = input("Limiting reagent name: ").strip()  # Needs to be parsed directly from calculator
    date_str = datetime.date.today().strftime("%Y%m%d")
    prefix = f"{date_str}_{exp_name}_{exp_number}"

    preload = input("Preloaded reagents file (blank if none): ").strip() or None

    calculator_file = run_hte_calculator(prefix, preload)

    print(f"Reagents calculated and saved to {calculator_file}.")
    input("Prepare the digital twin and download the workflow template. Press Enter to continue...")

    workflow_file = run_workflow_checker(prefix, calculator_file)

    print("Check if the experiment setup is correct.")
    print("Upload the workflow file to Arcsuite and run the reaction.")
    input("Perform the reaction analysis (HPLC) and the calibration. Press Enter to continue to the analysis once finished...")

    analysis_path = run_reaction_analysis(prefix, limiting, calculator_file)



    # Placeholder for forwarding results to external systems
    # e.g., send ``yields`` to a Bayesian optimizer in future iterations


if __name__ == "__main__":
    main()
