from __future__ import annotations

import os
import sys
import ast
import shutil
import datetime
import subprocess
import re
import json

from importlib import import_module
from typing import Optional, Tuple, List, Any, Dict

import pandas as pd

from pathlib import Path

from sympy.simplify.radsimp import expand_numer

from hte_workflow.paths import DATA_DIR, OUT_DIR, ensure_dirs
import argparse

from hte_workflow import hte_calculator, reaction_analyser
from json_handling.hci_file_creator import build_chemical_space_from_spec, load_library
from json_handling.library_and_hci_adapter import hci_to_optimizer_dict


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _rename(src: str, dst: str) -> Optional[str]:
    """Rename *src* to *dst* if it exists and return the destination path."""
    if os.path.exists(src):
        os.rename(src, dst)
        return dst
    return None

def _ask(prompt: str, default: Optional[str] = None) -> str:
    sfx = f" [{default}]" if default is not None else ""
    val = input(f"{prompt}{sfx}: ").strip()

    if prompt == "Campaign name":
        # Ensure campaign name is not empty
        if not val:
            raise ValueError("Give a campaign name you fool!")

    return default if (val == "" and default is not None) else val

def _ask_float(prompt: str, default: Optional[float] = None) -> Optional[float]:
    while True:
        raw = _ask(prompt, str(default) if default is not None else None)
        if raw == "" and default is None:
            return None
        try:
            return float(raw)
        except ValueError:
            print("Please enter a number.")

def _ask_yesno(prompt: str, default: bool = True) -> bool:
    dv = "y" if default else "n"
    while True:
        raw = _ask(f"{prompt} (y/n)", dv).lower()
        if raw in ("y", "yes"): return True
        if raw in ("n", "no"):  return False
        print("Please answer y/n.")

def _ask_range(name: str, unit_default: str = "", step_optional: bool = True) -> Optional[Dict[str, Any]]:
    if not _ask_yesno(f"Add global range for '{name}'?", True):
        return None
    r: Dict[str, Any] = {}
    r["min"]  = _ask_float(f"  {name} min")
    r["max"]  = _ask_float(f"  {name} max")
    r["unit"] = _ask(f"  {name} unit", unit_default)
    if not step_optional or _ask_yesno("  Add step?", False):
        step = _ask_float("  step", None)
        if step is not None:
            r["step"] = step
    return r


def _ask_group_metadata() -> Optional[Dict[str, Any]]:
    if not _ask_yesno("Add a group (e.g., catalyst/solvent/base)?", True):
        return None
    g: Dict[str, Any] = {}
    g["groupName"] = _ask("  Group name", "e.g. catalyst")
    g["description"] = _ask("  Description", "")
    g["selectionMode"] = _ask("  Selection mode [one-of/any/at-least-one]", "one-of")

    print("  Equivalents range for this group (applies to selected member at runtime):")
    if g["groupName"].lower() == "solvent":
        print("    NOTE: For solvents, the volume will be calculated via the concentration.")
        eq: Dict[str, Any] = {}
        eq["min"] = 1
        eq["max"] = 1
        eq["unit"] = "eq"
    else:
        eq: Dict[str, Any] = {}
        eq["min"]  = _ask_float("    eq min", 0.01)
        eq["max"]  = _ask_float("    eq max", 0.10)
        eq["unit"] = _ask("    eq unit", "eq")
        if _ask_yesno("    Add step?", False):
            step = _ask_float("    step", None)
            if step is not None:
                eq["step"] = step
    g["equivalents"] = eq
    g["fixed"] = _ask_yesno("  Is this group fixed (always included)?", True)
    # still need to include what happens if this is not fixed

    # NOTE: no members collected here
    return g

def _ask_catalog_chemicals_with_group_assignment(
    lib_names: List[str],
    groups_meta: List[Dict[str, Any]],
) -> tuple[list[Dict[str, Any]], dict[str, list[str]]]:
    """
    Collect catalog chemicals (all go in HCI 'chemicals') and, for each, ask which group(s)
    it belongs to. Returns:
      - chemicals: list[{"chemicalName": name, "descriptors": {...}?}]
      - assignments: {groupName: [chemicalName, ...], ...}
    """

    chemicals: List[Dict[str, Any]] = []
    assignments: Dict[str, List[str]] = {g["groupName"]: [] for g in groups_meta}

    print("  Add chemicals by *chemicalName* (must exist in the library).")
    print("  Type 'list' to preview library names; press ENTER on empty to finish.")

    group_names = [g["groupName"] for g in groups_meta]

    while True:
        nm = _ask("    chemicalName", None)
        if not nm:
            break
        if nm.lower() == "list":
            print("    Library names (first 20):", ", ".join(lib_names[:20]), "...")
            continue

        # Optional descriptors for this chemical
        if _ask_yesno("    Add descriptors JSON for this chemical?", False):
            raw = _ask("      descriptors JSON", "{}")
            try:
                desc = json.loads(raw)
            except Exception as e:
                print(f"      Invalid JSON ({e}); skipping descriptors.")
                desc = None
        else:
            desc = None

        chemicals.append({"chemicalName": nm, **({"descriptors": desc} if desc else {})})

        # Ask group membership(s)
        if group_names:
            print("    Assign to groups (y/n):")
            for gname in group_names:
                if _ask_yesno(f"      - {gname}?", False):
                    assignments[gname].append(nm)

    return chemicals, assignments

def interactive_create_hci(*, library_path: str | Path, out_dir: str | Path) -> Path:
    """
    Interactively ask for a minimal spec, build the HCI JSON via your builder, and write it.
    Returns the written path.
    """
    # Load library (so we can validate names & show suggestions)
    lib = load_library(library_path)
    # We only need names list for nicer prompts
    try:
        # ChemicalLibrary stores records keyed by lower-case name internally;
        # expose a sorted list for user convenience
        lib_names = sorted([getattr(rec, "chemicalName") for rec in lib._by_name.values()])
    except Exception:
        lib_names = []
        print("Empty Library; refer to a different one or fill the library first.")

    # ---- campaign meta
    print("\n=== Campaign metadata ===")
    campaignName  = _ask("Campaign name")
    description   = _ask("Description", "")
    objective_txt = _ask("Objective (free text)", "")
    campaignClass = _ask("Campaign class", "Standard Research")
    type_txt      = _ask("Type", "optimization")
    reference     = _ask("Reference (URL/DOI/free text)", "")

    # Batch (minimal)
    print("\n=== Batch info ===")
    hasBatch = {
        "batchID": _ask("  batchID", "0"),
        "batchName": _ask("  batchName (e.g., YYYYMMDD)", str(datetime.date.today().strftime("%Y%m%d"))),
        "reactionType": _ask("  reactionType", ""),
        "reactionName": _ask("  reactionName", ""),
        "optimizationType": _ask("  optimizationType", ""),
        "link": _ask("  link", ""),
        "plate_size": int(_ask_float("  plate size (e.g. 48)", 48.0)),
    }

    # Objective block
    print("\n=== Objective block ===")
    hasObjective = {
        "criteria": _ask("  criteria", ),
        "condition": _ask("  condition", ""),
        "description": _ask("  description", ""),
        "objectiveName": _ask("  objectiveName", "")
    }

    # ---- global ranges
    print("\n=== Global ranges ===")
    print("Set min and max to the same value to disable a range.")
    ranges: Dict[str, Dict[str, Any]] = {}
    # Offer common knobs, and allow arbitrary
    for name, unit in (("temperature", "C"), ("concentration", "M"), ("time", "min")):
        r = _ask_range(name, unit_default=unit, step_optional=True)
        if r:
            ranges[name] = r
    while _ask_yesno("Add another custom global range (e.g. pressure)?", False):
        nm = _ask("  range name", "")
        if nm:
            r = _ask_range(nm, unit_default="")
            if r:
                ranges[nm] = r

    # ---- groups
    print("\n=== Groups (BO-controlled factors) ===")
    groups_meta: List[Dict[str, Any]] = []
    while True:
        g = _ask_group_metadata()
        if not g:
            break
        groups_meta.append(g)

    # ---- catalog chemicals (not in any group)
    print("\n=== Catalog chemicals (not part of any group) ===")
    chemicals, assignments = _ask_catalog_chemicals_with_group_assignment(lib_names, groups_meta)

    # --- materialize groups with members from assignments ---
    groups: List[Dict[str, Any]] = []
    for g in groups_meta:
        gname = g["groupName"]
        names_for_group = assignments.get(gname, [])
        members = [{"name": nm} for nm in names_for_group]
        groups.append({
            **g,
            "members": members,  # now populated
        })

    # ---- spec dict matches your build_chemical_space_from_spec
    spec: Dict[str, Any] = {
        "campaignName": campaignName,
        "description": description,
        "objective": objective_txt,
        "campaignClass": campaignClass,
        "type": type_txt,
        "reference": reference,
        "hasBatch": hasBatch,
        "hasObjective": hasObjective,
        "ranges": ranges,
        "groups": groups,
        "chemicals": chemicals,
    }

    # Build Campaign -> HCI JSON
    campaign = build_chemical_space_from_spec(spec, lib)
    hci_json = campaign.to_json()

    out_name = (f"{campaignName}_{hasBatch["batchID"]}_hci.json")
    out_path = Path(out_dir/out_name).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(hci_json, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote HCI file to: {out_path.resolve()}")
    return out_path, campaignName, hasBatch["batchID"]

#-- -------------------------------------------------------------------------
# Run the BO script
# ---------------------------------------------------------------------------
def run_bo_script(prefix: str, hci_file: str, limiting: str, well_volume_ul: float,
                  data_dir: Path, out_dir: Path, results: Optional[Path],
                  descriptors = True, fix_plate_temp = True) -> str:
    """
    Run the Bayesian optimization script with the given HCI file and directories.
    :param hci_file:
    :param data_dir:
    :param out_dir:
    :return:
    """
    hci_path = hci_file

    synthesis_file = str(f"{prefix}_synthesis.json")

    if not results:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "hte_workflow.bo_runner",
                "--hci-file", str(hci_path),
                "--out", synthesis_file,
                "--limiting-name", limiting,
                "--well-volume-uL", str(well_volume_ul),
                "--use-descriptors", str(descriptors),
                "--fix-plate-temp", str(fix_plate_temp),
                "--data-dir", str(data_dir),
                "--out-dir", str(out_dir),
            ],
            check=True,
        )
    else:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "hte_workflow.bo_runner",
                "--hci-file", str(hci_path),
                "--results", str(results),
                "--out", synthesis_file,
                "--limiting-name", limiting,
                "--well-volume-ul", str(well_volume_ul),
                "--use-descriptors", str(descriptors),
                "--fix-plate-temp", str(fix_plate_temp),
                "--data-dir", str(data_dir),
                "--out-dir", str(out_dir),
            ],
            check=True,
        )

    return synthesis_file

# ---------------------------------------------------------------------------
# Step 1: HTE calculator
# ---------------------------------------------------------------------------

def run_hte_calculator(prefix: str, data_dir: Path, out_dir: Path,
                        preload: Optional[str] = None, synthesis: Optional[str] = None,
                       hci: Optional[str]= None,
                       well_volume: Optional[float] = None) -> str:
    """Run ``hte_calculator`` and return the path to the generated Excel file."""
    output_file = f"{prefix}_calculator.xlsx"

    original_argv = sys.argv
    sys.argv = [
                "hte_calculator",
                "--output", output_file,
                "--data-dir", str(data_dir),
                "--out-dir", str(out_dir),]
    if preload:
        sys.argv.extend(["--preload", preload])
    if synthesis:
        sys.argv.extend(["--synthesis", synthesis])
    if hci:
        sys.argv.extend(["--hci", hci])

    import builtins

    AUTO_ANSWERS: list[tuple[re.Pattern, callable]] = [
        # Example: “Reaction name …”
        (re.compile(r"^Reaction name", re.I), lambda: prefix),

        # This will answer "y" if preload is provided, otherwise "n"
        (re.compile(r"^Use preloaded compounds\?\s*\[y/n\]", re.I), lambda: "y" if preload else "n"),

        # Example: limiting reagent prompt


        # Add more patterns here as needed...
    ]

    if preload or synthesis:
        AUTO_ANSWERS.extend([

            # No need to add more as autofilled correctly
            (re.compile(r"^Reagent name", re.I), lambda: ""),
            (re.compile(r"^Add more reagents/solvents", re.I), lambda: "N"),
        ])

    if hci and well_volume:
        AUTO_ANSWERS.extend(
            [(re.compile(r"^Choice for final volume per well \(uL\):\s", re.I), lambda: "c"),
             (re.compile(r"^Enter final volume per well", re.I), lambda:str(well_volume)),
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

def run_workflow_checker(prefix: str, calculator_file: str, data_dir: Path, out_dir: Path) -> str:
    """Fill and visualise the workflow based on the calculator output."""
    template = input("Workflow Excel template file: ").strip()
    workflow_file = f"{prefix}_workflow.xlsx"
    shutil.copyfile(str(data_dir / template), str(data_dir / workflow_file))

    subprocess.run(
        [
            sys.executable,
            "-m",
            "hte_workflow.workflow_checker",
            calculator_file,
            workflow_file,
            "--visualize",
            "--fill",
            "--data-dir", str(data_dir),
            "--out-dir", str(out_dir),
        ],
        check=True,
    )

    _rename(str(out_dir/ "layout.png"), str(out_dir/f"{prefix}_workflow_layout.png"))
    _rename(str(out_dir/"experiment_map.png"), str(out_dir/f"{prefix}_experiment_map.png"))
    _rename(str(out_dir/"workflow.png"), str(out_dir/f"{prefix}_workflow_diagram.png"))
    _rename(str(out_dir/"parsed_layout.xlsx"), str(out_dir/f"{prefix}_parsed_workflow_layout.xlsx"))
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


def run_reaction_analysis(prefix: str, limiting: str, calculator_file: str,
                          data_dir: Path, out_dir: Path) -> str:
    """Run the full reaction analysis workflow.

    Returns the path to the analysis Excel file and the normalised yield DataFrame.
    """
    analysis_output_file = f"{prefix}_analysis.xlsx"
    dispense_output_file = f"{prefix}_dispense.xlsx"

    original_argv = sys.argv
    sys.argv = [
                "reaction_analyser",
                "--data-dir", str(data_dir),
                "--out-dir", str(out_dir),]

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
    parser = argparse.ArgumentParser(description="HTE workflow orchestration script")
    parser.add_argument("--library-path",
                        default = None,
                        help = "Path to the chemical library file (HCI format)"
                        )
    parser.add_argument("--hci-file",
                        default=None,
                        help="Path to the HCI file to create (if not provided, will be created interactively)")
    parser.add_argument("--synth-file",
                        default=None,
                        help="Path to the synthesis file to create (if not provided, will be created interactively)")
    parser.add_argument("--BO",
                        default=False)
    parser.add_argument(
        "--data-dir",
        default=str(DATA_DIR),
        help="Directory for data files (default: DATA_DIR)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(OUT_DIR),
        help="Directory for output files (default: OUT_DIR)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    if args.BO and args.synth_file:
        print("Bayesian optimization mode doesn't take in synthesis files.")
        sys.exit(1)

    if not args.library_path:
        lib_path = input("Library path: ").strip()
    else:
        lib_path = args.library_path
    library_path = str(data_dir/lib_path)
    if not os.path.exists(library_path):
        print(f"Library file {library_path} does not exist.")
        sys.exit(1)

    if not args.hci_file:
        hci_file_path, exp_name, exp_number = interactive_create_hci(library_path=library_path, out_dir=out_dir)
    else:
        hci_file_path = str(out_dir/args.hci_file)
        if not os.path.exists(hci_file_path):
            print(f"HCI file {hci_file_path} does not exist.")
            sys.exit(1)

        hci_doc = hci_to_optimizer_dict(hci_file_path)
        exp_name = hci_doc.get("metadata").get("campaignName")
        exp_number = hci_doc.get("metadata").get("batch", {}).get("batchID")

    limiting = input("Limiting reagent name: ").strip()
    date_str = datetime.date.today().strftime("%Y%m%d")
    prefix = f"{date_str}_{exp_name}_{exp_number}"
    well_volume_ul = _ask_float("Well volume in microliters (default 500.0)", 500.0)

    bayesian_iteration = 1
    max_bayesian_iterations = 1  # Set a maximum number of iterations for Bayesian optimization
    while True: #  Loop for several plates via Bayesian
        if args.BO == "True":
            print("Bayesian optimization is running. ")
            synthesis_file = run_bo_script(
                prefix=prefix,
                hci_file=hci_file_path,
                limiting=limiting,
                well_volume_ul=well_volume_ul,
                data_dir=data_dir,
                out_dir=out_dir,
                results=args.synth_file,  # pretty sure this should be the analytical results file
                descriptors=True,
            )
        elif args.synth_file:
            synthesis_file = str(args.synth_file)
        else:
            synthesis_file = None

        """
        Need to implement generic workflow steps into synthesis file here (transport, washing, internal standard
        etc.
        Done by default value or by asking the user.
        If synthesis file is provided, it will be used to run the calculator.
        """

        if not synthesis_file and not args.hci_file:  # legacy mode
            preload = input("Preloaded reagents file (blank if none): ").strip() or None

            calculator_file = run_hte_calculator(prefix, data_dir, out_dir, preload=preload)
        elif synthesis_file:
            calculator_file = run_hte_calculator(prefix, data_dir, out_dir, synthesis=synthesis_file, hci=str(args.hci_file))
        else:
            calculator_file = run_hte_calculator(prefix, data_dir, out_dir, hci=str(args.hci_file), well_volume = well_volume_ul)

        print(f"Reagents calculated and saved to {calculator_file}.")
        print("Executing Lucas' file")
        # Placeholder for Lucas' file execution

        print("Workflow send to ArkSuite.")
        input("Prepare the digital twin and download the workflow template. Press Enter to continue...")

        # ideally be avoided by directly starting Lucas' file and running the excel
        workflow_file = run_workflow_checker(prefix, calculator_file, data_dir, out_dir)


        print("Check if the experiment setup is correct.")
        print("Upload the workflow file to Arcsuite and run the reaction.")
        input("Perform the reaction analysis (HPLC) and the calibration. Press Enter to continue to the analysis once finished...")

        analysis_path = run_reaction_analysis(prefix, limiting, calculator_file, data_dir, out_dir)



        # Placeholder for forwarding results to external systems
        # e.g., send ``yields`` to a Bayesian optimizer in future iterations
        if not args.BO:
            break
        elif bayesian_iteration >= max_bayesian_iterations:
            print("Maximum number of Bayesian iterations reached. Exiting.")
            break


if __name__ == "__main__":
    main()
