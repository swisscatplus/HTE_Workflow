#!/usr/bin/env python3
"""
manual_plate_builder.py

Interactive tool to build BO-style `selections` directly from user input
(using the same HCI/opt_spec as the BO), and immediately call
`synthesis_writer.write_synthesis_json`.

No intermediate selections.json is required.

Usage examples
--------------

# Simple: ask interactively, then write synthesis.json
python -m json_handling.manual_plate_builder \
    --hci-file my_campaign_hci.json \
    --out my_campaign_synthesis.json \
    --limiting-name Sub_01 \
    --limiting-moles 2e-6 \
    --well-volume-uL 50

"""
#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import warnings

from json_handling.library_and_hci_adapter import hci_to_optimizer_dict
from json_handling.synthesis_writer import write_synthesis_json
from hte_workflow.paths import DATA_DIR, OUT_DIR, ensure_dirs



def seq_labels(plate_size: int) -> list[str]:
    return [str(i + 1) for i in range(plate_size)]


# ---------- HCI parsing ----------
def load_hci(hci_path: Path) -> Dict[str, Any]:
    with hci_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def hci_plate_size(hci: Dict[str, Any]) -> int:
    return int(hci["hasCampaign"]["hasBatch"]["plate_size"])  # e.g. 18 in your example

def hci_globals_from_ranges(hci: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float, str]]]:
    """
    Returns:
      base_globals: constants auto-filled from hasRanges where min==max
      variable_globals: {name: (min, max, unit)} for which we'll prompt once
    """
    base_globals: Dict[str, float] = {}
    variable_globals: Dict[str, Tuple[float, float, str]] = {}
    ranges = hci["hasCampaign"].get("hasRanges", {})
    for name, spec in ranges.items():
        vmin = float(spec.get("min"))
        vmax = float(spec.get("max"))
        unit = spec.get("unit", "")
        if vmin == vmax:
            base_globals[name] = vmin
        else:
            variable_globals[name] = (vmin, vmax, unit)
    return base_globals, variable_globals

def prompt_variable_globals(variable_globals: Dict[str, Tuple[float, float, str]]) -> Dict[str, float]:
    """
    Prompt ONCE per variable global (e.g., temperature if range-open),
    enforce min/max, and return chosen values.
    """
    chosen: Dict[str, float] = {}
    for name, (vmin, vmax, unit) in variable_globals.items():
        while True:
            raw = input(f"Set global '{name}' in [{vmin} .. {vmax}] {unit}: ").strip()
            try:
                val = float(raw)
                if not (vmin <= val <= vmax):
                    print(f"Value out of range. Must be between {vmin} and {vmax}.")
                    continue
                chosen[name] = val
                break
            except ValueError:
                print("Please enter a number.")
    return chosen

def hci_groups(hci: Dict[str, Any]) -> List[Dict[str, Any]]:
    return hci["hasCampaign"].get("hasGroups", [])

def collect_group_catalog(hci: Dict[str, Any]) -> Tuple[List[str], Dict[str, List[str]], Dict[str, Tuple[float, float, str]]]:
    """
    Returns:
      group_names: list of group names
      members: {group: [chemicalName, ...]}
      eq_ranges: {group: (min, max, unit)} for groups that define 'equivalents'
    """
    group_names: List[str] = []
    members: Dict[str, List[str]] = {}
    eq_ranges: Dict[str, Tuple[float, float, str]] = {}

    for g in hci_groups(hci):
        gname = g.get("groupName")
        if not gname:
            continue
        group_names.append(gname)

        # collect member names from .members[*].reference.chemicalName
        mems: List[str] = []
        for m in g.get("members", []):
            ref = m.get("reference", {})
            nm = ref.get("chemicalName")
            if nm:
                mems.append(nm)
        if mems:
            members[gname] = mems

        # collect equivalents range if present
        if "equivalents" in g and isinstance(g["equivalents"], dict):
            eq = g["equivalents"]
            try:
                eq_min = float(eq.get("min"))
                eq_max = float(eq.get("max"))
                unit = eq.get("unit", "eq")
                eq_ranges[gname] = (eq_min, eq_max, unit)
            except (TypeError, ValueError):
                pass

    return group_names, members, eq_ranges


# ---------- Interactive selections builder ----------
def interactive_build_selections_from_hci(
    hci: Dict[str, Any],
    plate_size: int,
    base_globals: Dict[str, float],  # constants from hasRanges
    variable_globals_once: Dict[str, float],  # your answers for open ranges
) -> List[Dict[str, Any]]:
    group_names, members_by_group, eq_ranges = collect_group_catalog(hci)

    print(f"Detected groups: {', '.join(group_names)}")
    if base_globals:
        print("Auto-applied constant globals:", ", ".join(f"{k}={v}" for k, v in base_globals.items()))
    if variable_globals_once:
        print("User-specified globals:", ", ".join(f"{k}={v}" for k, v in variable_globals_once.items()))

    # globals for every well = union of base + chosen variable values
    globals_for_plate = {**base_globals, **variable_globals_once}

    selections: List[Dict[str, Any]] = []
    prev_groups: Dict[str, Dict[str, Any]] = {}

    for i, exp_id in enumerate(seq_labels(plate_size), start=1):
        print(f"\n=== Experiment {exp_id} ===")
        sel_globals = dict(globals_for_plate)  # same for all wells (as requested)

        # groups: reuse and override
        sel_groups: Dict[str, Dict[str, Any]] = {}
        reuse = input("Reuse previous experiment’s groups? [ENTER=yes / n=no]: ").strip().lower()
        if reuse in ("", "y", "yes") and prev_groups:
            sel_groups = {g: dict(v) for g, v in prev_groups.items()}
            print("Reused previous groups (you can override below).")

        for gname in group_names:
            # show options
            options = members_by_group.get(gname, [])
            current = sel_groups.get(gname, {}).get("member")
            if options:
                print(f"\nGroup '{gname}' options: {', '.join(options)}")

            # member
            prompt = f"Member for '{gname}'" + (f" [{current}]" if current else "") + " (ENTER keep/skip): "
            chosen = input(prompt).strip()
            if chosen:
                if options and chosen not in options:
                    print(f"Note: '{chosen}' not in known members for {gname}; using anyway.")
                sel_groups.setdefault(gname, {})["member"] = chosen

            # equivalents
            if gname in eq_ranges and gname in sel_groups:
                eq_min, eq_max, unit = eq_ranges[gname]
                if eq_min == eq_max:
                    # auto-fill fixed equivalents
                    sel_groups[gname]["equivalents"] = eq_min
                    print(f"Equivalents for '{gname}' fixed at {eq_min} {unit}.")
                else:
                    cur_eq = sel_groups[gname].get("equivalents")
                    while True:
                        raw = input(
                            f"Equivalents for '{gname}' in [{eq_min}..{eq_max}] {unit}"
                            + (f" [{cur_eq}]" if cur_eq is not None else "")
                            + " (ENTER keep): "
                        ).strip()
                        if not raw:
                            break
                        try:
                            val = float(raw)
                            if not (eq_min <= val <= eq_max):
                                print(f"Out of range; must be between {eq_min} and {eq_max}.")
                                continue
                            sel_groups[gname]["equivalents"] = val
                            break
                        except ValueError:
                            print("Please enter a number.")

        selections.append({"groups": sel_groups, "globals": sel_globals})
        prev_groups = sel_groups

    return selections


# ---------- Main ----------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Interactively answer BO decisions from HCI:\n"
            "- read globals from hasRanges (auto-fill constants; prompt once if range open)\n"
            "- read groups/members from hasGroups\n"
            "- build BO-style selections in memory\n"
            "- write synthesis.json via synthesis_writer"
        )
    )
    ap.add_argument("--hci-file", required=True, help="HCI JSON (your schema).")
    ap.add_argument("--out", required=True, help="Output synthesis.json.")
    ap.add_argument("--plate-size", type=int, help="Override plate size (else read from HCI).")
    ap.add_argument("--limiting-name", help="Limiting reagent name.")
    ap.add_argument("--well-volume-uL", type=float, help="Total volume per well (µL).")

    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--data-dir", default=str(DATA_DIR), help="Directory with data files for layout_parser")

    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    hci_path = Path(args.hci_file)
    if not hci_path.is_file():
        raise SystemExit(f"HCI file not found: {hci_path}")

    # 1) Load HCI (your schema) and derive plate, globals, groups
    hci = load_hci(hci_path)
    plate_size = int(args.plate_size or hci_plate_size(hci))  # e.g., 18 in your example

    base_globals, variable_defs = hci_globals_from_ranges(hci)  # constants + open ranges
    chosen_variable_globals = prompt_variable_globals(variable_defs)  # ask ONCE per open range

    # 2) Build selections interactively
    selections = interactive_build_selections_from_hci(
        hci=hci,
        plate_size=plate_size,
        base_globals=base_globals,
        variable_globals_once=chosen_variable_globals,
    )

    # Helper for moles from concentration
    globals_for_plate = {**base_globals, **chosen_variable_globals}

    if "concentration" in globals_for_plate:
        limiting_moles = float(globals_for_plate["concentration"]) * args.well_volume_uL * 1e-6
    else:
        warnings.warn(
            "'concentration' global not specified — falling back to manual input for limiting moles.",
            RuntimeWarning,
        )
        try:
            limiting_moles = float(input("Enter quantity of limiting reagent (in moles): ").strip())
        except ValueError:
            raise SystemExit("Invalid input: please enter a numeric value for moles.")

    # 3) Call synthesis_writer with opt_spec that BO would also use
    opt_spec = hci_to_optimizer_dict(str(hci_path))
    write_synthesis_json(
        opt_spec=opt_spec,
        selections=selections,
        out_path=str(out_dir/args.out),
        plate_size=plate_size,
        limiting_name=args.limiting_name,
        limiting_moles=limiting_moles,
        well_volume_uL=args.well_volume_uL,
    )
    print(f"\n✅ Wrote synthesis.json to {args.out}")


if __name__ == "__main__":
    main()
