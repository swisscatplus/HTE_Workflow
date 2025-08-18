#!/usr/bin/env python3
"""
chem_library_and_hci_adapter.py

Part A: Maintain a chemical metadata library (chem.json)
  - init: create empty library
  - add: add/update one chemical
  - import-csv: bulk import (headers flexible; see notes)
  - list/show: inspect

Part B: Load an HCI file (output of your Campaign.to_json()) and convert it
        into a neutral, optimizer-friendly dict that a Bayesian optimizer
        can consume with minimal glue code.

CLI examples
------------
# Initialize an empty library
python chem_library_and_hci_adapter.py init --out data/chem.json

# Add/update a chemical
python chem_library_and_hci_adapter.py add --lib data/chem.json \
  --name "Pd(PPh3)4" --id "Pd(PPh3)4" --cas "14221-01-3" \
  --mw 1155.55 --mw-unit g/mol \
  --smiles "P(c1ccccc1)(c2ccccc2)c3ccccc3.Pd" \
  --inchi "InChI=1S/..." --formula "C72H60P4Pd" \
  --density 1.20 --density-unit g/mL --physicalstate solid

# Import from CSV
python chem_library_and_hci_adapter.py import-csv --lib data/chem.json --csv data/chem.csv

# Convert an HCI file to optimizer spec JSON
python chem_library_and_hci_adapter.py hci-to-optimizer \
  --hci artifacts/space.json --out artifacts/optimizer_space.json
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
import argparse
import csv
import json
import sys

from hte_workflow.paths import DATA_DIR, OUT_DIR, ensure_dirs


# -------------------------
# Common utilities
# -------------------------

def _as_path(p: Union[str, Path]) -> Path:
    return p if isinstance(p, Path) else Path(p)

def _load_json(path: Union[str, Path]) -> Any:
    return json.loads(_as_path(path).read_text(encoding="utf-8"))

def _save_json(obj: Any, path: Union[str, Path]) -> None:
    _as_path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

# -------------------------
# Part A: Chemical library
# -------------------------

@dataclass
class Quantity:
    value: float
    unit: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {"value": float(self.value), "unit": self.unit}

@dataclass
class Chemical:
    chemicalID: str
    chemicalName: str
    CASNumber: str
    molecularMass: Quantity
    smiles: str
    Inchi: str
    molecularFormula: str
    swissCatNumber: str
    physicalstate: str  # "solid"|"liquid"|"solution" (validated below)
    density: Optional[Quantity] = None
    concentration: Optional[Quantity] = None
    descriptors: Optional[Dict[str, Any]] = None
    extras: Optional[Dict[str, Any]] = None

    def to_json(self) -> Dict[str, Any]:
        allowed_states = {"solid", "liquid", "solution"}
        if self.physicalstate not in allowed_states:
            raise ValueError(f"Invalid physicalstate='{self.physicalstate}'. Must be one of {allowed_states}")

        d: Dict[str, Any] = {
            "chemicalID": self.chemicalID,
            "chemicalName": self.chemicalName,
            "CASNumber": self.CASNumber,
            "molecularMass": self.molecularMass.to_json(),
            "smiles": self.smiles,
            "Inchi": self.Inchi,
            "molecularFormula": self.molecularFormula,
            "swissCatNumber": self.swissCatNumber,
            "physicalstate": self.physicalstate,
        }
        if self.density is not None:
            d["density"] = self.density.to_json()
        if self.concentration is not None:
            d["concentration"] = self.concentration.to_json()
        if self.descriptors is not None:
            d["descriptors"] = self.descriptors
        if self.extras:
            d["extras"] = self.extras
        return d

# On-disk format: list of chemical dicts (simple + stable)
def _lib_load(path: Union[str, Path]) -> List[Dict[str, Any]]:
    p = _as_path(path)
    if not p.exists():
        return []
    data = _load_json(p)
    return list(data) if isinstance(data, list) else list(data.values())

def _lib_save(path: Union[str, Path], items: List[Dict[str, Any]]) -> None:
    _save_json(items, path)

def _index_by_name(items: List[Dict[str, Any]]) -> Dict[str, int]:
    return {it.get("chemicalName", "").lower(): i for i, it in enumerate(items) if it.get("chemicalName")}

def _add_or_update(items: List[Dict[str, Any]], payload: Dict[str, Any]) -> str:
    name = payload.get("chemicalName", "")
    if not name:
        raise ValueError("chemicalName is required")
    idx = _index_by_name(items).get(name.lower())
    if idx is None:
        items.append(payload)
        return "added"
    items[idx] = payload
    return "updated"

def _q_from_cli(val: Optional[float], unit: Optional[str]) -> Optional[Quantity]:
    if val is None:
        return None
    return Quantity(float(val), unit or "")

def cmd_init(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).resolve()  # Library location is set in input data always

    out = _as_path(str(data_dir/args.out))
    infos = {
        "libraryName": "Chemical Library",
        "dateCreated": datetime.today().strftime('%Y-%m-%d'),
        "createdBy": args.creator,
        "LibraryIndex": args.LibraryID,
    }
    if int(args.LibraryID) == 1:  # This is just to mess with people could be removed
        raise ValueError(f"LibraryID must be unique; Do you really think noone has taken 1 yet? Do better @ {args.creator}!")

    information = [infos]
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists. Use --force to overwrite.")
    _lib_save(out, information)
    print(f"Initialized empty library at {out}")

def cmd_add(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).resolve()

    items = _lib_load(str(data_dir/args.lib))
    density = _q_from_cli(args.density, args.density_unit)
    concentration = _q_from_cli(args.conc, args.conc_unit)

    # Check for double entries and determine ID
    name_lower = args.name.lower()
    if any(c.get("chemicalName", "").lower() == name_lower for c in items):
        raise SystemExit(f"Error: A chemical with name '{args.name}' already exists in {args.lib}")

    max_id = 0
    for c in items:
        try:
            c_id = int(c.get("chemicalID"))
            if c_id > max_id:
                max_id = c_id
        except (ValueError, TypeError):
            print("Warning: Invalid chemicalID found in existing items, skipping ID check.")
            continue

    chem = Chemical(
        chemicalID=str(max_id +1),
        chemicalName=args.name,
        CASNumber=args.cas,
        molecularMass=Quantity(float(args.mw), args.mw_unit or "g/mol"),
        smiles=args.smiles,
        Inchi=args.inchi,
        molecularFormula=args.formula ,
        swissCatNumber=args.swisscatnumber,
        physicalstate=args.physicalstate,
        density=density,
        concentration=concentration,
    )
    action = _add_or_update(items, chem.to_json())
    _lib_save(str(data_dir/args.lib), items)
    print(f"{action.capitalize()} '{args.name}'. Total records: {len(items)}")


def cmd_list(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).resolve()

    items = _lib_load(str(data_dir/args.lib))
    for it in items:
        name = it.get("chemicalName")
        cid = it.get("chemicalID")
        mw = it.get("molecularMass").get("value")
        ps = it.get("physicalstate")
        print(f"- {name}  (ID={cid}, MW={mw}, state={ps})")
    print(f"Total: {(len(items)-1)}")

def cmd_show(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).resolve()

    items = _lib_load(str(data_dir/args.lib))
    idx = _index_by_name(items).get(args.name.lower())
    if idx is None:
        raise SystemExit(f"Not found: {args.name}")
    print(json.dumps(items[idx], indent=2, ensure_ascii=False))

# -------------------------
# Part B: HCI -> Optimizer adapter
# -------------------------

def _coerce_range(r: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not r:
        return None
    out = {
        "min": float(r.get("min")),
        "max": float(r.get("max")),
        "unit": r.get("unit", ""),
    }
    if "step" in r and r["step"] is not None:
        out["step"] = float(r["step"])
    return out

def hci_to_optimizer_dict(hci: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert your HCI JSON (as produced by Campaign.to_json()) into an
    optimizer-friendly dictionary.

    Output structure (neutral, easy to adapt to BayBE/BoFire):
    {
      "metadata": {campaign fields...},
      "globals": {hasRanges...},                      # numeric ranges (e.g., temperature, concentration)
      "groups": [
        {
          "name": str,
          "selectionMode": "one-of"|"any"|...,
          "fixed": bool,
          "equivalents": {"min":..,"max":..,"unit":"eq","step":..},
          "members": [
            {
              "id": str,
              "name": str,
              "reference": {...},                    # original reference block
              "descriptors": {...} | None
            }, ...
          ]
        }, ...
      ],
      "chemicals": {id_or_name: {catalog info...}, ...},
      "parameters": [
        # Categorical choice per group (member names)
        {"name": "group:catalyst", "type": "categorical", "choices": ["Pd(PPh3)4","Pd2(dba)3"], "meta": {...}},
        # Numeric equivalents per group
        {"name": "eq:catalyst", "type": "numerical", "low": 0.01, "high": 0.1, "unit": "eq", "step": 0.01}
        # Plus one numeric per global range (e.g., temperature)
      ]
    }
    """
    doc = hci if isinstance(hci, dict) else _load_json(hci)
    c = doc.get("hasCampaign")

    # Metadata
    metadata = {
        "campaignName": c.get("campaignName"),
        "description": c.get("description"),
        "objective": c.get("objective"),
        "campaignClass": c.get("campaignClass"),
        "type": c.get("type"),
        "reference": c.get("reference"),
        "batch": c.get("hasBatch"),
        "objectiveBlock": c.get("hasObjective"),
    }

    # Globals / ranges
    globals_ranges = {}
    for k, v in (c.get("hasRanges") or {}).items():
        globals_ranges[k] = _coerce_range(v)

    # Chemicals catalog (flatten references, keep by ID or Name as key)
    catalog: Dict[str, Any] = {}
    for ch in (c.get("hasChemical") or []):
        key = ch.get("chemicalName")
        catalog[str(key)] = ch

    # Groups
    groups_out: List[Dict[str, Any]] = []
    parameters: List[Dict[str, Any]] = []

    for g in (c.get("hasGroups") or []):
        gname = g.get("groupName")
        sel = g.get("selectionMode", "one-of")
        fixed = bool(g.get("fixed", True))
        eq = _coerce_range(g.get("equivalents"))

        members_in = g.get("members") or []
        members_out = []
        choices = []
        for m in members_in:
            ref = m.get("reference") or {}
            cid = ref.get("chemicalID")
            name = ref.get("chemicalName")
            desc = ref.get("descriptors")  or None
            members_out.append({
                "id": str(cid),
                "name": name,
                "reference": ref,
                "descriptors": desc if isinstance(desc, dict) else None,
                "overrides": m.get("overrides", {}),
            })
            choices.append(name)

        groups_out.append({
            "name": gname,
            "selectionMode": sel,
            "fixed": fixed,
            "equivalents": eq,
            "members": members_out,
        })

        # One categorical parameter per group (member choice)
        if choices:
            parameters.append({
                "name": f"group:{gname}",
                "type": "categorical",
                "choices": choices,
                "meta": {"selectionMode": sel, "fixed": fixed}
            })

        # One numeric parameter per group's equivalents range (if present)
        if eq:
            p = {
                "name": f"eq:{gname}",
                "type": "numerical",
                "low": eq["min"],
                "high": eq["max"],
                "unit": eq.get("unit", "eq")
            }
            if "step" in eq:
                p["step"] = eq["step"]
            parameters.append(p)

    # Also expose global ranges as numerical parameters (e.g., temperature)
    for k, r in globals_ranges.items():
        if not r:
            continue
        p = {"name": f"global:{k}", "type": "numerical", "low": r["min"], "high": r["max"], "unit": r.get("unit", "")}
        if "step" in r:
            p["step"] = r["step"]
        parameters.append(p)

    optimizer_dict = {
        "metadata": metadata,
        "globals": globals_ranges,
        "groups": groups_out,
        "chemicals": catalog,
        "parameters": parameters,
    }
    return optimizer_dict

# Optional helpers to dump the optimizer dict to JSON for inspection
def cmd_hci_to_optimizer(args: argparse.Namespace) -> None:
    spec = hci_to_optimizer_dict(args.hci)
    if args.out:
        _save_json(spec, args.out)
        print(f"Wrote optimizer spec to {args.out}")
    else:
        print(json.dumps(spec, indent=2, ensure_ascii=False))

# -------------------------
# CLI
# -------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Chemical library tools + HCI -> optimizer adapter")
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--data-dir", default=str(DATA_DIR), help="Directory with data files for layout_parser")

    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("init", help="Initialize an empty chemical library JSON")
    sp.add_argument("--out", required=True)
    sp.add_argument("--force", action="store_true")
    sp.add_argument("--creator", required=True, help="Creator name for the library")
    sp.add_argument("--LibraryID", required=True, help="Library ID, needs to be unique")
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("add", help="Add or update one chemical")
    sp.add_argument("--lib", required=True)
    sp.add_argument("--name", required=True)
    sp.add_argument("--cas", required=True)
    sp.add_argument("--mw", required=True, type=float)
    sp.add_argument("--mw-unit", default="g/mol")
    sp.add_argument("--smiles", required=True)
    sp.add_argument("--inchi", required=True)
    sp.add_argument("--formula", required=True)
    sp.add_argument("--swisscatnumber", required=True)
    sp.add_argument("--physicalstate", choices=["solid","liquid","solution"])
    sp.add_argument("--density", type=float, default=None)
    sp.add_argument("--density-unit", default="g/mL")
    sp.add_argument("--conc", type=float, default=None, help="Concentration value for solutions")
    sp.add_argument("--conc-unit", default="M")
    sp.set_defaults(func=cmd_add)

    sp = sub.add_parser("list", help="List chemicals in a library")
    sp.add_argument("--lib", required=True)
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("show", help="Show one chemical by name")
    sp.add_argument("--lib", required=True)
    sp.add_argument("--name", required=True)
    sp.set_defaults(func=cmd_show)

    sp = sub.add_parser("hci-to-optimizer", help="Convert HCI JSON to optimizer-friendly dict/JSON")
    sp.add_argument("--hci", required=True, help="Path to HCI JSON file")
    sp.add_argument("--out", help="Write result to this path (JSON). If omitted, prints to stdout.")
    sp.set_defaults(func=cmd_hci_to_optimizer)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
