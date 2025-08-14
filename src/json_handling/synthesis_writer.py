#!/usr/bin/env python3
"""
synthesis_writer.py
Turn BO selections into a synthesis.json describing a plate's dispense plan.

This best-effort computes per-component moles, masses (mg), and volumes (µL) when possible:
- If a component has 'equivalents', it uses:
    moles_component = equivalents * moles_limiting_per_well
- If 'molecularMass' is present -> mass = moles * MW
- If 'density' is present and physicalstate is liquid -> volume = mass / density
- If 'concentration' is present (solutions) -> volume = moles / concentration
- Otherwise, leaves fields as null and includes a 'notes' string.
"""

from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import math

# ---------------------------
# Helpers
# ---------------------------

def _save_json(obj: Any, p: str | Path) -> None:
    Path(p).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def _plate_dims(plate_size: int) -> Tuple[int, int]:
    # Common SBS:
    # 24 -> 4x6, 48 -> 6x8, 96 -> 8x12, 384 -> 16x24, 1536 -> 32x48
    mapping = {24: (4, 6), 48: (6, 8), 96: (8, 12), 384: (16, 24), 1536: (32, 48)}
    if plate_size not in mapping:
        # best guess: make it roughly square
        cols = int(math.ceil(math.sqrt(plate_size)))
        rows = int(math.ceil(plate_size / cols))
        return rows, cols
    return mapping[plate_size]

def _well_labels(rows: int, cols: int) -> List[str]:
    labels = []
    for r in range(rows):
        row_letter = chr(ord('A') + r)
        for c in range(1, cols + 1):
            labels.append(f"{row_letter}{c}")
    return labels

def _chem_by_name(opt_spec: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    # chemicals catalog is opt_spec["chemicals"] keyed by id or name; search by chemicalName
    for k, v in opt_spec.get("chemicals", {}).items():
        if v.get("chemicalName") == name:
            return v
    return None

def _member_reference(opt_spec: Dict[str, Any], group_name: str, member_name: str) -> Optional[Dict[str, Any]]:
    # Look up the reference block for a group member by its chemicalName
    for g in opt_spec.get("groups", []):
        if g["name"] != group_name:
            continue
        for m in g.get("members", []):
            ref = m.get("reference") or {}
            if ref.get("chemicalName") == member_name:
                return ref
    return None

def _get_float(d: Dict[str, Any], path: List[str]) -> Optional[float]:
    cur = d
    for key in path:
        cur = cur.get(key) if isinstance(cur, dict) else None
        if cur is None:
            return None
    try:
        return float(cur)
    except Exception:
        return None

# ---------------------------
# Core
# ---------------------------

# ----- replace in synthesis_writer.py -----

def _enumerate_plate(rows: int, cols: int) -> list[str]:
    """A1, A2, ..., B1, ... (column-wise or row-wise if you prefer)."""
    labels = []
    for r in range(rows):
        row_letter = chr(ord('A') + r)
        for c in range(1, cols + 1):
            labels.append(f"{row_letter}{c}")
    return labels

def _exp_key(n: int, start: int = 1) -> str:
    """Return the experiment key string: '1', '2', ..."""
    return str(start + n)

def _non_group_catalog_chemicals(opt_spec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return {chemicalName: reference_block} for chemicals present in catalog but not in any group."""
    # names in groups
    group_names = set()
    for g in opt_spec.get("groups", []):
        for m in g.get("members", []) or []:
            ref = m.get("reference") or {}
            nm = ref.get("chemicalName")
            if nm:
                group_names.add(nm)

    # catalog entries keyed by chemicalName
    out = {}
    for _, ref in (opt_spec.get("chemicals") or {}).items():
        nm = ref.get("chemicalName")
        if nm and nm not in group_names:
            out[nm] = ref
    return out

def write_synthesis_json(
    opt_spec: Dict[str, Any],
    selections: List[Dict[str, Any]],
    out_path: str | Path,
    plate_size: int,
    *,
    limiting_name: Optional[str] = None,
    limiting_moles: Optional[float] = None,
    limiting_conc: Optional[float] = None,   # M
    well_volume_uL: Optional[float] = None,
    experiment_start_index: int = 1,          # experiments numbered 1..N
    fixed_equivalents_for_catalog: Optional[Dict[str, float]] = None,  # Non-SM and not in any group in hci
    default_catalog_equivalents: Optional[float] = None,
) -> None:
    """
    Build synthesis.json where the primary entries are keyed by **experiment number**.
    Plate (rows/cols) and well labels are kept in meta as extra info.
    """
    # Determine moles of limiting reagent
    moles_lim: Optional[float] = None
    lim_basis = None
    if limiting_moles is not None:
        moles_lim = float(limiting_moles)
        lim_basis = "moles"
    elif limiting_conc is not None and well_volume_uL is not None:
        moles_lim = float(limiting_conc) * (float(well_volume_uL) * 1e-6)  # mol/L * L
        lim_basis = "concentration"

    # Plate layout (info only)
    rows, cols = _plate_dims(plate_size)
    well_labels = _enumerate_plate(rows, cols)

    # Trim selections to plate_size
    selections = selections[:plate_size]

    # Build experiments keyed by number
    experiments: Dict[str, Any] = {}
    exp_to_well_map: list[Dict[str, Any]] = []

    for i, sel in enumerate(selections):
        exp_key = _exp_key(i, start=experiment_start_index)
        well_label = well_labels[i] if i < len(well_labels) else None

        dispenses = []
        for gname, gsel in sel.get("groups", {}).items():
            member = gsel.get("member")
            eq = gsel.get("equivalents")

            # reference lookup
            ref = _member_reference(opt_spec, gname, member) or {}
            phys = ref.get("physicalstate")
            mw = _get_float(ref, ["molecularMass", "value"])
            dens = _get_float(ref, ["density", "value"])
            conc = _get_float(ref, ["concentration", "value"])

            notes = []
            moles = mass_mg = vol_uL = None

            if eq is not None and moles_lim is not None:
                moles = float(eq) * moles_lim
            elif eq is not None:
                notes.append("equivalents provided but limiting moles unknown")

            if moles is not None and mw is not None:
                mass_mg = moles * mw * 1e3
            elif moles is not None:
                notes.append("no molecular mass; cannot compute mass")

            if moles is not None and conc:
                vol_uL = (moles / conc) * 1e6
            elif mass_mg is not None and dens and phys in {"liquid", "solution"}:
                vol_uL = (mass_mg / 1e3) / dens * 1e6
            else:
                if phys in {"liquid", "solution"} and dens is None and conc is None:
                    notes.append("no density/concentration; cannot compute volume")

            dispenses.append({
                "group": gname,
                "chemicalID": ref.get("chemicalID"),
                "chemicalName": ref.get("chemicalName"),
                "physicalstate": phys,
                "equivalents": eq,
                "moles": moles,
                "mass_mg": mass_mg,
                "volume_uL": vol_uL,
                "reference": ref,
                "notes": "; ".join(notes) if notes else ""
            })

        if i == 0:
            non_group_catalog = _non_group_catalog_chemicals(opt_spec)

            # 2) Append each such chemical to dispenses
        for cat_name, ref in non_group_catalog.items():
            # Decide equivalents:
            # - If it's the limiting reagent (by name) -> 1.0 unless overridden
            # - Else use fixed_equivalents_for_catalog[cat_name] if provided
            # - Else use default_catalog_equivalents (if provided)
            eq = None
            if limiting_name and cat_name == limiting_name:
                eq = 1.0
            if fixed_equivalents_for_catalog and cat_name in fixed_equivalents_for_catalog:
                eq = float(fixed_equivalents_for_catalog[cat_name])
            elif eq is None and default_catalog_equivalents is not None:
                eq = float(default_catalog_equivalents)

            phys = ref.get("physicalstate", "")
            mw = _get_float(ref, ["molecularMass", "value"])
            dens = _get_float(ref, ["density", "value"])
            conc = _get_float(ref, ["concentration", "value"])

            notes = []
            moles = mass_mg = vol_uL = None

            if eq is not None and moles_lim is not None:
                moles = float(eq) * moles_lim
            elif eq is not None:
                notes.append("equivalents provided but limiting moles unknown")
            elif limiting_name and cat_name == limiting_name and moles_lim is not None:
                # limiting reagent with implicit 1.0 eq
                eq = 1.0
                moles = 1.0 * moles_lim
            else:
                notes.append("no equivalents specified; skipping quantity computation")

            if moles is not None and mw is not None:
                mass_mg = moles * mw * 1e3
            elif moles is not None:
                notes.append("no molecular mass; cannot compute mass")

            if moles is not None and conc:
                vol_uL = (moles / conc) * 1e6
            elif mass_mg is not None and dens and phys in {"liquid", "solution"}:
                vol_uL = (mass_mg / 1e3) / dens * 1e6
            else:
                if phys in {"liquid", "solution"} and dens is None and conc is None:
                    notes.append("no density/concentration; cannot compute volume")

            dispenses.append({
                "group": None,  # not part of a BO group
                "chemicalID": ref.get("chemicalID"),
                "chemicalName": cat_name,
                "physicalstate": phys,
                "equivalents": eq,
                "moles": moles,
                "mass_mg": mass_mg,
                "volume_uL": vol_uL,
                "reference": ref,
                "notes": "; ".join(notes) if notes else ""
            })

        experiments[exp_key] = {
            "globals": sel.get("globals", {}),
            "dispenses": dispenses
            # Note: no 'well' here; experiment number is the primary key
        }

        exp_to_well_map.append({"experiment": exp_key, "well": well_label})

    synthesis = {
        "meta": {
            "layout": {"rows": rows, "cols": cols, "plate_size": plate_size},
            "well_labels": well_labels,
            "experiment_to_well": exp_to_well_map,
            "limiting": {
                "name": limiting_name,
                "basis": lim_basis,
                "moles_per_well": moles_lim,
                "well_volume_uL": well_volume_uL
            }
        },
        "experiments": experiments
    }

    _save_json(synthesis, out_path)

