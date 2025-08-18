import argparse
import importlib.util
import string
from dataclasses import dataclass, field
from operator import truediv
from typing import List, Optional, Dict, Tuple, Any, cast
import math
from pathlib import Path
from matplotlib.axes import Axes

from rdkit import Chem
from rdkit.Chem import inchi

import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

from hte_workflow.paths import DATA_DIR, OUT_DIR, ensure_dirs
from json_handling.library_and_hci_adapter import hci_to_optimizer_dict
from json_handling.synthesis_writer import write_synthesis_json

PUBCHEM_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{}/property/MolecularWeight/TXT"
)


def fetch_molar_mass(inchikey: str) -> float:
    """Return molecular weight from PubChem given an InChIKey."""
    response = requests.get(PUBCHEM_URL.format(inchikey))
    response.raise_for_status()
    text = response.text.strip()
    # PubChem occasionally returns the value twice separated by a newline,
    # so parse only the first line to avoid conversion errors
    first_line = text.splitlines()[0]
    return float(first_line)


def _ask_with_default(prompt: str, default: str | None) -> str:
    """Prompt once; ENTER keeps the default."""
    sfx = f" [{default}]" if (default is not None and default != "") else ""
    val = input(f"{prompt}{sfx}: ").strip()
    return default if (val == "" and default is not None) else val

def _ask_float_with_default(prompt: str, default: float | None) -> float | None:
    while True:
        sfx = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt}{sfx}: ").strip()
        if raw == "":
            return default
        try:
            return float(raw)
        except ValueError:
            print("Please enter a number or press ENTER to keep default.")

def load_preloaded_reagents(path: str, plate: "Plate") -> Tuple[List["Reagent"], List["Solvent"]]:
    """Load reagents from a Python file and return lists of Reagent and Solvent objects."""
    if not path:
        return [], []
    spec = importlib.util.spec_from_file_location("preloaded", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = getattr(module, "PRELOADED_REAGENTS", [])
    reagents: List[Reagent] = []
    solvents: List[Solvent] = []
    for entry in data:
        if isinstance(entry, Reagent):
            reagent = entry
        else:
            reagent = Reagent(
                name=entry.get("name", ""),
                inchikey=entry.get("inchikey", ""),
                rtype=entry.get("rtype", ""),
                equivalents=entry.get("equivalents", 0.0),
                is_limiting=entry.get("is_limiting"),
                density=entry.get("density"),
                concentration=entry.get("concentration"),
                stock_solution=entry.get("stock_solution"),
            )
        if not reagent.name:
            reagent.name = input("Reagent name: ").strip()
        if not reagent.inchikey:
            reagent.inchikey = input(f"InChIKey for {reagent.name}: ").strip()
        if not reagent.rtype:
            reagent.rtype = input(f"Type for {reagent.name} [solid/liquid/solvent]: ").strip().lower()
        if reagent.rtype == "solvent":
            solv = Solvent(name=reagent.name, inchikey=reagent.inchikey)
            solv.locations = plate.parse_location(f"solvent {solv.name}")
            solvents.append(solv)
            continue
        if reagent.equivalents == 0.0:
            reagent.equivalents = float(input(f"Equivalents for {reagent.name}: "))
        if reagent.rtype == "liquid":
            if reagent.density is None:
                d = input(f"Density for {reagent.name} (g/mL, blank if not known): ").strip()
                if d:
                    reagent.density = float(d)
            if reagent.concentration is None:
                c = input(f"Concentration for {reagent.name} (mol/L, blank if not known): ").strip()
                if c:
                    reagent.concentration = float(c)
        if reagent.is_limiting is None:
            is_limiting = input(f"Is {reagent.name} the limiting reagent? [y/N]: ").strip().lower()
            reagent.is_limiting = is_limiting == 'y'
        if reagent.stock_solution is None:
            stock_solution = input(f"Is {reagent.name} a stock solution? [y/N]: ").strip().lower()
            reagent.stock_solution = stock_solution == 'y'
        reagent.locations = plate.parse_location(f"reagent {reagent.name}")
        reagents.append(reagent)
    return reagents, solvents



@dataclass
class Reagent:
    name: str
    inchikey: str
    rtype: str  # 'solid' or 'liquid'
    equivalents: float
    is_limiting: bool = False
    density: Optional[float] = None  # g/mL
    concentration: Optional[float] = None  # mol/L
    locations: pd.DataFrame = field(default_factory=pd.DataFrame)
    molar_mass: Optional[float] = None
    stock_solution: Optional[bool] =False
    moles: pd.DataFrame = field(default_factory=pd.DataFrame)


    def ensure_molar_mass(self) -> None:
        if self is not None:
            return
        if self.inchikey:
            self.molar_mass = fetch_molar_mass(self.inchikey)
        else:
            raise ValueError(f"Molar mass unknown for {self.name}")


@dataclass
class Solvent:
    name: str
    inchikey: str
    locations: pd.DataFrame = field(default_factory=pd.DataFrame)

@dataclass
class Stock_Solution:
    name: str
    reagent: Reagent
    solvent: Solvent
    locations: pd.DataFrame = field(default_factory=pd.DataFrame)
    concentration: Optional[float] = None
    volume_dispensed: pd.DataFrame = field(default_factory=pd.DataFrame)


class Plate:
    def __init__(self, rows: int, cols: int) -> None:
        self.rows = rows
        self.cols = cols
        row_labels = list(string.ascii_uppercase[:rows])
        col_labels = list(range(1, cols + 1))
        self.template = pd.DataFrame(False, index=row_labels, columns=col_labels)

    def parse_location(self, prompt: str) -> pd.DataFrame:
        print(f"\nSpecify location for {prompt}:")
        print("  [a] all wells")
        print("  [c] specific columns")
        print("  [r] specific rows")
        print("  [w] individual wells (e.g., A1,B2)")
        choice = input("Choice: ").strip().lower()
        mat = self.template.copy()
        if choice == 'a':
            mat.loc[:, :] = True
        elif choice == 'c':
            cols = input("Enter columns separated by comma: ").split(',')
            cols = [int(c.strip()) for c in cols if c.strip()]
            mat.loc[:, cols] = True
        elif choice == 'r':
            rows = input("Enter rows separated by comma (A,B,...): ").split(',')
            rows = [r.strip().upper() for r in rows if r.strip()]
            mat.loc[rows, :] = True
        elif choice == 'w':
            wells = input("Enter wells separated by comma (e.g., A1,B2): ").split(',')
            for w in wells:
                w = w.strip()
                if not w:
                    continue
                row = w[0].upper()
                col = int(w[1:])
                mat.loc[row, col] = True
        else:
            print("Invalid choice; none selected")
        return mat

    def input_matrix(self, label: str) -> pd.DataFrame:
        print(f"\nInput method for {label}:")
        print("  [c] constant across all wells")
        print("  [r] per row")
        print("  [C] per column")
        print("  [w] per well")
        choice = input("Choice: ").strip().lower()
        df = pd.DataFrame(0.0, index=self.template.index, columns=self.template.columns)
        if choice == 'c':
            val = float(input(f"Enter {label}: "))
            df.loc[:, :] = val
        elif choice == 'r':
            for row in self.template.index:
                val = float(input(f"{label} for row {row}: "))
                df.loc[row, :] = val
        elif choice == 'C':
            for col in self.template.columns:
                val = float(input(f"{label} for column {col}: "))
                df.loc[:, col] = val
        elif choice == 'w':
            for row in self.template.index:
                for col in self.template.columns:
                    val = float(input(f"{label} for well {row}{col}: "))
                    df.loc[row, col] = val
        else:
            print("Invalid choice; defaulting to 0")
        return df

def _iter_wells(rows: int, cols: int):
    for r in range(rows):
        row_letter = string.ascii_uppercase[r]
        for c in range(1, cols + 1):
            yield r, c, f"{row_letter}{c}"

def build_selections_from_interactive(
    *,
    rows: int,
    cols: int,
    reagents: List[Reagent],
    solvents: List[Solvent],
    conc_limiting_df: pd.DataFrame,  # M, per well
    temperature_plate: Optional[float],
    time_plate: Optional[float],
    hci_groups: List[Dict],          # from HCI: hasGroups
) -> List[Dict[str, Any]]:
    """
    Build 'selections' for synthesis_writer with BOTH reagents and solvents.
    Reagents: chosen by presence in well & membership in their HCI group.
    Solvent group (if present in HCI): chosen by Solvent.locations for that well.
    """
    # Normalize column dtypes once (so .loc[row_letter, col_number] works)
    try:
        conc_limiting_df.columns = conc_limiting_df.columns.astype(int)
    except Exception:
        pass
    for r in reagents:
        if not getattr(r, "locations", pd.DataFrame()).empty:
            try:
                r.locations.columns = r.locations.columns.astype(int)
            except Exception:
                pass
    for s in solvents:
        if not getattr(s, "locations", pd.DataFrame()).empty:
            try:
                s.locations.columns = s.locations.columns.astype(int)
            except Exception:
                pass

    # Index HCI group memberships
    group_to_members: Dict[str, set] = {}
    lower_to_actual_name: Dict[str, str] = {}
    for g in hci_groups or []:
        gname = (g.get("groupName") or g.get("name") or "").strip()
        if not gname:
            continue
        lower_to_actual_name[gname.lower()] = gname
        members = []
        for m in g.get("members", []) or []:
            ref = m.get("reference") or {}
            nm = ref.get("chemicalName")
            if nm:
                members.append(nm)
        group_to_members[gname] = set(members)

    # Detect the actual "solvent" group name in HCI, if present
    solvent_group_name: Optional[str] = None
    if "solvent" in lower_to_actual_name:
        solvent_group_name = lower_to_actual_name["solvent"]

    # Quick lookups
    reag_by_name = {r.name: r for r in reagents}
    solvent_names = {s.name for s in solvents}

    selections: List[Dict[str, Any]] = []

    warned_multi_solvent = False  # print this warning only once

    for r_idx in range(rows):
        row_letter = string.ascii_uppercase[r_idx]
        for c_idx in range(1, cols + 1):
            col_label = c_idx

            # Per-well globals
            globals_block: Dict[str, Any] = {}
            if not conc_limiting_df.empty:
                try:
                    conc_here = float(conc_limiting_df.loc[row_letter, col_label])
                    globals_block["concentration"] = conc_here
                except KeyError:
                    pass
            if temperature_plate is not None:
                globals_block["temperature"] = float(temperature_plate)
            if time_plate is not None:
                globals_block["time"] = float(time_plate)

            groups_block: Dict[str, Dict[str, Any]] = {}

            # Reagent-driven group membership (one member per group)
            for gname, member_names in group_to_members.items():
                if gname == solvent_group_name:
                    # handle solvent group separately below
                    continue

                chosen_name: Optional[str] = None
                chosen_eq: Optional[float] = None

                for nm in member_names:
                    r = reag_by_name.get(nm)
                    if r is None or r.locations.empty:
                        continue
                    try:
                        if bool(r.locations.loc[row_letter, col_label]):
                            chosen_name = r.name
                            chosen_eq = r.equivalents
                            break
                    except KeyError:
                        continue

                if chosen_name is not None:
                    entry = {"member": chosen_name}
                    if chosen_eq is not None:
                        entry["equivalents"] = float(chosen_eq)
                    groups_block[gname] = entry

            # Solvent group selection (if present in HCI)
            if solvent_group_name:
                # Which placed solvents are in this well?
                placed_here: List[str] = []
                for s in solvents:
                    if s.locations.empty:
                        continue
                    try:
                        if bool(s.locations.loc[row_letter, col_label]):
                            placed_here.append(s.name)
                    except KeyError:
                        continue

                # Filter to those that are valid members of the solvent group
                valid_members = group_to_members.get(solvent_group_name, set())
                candidates = sorted([nm for nm in placed_here if nm in valid_members])

                if len(candidates) == 1:
                    groups_block[solvent_group_name] = {"member": candidates[0]}
                elif len(candidates) > 1:
                    # pick deterministically (alphabetical) and warn once
                    if not warned_multi_solvent:
                        print("Warning: multiple solvents placed in some wells; "
                              f"choosing the first alphabetically for group '{solvent_group_name}'.")
                        warned_multi_solvent = True
                    groups_block[solvent_group_name] = {"member": candidates[0]}
                # else: no solvent placed here (or not in group) → leave unset for this well

            selections.append({"groups": groups_block, "globals": globals_block})

    return selections

def visualize_distribution(
    reagents: List[Reagent],
    solvents: List[Solvent],
    plate: Plate,
    final_volume: pd.DataFrame,
    output: str,
    *,
    vmax_percentile: Optional[float] = None,  # e.g. 99.0 to cap outliers; None = use absolute max
) -> None:
    """
    Heatmaps for reagents are colored by concentration (M) on a **shared** color scale.
    Solvents remain binary maps (Blues). If final_volume has zeros, those wells will show 0 M.
    """
    items = reagents + solvents
    if not items:
        return

    # --- Precompute reagent concentrations to find global vmax ---
    conc_arrays: List[np.ndarray] = []
    for item in reagents:
        if item.moles is not None and not item.moles.empty:
            conc = (item.moles / (final_volume / 1_000_000)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            conc_arrays.append(conc.values.astype(float))

    if conc_arrays:
        all_vals = np.concatenate([arr.ravel() for arr in conc_arrays])
        all_vals = all_vals[np.isfinite(all_vals)]
        if all_vals.size == 0:
            global_vmax = 1.0
        elif vmax_percentile is not None:
            global_vmax = float(np.percentile(all_vals, vmax_percentile))
            if global_vmax <= 0:
                global_vmax = float(all_vals.max(initial=1.0))
        else:
            global_vmax = float(all_vals.max(initial=1.0))
    else:
        global_vmax = 1.0  # fallback

    # --- Plot grid ---
    n = len(items)
    cols = min(4, n)
    rows = int(math.ceil(n / cols))
    fig, axarr = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))

    axes_list = np.atleast_1d(axarr).ravel().tolist()
    axes: List[Axes] = cast(List[Axes], axes_list)

    # Turn off extra axes
    for ax in axes[n:]:
        ax.axis('off')

    # Draw
    for ax, item in zip(axes, items):
        if isinstance(item, Reagent) and item.moles is not None and not item.moles.empty:
            conc = (item.moles / (final_volume / 1_000_000)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            im = ax.imshow(conc.values, cmap="viridis", vmin=0.0, vmax=global_vmax)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("M")
        else:
            # solvents (and any non-reagent) remain binary maps
            mat = item.locations.astype(int) if hasattr(item, "locations") else plate.template*0
            im = ax.imshow(mat.values, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(plate.cols))
        ax.set_xticklabels(plate.template.columns)
        ax.set_yticks(range(plate.rows))
        ax.set_yticklabels(plate.template.index)
        ax.set_title(item.name)

    plt.tight_layout()
    plt.savefig(output)
    plt.close(fig)


def ensure_molar_mass(self) -> None:
    # Prefer pre-populated molar mass (e.g., from HCI), otherwise fetch by InChIKey if possible
    if self.molar_mass is not None:
        return
    if self.inchikey:
        self.molar_mass = fetch_molar_mass(self.inchikey)
    else:
        raise ValueError(f"Molar mass unknown for {self.name}; provide InChIKey or preload molar_mass.")

# ---------- synthesis.json pipeline ----------

def _df_bool(rows: List[str], cols: List[int], default=False) -> pd.DataFrame:
    return pd.DataFrame(default, index=rows, columns=cols)

def _well_labels(rows: int, cols: int) -> List[str]:
    return [f"{string.ascii_uppercase[r]}{c}" for r in range(rows) for c in range(1, cols+1)]

def _experiments_to_plate_map(meta: Dict) -> Dict[str, str]:
    """
    meta["experiment_to_well"]: [{experiment: "1", well: "A1"}, ...]
    Returns {"1":"A1", "2":"A2", ...}
    """
    mapping = {}
    for item in meta.get("experiment_to_well", []):
        mapping[str(item["experiment"])] = item.get("well")
    return mapping

def _empty_plate_like(rows: int, cols: int, fill=0.0) -> pd.DataFrame:
    return pd.DataFrame(fill, index=list(string.ascii_uppercase[:rows]), columns=list(range(1, cols+1)))

def load_synthesis_and_emit_outputs(
    synthesis_path: Path,
    out_dir: Path,
    reaction_name: Optional[str] = None,
) -> None:
    """
    Read synthesis.json (experiment-keyed), compute per-well tables and totals, and write Excel + visualization.
    - Stock solutions are assumed already expanded inside synthesis.json (we don't re-derive).
    - Visualization also includes per-well heatmaps for 'temperature' and 'time' when present.
    """
    synth = pd.read_json(synthesis_path, typ="dictionary")
    meta = synth["meta"]
    exps: Dict[str, Any] = synth["experiments"]

    # Plate layout
    rows = int(meta["layout"]["rows"])
    cols = int(meta["layout"]["cols"])
    well_volume_uL_global = meta.get("limiting", {}).get("well_volume_uL")  # may be used as a check only
    exp2well = _experiments_to_plate_map(meta)
    row_index = list(string.ascii_uppercase[:rows])
    col_index = list(range(1, cols+1))

    # Build per-well globals (temperature/time/concentration if present)
    globals_keys = set()
    for e in exps.values():
        globals_keys.update((e.get("globals") or {}).keys())

    globals_frames: Dict[str, pd.DataFrame] = {
        k: _empty_plate_like(rows, cols, fill=np.nan) for k in globals_keys
    }

    # Collect dispenses → one DataFrame per chemical (uL or mg), and “moles” if present
    per_chem_vol: Dict[str, pd.DataFrame] = {}
    per_chem_mass: Dict[str, pd.DataFrame] = {}
    per_chem_moles: Dict[str, pd.DataFrame] = {}

    # Final per-well volume = sum of all liquid volumes reported in dispersion entries (uL)
    final_volume = _empty_plate_like(rows, cols, fill=0.0)

    # Pass over experiments
    for exp_key, payload in exps.items():
        well = exp2well.get(str(exp_key))
        if not well:
            continue
        row_label, col_label = well[0], int(well[1:])
        # globals
        for k, v in (payload.get("globals") or {}).items():
            globals_frames[k].loc[row_label, col_label] = float(v)

        # dispenses
        for d in payload.get("dispenses", []):
            chem = d.get("chemicalName") or d.get("chemicalID") or "unknown"
            vol = d.get("volume_uL")
            mass_mg = d.get("mass_mg")
            moles = d.get("moles")

            if vol is not None:
                df = per_chem_vol.setdefault(chem, _empty_plate_like(rows, cols, fill=0.0))
                df.loc[row_label, col_label] += float(vol)
                final_volume.loc[row_label, col_label] += float(vol)

            if mass_mg is not None:
                dfm = per_chem_mass.setdefault(chem, _empty_plate_like(rows, cols, fill=0.0))
                dfm.loc[row_label, col_label] += float(mass_mg)

            if moles is not None:
                dfmol = per_chem_moles.setdefault(chem, _empty_plate_like(rows, cols, fill=0.0))
                dfmol.loc[row_label, col_label] += float(moles)

    # Build Excel sheets-like structure
    results: Dict[str, pd.DataFrame] = {}
    totals: Dict[str, float] = {}

    # Volumes (uL)
    for chem, df in per_chem_vol.items():
        key = f"{chem} (uL)"
        results[key] = df
        totals[key] = float(df.sum().sum())

    # Masses (mg)
    for chem, df in per_chem_mass.items():
        key = f"{chem} (mg)"
        results[key] = df
        totals[key] = float(df.sum().sum())

    # Moles
    for chem, df in per_chem_moles.items():
        key = f"{chem} (mmol)"
        results[key] = df / 1000.0  # optional: change units or keep as "mol" if you prefer

    # Per-well globals into results (for convenience; also used in visualization)
    for k, df in globals_frames.items():
        results[f"[global] {k}"] = df

    # Write Excel
    reaction_name = reaction_name or "synthesis"
    output_file = out_dir / f"{reaction_name}"
    with pd.ExcelWriter(output_file) as writer:
        # Per-well sheet (multicol — same as your previous structure)
        per_well = pd.concat(results, axis=1)
        per_well.index.name = "Row"
        per_well.to_excel(writer, sheet_name="per_well")

        # Totals
        if totals:
            pd.Series(totals, name="Total").to_frame().to_excel(writer, sheet_name="totals")

    # Visualization:
    # 1) heatmap per chemical based on moles / final_volume (M), if moles present
    # 2) heatmaps for temperature and time if present
    viz_file = out_dir / f"{reaction_name}_layout.png"
    visualize_distribution_synthesis(per_chem_moles, final_volume, rows, cols, results_globals=globals_frames, out_path=viz_file)
    print(f"Results written to {output_file}\nLayout visualization saved to {viz_file}")

def visualize_distribution_synthesis(
    per_chem_moles: Dict[str, pd.DataFrame],
    final_volume: pd.DataFrame,
    rows: int, cols: int,
    results_globals: Optional[Dict[str, pd.DataFrame]] = None,
    out_path: Path = Path("layout.png"),
    *,
    vmax_percentile: Optional[float] = None,   # e.g. 99.0 to cap outliers; None = absolute max
) -> None:
    """
    Grid of heatmaps: chemicals by concentration (M) with a **shared** color scale,
    plus global fields (temperature/time/etc.) with their own independent scales.
    """
    chem_names = list(per_chem_moles.keys())
    global_names = list((results_globals or {}).keys())

    # --- Global vmax across *all chemical* concentration maps ---
    conc_arrays: List[np.ndarray] = []
    for nm in chem_names:
        mol = per_chem_moles[nm]
        conc = (mol / (final_volume / 1_000_000)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        conc_arrays.append(conc.values.astype(float))

    if conc_arrays:
        all_vals = np.concatenate([a.ravel() for a in conc_arrays])
        all_vals = all_vals[np.isfinite(all_vals)]
        if all_vals.size == 0:
            global_vmax = 1.0
        elif vmax_percentile is not None:
            global_vmax = float(np.percentile(all_vals, vmax_percentile))
            if global_vmax <= 0:
                global_vmax = float(all_vals.max(initial=1.0))
        else:
            global_vmax = float(all_vals.max(initial=1.0))
    else:
        global_vmax = 1.0

    # --- Layout ---
    n_items = len(chem_names) + len(global_names)
    if n_items == 0:
        return

    ncols = min(4, max(1, n_items))
    nrows = int(math.ceil(n_items / ncols))
    fig, axarr = plt.subplots(nrows, ncols, figsize=(ncols*3.2, nrows*3.2))

    axes_list = np.atleast_1d(axarr).ravel().tolist()
    axes: List[Axes] = cast(List[Axes], axes_list)

    idx = 0
    # Chemicals (shared vmin/vmax)
    for chem in chem_names:
        ax = axes[idx]; idx += 1
        mol = per_chem_moles[chem]
        conc = (mol / (final_volume / 1_000_000)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        im = ax.imshow(conc.values, cmap="viridis", vmin=0.0, vmax=global_vmax)
        ax.set_title(chem)
        ax.set_xticks(range(cols)); ax.set_yticks(range(rows))
        ax.set_xticklabels(range(1, cols+1)); ax.set_yticklabels(list(string.ascii_uppercase[:rows]))
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.set_label("M")

    # Globals (own scales)
    for g in global_names:
        ax = axes[idx]; idx += 1
        df = results_globals[g].astype(float)
        im = ax.imshow(df.values, cmap="coolwarm")
        ax.set_title(g)
        ax.set_xticks(range(cols)); ax.set_yticks(range(rows))
        ax.set_xticklabels(range(1, cols+1)); ax.set_yticklabels(list(string.ascii_uppercase[:rows]))
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.set_label(g)

    # Hide unused axes
    for j in range(idx, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

# ---------- HCI preload helpers ----------

def preload_from_hci(hci_path: Path, plate: "Plate") -> Tuple[List["Reagent"], List["Solvent"]]:
    """
    Parse HCI JSON (your hasCampaign shape), construct preloaded Reagent/Solvent instances.
    - chemicalName, physicalstate -> map to rtype ('solid'/'liquid'/'solvent')
    - density/concentration/molecularMass if present
    - InChIKey may be absent; if so, molar_mass must be present in HCI to avoid fetch.
    """
    data = pd.read_json(hci_path, typ="dictionary")
    c = data["hasCampaign"]

    reagents: List[Reagent] = []
    solvents: List[Solvent] = []

    # Catalog chemicals (not necessarily in groups)
    for ref in c.get("hasChemical", []):
        name = ref.get("chemicalName") or ref.get("chemicalID")
        phys = (ref.get("physicalstate") or "").lower()
        rtype = "solid"
        if phys in {"liquid", "solution"}:
            rtype = "liquid"

        mm = None
        if isinstance(ref.get("molecularMass"), dict):
            try:
                mm = float(ref["molecularMass"].get("value"))
            except Exception:
                pass
        dens = None
        if isinstance(ref.get("density"), dict):
            try:
                dens = float(ref["density"].get("value"))
            except Exception:
                pass
        conc = None
        if isinstance(ref.get("concentration"), dict):
            try:
                conc = float(ref["concentration"].get("value"))
            except Exception:
                pass

        inchikey = ref.get("InchiKey") or ref.get("InChIKey") or ""  # HCI may not carry this

        # Heuristic: if name looks like a solvent (or physicalstate == 'solution' with no MW needed), let the user place it as solvent
        # We'll default to reagent; user can reclassify when prompted for "Type" if needed.
        reagent = Reagent(
            name=str(name),
            inchikey=inchikey,
            rtype=rtype,
            equivalents=1.0,       # default; user will adjust interactively
            is_limiting=False,
            density=dens,
            concentration=conc,
            molar_mass=mm,
        )
        # Ask for placement on plate during interactive flow; here we only scaffold.
        reagents.append(reagent)

    # Groups can include solvents explicitly — if you’d like to pre-mark solvents from group "solvent", do it here:
    for g in c.get("hasGroups", []):
        if g.get("groupName", "").lower() == "solvent":
            for m in g.get("members", []):
                ref = m.get("reference") or {}
                nm = ref.get("chemicalName")
                if nm:
                    # create a Solvent entry (placement still asked interactively)
                    solvents.append(Solvent(name=nm, inchikey=ref.get("InChIKey","")))

    return reagents, solvents


def main() -> None:
    parser = argparse.ArgumentParser(description="HTE Dispense Calculator")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Directory with data files for layout_parser")
    parser.add_argument("--output", default=None, help="Excel output file")
    parser.add_argument("--preload", default=None, help="Path to python file with PRELOADED_REAGENTS list")
    parser.add_argument("--synthesis", default=None, help="Synthesis.json file")
    parser.add_argument("--hci", default=None, help="HCI file")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.synthesis:
        synth_path = str(out_dir / args.synthesis)
        synth_path = Path(synth_path)
        if not synth_path.exists():
            raise FileNotFoundError(f"synthesis.json file not found: {synth_path}")
        reaction_name = args.output
        ensure_dirs()
        load_synthesis_and_emit_outputs(synth_path, out_dir, reaction_name=reaction_name)
        return

    if not args.hci and not args.preload:
        raise SystemExit("When --synthesis is not given, please supply --hci (preferred) or --preload.")

    layout = input("Plate layout [24/48/96/custom]: ").strip()
    while layout not in {"24", "48", "96", "custom"}:
        layout = input("Please enter 24, 48, 96 or custom: ").strip()

    if layout == 'custom':
        dims = input("Enter custom plate dimensions (columns x rows or single number, e.g. 12x8 or 6): ").strip().lower()
        while True:
            try:
                if 'x' in dims:
                    cols, rows = map(int, dims.split('x'))
                else:
                    cols = int(dims)
                    rows = 1
                if cols <= 0 or rows <= 0:
                    raise ValueError("Dimensions must be positive integers")
                break
            except ValueError:
                dims = input("Invalid dimensions. Please enter in format 'columns x rows' or single number: ").strip().lower()
    else:
        layout = int(layout)
        mapping = {24: (4, 6), 48: (6, 8), 96: (8, 12)}
        rows, cols = mapping[layout]
    plate = Plate(rows, cols)

    reaction_name = input("Reaction name: ").strip()

    output_file = Path(out_dir / args.output) if args.output else out_dir / f"{reaction_name}.xlsx"

    reagents: List[Reagent] = []
    solvents: List[Solvent] = []
    reagents_hci: List[Reagent] = []
    solvents_hci: List[Solvent] = []
    stock_solutions: List[Stock_Solution] = []
    hci_catalog: dict[str, dict] = {}

    # Preferred preload source: HCI

    if args.hci:
        import json
        try:
            hci_path = str(out_dir / args.hci)
            pre_r, pre_s = preload_from_hci(Path(hci_path), plate)
            reagents_hci.extend(pre_r)
            solvents_hci.extend(pre_s)
            if reagents or solvents:
                print(
                    f"Preloaded from HCI: {len(reagents)} reagents, {len(solvents)} solvents (you can edit/confirm in prompts).")
            hci_doc = json.loads(Path(hci_path).read_text(encoding="utf-8"))
            camp = hci_doc.get("hasCampaign", {})
            for ref in camp.get("hasChemical", []):
                nm = (ref.get("chemicalName") or ref.get("chemicalID") or "").strip()
                if not nm:
                    continue
                hci_catalog[nm] = ref  # keep the full reference block
        except Exception as e:
            print(f"Warning: failed to preload from HCI: {e}")

    # Legacy preload (optional)
    if args.preload:
        loaded_reagents, loaded_solvents = load_preloaded_reagents(args.preload, plate)
        reagents.extend(loaded_reagents)
        solvents.extend(loaded_solvents)

    limiting_set = any(r.is_limiting for r in reagents)
    while True:
        while True:
            name = input("Reagent name (blank to finish): ").strip()
            if not name:
                break
            if name.lower() == "list" and hci_catalog:
                print("HCI chemicals:", ", ".join(sorted(hci_catalog.keys())[:40]), "...")
                continue

            # Defaults from HCI (if available)
            ref = hci_catalog.get(name, {})
            # Map HCI physicalstate -> our rtype
            phys = (ref.get("physicalstate") or "").lower()
            default_rtype = "solid"
            if phys in {"liquid", "solution"}:
                default_rtype = "liquid"

            default_inchi = ref.get("Inchi") or None
            if default_inchi is not None:
                rdkit_inchi = f"InChI={default_inchi}"
                default_inchikey = inchi.InchiToInchiKey(rdkit_inchi)
            else:
                default_inchikey = ""

            default_density = None
            d = ref.get("density")
            if isinstance(d, dict):
                try:
                    default_density = float(d.get("value"))
                except Exception:
                    pass

            default_conc = None
            c = ref.get("concentration")
            if isinstance(c, dict):
                try:
                    default_conc = float(c.get("value"))
                except Exception:
                    pass

            default_mm = None
            mm = ref.get("molecularMass")
            if isinstance(mm, dict):
                try:
                    default_mm = float(mm.get("value"))
                except Exception:
                    pass

            # If HCI groups contain "solvent" and this name appears there, suggest rtype=solvent
            default_is_solvent = False
            try:
                for g in (hci_doc.get("hasCampaign", {}).get("hasGroups", []) or []):
                    if g.get("groupName", "").lower() == "solvent":
                        for m in g.get("members", []) or []:
                            rref = m.get("reference") or {}
                            if rref.get("chemicalName") == name:
                                default_is_solvent = True
                                break
            except Exception:
                pass

            # --- Ask with defaults (ENTER keeps HCI-derived value) ---
            inchikey = _ask_with_default("InChIKey", default_inchikey)

            # rtype: if HCI flagged it as solvent, default to 'solvent'
            suggested_rtype = "solvent" if default_is_solvent else default_rtype
            rtype = _ask_with_default("Type [solid/liquid/solvent]", suggested_rtype).lower()

            if rtype == "solvent":
                solv = Solvent(name=name, inchikey=inchikey)
                # Placement still needed
                solv.locations = plate.parse_location(f"solvent {name}")
                solvents.append(solv)
                continue

            # Non-solvent reagent
            # Equivalents usually aren’t in HCI; ask, but allow empty to keep previous (None)
            eqv = _ask_float_with_default("Equivalents", None)
            if eqv is None:
                # sensible fallback if left blank
                eqv = 1.0

            is_limiting = False
            if not limiting_set:
                ans = _ask_with_default("Is this the limiting reagent? [y/N]", "N").lower()
                is_limiting = (ans in ("y", "yes"))
                limiting_set = limiting_set or is_limiting

            density = _ask_float_with_default("Density (g/mL)", default_density) if rtype == "liquid" else None
            concentration = _ask_float_with_default("Concentration (mol/L)",
                                                    default_conc) if rtype == "liquid" else None

            # Stock solution?
            stock_solution_ans = _ask_with_default("Is this a stock solution? [y/N]", "N").lower()
            stock_solution = stock_solution_ans in ("y", "yes")

            reagent = Reagent(
                name=name,
                inchikey=inchikey,
                rtype=rtype,
                equivalents=float(eqv),
                is_limiting=is_limiting,
                density=density,
                concentration=concentration,
                stock_solution=stock_solution,
                molar_mass=default_mm,  # <- prefill MW from HCI; ensure_molar_mass will respect it
            )

            reagent.locations = plate.parse_location(f"reagent {name}")
            reagents.append(reagent)
        all_names = [r.name for r in reagents] + [s.name for s in solvents]
        print("Current reagents:", ", ".join(all_names))
        cont = input("Add more reagents/solvents? [y/N]: ").strip().lower()
        if cont != 'y':
            break

    if not any(r.is_limiting for r in reagents):
        raise RuntimeError("No limiting reagent specified")

    final_volume = plate.input_matrix("final volume per well (uL)")
    conc_limiting = plate.input_matrix("desired concentration of limiting reagent (mol/L)")

    results: Dict[str, pd.DataFrame] = {}
    totals: Dict[str, float] = {}

    limiting = next(r for r in reagents if r.is_limiting)
    limiting.ensure_molar_mass()
    moles_limiting = conc_limiting * (final_volume / 1_000_000)
    missing_lim = (~limiting.locations) & (final_volume > 0)
    if missing_lim.any().any():
        print("Warning: Some wells have no limiting reagent.")

    for reagent in reagents:
        reagent.ensure_molar_mass()
        moles = moles_limiting * reagent.equivalents * reagent.locations
        reagent.moles = moles_limiting * reagent.equivalents * reagent.locations
        if reagent.rtype == 'solid':
            mass_g = moles * reagent.molar_mass
            if reagent.stock_solution:
                results[f"{reagent.name} (mg) as stock solution"] = mass_g * 1000
                totals[f"{reagent.name} (mg) as stock solution"] = (mass_g * 1000).sum().sum()
            else:
                results[f"{reagent.name} (mg)"] = mass_g * 1000
                totals[f"{reagent.name} (mg)"] = (mass_g * 1000).sum().sum()

        else:
            mass_g = moles * reagent.molar_mass
            if reagent.concentration:
                volume_l = moles / reagent.concentration
            elif reagent.density:
                volume_l = mass_g / reagent.density / 1000
            else:
                raise ValueError(f"Liquid reagent {reagent.name} requires density or concentration")
            if reagent.stock_solution:
                results[f"{reagent.name} (uL) as stock solution"] = volume_l * 1_000_000
                totals[f"{reagent.name} (uL) as stock solution"] = (volume_l * 1_000_000).sum().sum()
            else:
                results[f"{reagent.name} (uL)"] = volume_l * 1_000_000
                totals[f"{reagent.name} (uL)"] = (volume_l * 1_000_000).sum().sum()

    volume_used = pd.DataFrame(0.0, index=plate.template.index, columns=plate.template.columns)

    for key, df in results.items():
        if key.endswith('(uL)'):
            volume_used += df
    solvent_vol = final_volume - volume_used
    vol_available = solvent_vol  # Check if there is a second stock solution required anywhere, adjust vol_available adequately
    stock_reagents = [r for r in reagents if r.stock_solution]
    if stock_reagents:
        print("Stock reagents found, adjusting available volume accordingly.")
        overlap_reagents = pd.DataFrame(np.zeros(plate.template.shape), index=plate.template.index,columns=plate.template.columns)
        for reagent in stock_reagents:
            overlap_reagents += reagent.locations.astype(int)
        if (overlap_reagents > 1).values.any():
            vol_available = vol_available / overlap_reagents.where(overlap_reagents > 0, 1).astype(float)

    for reagent in reagents:
        if reagent.stock_solution:
            solvents_required = []
            unique_solvents_required = []
            for solvent in solvents:
                solvent_overlap = reagent.locations & solvent.locations
                if solvent_overlap.any().any():
                    solvents_required.append(solvent)
            if solvents_required:
                seen = set()
                for s in solvents_required:
                    if id(s) not in seen:
                        unique_solvents_required.append(s)
                        seen.add(id(s))
            solvents_required = unique_solvents_required

            for solvent in solvents_required:
                available_volumes_stock_solution = vol_available[reagent.locations & solvent.locations]
                required_concentration = (reagent.moles / available_volumes_stock_solution).replace([np.inf, -np.inf], 0).fillna(0) * 1_000_000
                max_required_concentration = required_concentration.max().max()

                # safeguard if too much liquid is requested
                stock_solution_volume = vol_available[reagent.locations & solvent.locations].min().min()
                if stock_solution_volume <= 0:
                    raise RuntimeError(f"Not enough space for stock solution {reagent.name} in {solvent.name}")

                stock_solution_dispensed = pd.DataFrame(0.0, index=plate.template.index, columns=plate.template.columns)
                stock_solution_dispensed[reagent.locations & solvent.locations] = reagent.moles / max_required_concentration * 1_000_000
                new_stock_solution = Stock_Solution(
                    name=f"{reagent.name} in {solvent.name}",
                    reagent=reagent,
                    solvent=solvent,
                    locations=reagent.locations & solvent.locations,
                    concentration=max_required_concentration,
                    volume_dispensed=stock_solution_dispensed
                )
                solvent_vol = solvent_vol - stock_solution_dispensed
                stock_solutions.append(new_stock_solution)
                results[f"{new_stock_solution.name} (uL)"] = stock_solution_dispensed
                totals[f"{new_stock_solution.name} {new_stock_solution.concentration} M (uL)"] = stock_solution_dispensed.sum().sum()
    if solvents:
        # Allocate the remaining volume to each solvent according to its
        # location mask without averaging across solvents
        for solv in solvents:
            vol = solvent_vol * solv.locations
            results[f"{solv.name} (uL)"] = vol
            totals[f"{solv.name} (uL)"] = vol.sum().sum()
        if (solvent_vol < 0).any().any():
            print("Warning: total reagent volumes exceed final volume in some wells")
        union = sum(s.locations for s in solvents)
        if (union == 0).any().any():
            print("Warning: Some wells have no solvent added.")
    else:
        if (solvent_vol < 0).any().any():
            print("Warning: total reagent volumes exceed final volume in some wells")
        else:
            print("No solvents specified, assuming all wells are dry.")


    df = pd.concat(results, axis=1)
    df.index.name = 'Row'

    totals_series = pd.Series(totals, name='Total')

    #  -----------------------------
    # Write Synthesis.json file
    #  -----------------------------

    # Ask for plate-constant temperature & time (experimental constraints)

    def _ask_float_or_blank(prompt: str) -> Optional[float]:
        s = input(prompt).strip()
        return float(s) if s else None

    print("\nEnter plate-constant conditions (leave blank to skip; but please don't skip :'( ):")
    temperature_plate = _ask_float_or_blank("  Temperature (°C): ")
    time_plate = _ask_float_or_blank("  Time (min): ")

    # Synthesis writer (current implementation) assumes a SINGLE well volume (µL) for dosing
    # If your final_volume matrix varies, we’ll take the most common value and warn.
    unique_vols = pd.unique(final_volume.values.ravel())
    unique_vols = [float(v) for v in unique_vols if not pd.isna(v)]
    if not unique_vols:
        raise RuntimeError("Final volume matrix is empty.")
    mode_volume = max(set(unique_vols), key=unique_vols.count)
    if len(set(unique_vols)) > 1:
        print(f"Warning: final volume varies across wells; using the mode: {mode_volume} µL "
              f"(synthesis_writer expects a constant per-well volume).")

    # Load HCI (required on this path), convert to optimizer dict for writer
    hci_path = Path(str(out_dir / args.hci)) if args.hci else None
    if not hci_path or not hci_path.exists():
        raise SystemExit("HCI file (--hci) is required when --synthesis is not provided.")

    import json
    hci_doc = json.loads(hci_path.read_text(encoding="utf-8"))
    opt_spec = hci_to_optimizer_dict(hci_doc)  # dict -> optimizer dict

    # Build 'selections' (groups/globals per well) from interactive placements & equivalents
    hci_groups = (hci_doc.get("hasCampaign") or {}).get("hasGroups", [])
    selections = build_selections_from_interactive(
        rows=rows,
        cols=cols,
        reagents=reagents,
        solvents=solvents,
        conc_limiting_df=conc_limiting,
        temperature_plate=temperature_plate,
        time_plate=time_plate,
        hci_groups=hci_groups,
    )

    # Emit synthesis.json via synthesis_writer (it will also append catalog chemicals not in any group)
    plate_size = rows * cols
    synthesis_out = out_dir / f"{reaction_name}_synthesis.json"

    write_synthesis_json(
        opt_spec=opt_spec,
        selections=selections,
        out_path=synthesis_out,
        plate_size=plate_size,
        well_volume_uL=mode_volume,  # constant per the writer’s current API
        limiting_name=None,  # optional; you can pass your limiting reagent name if desired
        experiment_start_index=1,
        concentration_key="concentration",
        # You can optionally control equivalents for catalog (non-group) chemicals:
        # fixed_equivalents_for_catalog={"Substrate A": 1.0},
        # default_catalog_equivalents=2.0,
    )

    print(f"Created synthesis.json: {synthesis_out}")

    # Reuse the synthesis fast-path to generate Excel + plots (includes temp/time heatmaps)
    reaction_name_for_outputs = args.output or reaction_name
    load_synthesis_and_emit_outputs(synthesis_out, out_dir, reaction_name=reaction_name_for_outputs)

    # Write the final DataFrame to Excel; legacy shouldn't be required anymore, but kept for reference.

    """
    #with pd.ExcelWriter(output_file) as writer:
        df.to_excel(writer, sheet_name='per_well')
        totals_series.to_frame().to_excel(writer, sheet_name='totals')
    print(f"Results written to {output_file}")

    viz_file = f"{reaction_name}_layout.png"
    visualize_distribution(reagents, solvents, plate, final_volume, str(out_dir/viz_file))
    print(f"Layout visualization saved to {viz_file}")
    """



if __name__ == '__main__':
    main()
