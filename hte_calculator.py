import argparse
import importlib.util
import string
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import math

import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

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
        if self.molar_mass is None:
            self.molar_mass = fetch_molar_mass(self.inchikey)


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


def visualize_distribution(reagents: List[Reagent], solvents: List[Solvent], plate: Plate, final_volume: pd.DataFrame, output: str) -> None:
    """Create heatmaps showing where each reagent/solvent is dispensed.

    Reagents are colour coded by their final concentration in each well while
    solvents remain binary.
    """
    items = reagents + solvents
    if not items:
        return
    n = len(items)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if n > 1 else [axes]
    for ax in axes[n:]:
        ax.axis('off')
    for ax, item in zip(axes, items):
        if isinstance(item, Reagent) and not item.moles.empty:
            conc = item.moles / (final_volume / 1_000_000)
            conc = conc.fillna(0.0)
            im = ax.imshow(conc.values, cmap="viridis")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("M")
        else:
            mat = item.locations.astype(int)
            im = ax.imshow(mat.values, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(plate.cols))
        ax.set_xticklabels(plate.template.columns)
        ax.set_yticks(range(plate.rows))
        ax.set_yticklabels(plate.template.index)
        ax.set_title(item.name)
    plt.tight_layout()
    plt.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="HTE Dispense Calculator")
    parser.add_argument("--output", default=None, help="Excel output file")
    parser.add_argument("--preload", default=None, help="Path to python file with PRELOADED_REAGENTS list")
    args = parser.parse_args()

    layout = input("Plate layout [24/48/96]: ").strip()
    while layout not in {"24", "48", "96"}:
        layout = input("Please enter 24, 48 or 96: ").strip()
    layout = int(layout)
    mapping = {24: (4, 6), 48: (6, 8), 96: (8, 12)}
    rows, cols = mapping[layout]

    plate = Plate(rows, cols)

    reaction_name = input("Reaction name: ").strip()
    output_file = args.output or f"{reaction_name}.xlsx"

    reagents: List[Reagent] = []
    solvents: List[Solvent] = []
    stock_solutions: List[Stock_Solution] = []

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
            inchikey = input("InChIKey: ").strip()
            rtype = input("Type [solid/liquid/solvent]: ").strip().lower()
            if rtype == 'solvent':
                solv = Solvent(name=name, inchikey=inchikey)
                solv.locations = plate.parse_location(f"solvent {name}")
                solvents.append(solv)
                continue
            eqv = float(input("Equivalents: "))
            is_limiting = False
            if not limiting_set:
                ans = input("Is this the limiting reagent? [y/N]: ").strip().lower()
                is_limiting = ans == 'y'
                limiting_set = is_limiting
            density = None
            concentration = None
            if rtype == 'liquid':
                d = input("Density (g/mL, blank if not known): ").strip()
                if d:
                    density = float(d)
                c = input("Concentration (mol/L, blank if not known): ").strip()
                if c:
                    concentration = float(c)
            stock_solution = input("Is this a stock solution? [y/N]: ").strip().lower() == 'y'
            reagent = Reagent(name=name, inchikey=inchikey, rtype=rtype,
                              equivalents=eqv, is_limiting=is_limiting,
                              density=density, concentration=concentration, stock_solution=stock_solution)

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
        reagent.moles = moles_limiting * reagent.equivalents
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
        print("Several stock reagents found, adjusting available volume accordingly.")
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
        each = solvent_vol / len(solvents)
        for solv in solvents:
            vol = each * solv.locations
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

    with pd.ExcelWriter(output_file) as writer:
        df.to_excel(writer, sheet_name='per_well')
        totals_series.to_frame().to_excel(writer, sheet_name='totals')
    print(f"Results written to {output_file}")

    viz_file = f"{reaction_name}_layout.png"
    visualize_distribution(reagents, solvents, plate, final_volume, viz_file)
    print(f"Layout visualization saved to {viz_file}")


if __name__ == '__main__':
    main()
